"""Walk-forward training of the Deep Growth Network, and its out-of-sample run.

Protocol (fixed in docs/wealth_protocol.md before the out-of-sample run)
------------------------------------------------------------------------
* Refits every ``refit_every_years`` at the end of December, from the year before
  ``oos_start``. A refit at decision index d may only use returns up to month d.
* Inside each refit: the last ``val_years`` of decisions are validation, the rest
  training. Training episodes are random ``episode_months``-long stretches whose
  every return is dated at or before the first validation decision; the code
  asserts it. Validation is one continuous run, started from market weights.
* Each seed trains its own chain of refits (warm-started from its previous refit)
  with early stopping on validation growth. The deployed policy averages the
  seeds' targets and trade fractions.
* Out of sample, every strategy, including this one, starts from cash, pays the
  same costs in the same simulator (engine.py), and is never refit on future data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from ..logging_utils import get_logger
from .data import IndustryPanel
from .engine import RunResult, run
from .features import Features
from .model import DeepGrowthNet, crra, simulate

log = get_logger(__name__)


@dataclass(frozen=True)
class Variant:
    """A version of the network; 'deep' is the pre-specified primary model."""

    name: str
    attention: bool = True
    anchored: bool = True
    trade_all: bool = False
    max_tilt: float | None = None        # None: the study's max_tilt
    risk_aversion: float | None = None   # None: the study's risk_aversion
    description: str = ""


VARIANTS = {
    "deep": Variant("deep", description="Primary: attention across industries, anchored to market weights, learned trading"),
    "deep_trade_all": Variant("deep_trade_all", trade_all=True, description="Ablation: always trades fully to its target (no learned holding)"),
    "deep_mlp": Variant("deep_mlp", attention=False, description="Ablation: no attention, each industry judged alone"),
    "deep_unanchored": Variant("deep_unanchored", anchored=False, description="Ablation: no market anchor, allocates from scratch"),
}


@dataclass(frozen=True)
class TrainSettings:
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    init_trade: float
    learning_rate: float
    weight_decay: float
    episode_months: int
    batch_episodes: int
    max_steps: int
    eval_every: int
    patience: int
    val_years: int
    refit_every_years: int
    warm_start: bool
    grad_clip: float
    risk_aversion: float
    max_tilt: float
    cost_bps: float


@dataclass
class Tensors:
    X: torch.Tensor
    C: torch.Tensor
    R: torch.Tensor
    W: torch.Tensor
    first_valid: int

    @staticmethod
    def build(panel: IndustryPanel, feats: Features) -> "Tensors":
        f32 = torch.float32
        return Tensors(X=torch.tensor(np.nan_to_num(feats.assets), dtype=f32), C=torch.tensor(np.nan_to_num(feats.context), dtype=f32),
                       R=torch.tensor(panel.returns, dtype=f32), W=torch.tensor(panel.market_weights(), dtype=f32),
                       first_valid=feats.first_valid)


def refit_points(panel: IndustryPanel, oos_start: str, every_years: int) -> list[int]:
    """Decision indices (end of December) at which the policy is refit."""
    year = int(oos_start[:4]) - 1
    out = []
    while True:
        try:
            d = panel.index_of(f"{year}-12")
        except KeyError:
            break
        if d >= panel.n_months - 1:
            break
        out.append(d)
        year += every_years
    return out


def make_model(t: Tensors, s: TrainSettings, v: Variant) -> DeepGrowthNet:
    return DeepGrowthNet(t.X.shape[-1], t.C.shape[-1], s.d_model, s.n_heads, s.n_layers, s.dropout,
                         v.max_tilt or s.max_tilt, s.init_trade, attention=v.attention, anchored=v.anchored)


def validation_growth(model: DeepGrowthNet, t: Tensors, v0: int, refit: int, s: TrainSettings, v: Variant) -> float:
    model.eval()
    with torch.no_grad():
        logs, _, _ = simulate(model, t.X, t.C, t.R, t.W, torch.tensor([v0]), refit - v0, s.cost_bps / 1e4, trade_all=v.trade_all)
    return float(logs.mean())


def train_member(t: Tensors, refit: int, seed: int, s: TrainSettings, v: Variant, init: dict | None) -> tuple[dict, dict]:
    """One seed at one refit. Returns (best state, history)."""
    torch.manual_seed(seed * 100003 + refit)
    gen = torch.Generator().manual_seed(seed * 100003 + refit)
    v0 = refit - 12 * s.val_years              # first validation decision
    last_label = v0                            # training labels are returns of months <= v0
    hi = last_label - s.episode_months         # last episode start: decisions hi .. v0-1 earn months .. v0
    lo = t.first_valid
    if hi < lo:
        raise ValueError(f"refit at {refit}: not enough history for {s.episode_months}-month episodes")
    assert hi + s.episode_months <= v0, "a training label would fall inside validation"
    model = make_model(t, s, v)
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.AdamW(model.parameters(), lr=s.learning_rate, weight_decay=s.weight_decay)
    gamma = v.risk_aversion or s.risk_aversion
    best_val = validation_growth(model, t, v0, refit, s, v)
    best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
    history = {"val": [best_val], "best_step": 0}
    bad = 0
    for step in range(1, s.max_steps + 1):
        model.train()
        starts = torch.randint(lo, hi + 1, (s.batch_episodes,), generator=gen)
        logs, _, _ = simulate(model, t.X, t.C, t.R, t.W, starts, s.episode_months, s.cost_bps / 1e4, trade_all=v.trade_all)
        loss = -crra(logs, gamma).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), s.grad_clip)
        opt.step()
        if step % s.eval_every == 0:
            val = validation_growth(model, t, v0, refit, s, v)
            history["val"].append(val)
            if val > best_val + 1e-7:
                best_val, bad = val, 0
                best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
                history["best_step"] = step
            else:
                bad += 1
                if bad >= s.patience:
                    break
    history["best_val_monthly_log_growth"] = best_val
    history["steps"] = step
    return best_state, history


def _chain(args: tuple) -> dict:
    """All refits for one seed, in order (a worker process)."""
    panel, feats, s, v, seed, refits, out_dir = args
    torch.set_num_threads(1)
    t = Tensors.build(panel, feats)
    state, hist = None, {}
    for refit in refits:
        t0 = time.time()
        state, h = train_member(t, refit, seed, s, v, state if s.warm_start else None)
        torch.save(state, out_dir / f"refit{panel.dates[refit]:%Y}_seed{seed}.pt")
        h["seconds"] = round(time.time() - t0, 1)
        hist[f"{panel.dates[refit]:%Y-%m}"] = h
        log.info("%s seed %d refit %s: best val %.5f/month at step %d (%.0fs)", v.name, seed, f"{panel.dates[refit]:%Y-%m}",
                 h["best_val_monthly_log_growth"], h["best_step"], h["seconds"])
    return {"seed": seed, "history": hist}


def train_walk_forward(panel: IndustryPanel, feats: Features, s: TrainSettings, v: Variant, seeds: list[int],
                       oos_start: str, out_dir: Path, workers: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    refits = refit_points(panel, oos_start, s.refit_every_years)
    jobs = [(panel, feats, s, v, seed, refits, out_dir) for seed in seeds]
    t0 = time.time()
    if workers > 1 and len(seeds) > 1:
        with get_context("fork").Pool(min(workers, len(seeds))) as pool:
            results = pool.map(_chain, jobs)
    else:
        results = [_chain(j) for j in jobs]
    meta = {"variant": v.name, "description": v.description, "seeds": seeds,
            "refits": [f"{panel.dates[r]:%Y-%m}" for r in refits], "seconds": round(time.time() - t0, 1),
            "histories": {str(r["seed"]): r["history"] for r in results}}
    (out_dir / "training.json").write_text(json.dumps(meta, indent=1))
    return meta


@dataclass
class EnsemblePolicy:
    """The deployed policy: the average of the seeds' targets and trade fractions."""

    t: Tensors
    s: TrainSettings
    v: Variant
    refits: list[int]
    members: dict[int, list[DeepGrowthNet]] = field(default_factory=dict)
    kappas: list[float] = field(default_factory=list)

    @staticmethod
    def load(panel: IndustryPanel, feats: Features, s: TrainSettings, v: Variant, seeds: list[int], oos_start: str, model_dir: Path) -> "EnsemblePolicy":
        t = Tensors.build(panel, feats)
        refits = refit_points(panel, oos_start, s.refit_every_years)
        pol = EnsemblePolicy(t=t, s=s, v=v, refits=refits)
        for r in refits:
            nets = []
            for seed in seeds:
                m = make_model(t, s, v)
                m.load_state_dict(torch.load(model_dir / f"refit{panel.dates[r]:%Y}_seed{seed}.pt"))
                m.eval()
                nets.append(m)
            pol.members[r] = nets
        return pol

    def decide(self, t: int, w_drift: np.ndarray) -> np.ndarray:
        r = max(x for x in self.refits if x <= t)               # the latest refit at or before this decision
        w = torch.tensor(w_drift, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outs = [m(self.t.X[t:t + 1], self.t.C[t:t + 1], w, self.t.W[t:t + 1]) for m in self.members[r]]
        target = torch.stack([o[0] for o in outs]).mean(0)[0].numpy().astype(float)
        kappa = 1.0 if self.v.trade_all or w_drift.sum() < 0.5 else float(torch.stack([o[1] for o in outs]).mean())
        self.kappas.append(kappa)
        new = w_drift + kappa * (target - w_drift)
        return np.clip(new, 0, None) / np.clip(new, 0, None).sum()


def run_policy(panel: IndustryPanel, policy: EnsemblePolicy, oos_start: str, cost_bps: float, name: str) -> RunResult:
    start = panel.index_of(oos_start)
    if policy.refits[0] != start - 1:
        raise ValueError("the first refit must be the decision just before the out-of-sample start")
    return run(panel, policy.decide, start, panel.n_months - 1, cost_bps, name)
