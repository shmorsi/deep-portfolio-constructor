"""Stage 5: end-to-end ("decision-focused") portfolio learning through a convex optimiser.

Two-stage ("predict-then-optimise") pipelines fit the forecasts by forecast
error and then optimise. But the optimiser does not care about forecast error
everywhere equally; it cares about the errors that change the *portfolio*.
Here the whole chain is trained on the quantity we actually care about:

    features --> [alpha net] --> mu ----\\
                                          >-- [convex optimiser layer] --> w* --> realised
    returns  --> [deep risk net] --> B,s -/     (cvxpylayers, exact KKT          next-month
                                                  gradients)                    utility

    loss(month t) = -( w*' r_{t+1} - cost * ||w* - w_prev||_1 ) + lambda * sum_d (w*' r_d)^2

The optimiser layer is the *same* constrained problem every long-short
strategy uses (dollar-, sector-, beta-, size-, value-neutral, position caps,
turnover limit), so every portfolio the model ever produces is feasible.

Design choices that keep this honest and stable
-----------------------------------------------
* **Residual alpha, zero-initialised**:
      mu_i = IC * vol_i * ( exp(a) * z_combined_i + g(z_i, h_i) )
  with g's output layer and ``a`` initialised to 0. At step 0 the model *is*
  the classical two-stage strategy; it moves away only where that improves
  realised out-of-sample utility on the validation window.
* **Risk net initialised from Stage 4** (the walk-forward checkpoint for the
  same refit date) and fine-tuned with a smaller learning rate.
* **Previous weights**: turnover depends on last month's portfolio. We keep a
  buffer of each month's latest solution (at most one epoch stale) instead of
  re-solving the whole history sequentially every step.
* **Walk-forward**: same yearly refit dates; a month is a training example
  only if its realised next-month return is fully known at the refit date.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from cvxpylayers.torch import CvxpyLayer
from torch import nn

from dpc.backtest import LookAheadError, PointInTimeView, rebalance_dates, run_backtest, save_result
from dpc.config import Config, RiskModelConfig
from dpc.covariance import DAYS_PER_MONTH
from dpc.data.panel import Panel
from dpc.deep.alpha_data import AlphaBank, build_alpha_bank
from dpc.deep.alpha_model import AlphaEnsemble
from dpc.deep.features import SampleBank, build_sample_bank
from dpc.deep.risk_model import DeepRiskModel, to_return_space
from dpc.deep.train_risk import new_model, refit_dates
from dpc.logging_utils import get_logger
from dpc.optimizer import LongShortOptimizer, OptimizerInputs, build_long_short_problem
from dpc.reproducibility import make_torch_generator, set_global_seed
from dpc.signals import SIGNAL_NAMES

log = get_logger(__name__)
F64 = torch.float64


# --------------------------------------------------------------------------- #
# Month-level data
# --------------------------------------------------------------------------- #
@dataclass
class MonthBank:
    dates: pd.DatetimeIndex  # month-ends usable as decision dates
    bank_idx: np.ndarray  # position in the SampleBank
    z: np.ndarray  # (M, N, 3) component z-scores (0 where missing)
    zc: np.ndarray  # (M, N) composite z-score
    vol_m: np.ndarray  # (M, N) trailing monthly vol (cov window)
    inv: np.ndarray  # (M, N) investable mask
    X: np.ndarray  # (M, F, N) betas to neutralise
    S: np.ndarray  # (n_sec, N) sector membership
    r_next: np.ndarray  # (M, Dmax, N) next-month daily returns (0 padded)
    label_end: pd.DatetimeIndex  # last day of each month's label window
    ab: AlphaBank | None = None  # AlphaFormer inputs (fully deep variant only)
    ab_rows: np.ndarray | None = None  # row of each month in ``ab``

    def __len__(self) -> int:
        return len(self.dates)


def build_month_bank(cfg: Config, panel: Panel, bank: SampleBank, signals: dict, betas: dict,
                     ab: AlphaBank | None = None) -> MonthBank:
    me = panel.month_ends
    me = me[me.isin(bank.origins)]
    if ab is not None:  # fully deep variant: only months the AlphaFormer can score
        me = me[me.isin(ab.dates)]
    rets = panel.returns
    vol = rets.rolling(cfg.backtest.cov_window_days, min_periods=cfg.backtest.cov_window_days).std() * math.sqrt(DAYS_PER_MONTH)
    tick = panel.tickers
    inv = panel.eligible.loc[me].to_numpy().copy()
    for df in (*signals.values(), *betas.values()):
        inv &= df.loc[me, tick].notna().to_numpy()
    inv &= bank.mask_in[[bank.index_of(d) for d in me]]
    inv &= vol.loc[me].notna().to_numpy()
    if ab is not None:
        inv &= ab.mask[ab.rows_for(me)]
    keep = inv.sum(axis=1) >= 20
    me, inv = me[keep], inv[keep]
    z = np.stack([signals[k].loc[me, tick].fillna(0).to_numpy() for k in SIGNAL_NAMES], axis=-1)
    zc = signals["combined"].loc[me, tick].fillna(0).to_numpy()
    X = np.stack([betas[f].loc[me, tick].fillna(0).to_numpy() for f in cfg.optimizer.neutral_factors], axis=1)
    sec = panel.universe["sector"]
    S = np.stack([(sec == s).to_numpy(float) for s in sorted(sec.unique())])
    all_me = panel.month_ends
    dmax = 24
    r_next = np.zeros((len(me), dmax, len(tick)))
    label_end = []
    for k, d in enumerate(me):
        pos = all_me.get_loc(d)
        nxt = all_me[pos + 1] if pos + 1 < len(all_me) else rets.index[-1]
        block = rets.loc[(rets.index > d) & (rets.index <= nxt)].fillna(0).to_numpy()
        r_next[k, : len(block)] = block[:dmax]
        label_end.append(nxt)
    return MonthBank(
        dates=me, bank_idx=np.array([bank.index_of(d) for d in me]), z=z, zc=zc,
        vol_m=np.nan_to_num(vol.loc[me].to_numpy()), inv=inv, X=X, S=S, r_next=r_next,
        label_end=pd.DatetimeIndex(label_end), ab=ab, ab_rows=None if ab is None else ab.rows_for(me),
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class AlphaNet(nn.Module):
    """Residual expected-return model. Output is 0 at initialisation."""

    def __init__(self, d_embed: int, hidden: int, residual_scale: float):
        super().__init__()
        self.residual_scale = residual_scale
        self.log_scale = nn.Parameter(torch.zeros(()))  # scales the classical composite
        self.linear = nn.Linear(len(SIGNAL_NAMES), 1, bias=False)  # learned re-weighting of the 3 signals
        self.mlp = nn.Sequential(nn.Linear(d_embed + len(SIGNAL_NAMES), hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, zc: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        g = self.linear(z).squeeze(-1) + self.mlp(torch.cat([h, z], dim=-1)).squeeze(-1)
        return torch.exp(self.log_scale) * zc + self.residual_scale * g


class E2EModel(nn.Module):
    """views (optional AlphaFormer) + risk (DeepRiskModel) -> mu, B, s for the optimiser layer.

    Without ``views`` the base score is the classical composite z-score (the
    pre-specified e2e model). With ``views`` the base score is the
    AlphaFormer's cross-sectional forecast: deep learning at every layer.
    """

    def __init__(self, risk: DeepRiskModel, rc: RiskModelConfig, alpha_hidden: int, ic: float, residual_scale: float,
                 views: AlphaEnsemble | None = None):
        super().__init__()
        self.risk = risk
        self.views = views
        self.alpha = AlphaNet(rc.d_model, alpha_hidden, residual_scale).double()
        self.ic = ic
        self.horizon = rc.horizon_days

    def forward(self, bank: SampleBank, mb: MonthBank, rows: np.ndarray):
        idx = mb.bank_idx[rows]
        mask = torch.as_tensor(mb.inv[rows])
        x = torch.as_tensor(bank.x[idx], dtype=torch.float32)
        f = torch.as_tensor(bank.f[idx], dtype=torch.float32)
        st = torch.as_tensor(bank.static[idx], dtype=torch.float32)
        Bz, psi, h = self.risk(x, f, st, mask)
        Bh, sh = to_return_space(Bz.double(), psi.double(), torch.as_tensor(bank.sigma[idx], dtype=F64), self.horizon)
        m = mask.to(F64)
        z = torch.as_tensor(mb.z[rows], dtype=F64)
        if self.views is not None:
            ar = mb.ab_rows[rows]
            base, _ = self.views(torch.from_numpy(mb.ab.seq[ar]), torch.from_numpy(mb.ab.static[ar]), mask)
            base = base.double()
        else:
            base = torch.as_tensor(mb.zc[rows], dtype=F64)
        score = self.alpha(z, base, h.double())
        mu = self.ic * torch.as_tensor(mb.vol_m[rows], dtype=F64) * score * m
        return mu, Bh * m.unsqueeze(-1), sh * m


def make_layer(cfg: Config, n: int, n_sec: int) -> CvxpyLayer:
    lsp = build_long_short_problem(n, cfg.risk_model.n_factors, n_sec, len(cfg.optimizer.neutral_factors),
                                   cfg.optimizer, cfg.backtest.transaction_cost_bps)
    return CvxpyLayer(lsp.problem, parameters=lsp.parameters, variables=[lsp.w])


def solve_layer(layer: CvxpyLayer, cfg: Config, mb: MonthBank, rows: np.ndarray, mu, Bh, sh, w0: torch.Tensor) -> torch.Tensor:
    oc = cfg.optimizer
    b = len(rows)
    ub = torch.as_tensor(mb.inv[rows], dtype=F64) * oc.max_weight
    S = torch.as_tensor(mb.S, dtype=F64).expand(b, -1, -1).contiguous()
    X = torch.as_tensor(mb.X[rows], dtype=F64)
    first = w0.abs().sum(-1) < 1e-12
    tl = torch.where(first, torch.full((b,), oc.gross_leverage, dtype=F64), torch.full((b,), oc.turnover_limit, dtype=F64))
    w0 = w0 * ub.gt(0)  # anything no longer investable must be sold
    (w,) = layer(mu, Bh, sh, w0, ub, S, X, tl, solver_args={"eps": oc.solver_eps})
    return w


def month_utility(w: torch.Tensor, w0: torch.Tensor, r_next: torch.Tensor, cost: float, lam: float) -> torch.Tensor:
    """Realised net return minus lambda * realised daily variance, per month (b,)."""
    p = torch.einsum("bdn,bn->bd", r_next, w)  # daily P&L (padded days are 0)
    return p.sum(-1) - cost * (w - w0).abs().sum(-1) - lam * (p**2).sum(-1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _sequential_weights(model, layer, cfg, bank, mb, rows: np.ndarray, w_start: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Chain decisions month by month (true previous weights). No gradients."""
    ws, w0s = [], []
    w_prev = w_start
    with torch.no_grad():
        mu, Bh, sh = model(bank, mb, rows)
        for k in range(len(rows)):
            w = solve_layer(layer, cfg, mb, rows[k : k + 1], mu[k : k + 1], Bh[k : k + 1], sh[k : k + 1], w_prev.unsqueeze(0))[0]
            w0s.append(w_prev)
            ws.append(w)
            w_prev = w
    return torch.stack(ws), torch.stack(w0s)


def train_e2e(cfg: Config, bank: SampleBank, mb: MonthBank, refit: pd.Timestamp, risk_state: dict,
              alpha_state: dict | None, rc: RiskModelConfig, seed: int, layer: CvxpyLayer,
              views: AlphaEnsemble | None = None) -> tuple[E2EModel, dict]:
    ec = cfg.e2e
    usable = np.where(mb.label_end <= refit)[0]
    if len(usable) and mb.label_end[usable].max() > refit:
        raise LookAheadError("E2E training label extends beyond the refit date")
    val = usable[-ec.val_months:]
    train = usable[: -ec.val_months]
    set_global_seed(seed)
    risk = new_model(bank, rc)
    risk.load_state_dict(risk_state)
    model = E2EModel(risk, rc, ec.alpha_hidden, cfg.alpha.assumed_ic, ec.alpha_residual_scale, views=views)
    if alpha_state is not None:
        model.alpha.load_state_dict(alpha_state)
    groups = [{"params": model.alpha.parameters(), "lr": ec.learning_rate}]
    if views is not None:  # pre-trained AlphaFormer fine-tunes through the optimiser too
        groups.append({"params": model.views.parameters(), "lr": ec.learning_rate * ec.pretrained_lr_mult})
    if ec.finetune_risk_model:
        groups.append({"params": model.risk.parameters(), "lr": ec.learning_rate * ec.pretrained_lr_mult})
    else:
        for p in model.risk.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(groups, weight_decay=ec.weight_decay)
    gen = make_torch_generator(seed, refit.year + 1000)
    cost = cfg.backtest.transaction_cost_bps / 1e4
    lam = ec.loss_risk_aversion
    R_all = torch.as_tensor(mb.r_next, dtype=F64)

    model.risk.eval()  # dropout off: the optimiser input should be deterministic
    if model.views is not None:
        model.views.eval()
    zero = torch.zeros(mb.inv.shape[1], dtype=F64)
    W_buf, _ = _sequential_weights(model, layer, cfg, bank, mb, usable, zero)  # initial previous-weights buffer
    pos = {r: k for k, r in enumerate(usable)}

    def val_score() -> float:
        w_start = W_buf[pos[train[-1]]] if len(train) else zero
        w, w0 = _sequential_weights(model, layer, cfg, bank, mb, val, w_start)
        return float(month_utility(w, w0, R_all[val], cost, lam).mean())

    best, best_state, bad, hist = val_score(), copy.deepcopy(model.state_dict()), 0, []
    hist.append((float("nan"), best))
    for epoch in range(ec.max_epochs):
        perm = train[torch.randperm(len(train), generator=gen).numpy()]
        losses = []
        for i in range(0, len(perm), ec.batch_size):
            rows = np.sort(perm[i : i + ec.batch_size])
            w0 = torch.stack([W_buf[pos[r] - 1] if pos[r] > 0 else zero for r in rows])
            mu, Bh, sh = model(bank, mb, rows)
            w = solve_layer(layer, cfg, mb, rows, mu, Bh, sh, w0)
            loss = -month_utility(w, w0, R_all[rows], cost, lam).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ec.grad_clip)
            opt.step()
            losses.append(loss.item())
            for k, r in enumerate(rows):
                W_buf[pos[r]] = w[k].detach()
        v = val_score()
        hist.append((float(np.mean(losses)), v))
        if v > best + 1e-6:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= ec.early_stopping_patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    info = {"n_train": len(train), "n_val": len(val), "epochs": len(hist) - 1, "val_utility": best,
            "val_utility_init": hist[0][1], "log_scale": float(model.alpha.log_scale.detach()),
            **{f"w_{k}": float(v) for k, v in zip(SIGNAL_NAMES, model.alpha.linear.weight.detach().ravel())}}
    return model, info


# --------------------------------------------------------------------------- #
# Inference strategy for the shared backtest engine
# --------------------------------------------------------------------------- #
@dataclass
class E2EForecasts:
    dates: pd.DatetimeIndex
    mu: np.ndarray  # (D, N)
    B: np.ndarray  # (D, N, K)
    s: np.ndarray  # (D, N)


def make_e2e_strategy(cfg: Config, fc: E2EForecasts, tickers: list[str], n_sec: int):
    from dpc.baselines import long_short_inputs

    n = len(tickers)
    opt = LongShortOptimizer(n, fc.B.shape[-1], n_sec, len(cfg.optimizer.neutral_factors), cfg.optimizer, cfg.backtest.transaction_cost_bps)

    def fn(view: PointInTimeView, w0: pd.Series) -> pd.Series:
        k = fc.dates.get_loc(view.date)
        # Only stocks the model actually forecast (positive idiosyncratic risk). A stock with no
        # forecast would look riskless to the optimiser and could be misused as a free hedge.
        keep = np.isin(view.tickers, view.investable) & (fc.s[k] > 0)
        x: OptimizerInputs = long_short_inputs(view, w0, fc.mu[k] * keep, fc.B[k] * keep[:, None], fc.s[k] * keep, cfg)
        return pd.Series(opt.solve(x), index=view.tickers)

    return fn, opt


def walk_forward_e2e(cfg: Config, panel: Panel, bank: SampleBank, mb: MonthBank, rc: RiskModelConfig,
                     seed: int, risk_tag: str, tag: str, views_tag: str | None = None) -> tuple[E2EForecasts, pd.DataFrame]:
    rebal = rebalance_dates(panel, cfg.backtest.oos_start)
    refits = refit_dates(rebal, rc.refit_every_months)
    layer = make_layer(cfg, len(panel.tickers), mb.S.shape[0])
    D, N, K = len(rebal), len(panel.tickers), rc.n_factors
    mu_o, B_o, s_o = np.zeros((D, N)), np.zeros((D, N, K)), np.zeros((D, N))
    alpha_state, rows_log = None, []
    for j, R in enumerate(refits):
        ck = torch.load(cfg.paths.models / f"risk_{risk_tag}_{R.date()}.pt", weights_only=False)
        views = None
        if views_tag is not None:  # the AlphaFormer ensemble trained at the same refit date (Stage 4b)
            from dpc.deep.train_alpha import load_alpha_ensemble

            views = load_alpha_ensemble(cfg, mb.ab, R, tag=views_tag)
        model, info = train_e2e(cfg, bank, mb, R, ck["state"], alpha_state, rc, seed, layer, views=views)
        alpha_state = copy.deepcopy(model.alpha.state_dict())
        torch.save({"state": model.state_dict()}, cfg.paths.models / f"e2e_{tag}_{R.date()}.pt")
        info["refit"] = R
        rows_log.append(info)
        log.info("[%s] E2E refit %s: %d train / %d val months, %d epochs, val utility %.5f (start %.5f), "
                 "scale %.2f, w_mom %+.3f w_rev %+.3f w_lowvol %+.3f", tag, R.date(), info["n_train"], info["n_val"],
                 info["epochs"], info["val_utility"], info["val_utility_init"], math.exp(info["log_scale"]),
                 info["w_momentum"], info["w_reversal"], info["w_low_vol"])
        nxt = refits[j + 1] if j + 1 < len(refits) else pd.Timestamp.max
        use = np.where((rebal >= R) & (rebal < nxt))[0]
        rows = np.array([mb.dates.get_loc(rebal[u]) for u in use])
        with torch.no_grad():
            mu, Bh, sh = model(bank, mb, rows)
        mu_o[use], B_o[use], s_o[use] = mu.numpy(), Bh.numpy(), sh.numpy()
    return E2EForecasts(rebal, mu_o, B_o, s_o), pd.DataFrame(rows_log)


def run_stage5(cfg: Config, seed: int | None = None, tag: str = "main", risk_tag: str = "main",
               views: bool = False) -> pd.DataFrame:
    """views=False: the pre-specified e2e (classical composite + deep risk + optimiser layer).
    views=True: the fully deep e2e (AlphaFormer + deep risk + optimiser layer), saved as e2e_<tag>."""
    from dpc.baselines import get_betas, load_signals, summarize

    seed = cfg.project.seed if seed is None else seed
    panel = Panel.load(cfg.paths.data_processed)
    rc = cfg.risk_model
    signals, betas = load_signals(cfg), get_betas(cfg, panel)
    bank = build_sample_bank(panel, rc.seq_len_days, rc.horizon_days, rc.sample_stride_days)
    ab = build_alpha_bank(panel, signals, betas, cfg.alpha_model) if views else None
    mb = build_month_bank(cfg, panel, bank, signals, betas, ab=ab)
    fc, refit_log = walk_forward_e2e(cfg, panel, bank, mb, rc, seed, risk_tag, tag, views_tag="main" if views else None)
    refit_log.to_csv(cfg.paths.reports / f"stage5_refits_{tag}.csv", index=False)
    np.savez_compressed(cfg.paths.data_processed / f"e2e_forecasts_{tag}.npz", mu=fc.mu, B=fc.B, s=fc.s,
                        dates=fc.dates.values.astype("datetime64[ns]").astype(np.int64))
    fn, opt = make_e2e_strategy(cfg, fc, panel.tickers, mb.S.shape[0])
    name = "e2e" if tag == "main" else f"e2e_{tag}"
    res = run_backtest(name, panel, fn, signals, betas, cfg.backtest.oos_start, cfg.backtest.transaction_cost_bps, False)
    res.info["turnover_slack_months"] = int(np.sum(np.array(opt.slack_used) > 1e-6))
    save_result(res, cfg.paths.results)
    table = summarize([res], panel, cfg)
    log.info("Stage 5 (%s):\n%s", name, table.round(3).to_string())
    return table
