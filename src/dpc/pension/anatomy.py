"""What the learned policy actually does. Written and run after the test; not part of the protocol.

The pre-registered study (run.py) answers one question: does the learned policy cost less? This
module answers the next one a reader asks: what did it learn to do? It is a description, not a
test, and nothing here changes a verdict.

It retrains the three networks of the simulated-futures test from the same seeds and stops unless
they reproduce the committed result to the last digit, so what is described is the policy that was
scored. Then it follows the plan through the same 10,000 test futures and records:

  by_funded   what the learned policy and the glide path hold, on average, at each level of the
              funded ratio (both looked up in the situations the learned policy actually reached);
  by_yield    the same split by the level of the yield, for plans between 85% and 100% funded;
  by_year     the learned policy's average holdings in each year of the plan;
  curve       the weights each rule would choose as the funded ratio alone is varied, with
              everything else held at a middling value, at three levels of the yield.

And one question the protocol did not ask. The glide path in the protocol has fixed, textbook
numbers. `tuned_glide` trains those four numbers (where the slide starts and ends, how much is in
stocks at each end) on the same scenarios, by the same method and for the same score as the
network, and runs it on the same test futures. It separates "a rule tuned to the cost" from
"a network": if the tuned rule does as well, the network's extra inputs bought nothing.

And one check on where the network's advantage comes from. The simulated futures re-use history in runs
averaging 12 months, and the network sees the last 12 months of returns, yields and prices. It could
partly recognise a run and act on what followed it in history: knowledge no plan has. `one_month_blocks`
runs the same policies on futures whose months are drawn one at a time, so there is no run to recognise.

Writes reports/pension/anatomy.json and reports/pension/learned_policy.pt (the scored networks' weights).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from torch import nn

from . import data, evaluate as ev, scenarios as sc
from .plan import ASSETS, Plan, exogenous, run
from .policy import Ensemble, Glide, Learned
from .run import DRAWS, LEARNED, N_TEST, SEEDS, TEST_SEED, WALK_SEED
from .train import TrainConfig, score_t, train

SHUFFLE_SEED = 2028          # the futures drawn one month at a time; no other set uses it
FUNDED_EDGES = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.50)
YIELD_EDGES = (0.03, 0.05, 0.07, 0.10)
CURVE_YIELDS = (0.03, 0.06, 0.09)


class TunedGlide(nn.Module):
    """A glide path whose four numbers can be trained. It starts as the protocol's glide path (80%, 110%, 70%, 20%).

    Like it, it looks at the funded ratio only and holds stocks and matching bonds only.
    """

    def __init__(self) -> None:
        super().__init__()
        logit = lambda x: float(np.log(x / (1.0 - x)))  # noqa: E731
        self.raw = nn.Parameter(torch.tensor([0.0, logit((0.30 - 0.05) / 0.90), logit(0.70), logit(0.20)]))

    def numbers(self) -> dict[str, torch.Tensor]:
        r = torch.sigmoid(self.raw)
        start = 0.5 + 0.6 * r[0]                       # funded ratio where the slide out of stocks begins: 50% to 110%
        return {"slide_from": start, "slide_to": start + 0.05 + 0.90 * r[1], "stocks_when_short": r[2], "stocks_when_funded": r[3]}

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        n = self.numbers()
        along = torch.clamp((state[:, 0] - n["slide_from"]) / (n["slide_to"] - n["slide_from"]), 0.0, 1.0)
        stocks = n["stocks_when_short"] - (n["stocks_when_short"] - n["stocks_when_funded"]) * along
        return torch.stack([stocks, torch.zeros_like(stocks), 1.0 - stocks], dim=1)


def tune_glide(model: sc.Model, plan: Plan, seed: int, cfg: TrainConfig = TrainConfig()) -> TunedGlide:
    """Train the four numbers exactly as train.train trains a network: same scenarios, same score, same steps, best on validation."""
    torch.manual_seed(seed)
    train_set = exogenous(sc.simulate(model, cfg.n_train, plan.horizon, seed=1000 + seed), plan)
    val_set = exogenous(sc.simulate(model, cfg.n_val, plan.horizon, seed=5000 + seed), plan)
    rule = TunedGlide()
    opt = torch.optim.Adam(rule.parameters(), lr=cfg.lr * 5)      # four numbers on a logit scale: a larger step than a network's
    gen = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    for step in range(cfg.steps + 1):
        if step % cfg.check_every == 0:
            with torch.no_grad():
                val = float(score_t(run(rule, val_set, plan)["cost"]))
            if val < best - 1e-6:
                best, best_state = val, {k: v.clone() for k, v in rule.state_dict().items()}
        if step == cfg.steps:
            break
        idx = torch.randint(0, len(train_set), (cfg.batch,), generator=gen)
        loss = score_t(run(rule, train_set.take(idx), plan)["cost"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    rule.load_state_dict(best_state)
    rule.eval()
    return rule


def _labels(edges: tuple[float, ...], fmt) -> list[str]:
    return [f"under {fmt(edges[0])}"] + [f"{fmt(a)} to {fmt(b)}" for a, b in zip(edges[:-1], edges[1:])] + [f"over {fmt(edges[-1])}"]


def bucket_means(key: np.ndarray, edges: tuple[float, ...], values: dict[str, np.ndarray], keep: np.ndarray | None = None) -> dict:
    """Average each (n, 3) array of weights within the buckets of `key` cut at `edges`."""
    which = np.digitize(key, edges)
    if keep is None:
        keep = np.ones_like(key, dtype=bool)
    out = {"share_of_months": [], **{name: [] for name in values}}
    total = int(keep.sum())
    for b in range(len(edges) + 1):
        sel = keep & (which == b)
        out["share_of_months"].append(float(sel.sum() / total) if total else 0.0)
        for name, w in values.items():
            out[name].append([float(x) for x in w[sel].mean(axis=0)] if sel.any() else None)
    return out


def describe(learned, exo, plan: Plan) -> dict:
    """Follow the plan under `learned` and describe its holdings next to the glide path's in the same situations."""
    glide = Glide()
    with torch.no_grad():
        out = run(learned, exo, plan)
    T = plan.horizon
    funded = out["funded"][:, :T].numpy().reshape(-1).astype(np.float64)   # the funded ratio each decision saw
    w = out["weights"].numpy().reshape(-1, 3).astype(np.float64)           # float64, so that averages of millions of rows are exact
    y = exo.feats[:, :, 0].numpy().reshape(-1).astype(np.float64)
    state = torch.zeros(funded.size, 10)
    state[:, 0] = torch.from_numpy(funded).float()
    with torch.no_grad():
        w_glide = glide(state).numpy().astype(np.float64)         # the glide path looks at the funded ratio only
    year = np.tile(np.arange(T) // 12, len(exo))
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731

    # The response to the funded ratio alone: every other input at the middle of its range, the plan halfway through.
    grid = np.linspace(0.6, 1.5, 46)
    curve = {"funded": [float(x) for x in grid], "glide": None, "learned": {}}
    for yl in CURVE_YIELDS:
        s = Learned.CENTRE.repeat(len(grid), 1).clone()
        s[:, 0] = torch.from_numpy(grid).float()
        s[:, 1] = yl
        with torch.no_grad():
            g = glide(s)
            s[:, 7:10] = g                                         # holding what the glide path holds there
            curve["learned"][pct(yl)] = [[float(x) for x in row] for row in learned(s).numpy()]
            curve["glide"] = [[float(x) for x in row] for row in g.numpy()]

    return {
        "assets": list(ASSETS),
        "by_funded": {"buckets": _labels(FUNDED_EDGES, pct), **bucket_means(funded, FUNDED_EDGES, {"learned": w, "glide": w_glide})},
        "by_yield": {"buckets": _labels(YIELD_EDGES, pct), "funded_between": [0.85, 1.00],
                     **bucket_means(y, YIELD_EDGES, {"learned": w, "glide": w_glide}, keep=(funded >= 0.85) & (funded < 1.00))},
        "by_year": {"years": list(range(1, T // 12 + 1)),
                    "learned": [[float(x) for x in w[year == k].mean(axis=0)] for k in range(T // 12)],
                    "glide": [[float(x) for x in w_glide[year == k].mean(axis=0)] for k in range(T // 12)]},
        "overall": {"learned": [float(x) for x in w.mean(axis=0)], "glide": [float(x) for x in w_glide.mean(axis=0)],
                    # how much the learned policy's target weights move in a year (one-way, the sum over the three assets halved)
                    "weight_change_per_year_learned": float(np.abs(np.diff(out["weights"].numpy(), axis=1)).sum(axis=2).mean() * 12 / 2)},
        "curve": curve,
    }


def anatomy(root: Path, cfg: TrainConfig = TrainConfig(), log=print) -> dict:
    out_dir = root / "reports" / "pension"
    summary = json.loads((out_dir / "summary.json").read_text())
    panel = data.load(root / "data" / "raw" / "pension")
    plan = Plan()
    model = sc.fit(panel)
    nets = []
    for seed in SEEDS:
        net, _ = train(model, plan, seed, cfg)
        nets.append(net)
        log(f"  retrained seed {seed}")
    learned = Ensemble(nets)
    test = exogenous(sc.simulate(model, N_TEST, plan.horizon, seed=TEST_SEED), plan)
    with torch.no_grad():
        cost = float(run(learned, test, plan)["cost"].mean())
    committed = summary["simulated"]["policies"][LEARNED]["mean_cost"]
    if abs(cost - committed) > 1e-6:
        raise SystemExit(f"the retrained policy costs {cost:.6f} on the test futures, the committed result says {committed:.6f}: "
                         "this is not the policy that was scored, so it is not described")
    log(f"  reproduces the committed result: mean cost {cost:.6f}")
    torch.save({f"seed_{seed}": net.state_dict() for seed, net in zip(SEEDS, nets)}, out_dir / "learned_policy.pt")
    result = {"note": "A description of the policy scored in summary.json, made after the test. Not pre-registered; it changes no verdict.",
              "commit_of_results": summary["commit"], "reproduced_mean_cost": cost, **describe(learned, test, plan)}

    # The glide path with its four numbers trained on the same scenarios, for the same score.
    rules = []
    for seed in SEEDS:
        rules.append(tune_glide(model, plan, seed, cfg))
        log(f"  tuned the glide path, seed {seed}: " + ", ".join(f"{k} {float(v.detach()):.3f}" for k, v in rules[-1].numbers().items()))
    tuned = Ensemble(rules)
    walk = exogenous(sc.simulate(sc.fit(panel, random_walk=True), N_TEST, plan.horizon, seed=WALK_SEED), plan)
    block = {"note": "The protocol's glide path with its four numbers trained like the network (same scenarios, score and steps). "
                     "Added after the test; the verdicts below use the protocol's rule but were not pre-registered.",
             "numbers": [{k: float(v.detach()) for k, v in r.numbers().items()} for r in rules]}
    for name, exo in (("simulated", test), ("model_risk", walk)):
        with torch.no_grad():
            out_l = {k: v.numpy() for k, v in run(learned, exo, plan).items()}
            out_t = {k: v.numpy() for k, v in run(tuned, exo, plan).items()}
            out_g = {k: v.numpy() for k, v in run(Glide(), exo, plan).items()}
        block[name] = {"policy": ev.summarise(out_t),
                       "learned_minus_tuned": ev.paired(out_l["cost"], out_t["cost"], DRAWS, seed=7),
                       "tuned_minus_glide": ev.paired(out_t["cost"], out_g["cost"], DRAWS, seed=7)}
        log(f"  tuned glide path, {name}: mean cost {block[name]['policy']['mean_cost']:.3f}, worst-5% cost {block[name]['policy']['cvar_cost']:.3f}; "
            f"learned minus tuned: {block[name]['learned_minus_tuned']['verdict']}")
    result["tuned_glide"] = block

    # Futures with nothing to recognise: every month's shocks drawn on their own, not in runs of history.
    loose = exogenous(sc.simulate(model, N_TEST, plan.horizon, seed=SHUFFLE_SEED, block=1.0), plan)
    with torch.no_grad():
        outs = {name: {k: v.numpy() for k, v in run(policy, loose, plan).items()}
                for name, policy in (("Glide path", Glide()), ("Tuned glide path", tuned), (LEARNED, learned))}
    result["one_month_blocks"] = {
        "note": "The same policies, not retrained, on futures whose months are drawn one at a time instead of in runs of history "
                "averaging 12 months. Added after the test; not pre-registered.",
        "seed": SHUFFLE_SEED, "scenarios": N_TEST,
        "policies": {name: ev.summarise(o) for name, o in outs.items()},
        "learned_minus_glide": ev.paired(outs[LEARNED]["cost"], outs["Glide path"]["cost"], DRAWS, seed=7),
        "learned_minus_tuned": ev.paired(outs[LEARNED]["cost"], outs["Tuned glide path"]["cost"], DRAWS, seed=7)}
    for name, row in result["one_month_blocks"]["policies"].items():
        log(f"  months drawn one at a time: {name:18s} mean cost {row['mean_cost']:.3f}  worst-5% cost {row['cvar_cost']:.3f}")
    (out_dir / "anatomy.json").write_text(json.dumps(result, indent=1) + "\n")
    log(f"Wrote {(out_dir / 'anatomy.json').relative_to(root)}")
    return result
