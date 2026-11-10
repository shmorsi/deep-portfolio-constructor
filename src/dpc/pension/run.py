"""The whole pension study, in the order the protocol (docs/pension_protocol.md) fixes it.

1. Simulated futures. Fit the scenario generator on all the data, train the learned policy on its
   scenarios, and compare the five policies on 10,000 scenarios that training never saw.
   Then the model-risk test: the same policies, not retrained, on 10,000 scenarios in which the yield
   has no pull back to an average.
2. Real history, walking forward. For every year from 1973 to 2006, start a plan that December and
   follow it through the 20 years that actually came next. The learned policy used for a start year
   was trained only on data up to that start (refitted every five years).

Everything is written to reports/pension/.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data, evaluate as ev, scenarios as sc
from .fixed_income import convexity, duration
from .plan import ASSETS, Plan, exogenous, run
from .policy import Ensemble, baselines
from .train import TrainConfig, train

SEEDS = (1, 2, 3)            # one network per seed; the learned policy is their average
TEST_SEED = 2026             # the 10,000 test scenarios; no training or validation set uses it
WALK_SEED = 2027             # the 10,000 scenarios of the model-risk test (yields with no pull to an average)
N_TEST = 10_000
FIRST_START, LAST_START = 1973, 2006
REFIT_YEARS = 5
DRAWS = 2000
HISTORY_BLOCK = 10           # start years resampled together on the walk through history
LEARNED = "Learned policy"


def _np(out: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().numpy() for k, v in out.items()}


def _follow(policies: dict, exo, plan: Plan) -> dict[str, dict[str, np.ndarray]]:
    with torch.no_grad():
        return {name: _np(run(p, exo, plan)) for name, p in policies.items()}


def _train_ensemble(model: sc.Model, plan: Plan, cfg: TrainConfig, log, tag: str) -> tuple[Ensemble, list[dict]]:
    nets, history = [], []
    for seed in SEEDS:
        net, h = train(model, plan, seed, cfg, log=None)
        nets.append(net)
        history += [{"fit": tag, **row} for row in h]
        log(f"  {tag}: seed {seed}: validation score {h[0]['validation_score']:.4f} at the start, {min(r['validation_score'] for r in h):.4f} at best")
    return Ensemble(nets), history


def _commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def study(root: Path, refresh: bool = False, cfg: TrainConfig = TrainConfig(), log=print) -> dict:
    out_dir = root / "reports" / "pension"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = data.load(root / "data" / "raw" / "pension", refresh)
    plan = Plan()
    log(f"Data: {len(panel)} months, {panel.index[0]:%Y-%m} to {panel.index[-1]:%Y-%m}")

    # ---- 1. simulated futures
    model = sc.fit(panel)
    log(f"Scenario generator: yield AR(1) {model.phi_y:.4f} around {model.mean_yield:.2%}; inflation AR(1) {model.phi_pi:.3f}")
    learned, training = _train_ensemble(model, plan, cfg, log, "all data")
    test = exogenous(sc.simulate(model, N_TEST, plan.horizon, seed=TEST_SEED), plan)
    policies = {**baselines(), LEARNED: learned}
    sim = _follow(policies, test, plan)
    table = {name: ev.summarise(o) for name, o in sim.items()}
    compare = {b: ev.paired(sim[LEARNED]["cost"], sim[b]["cost"], DRAWS, seed=7) for b in baselines()}
    fans = {name: ev.fan(o["funded"]) for name, o in sim.items()}
    pick = np.random.default_rng(11).choice(N_TEST, 120, replace=False)   # a few whole paths, for the pictures
    sample = {name: np.round(sim[name]["funded"][pick][:, ::12], 4).tolist() for name in ("Glide path", LEARNED, "60/40")}
    for name, row in table.items():
        log(f"  {name:24s} mean cost {row['mean_cost']:.3f}  worst-5% cost {row['cvar_cost']:.3f}  funded at 20y {row['funded_end_share']:.1%}")

    # ---- 1b. model risk: yields that never come back, policies as they are
    walk = _follow(policies, exogenous(sc.simulate(sc.fit(panel, random_walk=True), N_TEST, plan.horizon, seed=WALK_SEED), plan), plan)
    walk_table = {name: ev.summarise(o) for name, o in walk.items()}
    walk_compare = {b: ev.paired(walk[LEARNED]["cost"], walk[b]["cost"], DRAWS, seed=7) for b in baselines()}
    for name, row in walk_table.items():
        log(f"  no pull to an average: {name:24s} mean cost {row['mean_cost']:.3f}  worst-5% cost {row['cvar_cost']:.3f}")

    # ---- 2. real history, walking forward
    rows = []
    hist_fit: Ensemble | None = None
    for year in range(FIRST_START, LAST_START + 1):
        start = panel.index.get_loc(pd.Timestamp(year=year - 1, month=12, day=31))
        if (year - FIRST_START) % REFIT_YEARS == 0:
            past = panel.iloc[:start + 1]                       # nothing after the plan's first decision
            hist_fit, h = _train_ensemble(sc.fit(past), plan, cfg, log, f"data to {year - 1}")
            training += h
        exo = exogenous(sc.from_history(panel, start, plan.horizon), plan)
        got = _follow({**baselines(), LEARNED: hist_fit}, exo, plan)
        rows.append({"start_year": year, "start_yield": float(panel["yield"].iloc[start]),
                     **{f"cost | {n}": float(o["cost"][0]) for n, o in got.items()},
                     **{f"funded_end | {n}": float(o["funded"][0, -1]) for n, o in got.items()},
                     **{f"stocks | {n}": float(o["weights"][0, :, 0].mean()) for n, o in got.items()}})
    history = pd.DataFrame(rows)
    hist_table = {n: {"mean_cost": float(history[f"cost | {n}"].mean()), "worst_cost": float(history[f"cost | {n}"].max()),
                      "funded_end_share": float((history[f"funded_end | {n}"] >= 1.0).mean()),
                      "stocks_mean": float(history[f"stocks | {n}"].mean())} for n in policies}
    hist_compare = {b: ev.paired_mean_blocks(history[f"cost | {LEARNED}"].to_numpy(), history[f"cost | {b}"].to_numpy(), HISTORY_BLOCK, DRAWS, seed=7)
                    for b in baselines()}

    # ---- 3. write
    cash = plan.schedule.cash()
    summary = {
        "commit": _commit(root),
        "data": data.summary(panel),
        "plan": {"funded_at_start": plan.funded0, "amortisation_months": plan.amort_months, "cost_bps": plan.cost_bps, "horizon_months": plan.horizon,
                 "schedule_years": plan.schedule.years, "assets": list(ASSETS),
                 "liability_duration_at_4pct": duration(cash, 0.04), "liability_convexity_at_4pct": convexity(cash, 0.04),
                 "first_year_benefit_share_at_4pct": float(cash[:12].sum() / (plan.schedule.unit_value(0.04)))},
        "generator": {"yield_phi": model.phi_y, "yield_mean": model.mean_yield, "inflation_phi": model.phi_pi, "block_months": 12, "months_fitted": int(len(model.shocks))},
        "training": {"seeds": list(SEEDS), "steps": cfg.steps, "batch": cfg.batch, "train_scenarios": cfg.n_train, "validation_scenarios": cfg.n_val,
                     "parameters": int(sum(p.numel() for p in learned.members[0].parameters())), "networks": len({(r["fit"], r["seed"]) for r in training})},
        "simulated": {"scenarios": N_TEST, "seed": TEST_SEED, "policies": table, "learned_minus": compare},
        "model_risk": {"scenarios": N_TEST, "seed": WALK_SEED, "policies": walk_table, "learned_minus": walk_compare},
        "history": {"first_start": FIRST_START, "last_start": LAST_START, "starts": int(len(history)), "refit_every_years": REFIT_YEARS,
                    "block_years": HISTORY_BLOCK, "policies": hist_table, "learned_minus": hist_compare},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out_dir / "fans.json").write_text(json.dumps({"years": list(range(plan.horizon // 12 + 1)), "percentiles": fans, "paths": sample}) + "\n")
    pd.DataFrame(table).T.rename_axis("policy").to_csv(out_dir / "simulated.csv", float_format="%.6f")
    pd.DataFrame(compare).T.rename_axis("learned minus").to_csv(out_dir / "comparisons.csv", float_format="%.6f")
    pd.DataFrame(walk_table).T.rename_axis("policy").to_csv(out_dir / "model_risk.csv", float_format="%.6f")
    history.to_csv(out_dir / "history.csv", index=False, float_format="%.6f")
    pd.DataFrame(training).to_csv(out_dir / "training.csv", index=False, float_format="%.6f")
    log(f"Wrote {out_dir.relative_to(root)}/")
    return summary
