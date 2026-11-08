"""Stages of the wealth study: data, baselines, train, evaluate, site.

Every stage reads ``config.yaml`` and writes its results under
``reports/wealth/`` (tables, summaries) and ``reports/results/wealth/`` (one
parquet per strategy: monthly net and gross return, turnover and the weights
held each month), so later stages and the website read committed files.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .baselines import all_baselines
from .data import IndustryPanel, load_panel
from .engine import RunResult, run
from .features import Features, build_features
from .train import VARIANTS, EnsemblePolicy, TrainSettings, run_policy, train_walk_forward

log = get_logger(__name__)


def settings(cfg: Config) -> TrainSettings:
    w = cfg.wealth
    train = {k: v for k, v in w.train.__dict__.items() if k not in ("seeds", "workers")}
    return TrainSettings(**w.model.__dict__, **train, max_tilt=w.max_tilt, cost_bps=w.cost_bps)


def load(cfg: Config, refresh: bool = False) -> tuple[IndustryPanel, Features]:
    w = cfg.wealth
    panel = load_panel(w.base_url, w.industry_file, w.factor_file, cfg.paths.data_raw / "french", refresh=refresh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing rows before the first full lookback
        feats = build_features(panel, clip=w.feature_clip)
    return panel, feats


def results_dir(cfg: Config):
    d = cfg.paths.reports / "results" / "wealth"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_result(cfg: Config, panel: IndustryPanel, res: RunResult, extra: dict[str, np.ndarray] | None = None) -> None:
    df = pd.DataFrame({"net": res.net, "gross": res.gross, "turnover": res.turnover}, index=res.dates)
    for k, v in (extra or {}).items():
        df[k] = v
    w = pd.DataFrame(res.weights, index=res.dates, columns=[f"w:{n}" for n in panel.names])
    pd.concat([df, w], axis=1).to_parquet(results_dir(cfg) / f"{res.name}.parquet")


def load_result(cfg: Config, name: str) -> pd.DataFrame:
    return pd.read_parquet(results_dir(cfg) / f"{name}.parquet")


def stage_data(cfg: Config, refresh: bool = False) -> None:
    panel, feats = load(cfg, refresh=refresh)
    w = panel.cap_start / panel.cap_start.sum(1, keepdims=True)
    cap_weighted = (w * panel.returns).sum(1)
    french = panel.mkt_rf + panel.rf
    summary = {
        "industries": panel.names,
        "first_month": f"{panel.dates[0]:%Y-%m}", "last_month": f"{panel.dates[-1]:%Y-%m}", "months": panel.n_months,
        "first_decision": f"{panel.dates[feats.first_valid]:%Y-%m}",
        "check_cap_weighted_vs_french_market": {
            "correlation": float(np.corrcoef(cap_weighted, french)[0, 1]),
            "mean_abs_diff_monthly": float(np.abs(cap_weighted - french).mean()),
        },
    }
    out = cfg.paths.reports / "wealth"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_summary.json").write_text(json.dumps(summary, indent=1))
    log.info("Data summary written: %s", out / "data_summary.json")


def stage_baselines(cfg: Config) -> None:
    w = cfg.wealth
    panel, _ = load(cfg)
    start, end = panel.index_of(w.oos_start), panel.n_months - 1
    for name, decide in all_baselines(panel, w.rebalance_month, w.min_variance_max_weight, w.max_tilt, w.growth_shrink).items():
        res = run(panel, decide, start, end, w.cost_bps, name)
        save_result(cfg, panel, res)
        log.info("%s: $1 -> $%.0f", name, res.wealth()[-1])


def stage_train(cfg: Config, variants: list[str]) -> None:
    w = cfg.wealth
    panel, feats = load(cfg)
    s = settings(cfg)
    for name in variants:
        v = VARIANTS[name]
        seeds = w.train.seeds if name == "deep" else w.evaluation.ablation_seeds
        model_dir = cfg.paths.models / "wealth" / name
        train_walk_forward(panel, feats, s, v, seeds, w.oos_start, model_dir, w.train.workers)
        policy = EnsemblePolicy.load(panel, feats, s, v, seeds, w.oos_start, model_dir)
        res = run_policy(panel, policy, w.oos_start, w.cost_bps, name)
        save_result(cfg, panel, res, {"kappa": np.array(policy.kappas)})
        log.info("%s (ensemble of %d): $1 -> $%.0f", name, len(seeds), res.wealth()[-1])
        if name == "deep":  # each seed on its own, for the robustness table
            for seed in seeds:
                single = EnsemblePolicy.load(panel, feats, s, v, [seed], w.oos_start, model_dir)
                r1 = run_policy(panel, single, w.oos_start, w.cost_bps, f"deep_seed{seed}")
                save_result(cfg, panel, r1, {"kappa": np.array(single.kappas)})
                log.info("deep seed %d alone: $1 -> $%.0f", seed, r1.wealth()[-1])
