"""Stage 6 robustness runs that need retraining (slow; run once).

* Model size: retrain the walk-forward deep risk model with each d_model in
  ``evaluation.sensitivity_d_model`` and report out-of-sample NLL, GMV vol and
  the two-stage strategy Sharpe. (The end-to-end model is not re-trained per
  size: compute budget. Stated as a limitation.)
* Seeds: re-run the end-to-end walk-forward with each extra seed (same risk
  checkpoints), and report the spread of Sharpe ratios. A deep model whose
  edge is smaller than its seed-to-seed spread has no reliable edge.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from dpc.backtest import run_backtest
from dpc.baselines import get_betas, load_signals
from dpc.config import Config
from dpc.data.panel import Panel
from dpc.deep.features import build_sample_bank
from dpc.deep.train_risk import evaluate_risk, make_ls_deep_twostage, walk_forward_risk
from dpc.logging_utils import get_logger
from dpc.metrics import performance_metrics

log = get_logger(__name__)


def model_size(cfg: Config) -> pd.DataFrame:
    panel = Panel.load(cfg.paths.data_processed)
    signals, betas = load_signals(cfg), get_betas(cfg, panel)
    rows = {}
    for d in cfg.evaluation.sensitivity_d_model:
        rc = dataclasses.replace(cfg.risk_model, d_model=d)
        bank = build_sample_bank(panel, rc.seq_len_days, rc.horizon_days, rc.sample_stride_days)
        fc, _, _ = walk_forward_risk(cfg, panel, bank, rc, cfg.project.seed, tag=f"d{d}", save_models=False)
        summ, _ = evaluate_risk(cfg, panel, bank, fc)
        fn, _ = make_ls_deep_twostage(cfg, fc, panel.universe["sector"].nunique())
        r = run_backtest("tmp", panel, fn, signals, betas, cfg.backtest.oos_start, cfg.backtest.transaction_cost_bps, False)
        m = performance_metrics(r.daily_net, panel.ff, r.turnover, r.costs, cfg.backtest.annualization_days, cfg.signals.newey_west_lags)
        n_params = None
        try:
            from dpc.deep.train_risk import new_model

            n_params = sum(p.numel() for p in new_model(bank, rc).parameters())
        except Exception:  # pragma: no cover
            pass
        rows[f"d_model={d}"] = {
            "n_params": n_params,
            "oos_nll_deep": summ.loc["deep", "mean_oos_nll"],
            "oos_nll_ledoit_wolf": summ.loc["ledoit_wolf", "mean_oos_nll"],
            "gmv_vol_deep": summ.loc["deep", "gmv_realised_vol_ann"],
            "gmv_vol_ledoit_wolf": summ.loc["ledoit_wolf", "gmv_realised_vol_ann"],
            "twostage_sharpe": m["sharpe"],
            "twostage_vol": m["ann_vol"],
        }
        log.info("model size d=%d: %s", d, rows[f"d_model={d}"])
    df = pd.DataFrame(rows).T
    df.to_csv(cfg.paths.reports / "stage6_model_size.csv")
    return df


def seeds(cfg: Config, views: bool = False) -> pd.DataFrame:
    """Re-run the end-to-end walk-forward with each seed. views=True: the fully deep model."""
    from dpc.backtest import load_result
    from dpc.deep.e2e import run_stage5

    panel = Panel.load(cfg.paths.data_processed)
    rows = {}
    base = "deep" if views else "main"
    for s in cfg.evaluation.seeds:
        tag = base if s == cfg.project.seed else (f"deep_seed{s}" if views else f"seed{s}")
        name = "e2e" if tag == "main" else f"e2e_{tag}"
        try:
            r = load_result(name, cfg.paths.results)
        except FileNotFoundError:
            run_stage5(cfg, seed=s, tag=tag, views=views)
            r = load_result(name, cfg.paths.results)
        m = performance_metrics(r.daily_net, panel.ff, r.turnover, r.costs, cfg.backtest.annualization_days, cfg.signals.newey_west_lags)
        rows[f"seed={s}"] = {k: m[k] for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "alpha_t_ff6", "turnover_ann")}
    df = pd.DataFrame(rows).T
    df.loc["mean"] = df.mean()
    df.loc["std"] = df.iloc[:-1].std()
    df.to_csv(cfg.paths.reports / ("stage6_seeds_deep.csv" if views else "stage6_seeds.csv"))
    log.info("Seed robustness:\n%s", df.round(3).to_string())
    return df


def run_robustness(cfg: Config) -> None:
    model_size(cfg)
    seeds(cfg)
    seeds(cfg, views=True)


__all__ = ["run_robustness", "model_size", "seeds", "np"]
