"""The risk study's stages: data, classical forecasts, evaluation (docs/risk_protocol.md).

    python -m dpc.cli risk-study        # risk-data, risk-classical, risk-evaluate, risk-site
    python scripts/risk_tsfm_forecast.py ...   # the foundation model's forecasts, in tsfm's environment

``risk-evaluate`` scores every forecast file it finds in reports/risk/forecasts/. The classical
four are required; a foundation-model checkpoint whose files are not there is reported as not run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from ..wealth.baselines import market
from ..wealth.engine import RunResult, run
from ..wealth.frequency import load_daily
from ..wealth.pipeline import load as load_wealth
from . import forecasters as fc
from . import portfolio as pf
from . import scores as sc
from .data import RiskPanel, build, save_npz

log = get_logger(__name__)

# The two checkpoints the protocol names. `files` are the forecast files each one writes.
MODELS = {
    "cpu_s0": {"label": "CPU model", "parameters": 833_440, "sha256": "d41b77fd19881097a4d8fc400b4f198a8edbf242a6691147308d5ce2f4a08a91",
               "weights": "public (tsfm-from-scratch results/runs/cpu_s0_seed0/model.pt)"},
    "patch_s2": {"label": "T4 model", "parameters": 10_083_392, "sha256": "85c46256cefa665b7eb0822f9877831563df85e22345670b40c4a115c16d0fd2",
                 "weights": "not public yet"},
}
ROBUSTNESS = ("cpu_s0_seed1", "cpu_s0_seed2")   # the CPU model's other two pretraining seeds
VARIANTS = {"zero": "zero-shot", "ft": "fine-tuned"}
LABELS = {"last": "Last month", "mean60": "60-month mean", "ewma": "RiskMetrics EWMA", "har": "HAR regression",
          "market": "Market (reference)", "oracle": "Oracle (not investable)"}
RULES = {"min_variance": "Minimum variance", "inverse_volatility": "Inverse volatility"}


def model_file(variant: str, checkpoint: str) -> str:
    return f"tsfm_{variant}_{checkpoint}"


def label(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    _, variant, checkpoint = name.split("_", 2)
    base = MODELS[checkpoint]["label"] if checkpoint in MODELS else f"CPU model, seed {checkpoint[-1]}"
    return f"{base}, {VARIANTS[variant]}"


def out_dir(cfg: Config) -> Path:
    d = cfg.paths.reports / "risk"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(cfg: Config, refresh: bool = False) -> tuple:
    panel, _ = load_wealth(cfg, refresh=refresh)
    daily = load_daily(cfg.wealth.base_url, cfg.paths.data_raw / "french", panel.names, refresh=refresh)
    return panel, build(panel, daily)


def origin_labels(rp: RiskPanel, idx: np.ndarray) -> list[str]:
    return [f"{rp.dates[t]:%Y-%m}" for t in idx]


def stage_data(cfg: Config, refresh: bool = False) -> None:
    panel, rp = load(cfg, refresh=refresh)
    idx = fc.test_origins(rp, cfg.wealth.oos_start)
    save_npz(rp, out_dir(cfg) / "panel.npz")
    summary = {
        "source": "Kenneth French, 30 Industry Portfolios, daily and monthly, value-weighted",
        "industries": rp.names, "first_month": f"{rp.dates[0]:%Y-%m}", "last_month": f"{rp.dates[-1]:%Y-%m}",
        "daily_returns": int(len(rp.daily)), "first_day": f"{rp.daily.index[0]:%Y-%m-%d}", "last_day": f"{rp.daily.index[-1]:%Y-%m-%d}",
        "first_scored_month": f"{rp.dates[idx[0] + 1]:%Y-%m}", "last_scored_month": f"{rp.dates[idx[-1] + 1]:%Y-%m}", "scored_months": int(len(idx)),
        "missing_industry_months_from_1949_12": int(np.isnan(rp.rv[rp.index_of("1949-12"):]).sum()),
        "median_realized_volatility_scored": float(np.median(rp.rv[idx + 1])),
    }
    (out_dir(cfg) / "data_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log.info("Risk panel: %d months x %d industries; %d scored months", len(rp.dates), len(rp.names), len(idx))


def stage_classical(cfg: Config) -> None:
    _, rp = load(cfg)
    idx = fc.test_origins(rp, cfg.wealth.oos_start)
    for name, point in fc.point_forecasts(rp).items():
        q = fc.quantiles_from_errors(point, rp.rv)[idx]
        if not np.isfinite(q).all():
            raise ValueError(f"{name}: a scored month has no forecast")
        fc.save_forecast(out_dir(cfg) / "forecasts" / f"{name}.npz", origin_labels(rp, idx), rp.names, q, {"kind": "classical", "forecaster": name})
        log.info("%s: %d months x %d industries", name, *q.shape[:2])


def available(cfg: Config, rp: RiskPanel, idx: np.ndarray) -> dict[str, fc.Forecast]:
    """Every forecast file present: the classical four (required), then the model files."""
    folder, origins = out_dir(cfg) / "forecasts", origin_labels(rp, idx)
    out = {name: fc.load_forecast(folder / f"{name}.npz", origins, rp.names) for name in fc.CLASSICAL}
    for checkpoint in (*MODELS, *ROBUSTNESS):
        for variant in VARIANTS:
            path = folder / f"{model_file(variant, checkpoint)}.npz"
            if path.exists():
                out[path.stem] = fc.load_forecast(path, origins, rp.names)
    return out


def portfolio_stats(res: RunResult, rvar: np.ndarray) -> dict[str, float]:
    wealth = res.wealth()
    return {
        "realized_volatility": float(np.sqrt(12.0 * rvar.mean())),          # yearly, from daily returns
        "volatility_monthly_returns": float(res.net.std(ddof=1) * np.sqrt(12.0)),
        "terminal_wealth": float(wealth[-1]), "cagr": float(wealth[-1] ** (12.0 / len(wealth)) - 1.0),
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
        "traded_per_year": float(12.0 * res.turnover[1:].mean()),            # the first month buys the whole portfolio from cash
    }


def decade_table(y: np.ndarray, forecasts: dict[str, fc.Forecast], years: np.ndarray) -> dict[str, dict[str, float]]:
    """Pooled quantile loss of each forecaster by decade of the scored month, as a ratio to HAR."""
    out: dict[str, dict[str, float]] = {}
    for start in range(int(years.min()) // 10 * 10, int(years.max()) + 1, 10):
        rows = (years >= start) & (years < start + 10)
        har = sc.month_sums(y[rows], forecasts["har"].q[rows])["wql_num"].sum()
        out[f"{start}s"] = {name: float(sc.month_sums(y[rows], f.q[rows])["wql_num"].sum() / har) for name, f in forecasts.items()}
    return out


def evaluate(cfg: Config, panel, rp: RiskPanel, forecasts: dict[str, fc.Forecast]) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Every number of the study from the forecasts. Returns the summary and the portfolio runs."""
    w, ev = cfg.wealth, cfg.wealth.evaluation
    idx = fc.test_origins(rp, w.oos_start)
    y = rp.rv[idx + 1]
    if not np.isfinite(y).all():
        raise ValueError("a scored month has no realized volatility")
    boot = sc.block_indices(len(idx), ev.bootstrap_block_months, ev.bootstrap_draws, cfg.project.seed)
    sums = {name: sc.month_sums(y, f.q) for name, f in forecasts.items()}
    accuracy = {name: {**sc.pooled(s), "vs_har": sc.against(s, sums["har"], boot)} for name, s in sums.items()}

    start, end, first = int(idx[0]) + 1, int(idx[-1]) + 1, int(idx[0])
    sigmas = {name: f.median for name, f in forecasts.items()}
    sigmas["oracle"] = y                                                     # next month's realized volatility: not a forecast
    runs: dict[str, pd.DataFrame] = {}
    portfolios: dict[str, dict] = {}

    def record(key: str, res: RunResult) -> np.ndarray:
        rvar = pf.realized_variance(rp.daily, res.dates, res.weights)
        runs[key] = pd.DataFrame({"net": res.net, "gross": res.gross, "turnover": res.turnover, "realized_variance": rvar}, index=res.dates)
        return rvar

    mkt = run(panel, market(panel), start, end, w.cost_bps, "market")
    references = {"market": portfolio_stats(mkt, record("market", mkt))}
    for rule in RULES:
        rows, rvars = {}, {}
        for name, sigma in sigmas.items():
            decide = (pf.min_variance_rule(panel, sigma, first, w.min_variance_max_weight) if rule == "min_variance"
                      else pf.inverse_volatility_rule(sigma, first))
            res = run(panel, decide, start, end, w.cost_bps, f"{rule}:{name}")
            rvars[name] = record(f"{rule}_{name}", res)
            rows[name] = portfolio_stats(res, rvars[name])
        for name in rows:
            if name != "har":
                rows[name]["vs_har"] = pf.compare_variance(rvars[name], rvars["har"], ev.nw_lags, ev.bootstrap_block_months, ev.bootstrap_draws, cfg.project.seed)
        portfolios[rule] = rows

    years = np.array([rp.dates[t + 1].year for t in idx])
    models = {}
    for checkpoint, info in MODELS.items():
        ft, zero = model_file("ft", checkpoint), model_file("zero", checkpoint)
        entry = {**info, "status": "run" if ft in forecasts and zero in forecasts else "not run"}
        if entry["status"] == "run":
            meta = forecasts[ft].meta
            refits = meta.get("refits", [])
            entry.update({
                "accuracy_verdict": accuracy[ft]["vs_har"]["wql"]["verdict"],
                "portfolio_verdict": portfolios["min_variance"][ft]["vs_har"]["verdict"],
                "earns_its_place": accuracy[ft]["vs_har"]["wql"]["verdict"] == "better" and portfolios["min_variance"][ft]["vs_har"]["verdict"] == "lower risk",
                "refits": len(refits), "refits_kept_pretrained": sum(1 for r in refits if r["kept_step"] == 0),
                "tsfm_commit": meta.get("tsfm_commit"), "torch": meta.get("torch"), "checkpoint_sha256": meta.get("checkpoint_sha256"),
            })
        models[checkpoint] = entry
    summary = {
        "protocol": "docs/risk_protocol.md",
        "window": {"first_month": f"{rp.dates[start]:%Y-%m}", "last_month": f"{rp.dates[end]:%Y-%m}", "months": int(len(idx)), "industries": len(rp.names)},
        "tests": {"bootstrap_block_months": ev.bootstrap_block_months, "bootstrap_draws": ev.bootstrap_draws, "nw_lags": ev.nw_lags, "seed": cfg.project.seed,
                  "cost_bps": w.cost_bps, "max_weight": w.min_variance_max_weight},
        "labels": {name: label(name) for name in (*forecasts, "market", "oracle")},
        "rules": RULES,
        "accuracy": accuracy,
        "decades_wql_vs_har": decade_table(y, forecasts, years),
        "portfolios": portfolios,
        "references": references,
        "models": models,
    }
    return summary, runs


def stage_evaluate(cfg: Config) -> None:
    from .report import write_reports

    panel, rp = load(cfg)
    forecasts = available(cfg, rp, fc.test_origins(rp, cfg.wealth.oos_start))
    summary, runs = evaluate(cfg, panel, rp, forecasts)
    out = out_dir(cfg)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    results = cfg.paths.reports / "results" / "risk"
    results.mkdir(parents=True, exist_ok=True)
    for key, df in runs.items():
        df.to_parquet(results / f"{key}.parquet")
    write_reports(cfg.root, summary)
    log.info("Risk study evaluated: %d forecasters, %d portfolios", len(forecasts), len(runs))
