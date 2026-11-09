"""Evaluation of the wealth study, as fixed in docs/wealth_protocol.md.

Reads the per-strategy results in reports/results/wealth/, writes tables to
reports/wealth/ and a machine-readable summary (summary.json) that the README,
the research note and the website quote. Nothing here is fitted: every number
is a statistic of the out-of-sample monthly returns.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import Config
from ..logging_utils import get_logger
from .data import IndustryPanel
from .engine import run
from .names import ABLATIONS, LABELS, MAIN, REFERENCE, SECONDARY  # noqa: F401 (re-exported for the report)
from .pipeline import load, load_result, results_dir

log = get_logger(__name__)


def _ann_log_growth(net: np.ndarray) -> float:
    return float(np.log1p(net).mean() * 12)


def metrics(df: pd.DataFrame, rf: pd.Series, w_mkt: pd.DataFrame | None = None) -> dict[str, float]:
    net = df["net"].to_numpy()
    years = len(net) / 12
    wealth = np.cumprod(1 + net)
    ex = net - rf.reindex(df.index).to_numpy()
    yearly = (1 + df["net"]).groupby(df.index.year).prod() - 1
    full_years = yearly[df.index.to_series().groupby(df.index.year).count() == 12]
    one_way = df["turnover"].iloc[1:].sum() / 2 / years           # the first purchase from cash is not turnover
    out = {
        "terminal_wealth": float(wealth[-1]),
        "cagr": float(wealth[-1] ** (1 / years) - 1),
        "log_growth": _ann_log_growth(net),
        "volatility": float(net.std(ddof=1) * np.sqrt(12)),
        "sharpe": float(ex.mean() / ex.std(ddof=1) * np.sqrt(12)),
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "worst_year": float(full_years.min()),
        "best_year": float(full_years.max()),
        "turnover_per_year": float(one_way),
        "holding_years": float(1 / one_way) if one_way > 0 else float("inf"),
    }
    out["calmar"] = out["cagr"] / abs(out["max_drawdown"])
    if w_mkt is not None:
        w = df[[c for c in df.columns if c.startswith("w:")]].to_numpy()
        out["active_share"] = float(0.5 * np.abs(w - w_mkt.to_numpy()).sum(1).mean())
    return out


def newey_west(x: np.ndarray, lags: int) -> tuple[float, float]:
    res = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def block_bootstrap_mean(x: np.ndarray, block: int, draws: int, seed: int) -> tuple[float, float]:
    """95% interval for the mean by the moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(x)
    starts_max = n - block + 1
    k = int(np.ceil(n / block))
    means = np.empty(draws)
    for i in range(draws):
        idx = (rng.integers(0, starts_max, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[i] = x[idx].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compare(a: pd.DataFrame, b: pd.DataFrame, lags: int, block: int, draws: int, seed: int) -> dict[str, float | str]:
    """a minus b in annualised net log growth, with the protocol's verdict."""
    d = np.log1p(a["net"].to_numpy()) - np.log1p(b["net"].to_numpy())
    mean, t = newey_west(d, lags)
    lo, hi = block_bootstrap_mean(d, block, draws, seed)
    verdict = "better" if t >= 2 and lo > 0 else "worse" if t <= -2 and hi < 0 else "no detectable difference"
    return {"diff_log_growth": 12 * mean, "t_nw": t, "ci_low": 12 * lo, "ci_high": 12 * hi, "verdict": verdict}


def ff3_alpha(df: pd.DataFrame, panel: IndustryPanel, lags: int) -> dict[str, float]:
    idx = panel.dates.get_indexer(df.index)
    y = df["net"].to_numpy() - panel.rf[idx]
    X = sm.add_constant(np.column_stack([panel.mkt_rf[idx], panel.smb[idx], panel.hml[idx]]))
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha_annual": float(res.params[0] * 12), "alpha_t": float(res.tvalues[0]), "beta_mkt": float(res.params[1]),
            "beta_smb": float(res.params[2]), "beta_hml": float(res.params[3])}


def decades(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, df in results.items():
        g = (1 + df["net"]).groupby((df.index.year // 10) * 10)
        n = g.count()
        rows[name] = (g.prod() ** (12 / n) - 1)
    out = pd.DataFrame(rows).T
    out.columns = [f"{c}s" for c in out.columns]
    return out


def rolling_beat_share(df: pd.DataFrame, mkt: pd.DataFrame, years: int) -> float:
    w = 12 * years
    a = np.log1p(df["net"]).rolling(w).sum()
    b = np.log1p(mkt["net"]).rolling(w).sum()
    valid = a.notna()
    return float((a[valid] > b[valid]).mean())


def reprice(df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Same decisions, different cost: net = (1 - c * turnover)(1 + gross) - 1."""
    out = df.copy()
    out["net"] = (1 - cost_bps / 1e4 * df["turnover"]) * (1 + df["gross"]) - 1
    return out


def oracle_wealth(panel: IndustryPanel, start: int, max_tilt: float, cost_bps: float) -> float:
    """A deliberately cheating policy: tilts toward next month's best industries. Calibrates the simulator."""
    w_mkt = panel.market_weights()

    def decide(t: int, w_drift: np.ndarray) -> np.ndarray:
        nxt = panel.returns[t + 1]
        ranks = nxt.argsort().argsort() / (len(nxt) - 1)
        w = w_mkt[t] * max_tilt ** (2 * ranks - 1)
        return w / w.sum()

    return float(run(panel, decide, start, panel.n_months - 1, cost_bps, "oracle").wealth()[-1])


def run_evaluation(cfg: Config) -> dict:
    w = cfg.wealth
    ev = w.evaluation
    panel, _ = load(cfg)
    rf = pd.Series(panel.rf, index=panel.dates)
    available = {p.stem for p in results_dir(cfg).glob("*.parquet")}
    results = {name: load_result(cfg, name) for name in MAIN + ABLATIONS + REFERENCE if name in available}
    seeds = {p: load_result(cfg, p) for p in sorted(available) if p.startswith("deep_seed")}
    mkt = results["market"]
    start = panel.index_of(w.oos_start)
    w_mkt = pd.DataFrame(panel.market_weights()[start - 1:panel.n_months - 1], index=mkt.index)

    perf = {name: metrics(df, rf, w_mkt) for name, df in results.items()}
    for name, df in results.items():
        perf[name].update(ff3_alpha(df, panel, ev.nw_lags))
        perf[name]["beat_market_rolling"] = rolling_beat_share(df, mkt, ev.rolling_years)
    table = pd.DataFrame(perf).T
    table.insert(0, "label", [LABELS[n] for n in table.index])
    out_dir = cfg.paths.reports / "wealth"
    table.to_csv(out_dir / "performance.csv")

    comps = {}
    if "deep" in results:
        comps["deep - market"] = compare(results["deep"], mkt, ev.nw_lags, ev.bootstrap_block_months, ev.bootstrap_draws, cfg.project.seed)
        for name in SECONDARY:
            comps[f"deep - {name}"] = compare(results["deep"], results[name], ev.nw_lags, ev.bootstrap_block_months, ev.bootstrap_draws, cfg.project.seed)
    for name in ["equal_weight", "min_variance", "risk_parity", "growth_classical", "momentum_tilt", *ABLATIONS, *REFERENCE]:
        if name in results:
            comps[f"{name} - market"] = compare(results[name], mkt, ev.nw_lags, ev.bootstrap_block_months, ev.bootstrap_draws, cfg.project.seed)
    for name in ABLATIONS:  # each ablation against the full network with the same two seeds
        if name in results and "deep_pair" in results:
            comps[f"{name} - deep_pair"] = compare(results[name], results["deep_pair"], ev.nw_lags, ev.bootstrap_block_months, ev.bootstrap_draws, cfg.project.seed)
    pd.DataFrame(comps).T.to_csv(out_dir / "comparisons.csv")

    dec = decades(results)
    dec.to_csv(out_dir / "decades.csv")

    costs = {}
    for name in ("deep", "market", "momentum_tilt", "equal_weight"):
        if name not in results:
            continue
        for c in ev.sensitivity_cost_bps:
            df = reprice(results[name], c)
            m = metrics(df, rf)
            costs[f"{name}@{c:g}bps"] = {"terminal_wealth": m["terminal_wealth"], "cagr": m["cagr"],
                                        "vs_market_log_growth": _ann_log_growth(df["net"].to_numpy()) - _ann_log_growth(reprice(mkt, c)["net"].to_numpy())}
    pd.DataFrame(costs).T.to_csv(out_dir / "costs.csv")

    seed_rows = {name: metrics(df, rf) for name, df in seeds.items()}
    if seed_rows:
        pd.DataFrame(seed_rows).T.to_csv(out_dir / "seeds.csv")

    summary = {
        "oos": {"start": f"{results['market'].index[0]:%Y-%m}", "end": f"{results['market'].index[-1]:%Y-%m}", "months": len(mkt)},
        "cost_bps": w.cost_bps,
        "performance": {k: {kk: (None if not np.isfinite(vv) else vv) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in perf.items()},
        "comparisons": comps,
        "decades": json.loads(dec.to_json()),
        "costs": costs,
        "seeds": seed_rows,
        "oracle_terminal_wealth": oracle_wealth(panel, start, w.max_tilt, w.cost_bps),
    }
    if "deep" in results and "kappa" in results["deep"].columns:
        k = results["deep"]["kappa"].iloc[1:]
        summary["deep_trade_fraction"] = {"mean": float(k.mean()), "median": float(k.median()), "min": float(k.min()), "max": float(k.max())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("Evaluation written to %s", out_dir)
    return summary
