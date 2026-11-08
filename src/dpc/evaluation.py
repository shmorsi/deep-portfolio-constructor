"""Stage 6: rigorous, walk-forward-only evaluation.

Headline: **where does each portfolio's risk come from?** Intended stock bets
(stock-specific risk) versus unintended factor exposure (style + sector).

To avoid grading the deep model with its own homework, risk is decomposed
with an independent, evaluation-only Barra-style model:

* exposures X_t (known at month-end t): rolling betas to the six
  Fama-French/momentum factors + 11 GICS sector dummies;
* each day d in the following month: cross-sectional regression
  r_d = X_t f_d + u_d gives factor returns f_d and specific returns u_d;
* ex-ante at rebalance t: F = cov(f) and specific variances from the
  trailing window (data <= t);  risk(w) = e'Fe + sum w_i^2 delta_i, e = X_t'w;
* ex-post: realised portfolio return split into e'f_d (factor) and w'u_d
  (stock-specific).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dpc.backtest import BacktestResult, load_result, run_backtest
from dpc.baselines import LONG_ONLY, get_betas, load_signals, make_ls_classical, summarize
from dpc.config import Config
from dpc.data.panel import Panel
from dpc.exposures import STYLE_FACTORS
from dpc.logging_utils import get_logger
from dpc.metrics import metrics_table, performance_metrics, to_monthly
from dpc.stats import newey_west_mean

log = get_logger(__name__)

LS_ORDER = ["ls_naive", "ls_classical", "ls_deep_twostage", "e2e"]
LABELS = {
    "equal_weight": "Equal weight (LO)",
    "min_variance": "Min variance (LO)",
    "mean_variance_lw": "Mean-variance LW (LO)",
    "risk_parity": "Risk parity (LO)",
    "hrp": "HRP (LO)",
    "ls_naive": "L/S naive signal",
    "ls_classical": "L/S classical (LW + optimiser)",
    "ls_deep_twostage": "L/S deep risk + optimiser",
    "e2e": "L/S end-to-end deep",
}


# --------------------------------------------------------------------------- #
# Evaluation-only Barra-style risk model
# --------------------------------------------------------------------------- #
@dataclass
class EvalRiskModel:
    factor_names: list[str]
    exposures: dict[pd.Timestamp, pd.DataFrame]  # month-end -> (ticker x factor)
    f: pd.DataFrame  # day x factor returns
    u: pd.DataFrame  # day x ticker specific returns
    month_of_day: pd.Series  # day -> month-end whose exposures were used


def build_eval_risk_model(panel: Panel, betas: dict[str, pd.DataFrame]) -> EvalRiskModel:
    sec = panel.universe["sector"]
    sectors = sorted(sec.unique())
    dummies = pd.DataFrame({f"sector:{s}": (sec == s).astype(float) for s in sectors})
    names = [f"style:{f}" for f in STYLE_FACTORS] + list(dummies.columns)
    me = panel.month_ends
    R = panel.returns
    f_rows, u_rows, month_of = [], [], []
    exposures = {}
    for k, t in enumerate(me[:-1]):
        X = pd.concat([pd.DataFrame({f"style:{f}": betas[f].loc[t] for f in STYLE_FACTORS}), dummies], axis=1)
        ok = panel.eligible.loc[t] & X.notna().all(axis=1)
        if ok.sum() < 3 * len(names):
            continue
        Xo = X[ok]
        exposures[t] = Xo
        block = R.loc[(R.index > t) & (R.index <= me[k + 1]), Xo.index]
        Y = block.fillna(0.0).to_numpy()
        coef, *_ = np.linalg.lstsq(Xo.to_numpy(), Y.T, rcond=None)  # (F, D)
        resid = Y - (Xo.to_numpy() @ coef).T
        f_rows.append(pd.DataFrame(coef.T, index=block.index, columns=names))
        u_rows.append(pd.DataFrame(resid, index=block.index, columns=Xo.index))
        month_of.append(pd.Series(t, index=block.index))
    return EvalRiskModel(
        factor_names=names,
        exposures=exposures,
        f=pd.concat(f_rows),
        u=pd.concat(u_rows).reindex(columns=panel.tickers),
        month_of_day=pd.concat(month_of),
    )


def decompose(res: BacktestResult, erm: EvalRiskModel, window: int, horizon: int = 21) -> tuple[pd.DataFrame, dict]:
    """Ex-ante (per rebalance) and ex-post (realised) risk split for one strategy."""
    rows = []
    fac_parts, spec_parts = [], []
    for t, w in res.weights.iterrows():
        if t not in erm.exposures:
            continue
        X = erm.exposures[t]
        wv = w.reindex(X.index).fillna(0.0)
        if np.abs(w.drop(X.index, errors="ignore")).sum() > 1e-6:
            continue  # holds a name the eval model cannot price (should not happen)
        hist_f = erm.f.loc[: t].iloc[-window:]
        hist_u = erm.u.loc[: t].iloc[-window:]
        if len(hist_f) < window // 2:
            continue
        F = hist_f.cov().to_numpy() * horizon
        delta = hist_u[X.index].var().fillna(hist_u.var().median()).to_numpy() * horizon
        e = X.to_numpy().T @ wv.to_numpy()
        Fe = F @ e
        contrib = e * Fe
        style = contrib[: len(STYLE_FACTORS)].sum()
        sector = contrib[len(STYLE_FACTORS) :].sum()
        spec = float((wv.to_numpy() ** 2 * delta).sum())
        total = style + sector + spec
        row = {"date": t, "total_vol_ann": np.sqrt(total * 12), "share_specific": spec / total,
               "share_style": style / total, "share_sector": sector / total}
        row.update({f"exp_{n}": v for n, v in zip(X.columns, e)})
        rows.append(row)
        days = erm.month_of_day.index[erm.month_of_day == t]
        fac_parts.append(pd.Series(erm.f.loc[days].to_numpy() @ e, index=days))
        spec_parts.append(pd.Series(erm.u.loc[days, X.index].fillna(0).to_numpy() @ wv.to_numpy(), index=days))
    ex_ante = pd.DataFrame(rows).set_index("date")
    fac, spec = pd.concat(fac_parts), pd.concat(spec_parts)
    vf, vs, cv = fac.var(), spec.var(), 2 * fac.cov(spec)
    ex_post = {"realised_share_factor": vf / (vf + vs), "realised_share_specific": vs / (vf + vs),
               "realised_cov_term_share": cv / (vf + vs + cv)}
    return ex_ante, ex_post


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def period_table(results: dict[str, BacktestResult], panel: Panel, year: int, ann: int) -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        x = r.excess_returns(panel.ff["RF"])
        x = x[x.index.year == year]
        if len(x) < 20:
            continue
        wealth = (1 + x).cumprod()
        rows[LABELS.get(name, name)] = {
            "return": float(wealth.iloc[-1] - 1),
            "vol_ann": float(x.std() * np.sqrt(ann)),
            "sharpe": float(x.mean() / x.std() * np.sqrt(ann)),
            "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
        }
    return pd.DataFrame(rows).T


def cost_sensitivity(results: dict[str, BacktestResult], panel: Panel, cfg: Config) -> pd.DataFrame:
    """Re-price the same trades at different cost levels (weights held fixed; the
    optimiser's own turnover penalty was set for 10 bps, noted in the report)."""
    out = {}
    for name, r in results.items():
        row = {}
        for bps in cfg.evaluation.sensitivity_cost_bps:
            net = r.daily_gross.copy()
            net.loc[r.turnover.index] -= r.turnover * bps / 1e4
            x = net - panel.ff["RF"].reindex(net.index).fillna(0) if r.long_only else net
            row[f"sharpe@{bps:g}bps"] = float(x.mean() / x.std() * np.sqrt(cfg.backtest.annualization_days))
        out[LABELS.get(name, name)] = row
    return pd.DataFrame(out).T


def difference_test(a: BacktestResult, b: BacktestResult, lags: int) -> dict[str, float]:
    """Is strategy a's monthly excess return higher than b's? (NW t-stat on the difference)."""
    ma, mb_ = to_monthly(a.daily_net), to_monthly(b.daily_net)
    d = (ma - mb_).dropna()
    mean, t, n = newey_west_mean(d, lags)
    return {"mean_monthly_diff": mean, "t_stat": t, "n_months": n}


def oracle_check(panel: Panel, signals: dict, betas: dict, cfg: Config) -> dict[str, float]:
    """Calibration of what look-ahead looks like: trade on NEXT month's return."""
    from dpc.signals import forward_returns, monthly_returns

    fwd = forward_returns(monthly_returns(panel.returns, panel.month_ends))

    def fn(view, w0):
        z = fwd.loc[view.date, view.investable]  # deliberately illegal
        z = (z - z.mean()).fillna(0)
        return z / z.abs().sum() * cfg.optimizer.gross_leverage

    r = run_backtest("oracle", panel, fn, signals, betas, cfg.backtest.oos_start, cfg.backtest.transaction_cost_bps, False)
    return {"oracle_sharpe": float(r.daily_net.mean() / r.daily_net.std() * np.sqrt(cfg.backtest.annualization_days))}


def truncation_check(panel: Panel, signals: dict, betas: dict, cfg: Config, T: pd.Timestamp) -> bool:
    """Runtime look-ahead check on the real data: scramble every return after T and
    confirm ls_classical's weights up to T are bit-for-bit unchanged."""
    n_sec = panel.universe["sector"].nunique()
    end = T + pd.DateOffset(months=3)
    fn, _ = make_ls_classical(cfg, len(panel.tickers), n_sec)
    base = run_backtest("a", panel, fn, signals, betas, cfg.backtest.oos_start, 10.0, False, end=end)
    p2 = dataclasses.replace(panel, returns=panel.returns.copy())
    fut = p2.returns.index > T
    rng = np.random.default_rng(0)
    p2.returns.loc[fut] = rng.normal(0, 0.05, (int(fut.sum()), p2.returns.shape[1]))
    fn2, _ = make_ls_classical(cfg, len(panel.tickers), n_sec)
    alt = run_backtest("b", p2, fn2, signals, betas, cfg.backtest.oos_start, 10.0, False, end=end)
    a, b = base.weights.loc[:T], alt.weights.loc[:T]
    return bool(np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-7)) and not np.allclose(
        base.weights.loc[T:].iloc[1:].to_numpy(), alt.weights.loc[T:].iloc[1:].to_numpy(), atol=1e-4
    )


def feature_truncation_check(panel: Panel, cfg: Config, T: pd.Timestamp) -> bool:
    """Deep-model inputs at origins <= T must not change when the future changes."""
    from dpc.deep.features import build_sample_bank

    rc = cfg.risk_model
    a = build_sample_bank(panel, rc.seq_len_days, rc.horizon_days, rc.sample_stride_days)
    p2 = dataclasses.replace(panel, returns=panel.returns.copy())
    p2.returns.loc[p2.returns.index > T] = 0.123
    b = build_sample_bank(p2, rc.seq_len_days, rc.horizon_days, rc.sample_stride_days)
    k = a.origins <= T
    return bool(np.array_equal(a.x[k], b.x[k]) and np.array_equal(a.static[k], b.static[k]) and np.array_equal(a.mask_in[k], b.mask_in[k]))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def load_all_results(cfg: Config) -> dict[str, BacktestResult]:
    names = [*LONG_ONLY, *LS_ORDER]
    out = {}
    for n in names:
        try:
            out[n] = load_result(n, cfg.paths.results)
        except FileNotFoundError:
            log.warning("Result %s not found; skipping", n)
    return out


def run_stage6(cfg: Config) -> dict:
    from dpc import report_figures

    panel = Panel.load(cfg.paths.data_processed)
    signals, betas = load_signals(cfg), get_betas(cfg, panel)
    results = load_all_results(cfg)
    ann, lags = cfg.backtest.annualization_days, cfg.signals.newey_west_lags
    rep = cfg.paths.reports
    out: dict = {}

    # 1. Headline performance (net of costs, OOS only)
    perf = summarize(list(results.values()), panel, cfg)
    perf.index = [LABELS.get(i, i) for i in perf.index]
    gross = {LABELS.get(n, n): performance_metrics(
        (r.daily_gross - panel.ff["RF"].reindex(r.daily_gross.index).fillna(0)) if r.long_only else r.daily_gross,
        panel.ff, None, None, ann, lags)["sharpe"] for n, r in results.items()}
    perf.insert(3, "sharpe_gross", pd.Series(gross))
    perf.to_csv(rep / "stage6_performance.csv")
    out["performance"] = perf

    # 2. Risk decomposition (headline)
    erm = build_eval_risk_model(panel, betas)
    decomp_rows, exposures_ts = {}, {}
    for name, r in results.items():
        ex_ante, ex_post = decompose(r, erm, cfg.evaluation.risk_model_window_days)
        decomp_rows[LABELS.get(name, name)] = {
            "exante_share_specific": ex_ante["share_specific"].mean(),
            "exante_share_style": ex_ante["share_style"].mean(),
            "exante_share_sector": ex_ante["share_sector"].mean(),
            "exante_vol_ann": ex_ante["total_vol_ann"].mean(),
            **ex_post,
            "mean_abs_beta_mkt": ex_ante["exp_style:Mkt-RF"].abs().mean(),
            "mean_abs_beta_smb": ex_ante["exp_style:SMB"].abs().mean(),
            "mean_abs_beta_hml": ex_ante["exp_style:HML"].abs().mean(),
            "max_abs_sector_net": ex_ante[[c for c in ex_ante if c.startswith("exp_sector:")]].abs().max(axis=1).mean(),
        }
        exposures_ts[name] = ex_ante
        ex_ante.to_parquet(cfg.paths.results / f"{name}__risk_decomp.parquet")
    decomp = pd.DataFrame(decomp_rows).T
    decomp.to_csv(rep / "stage6_risk_decomposition.csv")
    out["risk_decomposition"] = decomp

    # 3. Crisis years
    crisis = {y: period_table(results, panel, y, ann) for y in cfg.evaluation.crisis_years}
    for y, t in crisis.items():
        t.to_csv(rep / f"stage6_year_{y}.csv")
    out["crisis"] = crisis

    # 4. Is deep better? Paired tests on monthly net returns
    tests = {}
    for a, b in [("e2e", "ls_classical"), ("ls_deep_twostage", "ls_classical"), ("e2e", "ls_deep_twostage")]:
        if a in results and b in results:
            tests[f"{a} - {b}"] = difference_test(results[a], results[b], lags)
    out["tests"] = pd.DataFrame(tests).T
    out["tests"].to_csv(rep / "stage6_difference_tests.csv")

    # 5. Sensitivity: costs
    ls = {k: v for k, v in results.items() if k in LS_ORDER}
    out["cost_sensitivity"] = cost_sensitivity(ls, panel, cfg)
    out["cost_sensitivity"].to_csv(rep / "stage6_cost_sensitivity.csv")

    # 6. Sensitivity: constraints (re-optimise with the saved forecasts; models not retrained)
    out["constraint_sensitivity"] = constraint_sensitivity(cfg, panel, signals, betas)
    out["constraint_sensitivity"].to_csv(rep / "stage6_constraint_sensitivity.csv")

    # 7. Robustness results produced by `python -m dpc.cli robustness`, if present
    for f in ("stage6_model_size.csv", "stage6_seeds.csv"):
        p = rep / f
        if p.exists():
            out[f[:-4]] = pd.read_csv(p, index_col=0)

    # 8. Look-ahead checks
    T = pd.Timestamp("2016-06-30")
    T = panel.month_ends[panel.month_ends <= T][-1]
    checks = {
        "engine_truncation_invariance_ls_classical": truncation_check(panel, signals, betas, cfg, T),
        "deep_feature_truncation_invariance": feature_truncation_check(panel, cfg, T),
        **oracle_check(panel, signals, betas, cfg),
    }
    (rep / "stage6_lookahead_checks.json").write_text(json.dumps(checks, indent=2))
    out["lookahead"] = checks
    log.info("Look-ahead checks: %s", checks)

    report_figures.make_all(cfg, panel, results, exposures_ts, decomp)
    write_markdown(cfg, out)
    log.info("Performance (OOS, net):\n%s", perf.round(3).to_string())
    log.info("Risk decomposition:\n%s", decomp.round(3).to_string())
    return out


def constraint_sensitivity(cfg: Config, panel: Panel, signals: dict, betas: dict) -> pd.DataFrame:
    from dpc.deep.e2e import E2EForecasts, make_e2e_strategy
    from dpc.deep.train_risk import RiskForecasts, make_ls_deep_twostage

    n_sec = panel.universe["sector"].nunique()
    deep_fc = RiskForecasts.load(cfg.paths.data_processed / "deep_risk_forecasts.npz")
    e2e_path = cfg.paths.data_processed / "e2e_forecasts_main.npz"
    e2e_fc = None
    if e2e_path.exists():
        z = np.load(e2e_path)
        e2e_fc = E2EForecasts(pd.DatetimeIndex(z["dates"].astype("datetime64[ns]")), z["mu"], z["B"], z["s"])
    variants = [("sector_limit", v) for v in cfg.evaluation.sensitivity_sector_limits] + \
               [("max_weight", v) for v in cfg.evaluation.sensitivity_max_weights]
    rows = []
    for key, val in variants:
        c2 = cfg.replace(optimizer=dataclasses.replace(cfg.optimizer, **{key: val}))
        builders = {
            "ls_classical": lambda: make_ls_classical(c2, len(panel.tickers), n_sec)[0],
            "ls_deep_twostage": lambda: make_ls_deep_twostage(c2, deep_fc, n_sec)[0],
        }
        if e2e_fc is not None:
            builders["e2e"] = lambda: make_e2e_strategy(c2, e2e_fc, panel.tickers, n_sec)[0]
        for name, b in builders.items():
            r = run_backtest(name, panel, b(), signals, betas, cfg.backtest.oos_start, cfg.backtest.transaction_cost_bps, False)
            m = performance_metrics(r.daily_net, panel.ff, r.turnover, r.costs, cfg.backtest.annualization_days, cfg.signals.newey_west_lags)
            rows.append({"strategy": LABELS[name], "setting": f"{key}={val:g}", "sharpe": m["sharpe"],
                         "ann_vol": m["ann_vol"], "max_drawdown": m["max_drawdown"], "turnover_ann": m["turnover_ann"]})
            log.info("constraint sensitivity %s %s=%g: Sharpe %.2f", name, key, val, m["sharpe"])
    return pd.DataFrame(rows).set_index(["setting", "strategy"]).sort_index()


def _md(df: pd.DataFrame, fmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=fmt)


def write_markdown(cfg: Config, out: dict) -> None:
    lines = ["# Stage 6 - Evaluation (all results out-of-sample, walk-forward, net of costs)", ""]
    lines += ["## Performance", "", _md(out["performance"]), ""]
    lines += ["## Risk decomposition (headline)", "",
              "Ex-ante shares from an independent Barra-style model (6 style + 11 sector factors); "
              "realised shares split each day's P&L into factor and stock-specific parts.", "",
              _md(out["risk_decomposition"]), ""]
    for y, t in out["crisis"].items():
        lines += [f"## {y}", "", _md(t), ""]
    lines += ["## Paired tests on monthly net returns (Newey-West)", "", _md(out["tests"], ".4f"), ""]
    lines += ["## Sensitivity: transaction costs (Sharpe, same trades re-priced)", "", _md(out["cost_sensitivity"], ".2f"), ""]
    lines += ["## Sensitivity: constraints (re-optimised; deep models not retrained)", "", _md(out["constraint_sensitivity"]), ""]
    for k in ("stage6_model_size", "stage6_seeds"):
        if k in out:
            lines += [f"## {k.replace('stage6_', '').replace('_', ' ').title()}", "", _md(out[k]), ""]
    lines += ["## Look-ahead checks", "", "```json", json.dumps(out["lookahead"], indent=2), "```", ""]
    (cfg.paths.reports / "stage6_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
