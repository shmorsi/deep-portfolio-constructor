"""`python -m dpc.cli institutional`: run the toolkit on public data and write reports/institutional/.

Everything is computed here from Kenneth French's library and FRED; nothing is typed by hand into
the report. Not pre-registered: this is a toolkit applied once, and the report says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from . import active, attribution, curve, data, fx, ldi, liabilities, privates

log = get_logger(__name__)


def _active(panel) -> dict:
    out = {"budgets": {}, "risk_models": {}}
    for te in (0.01, 0.02, 0.04):
        res = active.run(panel, active.ActiveConfig(te_budget=te))
        out["budgets"][f"{te:.2f}"] = active.summarise(res)
        if te == 0.02:
            main = res
    for kind in ("sample", "shrunk"):
        out["risk_models"][kind] = active.summarise(active.run(panel, active.ActiveConfig(risk=kind)))
    out["risk_models"]["factor"] = out["budgets"]["0.02"]
    out["final_risk"] = active.final_risk_report(main)
    h = attribution.holdings_frame(main.dates, panel.names, active.SECTORS, main.weights, main.bench, main.returns)
    eff = attribution.brinson_fachler(h)
    last10 = h[h.date > h.date.max() - pd.DateOffset(years=10)]
    linked = attribution.carino_link(attribution.brinson_fachler(last10))
    sql = attribution.carino_link_sql(last10)
    yearly = eff.groupby("sector")[attribution.EFFECTS].sum() * 12 / eff.date.nunique()
    yearly["total"] = yearly.sum(axis=1)
    out["attribution"] = {
        "window": f"{last10.date.min():%Y-%m} to {last10.date.max():%Y-%m}",
        "cumulative_portfolio": linked.attrs["Rp"], "cumulative_benchmark": linked.attrs["Rb"],
        "cumulative_active": linked.attrs["active"],
        "by_sector": linked.round(6).to_dict(orient="index"),
        "totals": linked[attribution.EFFECTS].sum().to_dict(),
        "full_span_average_per_year": yearly.round(6).to_dict(orient="index"),
        "full_span_average_totals": yearly[attribution.EFFECTS].sum().to_dict(),
        "sql_max_abs_difference": float(np.abs(linked.sort_index()[sql.columns].to_numpy() - sql.sort_index().to_numpy()).max()),
        "note": "gross of trading costs: attribution explains holdings x returns; costs are reported separately",
    }
    out["rolling_bias"] = active.rolling_bias(main).dropna()
    out["_main"] = main
    return out


def _curve(cache: Path) -> dict:
    daily = data.treasury_daily(cache)
    monthly = data.month_end(daily).loc["1990-01":]
    fitted, rmse = curve.fit_history(monthly)
    changes = fitted.diff().dropna()
    p = curve.pca(changes)
    last = monthly.dropna(how="all").iloc[-1]
    nss = curve.NSS.fit(np.asarray(last.index, float), last.to_numpy(float))
    # par yields treated as annually compounded; convert to continuous for discounting
    zero = lambda t: np.log1p(nss(t))
    return {"as_of": f"{monthly.dropna(how='all').index[-1]:%Y-%m}", "fit_rmse_bp_median": float(rmse.median() * 1e4),
            "fit_rmse_bp_p95": float(rmse.quantile(0.95) * 1e4), "months": int(len(fitted)),
            "pca_explained": p["explained"], "pca_cumulative": p["cumulative"], "pca_months": p["months"],
            "pca_loadings": {name: dict(zip([int(m) for m in p["maturities"]], np.round(row, 4).tolist()))
                             for name, row in zip(("level", "slope", "curvature"), p["loadings"])},
            "latest_fit": {"beta": nss.beta.tolist(), "l1": nss.l1, "l2": nss.l2, "rmse_bp": nss.rmse * 1e4,
                           "observed": {str(k): float(v) for k, v in last.items()},
                           "fitted": {str(k): float(nss(k)[0]) for k in last.index}},
            "_zero": zero, "_daily10": daily[10.0], "_daily30": daily[30.0], "_fitted": fitted}


def _liabilities(zero) -> dict:
    c = liabilities.example_cohort()
    base = liabilities.valuations(c, zero)
    longevity = liabilities.valuations(c, zero, mult=0.8)
    gc_low = liabilities.valuations(c, zero, going_concern_rate=0.05)
    return {
        "cohort": {"ages": c.ages.tolist(), "pensions": c.pensions.round(2).tolist(), "cola": c.cola},
        "life_expectancy_65": liabilities.life_expectancy(65), "life_expectancy_65_longevity": liabilities.life_expectancy(65, 0.8),
        "solvency": base["solvency"], "going_concern_6pct": base["going_concern"], "going_concern_5pct": gc_low["going_concern"],
        "solvency_duration": base["solvency_duration"], "going_concern_duration": base["going_concern_duration"],
        "krd": {str(int(k)): v for k, v in base["krd"].items()},
        "longevity_increase": longevity["solvency"] / base["solvency"] - 1,
    }


def _ldi(liab: dict, d10: pd.Series, d30: pd.Series) -> dict:
    plan = ldi.Plan(liability_duration=round(liab["solvency_duration"], 2))
    rows = []
    for h in (0.0, 0.5, 0.75, 1.0):
        o = ldi.overlay(plan, h)
        h = o["achieved_hedge_ratio"]
        rows.append({"hedge_ratio": h, "bonds_only": o["swap_notional"] == 0, "leverage": o["leverage"], "swap_notional": o["swap_notional"],
                     "funded_change_down100": ldi.shock(plan, h, -0.01)["change_pts"],
                     "funded_change_up100": ldi.shock(plan, h, 0.01)["change_pts"],
                     "funded_change_up100_equity_down20": ldi.shock(plan, h, 0.01, -0.20)["change_pts"],
                     "margin_call_up100": ldi.shock(plan, h, 0.01)["margin_call"],
                     "max_survivable_rise_bp": ldi.max_survivable_rise(plan, h) * 1e4})
    return {"plan": plan.__dict__, "rows": rows,
            "worst_us10": ldi.worst_rises(d10), "worst_us30": ldi.worst_rises(d30),
            "worst_us30_since_2000": ldi.worst_rises(d30.loc["2000":])}


def _fx(panel, cache: Path) -> dict:
    lvl = data.month_end(data.fred_series(data.FX, cache))
    r_usd = pd.Series(panel.mkt_rf + panel.rf, index=panel.dates)
    df = fx.cad_returns(r_usd, lvl).loc["1971-02":]
    roll = fx.rolling_ratio(df)
    stress = df.loc[df.r_usd.nsmallest(24).index]
    return {"start": f"{df.index[0]:%Y-%m}", "end": f"{df.index[-1]:%Y-%m}", "months": len(df),
            "min_variance_ratio": fx.min_variance_ratio(df), "table": fx.table(df),
            "rolling_10y_min": float(roll.min()), "rolling_10y_max": float(roll.max()),
            "rolling_10y_last": float(roll.iloc[-1]),
            "usd_move_in_worst_24_equity_months": float(stress.s.mean()),
            "_rolling": roll}


def _breakeven(sim: dict) -> int:
    cs = np.cumsum(sim["distributions"][1:] - sim["calls"][1:])
    lo = int(np.argmin(cs))
    after = np.nonzero(cs[lo:] >= 0)[0]
    return int(lo + after[0] + 1) if len(after) else -1


def _privates(panel) -> dict:
    fm = privates.FundModel()
    sim = fm.simulate(1.0)
    tvpi, irr = privates.tvpi_irr(sim)
    years = 40
    rate = privates.rate_for_target(0.25, years=years, net_outflow=0.02)
    base = privates.pace(rate, years=years, net_outflow=0.02)
    crash = np.full(years, 0.06)
    crash[25] = -0.30
    shocked = privates.pace(rate, years=years, public_return=crash, net_outflow=0.02)
    dynamic = privates.pace(rate, years=years, public_return=crash, net_outflow=0.02, dynamic_target=0.25)
    first = int(np.argmax(base["share"] >= 0.9 * 0.25))
    # desmoothing: industry returns smoothed quarterly as an appraisal would be
    q = (1 + pd.DataFrame(panel.returns, index=panel.dates)).resample("QE").prod() - 1
    mkt = (1 + pd.Series(panel.mkt_rf + panel.rf, index=panel.dates)).resample("QE").prod() - 1
    real = (q.loc[:, panel.names.index("Cnstr")]).to_numpy()
    check = privates.desmoothing_check(real, 0.6, other=mkt.to_numpy())
    return {
        "fund": {"params": {"life": fm.life, "bow": fm.bow, "growth": fm.growth, "calls": list(fm.calls)},
                 "calls": sim["calls"][1:].round(4).tolist(), "distributions": sim["distributions"][1:].round(4).tolist(),
                 "nav": sim["nav"][1:].round(4).tolist(), "paid_in": sim["paid_in"], "tvpi": tvpi, "irr": irr,
                 "trough_cumulative_net": float(np.min(np.cumsum(sim["distributions"] - sim["calls"]))),
                 "breakeven_year": _breakeven(sim)},
        "pacing": {"target": 0.25, "commit_rate": rate, "net_outflow": 0.02, "years_to_90pct_of_target": first,
                   "share_base": base["share"].round(4).tolist(), "share_crash": shocked["share"].round(4).tolist(),
                   "share_crash_dynamic": dynamic["share"].round(4).tolist(),
                   "commit_dynamic": dynamic["commit_rate"].round(4).tolist(),
                   "crash_year": 25, "share_before": float(base["share"][25]), "share_after_crash": float(shocked["share"][25])},
        "desmoothing": {"series": "the construction industry's quarterly returns, a stand-in with known volatility", **check},
    }


def _fmt_pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def write_markdown(s: dict, path: Path) -> None:
    a = s["active"]
    m = a["budgets"]["0.02"]
    L = []
    L.append("# Institutional toolkit: results\n")
    L.append("<!-- Generated by `python -m dpc.cli institutional` from reports/institutional/summary.json. Do not edit. -->\n")
    L.append("Not pre-registered. These are tools with tests (`tests/test_institutional.py`), applied once to public data "
             "(Kenneth French's library and FRED). Every number below comes from `summary.json`.\n")
    L.append("## 1. Active management against the market: factor risk model, Black-Litterman, a tracking-error budget\n")
    L.append(f"Long-only across the 30 industries, benchmark the cap-weighted market, decided monthly with past data only, "
             f"{m['start']} to {m['end']} ({m['months']} months), net of 10 bps per unit traded.\n")
    L.append("| Tracking-error budget | Active return a year | Newey-West t | Realised tracking error | Information ratio | Bias statistic | Turnover a year |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, v in a["budgets"].items():
        L.append(f"| {_fmt_pct(float(k), 0)} | {_fmt_pct(v['active_return_ann'], 2)} | {v['active_t_nw']:.2f} | {_fmt_pct(v['tracking_error_realised'], 2)} | "
                 f"{v['information_ratio']:.2f} | {v['bias_statistic']:.2f} | {_fmt_pct(v['turnover_ann'], 0)} |")
    lo, hi = m["bias_band"]
    L.append(f"\n**Is the risk model right?** The bias statistic is the standard deviation of active return divided by the model's "
             f"forecast of tracking error; 1 is a perfect forecast, and with {m['months']} months the band is {lo:.2f} to {hi:.2f}. "
             "Three risk models, same budget (2%):\n")
    L.append("| Risk model | Bias statistic | Realised tracking error | Information ratio | Verdict |")
    L.append("|---|---:|---:|---:|---|")
    for k, lab in (("factor", "Factor model (market, size, value) + specific"), ("shrunk", "Sample covariance, Ledoit-Wolf shrinkage"), ("sample", "Sample covariance")):
        v = a["risk_models"][k]
        L.append(f"| {lab} | {v['bias_statistic']:.2f} | {_fmt_pct(v['tracking_error_realised'], 2)} | {v['information_ratio']:.2f} | {v['bias_verdict']} |")
    L.append("\nA three-factor model treats what is left after market, size and value as unrelated across industries; "
             "industries in the same sector are not, so the model under-forecasts the risk of bets that lean on a sector. "
             "That is the reason commercial models carry industry factors.\n")
    fr = a["final_risk"]
    L.append(f"**Where the active risk sits ({fr['as_of']}).** {_fmt_pct(fr['factor_share'], 0)} of the active variance comes from the factors "
             f"and {_fmt_pct(fr['specific_share'], 0)} is specific. Largest contributors:\n")
    L.append("| Industry | Active weight | Share of active risk |")
    L.append("|---|---:|---:|")
    for r in fr["top_contributors"]:
        L.append(f"| {r['industry']} | {_fmt_pct(r['active_weight'], 2)} | {_fmt_pct(r['pct_of_active_risk'], 1)} |")
    at = a["attribution"]
    L.append(f"\n**Brinson-Fachler attribution, 2% budget, gross of costs, sectors grouped from the 30 industries.** Average effect per year over "
             f"{m['start']} to {m['end']}: allocation {_fmt_pct(at['full_span_average_totals']['allocation'], 2)}, selection "
             f"{_fmt_pct(at['full_span_average_totals']['selection'], 2)}, interaction {_fmt_pct(at['full_span_average_totals']['interaction'], 2)}. "
             f"Linked over the last ten years ({at['window']}, Carino): {_fmt_pct(at['cumulative_portfolio'], 1)} against {_fmt_pct(at['cumulative_benchmark'], 1)}, "
             f"active {_fmt_pct(at['cumulative_active'], 1)}, split {_fmt_pct(at['totals']['allocation'], 1)} allocation, {_fmt_pct(at['totals']['selection'], 1)} selection, "
             f"{_fmt_pct(at['totals']['interaction'], 1)} interaction. The same attribution in SQL (DuckDB) agrees with pandas to {at['sql_max_abs_difference']:.0e}.\n")
    L.append("| Sector | Allocation a year | Selection a year | Interaction a year | Total a year | Last 10 years, linked |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for sec, v in sorted(at["full_span_average_per_year"].items(), key=lambda kv: -kv[1]["total"]):
        L.append(f"| {sec} | {_fmt_pct(v['allocation'], 2)} | {_fmt_pct(v['selection'], 2)} | {_fmt_pct(v['interaction'], 2)} | {_fmt_pct(v['total'], 2)} | "
                 f"{_fmt_pct(at['by_sector'][sec]['total'], 2)} |")
    L.append("\nSectors made of one industry (Health, Financials, Telecom, Utilities, Other) can only show allocation.\n")
    L.append("**Read with care.** The view is momentum because the wealth study (this repository) had already found a momentum tilt to be "
             "the strongest classical rule over the same history. The view was chosen knowing that, so a t-statistic here is not independent "
             "evidence of an edge; what this section demonstrates is the machinery and the risk-model check.\n")
    c = s["curve"]
    L.append("\n## 2. The yield curve: Nelson-Siegel-Svensson, principal components, key-rate durations\n")
    L.append(f"US Treasury constant-maturity yields (FRED), every month-end since 1990 ({c['months']} curves). Median fit error "
             f"{c['fit_rmse_bp_median']:.1f} bp (95th percentile {c['fit_rmse_bp_p95']:.1f} bp). Monthly changes of the fitted 1- to 30-year curve, "
             f"{c['pca_months']} months: level explains {_fmt_pct(c['pca_explained'][0])}, slope {_fmt_pct(c['pca_explained'][1])}, curvature "
             f"{_fmt_pct(c['pca_explained'][2])}; together {_fmt_pct(c['pca_cumulative'])}.\n")
    li = s["liabilities"]
    L.append("## 3. The pension promise with mortality: solvency and going-concern\n")
    L.append(f"A closed cohort (actives 30 to 60, retirees 65 to 95), pensions indexed at {_fmt_pct(li['cohort']['cola'], 0)} a year, "
             f"Gompertz-Makeham mortality (the textbook Standard Ultimate Survival Model; life expectancy at 65: {li['life_expectancy_65']:.1f} years). "
             f"Valued on the Treasury curve of {c['as_of']}.\n")
    L.append("| Basis | Value | Duration |")
    L.append("|---|---:|---:|")
    L.append(f"| Solvency (zero curve) | {li['solvency']:,.0f} | {li['solvency_duration']:.1f} |")
    L.append(f"| Going concern, 6% assumed return | {li['going_concern_6pct']:,.0f} | {li['going_concern_duration']:.1f} |")
    L.append(f"| Going concern, 5% assumed return | {li['going_concern_5pct']:,.0f} | |")
    L.append(f"\nKey-rate durations: " + ", ".join(f"{k}y {v:.2f}" for k, v in li["krd"].items()) +
             f" (they add up to the solvency duration). Death rates 20% lower at every age (life expectancy at 65: "
             f"{li['life_expectancy_65_longevity']:.1f}) raise the liability by {_fmt_pct(li['longevity_increase'])}.\n")
    d = s["ldi"]
    L.append("## 4. LDI with a swap overlay: funding risk against liquidity risk\n")
    L.append(f"A plan {_fmt_pct(d['plan']['funded_ratio'], 0)} funded: {_fmt_pct(d['plan']['bond_share'], 0)} in long bonds, "
             f"{_fmt_pct(d['plan']['cash_share'], 0)} cash, {_fmt_pct(d['plan']['return_seeking_share'], 0)} return-seeking; receive-fixed swaps make up the hedge.\n")
    L.append("| Hedge ratio | Leverage | Funded ratio, yields −1 pt | yields +1 pt | +1 pt and equities −20% | Margin call, +1 pt | Largest rise the collateral meets |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in d["rows"]:
        msr = "no swaps" if r["max_survivable_rise_bp"] is None else f"{r['max_survivable_rise_bp']:.0f} bp"
        lab = f"{_fmt_pct(r['hedge_ratio'], 0)} (bonds only)" if r["bonds_only"] else _fmt_pct(r["hedge_ratio"], 0)
        L.append(f"| {lab} | {r['leverage']:.2f}x | {r['funded_change_down100']:+.1f} pts | {r['funded_change_up100']:+.1f} pts | "
                 f"{r['funded_change_up100_equity_down20']:+.1f} pts | {r['margin_call_up100']:.1f} | {msr} |")
    w30, w00 = d["worst_us30"], d["worst_us30_since_2000"]
    L.append(f"\nThe worst rises in the US 30-year yield since 1977: {w30['1']['rise'] * 1e4:.0f} bp in a day ({w30['1']['date']}), "
             f"{w30['5']['rise'] * 1e4:.0f} bp in a week ({w30['5']['date']}), {w30['20']['rise'] * 1e4:.0f} bp in a month ({w30['20']['date']}); "
             f"since 2000: {w00['5']['rise'] * 1e4:.0f} bp in a week ({w00['5']['date']}), {w00['20']['rise'] * 1e4:.0f} bp in a month ({w00['20']['date']}). "
             "Hedging steadies the funded ratio; it is the swaps' cash calls, not the funding position, that a fast rise tests. "
             "A plan below full funding that hedges all of its liability's DV01 is over-hedged in funded-ratio terms (its assets are smaller "
             "than what it hedges), which is why the 100% row loses when yields rise.\n")
    f = s["fx"]
    L.append("## 5. Currency: hedging US equities for a Canadian investor\n")
    L.append(f"US stock market in Canadian dollars, {f['start']} to {f['end']} ({f['months']} months). The minimum-variance hedge ratio is "
             f"**{f['min_variance_ratio']:.2f}** (no hedge at all); over rolling 10-year windows it ranged from {f['rolling_10y_min']:.2f} to {f['rolling_10y_max']:.2f} "
             f"(latest {f['rolling_10y_last']:.2f}). In the 24 worst months for US stocks the US dollar rose {_fmt_pct(f['usd_move_in_worst_24_equity_months'])} on average against the Canadian dollar.\n")
    L.append("| Hedge ratio | Volatility a year | Worst month | Worst fall |")
    L.append("|---:|---:|---:|---:|")
    for i, r in enumerate(f["table"]):
        lab = f"{r['hedge_ratio']:.2f} (minimum variance)" if i == len(f["table"]) - 1 else f"{r['hedge_ratio']:.2f}"
        L.append(f"| {lab} | {_fmt_pct(r['vol_ann'])} | {_fmt_pct(r['worst_month'])} | {_fmt_pct(r['max_drawdown'])} |")
    p = s["privates"]
    fu, pa, ds = p["fund"], p["pacing"], p["desmoothing"]
    L.append("\n## 6. Private assets: fund cash flows, pacing, the denominator effect, desmoothing\n")
    L.append(f"**One fund (Takahashi-Alexander, illustrative buyout parameters).** {_fmt_pct(fu['paid_in'], 0)} of the commitment is ever called; "
             f"the cumulative net cash flow bottoms at {_fmt_pct(fu['trough_cumulative_net'], 0)} of commitment (the J-curve) and turns positive in year {fu['breakeven_year']}; "
             f"TVPI {fu['tvpi']:.2f}x, IRR {_fmt_pct(fu['irr'])}.\n")
    L.append(f"**A programme.** To hold 25% in private assets while paying out 2% of assets a year in net benefits, the plan commits "
             f"{_fmt_pct(pa['commit_rate'])} of its assets every year, and needs {pa['years_to_90pct_of_target']} years to reach 90% of the target. "
             f"A 30% fall in public markets in year {pa['crash_year']} lifts the private share from {_fmt_pct(pa['share_before'])} to "
             f"{_fmt_pct(pa['share_after_crash'])} overnight: the denominator effect. "
             f"A rule that commits less while over target cuts new commitments to {_fmt_pct(min(pa['commit_dynamic'][25:30]))} of assets in the years right after the crash, the step a plan takes to get back to target (and the vintages it then skips).\n")
    L.append(f"**Desmoothing.** Quarterly returns of a series with known volatility ({ds['series']}) smoothed as an appraisal would (phi = {ds['phi']}): "
             f"reported volatility {_fmt_pct(ds['vol_observed'])} a quarter against a true {_fmt_pct(ds['vol_true'])}; Geltner's correction "
             f"(estimated phi {ds['phi_hat']:.2f}) gives {_fmt_pct(ds['vol_desmoothed'])}. Correlation with the stock market: true {ds['corr_with_other_true']:.2f}, "
             f"smoothed {ds['corr_with_other_observed']:.2f}, desmoothed {ds['corr_with_other_desmoothed']:.2f}.\n")
    L.append("## Limits\n")
    L.append("- 30 industries are not stocks: the active study has 30 assets and one view; its information ratio is not a claim about a stock picker.\n"
             "- The risk model has three public factors; commercial models have dozens. The bias statistic shows what that costs.\n"
             "- The mortality table is a textbook Gompertz-Makeham model, not CPM2014; the plan is illustrative, not a real plan's membership.\n"
             "- LDI: parallel shifts, one swap tenor, no repo, no rebalancing during the shock.\n"
             "- FX: the hedge ignores the interest-rate differential (carry), which changes returns, not this variance.\n"
             "- Private-fund parameters are illustrative, not fitted to managers' cash flows.\n")
    path.write_text("\n".join(L))


def figures(s: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    main = s["active"]["_main"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ca = np.cumprod(1 + main.port) / np.cumprod(1 + main.bench_ret)
    ax[0].plot(main.dates, ca, color="#1f4e79")
    ax[0].axhline(1, color="grey", lw=0.8)
    ax[0].set_title("Portfolio / benchmark, 2% tracking-error budget")
    rb = s["active"]["rolling_bias"]
    ax[1].plot(rb.index, rb.values, color="#b5651d")
    ax[1].axhline(1, color="grey", lw=0.8)
    ax[1].set_title("Rolling 36-month bias statistic (1 = risk forecast right)")
    fig.tight_layout()
    fig.savefig(out / "active.png", dpi=150)
    plt.close(fig)

    p = s["privates"]["pacing"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    yrs = np.arange(len(p["share_base"]))
    ax.plot(yrs, np.array(p["share_base"]) * 100, label="steady markets", color="#1f4e79")
    ax.plot(yrs, np.array(p["share_crash"]) * 100, label="public markets −30% in year 25", color="#b5651d")
    ax.plot(yrs, np.array(p["share_crash_dynamic"]) * 100, "--", label="same, pacing to target", color="#6b8e23")
    ax.axhline(25, color="grey", lw=0.8)
    ax.set_xlabel("year")
    ax.set_ylabel("private assets, % of plan")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Building a private-markets programme")
    fig.tight_layout()
    fig.savefig(out / "pacing.png", dpi=150)
    plt.close(fig)

    rows = s["ldi"]["rows"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    h = [r["hedge_ratio"] * 100 for r in rows]
    ax.plot(h, [abs(r["funded_change_up100_equity_down20"]) for r in rows], "o-", color="#1f4e79", label="funded-ratio hit, +1 pt yields & equities −20%")
    ax2 = ax.twinx()
    ax2.plot(h[1:], [r["max_survivable_rise_bp"] for r in rows[1:]], "s-", color="#b5651d", label="largest yield rise the collateral meets (bp)")
    ax.set_xlabel("interest-rate hedge ratio, %")
    ax.set_ylabel("funded-ratio points")
    ax2.set_ylabel("bp")
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.legend(frameon=False, fontsize=8, loc="lower center", ncol=2)
    ax.set_title("LDI: more hedge, less room for a yield spike")
    fig.savefig(out / "ldi.png", dpi=150)
    plt.close(fig)

    roll = s["fx"]["_rolling"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(roll.index, roll.values, color="#1f4e79")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(1, color="grey", lw=0.8, ls=":")
    ax.set_title("Minimum-variance USD hedge ratio for a Canadian investor, rolling 10 years")
    fig.tight_layout()
    fig.savefig(out / "fx.png", dpi=150)
    plt.close(fig)

    c = s["curve"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    mats = sorted(int(k) for k in c["pca_loadings"]["level"])
    for name, col in zip(("level", "slope", "curvature"), ("#1f4e79", "#b5651d", "#6b8e23")):
        ax.plot(mats, [c["pca_loadings"][name][m] for m in mats], label=name, color=col)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("maturity, years")
    ax.legend(frameon=False)
    ax.set_title("Principal components of monthly US Treasury curve changes")
    fig.tight_layout()
    fig.savefig(out / "curve_pca.png", dpi=150)
    plt.close(fig)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_") and k != "rolling_bias"}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main(cfg, refresh: bool = False) -> int:
    from ..wealth.data import load_panel

    w = cfg.wealth
    panel = load_panel(w.base_url, w.industry_file, w.factor_file, cfg.paths.data_raw, refresh)
    cache = cfg.paths.data_raw / "institutional"
    log.info("Active management")
    s = {"active": _active(panel)}
    log.info("Yield curve")
    s["curve"] = _curve(cache)
    s["liabilities"] = _liabilities(s["curve"]["_zero"])
    s["ldi"] = _ldi(s["liabilities"], s["curve"]["_daily10"], s["curve"]["_daily30"])
    log.info("Currency")
    s["fx"] = _fx(panel, cache)
    log.info("Private assets")
    s["privates"] = _privates(panel)
    s["data"] = {"french_vintage_end": f"{panel.dates[-1]:%Y-%m}", "sources": ["Kenneth French data library", "FRED (DGS*, DEXCAUS)"]}
    out = cfg.paths.reports / "institutional"
    out.mkdir(parents=True, exist_ok=True)
    clean = _clean(s)
    (out / "summary.json").write_text(json.dumps(clean, indent=1))
    write_markdown(clean, out / "results.md")
    figures(s, cfg.paths.figures / "institutional")
    update_readme(cfg.root, clean)
    log.info("Wrote %s", out)
    return 0


def readme_block(s: dict) -> str:
    a, m = s["active"], s["active"]["budgets"]["0.02"]
    li, d, f, p, c = s["liabilities"], s["ldi"], s["fx"], s["privates"], s["curve"]
    full = next(r for r in d["rows"] if r["hedge_ratio"] > 0.99)
    return "\n".join([
        "<!-- institutional:start -->",
        "<!-- Generated by `python -m dpc.cli institutional` from reports/institutional/summary.json. Edit src/dpc/institutional/run.py, not this block. -->",
        "",
        "**What a pension fund or asset manager runs around a portfolio**, built as tested tools ([`src/dpc/institutional/`](src/dpc/institutional/), "
        "[`tests/test_institutional.py`](tests/test_institutional.py)) and applied once to public data. Not pre-registered; every table is in "
        "[`reports/institutional/results.md`](reports/institutional/results.md).",
        "",
        "| Tool | What it shows on the data |",
        "|---|---|",
        f"| **Factor risk model + Black-Litterman + tracking-error budget** | Long-only against the cap-weighted market, {m['start']} to {m['end']}: "
        f"{_fmt_pct(m['active_return_ann'], 2)} a year active, net of costs, at {_fmt_pct(m['tracking_error_realised'], 1)} realised tracking error (information ratio "
        f"{m['information_ratio']:.2f}). The view was chosen knowing the wealth study's momentum result, so this is a demonstration, not evidence of an edge. |",
        f"| **Risk-model validation (bias statistic)** | The three-factor model under-forecast tracking error (bias {m['bias_statistic']:.2f}; "
        f"band {m['bias_band'][0]:.2f} to {m['bias_band'][1]:.2f}); Ledoit-Wolf shrinkage did better ({a['risk_models']['shrunk']['bias_statistic']:.2f}). |",
        f"| **Brinson-Fachler attribution, Carino-linked, in pandas and SQL (DuckDB)** | Average {_fmt_pct(a['attribution']['full_span_average_totals']['allocation'], 2)} a year from "
        f"allocation, {_fmt_pct(a['attribution']['full_span_average_totals']['selection'], 2)} from selection; the SQL and pandas versions agree to {a['attribution']['sql_max_abs_difference']:.0e}. |",
        f"| **Nelson-Siegel-Svensson curve, PCA, key-rate durations** | {c['months']} monthly Treasury curves fitted to a median {c['fit_rmse_bp_median']:.1f} bp; level, slope and "
        f"curvature explain {_fmt_pct(c['pca_cumulative'])} of curve moves. |",
        f"| **Pension liability with mortality, solvency vs going concern** | Gompertz-Makeham cohort: solvency duration {li['solvency_duration']:.1f}; 20% lower death rates "
        f"add {_fmt_pct(li['longevity_increase'])} to the liability. |",
        f"| **LDI swap overlay and liquidity stress** | A full hedge takes {full['leverage']:.2f}x leverage and its collateral meets a rise of {full['max_survivable_rise_bp']:.0f} bp; "
        f"the worst month for the US 30-year since 1977 was {d['worst_us30']['20']['rise'] * 1e4:.0f} bp. Funding risk falls as liquidity risk rises (the 2022 UK LDI lesson). |",
        f"| **FX hedge ratio for a Canadian investor** | {f['start']} to {f['end']}: the minimum-variance hedge of US equities is {f['min_variance_ratio']:.2f}; the US dollar rose "
        f"{_fmt_pct(f['usd_move_in_worst_24_equity_months'])} on average in the worst 24 months for US stocks. |",
        f"| **Private assets: Takahashi-Alexander, pacing, denominator effect, desmoothing** | Committing {_fmt_pct(p['pacing']['commit_rate'])} of assets a year reaches 90% of a 25% target in "
        f"{p['pacing']['years_to_90pct_of_target']} years; a 30% public crash lifts the share to {_fmt_pct(p['pacing']['share_after_crash'])}. |",
        "",
        "![Active management: growth against the benchmark and the rolling bias statistic](reports/figures/institutional/active.png)",
        "",
        "![LDI: funding risk against liquidity risk](reports/figures/institutional/ldi.png)",
        "",
        "`python -m dpc.cli institutional` downloads the public data and rewrites every number above.",
        "<!-- institutional:end -->",
    ])


def update_readme(root: Path, s: dict) -> None:
    readme = root / "README.md"
    text = readme.read_text()
    block = readme_block(s)
    start, end = "<!-- institutional:start -->", "<!-- institutional:end -->"
    if start in text:
        i, j = text.index(start), text.index(end) + len(end)
        text = text[:i] + block + text[j:]
    else:
        anchor = "## How it works"
        text = text.replace(anchor, "## Institutional toolkit: risk models, attribution, LDI, currency, private assets\n\n" + block + "\n\n" + anchor, 1)
    readme.write_text(text)
