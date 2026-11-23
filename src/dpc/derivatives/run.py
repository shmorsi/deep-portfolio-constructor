"""`python -m dpc.cli derivatives`: the library applied to public data, written to reports/derivatives/.

Not pre-registered: a toolkit run once on public data; every number in the report comes from
summary.json, which this module writes.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..institutional import data as idata
from ..logging_utils import get_logger
from ..stats import newey_west_mean
from . import black, hedging, rates, sabr, trees

log = get_logger(__name__)
FRENCH_DAILY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


def french_daily(cache: Path, refresh: bool = False) -> pd.DataFrame:
    path = cache / "french_factors_daily.csv"
    if refresh or not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        r = requests.get(FRENCH_DAILY, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            path.write_text(z.read(z.namelist()[0]).decode("latin-1"))
    lines = [ln for ln in path.read_text().splitlines() if ln[:8].strip().isdigit() and len(ln.split(",")[0].strip()) == 8]
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=["date", "mkt_rf", "smb", "hml", "rf"])
    df.index = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    out = pd.DataFrame({"ret": (df.mkt_rf + df.rf) / 100, "rf": df.rf / 100})
    return out


def _vrp(cache: Path) -> tuple[dict, pd.DataFrame]:
    d = french_daily(cache)
    vix = idata.fred_series("VIXCLS", cache)
    daily = d.join(vix.rename("vix"), how="inner")
    trades = hedging.vrp_study(daily)
    pnl = trades["hedged_pnl"]
    mean, t, n = newey_west_mean(pnl, 3)
    worst = trades.nsmallest(3, "hedged_pnl")
    by = trades.groupby(trades.index.year)["hedged_pnl"].sum()
    out = {
        "start": f"{trades.index[0]:%Y-%m}", "end": f"{trades.index[-1]:%Y-%m}", "trades": int(n),
        "mean_premium": float(trades["premium"].mean()),
        "mean_hedged_pnl": float(mean), "t_nw": float(t), "hit_rate": float((pnl > 0).mean()),
        "mean_unhedged_pnl": float(trades["unhedged_pnl"].mean()),
        "std_hedged": float(pnl.std(ddof=1)), "std_unhedged": float(trades["unhedged_pnl"].std(ddof=1)),
        "skew_hedged": float(pnl.skew()),
        "mean_vix": float(trades["vix"].mean()), "mean_realised_vol": float(trades["realised_vol"].mean() * 100),
        "share_months_implied_above_realised": float((trades["vix"] / 100 > trades["realised_vol"]).mean()),
        "worst": [{"month": f"{i:%Y-%m}", "pnl": float(v)} for i, v in worst["hedged_pnl"].items()],
        "losing_years": [int(y) for y, v in by.items() if v < 0],
    }
    return out, trades


def _curve(cache: Path) -> tuple[dict, rates.Curve]:
    daily = idata.treasury_daily(cache)
    last = daily.dropna(how="all").iloc[-1]
    date = daily.dropna(how="all").index[-1]
    c = rates.bootstrap_par(np.asarray(last.index, float), last.to_numpy(float))
    mats = [1, 2, 5, 10, 20, 30]
    sw = {str(m): rates.swap(c, m, notional=100.0) for m in (2, 5, 10, 30)}
    reprice = max(abs(rates.bond_price(c, y, m) - 100) for m, y in last.dropna().items() if m >= 0.5)
    return {"as_of": f"{date:%Y-%m-%d}", "par": {str(k): float(v) for k, v in last.dropna().items()},
            "zero": {str(m): float(c.zero(m)[0]) for m in mats},
            "forward_1y": {str(m): c.forward(m - 1, m) for m in (2, 5, 10, 20, 30)},
            "swaps_per_100": sw, "max_par_bond_reprice_error": float(reprice)}, c


def _overlay(c: rates.Curve, root: Path) -> dict:
    """Price the swap overlay that the LDI stress (reports/institutional) calls for, on today's curve."""
    summ = root / "reports" / "institutional" / "summary.json"
    if not summ.exists():
        return {}
    s = json.loads(summ.read_text())
    full = next(r for r in s["ldi"]["rows"] if r["hedge_ratio"] > 0.99)
    need_dv01 = full["swap_notional"] * s["ldi"]["plan"]["swap_duration"] * 1e-4
    s30 = rates.swap(c, 30, notional=1.0)
    notional = need_dv01 / s30["dv01"]
    return {"liability": s["ldi"]["plan"]["liability"], "dv01_needed": need_dv01, "tenor": 30,
            "par_rate": s30["par_rate"], "notional_30y": notional, "dv01_per_unit": s30["dv01"],
            "notional_assumed_by_stress": full["swap_notional"], "stress_duration": s["ldi"]["plan"]["swap_duration"]}


def _examples() -> dict:
    hull = {"call": float(black.bsm(42, 40, 0.5, 0.10, 0.20)), "put": float(black.bsm(42, 40, 0.5, 0.10, 0.20, kind="put"))}
    am = trees.crr(50, 50, 5 / 12, 0.10, 0.40, steps=2000)
    eu = float(black.bsm(50, 50, 5 / 12, 0.10, 0.40, kind="put"))
    F, T = 0.04, 5.0
    K = F * np.array([0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5])
    true = dict(alpha=0.035, beta=0.5, rho=-0.3, nu=0.45)
    fit = sabr.calibrate(F, T, K, sabr.implied_vol(F, K, T, **true))
    hedge = {str(n): hedging.simulate_short_call(rebalances=n, paths=20_000)["std"] for n in (4, 16, 64, 256)}
    return {"hull_15_6": hull, "american_put": am, "european_put": eu, "early_exercise_premium": am - eu,
            "sabr_true": true, "sabr_fit": fit, "hedge_error_std_by_rebalances": hedge}


def _pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def write_markdown(s: dict, path: Path) -> None:
    v, c, o, e = s["vrp"], s["curve"], s.get("overlay", {}), s["examples"]
    L = ["# Derivatives and rates: results", "",
         "<!-- Generated by `python -m dpc.cli derivatives` from reports/derivatives/summary.json. Do not edit. -->", "",
         "Not pre-registered: a tested library (`src/dpc/derivatives/`, `tests/test_derivatives.py`) run once on public data.", "",
         "## 1. The volatility risk premium, delta-hedged, on real data", "",
         f"Each month from {v['start']} to {v['end']} ({v['trades']} trades): sell a one-month at-the-money straddle on the US stock market at the VIX's "
         "volatility, hedge its delta every trading day with the market's actual returns, close at expiry. P&L as a share of the index.", "",
         "| | Average | Standard deviation | Note |", "|---|---:|---:|---|",
         f"| Premium collected | {_pct(v['mean_premium'])} | | |",
         f"| P&L, delta-hedged | {_pct(v['mean_hedged_pnl'], 3)} | {_pct(v['std_hedged'])} | Newey-West t {v['t_nw']:.2f}; positive in {_pct(v['hit_rate'], 0)} of months; skew {v['skew_hedged']:.1f} |",
         f"| P&L, unhedged | {_pct(v['mean_unhedged_pnl'], 3)} | {_pct(v['std_unhedged'])} | |", "",
         f"Implied volatility (VIX) averaged {v['mean_vix']:.1f}% against {v['mean_realised_vol']:.1f}% realised; implied exceeded realised in "
         f"{_pct(v['share_months_implied_above_realised'], 0)} of months. Worst months for the hedged seller: "
         + ", ".join(f"{w['month']} ({_pct(w['pnl'])})" for w in v["worst"]) + ". "
         "The premium is real and so is the crash risk: the seller earns small amounts most months and gives a large share back in a few. "
         "The VIX is a 30-day implied volatility across strikes, above at-the-money volatility, so these figures overstate the premium somewhat.", "",
         "![Delta-hedged straddle P&L by month](../figures/derivatives/vrp.png)", "",
         "## 2. The Treasury zero curve, bootstrapped", "",
         f"Par yields of {c['as_of']} (FRED), semiannual coupons, bootstrapped into discount factors; every quoted par bond reprices to 100 within "
         f"{c['max_par_bond_reprice_error']:.1e}.", "",
         "| Maturity | Zero rate (continuous) | One-year forward ending then |", "|---:|---:|---:|"]
    for m in ("1", "2", "5", "10", "20", "30"):
        L.append(f"| {m}y | {_pct(c['zero'][m])} | {_pct(c['forward_1y'][m]) if m in c['forward_1y'] else ''} |")
    L += ["", "| Receive-fixed swap | Par rate | Annuity | DV01 per 100 notional |", "|---:|---:|---:|---:|"]
    for m, sw in c["swaps_per_100"].items():
        L.append(f"| {m}y | {_pct(sw['par_rate'])} | {sw['annuity']:.3f} | {sw['dv01']:.4f} |")
    if o:
        L += ["", "## 3. The pension's swap overlay, priced", "",
              f"The LDI stress test (reports/institutional) needs receive-fixed swaps with a DV01 of {o['dv01_needed']:.4f} per {o['liability']:.0f} of liability. "
              f"On today's curve a 30-year swap at par ({_pct(o['par_rate'])}) has a DV01 of {o['dv01_per_unit']:.5f} per unit of notional (a duration of "
              f"{o['dv01_per_unit'] * 1e4:.1f} years), so the overlay is {o['notional_30y']:.1f} of 30-year notional per {o['liability']:.0f} of liability; "
              f"the stress test, which uses a duration of {o['stress_duration']:.0f}, assumed {o['notional_assumed_by_stress']:.1f}."]
    h = e["hull_15_6"]
    L += ["", "## 4. Checks against the textbook", "",
          f"- Black-Scholes, Hull Example 15.6: call {h['call']:.4f}, put {h['put']:.4f} (book: 4.76, 0.81).",
          f"- American put on a 2,000-step tree (S = K = 50, r 10%, sigma 40%, 5 months): {e['american_put']:.4f}; European {e['european_put']:.4f}; "
          f"early-exercise premium {e['early_exercise_premium']:.4f}.",
          f"- SABR fitted to a smile it generated: rho {e['sabr_fit']['rho']:.4f} (true {e['sabr_true']['rho']}), nu {e['sabr_fit']['nu']:.4f} "
          f"(true {e['sabr_true']['nu']}), error {e['sabr_fit']['rmse']:.1e}.",
          "- Delta-hedging error of a short 3-month call (standard deviation of P&L per 100 of stock) by rebalances: "
          + ", ".join(f"{k}: {val:.3f}" for k, val in e["hedge_error_std_by_rebalances"].items()) + " (halves for each fourfold increase, as theory says).", ""]
    path.write_text("\n".join(L))


def figures(trades: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(trades.index, trades["hedged_pnl"] * 100, width=25, color=np.where(trades["hedged_pnl"] > 0, "#1f4e79", "#b5651d"))
    ax[0].set_title("Delta-hedged short straddle, P&L per month (% of index)")
    ax[1].plot(trades.index, trades["hedged_pnl"].cumsum() * 100, color="#1f4e79")
    ax[1].set_title("Cumulative P&L (% of index, not compounded)")
    fig.tight_layout()
    fig.savefig(out / "vrp.png", dpi=150)
    plt.close(fig)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.floating | np.integer):
        return o.item()
    return o


def main(cfg, refresh: bool = False) -> int:
    cache = cfg.paths.data_raw / "institutional"
    log.info("Volatility risk premium")
    vrp, trades = _vrp(cache)
    log.info("Treasury curve")
    curve, c = _curve(cache)
    s = {"vrp": vrp, "curve": curve, "overlay": _overlay(c, cfg.root), "examples": _examples()}
    out = cfg.paths.reports / "derivatives"
    out.mkdir(parents=True, exist_ok=True)
    s = _clean(s)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    write_markdown(s, out / "results.md")
    figures(trades, cfg.paths.figures / "derivatives")
    update_readme(cfg.root, s)
    log.info("Wrote %s", out)
    return 0


def readme_block(s: dict) -> str:
    v, c, e = s["vrp"], s["curve"], s["examples"]
    return "\n".join([
        "<!-- derivatives:start -->",
        "<!-- Generated by `python -m dpc.cli derivatives` from reports/derivatives/summary.json. Edit src/dpc/derivatives/run.py, not this block. -->",
        "",
        "Black-Scholes-Merton, Black-76 and Bachelier with Greeks and implied volatility; binomial trees for American options; Monte Carlo with "
        "antithetic and control variates; SABR calibration; Treasury curve bootstrapping, swaps and DV01; delta-hedging simulation "
        "([`src/dpc/derivatives/`](src/dpc/derivatives/), checked against Hull's textbook values in [`tests/test_derivatives.py`](tests/test_derivatives.py)). "
        "Every table is in [`reports/derivatives/results.md`](reports/derivatives/results.md).",
        "",
        f"* **The volatility risk premium, on real data.** Selling a one-month at-the-money straddle on the US market at the VIX each month from {v['start']} "
        f"to {v['end']} and delta-hedging it daily earned {_pct(v['mean_hedged_pnl'], 2)} of the index a month (Newey-West t {v['t_nw']:.1f}), positive in "
        f"{_pct(v['hit_rate'], 0)} of months, with losses concentrated in a few crashes ({', '.join(w['month'] for w in v['worst'])}).",
        f"* **The Treasury curve, bootstrapped** ({c['as_of']}): par bonds reprice to {c['max_par_bond_reprice_error']:.0e}; a 30-year receive-fixed swap at "
        f"{_pct(c['swaps_per_100']['30']['par_rate'])} has a DV01 of {c['swaps_per_100']['30']['dv01']:.3f} per 100, which sizes the pension's swap overlay above.",
        f"* **Textbook checks:** Hull's Example 15.6 call {e['hull_15_6']['call']:.2f}; an American put's early-exercise premium of {e['early_exercise_premium']:.3f}; "
        "hedging error halving with each fourfold increase in rebalancing.",
        "",
        "![Delta-hedged straddle P&L by month](reports/figures/derivatives/vrp.png)",
        "<!-- derivatives:end -->",
    ])


def update_readme(root: Path, s: dict) -> None:
    readme = root / "README.md"
    text = readme.read_text()
    block = readme_block(s)
    start, end = "<!-- derivatives:start -->", "<!-- derivatives:end -->"
    if start in text:
        i, j = text.index(start), text.index(end) + len(end)
        text = text[:i] + block + text[j:]
    else:
        anchor = "## How it works"
        text = text.replace(anchor, "## Derivatives and rates\n\n" + block + "\n\n" + anchor, 1)
    readme.write_text(text)
