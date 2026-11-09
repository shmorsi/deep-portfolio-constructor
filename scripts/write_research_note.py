"""Write reports/wealth/research_note.md from the committed results.

Numbers come from reports/site_wealth.json and reports/results/wealth/*.parquet;
sentences that depend on a result follow the pre-registered verdict. Run after
``python -m dpc.cli wealth-site``:

    python scripts/write_research_note.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MINUS = "−"
NAMES = {"Food": "food", "Beer": "beer and liquor", "Smoke": "tobacco", "Games": "recreation", "Books": "printing and publishing",
         "Hshld": "consumer goods", "Clths": "apparel", "Hlth": "healthcare", "Chems": "chemicals", "Txtls": "textiles",
         "Cnstr": "construction", "Steel": "steel", "FabPr": "fabricated products", "ElcEq": "electrical equipment", "Autos": "autos",
         "Carry": "aircraft, ships and rail", "Mines": "mining", "Coal": "coal", "Oil": "oil and gas", "Util": "utilities",
         "Telcm": "communication", "Servs": "business services", "BusEq": "business equipment", "Paper": "paper and packaging",
         "Trans": "transportation", "Whlsl": "wholesale", "Rtail": "retail", "Meals": "restaurants and hotels", "Fin": "finance",
         "Other": "other"}


def money(x: float) -> str:
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def pct(x: float, digits: int = 2, sign: bool = False) -> str:
    text = f"{abs(100 * x):.{digits}f}"
    if float(text) == 0:
        return f"{text}%"
    return (("+" if x > 0 else MINUS) if sign else (MINUS if x < 0 else "")) + text + "%"


def num(x: float, digits: int = 2) -> str:
    text = f"{abs(x):.{digits}f}"
    return (MINUS if x < 0 and float(text) != 0 else "") + text

def month_name(ym: str) -> str:
    """'2026-09' -> 'September 2026'."""
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    return f"{months[int(ym[5:7]) - 1]} {ym[:4]}"



def learned(results: Path) -> dict:
    """Descriptive facts about the deployed network, from its committed holdings and returns."""
    deep, mkt, mom = (pd.read_parquet(results / f"{n}.parquet") for n in ("deep", "market", "momentum_tilt"))
    wcols = [c for c in deep.columns if c.startswith("w:")]
    avg = (deep[wcols].mean() / mkt[wcols].mean()).rename(lambda c: c[2:])  # its average weight over the market's
    active_deep = np.log1p(deep["net"]) - np.log1p(mkt["net"])
    active_mom = np.log1p(mom["net"]) - np.log1p(mkt["net"])
    wealth = (1 + mkt["net"]).cumprod()
    before = wealth.shift(1).fillna(1.0)                                      # market wealth when each decision was made
    stressed = (before / before.cummax() <= 0.8).iloc[1:]
    kappa = deep["kappa"].iloc[1:] if "kappa" in deep else None
    ratio = (deep[wcols] / mkt[wcols]).to_numpy()
    return {
        "over": [(k, float(v)) for k, v in avg.sort_values(ascending=False).head(3).items()],
        "under": [(k, float(v)) for k, v in avg.sort_values().head(3).items()],
        "corr_momentum": float(active_deep.corr(active_mom)),
        "kappa_stress": float(kappa[stressed].mean()) if kappa is not None and stressed.any() else None,
        "kappa_calm": float(kappa[~stressed].mean()) if kappa is not None else None,
        "stress_months": int(stressed.sum()),
        "max_ratio": float(ratio.max()), "min_ratio": float(ratio.min()),
    }


def vs_block(block: dict, key: str) -> str:
    x = block["comparisons"][key]
    return f"{pct(x['diff_log_growth'], sign=True)} a year (Newey-West t {num(x['t_nw'])}): {x['verdict']}"


def main() -> None:
    site = json.loads((ROOT / "reports" / "site_wealth.json").read_text())
    perf, comps, labels, oos = site["performance"], site["comparisons"], site["labels"], site["oos"]
    end_name = month_name(oos["end"])
    d, m, mom = perf["deep"], perf["market"], perf["momentum_tilt"]
    c = comps["deep - market"]
    verdict = c["verdict"]
    L = learned(ROOT / "reports" / "results" / "wealth")
    seeds = [site["seeds"][k]["terminal_wealth"] for k in sorted(site["seeds"])]
    tf = site.get("trade_fraction") or {}
    start = site.get("init_trade", 0.25)

    def vs(key: str) -> str:
        x = comps[key]
        return f"{pct(x['diff_log_growth'], sign=True)} a year (Newey-West t {num(x['t_nw'])}, 95% interval {pct(x['ci_low'], sign=True)} to {pct(x['ci_high'], sign=True)}): {x['verdict']}"

    headline = {
        "better": "the network grew wealth faster than holding the market, and the pre-registered test says the gap is not luck",
        "worse": "the network grew wealth more slowly than holding the market, and the pre-registered test says the gap is real",
    }.get(verdict, "the network's edge over holding the market is too small to tell from luck by the pre-registered test")
    vm = comps["deep - momentum_tilt"]["verdict"]
    rival = {"better": "it also beat a simple momentum tilt by the same test",
             "worse": "a simple momentum tilt did better by the same test"}.get(vm, "its edge cannot be told apart from a simple momentum tilt's")
    trading = (f"it trades {pct(d['turnover_per_year'], 0)} of the portfolio a year against the index's {pct(m['turnover_per_year'], 0)}"
               if d["turnover_per_year"] > 5 * m["turnover_per_year"] else f"it trades only {pct(d['turnover_per_year'], 0)} of the portfolio a year")
    conclusion = {
        "better": (f"A network trained only on the growth of wealth, long-only and after costs, beat holding the market over "
                   f"{oos['months'] // 12} years out of sample by a test fixed in advance. The honest reading is narrower than the headline: "
                   f"{rival}, {trading}, and the result comes from one country's century with features that research discovered later."),
        "worse": ("Trained on the right objective, with a market anchor and a learned trade fraction, the network still lost to holding "
                  "the market by a test fixed in advance. That is a useful negative: for a long-only investor in broad industries, "
                  "learned tilts did not pay for themselves after costs."),
    }.get(verdict, ("Trained on the right objective, with a market anchor and a learned trade fraction, the network did not separate "
                    "from holding the market by a test fixed in advance. For a long-only investor in 30 broad industries, the index is a "
                    "hard benchmark; the learned tilts neither clearly helped nor clearly hurt."))
    if tf and tf["median"] > 2 * start:
        conclusion += (f" Two things it learned are worth knowing. Given the choice, it did not sit still: its trade fraction rose from "
                       f"{pct(start, 0)} to a median of {pct(tf['median'], 0)}, so the {site['cost_bps']:g} bps cost per trade was not enough "
                       f"to teach patience on these features.")
    if abs(d["alpha_t"]) < 2 and d["beta_hml"] > 0.1:
        conclusion += (f" And its gain over the market is about what its value tilt would have paid: a value loading of "
                       f"{num(d['beta_hml'])} and a three-factor alpha of {pct(d['alpha_annual'], sign=True)} (t {num(d['alpha_t'])}).")
    conclusion += (" A fresh test needs fresh data: the honest next step is forward, publishing the network's holdings each month "
                   "from now on and scoring them as the months arrive, under the same protocol.")
    over = ", ".join(f"{NAMES.get(k, k)} ({r:.2f}x)" for k, r in L["over"])
    under = ", ".join(f"{NAMES.get(k, k)} ({r:.2f}x)" for k, r in L["under"])
    kappa_line = ""
    if L["kappa_calm"] is not None:
        kappa_line = (f" Its trade fraction, which training starts at {pct(start, 0)}, had a median of {pct(tf['median'], 0)} out of sample. In the "
                      f"{L['stress_months']} months when the market stood 20% or more below its peak it averaged {pct(L['kappa_stress'], 0)}, "
                      f"against {pct(L['kappa_calm'], 0)} in other months." if L["kappa_stress"] is not None else
                      f" Its trade fraction, which training starts at {pct(start, 0)}, had a median of {pct(tf['median'], 0)} out of sample.")
    abl_rows = []
    for n in ("deep_trade_all", "deep_mlp", "deep_unanchored"):
        if n in perf and f"{n} - deep_pair" in comps:
            abl_rows.append(f"| {labels[n]} | {money(perf[n]['terminal_wealth'])} | {pct(perf[n]['turnover_per_year'], 0)} | {vs(f'{n} - deep_pair')} |")

    added = ""
    if site.get("benchmarks") and site.get("schedules"):
        sp, nq, sc = site["benchmarks"]["sp500"], site["benchmarks"]["nasdaq"], site["schedules"]
        added = (f"## 6b. Added after the test: benchmarks and rebalancing schedules\n\n"
                 f"Against the S&P 500 with dividends reinvested ($1 became {money(sp['wealth']['sp500'])} from 1960), the network's difference "
                 f"was {vs_block(sp, 'deep - sp500')}. Against the NASDAQ Composite from March 1971 (price only, so understated), "
                 f"{vs_block(nq, 'deep - nasdaq')}.\n\n"
                 f"The same trained network on four rebalancing schedules, net of {site['cost_bps']:g} bps per unit traded: "
                 + "; ".join(f"{n} {money(sc[n]['terminal_wealth'])} ({pct(sc[n]['turnover_per_year'], 0)} traded a year, "
                             f"{pct(sc[n]['vs_market']['diff_log_growth'], sign=True)} against the market, t {num(sc[n]['vs_market']['t_nw'])})"
                             for n in ("daily", "monthly", "quarterly", "yearly"))
                 + ". Daily trading only re-centres the month's chosen mix and pays more for it; less frequent trading lets the tilts go "
                   "stale. None of these were pre-registered, and none changes the primary verdict.\n\n")

    text = f"""# The Deep Growth Network: a pre-registered test of deep learning for long-term, long-only investing

Sherif Morsi · Deep Portfolio Constructor, commit {site['commit']} · every number below is generated from the committed results by `scripts/write_research_note.py`.

**Abstract.** A transformer allocates a long-only, fully invested portfolio across Kenneth French's 30 US industry portfolios and learns what fraction of the gap to its target to trade each month. It is trained in PyTorch on minus the log growth of wealth after trading costs, through a differentiable simulator of the investing rules, walk-forward with refits every five years. Out of sample from January 1960 to {end_name} ({oos['months']} months, {site['cost_bps']:g} bps per unit traded), $1 grew to {money(d['terminal_wealth'])} with the network and {money(m['terminal_wealth'])} in the market: {vs('deep - market')}. In words: {headline}. The test was fixed in git before the network was run out of sample.

## 1. Question

Can a neural network, trained on the growth of wealth after trading costs, build a better long-only, fully invested, long-term portfolio than simply holding the market, and than the classical rules? The protocol ([`docs/wealth_protocol.md`](../../docs/wealth_protocol.md)) fixed the data, the rules, the model, the comparisons and the decision rule before the out-of-sample run. The primary metric is the annualised difference in net log growth, network minus market; "better" needs a Newey-West t of at least 2 (12 lags) and a 95% moving-block bootstrap interval (12-month blocks, 5,000 draws) that excludes zero.

## 2. Data and rules

Kenneth French's 30 value-weighted industry portfolios, July 1926 to {end_name}. Each holds every listed firm in its industry at each date, so there is no survivorship bias among firms. Industry market caps give the market portfolio, which tracks French's own market return with a correlation of {site['data_check']['correlation']:.4f}. Every strategy is long-only and fully invested, with no leverage and no cash; a decision at the end of month t is held through month t+1, holdings drift with prices between trades, and every unit traded costs {site['cost_bps']:g} bps.

## 3. Model and training

Each industry is a token of 13 point-in-time features (returns over 1, 3, 12-1, 36 and 60 months, volatility, downside risk, drawdown, beta and correlation with the market, book-to-market, index weight) plus its current holding relative to the market; a market token carries 7 market-wide features. Two transformer blocks (d = 32, 4 heads; {site['params']:,} parameters) let every industry attend to every other. A tilt head sets the target `softmax(log w_mkt + log 3 · tanh(score))`, so each market weight is scaled by a factor between a third and three before rescaling; it starts at zero, so the untrained network is the index. A trade-fraction head sets κ, and new holdings are `w + κ (target − w)`.

Training plays the policy through a differentiable copy of the backtest engine (tested to reproduce it month by month) over 60-month episodes and minimises minus the average monthly log growth after costs, the Kelly criterion. Refits every five years from December 1959 to December 2024 use only earlier data; the last ten years of each window are validation for early stopping, and no training label is dated after the first validation decision. Five seeds are warm-started from their own previous refits, and the deployed policy averages their targets and trade fractions.

## 4. Results

| Strategy | $1 became | CAGR | Volatility | Sharpe | Max drawdown | Traded a year | Against the market |
|---|---:|---:|---:|---:|---:|---:|---|
""" + "\n".join(
        f"| {labels[n]} | {money(perf[n]['terminal_wealth'])} | {pct(perf[n]['cagr'])} | {pct(perf[n]['volatility'], 1)} | {num(perf[n]['sharpe'])} | "
        f"{pct(perf[n]['max_drawdown'], 0)} | {pct(perf[n]['turnover_per_year'], 0)} | {'benchmark' if n == 'market' else vs(f'{n} - market')} |"
        for n in ("market", "equal_weight", "min_variance", "risk_parity", "growth_classical", "momentum_tilt", "deep")) + f"""

**Primary test.** Network minus market: {vs('deep - market')}. The secondary comparisons, reported but not used to choose anything: against equal weight, {vs('deep - equal_weight')}; against the momentum tilt, {vs('deep - momentum_tilt')}; against the classical growth optimiser, {vs('deep - growth_classical')}.

**Over time.** The network grew faster than the market in {pct(d['beat_market_rolling'], 0)} of rolling ten-year windows (the momentum tilt in {pct(mom['beat_market_rolling'], 0)}). Its Fama-French three-factor alpha was {pct(d['alpha_annual'], sign=True)} a year (t {num(d['alpha_t'])}), with a market beta of {num(d['beta_mkt'])}. Its worst calendar year was {pct(d['worst_year'], 0)}, against {pct(m['worst_year'], 0)} for the market.

**Costs and seeds.** Re-pricing the same decisions, its growth against the market (also re-priced) was {pct(site['costs']['deep@0bps']['vs_market_log_growth'], sign=True)} a year with free trading, {pct(site['costs']['deep@25bps']['vs_market_log_growth'], sign=True)} at 25 bps and {pct(site['costs']['deep@50bps']['vs_market_log_growth'], sign=True)} at 50 bps. Each seed's network on its own turned $1 into {', '.join(money(x) for x in seeds)}.

## 5. What the network learned

Averaged over the {oos['months']} months, its largest overweights were {over}, and its largest underweights {under} (its average weight over the market's average weight). Its monthly active return (network minus market, in log growth) had a correlation of {num(L['corr_momentum'])} with the momentum tilt's.{kappa_line} It traded {pct(d['turnover_per_year'], 0)} of the portfolio a year, an average holding period of {d['holding_years']:.1f} years, against {pct(m['turnover_per_year'], 0)} for the index and {pct(mom['turnover_per_year'], 0)} for the momentum tilt. Its weights ranged from {L['min_ratio']:.2f} to {L['max_ratio']:.2f} times the market's. Its loadings on the Fama-French factors were {num(d['beta_mkt'])} on the market, {num(d['beta_smb'])} on size and {num(d['beta_hml'])} on value (the market's own: {num(m['beta_smb'])} and {num(m['beta_hml'])}), with a three-factor alpha of {pct(d['alpha_annual'], sign=True)} a year (t {num(d['alpha_t'])}).

## 6. Ablations

Fixed in the protocol, two seeds each (42 and 43), compared here with the full network averaged over the same two seeds ({money(perf['deep_pair']['terminal_wealth']) if 'deep_pair' in perf else 'n/a'}):

| Ablation | $1 became | Traded a year | Against the full network, same seeds |
|---|---:|---:|---|
{chr(10).join(abl_rows)}

{added}## 7. Limits

The 30 portfolios are industries, not funds an investor could have bought in 1960; costs are a flat {site['cost_bps']:g} bps with no taxes or market impact. The features carry hindsight: momentum, value and low volatility were documented decades after 1960, and the pre-registration fixes the test, not the feature list. The data are one country's century, among the best-performing markets in the world, and several strategies were compared; the pre-registered primary test is the one to read.

## 8. Conclusion

{conclusion}
"""
    (ROOT / "reports" / "wealth" / "research_note.md").write_text(text)
    print(f"research note written ({verdict})")


if __name__ == "__main__":
    main()
