"""Deep Portfolio Constructor: explore the long-only wealth study.

Run from the project root:   streamlit run dashboard/app.py

It reads only committed results (reports/results/wealth/*.parquet and
reports/wealth/summary.json): no download, no training, no PyTorch. The
archived long-short dashboard is dashboard/longshort_app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dpc.wealth.names import ABLATIONS, COLORS, LABELS, MAIN, REFERENCE

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results" / "wealth"
SUMMARY = ROOT / "reports" / "wealth" / "summary.json"
GRID, MARKET_BAR = "#e1e0d9", "#c9c8c1"

st.set_page_config(page_title="Deep Portfolio Constructor", layout="wide")


@st.cache_data
def load() -> tuple[dict[str, pd.DataFrame], dict]:
    names = [n for n in MAIN + ABLATIONS + REFERENCE if (RESULTS / f"{n}.parquet").exists()]
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    return {n: pd.read_parquet(RESULTS / f"{n}.parquet") for n in names}, summary


def style(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def pct(x: float | None, digits: int = 1, sign: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:+.{digits}f}%" if sign else f"{100 * x:.{digits}f}%"


results, summary = load()
if "market" not in results:
    st.error("No committed results in reports/results/wealth/. Run `python -m dpc.cli wealth` first.")
    st.stop()
mkt = results["market"]
wealth = {n: (1 + df["net"]).cumprod() for n, df in results.items()}

st.title("Deep Portfolio Constructor")
st.caption(f"A neural network that builds a long-only portfolio for long-term wealth. Out of sample {mkt.index[0]:%B %Y} to "
           f"{mkt.index[-1]:%B %Y}, fully invested, never short, every trade costs {summary.get('cost_bps', 10):g} bps. "
           "Every number on this page is read from the committed results.")

head = st.columns(4)
if "deep" in wealth:
    head[0].metric("$1 in January 1960: the network", f"${wealth['deep'].iloc[-1]:,.0f}")
head[1].metric("$1 in January 1960: the market", f"${wealth['market'].iloc[-1]:,.0f}")
primary = summary.get("comparisons", {}).get("deep - market")
if primary:
    head[2].metric("Growth vs the market, a year", pct(primary["diff_log_growth"], 2, sign=True),
                   help="Annualised difference in net log growth, the protocol's primary metric")
    head[3].metric("Pre-registered verdict", primary["verdict"],
                   help=f"Newey-West t {primary['t_nw']:.2f}; 95% block-bootstrap interval "
                        f"{pct(primary['ci_low'], 2, sign=True)} to {pct(primary['ci_high'], 2, sign=True)} a year")

tab_growth, tab_hold, tab_roll, tab_tables = st.tabs(["Growth of $1", "Holdings", "Rolling 10 years", "Tables"])

with tab_growth:
    pick = st.multiselect("Strategies", list(results), format_func=LABELS.get,
                          default=[n for n in ("market", "momentum_tilt", "deep") if n in results])
    fig = go.Figure()
    for n in pick:
        w = wealth[n]
        fig.add_trace(go.Scatter(x=w.index, y=w.values, name=LABELS[n],
                                 line=dict(color=COLORS[n], width=3 if n == "deep" else 1.6)))
    fig.update_yaxes(type="log", title="Growth of $1 (log scale)", tickprefix="$")
    st.plotly_chart(style(fig, 480), width="stretch")

with tab_hold:
    if "deep" not in results:
        st.info("The Deep Growth Network has not been run yet: `python -m dpc.cli wealth-train`.")
    else:
        deep = results["deep"]
        wcols = [c for c in deep.columns if c.startswith("w:")]
        dates = list(deep.index)
        when = st.select_slider("Month", options=dates, value=dates[-1], format_func=lambda d: f"{d:%b %Y}")
        net_w = deep.loc[when, wcols].astype(float)
        mkt_w = mkt.loc[when, wcols].astype(float)
        order = mkt_w.sort_values(ascending=False).index
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[c[2:] for c in order], y=100 * mkt_w[order], name="Market", marker_color=MARKET_BAR))
        fig.add_trace(go.Bar(x=[c[2:] for c in order], y=100 * net_w[order], name=LABELS["deep"], marker_color=COLORS["deep"]))
        fig.update_layout(barmode="group", hovermode="x")
        fig.update_yaxes(title="Weight, %")
        st.plotly_chart(style(fig, 440), width="stretch")
        ratio = net_w / mkt_w
        cols = st.columns(4)
        cols[0].metric("Differs from the index", pct(0.5 * float(np.abs(net_w - mkt_w).sum()), 0), help="Active share")
        cols[1].metric("Biggest overweight", f"{ratio.idxmax()[2:]} {ratio.max():.2f}×")
        cols[2].metric("Biggest underweight", f"{ratio.idxmin()[2:]} {ratio.min():.2f}×")
        if "kappa" in deep:
            cols[3].metric("Share of the gap to its target traded", pct(float(deep.loc[when, "kappa"]), 0))
        st.caption("Bars are the weights held during the month after the decision. The network scales each industry's market "
                   "weight by a factor between a third and three, then rescales the portfolio to 100%.")

with tab_roll:
    pick = st.multiselect("Strategies", [n for n in results if n != "market"], format_func=LABELS.get, key="roll",
                          default=[n for n in ("momentum_tilt", "deep") if n in results])
    base = np.log1p(mkt["net"]).rolling(120).sum() / 10
    fig = go.Figure()
    for n in pick:
        r = 100 * (np.log1p(results[n]["net"]).rolling(120).sum() / 10 - base)
        fig.add_trace(go.Scatter(x=r.index, y=r.values, name=LABELS[n], line=dict(color=COLORS[n], width=2.6 if n == "deep" else 1.4)))
    fig.add_hline(y=0, line_color="#111827", line_width=1)
    fig.update_yaxes(title="Growth above the market over 10 years, % a year")
    st.plotly_chart(style(fig, 420), width="stretch")

with tab_tables:
    if not summary:
        st.info("Run `python -m dpc.cli wealth-evaluate` to write reports/wealth/summary.json.")
    else:
        perf = summary["performance"]
        rows = [{"Strategy": LABELS[n], "$1 became": f"${perf[n]['terminal_wealth']:,.0f}", "CAGR": pct(perf[n]["cagr"], 2),
                 "Volatility": pct(perf[n]["volatility"]), "Sharpe": f"{perf[n]['sharpe']:.2f}", "Max drawdown": pct(perf[n]["max_drawdown"], 0),
                 "Traded a year": pct(perf[n]["turnover_per_year"], 0), "Active share": pct(perf[n].get("active_share"), 0),
                 "FF3 alpha (t)": f"{pct(perf[n]['alpha_annual'], 2, sign=True)} ({perf[n]['alpha_t']:.2f})"}
                for n in MAIN + ABLATIONS + REFERENCE if n in perf]
        st.subheader("Performance")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.subheader("Pre-registered comparisons")
        comps = [{"Comparison": k.replace(" - ", " minus "), "Growth a year": pct(c["diff_log_growth"], 2, sign=True),
                  "Newey-West t": f"{c['t_nw']:.2f}", "95% interval": f"{pct(c['ci_low'], 2, sign=True)} to {pct(c['ci_high'], 2, sign=True)}",
                  "Verdict": c["verdict"]} for k, c in summary["comparisons"].items()]
        st.dataframe(pd.DataFrame(comps), hide_index=True, width="stretch")
        st.subheader("Compound growth by decade")
        dec = pd.DataFrame(summary["decades"]).rename(index=LABELS)
        st.dataframe(dec.map(lambda v: pct(v) if pd.notna(v) else ""), width="stretch")
        st.caption("Full tables: reports/wealth/results.md. Protocol: docs/wealth_protocol.md.")
