"""Deep Portfolio Constructor - Streamlit dashboard.

Run from the project root:   streamlit run dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dpc.config import load_config
from dpc.dashboard_data import load_dashboard_data
from dpc.evaluation import LABELS

# Same validated palette as the static report; colour follows the strategy.
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
RED, GRAY, GRID = "#e34948", "#898781", "#e1e0d9"
COLOR = {"e2e": BLUE, "ls_naive": ORANGE, "ls_classical": AQUA, "ls_deep_twostage": VIOLET}
LO_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

st.set_page_config(page_title="Deep Portfolio Constructor", layout="wide")


@st.cache_resource
def data():
    return load_dashboard_data(load_config())


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


D = data()
cfg = D.cfg
ls_names = [n for n in ("e2e", "ls_deep_twostage", "ls_classical", "ls_naive") if n in D.results]
default = cfg.dashboard.default_strategy if cfg.dashboard.default_strategy in ls_names else ls_names[0]

st.title("Deep Portfolio Constructor")
st.caption("Market-neutral long-short equity: keep the intended stock bets, remove unintended sector and factor risk. "
           "All results are walk-forward and out-of-sample, net of transaction costs. Universe: 99 US & Canadian large caps "
           "(today's constituents, so survivorship bias applies).")

with st.sidebar:
    strat = st.selectbox("Strategy", ls_names, index=ls_names.index(default), format_func=lambda s: LABELS.get(s, s))
    dates = D.results[strat].weights.index
    date = st.selectbox("Rebalance date", list(dates[::-1]), format_func=lambda d: d.strftime("%Y-%m-%d"))
    st.markdown("---")
    oc = cfg.optimizer
    st.markdown(f"**Limits**  \nmax position ±{oc.max_weight:.0%}  \nsector net ±{oc.sector_limit:.1%}  \n"
                f"beta/size/value ±{oc.factor_limit:g}  \ngross ≤ {oc.gross_leverage:.0%}  \nturnover ≤ {oc.turnover_limit:.0%} / month  \n"
                f"costs {cfg.backtest.transaction_cost_bps:g} bps")

# KPI row
perf = D.performance
label = LABELS.get(strat, strat)
if perf is not None and label in perf.index:
    p = perf.loc[label]
    dec = D.decomposition.loc[label] if D.decomposition is not None and label in D.decomposition.index else None
    c = st.columns(6)
    c[0].metric("Net Sharpe", f"{p['sharpe']:.2f}")
    c[1].metric("Ann. return", f"{p['ann_return']:.1%}")
    c[2].metric("Ann. volatility", f"{p['ann_vol']:.1%}")
    c[3].metric("Max drawdown", f"{p['max_drawdown']:.1%}")
    c[4].metric("FF6 alpha t-stat", f"{p['alpha_t_ff6']:.2f}")
    if dec is not None:
        c[5].metric("Risk from stock bets", f"{dec['exante_share_specific']:.0%}")

tab1, tab2, tab3, tab4 = st.tabs(["Current weights", "Risk breakdown", "Exposures vs limits", "Performance vs baselines"])

with tab1:
    w = D.weights_at(strat, date)
    left, right = st.columns([3, 2])
    top = pd.concat([w.head(15), w.tail(15)]).drop_duplicates()
    fig = go.Figure(go.Bar(x=top["weight"] * 100, y=top.index, orientation="h",
                           marker_color=[BLUE if v > 0 else RED for v in top["weight"]],
                           customdata=top[["name", "sector"]], hovertemplate="%{y} · %{customdata[0]}<br>%{customdata[1]}<br>%{x:.2f}%<extra></extra>"))
    fig.add_vline(x=cfg.optimizer.max_weight * 100, line_dash="dot", line_color=GRAY)
    fig.add_vline(x=-cfg.optimizer.max_weight * 100, line_dash="dot", line_color=GRAY)
    fig.update_yaxes(autorange="reversed")
    left.plotly_chart(style(fig, 640).update_layout(title=f"Largest longs (blue) and shorts (red) on {date:%Y-%m-%d}", xaxis_title="Weight (%)"), use_container_width=True)
    right.markdown(f"**{(w['weight'] > 0).sum()} longs, {(w['weight'] < 0).sum()} shorts**  \n"
                   f"Gross {w['weight'].abs().sum():.0%} · net {w['weight'].sum():+.2%}")
    right.dataframe(w.assign(weight=lambda d: (d["weight"] * 100).round(2)).rename(columns={"weight": "weight %"}), height=560)

with tab2:
    rd = D.risk_decomp(strat)
    if rd is None:
        st.info("Run `python -m dpc.cli evaluate` to compute the risk decomposition.")
    else:
        fig = go.Figure()
        for col, name, color in [("share_specific", "Stock-specific (intended)", BLUE), ("share_style", "Style factors", ORANGE), ("share_sector", "Sectors", YELLOW)]:
            fig.add_trace(go.Scatter(x=rd.index, y=rd[col].clip(lower=0) * 100, name=name, stackgroup="one", line=dict(width=0.5, color=color)))
        st.plotly_chart(style(fig, 380).update_layout(title="Share of predicted variance over time (independent Barra-style model)", yaxis_title="%"), use_container_width=True)
        if D.decomposition is not None:
            st.markdown("**Average over all out-of-sample rebalances, every strategy**")
            st.dataframe(D.decomposition[["exante_share_specific", "exante_share_style", "exante_share_sector", "realised_share_specific", "exante_vol_ann"]]
                         .style.format("{:.1%}"))

with tab3:
    a, b = st.columns(2)
    sec = D.sector_exposure(strat, date) * 100
    lim = cfg.optimizer.sector_limit * 100
    fig = go.Figure(go.Bar(x=sec.values, y=sec.index, orientation="h", marker_color=[BLUE if abs(v) <= lim + 1e-6 else RED for v in sec.values],
                           hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig.add_vrect(x0=-lim, x1=lim, fillcolor=GRID, opacity=0.5, line_width=0)
    a.plotly_chart(style(fig, 420).update_layout(title=f"Net sector exposure vs ±{lim:g}% limit", xaxis_title="% of capital"), use_container_width=True)
    fx = D.factor_exposure(strat, date)
    flim = cfg.optimizer.factor_limit
    fig = go.Figure(go.Bar(x=fx.index, y=fx.values, marker_color=[BLUE if abs(v) <= flim + 1e-6 else RED for v in fx.values],
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_hrect(y0=-flim, y1=flim, fillcolor=GRID, opacity=0.5, line_width=0)
    b.plotly_chart(style(fig, 420).update_layout(title=f"Portfolio beta to market, size, value vs ±{flim:g}", yaxis_title="beta"), use_container_width=True)
    to = D.results[strat].turnover
    fig = go.Figure(go.Scatter(x=to.index, y=to.values * 100, mode="lines", line=dict(color=COLOR.get(strat, BLUE), width=2), name="turnover"))
    fig.add_hline(y=cfg.optimizer.turnover_limit * 100, line_dash="dot", line_color=GRAY, annotation_text="limit")
    st.plotly_chart(style(fig, 280).update_layout(title="Turnover per rebalance (sum |Δw|)", yaxis_title="% of capital"), use_container_width=True)

with tab4:
    sel = st.multiselect("Compare with", [n for n in D.results if n != strat],
                         default=[n for n in ("ls_classical", "ls_deep_twostage", "ls_naive", "equal_weight") if n in D.results and n != strat],
                         format_func=lambda s: LABELS.get(s, s))
    names = [strat, *sel]
    cum = D.cumulative(names)
    fig = go.Figure()
    lo_i = 0
    for s, col in zip(names, cum.columns):
        if s in COLOR:
            color = COLOR[s]
        else:
            color = LO_COLORS[lo_i % len(LO_COLORS)] if len(sel) <= 4 else GRAY
            lo_i += 1
        fig.add_trace(go.Scatter(x=cum.index, y=cum[col], name=col, line=dict(color=color, width=2.5 if s == strat else 1.5)))
    fig.update_yaxes(type="log")
    st.plotly_chart(style(fig, 460).update_layout(title="Growth of $1 (excess of cash, net of costs, log scale)"), use_container_width=True)
    if perf is not None:
        cols = ["sharpe", "sortino", "ann_return", "ann_vol", "max_drawdown", "hit_rate_monthly", "alpha_ann_ff6", "alpha_t_ff6", "turnover_ann"]
        st.dataframe(perf[cols].style.format({c: "{:.2f}" for c in ("sharpe", "sortino", "alpha_t_ff6", "turnover_ann")} |
                                               {c: "{:.1%}" for c in ("ann_return", "ann_vol", "max_drawdown", "hit_rate_monthly", "alpha_ann_ff6")}))
