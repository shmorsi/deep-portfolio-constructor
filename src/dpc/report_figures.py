"""Figures for the Stage 6 report and the research note (PNG, light surface)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dpc.backtest import BacktestResult
from dpc.config import Config
from dpc.data.panel import Panel
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.ticker import FuncFormatter

from dpc.plotting import BASELINE, CATEGORICAL, CONTEXT_GRAY, INK_2, apply_style, save

from dpc.strategies import COLORS as COLOR, LONG_SHORT, SHORT  # one source for names and colours


def _label_ends(ax, series: dict[str, pd.Series], log: bool = False) -> None:
    """Direct labels at line ends, nudged apart (in log space for log axes) so they never collide."""
    tf = (lambda v: np.log10(v)) if log else (lambda v: v)
    inv = (lambda v: 10**v) if log else (lambda v: v)
    lo, hi = (tf(v) for v in ax.get_ylim())
    gap = (hi - lo) * 0.05
    ends = sorted(((tf(s.iloc[-1]), n) for n, s in series.items()), key=lambda x: x[0])
    placed: list[float] = []
    for y, n in ends:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(SHORT.get(n, n), (series[n].index[-1], inv(y)), xytext=(6, 0), textcoords="offset points",
                    color=INK_2, fontsize=8.5, va="center", annotation_clip=False)


def cumulative_returns(cfg: Config, panel: Panel, results: dict[str, BacktestResult]) -> None:
    import matplotlib.pyplot as plt

    groups = [("Long-only baselines (excess of cash)", [n for n in COLOR if n in results and results[n].long_only]),
              ("Market-neutral long-short", [n for n in LONG_SHORT if n in results])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    for ax, (title, names) in zip(axes, groups):
        series = {}
        for n in names:
            x = results[n].excess_returns(panel.ff["RF"])
            w = (1 + x).cumprod()
            series[n] = w
            ax.plot(w.index, w.values, color=COLOR[n], lw=2.2 if n == "e2e" else 1.6, label=SHORT[n])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"${v:g}" if f"{v:g}".lstrip("0.")[:1] in ("2", "5") else ""))
        ax.axhline(1, color=BASELINE, lw=1)
        ax.set_title(title)
        ax.set_ylabel("Growth of $1 (log scale)")
        ax.legend(loc="upper left", fontsize=8.5)
        if len(series) <= 4:  # direct labels only when few enough to read
            _label_ends(ax, series, log=True)
    fig.suptitle("Out-of-sample cumulative net returns, monthly rebalancing, 10 bps costs", x=0.01, ha="left", fontsize=11, color=INK_2)
    save(fig, cfg.paths.figures / "stage6_cumulative.png")


def drawdowns(cfg: Config, panel: Panel, results: dict[str, BacktestResult]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.8))
    for n in LONG_SHORT:
        if n not in results:
            continue
        w = (1 + results[n].daily_net).cumprod()
        dd = w / w.cummax() - 1
        ax.plot(dd.index, dd.values * 100, color=COLOR[n], lw=1.8 if n == "e2e" else 1.3, label=SHORT[n])
    ax.set_title("Drawdowns of the market-neutral strategies")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", ncols=3, fontsize=8.5)
    save(fig, cfg.paths.figures / "stage6_drawdowns.png")


def risk_decomposition_bars(cfg: Config, decomp: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    d = decomp[["exante_share_specific", "exante_share_style", "exante_share_sector"]].clip(lower=0)
    d = d.div(d.sum(axis=1), axis=0) * 100
    parts = [("exante_share_specific", "Stock-specific (intended bets)", CATEGORICAL[0]),
             ("exante_share_style", "Style factors (beta, size, value, ...)", CATEGORICAL[1]),
             ("exante_share_sector", "Sectors", CATEGORICAL[3])]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(d))[::-1]
    left = np.zeros(len(d))
    for col, label, color in parts:
        vals = d[col].to_numpy()
        ax.barh(y, vals, left=left, color=color, height=0.62, label=label, edgecolor="#fcfcfb", linewidth=2)
        for yi, l, v in zip(y, left, vals):
            if v >= 8:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", color="#ffffff" if color != CATEGORICAL[3] else "#0b0b0b", fontsize=8.5)
        left += vals
    ax.set_yticks(y, d.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of predicted portfolio variance (%)")
    ax.set_title("Where does the risk come from? (average over out-of-sample rebalances)")
    fig.text(0.01, -0.08, "Independent Barra-style model: 6 style betas + 11 sector factors. For long-only books the sector "
             "bucket also carries the overall market move (sector dummies span the intercept).", fontsize=8, color=INK_2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=3, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    save(fig, cfg.paths.figures / "stage6_risk_decomposition.png")


def beta_exposure(cfg: Config, exposures: dict[str, pd.DataFrame]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, [("exp_style:Mkt-RF", "Market beta"), ("exp_style:SMB", "Size (SMB) beta"), ("exp_style:HML", "Value (HML) beta")]):
        for n in ("ls_naive", "ls_classical", "e2e", "e2e_deep"):
            if n in exposures:
                s = exposures[n][col]
                ax.plot(s.index, s.values, color=COLOR[n], lw=1.6 if n == "e2e" else 1.2, label=SHORT[n])
        ax.axhspan(-cfg.optimizer.factor_limit, cfg.optimizer.factor_limit, color="#e1e0d9", alpha=0.7, lw=0)
        ax.axhline(0, color=BASELINE, lw=1)
        ax.xaxis.set_major_locator(YearLocator(4))
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.set_title(title)
    axes[0].set_ylabel("Portfolio exposure (beta)")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle(f"Factor exposure at each rebalance (shaded band = ±{cfg.optimizer.factor_limit:g} limit)", x=0.01, y=1.04, ha="left", fontsize=11, color=INK_2)
    save(fig, cfg.paths.figures / "stage6_factor_exposures.png")


def crisis_years(cfg: Config, panel: Panel, results: dict[str, BacktestResult]) -> None:
    import matplotlib.pyplot as plt

    years = cfg.evaluation.crisis_years
    names = [n for n in ("e2e_deep", "e2e", "ls_deep_alpha", "ls_classical", "ls_naive", "equal_weight", "hrp") if n in results]
    fig, axes = plt.subplots(1, len(years), figsize=(13, 3.8), sharey=True)
    for ax, y in zip(axes, years):
        vals = []
        for n in names:
            x = results[n].excess_returns(panel.ff["RF"])
            x = x[x.index.year == y]
            vals.append(((1 + x).prod() - 1) * 100)
        pos = np.arange(len(names))[::-1]
        ax.barh(pos, vals, color=[COLOR[n] if not results[n].long_only else CONTEXT_GRAY for n in names], height=0.6)  # long-only = grey context
        for p, v in zip(pos, vals):
            ax.text(v + (1 if v >= 0 else -1), p, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK_2)
        ax.axvline(0, color=BASELINE, lw=1)
        ax.set_title(str(y))
        ax.set_yticks(pos, [SHORT[n] + (" (LO)" if results[n].long_only else "") for n in names])
        ax.grid(axis="y", visible=False)
        lim = max(abs(v) for v in vals) * 1.35
        ax.set_xlim(-lim, lim)
    axes[0].set_xlabel("Excess return in the year (%)")
    fig.suptitle("Stress years: 2008 (GFC), 2020 (COVID), 2022 (rate shock). Grey = long-only", x=0.01, ha="left", fontsize=11, color=INK_2)
    save(fig, cfg.paths.figures / "stage6_crisis_years.png")


def learned_signal_weights(cfg: Config) -> None:
    import matplotlib.pyplot as plt

    p = cfg.paths.reports / "stage5_refits_main.csv"
    if not p.exists():
        return
    df = pd.read_csv(p, parse_dates=["refit"]).set_index("refit")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, k in enumerate(("w_momentum", "w_reversal", "w_low_vol")):
        ax.plot(df.index, df[k], marker="o", ms=4, color=CATEGORICAL[i], label=k[2:])
    ax.plot(df.index, np.exp(df["log_scale"]) - 1, marker="o", ms=4, color=CATEGORICAL[6], label="composite scale - 1")
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_title("What the end-to-end model learned, refit by refit (0 = classical)")
    ax.set_ylabel("Learned adjustment")
    ax.legend(loc="upper left", ncols=4, fontsize=8.5)
    save(fig, cfg.paths.figures / "stage5_learned_weights.png")


def make_all(cfg: Config, panel: Panel, results: dict[str, BacktestResult], exposures: dict[str, pd.DataFrame], decomp: pd.DataFrame) -> None:
    apply_style()
    cumulative_returns(cfg, panel, results)
    drawdowns(cfg, panel, results)
    risk_decomposition_bars(cfg, decomp)
    beta_exposure(cfg, exposures)
    crisis_years(cfg, panel, results)
    learned_signal_weights(cfg)
