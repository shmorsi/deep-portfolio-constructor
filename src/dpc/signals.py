"""Stage 2: three well-known equity signals ("analyst views") and their ICs.

All signals are computed at month-end ``t`` from data dated ``<= t``:

* **Momentum (12-1)**: return from month t-11 to month t-1 (11 months),
  skipping the most recent month, which tends to *reverse* (Jegadeesh-Titman).
* **Short-term reversal (1m)**: minus last month's return. Last month's losers
  tend to bounce (liquidity provision / overreaction).
* **Low volatility**: minus trailing 1-year daily volatility. Low-risk stocks
  have historically earned similar or higher returns than high-risk ones
  (the low-volatility anomaly, Frazzini-Pedersen "betting against beta").

Each raw signal is turned into a cross-sectional z-score among investable
stocks, winsorised at +/- ``winsorize_z`` and re-standardised. The composite
is a weighted sum of the three z-scores, re-standardised.

Predictive power is the information coefficient (IC): the Spearman rank
correlation between the signal at t and the stock return over month t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dpc.config import Config, SignalsConfig
from dpc.data.panel import Panel
from dpc.logging_utils import get_logger
from dpc.stats import newey_west_mean

log = get_logger(__name__)

SIGNAL_NAMES = ("momentum", "reversal", "low_vol")


def monthly_returns(daily: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Compounded return of each calendar month, indexed by that month's last trading day.

    A month with any missing daily return (stock not listed for the whole
    month) gets NaN, so partial months never masquerade as full ones.
    """
    period = daily.index.to_period("M")
    comp = (1 + daily).groupby(period).prod(min_count=1) - 1
    complete = daily.notna().groupby(period).all()
    comp = comp.where(complete)
    comp.index = comp.index.to_timestamp(how="end").normalize()
    me = pd.Series(month_ends, index=month_ends.to_period("M"))
    comp.index = pd.DatetimeIndex(me.reindex(comp.index.to_period("M")).values)
    return comp.loc[comp.index.notna()].reindex(month_ends)


def forward_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return earned over the month *after* each month-end (the prediction target)."""
    return monthly.shift(-1)


def raw_signals(panel: Panel, sc: SignalsConfig) -> dict[str, pd.DataFrame]:
    me = panel.month_ends
    mret = monthly_returns(panel.returns, me)
    log1p = np.log1p(mret)
    window = sc.momentum_lookback_months - sc.momentum_skip_months
    mom = np.expm1(log1p.shift(sc.momentum_skip_months).rolling(window, min_periods=window).sum())
    rev = -mret.rolling(sc.reversal_lookback_months, min_periods=sc.reversal_lookback_months).apply(
        lambda x: np.prod(1 + x) - 1, raw=True
    )
    vol = panel.returns.rolling(sc.low_vol_lookback_days, min_periods=sc.low_vol_lookback_days).std()
    low_vol = -vol.reindex(me)
    return {"momentum": mom, "reversal": rev, "low_vol": low_vol}


WINSOR_MAX_ITER = 20  # iterations of clip + re-standardise (converges in a few)
WINSOR_TOL = 1e-6


def cross_sectional_z(x: pd.DataFrame, mask: pd.DataFrame, clip: float) -> pd.DataFrame:
    """Row-wise z-score over ``mask``-ed stocks, iteratively winsorised.

    A single "z-score, clip, re-standardise" pass is not enough: one huge
    outlier inflates the first standard deviation, so after re-scaling the
    clipped point can sit far beyond ``clip`` again. We therefore repeat
    clip + re-standardise until every |z| <= clip (to tolerance).
    """
    x = x.where(mask)

    def _z(df: pd.DataFrame) -> pd.DataFrame:
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    z = _z(x)
    for _ in range(WINSOR_MAX_ITER):
        if not (z.abs() > clip + WINSOR_TOL).any().any():
            break
        z = _z(z.clip(-clip, clip))
    return z


def build_signals(panel: Panel, sc: SignalsConfig) -> dict[str, pd.DataFrame]:
    """z-scored component signals plus the composite, each month_end x ticker."""
    raw = raw_signals(panel, sc)
    mask = panel.eligible & pd.concat(raw.values()).notna().groupby(level=0).all().reindex(panel.month_ends)
    z = {k: cross_sectional_z(v, mask, sc.winsorize_z) for k, v in raw.items()}
    w = {"momentum": sc.weights.momentum, "reversal": sc.weights.reversal, "low_vol": sc.weights.low_vol}
    comp = sum(w[k] * z[k] for k in SIGNAL_NAMES)
    z["combined"] = cross_sectional_z(comp, mask, sc.winsorize_z)
    return z


def information_coefficients(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Monthly Spearman rank IC between signal at t and return over t+1."""
    s_rank = signal.rank(axis=1)
    f_rank = fwd.where(signal.notna()).rank(axis=1)
    s_rank = s_rank.where(f_rank.notna())
    ic = s_rank.corrwith(f_rank, axis=1, method="pearson")  # Pearson on ranks = Spearman
    return ic.dropna()


def quantile_spread(signal: pd.DataFrame, fwd: pd.DataFrame, n_q: int = 5) -> pd.Series:
    """Top-minus-bottom quantile equal-weight return each month."""
    out = {}
    for d in signal.index:
        s, f = signal.loc[d], fwd.loc[d]
        ok = s.notna() & f.notna()
        if ok.sum() < 2 * n_q:
            continue
        q = pd.qcut(s[ok].rank(method="first"), n_q, labels=False)
        out[d] = f[ok][q == n_q - 1].mean() - f[ok][q == 0].mean()
    return pd.Series(out, dtype=float)


def ic_table(signals: dict[str, pd.DataFrame], fwd: pd.DataFrame, lags: int, start: str | None = None) -> pd.DataFrame:
    rows = []
    for name, sig in signals.items():
        s = sig.loc[start:] if start else sig
        ic = information_coefficients(s, fwd.reindex(s.index))
        mean, t, n = newey_west_mean(ic, lags)
        spread = quantile_spread(s, fwd.reindex(s.index))
        sp_mean, sp_t, _ = newey_west_mean(spread, lags)
        rows.append(
            {
                "signal": name,
                "mean_IC": mean,
                "IC_t_stat_NW": t,
                "IC_IR_ann": ic.mean() / ic.std() * np.sqrt(12),
                "IC_hit_rate": float((ic > 0).mean()),
                "Q5_minus_Q1_monthly": sp_mean,
                "Q5_Q1_t_stat_NW": sp_t,
                "n_months": n,
            }
        )
    return pd.DataFrame(rows).set_index("signal")


def run_stage2(cfg: Config) -> dict[str, pd.DataFrame]:
    from dpc.plotting import apply_style, save, CATEGORICAL, INK_2
    import matplotlib.pyplot as plt

    panel = Panel.load(cfg.paths.data_processed)
    sigs = build_signals(panel, cfg.signals)
    fwd = forward_returns(monthly_returns(panel.returns, panel.month_ends))
    out_dir = cfg.paths.data_processed
    for k, v in sigs.items():
        v.to_parquet(out_dir / f"signal_{k}.parquet")
    fwd.to_parquet(out_dir / "forward_returns.parquet")

    lags = cfg.signals.newey_west_lags
    full = ic_table(sigs, fwd, lags)
    oos = ic_table(sigs, fwd, lags, start=cfg.backtest.oos_start)
    corr = pd.DataFrame(
        {a: [pd.concat([sigs[a].stack(), sigs[b].stack()], axis=1).corr().iloc[0, 1] for b in sigs] for a in sigs},
        index=list(sigs),
    )
    md = [
        "# Stage 2 - Signal information coefficients",
        "",
        "IC = monthly Spearman rank correlation between the signal at month-end t and the",
        "stock return over month t+1. t-statistics use Newey-West errors with "
        f"{lags} lags. Q5-Q1 = equal-weight top-quintile minus bottom-quintile return.",
        "",
        "## Full sample",
        "",
        full.to_markdown(floatfmt=".4f"),
        "",
        f"## Out-of-sample window only (from {cfg.backtest.oos_start})",
        "",
        oos.to_markdown(floatfmt=".4f"),
        "",
        "## Correlation between signals (pooled z-scores)",
        "",
        corr.to_markdown(floatfmt=".2f"),
        "",
    ]
    (cfg.paths.reports / "signals_ic.md").write_text("\n".join(md), encoding="utf-8")

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (name, sig) in enumerate(sigs.items()):
        ic = information_coefficients(sig, fwd.reindex(sig.index))
        cum = ic.cumsum()
        ax.plot(cum.index, cum.values, color=CATEGORICAL[i], label=name)
        ax.annotate(name, (cum.index[-1], cum.values[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK_2, fontsize=9, va="center")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_title("Cumulative information coefficient by signal")
    ax.set_ylabel("Sum of monthly rank IC")
    ax.legend(loc="upper left")
    save(fig, cfg.paths.figures / "stage2_cumulative_ic.png")
    log.info("Signal IC (full sample):\n%s", full.round(4).to_string())
    log.info("Signal IC (OOS window):\n%s", oos.round(4).to_string())
    return sigs
