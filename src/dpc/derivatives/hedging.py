"""Delta hedging, in theory and on 30 years of real data.

**Discrete hedging error.** Sell a call, hedge its delta n times over its life. With the right
volatility the P&L is centred on zero and its standard deviation falls like 1/sqrt(n)
(Derman and Kamal); with the wrong one, the P&L is about the sum of
1/2 Gamma S^2 (sigma_implied^2 - sigma_realised^2) dt: the option seller earns the gap between
the volatility sold and the volatility that happened.

**The volatility risk premium on real data.** Each month, sell a one-month at-the-money straddle on
the US stock market at the VIX's volatility, hedge its delta every trading day with the market's
actual daily returns, and close at expiry. The P&L, as a share of the index, is what an option
seller earns for carrying volatility risk. Two stated approximations: the VIX is a 30-day
implied volatility across strikes (it sits above at-the-money volatility, so this overstates the
premium somewhat), and the index is Kenneth French's market return including dividends, priced
as if it paid none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .black import bsm


def _call_delta(S, K, tau, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def simulate_short_call(S0=100.0, K=100.0, T=0.25, r=0.0, sigma_implied=0.2, sigma_real=0.2,
                        rebalances=63, paths=20_000, seed=0) -> dict[str, float]:
    """Sell one call at the implied volatility, delta-hedge `rebalances` times on GBM paths at sigma_real."""
    rng = np.random.default_rng(seed)
    dt = T / rebalances
    S = np.full(paths, S0)
    premium = float(bsm(S0, K, T, r, sigma_implied))
    cash = np.full(paths, premium)
    delta = _call_delta(S, K, T, r, sigma_implied)
    cash -= delta * S
    for i in range(1, rebalances + 1):
        S = S * np.exp((r - 0.5 * sigma_real ** 2) * dt + sigma_real * np.sqrt(dt) * rng.standard_normal(paths))
        cash *= np.exp(r * dt)
        if i < rebalances:
            new = _call_delta(S, K, T - i * dt, r, sigma_implied)
            cash -= (new - delta) * S
            delta = new
    pnl = cash + delta * S - np.maximum(S - K, 0.0)
    return {"mean": float(pnl.mean()), "std": float(pnl.std(ddof=1)), "premium": premium}


def straddle_delta(S, K, tau, r, sigma):
    return 2 * _call_delta(S, K, tau, r, sigma) - 1


def vrp_study(daily: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """`daily` has columns ret (market total return), rf (daily T-bill) and vix (percent), indexed by date.

    A straddle is sold on the first trading day of each month and held `horizon` trading days.
    Returns one row per trade: the premium, the delta-hedged P&L and the unhedged P&L, all as a
    share of the index level at the sale.
    """
    d = daily.dropna().copy()
    level = (1 + d["ret"]).cumprod()
    starts = d.index.to_series().groupby([d.index.year, d.index.month]).first().to_numpy()
    rows = []
    idx = {t: i for i, t in enumerate(d.index)}
    for t0 in starts:
        i0 = idx[t0]
        if i0 + horizon >= len(d):
            break
        S0 = float(level.iloc[i0])
        sigma = float(d["vix"].iloc[i0]) / 100
        r = float(d["rf"].iloc[i0]) * 252
        T = horizon / 252
        prem = float(bsm(S0, S0, T, r, sigma) + bsm(S0, S0, T, r, sigma, kind="put"))
        cash = prem
        delta = float(straddle_delta(S0, S0, T, r, sigma))
        cash -= delta * S0
        S = S0
        for k in range(1, horizon + 1):
            S = float(level.iloc[i0 + k])
            cash *= 1 + float(d["rf"].iloc[i0 + k])
            if k < horizon:
                new = float(straddle_delta(S, S0, (horizon - k) / 252, r, sigma))
                cash -= (new - delta) * S
                delta = new
        payoff = abs(S - S0)
        hedged = cash + delta * S - payoff
        rows.append({"date": pd.Timestamp(t0), "vix": sigma * 100, "premium": prem / S0,
                     "hedged_pnl": hedged / S0, "unhedged_pnl": (prem - payoff) / S0,
                     "realised_vol": float(d["ret"].iloc[i0 + 1:i0 + horizon + 1].std(ddof=1) * np.sqrt(252))})
    return pd.DataFrame(rows).set_index("date")
