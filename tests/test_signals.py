"""Stage 2: signal construction is correct and uses no future data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dpc.data.panel import Panel, month_end_dates
from dpc.signals import (
    build_signals,
    cross_sectional_z,
    forward_returns,
    information_coefficients,
    monthly_returns,
    raw_signals,
)
from dpc.stats import newey_west_mean


def _toy_panel(n_days: int = 700, n_stocks: int = 12, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2015-01-01", periods=n_days)
    tick = [f"S{i}" for i in range(n_stocks)]
    rets = pd.DataFrame(rng.normal(0.0004, 0.015, (n_days, n_stocks)), index=cal, columns=tick)
    me = month_end_dates(cal)[:-1]
    elig = pd.DataFrame(True, index=me, columns=tick)
    elig.iloc[:12] = False
    uni = pd.DataFrame({"name": tick, "country": "US", "sector": ["A", "B"] * (n_stocks // 2)}, index=tick)
    ff = pd.DataFrame(0.0, index=cal, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"])
    return Panel(returns=rets, prices=(1 + rets).cumprod(), ff=ff, benchmark=rets.mean(axis=1), eligible=elig, universe=uni)


def test_monthly_returns_compound_daily() -> None:
    cal = pd.bdate_range("2024-01-01", "2024-02-29")
    daily = pd.DataFrame({"A": 0.01}, index=cal)
    m = monthly_returns(daily, month_end_dates(cal))
    n_jan = (cal.month == 1).sum()
    assert np.isclose(m["A"].iloc[0], 1.01**n_jan - 1)
    assert m.index[0] == cal[cal.month == 1][-1]


def test_monthly_return_is_nan_for_partial_month() -> None:
    cal = pd.bdate_range("2024-01-01", "2024-02-29")
    daily = pd.DataFrame({"A": 0.01}, index=cal)
    daily.iloc[:5] = np.nan  # listed mid-January
    m = monthly_returns(daily, month_end_dates(cal))
    assert np.isnan(m["A"].iloc[0]) and np.isfinite(m["A"].iloc[1])


def test_momentum_skips_last_month(cfg) -> None:
    p = _toy_panel()
    mret = monthly_returns(p.returns, p.month_ends)
    mom = raw_signals(p, cfg.signals)["momentum"]
    t = 20
    expected = (1 + mret.iloc[t - 11 : t]).prod() - 1  # months t-11 .. t-1
    np.testing.assert_allclose(mom.iloc[t].values, expected.values)


def test_signals_are_point_in_time(cfg) -> None:
    """Changing returns AFTER month-end t must not change any signal at <= t."""
    p = _toy_panel()
    base = build_signals(p, cfg.signals)
    t = p.month_ends[20]
    p2 = _toy_panel()
    p2.returns.loc[p2.returns.index > t] = 0.5  # absurd future
    shocked = build_signals(p2, cfg.signals)
    for k in base:
        pd.testing.assert_frame_equal(base[k].loc[:t], shocked[k].loc[:t])


def test_zscores_standardised_and_winsorised() -> None:
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.standard_t(2, (30, 50)))
    z = cross_sectional_z(x, pd.DataFrame(True, index=x.index, columns=x.columns), clip=3.0)
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=1), 1, atol=1e-12)
    assert z.abs().max().max() <= 3.0 + 1e-4  # iterated winsorisation holds the bound


def test_ic_of_perfect_and_inverse_signal() -> None:
    rng = np.random.default_rng(2)
    fwd = pd.DataFrame(rng.normal(size=(24, 40)))
    assert np.allclose(information_coefficients(fwd, fwd), 1.0)
    assert np.allclose(information_coefficients(-fwd, fwd), -1.0)


def test_forward_returns_shift() -> None:
    m = pd.DataFrame({"A": [0.1, 0.2, 0.3]})
    assert forward_returns(m)["A"].tolist()[:2] == [0.2, 0.3]


def test_newey_west_on_noise_is_insignificant() -> None:
    x = pd.Series(np.random.default_rng(3).normal(size=500))
    _, t, n = newey_west_mean(x, lags=6)
    assert abs(t) < 3 and n == 500
