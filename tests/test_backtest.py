"""The backtest engine: accounting, costs, drift and look-ahead guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dpc.backtest import LookAheadError, PointInTimeView, run_backtest
from dpc.baselines import make_long_only, make_ls_classical
from dpc.metrics import max_drawdown
from helpers import toy_panel, toy_signals_betas


def _const(weights: dict[str, float]):
    return lambda view, w0: pd.Series(weights)


def test_single_stock_portfolio_earns_stock_return() -> None:
    p = toy_panel()
    s, b = toy_signals_betas(p)
    start = str(p.month_ends[2].date())
    res = run_backtest("one", p, _const({"S00": 1.0}), s, b, start, cost_bps=0.0, long_only=True)
    days = res.daily_gross.index[1:]
    np.testing.assert_allclose(res.daily_gross.loc[days], p.returns.loc[days, "S00"])


def test_buy_and_hold_drift_matches_manual() -> None:
    p = toy_panel()
    s, b = toy_signals_betas(p)
    start = str(p.month_ends[2].date())
    end = p.month_ends[3]
    res = run_backtest("pair", p, _const({"S00": 0.5, "S01": 0.5}), s, b, start, 0.0, True, end=end)
    days = res.daily_gross.loc[p.month_ends[2] : end].index[1:]
    # holdings grow independently; portfolio wealth is their sum
    growth = (1 + p.returns.loc[days, ["S00", "S01"]]).cumprod()
    wealth = 0.5 * growth["S00"] + 0.5 * growth["S01"]
    np.testing.assert_allclose((1 + res.daily_gross.loc[days]).cumprod(), wealth, rtol=1e-12)


def test_costs_equal_bps_times_turnover() -> None:
    p = toy_panel()
    s, b = toy_signals_betas(p)
    start = str(p.month_ends[2].date())
    res = run_backtest("ew", p, lambda v, w0: pd.Series(1 / 12, index=v.tickers), s, b, start, 10.0, True)
    assert np.isclose(res.turnover.iloc[0], 1.0)  # building from cash
    np.testing.assert_allclose(res.costs, res.turnover * 10 / 1e4)
    np.testing.assert_allclose(res.daily_gross - res.daily_net, res.costs.reindex(res.daily_net.index).fillna(0))
    # after the first month, turnover is only the drift back to equal weight
    assert res.turnover.iloc[1:].max() < 0.2


def test_dollar_neutral_return_is_long_minus_short() -> None:
    p = toy_panel()
    s, b = toy_signals_betas(p)
    start = str(p.month_ends[2].date())
    res = run_backtest("ls", p, _const({"S00": 1.0, "S01": -1.0}), s, b, start, 0.0, False, end=p.month_ends[3])
    d = res.daily_gross.index[1]
    r = p.returns.loc[d]
    assert np.isclose(res.daily_gross.loc[d], r["S00"] - r["S01"])


def test_engine_rejects_weight_on_non_investable_stock() -> None:
    p = toy_panel()
    p.eligible.loc[:, "S05"] = False
    s, b = toy_signals_betas(p)
    with pytest.raises(LookAheadError):
        run_backtest("bad", p, _const({"S05": 1.0}), s, b, str(p.month_ends[2].date()), 0.0, True)


def test_point_in_time_view_hides_the_future() -> None:
    p = toy_panel()
    s, b = toy_signals_betas(p)
    t = p.month_ends[5]
    v = PointInTimeView(t, p, s, b)
    assert v.returns.index.max() == t
    assert v.rf.index.max() == t


@pytest.mark.parametrize("strategy", ["equal_weight", "min_variance", "hrp", "ls_classical"])
def test_truncation_invariance(cfg, strategy: str) -> None:
    """Look-ahead check: weights decided at or before T must not depend on data after T."""
    import dataclasses

    cfg = cfg.replace(backtest=dataclasses.replace(cfg.backtest, cov_window_days=60))
    p = toy_panel(n_days=520)
    s, b = toy_signals_betas(p)
    start = str(p.month_ends[3].date())
    T = p.month_ends[12]

    def fn():
        if strategy == "ls_classical":
            return make_ls_classical(cfg, len(p.tickers), 3)[0]
        return make_long_only(strategy, cfg)

    full = run_backtest("f", p, fn(), s, b, start, 10.0, strategy != "ls_classical")
    p_cut = toy_panel(n_days=520)
    future = p_cut.returns.index > T
    noise = np.random.default_rng(99).normal(0.01, 0.2, (int(future.sum()), len(p.tickers)))
    p_cut.returns.loc[future] = noise  # a completely different future
    cut = run_backtest("c", p_cut, fn(), s, b, start, 10.0, strategy != "ls_classical")
    pd.testing.assert_frame_equal(full.weights.loc[:T], cut.weights.loc[:T], atol=1e-7)


def test_max_drawdown() -> None:
    r = pd.Series([0.10, -0.50, 0.20])
    assert np.isclose(max_drawdown(r), -0.5)
