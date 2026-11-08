"""Stage 6: the independent risk decomposition gets known answers right."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dpc.backtest import BacktestResult
from dpc.data.panel import Panel, month_end_dates
from dpc.evaluation import build_eval_risk_model, decompose
from dpc.exposures import STYLE_FACTORS


def _factor_world(n_days: int = 900, n: int = 60, seed: int = 0):
    """Returns = beta * market + small sector effect + idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2012-01-02", periods=n_days)
    tick = [f"S{i:02d}" for i in range(n)]
    beta = rng.uniform(0.5, 1.5, n)
    mkt = rng.normal(0, 0.012, n_days)
    rets = pd.DataFrame(np.outer(mkt, beta) + rng.normal(0, 0.01, (n_days, n)), index=cal, columns=tick)
    me = month_end_dates(cal)[:-1]
    uni = pd.DataFrame({"name": tick, "country": "US", "sector": [f"Sec{i % 4}" for i in range(n)]}, index=tick)
    ff = pd.DataFrame(0.0, index=cal, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"])
    panel = Panel(rets, (1 + rets).cumprod(), ff, rets.mean(axis=1), pd.DataFrame(True, index=me, columns=tick), uni)
    betas = {f: pd.DataFrame(rng.normal(0, 0.05, (len(me), n)), index=me, columns=tick) for f in STYLE_FACTORS}
    betas["Mkt-RF"] = pd.DataFrame(np.tile(beta, (len(me), 1)), index=me, columns=tick)
    return panel, betas, beta


def _result(panel: Panel, w: np.ndarray, start: int = 14) -> BacktestResult:
    dates = panel.month_ends[start:-1]
    W = pd.DataFrame(np.tile(w, (len(dates), 1)), index=dates, columns=panel.tickers)
    z = pd.Series(0.0, index=panel.returns.index)
    return BacktestResult("x", False, W, z, z, pd.Series(dtype=float), pd.Series(dtype=float))


def test_market_bet_is_almost_all_factor_risk() -> None:
    panel, betas, beta = _factor_world()
    erm = build_eval_risk_model(panel, betas)
    w = np.full(len(beta), 1 / len(beta))  # long-only, fully invested: a market bet
    ex_ante, ex_post = decompose(_result(panel, w), erm, window=200)
    assert ex_ante["share_specific"].mean() < 0.2
    assert ex_post["realised_share_factor"] > 0.8


def test_beta_and_sector_neutral_book_is_mostly_specific_risk() -> None:
    panel, betas, beta = _factor_world()
    erm = build_eval_risk_model(panel, betas)
    rng = np.random.default_rng(5)
    sec = panel.universe["sector"].to_numpy()
    w = rng.normal(size=len(beta))
    # project out market beta and sector exposure -> stock-specific bets only
    X = np.column_stack([beta, *[(sec == s).astype(float) for s in sorted(set(sec))]])
    w = w - X @ np.linalg.lstsq(X, w, rcond=None)[0]
    w = w / np.abs(w).sum() * 2
    ex_ante, ex_post = decompose(_result(panel, w), erm, window=200)
    assert ex_ante["share_specific"].mean() > 0.8
    assert ex_post["realised_share_specific"] > 0.8
    assert abs(ex_ante["exp_style:Mkt-RF"]).max() < 1e-8
