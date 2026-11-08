"""Synthetic panels for engine tests (known answers, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dpc.data.panel import Panel, month_end_dates

FF_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]


def toy_panel(n_days: int = 520, n_stocks: int = 12, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2015-01-01", periods=n_days)
    tick = [f"S{i:02d}" for i in range(n_stocks)]
    rets = pd.DataFrame(rng.normal(0.0004, 0.015, (n_days, n_stocks)), index=cal, columns=tick)
    me = month_end_dates(cal)[:-1]
    elig = pd.DataFrame(True, index=me, columns=tick)
    uni = pd.DataFrame(
        {"name": tick, "country": "US", "sector": ["Alpha", "Beta", "Gamma"] * (n_stocks // 3)}, index=tick
    )
    ff = pd.DataFrame(rng.normal(0, 0.01, (n_days, len(FF_COLS))), index=cal, columns=FF_COLS)
    ff["RF"] = 0.0001
    return Panel(returns=rets, prices=(1 + rets).cumprod(), ff=ff, benchmark=rets.mean(axis=1), eligible=elig, universe=uni)


def toy_signals_betas(panel: Panel, seed: int = 1) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    shape = (len(panel.month_ends), len(panel.tickers))
    mk = lambda: pd.DataFrame(rng.normal(size=shape), index=panel.month_ends, columns=panel.tickers)  # noqa: E731
    signals = {k: mk() for k in ("momentum", "reversal", "low_vol", "combined")}
    betas = {f: mk() * 0.3 + (1.0 if f == "Mkt-RF" else 0.0) for f in ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")}
    return signals, betas
