"""Monthly realized volatility of the 30 industries, from Kenneth French's daily returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..wealth.data import IndustryPanel

MIN_DAYS = 15          # a month with fewer daily returns for an industry is missing for it
EWMA_DECAY = 0.94      # RiskMetrics, per day
MONTH_DAYS = 21        # trading days a month is scaled to


@dataclass(frozen=True)
class RiskPanel:
    """Everything the risk study reads, aligned on the wealth panel's month-ends."""

    dates: pd.DatetimeIndex      # month-ends, the wealth panel's
    names: list[str]
    rv: np.ndarray               # [T, N] realized volatility of month t (nan if missing)
    ewma: np.ndarray             # [T, N] RiskMetrics volatility for a 21-day month, known at the end of month t
    daily: pd.DataFrame          # daily decimal returns, the wealth study's daily file

    def index_of(self, month: str) -> int:
        ts = pd.Timestamp(month) + pd.offsets.MonthEnd(0)
        pos = self.dates.get_indexer([ts])[0]
        if pos < 0:
            raise KeyError(f"{month} is outside the panel {self.dates[0]:%Y-%m}..{self.dates[-1]:%Y-%m}")
        return int(pos)


def monthly_rv(daily: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """[T, N]: the square root of the sum of squared daily returns in each month of ``dates``."""
    month = daily.index.to_period("M")
    sq = (daily ** 2).groupby(month).sum(min_count=1)
    n = daily.notna().groupby(month).sum()
    rv = np.sqrt(sq.where(n >= MIN_DAYS))
    return rv.reindex(dates.to_period("M")).to_numpy(float)


def ewma_month_end(daily: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """[T, N]: sqrt(21 x the exponentially weighted daily variance) at the last trading day of each month.

    The recursion starts from the first squared return and skips a missing day (the variance carries over)."""
    r = daily.to_numpy(float)
    var = np.full_like(r, np.nan)
    state = np.full(r.shape[1], np.nan)
    for d in range(r.shape[0]):
        x = r[d] ** 2
        seen = np.isfinite(x)
        start = seen & ~np.isfinite(state)
        state = np.where(start, x, np.where(seen, EWMA_DECAY * state + (1.0 - EWMA_DECAY) * x, state))
        var[d] = state
    last = pd.DataFrame(var, index=daily.index, columns=daily.columns).groupby(daily.index.to_period("M")).last()
    return np.sqrt(MONTH_DAYS * last.reindex(dates.to_period("M")).to_numpy(float))


def build(panel: IndustryPanel, daily: pd.DataFrame) -> RiskPanel:
    if list(daily.columns) != list(panel.names):
        raise ValueError("daily industry columns do not match the monthly panel")
    return RiskPanel(dates=panel.dates, names=list(panel.names), rv=monthly_rv(daily, panel.dates),
                     ewma=ewma_month_end(daily, panel.dates), daily=daily)


def save_npz(rp: RiskPanel, path: Path) -> None:
    """The realized-volatility panel in a form the forecast script reads with numpy alone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dates=np.array([f"{d:%Y-%m}" for d in rp.dates]), names=np.array(rp.names), rv=rp.rv)
