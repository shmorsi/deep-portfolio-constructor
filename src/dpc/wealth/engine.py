"""The monthly investing simulator, shared by every strategy.

At the end of each month a strategy sees its current holdings, which have
drifted with prices since the last trade, and chooses new long-only weights
that sum to one. It pays ``cost_bps`` per unit traded (one way), then holds
the new weights through the next month. Doing nothing is free: a buy-and-hold
investor's weights simply drift.

The same arithmetic, written in PyTorch, is what the Deep Growth Network is
trained through (see model.py); tests check that the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import IndustryPanel

Decide = Callable[[int, np.ndarray], np.ndarray]  # (decision index t, drifted weights) -> new weights


@dataclass(frozen=True)
class RunResult:
    name: str
    dates: pd.DatetimeIndex   # the months the returns were earned in
    net: np.ndarray           # [M] net-of-cost total return of each month
    gross: np.ndarray         # [M]
    turnover: np.ndarray      # [M] sum |w_new - w_drift| at the decision before the month
    weights: np.ndarray       # [M, N] weights held during the month

    def wealth(self) -> np.ndarray:
        return np.cumprod(1.0 + self.net)

    def series(self) -> pd.Series:
        return pd.Series(self.net, index=self.dates, name=self.name)


def check_weights(w: np.ndarray, where: str) -> np.ndarray:
    if np.any(w < -1e-9) or not np.isfinite(w).all() or abs(w.sum() - 1.0) > 1e-6:
        raise ValueError(f"{where}: weights must be long-only and sum to 1 (min {w.min():.3g}, sum {w.sum():.6f})")
    w = np.clip(w, 0.0, None)
    return w / w.sum()


def run(panel: IndustryPanel, decide: Decide, start: int, end: int, cost_bps: float, name: str,
        initial: np.ndarray | None = None) -> RunResult:
    """Simulate months start..end (inclusive). Decisions are taken at t = start-1 .. end-1.

    ``initial``: holdings before the first decision; None means cash, so the first
    purchase pays the cost of buying the whole portfolio."""
    if start < 1 or end >= panel.n_months or end < start:
        raise ValueError("bad simulation window")
    n = panel.n_assets
    c = cost_bps / 1e4
    w_drift = np.zeros(n) if initial is None else check_weights(np.asarray(initial, float), f"{name} initial")
    months = end - start + 1
    net, gross, turn = np.empty(months), np.empty(months), np.empty(months)
    held = np.empty((months, n))
    for k, t in enumerate(range(start - 1, end)):
        w_new = check_weights(np.asarray(decide(t, w_drift.copy()), float), f"{name} at {panel.dates[t]:%Y-%m}")
        traded = float(np.abs(w_new - w_drift).sum())
        r = panel.returns[t + 1]
        g = float(w_new @ r)
        net[k] = (1.0 - c * traded) * (1.0 + g) - 1.0
        gross[k], turn[k], held[k] = g, traded, w_new
        w_drift = w_new * (1.0 + r) / (1.0 + g)
    return RunResult(name=name, dates=panel.dates[start:end + 1], net=net, gross=gross, turnover=turn, weights=held)
