"""How often to trade: the same trained network on different rebalancing schedules.

Added after the pre-registered test. Nothing is retrained: the deployed policy
(the walk-forward networks of all five seeds, averaged) makes every decision.

* ``monthly``: as tested, a decision at every month-end.
* ``quarterly`` and ``yearly``: decisions only at the end of March, June,
  September and December, or only at the end of December; between them the
  holdings drift with prices, untouched.
* ``daily``: a decision at every month-end, then every trading day the
  portfolio is traded back to the weights the network chose, on Kenneth
  French's daily industry returns, paying the same cost per unit traded. The
  network itself still decides monthly: its inputs are monthly.

``run_daily`` with ``true_up=False`` trades only at month-ends, so it should
reproduce the monthly engine; that is how the daily engine is checked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import MISSING, IndustryPanel, _fetch_zip
from .engine import Decide, RunResult, check_weights

SCHEDULES = {"daily": None, "monthly": frozenset(range(1, 13)), "quarterly": frozenset({3, 6, 9, 12}), "yearly": frozenset({12})}
DAILY_FILE = "30_Industry_Portfolios_Daily_CSV.zip"
DAILY_TABLE = "Average Value Weighted Returns -- Daily"


def on_schedule(decide: Decide, panel: IndustryPanel, months: frozenset[int]) -> Decide:
    """Trade only at month-ends in ``months`` (and when starting from cash); hold the drifted weights otherwise."""
    def wrapped(t: int, w_drift: np.ndarray) -> np.ndarray:
        if w_drift.sum() < 0.5 or panel.dates[t].month in months:
            return decide(t, w_drift)
        return w_drift
    return wrapped


def parse_daily(text: str, names: list[str]) -> pd.DataFrame:
    """French's daily file: the value-weighted table, as decimal returns indexed by date."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == DAILY_TABLE)
    header = [h.strip() for h in lines[start + 1].split(",")[1:]]
    if header[:len(names)] != list(names):
        raise ValueError("daily industry columns do not match the monthly panel")
    rows = []
    for line in lines[start + 2:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append([parts[0]] + [float(x) for x in parts[1:len(names) + 1]])
    df = pd.DataFrame(rows, columns=["date", *names])
    df.index = pd.to_datetime(df.pop("date"), format="%Y%m%d")
    return df.replace(list(MISSING), np.nan) / 100.0


def load_daily(base_url: str, cache_dir: Path, names: list[str], refresh: bool = False) -> pd.DataFrame:
    cache = cache_dir / DAILY_FILE.replace(".zip", ".txt")
    if refresh or not cache.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(_fetch_zip(base_url + DAILY_FILE), encoding="latin-1")
    return parse_daily(cache.read_text(encoding="latin-1"), names)


def run_daily(panel: IndustryPanel, daily: pd.DataFrame, decide: Decide, start: int, end: int, cost_bps: float, name: str,
              true_up: bool = True) -> RunResult:
    """Months start..end on daily returns. Decisions at month-ends as in engine.run; with ``true_up`` the portfolio
    is also traded back to that month's chosen weights at every close before the month's last day."""
    c = cost_bps / 1e4
    n = panel.n_assets
    w = np.zeros(n)
    months = end - start + 1
    net, gross, turn = np.empty(months), np.empty(months), np.empty(months)
    held = np.empty((months, n))
    day_index = daily.index
    for k, t in enumerate(range(start - 1, end)):
        w_new = check_weights(np.asarray(decide(t, w.copy()), float), f"{name} at {panel.dates[t]:%Y-%m}")
        traded = float(np.abs(w_new - w).sum())
        lo, hi = panel.dates[t], panel.dates[t + 1]
        days = daily.to_numpy()[(day_index > lo) & (day_index <= hi)]
        if len(days) == 0 or np.isnan(days).any():
            raise ValueError(f"no complete daily returns for {hi:%Y-%m}")
        value, value_gross, w_cur = 1.0 - c * traded, 1.0, w_new.copy()
        for i, r in enumerate(days):
            g = float(w_cur @ r)
            value *= 1.0 + g
            value_gross *= 1.0 + g
            w_cur = w_cur * (1.0 + r) / (1.0 + g)
            if true_up and i < len(days) - 1:          # the month's last close is the next decision's job
                fix = float(np.abs(w_new - w_cur).sum())
                value *= 1.0 - c * fix
                traded += fix
                w_cur = w_new.copy()
        net[k], gross[k], turn[k], held[k] = value - 1.0, value_gross - 1.0, traded, w_new
        w = w_cur
    return RunResult(name=name, dates=panel.dates[start:end + 1], net=net, gross=gross, turnover=turn, weights=held)
