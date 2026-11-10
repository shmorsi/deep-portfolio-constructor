"""The scenario generator: possible futures for stocks, the bond yield and inflation.

Deliberately simple and transparent, as scenario generators go:

* the 10-year yield and monthly inflation each follow an AR(1) fitted by least squares
  (next month's value is pulled back toward a long-run level, plus a shock);
* stock returns are taken as they came;
* the three monthly shocks (stock return, yield shock, inflation shock) are drawn together
  from history in blocks (the stationary bootstrap of Politis and Romano, 1994), so a scenario
  keeps history's fat tails, its crashes that last several months, and the way stocks, yields
  and inflation moved together.

It has no views: no forecasts of returns, no regime model, no term structure. It can only
rearrange what has already happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

Y_FLOOR = 0.001   # a yield never falls below 0.10%
PHI_CAP = 0.995   # keeps a fitted AR(1) from being a random walk when a window trends


@dataclass(frozen=True)
class Model:
    a_y: float
    phi_y: float
    a_pi: float
    phi_pi: float
    shocks: np.ndarray   # (N, 3): stock return, yield shock, inflation shock, month by month
    states: np.ndarray   # (N, 2): the yield and the inflation at the start of each of those months

    @property
    def mean_yield(self) -> float:
        """The level the yield is pulled toward (not defined for a random walk)."""
        return float("nan") if self.phi_y >= 1.0 else self.a_y / (1.0 - self.phi_y)


def _ar1(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Least-squares AR(1): x[t] = a + phi x[t-1] + e[t], with phi capped just below one."""
    prev, now = x[:-1], x[1:]
    phi = float(np.cov(prev, now, bias=True)[0, 1] / np.var(prev))
    phi = min(phi, PHI_CAP)
    a = float(now.mean() - phi * prev.mean())
    return a, phi, now - a - phi * prev


def fit(panel: pd.DataFrame, random_walk: bool = False) -> Model:
    """Fit on a monthly panel with columns equity, yield, infl (see data.py). Uses every row given, and nothing else.

    random_walk: give the yield no pull back to an average (each month's yield is last month's plus a change
    that happened in history). Used for the model-risk test: futures in which yields never come back.
    """
    y, pi, eq = panel["yield"].to_numpy(), panel["infl"].to_numpy(), panel["equity"].to_numpy()
    a_y, phi_y, e_y = (0.0, 1.0, np.diff(y)) if random_walk else _ar1(y)
    a_pi, phi_pi, e_pi = _ar1(pi)
    shocks = np.column_stack([eq[1:], e_y, e_pi])
    states = np.column_stack([y[:-1], pi[:-1]])
    return Model(a_y, phi_y, a_pi, phi_pi, shocks, states)


BURN = 122   # months before the plan starts: ten years of published inflation (and a month's lag) for the plan to look back on


def simulate(model: Model, n: int, months: int, seed: int, block: float = 12.0) -> dict[str, np.ndarray]:
    """n scenarios: BURN months of run-up, then `months` months of plan life.

    Each scenario begins its run-up from a month of history drawn at random, so the set covers
    every past regime. Time 0 is the plan's first decision. Returns, with t counted from there:

      yield  (n, BURN + months + 1)  month-end yields at t = -BURN .. months
      equity (n, BURN + months)      stock returns over the months ending at t = -BURN+1 .. months
      infl   (n, BURN + months)      inflation over the same months
    """
    rng = np.random.default_rng(seed)
    N, total = len(model.shocks), BURN + months
    idx = np.empty((n, total), dtype=np.int64)
    cur = rng.integers(0, N, n)
    for t in range(total):                       # stationary bootstrap: run on, or jump with probability 1/block
        if t:
            jump = rng.random(n) < 1.0 / block
            cur = np.where(jump, rng.integers(0, N, n), (cur + 1) % N)
        idx[:, t] = cur
    drawn = model.shocks[idx]                    # (n, total, 3)
    s0 = model.states[rng.integers(0, N, n)]
    y = np.empty((n, total + 1))
    infl = np.empty((n, total))
    y[:, 0] = s0[:, 0]
    last_pi = s0[:, 1].copy()
    for t in range(total):
        y[:, t + 1] = np.maximum(Y_FLOOR, model.a_y + model.phi_y * y[:, t] + drawn[:, t, 1])
        last_pi = model.a_pi + model.phi_pi * last_pi + drawn[:, t, 2]
        infl[:, t] = last_pi
    return {"equity": drawn[:, :, 0].copy(), "yield": y, "infl": infl}


def from_history(panel: pd.DataFrame, start: int, months: int) -> dict[str, np.ndarray]:
    """What actually happened: the plan's first decision is taken at row `start`, with the BURN months before it as its past."""
    if start < BURN or start + months >= len(panel):
        raise ValueError("not enough history around the start")
    rows = panel.iloc[start - BURN:start + months + 1]
    return {
        "equity": rows["equity"].to_numpy()[None, 1:].copy(),
        "yield": rows["yield"].to_numpy()[None, :].copy(),
        "infl": rows["infl"].to_numpy()[None, 1:].copy(),
    }
