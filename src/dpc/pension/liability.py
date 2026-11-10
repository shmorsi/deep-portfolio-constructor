"""The plan's promise: a schedule of monthly payments, and what it is worth along a scenario.

The schedule runs for `years` and each payment is a fixed fraction smaller than the one before
(a mature plan running off). Payments are indexed to consumer prices: each one is paid in the
money of the month it falls due.

The liability is the value of that promise today. Future payments are projected to grow with
expected inflation and are discounted at the 10-year government yield, which is the same as
discounting today's-money payments at a real yield. Two simplifications, both stated in the
protocol: one yield for every maturity (a flat curve), and expected inflation is simply the
average of the last ten years.

Because the payments shrink geometrically, the value has a closed form (`Schedule.unit_value`),
checked against discounting every payment one by one in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _geometric(ratio: np.ndarray, n: int) -> np.ndarray:
    """1 + ratio + ... + ratio^(n-1), safe where the ratio is one."""
    ratio = np.asarray(ratio, dtype=float)
    near = np.abs(ratio - 1.0) < 1e-10
    safe = np.where(near, 0.5, ratio)
    return np.where(near, float(n), (1.0 - safe ** n) / (1.0 - safe))


@dataclass(frozen=True)
class Schedule:
    years: int = 60            # the last payment is this far away at the start
    decay_years: float = 40.0  # each payment is exp(-1 / (12 * decay_years)) times the one before

    @property
    def months(self) -> int:
        return self.years * 12

    @property
    def g(self) -> float:
        return float(np.exp(-1.0 / (12.0 * self.decay_years)))

    def cash(self, first: float = 1.0) -> np.ndarray:
        """The payments in today's money: `first` next month, then shrinking by g each month."""
        return first * self.g ** np.arange(self.months)

    def unit_value(self, y: np.ndarray, expected_inflation: np.ndarray = 0.0, elapsed: int = 0) -> np.ndarray:
        """Value of the payments still to come after `elapsed` months, in today's money, per unit of the first payment ever due.

        The payment due in k months is g^(elapsed + k - 1) in today's money, expected to be u^k times
        that when paid (u = 1 + expected inflation / 12), and is discounted by d^k (d = 1 / (1 + y / 12)):
        a geometric series with first term g^elapsed (u d) and ratio g u d.
        """
        y = np.asarray(y, dtype=float)
        ud = (1.0 + np.asarray(expected_inflation, dtype=float) / 12.0) / (1.0 + y / 12.0)
        return self.g ** elapsed * ud * _geometric(self.g * ud, self.months - elapsed)


def paths(sched: Schedule, y: np.ndarray, infl: np.ndarray, expected: np.ndarray, start_value: float = 100.0) -> dict[str, np.ndarray]:
    """The liability along each scenario.

    y:        (S, T+1) month-end yields, column 0 at the start
    infl:     (S, T)   inflation over each month
    expected: (S, T+1) expected annual inflation at each month-end

    Returns, per scenario:
      liability (S, T+1)  the promise's value; `start_value` at the start
      benefit   (S, T)    the payment made at the end of each month, in that month's money
      matching  (S, T)    the month's return on nominal bonds bought to pay exactly the schedule
                          as it was expected at the start of the month. It hedges the yield; it
                          does not hedge inflation turning out different from what was expected.
      index     (S, T+1)  cumulative inflation since the start
    """
    S, T1 = y.shape
    T = T1 - 1
    g = sched.g
    first = start_value / sched.unit_value(y[:, 0], expected[:, 0])       # scale each scenario's schedule to start_value
    index = np.concatenate([np.ones((S, 1)), np.cumprod(1.0 + infl, axis=1)], axis=1)
    unit = np.stack([sched.unit_value(y[:, t], expected[:, t], elapsed=t) for t in range(T1)], axis=1)
    due = g ** np.arange(T)[None, :]                                        # the payment due at the end of month t+1, in today's money
    # Nominal bonds bought at t pay due * u, due * g * u^2, ... (u from the expectation at t). A month on, the first has
    # been paid and the rest are worth u * sum over j = 1..n-1 of g^(t+j) (u d')^j at the new yield (d').
    u = 1.0 + expected[:, :-1] / 12.0
    ud_next = u / (1.0 + y[:, 1:] / 12.0)
    left = np.stack([g ** (t + 1) * ud_next[:, t] * _geometric(g * ud_next[:, t], sched.months - t - 1) for t in range(T)], axis=1)
    matching = (due * u + u * left) / unit[:, :-1] - 1.0
    return {
        "liability": first[:, None] * index * unit,
        "benefit": first[:, None] * index[:, 1:] * due,
        "matching": matching,
        "index": index,
    }
