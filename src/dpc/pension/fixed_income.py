"""Bond mathematics used by the pension study. Yields are annual, in decimals.

Two conventions, both standard:
* a coupon bond pays semi-annually and is priced with semi-annual compounding (US Treasuries);
* a schedule of monthly cash flows is discounted with monthly compounding of the annual yield.
"""

from __future__ import annotations

import numpy as np


def par_bond_return(y_prev: np.ndarray, y_now: np.ndarray, maturity: float = 10.0) -> np.ndarray:
    """One month's total return on a bond bought at par a month ago.

    The bond is bought at par (coupon = last month's yield) with `maturity` years to run, held
    for a month and sold at this month's yield with a month less to run. This is the usual way
    to turn a constant-maturity yield series into a total-return series: one month of coupon
    plus the price change from the move in yields and the shorter maturity.
    """
    y_prev = np.asarray(y_prev, dtype=float)
    y_now = np.asarray(y_now, dtype=float)
    years = maturity - 1.0 / 12.0
    disc = (1.0 + y_now / 2.0) ** (-2.0 * years)
    safe = np.where(np.abs(y_now) < 1e-9, 1.0, y_now)
    price = np.where(np.abs(y_now) < 1e-9, 1.0 + y_prev * years, y_prev / safe * (1.0 - disc) + disc)
    return price - 1.0 + y_prev / 12.0


def present_value(cash: np.ndarray, y: float) -> float:
    """Value of monthly payments cash[0], cash[1], ... due in 1, 2, ... months, at annual yield y."""
    k = np.arange(1, len(cash) + 1)
    return float(np.sum(cash * (1.0 + y / 12.0) ** (-k)))


def duration(cash: np.ndarray, y: float) -> float:
    """Modified duration in years: the percentage fall in value for a one-point rise in the yield."""
    k = np.arange(1, len(cash) + 1)
    d = 1.0 / (1.0 + y / 12.0)
    return float(np.sum(k / 12.0 * cash * d ** (k + 1)) / present_value(cash, y))


def convexity(cash: np.ndarray, y: float) -> float:
    """Convexity in years squared: how fast the duration itself changes as the yield moves."""
    k = np.arange(1, len(cash) + 1)
    d = 1.0 / (1.0 + y / 12.0)
    return float(np.sum(k * (k + 1) / 144.0 * cash * d ** (k + 2)) / present_value(cash, y))
