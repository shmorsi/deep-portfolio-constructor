"""Zero curves, interest-rate swaps and their risk.

**Bootstrapping.** US Treasury constant-maturity yields are *par* yields: the coupon at which a
bond prices at par. With semiannual coupons c, a par bond maturing at the n-th coupon date
satisfies 1 = (c/2) sum_{i<=n} D_i + D_n, so the discount factors follow one at a time:

    D_n = (1 - (c_n/2) sum_{i<n} D_i) / (1 + c_n/2).

Par yields between quoted maturities are interpolated linearly. Between grid points the curve
interpolates log discount factors linearly (piecewise-flat forwards).

**Swaps** (single curve, the textbook case): the floating leg of a swap is worth 1 - D(T) per unit
notional, the fixed leg K times the annuity A = sum tau_i D(t_i); the par rate is (1 - D(T)) / A.
A receive-fixed swap gains when rates fall: the instrument a pension buys to hedge its liability.
Its DV01 is measured by bumping the zero curve 1 bp. Real desks discount on OIS and project on a
separate curve; that distinction is stated, not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Curve:
    times: np.ndarray        # strictly increasing, years
    dfs: np.ndarray          # discount factors at those times

    def df(self, t) -> np.ndarray:
        """Log-linear interpolation; past the last point the last forward rate continues."""
        t = np.atleast_1d(np.asarray(t, float))
        lt = np.concatenate([[0.0], self.times])
        ld = np.concatenate([[0.0], np.log(self.dfs)])
        inside = np.interp(t, lt, ld)
        slope = (ld[-1] - ld[-2]) / (lt[-1] - lt[-2])
        return np.exp(np.where(t <= lt[-1], inside, ld[-1] + slope * (t - lt[-1])))

    def zero(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        return -np.log(self.df(t)) / t

    def forward(self, t1: float, t2: float) -> float:
        return float(np.log(self.df(t1)[0] / self.df(t2)[0]) / (t2 - t1))

    def shifted(self, bp: float, key: np.ndarray | None = None) -> "Curve":
        """Parallel shift of the continuously compounded zero curve (or a shift weighted by `key`)."""
        w = np.ones_like(self.times) if key is None else key
        return Curve(self.times, self.dfs * np.exp(-bp * 1e-4 * w * self.times))


def bootstrap_par(maturities, par_yields, freq: int = 2, last: float | None = None) -> Curve:
    m = np.asarray(maturities, float)
    y = np.asarray(par_yields, float)
    ok = np.isfinite(y)
    m, y = m[ok], y[ok]
    last = last or float(m.max())
    grid = np.arange(1, int(round(last * freq)) + 1) / freq
    c = np.interp(grid, m, y)
    D = np.empty_like(grid)
    for n in range(len(grid)):
        D[n] = (1 - c[n] / freq * D[:n].sum()) / (1 + c[n] / freq)
    return Curve(grid, D)


def bond_price(curve: Curve, coupon: float, maturity: float, freq: int = 2) -> float:
    t = np.arange(1, int(round(maturity * freq)) + 1) / freq
    return float(100 * (coupon / freq * curve.df(t).sum() + curve.df(maturity)[0]))


def swap(curve: Curve, maturity: float, fixed: float | None = None, freq: int = 2, notional: float = 1.0) -> dict[str, float]:
    """Receive-fixed swap. Returns the par rate, annuity, PV (to the fixed receiver) and DV01."""
    t = np.arange(1, int(round(maturity * freq)) + 1) / freq

    def value(c: Curve, k: float) -> tuple[float, float, float]:
        A = float((c.df(t) / freq).sum())
        flt = 1 - float(c.df(maturity)[0])
        return k * A - flt, A, flt / A

    _, A, par = value(curve, 0.0)
    k = par if fixed is None else fixed
    pv, _, _ = value(curve, k)
    up, _, _ = value(curve.shifted(1.0), k)
    dn, _, _ = value(curve.shifted(-1.0), k)
    return {"par_rate": par, "annuity": A, "pv": notional * pv, "dv01": notional * (dn - up) / 2}
