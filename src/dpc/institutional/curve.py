"""Yield curves: Nelson-Siegel-Svensson fits, the curve's principal components, key-rate durations.

Nelson-Siegel-Svensson (Svensson 1994):

    y(m) = b0 + b1 g1(m/l1) + b2 h(m/l1) + b3 h(m/l2),
    g1(x) = (1 - e^-x)/x,   h(x) = g1(x) - e^-x

For fixed decay parameters (l1, l2) the curve is linear in the betas, so each date is fitted by
least squares over a grid of (l1, l2): no local minima, no starting values. The inputs are
Treasury par yields, used here as if they were zero-coupon yields (the gap is a few basis points
at these maturities): a stated approximation.

Principal components of monthly changes in the fitted curve give the classic result of
Litterman and Scheinkman (1991): level, slope and curvature explain nearly all of it.

Key-rate durations (Ho 1992): bump the zero curve by 1 bp in a triangle around each key maturity,
reprice, and read the sensitivity. They add up to the effective duration, which is what lets
an LDI manager hedge each part of the curve separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

L1_GRID = np.linspace(0.3, 6.0, 20)
L2_GRID = np.linspace(4.0, 30.0, 14)


def _basis(m: np.ndarray, l1: float, l2: float | None) -> np.ndarray:
    m = np.asarray(m, float)
    x1 = m / l1
    g1 = (1 - np.exp(-x1)) / x1
    cols = [np.ones_like(m), g1, g1 - np.exp(-x1)]
    if l2 is not None:
        x2 = m / l2
        cols.append((1 - np.exp(-x2)) / x2 - np.exp(-x2))
    return np.column_stack(cols)


@dataclass(frozen=True)
class NSS:
    beta: np.ndarray
    l1: float
    l2: float | None
    rmse: float

    def __call__(self, m) -> np.ndarray:
        return _basis(np.atleast_1d(m), self.l1, self.l2) @ self.beta

    @classmethod
    def fit(cls, maturities, yields, svensson: bool = True) -> "NSS":
        m = np.asarray(maturities, float)
        y = np.asarray(yields, float)
        ok = np.isfinite(y)
        m, y = m[ok], y[ok]
        best = None
        for l1 in L1_GRID:
            for l2 in (L2_GRID if svensson and len(m) >= 6 else [None]):
                if l2 is not None and l2 <= l1:
                    continue
                X = _basis(m, l1, l2)
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                rmse = float(np.sqrt(np.mean((X @ beta - y) ** 2)))
                if best is None or rmse < best.rmse:
                    best = cls(beta, float(l1), None if l2 is None else float(l2), rmse)
        return best


def fit_history(yields: pd.DataFrame, grid=np.arange(1, 31)) -> tuple[pd.DataFrame, pd.Series]:
    """Fit every row (date) of a maturity-column frame; return the fitted curve on `grid` and the fit RMSE."""
    curves, err = {}, {}
    mats = np.asarray(yields.columns, float)
    for d, row in yields.iterrows():
        v = row.to_numpy(float)
        if np.isfinite(v).sum() < 5:
            continue
        f = NSS.fit(mats, v)
        curves[d] = f(grid)
        err[d] = f.rmse
    return pd.DataFrame(curves, index=grid).T, pd.Series(err)


def pca(changes: pd.DataFrame, k: int = 3) -> dict:
    X = changes.dropna().to_numpy(float)
    X = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    share = s ** 2 / np.sum(s ** 2)
    load = vt[:k]
    load = load * np.sign(load.sum(axis=1, keepdims=True) + 1e-15)   # level loads positively
    return {"explained": share[:k].tolist(), "cumulative": float(share[:k].sum()), "loadings": load,
            "maturities": list(changes.columns), "months": int(X.shape[0])}


def pv(cash: np.ndarray, times: np.ndarray, zero) -> float:
    """Present value of cash flows at `times` (years) on a continuously compounded zero curve."""
    return float(np.sum(cash * np.exp(-zero(times) * times)))


KEYS = (2.0, 5.0, 10.0, 20.0, 30.0)


def _triangle(t: np.ndarray, keys, j: int) -> np.ndarray:
    k = keys[j]
    left = keys[j - 1] if j > 0 else None
    right = keys[j + 1] if j + 1 < len(keys) else None
    out = np.zeros_like(t, dtype=float)
    if left is None:
        out[t <= k] = 1.0
    else:
        m = (t > left) & (t <= k)
        out[m] = (t[m] - left) / (k - left)
    if right is None:
        out[t > k] = 1.0
    else:
        m = (t > k) & (t < right)
        out[m] = (right - t[m]) / (right - k)
    return out


def key_rate_durations(cash: np.ndarray, times: np.ndarray, zero, keys=KEYS, bp: float = 1e-4) -> dict:
    """KRD at each key maturity and their sum; the triangles partition unity, so the sum is the effective duration."""
    base = pv(cash, times, zero)
    krd = {}
    for j, k in enumerate(keys):
        tri = _triangle(times, keys, j)
        up = float(np.sum(cash * np.exp(-(zero(times) + bp * tri) * times)))
        dn = float(np.sum(cash * np.exp(-(zero(times) - bp * tri) * times)))
        krd[k] = -(up - dn) / (2 * bp * base)
    up = pv(cash, times, lambda t: zero(t) + bp)
    dn = pv(cash, times, lambda t: zero(t) - bp)
    eff = -(up - dn) / (2 * bp * base)
    return {"pv": base, "krd": krd, "sum_krd": float(sum(krd.values())), "effective_duration": float(eff)}
