"""What a volatility forecast is worth to a long-only investor: one rule, fed by each forecaster.

Minimum variance with at most 20% in an industry (the wealth study's cap). The covariance is the
forecaster's volatilities times one correlation matrix, the same for every forecaster, so the
portfolios differ only through the forecasts. The outcome is the portfolio's realized variance,
from its own daily returns.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from ..wealth.baselines import _cov
from ..wealth.data import IndustryPanel
from ..wealth.engine import Decide
from ..wealth.evaluate import block_bootstrap_mean, newey_west


def correlation(panel: IndustryPanel, t: int) -> np.ndarray:
    """Ledoit-Wolf on the last 60 monthly returns, as a correlation matrix."""
    cov = _cov(panel, t)
    s = np.sqrt(np.diag(cov))
    return cov / np.outer(s, s)


class MinVariance:
    """min w' S w, sum w = 1, 0 <= w <= cap. Built once; each month only the covariance factor changes."""

    def __init__(self, n: int, cap: float) -> None:
        self.w = cp.Variable(n)
        self.root = cp.Parameter((n, n))
        self.problem = cp.Problem(cp.Minimize(cp.sum_squares(self.root @ self.w)), [cp.sum(self.w) == 1, self.w >= 0, self.w <= cap])

    def solve(self, sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
        scale = sigma / sigma.mean()                      # the weights do not depend on the level of risk; this keeps the problem well scaled
        cov = corr * np.outer(scale, scale)
        self.root.value = np.linalg.cholesky(cov + 1e-10 * np.eye(len(sigma))).T
        self.problem.solve(solver=cp.CLARABEL)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"minimum-variance optimisation failed: {self.problem.status}")
        w = np.clip(self.w.value, 0.0, None)
        return w / w.sum()


def inverse_volatility(sigma: np.ndarray) -> np.ndarray:
    w = 1.0 / sigma
    return w / w.sum()


def min_variance_rule(panel: IndustryPanel, sigma: np.ndarray, first: int, cap: float) -> Decide:
    """sigma [M, N]: row k is the forecast made at decision index first + k."""
    solver = MinVariance(panel.n_assets, cap)
    return lambda t, w_drift: solver.solve(sigma[t - first], correlation(panel, t))


def inverse_volatility_rule(sigma: np.ndarray, first: int) -> Decide:
    return lambda t, w_drift: inverse_volatility(sigma[t - first])


def realized_variance(daily: pd.DataFrame, dates: pd.DatetimeIndex, held: np.ndarray) -> np.ndarray:
    """[M]: the sum of the portfolio's squared daily returns in each month, weights drifting from ``held[m]``."""
    by_month = {p: g.to_numpy(float) for p, g in daily.groupby(daily.index.to_period("M"))}
    out = np.empty(len(dates))
    for m, date in enumerate(dates):
        r = by_month[date.to_period("M")]
        if not np.isfinite(r).all():
            raise ValueError(f"missing daily returns in {date:%Y-%m}")
        value = held[m][None, :] * np.cumprod(1.0 + r, axis=0)        # [D, N] value of each holding
        total = np.concatenate([[1.0], value.sum(axis=1)])
        out[m] = float(((total[1:] / total[:-1] - 1.0) ** 2).sum())
    return out


def compare_variance(a: np.ndarray, b: np.ndarray, lags: int, block: int, draws: int, seed: int) -> dict[str, float | str]:
    """a against b by the mean difference in log realized variance, with the protocol's verdict."""
    d = np.log(a) - np.log(b)
    mean, t = newey_west(d, lags)
    lo, hi = block_bootstrap_mean(d, block, draws, seed)
    verdict = "lower risk" if t <= -2 and hi < 0 else "higher risk" if t >= 2 and lo > 0 else "no detectable difference"
    return {"diff_log_variance": mean, "t_nw": t, "ci_low": lo, "ci_high": hi, "volatility_change": float(np.exp(mean / 2.0) - 1.0), "verdict": verdict}
