"""The classical long-only strategies every result is measured against.

* Market        : hold every industry at its market-cap weight. This is the
                  index fund, the strategy most professionals fail to beat.
* Equal weight  : 1/30 in each industry, rebalanced every January.
* Min variance  : the lowest-risk long-only mix (Ledoit-Wolf covariance of the
                  last 60 months, at most 20% per industry), every January.
* Risk parity   : equal risk from every industry, every January.
* Growth optimiser (classical): the same goal and mandate as the network,
                  maximise expected log growth, mu'w - w'Sigma w / 2, inside the
                  same bounds around market weights, with plug-in estimates
                  (shrunk long-run means, Ledoit-Wolf covariance), every January.
* Momentum tilt : the textbook rule; overweight the industries with the best
                  12-1 month momentum and underweight the worst, inside the same
                  bounds, rebalanced monthly.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from sklearn.covariance import LedoitWolf

from .data import IndustryPanel
from .engine import Decide

COV_WINDOW = 60


def _is_rebalance(panel: IndustryPanel, t: int, month: int) -> bool:
    """A January rebalance is decided at the end of December."""
    return panel.dates[t].month == (month - 2) % 12 + 1


def _cov(panel: IndustryPanel, t: int) -> np.ndarray:
    window = panel.returns[t - COV_WINDOW + 1:t + 1]
    return LedoitWolf().fit(window).covariance_


def tilt_bounds(w_mkt: np.ndarray, max_tilt: float) -> tuple[np.ndarray, np.ndarray]:
    """Each industry stays between 1/max_tilt and max_tilt times its market weight."""
    return w_mkt / max_tilt, np.minimum(w_mkt * max_tilt, 1.0)


def market(panel: IndustryPanel) -> Decide:
    w = panel.market_weights()
    return lambda t, w_drift: w[t]


def equal_weight(panel: IndustryPanel, month: int) -> Decide:
    n = panel.n_assets
    return lambda t, w_drift: np.full(n, 1.0 / n) if _is_rebalance(panel, t, month) or w_drift.sum() == 0 else w_drift


def _solve(problem: cp.Problem) -> None:
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"baseline optimisation failed: {problem.status}")


def min_variance(panel: IndustryPanel, month: int, max_weight: float) -> Decide:
    n = panel.n_assets

    def decide(t: int, w_drift: np.ndarray) -> np.ndarray:
        if not (_is_rebalance(panel, t, month) or w_drift.sum() == 0):
            return w_drift
        w = cp.Variable(n)
        _solve(cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(_cov(panel, t)))), [cp.sum(w) == 1, w >= 0, w <= max_weight]))
        return np.clip(w.value, 0, None) / np.clip(w.value, 0, None).sum()
    return decide


def risk_parity(panel: IndustryPanel, month: int) -> Decide:
    n = panel.n_assets

    def decide(t: int, w_drift: np.ndarray) -> np.ndarray:
        if not (_is_rebalance(panel, t, month) or w_drift.sum() == 0):
            return w_drift
        y = cp.Variable(n)
        _solve(cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(_cov(panel, t))) - cp.sum(cp.log(y)) / n)))
        return y.value / y.value.sum()
    return decide


def growth_optimizer(panel: IndustryPanel, month: int, max_tilt: float, shrink: float) -> Decide:
    """Classical Kelly-style allocation: maximise mu'w - w'Sigma w / 2 in the tilt mandate."""
    n = panel.n_assets
    w_mkt = panel.market_weights()

    def decide(t: int, w_drift: np.ndarray) -> np.ndarray:
        if not (_is_rebalance(panel, t, month) or w_drift.sum() == 0):
            return w_drift
        m = panel.returns[:t + 1].mean(axis=0)                 # long-run mean of each industry, to date
        mu = (1 - shrink) * m + shrink * m.mean()              # shrunk toward the cross-sectional average
        lo, hi = tilt_bounds(w_mkt[t], max_tilt)
        w = cp.Variable(n)
        obj = cp.Maximize(mu @ w - 0.5 * cp.quad_form(w, cp.psd_wrap(_cov(panel, t))))
        _solve(cp.Problem(obj, [cp.sum(w) == 1, w >= lo, w <= hi]))
        return np.clip(w.value, 0, None) / np.clip(w.value, 0, None).sum()
    return decide


def momentum_tilt(panel: IndustryPanel, max_tilt: float) -> Decide:
    """Market weights times max_tilt ** s, s from -1 (worst 12-1 momentum) to +1 (best)."""
    w_mkt = panel.market_weights()
    logr = np.log1p(panel.returns)

    def decide(t: int, w_drift: np.ndarray) -> np.ndarray:
        mom = logr[t - 11:t].sum(axis=0)                        # months t-11 .. t-1: skip the latest month
        ranks = mom.argsort().argsort() / (len(mom) - 1)
        w = w_mkt[t] * max_tilt ** (2 * ranks - 1)
        return w / w.sum()
    return decide


def all_baselines(panel: IndustryPanel, month: int, max_weight: float, max_tilt: float, shrink: float) -> dict[str, Decide]:
    return {
        "market": market(panel),
        "equal_weight": equal_weight(panel, month),
        "min_variance": min_variance(panel, month, max_weight),
        "risk_parity": risk_parity(panel, month),
        "growth_classical": growth_optimizer(panel, month, max_tilt, shrink),
        "momentum_tilt": momentum_tilt(panel, max_tilt),
    }
