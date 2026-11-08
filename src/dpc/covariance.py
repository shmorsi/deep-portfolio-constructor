"""Classical covariance estimators (all return *monthly* covariance matrices).

* Sample covariance: unbiased but noisy. With N=100 stocks and T=252 days,
  the matrix has 5,050 free parameters estimated from 25,200 numbers; its
  smallest eigenvalues are biased toward zero, so an optimiser that inverts it
  piles into spuriously "riskless" combinations.
* Ledoit-Wolf: shrinks the sample matrix toward a structured target
  (scaled identity), with the shrinkage intensity chosen analytically to
  minimise expected error. Simple, fast, and hard to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

DAYS_PER_MONTH = 21  # horizon scaling: daily -> monthly covariance


def _window(returns: pd.DataFrame) -> np.ndarray:
    x = returns.to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)  # rare gaps (<= max_missing_in_window) -> 0
    return x - x.mean(axis=0, keepdims=True)


def sample_cov(returns: pd.DataFrame, horizon_days: int = DAYS_PER_MONTH) -> np.ndarray:
    x = _window(returns)
    return np.cov(x, rowvar=False, ddof=1) * horizon_days


def ledoit_wolf_cov(returns: pd.DataFrame, horizon_days: int = DAYS_PER_MONTH) -> np.ndarray:
    x = _window(returns)
    return LedoitWolf(assume_centered=True).fit(x).covariance_ * horizon_days


def cov_to_factor_form(cov: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Return L with cov = L L' (Cholesky), the form the optimiser uses for risk."""
    n = cov.shape[0]
    return np.linalg.cholesky(cov + jitter * np.eye(n))
