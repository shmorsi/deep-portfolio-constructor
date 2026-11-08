"""Small statistics helpers used across stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def newey_west_mean(x: pd.Series, lags: int) -> tuple[float, float, int]:
    """Mean of ``x`` with a Newey-West (HAC) t-statistic.

    Monthly IC and return series are autocorrelated (overlapping information,
    persistent signals), which makes the plain t-stat too optimistic. HAC
    standard errors correct for that.
    Returns (mean, t_stat, n_obs).
    """
    y = pd.Series(x).dropna().astype(float)
    n = len(y)
    if n < 3:
        return float("nan"), float("nan"), n
    res = sm.OLS(y.values, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0]), n


def factor_regression(y: pd.Series, X: pd.DataFrame, lags: int) -> dict[str, float]:
    """OLS of ``y`` on factors with HAC errors. Returns alpha, its t-stat, betas, R^2."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df[X.columns])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {"alpha": float(res.params["const"]), "alpha_t": float(res.tvalues["const"]), "r2": float(res.rsquared)}
    out.update({f"beta_{c}": float(res.params[c]) for c in X.columns})
    out["n_obs"] = int(res.nobs)
    return out
