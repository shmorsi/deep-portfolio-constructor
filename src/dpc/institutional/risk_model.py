"""A fundamental factor risk model and the decomposition of ex-ante risk.

Each asset's excess return is regressed on the factors over a trailing window:

    r_i - rf = a_i + B_i f + e_i

so the covariance of returns is the factor part plus the specific part,

    Sigma = B F B' + D,      F = cov(f),  D = diag(var(e_i)).

That is the structure of a commercial (Barra-style) model, with three public factors
(market, size, value) instead of dozens of proprietary ones. Its use here is the one a
portfolio manager has for it: how much active risk a portfolio takes against its benchmark,
where that risk comes from, and whether the model's forecasts of it were right (the bias
statistic, in `active`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorRiskModel:
    names: list[str]
    factor_names: list[str]
    loadings: np.ndarray        # B, n_assets x n_factors
    factor_cov: np.ndarray      # F, n_factors x n_factors (per period)
    specific_var: np.ndarray    # diag(D), n_assets (per period)

    @property
    def cov(self) -> np.ndarray:
        B = self.loadings
        return B @ self.factor_cov @ B.T + np.diag(self.specific_var)

    @classmethod
    def fit(cls, excess: np.ndarray, factors: np.ndarray, names: list[str], factor_names: list[str],
            halflife: float | None = None) -> "FactorRiskModel":
        """Time-series OLS of each asset on the factors; optional exponential weights (half-life in periods)."""
        excess = np.asarray(excess, float)
        factors = np.asarray(factors, float)
        T = excess.shape[0]
        if factors.shape[0] != T:
            raise ValueError("returns and factors must cover the same periods")
        if halflife:
            w = 0.5 ** (np.arange(T)[::-1] / halflife)
        else:
            w = np.ones(T)
        w = w / w.sum()
        X = np.column_stack([np.ones(T), factors])
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(X * sw, excess * sw, rcond=None)
        B = coef[1:].T
        resid = excess - X @ coef
        k = X.shape[1]
        dof = 1.0 - k / T                             # small-sample correction of the weighted variance
        spec = (w[:, None] * resid ** 2).sum(axis=0) / dof
        fm = (w[:, None] * factors).sum(axis=0)
        fc = factors - fm
        F = (fc * w[:, None]).T @ fc / (1.0 - (w ** 2).sum())
        return cls(list(names), list(factor_names), B, F, spec)


def risk_decomposition(model: FactorRiskModel, weights: np.ndarray, benchmark: np.ndarray | None = None) -> dict:
    """Ex-ante risk of `weights` (or of `weights - benchmark`: active risk) and where it comes from.

    Marginal contribution of asset i: (Sigma a)_i / sigma. Contribution: a_i times that; the
    contributions add up to sigma exactly (Euler). The factor and specific shares of the variance
    add up to one.
    """
    w = np.asarray(weights, float)
    a = w - (np.asarray(benchmark, float) if benchmark is not None else 0.0)
    S = model.cov
    var = float(a @ S @ a)
    sigma = float(np.sqrt(max(var, 0.0)))
    mctr = S @ a / sigma if sigma > 0 else np.zeros_like(a)
    contrib = a * mctr
    x = model.loadings.T @ a                       # factor exposures of the (active) position
    factor_var = float(x @ model.factor_cov @ x)
    specific_var = float((a ** 2) @ model.specific_var)
    by_factor = x * (model.factor_cov @ x)          # each factor's share of the factor variance (adds to factor_var)
    return {
        "sigma": sigma,
        "mctr": mctr,
        "contribution": contrib,
        "pct_contribution": contrib / sigma if sigma > 0 else np.zeros_like(a),
        "exposures": dict(zip(model.factor_names, x.tolist())),
        "factor_share": factor_var / var if var > 0 else 0.0,
        "specific_share": specific_var / var if var > 0 else 0.0,
        "factor_variance_by_factor": dict(zip(model.factor_names, (by_factor / var if var > 0 else by_factor).tolist())),
    }
