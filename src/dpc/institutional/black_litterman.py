"""Black-Litterman: start from the returns the benchmark implies, then move toward views.

Reverse optimisation: if the benchmark b is what a mean-variance investor with risk aversion
delta would hold, the expected excess returns it implies are

    pi = delta Sigma b.

Views are P mu = Q + eps, eps ~ N(0, Omega). The posterior mean (Black and Litterman 1992,
in the form of He and Litterman 1999) is

    mu = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P' Omega^-1 Q],

and with Omega = diag(P tau Sigma P') ("He-Litterman"), each view is trusted as much as the
prior is. With no views the posterior is the prior and the optimal portfolio is the benchmark:
the property that makes the method usable by an institution, and the first thing the tests check.
"""

from __future__ import annotations

import numpy as np


def implied_returns(cov: np.ndarray, benchmark: np.ndarray, delta: float) -> np.ndarray:
    return delta * np.asarray(cov, float) @ np.asarray(benchmark, float)


def risk_aversion(excess_return: float, variance: float) -> float:
    """delta = E[r_m - rf] / var(r_m): the market's price of risk."""
    return excess_return / variance


def he_litterman_omega(P: np.ndarray, cov: np.ndarray, tau: float, confidence: float = 1.0) -> np.ndarray:
    """diag(P tau Sigma P') / confidence. confidence 1 trusts a view as much as the prior."""
    P = np.atleast_2d(P)
    return np.diag(np.diag(P @ (tau * cov) @ P.T)) / confidence


def posterior(pi: np.ndarray, cov: np.ndarray, P: np.ndarray, Q: np.ndarray, omega: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and covariance of expected returns; the predictive covariance is Sigma + M."""
    P = np.atleast_2d(np.asarray(P, float))
    Q = np.atleast_1d(np.asarray(Q, float))
    ts_inv = np.linalg.inv(tau * cov)
    om_inv = np.linalg.inv(omega)
    M = np.linalg.inv(ts_inv + P.T @ om_inv @ P)
    mu = M @ (ts_inv @ pi + P.T @ om_inv @ Q)
    return mu, M


def unconstrained_weights(mu: np.ndarray, cov: np.ndarray, delta: float) -> np.ndarray:
    """w = (delta Sigma)^-1 mu: the mean-variance portfolio for these returns (may short, may lever)."""
    return np.linalg.solve(delta * cov, mu)


def relative_view(names: list[str], winners: list[str], losers: list[str]) -> np.ndarray:
    """One row of P: equal-weighted winners minus equal-weighted losers."""
    p = np.zeros(len(names))
    for n in winners:
        p[names.index(n)] += 1.0 / len(winners)
    for n in losers:
        p[names.index(n)] -= 1.0 / len(losers)
    return p
