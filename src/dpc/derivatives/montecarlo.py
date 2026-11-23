"""Monte Carlo pricing with variance reduction: an arithmetic-average Asian call.

Paths follow geometric Brownian motion under the risk-neutral measure. Two reductions:
antithetic draws (each normal z paired with -z), and a control variate, the *geometric* Asian
call, whose price has a closed form (Kemna and Vorst 1990). The estimator is
    C_arith + b (C_geo_exact - C_geo_mc),   b = cov / var from the same paths,
and its standard error is reported, so the reduction is measured, not assumed.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def geometric_asian_call(S, K, T, r, sigma, n, q=0.0) -> float:
    """Closed form for a discretely monitored geometric-average call (n equally spaced fixings)."""
    t = (T / n) * np.arange(1, n + 1)
    mu = np.log(S) + (r - q - 0.5 * sigma ** 2) * t.mean()          # mean of the log of the geometric average
    var = sigma ** 2 * np.sum(np.minimum.outer(t, t)) / n ** 2       # its variance: sigma^2 sum_ij min(t_i, t_j) / n^2
    sd = np.sqrt(var)
    d1 = (mu - np.log(K) + var) / sd
    d2 = d1 - sd
    return float(np.exp(-r * T) * (np.exp(mu + 0.5 * var) * norm.cdf(d1) - K * norm.cdf(d2)))


def asian_call(S, K, T, r, sigma, n=12, paths=100_000, seed=0, q=0.0, antithetic=True, control=True) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dt = T / n
    z = rng.standard_normal((paths // 2 if antithetic else paths, n))
    if antithetic:
        z = np.vstack([z, -z])
    logs = np.log(S) + np.cumsum((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    prices = np.exp(logs)
    disc = np.exp(-r * T)
    arith = disc * np.maximum(prices.mean(axis=1) - K, 0.0)
    h = len(arith) // 2

    def pairs(x: np.ndarray) -> np.ndarray:   # with antithetics the pairs, not the draws, are independent
        return 0.5 * (x[:h] + x[h:]) if antithetic else x

    arith_p = pairs(arith)
    est, se = float(arith_p.mean()), float(arith_p.std(ddof=1) / np.sqrt(len(arith_p)))
    out = {"plain": est, "plain_se": se}
    if control:
        geo = disc * np.maximum(np.exp(logs.mean(axis=1)) - K, 0.0)
        geo_p = pairs(geo)
        exact = geometric_asian_call(S, K, T, r, sigma, n, q)
        c = np.cov(arith_p, geo_p)
        b = c[0, 1] / c[1, 1]
        adj = arith_p - b * (geo_p - exact)
        out.update({"control": float(adj.mean()), "control_se": float(adj.std(ddof=1) / np.sqrt(len(adj))),
                    "geometric_exact": exact, "beta": float(b)})
    return out
