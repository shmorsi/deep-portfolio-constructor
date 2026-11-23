"""Value-at-risk and its backtest, as a risk team validates its own model.

A 99% one-month VaR is the loss exceeded in 1% of months. Three estimators, each using only the
months before the forecast: historical simulation (the empirical quantile of the last 120 months),
a normal distribution with the window's mean and volatility, and a filtered (EWMA, RiskMetrics
lambda 0.97 on monthly data) normal. An *exception* is a month whose loss is bigger than the VaR
forecast for it.

Kupiec (1995), proportion of failures: with x exceptions in T months and target rate p,

    LR_pof = -2 ln[(1-p)^(T-x) p^x] + 2 ln[(1-x/T)^(T-x) (x/T)^x]  ~ chi2(1).

Christoffersen (1998), independence: exceptions should not cluster. With n_ij the count of
months in state j after a month in state i (1 = exception),

    LR_ind = -2 ln[(1-pi)^(n00+n10) pi^(n01+n11)] + 2 ln[(1-pi0)^n00 pi0^n01 (1-pi1)^n10 pi1^n11] ~ chi2(1).

Both are reported with their p-values; a model passes a test when p > 0.05. Expected shortfall
(the mean loss beyond VaR) is reported beside it, because Basel III moved to ES for that reason.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec(exceptions: np.ndarray, p: float) -> dict:
    e = np.asarray(exceptions, bool)
    T, x = len(e), int(e.sum())
    phat = x / T
    ll0 = _xlogy(T - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(T - x, 1 - phat) + _xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    return {"exceptions": x, "months": T, "expected": p * T, "rate": phat, "lr": lr, "p_value": float(chi2.sf(lr, 1))}


def christoffersen(exceptions: np.ndarray) -> dict:
    e = np.asarray(exceptions, int)
    a, b = e[:-1], e[1:]
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return {"n01": n01, "n11": n11, "lr": lr, "p_value": float(chi2.sf(lr, 1))}


def forecasts(returns: np.ndarray, level: float = 0.99, window: int = 120, lam: float = 0.97) -> dict[str, np.ndarray]:
    """VaR (a positive loss) for months window..T-1, each from the months before it."""
    r = np.asarray(returns, float)
    T = len(r)
    z = norm.ppf(level)
    out = {k: np.full(T, np.nan) for k in ("historical", "normal", "ewma")}
    es_hist = np.full(T, np.nan)
    var = np.var(r[:window], ddof=1)
    for t in range(window, T):
        w = r[t - window:t]
        q = np.quantile(w, 1 - level)
        out["historical"][t] = -q
        es_hist[t] = -w[w <= q].mean()
        out["normal"][t] = -(w.mean() - z * w.std(ddof=1))
        var = lam * var + (1 - lam) * r[t - 1] ** 2
        out["ewma"][t] = z * np.sqrt(var)
    out["es_historical"] = es_hist
    return out


def backtest(returns: np.ndarray, level: float = 0.99, window: int = 120) -> dict:
    r = np.asarray(returns, float)
    f = forecasts(r, level, window)
    res = {}
    for k in ("historical", "normal", "ewma"):
        v = f[k][window:]
        loss = -r[window:]
        exc = loss > v
        res[k] = {"mean_var": float(np.mean(v)), **kupiec(exc, 1 - level), "independence": christoffersen(exc),
                  "worst_excess": float(np.max(loss - v))}
    es = f["es_historical"][window:]
    loss = -r[window:]
    hit = loss > f["historical"][window:]
    res["historical"]["mean_es"] = float(np.mean(es))
    res["historical"]["mean_loss_when_exceeded"] = float(loss[hit].mean()) if hit.any() else None
    return res
