"""Active management against a benchmark, as an asset manager runs it.

Every month-end, with only the data known then:

1. **Risk model.** A Fama-French factor model fitted on the last 60 months (`risk_model`).
2. **Expected returns.** Black-Litterman: the returns the cap-weighted benchmark implies, moved
   toward one relative view, "the six industries with the best 12-month momentum (skipping the
   last month) beat the six with the worst". The size of the view is the average of that spread
   over the previous ten years (floored at zero), so the view is earned from the past only.
3. **Portfolio.** Maximise expected return subject to a tracking-error budget against the
   benchmark, long-only, fully invested, at most 10 points over the benchmark weight in any industry.

Then the portfolio is held for a month and pays 10 bps on what it traded. What is reported is
what a client or an investment committee asks: active return, realised tracking error,
information ratio, and whether the risk model's forecast of tracking error was right (the bias
statistic: the standard deviation of active return divided by its forecast; 1 is perfect,
the usual acceptance band for 60+ months is about 1 +/- sqrt(2/T)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .black_litterman import he_litterman_omega, implied_returns, posterior, relative_view
from .risk_model import FactorRiskModel, risk_decomposition

FACTORS = ["Mkt-RF", "SMB", "HML"]

# An approximate sector grouping of Kenneth French's 30 industries (for attribution).
SECTORS = {
    "Food": "Staples", "Beer": "Staples", "Smoke": "Staples", "Hshld": "Staples",
    "Games": "Discretionary", "Books": "Discretionary", "Clths": "Discretionary", "Autos": "Discretionary",
    "Rtail": "Discretionary", "Meals": "Discretionary", "Txtls": "Discretionary",
    "Hlth": "Health",
    "Chems": "Materials", "Steel": "Materials", "Mines": "Materials", "Paper": "Materials", "FabPr": "Materials",
    "Cnstr": "Industrials", "ElcEq": "Industrials", "Carry": "Industrials", "Trans": "Industrials", "Whlsl": "Industrials",
    "Coal": "Energy", "Oil": "Energy",
    "Util": "Utilities", "Telcm": "Telecom",
    "BusEq": "Technology", "Servs": "Technology",
    "Fin": "Financials", "Other": "Other",
}


@dataclass(frozen=True)
class ActiveConfig:
    te_budget: float = 0.02          # annual ex-ante tracking error
    window: int = 60                 # months of history for the risk model
    max_active: float = 0.10         # at most this much over the benchmark weight
    tau: float = 0.05
    delta: float = 2.5               # market risk aversion (the conventional value)
    n_view: int = 6                  # winners and losers in the momentum view
    view_lookback: int = 120         # months used to size the view
    cost_bps: float = 10.0
    start: str = "1970-01"
    risk: str = "factor"             # "factor" (B F B' + D), "sample", or "shrunk" (Ledoit-Wolf)


def momentum(returns: np.ndarray, t: int) -> np.ndarray:
    """12-1 momentum at the end of month t: months t-11..t-1, skipping month t."""
    return np.prod(1.0 + returns[t - 11:t], axis=0) - 1.0


def momentum_spread(returns: np.ndarray, t: int, n: int) -> float:
    """Next-month return of the top-n minus the bottom-n by momentum, formed at t-1, earned in t."""
    m = momentum(returns, t - 1)
    order = np.argsort(m)
    return float(returns[t, order[-n:]].mean() - returns[t, order[:n]].mean())


def _covariance(model: FactorRiskModel, excess: np.ndarray, kind: str) -> np.ndarray:
    if kind == "factor":
        return model.cov
    if kind == "sample":
        return np.cov(excess, rowvar=False)
    if kind == "shrunk":
        from sklearn.covariance import LedoitWolf

        return LedoitWolf().fit(excess).covariance_
    raise ValueError(f"unknown risk model {kind!r}")


def optimise(mu: np.ndarray, cov: np.ndarray, bench: np.ndarray, te_monthly: float, max_active: float) -> np.ndarray:
    """max mu'w  s.t.  ||L'(w - b)|| <= TE, sum w = 1, 0 <= w <= b + max_active."""
    import cvxpy as cp

    n = len(mu)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0, w <= bench + max_active, cp.norm(L.T @ (w - bench)) <= te_monthly]
    cp.Problem(cp.Maximize(mu @ w), cons).solve(solver=cp.CLARABEL)
    if w.value is None:
        return bench.copy()
    out = np.clip(np.asarray(w.value).ravel(), 0.0, None)
    return out / out.sum()


@dataclass
class ActiveResult:
    dates: pd.DatetimeIndex
    weights: np.ndarray           # held over each month
    bench: np.ndarray
    returns: np.ndarray           # asset returns over each month
    port: np.ndarray              # portfolio return, net of costs
    bench_ret: np.ndarray
    ex_ante_te: np.ndarray        # monthly forecast made at the start of each month
    turnover: np.ndarray
    last_model: FactorRiskModel | None = None
    meta: dict = field(default_factory=dict)

    @property
    def active(self) -> np.ndarray:
        return self.port - self.bench_ret


def run(panel, cfg: ActiveConfig = ActiveConfig()) -> ActiveResult:
    R = panel.returns
    names = panel.names
    bench_all = panel.market_weights()
    fac = np.column_stack([panel.mkt_rf, panel.smb, panel.hml])
    excess = R - panel.rf[:, None]
    first = max(panel.index_of(cfg.start) - 1, cfg.window, cfg.view_lookback + 13)
    te_m = cfg.te_budget / np.sqrt(12.0)
    cost = cfg.cost_bps / 1e4

    W, B, P_, BR, TE, TO, D = [], [], [], [], [], [], []
    w_prev = bench_all[first - 1]
    model = None
    for t in range(first, panel.n_months - 1):            # decide at the end of t, hold over t+1
        win = slice(t - cfg.window + 1, t + 1)
        model = FactorRiskModel.fit(excess[win], fac[win], names, FACTORS)
        S = _covariance(model, excess[win], cfg.risk)
        b = bench_all[t]
        pi = implied_returns(S, b, cfg.delta)
        mom = momentum(R, t)
        order = np.argsort(mom)
        winners = [names[i] for i in order[-cfg.n_view:]]
        losers = [names[i] for i in order[:cfg.n_view]]
        spreads = [momentum_spread(R, s, cfg.n_view) for s in range(t - cfg.view_lookback + 1, t + 1)]
        q = max(float(np.mean(spreads)), 0.0)
        P = relative_view(names, winners, losers)[None, :]
        mu, _ = posterior(pi, S, P, np.array([q]), he_litterman_omega(P, S, cfg.tau), cfg.tau)
        w = optimise(mu, S, b, te_m, cfg.max_active)
        a = w - b
        TE.append(float(np.sqrt(a @ S @ a)))
        # what the old holdings drifted to over month t, then what was traded
        drift = w_prev * (1.0 + R[t])
        drift = drift / drift.sum()
        turn = float(np.abs(w - drift).sum())
        r_next = R[t + 1]
        P_.append(float(w @ r_next) - cost * turn)
        BR.append(float(b @ r_next))
        W.append(w)
        B.append(b)
        TO.append(turn)
        D.append(panel.dates[t + 1])
        w_prev = w
    return ActiveResult(pd.DatetimeIndex(D), np.array(W), np.array(B), R[[panel.dates.get_loc(d) for d in D]],
                        np.array(P_), np.array(BR), np.array(TE), np.array(TO), model,
                        {"config": cfg.__dict__, "first_decision": f"{panel.dates[first]:%Y-%m}"})


def summarise(res: ActiveResult, nw_lags: int = 12) -> dict:
    from dpc.stats import newey_west_mean

    a = res.active
    T = len(a)
    te_real = float(np.std(a, ddof=1) * np.sqrt(12))
    mean_a, t_a, _ = newey_west_mean(pd.Series(a), nw_lags)
    z = a / res.ex_ante_te
    bias = float(np.std(z, ddof=1))
    band = float(np.sqrt(2.0 / T))
    cum = lambda r: float(np.prod(1 + r) - 1)
    return {
        "start": f"{res.dates[0]:%Y-%m}", "end": f"{res.dates[-1]:%Y-%m}", "months": T,
        "portfolio_cagr": float(np.prod(1 + res.port) ** (12 / T) - 1),
        "benchmark_cagr": float(np.prod(1 + res.bench_ret) ** (12 / T) - 1),
        "portfolio_growth": cum(res.port) + 1, "benchmark_growth": cum(res.bench_ret) + 1,
        "active_return_ann": float(mean_a * 12), "active_t_nw": float(t_a),
        "tracking_error_realised": te_real,
        "tracking_error_ex_ante_mean": float(np.mean(res.ex_ante_te) * np.sqrt(12)),
        "information_ratio": float(mean_a * 12 / te_real) if te_real > 0 else float("nan"),
        "bias_statistic": bias, "bias_band": [1 - band, 1 + band],
        "bias_verdict": "forecasts right-sized" if abs(bias - 1) <= band else ("risk under-forecast" if bias > 1 else "risk over-forecast"),
        "turnover_ann": float(np.mean(res.turnover) * 12),
        "hit_rate": float(np.mean(a > 0)),
    }


def rolling_bias(res: ActiveResult, window: int = 36) -> pd.Series:
    z = pd.Series(res.active / res.ex_ante_te, index=res.dates)
    return z.rolling(window).std()


def final_risk_report(res: ActiveResult) -> dict:
    d = risk_decomposition(res.last_model, res.weights[-1], res.bench[-1])
    names = res.last_model.names
    top = np.argsort(-np.abs(d["contribution"]))[:8]
    return {
        "as_of": f"{res.dates[-1]:%Y-%m}",
        "active_risk_ann": d["sigma"] * np.sqrt(12),
        "factor_share": d["factor_share"], "specific_share": d["specific_share"],
        "active_exposures": d["exposures"],
        "factor_variance_by_factor": d["factor_variance_by_factor"],
        "top_contributors": [{"industry": names[i], "active_weight": float(res.weights[-1][i] - res.bench[-1][i]),
                              "pct_of_active_risk": float(d["pct_contribution"][i])} for i in top],
    }
