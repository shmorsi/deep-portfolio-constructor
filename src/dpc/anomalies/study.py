"""Three questions of fifteen anomalies, all answered with data known at the time.

1. **Do they survive?** Full-sample means with Newey-West t-statistics, against the usual 1.96
   and against Harvey, Liu and Zhu's 3.0 hurdle for a field that has tested hundreds of signals,
   with a Benjamini-Hochberg false-discovery check.
2. **Did publication kill them?** McLean and Pontiff (2016): the average return before and after
   each anomaly's paper appeared, and a pooled regression of returns on a post-publication dummy
   with anomaly fixed effects and errors clustered by month.
3. **Can they be combined out of sample?** Each month from 1973, using only the past:
   equal weight; factor momentum (hold the anomalies whose last twelve months were positive,
   Ehsani and Linnainmaa 2022); shrinkage mean-variance (Ledoit-Wolf covariance, means shrunk
   halfway to their cross-sectional average); and gradient boosting that forecasts each
   anomaly's next month from its own recent history, refitted every January. Each is compared
   with equal weight by the difference in Sharpe ratio and a block bootstrap.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import ZOO

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
START = "1963-07"


def _text(file: str, cache: Path, refresh: bool = False) -> str:
    path = cache / file.replace(".zip", ".txt")
    if refresh or not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        r = requests.get(BASE + file, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            path.write_text(z.read(z.namelist()[0]).decode("latin-1"), encoding="latin-1")
    return path.read_text(encoding="latin-1")


def spreads(cache: Path, refresh: bool = False) -> pd.DataFrame:
    """Monthly long-minus-short decile returns (decimals), one column per anomaly, from 1963-07."""
    from ..wealth.data import parse_sections

    out = {}
    for a in ZOO:
        sec = parse_sections(_text(a.file, cache, refresh))
        key = next(k for k, v in sec.items() if "Value Weight" in k and "Monthly" in k and len(str(v.index[0])) == 6)
        df = sec[key]
        lo = "Lo PRIOR" if "Lo PRIOR" in df.columns else "Lo 10"
        hi = "Hi PRIOR" if "Hi PRIOR" in df.columns else "Hi 10"
        s = (df[hi] - df[lo]) / 100 if a.long == "hi" else (df[lo] - df[hi]) / 100
        s.index = pd.to_datetime(s.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
        out[a.key] = s
    return pd.DataFrame(out).loc[START:].dropna()


# ---------------------------------------------------------------- statistics
def nw_t(x: np.ndarray, lags: int = 12) -> tuple[float, float]:
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean()), float(x.mean() / np.sqrt(s / n))


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


def survival(R: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import norm

    rows = []
    for a in ZOO:
        m, t = nw_t(R[a.key].to_numpy())
        rows.append({"key": a.key, "name": a.name, "mean_ann": 12 * m, "t": t, "p": 2 * norm.sf(abs(t)),
                     "sharpe": float(np.sqrt(12) * R[a.key].mean() / R[a.key].std(ddof=1))})
    df = pd.DataFrame(rows).set_index("key")
    df["bh_5pct"] = benjamini_hochberg(df["p"].to_numpy())
    return df


def decay(R: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import statsmodels.api as sm

    rows, stacked = [], []
    for a in ZOO:
        s = R[a.key]
        pre, post = s[s.index.year <= a.year], s[s.index.year > a.year]
        rows.append({"key": a.key, "paper": f"{a.paper} ({a.year})", "pre_months": len(pre), "post_months": len(post),
                     "pre_ann": 12 * pre.mean() if len(pre) else np.nan, "post_ann": 12 * post.mean()})
        stacked.append(pd.DataFrame({"r": s.values, "post": (s.index.year > a.year).astype(float), "key": a.key, "month": s.index}))
    df = pd.DataFrame(rows).set_index("key")
    df["decay"] = 1 - df["post_ann"] / df["pre_ann"]
    st = pd.concat(stacked, ignore_index=True)
    X = pd.get_dummies(st["key"], dtype=float)
    X["post"] = st["post"]
    fit = sm.OLS(st["r"] * 12, X).fit(cov_type="cluster", cov_kwds={"groups": st["month"].dt.strftime("%Y%m").astype(int)})
    # the average in-sample return of anomalies with a pre-publication sample, for scale
    with_pre = df[df["pre_months"] >= 60]
    pooled = {"post_coef_ann": float(fit.params["post"]), "post_t": float(fit.tvalues["post"]),
              "mean_pre_ann": float(with_pre["pre_ann"].mean()), "mean_post_ann": float(with_pre["post_ann"].mean()),
              "anomalies_with_pre_sample": int(len(with_pre))}
    pooled["implied_decay"] = -pooled["post_coef_ann"] / pooled["mean_pre_ann"]
    return df, pooled


# ---------------------------------------------------------------- combining
def _features(R: pd.DataFrame | np.ndarray, t: int) -> np.ndarray:
    """Per anomaly at the end of month t: trailing 1, 3, 12 and 60-month mean returns and 12-month volatility."""
    x = R.to_numpy() if isinstance(R, pd.DataFrame) else R
    f = [x[t], x[t - 2:t + 1].mean(0), x[t - 11:t + 1].mean(0), x[t - 59:t + 1].mean(0), x[t - 11:t + 1].std(0, ddof=1)]
    return np.column_stack(f)


def combine(R: pd.DataFrame, start: str = "1973-07", window: int = 120, seed: int = 0) -> pd.DataFrame:
    """Out-of-sample monthly returns of four ways to hold the zoo (weights sum to one in absolute value)."""
    from sklearn.covariance import LedoitWolf
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = R.to_numpy()
    T, N = x.shape
    first = R.index.get_indexer([pd.Timestamp(start) + pd.offsets.MonthEnd(0)])[0]
    out = {k: [] for k in ("equal_weight", "factor_momentum", "shrinkage_mv", "boosting")}
    dates = []
    model, fitted_year = None, None
    for t in range(first - 1, T - 1):                     # decide at the end of t, earn month t+1
        nxt = x[t + 1]
        out["equal_weight"].append(nxt.mean())
        past12 = x[t - 11:t + 1].sum(0)
        w = (past12 > 0).astype(float)
        out["factor_momentum"].append(nxt @ w / w.sum() if w.sum() else 0.0)
        win = x[t - window + 1:t + 1]
        S = LedoitWolf().fit(win).covariance_
        mu = win.mean(0)
        mu = 0.5 * mu + 0.5 * mu.mean()
        w = np.linalg.solve(S, mu)
        out["shrinkage_mv"].append(nxt @ w / np.abs(w).sum())
        year = R.index[t + 1].year
        if model is None or year != fitted_year:          # refit every January on everything known
            Xs, ys = [], []
            for s in range(60, t):                        # features at s, target at s+1 (known by t)
                Xs.append(_features(x, s))
                ys.append(x[s + 1])
            model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=3, min_samples_leaf=50,
                                                  random_state=seed).fit(np.vstack(Xs), np.concatenate(ys))
            fitted_year = year
        pred = model.predict(_features(x, t))
        w = np.clip(pred, 0, None)
        out["boosting"].append(nxt @ w / w.sum() if w.sum() > 0 else nxt.mean())
        dates.append(R.index[t + 1])
    return pd.DataFrame(out, index=pd.DatetimeIndex(dates))


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    return float(np.sqrt(12) * r.mean() / r.std(ddof=1))


def sharpe_diff_bootstrap(a: np.ndarray, b: np.ndarray, block: int = 12, draws: int = 2000, seed: int = 0) -> dict:
    """Moving-block bootstrap of Sharpe(a) - Sharpe(b), months resampled together for both."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = len(a)
    nb = int(np.ceil(T / block))
    d = np.empty(draws)
    for k in range(draws):
        starts = rng.integers(0, T - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:T]
        d[k] = sharpe(a[idx]) - sharpe(b[idx])
    lo, hi = np.quantile(d, [0.05, 0.95])
    return {"difference": sharpe(a) - sharpe(b), "lo90": float(lo), "hi90": float(hi),
            "verdict": "better" if lo > 0 else ("worse" if hi < 0 else "no detectable difference")}


def summarise(C: pd.DataFrame) -> dict:
    out = {}
    for k in C.columns:
        r = C[k].to_numpy()
        m, t = nw_t(r)
        dd = (1 + C[k]).cumprod()
        out[k] = {"mean_ann": 12 * m, "t": t, "vol_ann": float(np.sqrt(12) * r.std(ddof=1)), "sharpe": sharpe(r),
                  "max_drawdown": float((dd / dd.cummax() - 1).min()),
                  "sharpe_since_2000": sharpe(C.loc["2000":, k].to_numpy())}
        if k != "equal_weight":
            out[k]["vs_equal_weight"] = sharpe_diff_bootstrap(r, C["equal_weight"].to_numpy())
    return out
