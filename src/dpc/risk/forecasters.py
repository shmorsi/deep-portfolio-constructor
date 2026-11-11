"""The four classical volatility forecasters, their quantiles, and the forecast file format.

Index convention, the wealth study's: ``f[t]`` is the forecast made at the end of month ``t``
for month ``t + 1``, from data up to and including month ``t`` only (tests perturb the future
and check that nothing earlier moves).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import RiskPanel

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN = LEVELS.index(0.5)
MEAN_WINDOW = 60       # months in `mean60`
HAR_LAGS = (1, 3, 12)  # this month, the last quarter, the last year
HAR_MIN_OBS = 36       # regressions start once an industry has this many earlier months
ERROR_WINDOW = 120     # a classical forecaster's quantiles come from its last 120 errors
CLASSICAL = ("last", "mean60", "ewma", "har")


def last_value(rv: np.ndarray) -> np.ndarray:
    return rv.copy()


def trailing_mean(rv: np.ndarray, window: int = MEAN_WINDOW) -> np.ndarray:
    """Mean of the last ``window`` months, missing unless all of them are there."""
    out = np.full_like(rv, np.nan)
    for t in range(window - 1, rv.shape[0]):
        block = rv[t - window + 1:t + 1]
        out[t] = np.where(np.isfinite(block).all(axis=0), block.mean(axis=0), np.nan)
    return out


def har_features(log_rv: np.ndarray) -> np.ndarray:
    """[T, N, 4]: 1, this month's log volatility, the mean of the last 3 and of the last 12 (nan until all exist)."""
    t_n = log_rv.shape
    x = np.full((*t_n, 1 + len(HAR_LAGS)), np.nan)
    x[..., 0] = 1.0
    for k, lag in enumerate(HAR_LAGS):
        for t in range(lag - 1, t_n[0]):
            x[t, :, 1 + k] = log_rv[t - lag + 1:t + 1].mean(axis=0)   # nan if any month in the window is missing
    return x


def har(rv: np.ndarray) -> np.ndarray:
    """One regression per industry on all earlier months, refit every month; the forecast is exp(fitted)."""
    log_rv = np.log(rv)
    x = har_features(log_rv)
    t_max, n = rv.shape
    out = np.full_like(rv, np.nan)
    for i in range(n):
        a, b, count = np.zeros((x.shape[2], x.shape[2])), np.zeros(x.shape[2]), 0
        for t in range(t_max):
            if t >= 1 and np.isfinite(x[t - 1, i]).all() and np.isfinite(log_rv[t, i]):   # the pair that became known this month
                a += np.outer(x[t - 1, i], x[t - 1, i])
                b += x[t - 1, i] * log_rv[t, i]
                count += 1
            if count >= HAR_MIN_OBS and np.isfinite(x[t, i]).all():
                beta = np.linalg.lstsq(a, b, rcond=None)[0]
                out[t, i] = float(np.exp(x[t, i] @ beta))
    return out


def point_forecasts(rp: RiskPanel) -> dict[str, np.ndarray]:
    return {"last": last_value(rp.rv), "mean60": trailing_mean(rp.rv), "ewma": rp.ewma.copy(), "har": har(rp.rv)}


def quantiles_from_errors(point: np.ndarray, rv: np.ndarray, window: int = ERROR_WINDOW, levels: tuple[float, ...] = LEVELS) -> np.ndarray:
    """[T, N, Q]: the point forecast times each quantile of its own last ``window`` errors (realized over forecast).

    The error of the forecast made at month s is known at the end of month s + 1, so the forecast
    at t uses the errors of s = t - window .. t - 1. Missing until an industry has all of them."""
    t_max, n = rv.shape
    err = np.full_like(rv, np.nan)
    err[:-1] = np.log(rv[1:]) - np.log(point[:-1])
    out = np.full((t_max, n, len(levels)), np.nan)
    for t in range(window, t_max):
        block = err[t - window:t]
        ok = np.isfinite(block).all(axis=0) & np.isfinite(point[t])
        if ok.any():
            q = np.quantile(block[:, ok], levels, axis=0)            # [Q, n_ok]
            out[t, ok] = (point[t, ok][None, :] * np.exp(q)).T
    return out


def test_origins(rp: RiskPanel, oos_start: str) -> np.ndarray:
    """Decision indices scored: the month-end before ``oos_start`` up to the second-to-last month."""
    return np.arange(rp.index_of(oos_start) - 1, len(rp.dates) - 1)


@dataclass(frozen=True)
class Forecast:
    name: str
    origins: list[str]           # 'YYYY-MM' of each decision month-end
    names: list[str]
    q: np.ndarray                # [M, N, Q] quantiles of next month's realized volatility
    meta: dict

    @property
    def median(self) -> np.ndarray:
        return self.q[:, :, MEDIAN]


def save_forecast(path: Path, origins: list[str], names: list[str], q: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, q=q.astype(np.float32), levels=np.array(LEVELS), origins=np.array(origins),
                        names=np.array(names), meta=np.array(json.dumps(meta, sort_keys=True)))


def load_forecast(path: Path, origins: list[str], names: list[str]) -> Forecast:
    """Read a forecast file and refuse one that is not for these months, industries and levels."""
    z = np.load(path, allow_pickle=False)
    if tuple(np.round(z["levels"], 6)) != LEVELS:
        raise ValueError(f"{path.name}: quantile levels {z['levels']} are not the protocol's")
    if list(z["origins"]) != list(origins) or list(z["names"]) != list(names):
        raise ValueError(f"{path.name}: not the protocol's months or industries")
    q = z["q"].astype(float)
    if q.shape != (len(origins), len(names), len(LEVELS)) or not np.isfinite(q).all() or (q <= 0).any():
        raise ValueError(f"{path.name}: forecasts must be finite, positive and of shape months x industries x levels")
    if (np.diff(q, axis=2) < -1e-6 * q[:, :, :-1]).any():
        raise ValueError(f"{path.name}: quantiles cross")
    return Forecast(path.stem, list(origins), list(names), q, json.loads(str(z["meta"])))
