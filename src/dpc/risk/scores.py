"""Forecast accuracy: pooled quantile loss, absolute error, QLIKE and coverage, with block-bootstrap ratios.

Everything is first summed within a month over the 30 industries, so the bootstrap resamples
whole months (in blocks) and keeps the industries of a month together, as the protocol says.
"""

from __future__ import annotations

import numpy as np

from .forecasters import LEVELS, MEDIAN


def month_sums(y: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """Per-month sums over industries. y [M, N] realized volatility, q [M, N, Q] quantile forecasts."""
    d = y[:, :, None] - q
    lv = np.asarray(LEVELS)[None, None, :]
    pin = np.maximum(lv * d, (lv - 1.0) * d).sum(axis=2)          # [M, N]
    med = q[:, :, MEDIAN]
    ratio = (y / med) ** 2
    return {
        "wql_num": 2.0 * pin.sum(axis=1) / len(LEVELS),
        "abs_y": np.abs(y).sum(axis=1),
        "abs_err": np.abs(y - med).sum(axis=1),
        "qlike": (ratio - np.log(ratio) - 1.0).sum(axis=1),
        "inside": ((y >= q[:, :, 0]) & (y <= q[:, :, -1])).sum(axis=1).astype(float),
        "n": np.full(y.shape[0], float(y.shape[1])),
    }


def pooled(s: dict[str, np.ndarray]) -> dict[str, float]:
    n = s["n"].sum()
    return {"wql": float(s["wql_num"].sum() / s["abs_y"].sum()), "mae": float(s["abs_err"].sum() / n),
            "qlike": float(s["qlike"].sum() / n), "cover80": float(s["inside"].sum() / n)}


def block_indices(n: int, block: int, draws: int, seed: int) -> np.ndarray:
    """[draws, n] month indices of a moving-block bootstrap (the wealth study's scheme)."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, (draws, k))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)[:, :n]


def ratio_ci(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    """sum(a) / sum(b) over months, with its 95% bootstrap interval. a and b are per-month sums."""
    r = a[idx].sum(axis=1) / b[idx].sum(axis=1)
    return float(a.sum() / b.sum()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def verdict_ratio(lo: float, hi: float) -> str:
    """The protocol's rule for a loss ratio against HAR."""
    return "better" if hi < 1.0 else "worse" if lo > 1.0 else "no detectable difference"


def against(s: dict[str, np.ndarray], ref: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, dict]:
    """Loss ratios to a reference forecaster on the same months, each with its interval and verdict."""
    out = {}
    for key, num in (("wql", "wql_num"), ("mae", "abs_err"), ("qlike", "qlike")):
        ratio, lo, hi = ratio_ci(s[num], ref[num], idx)
        out[key] = {"ratio": ratio, "ci_low": lo, "ci_high": hi, "verdict": verdict_ratio(lo, hi)}
    return out
