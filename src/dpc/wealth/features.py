"""Point-in-time features for each decision date.

Every feature at decision index t uses data up to the end of month t only
(tests/test_wealth.py checks this by truncating the panel). Industry features
are standardised across the 30 industries at each date, so the network sees
relative positions (which industry is cheapest, fastest-rising, calmest)
rather than levels that drift over a century. Market-level context features
are standardised with an expanding window of their own past.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .data import IndustryPanel

ASSET_FEATURES = ["ret_1m", "ret_3m", "mom_12_1", "ret_36m", "ret_60m", "vol_12m", "vol_60m",
                  "downside_36m", "drawdown_36m", "beta_60m", "corr_60m", "log_bm", "log_mkt_weight"]
CONTEXT_FEATURES = ["mkt_ret_12m", "mkt_ret_60m", "mkt_vol_12m", "mkt_drawdown_36m", "tbill", "dispersion_12m", "mkt_log_bm"]
LOOKBACK = 60  # months of history the longest feature needs


@dataclass(frozen=True)
class Features:
    assets: np.ndarray    # [T, N, F] standardised industry features (nan before the first valid date)
    context: np.ndarray   # [T, G] standardised market features
    first_valid: int      # first decision index with every feature defined


def _rolling(x: np.ndarray, window: int, fn) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    for t in range(window - 1, len(x)):
        out[t] = fn(x[t - window + 1:t + 1])
    return out


def _zscore_cross(x: np.ndarray, clip: float) -> np.ndarray:
    with warnings.catch_warnings():                    # rows before the first full lookback are all missing
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
    z = (x - mu) / np.where(sd > 1e-12, sd, 1.0)
    return np.clip(np.nan_to_num(z, nan=0.0), -clip, clip)


def build_features(panel: IndustryPanel, clip: float = 4.0, min_history: int = 120) -> Features:
    r = panel.returns
    T, N = r.shape
    logr = np.log1p(r)
    csum = np.vstack([np.zeros((1, N)), np.cumsum(logr, axis=0)])  # csum[k] = sum of logr[:k]

    def trailing(window: int, skip: int = 0) -> np.ndarray:
        out = np.full((T, N), np.nan)
        for t in range(window - 1, T):
            out[t] = csum[t + 1 - skip] - csum[t + 1 - window]
        return out

    mkt = panel.mkt_rf + panel.rf                      # French market total return, month t
    feats = {
        "ret_1m": logr.copy(),
        "ret_3m": trailing(3),
        "mom_12_1": trailing(12, skip=1),
        "ret_36m": trailing(36),
        "ret_60m": trailing(60),
        "vol_12m": _rolling(r, 12, lambda w: w.std(axis=0, ddof=1)),
        "vol_60m": _rolling(r, 60, lambda w: w.std(axis=0, ddof=1)),
        "downside_36m": _rolling(r, 36, lambda w: np.sqrt((np.minimum(w, 0.0) ** 2).mean(axis=0))),
    }
    level = csum[1:]                                   # log price index at the end of month t
    feats["drawdown_36m"] = _rolling(level, 36, lambda w: w[-1] - w.max(axis=0))
    beta = np.full((T, N), np.nan)
    corr = np.full((T, N), np.nan)
    for t in range(LOOKBACK - 1, T):
        y = r[t - LOOKBACK + 1:t + 1]
        x = mkt[t - LOOKBACK + 1:t + 1]
        xc = x - x.mean()
        yc = y - y.mean(axis=0)
        cov = xc @ yc / (LOOKBACK - 1)
        beta[t] = cov / xc.var(ddof=1)
        corr[t] = cov / (xc.std(ddof=1) * yc.std(axis=0, ddof=1))
    feats["beta_60m"], feats["corr_60m"] = beta, corr
    bm = np.where(panel.bm > 0, panel.bm, np.nan)
    feats["log_bm"] = np.log(bm)
    weights = panel.market_weights()
    feats["log_mkt_weight"] = np.log(weights)

    first_valid = LOOKBACK - 1
    assets = np.stack([_zscore_cross(feats[name], clip) for name in ASSET_FEATURES], axis=-1)
    assets[:first_valid] = np.nan

    lm = np.log1p(mkt)
    lcs = np.concatenate([[0.0], np.cumsum(lm)])
    ctx_raw = np.full((T, len(CONTEXT_FEATURES)), np.nan)
    for t in range(first_valid, T):
        level_t = lcs[1:t + 2]
        ctx_raw[t] = [
            lcs[t + 1] - lcs[t + 1 - 12],
            lcs[t + 1] - lcs[t + 1 - 60],
            mkt[t - 11:t + 1].std(ddof=1),
            level_t[-1] - level_t[-36:].max(),
            12 * panel.rf[t],
            np.nanstd(feats["mom_12_1"][t] if t >= 12 else np.nan),
            np.nansum(weights[t] * feats["log_bm"][t]) / np.sum(weights[t][~np.isnan(feats["log_bm"][t])]),
        ]
    context = np.full_like(ctx_raw, np.nan)
    for t in range(first_valid, T):                     # expanding standardisation: the past only
        hist = ctx_raw[first_valid:t + 1]
        mu = hist.mean(axis=0)
        sd = hist.std(axis=0) if t - first_valid + 1 >= min_history else np.ones(hist.shape[1])
        context[t] = np.clip((ctx_raw[t] - mu) / np.where(sd > 1e-12, sd, 1.0), -clip, clip)
    return Features(assets=assets, context=context, first_valid=first_valid)
