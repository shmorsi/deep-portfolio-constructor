"""Turn the panel into tensors for the deep risk model (point-in-time).

A *sample* is an origin date t. Inputs use returns dated t-L+1 .. t only;
the target is the next H daily returns t+1 .. t+H. Every stock is scaled by
its own trailing volatility measured over the input window, so the network
learns *relative* structure (correlations, vol changes) and the forecast
starts from a sensible "trailing volatility" prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dpc.data.panel import Panel

CONTEXT_FACTORS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")
INPUT_CLIP = 10.0  # standardised returns beyond +-10 sigma are clipped (inputs only)
VOL_FLOOR = 1e-4  # daily vol floor (numerical)


@dataclass
class SampleBank:
    """All samples, precomputed as arrays (one row per origin date)."""

    origins: pd.DatetimeIndex
    origin_idx: np.ndarray  # position of each origin in the trading calendar
    x: np.ndarray  # (n, L, N) standardised stock returns
    f: np.ndarray  # (n, L, nf) standardised factor returns
    static: np.ndarray  # (n, N, ns) log vols + sector one-hot
    sigma: np.ndarray  # (n, N) trailing daily vol used for scaling
    target: np.ndarray  # (n, H, N) next-H raw daily returns (0 where invalid)
    mask_in: np.ndarray  # (n, N) full input history available at t
    mask_out: np.ndarray  # (n, N) full target window available (training only!)
    tickers: list[str]
    calendar: pd.DatetimeIndex
    horizon: int

    def index_of(self, date: pd.Timestamp) -> int:
        return int(self.origins.get_loc(date))


def build_sample_bank(panel: Panel, seq_len: int, horizon: int, stride: int) -> SampleBank:
    R = panel.returns.to_numpy(dtype=np.float64)
    T, N = R.shape
    cal = panel.returns.index
    F = panel.ff[list(CONTEXT_FACTORS)].to_numpy(dtype=np.float64)
    long_win = 4 * seq_len
    sectors = sorted(panel.universe["sector"].unique())
    onehot = np.stack([(panel.universe["sector"] == s).to_numpy(dtype=np.float64) for s in sectors], axis=1)

    me_pos = set(cal.get_indexer(panel.month_ends))
    first = long_win  # need a long window for the slow vol feature
    cand = [t for t in range(first, T) if (t - first) % stride == 0 or t in me_pos]
    n = len(cand)
    x = np.zeros((n, seq_len, N), np.float32)
    f = np.zeros((n, seq_len, len(CONTEXT_FACTORS)), np.float32)
    static = np.zeros((n, N, 2 + onehot.shape[1]), np.float32)
    sig = np.ones((n, N), np.float64)
    tgt = np.zeros((n, horizon, N), np.float64)
    m_in = np.zeros((n, N), bool)
    m_out = np.zeros((n, N), bool)
    for k, t in enumerate(cand):
        w = R[t - seq_len + 1 : t + 1]
        ok_in = np.isfinite(w).all(axis=0)
        s = np.where(ok_in, np.nanstd(w, axis=0, ddof=1), 1.0)
        s = np.maximum(s, VOL_FLOOR)
        wl = R[t - long_win + 1 : t + 1]
        sl = np.nanstd(wl, axis=0, ddof=1)
        sl = np.where(np.isfinite(sl), np.maximum(sl, VOL_FLOOR), s)
        x[k] = np.clip(np.nan_to_num(w / s, nan=0.0), -INPUT_CLIP, INPUT_CLIP)
        fw = F[t - seq_len + 1 : t + 1]
        f[k] = np.clip(fw / np.maximum(fw.std(axis=0, ddof=1), VOL_FLOOR), -INPUT_CLIP, INPUT_CLIP)
        static[k, :, 0] = np.log(s * np.sqrt(252))
        static[k, :, 1] = np.log(sl * np.sqrt(252))
        static[k, :, 2:] = onehot
        sig[k] = s
        m_in[k] = ok_in
        if t + horizon < T:
            y = R[t + 1 : t + 1 + horizon]
            ok_out = np.isfinite(y).all(axis=0)
            tgt[k] = np.nan_to_num(y, nan=0.0)
            m_out[k] = ok_out
    return SampleBank(
        origins=cal[cand], origin_idx=np.array(cand), x=x, f=f, static=static, sigma=sig, target=tgt,
        mask_in=m_in, mask_out=m_out, tickers=panel.tickers, calendar=cal, horizon=horizon,
    )
