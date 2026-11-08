"""Point-in-time monthly cross-sections for the AlphaFormer, as a PyTorch Dataset.

One sample = one month-end t = the whole cross-section of stocks:

* ``seq``    (N, L, 2): each stock's last L monthly returns, as two channels:
  the cross-sectional z-score of that month's return (how it did versus
  peers) and the return divided by the stock's own trailing volatility (how
  unusual it was for that stock).
* ``static`` (N, F): log volatility at 1, 3 and 12 months, the six factor
  betas (cross-sectionally standardised), the three classical signal
  z-scores, and a sector one-hot.
* ``target`` (N,): the NEXT month's return, cross-sectionally standardised
  and clipped (the label; never an input).

Everything in ``seq`` and ``static`` is dated <= t. ``label_end[t]`` is the
last day the label depends on; walk-forward training uses a month only if
its label ended on or before the refit date.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dpc.config import AlphaModelConfig
from dpc.data.panel import Panel
from dpc.exposures import STYLE_FACTORS
from dpc.signals import SIGNAL_NAMES, forward_returns, monthly_returns

VOL_WINDOWS = (21, 63, 252)  # 1, 3 and 12 months of trading days
MIN_STOCKS = 20  # a month needs at least this many investable stocks
DAYS_PER_YEAR = 252
DAYS_PER_MONTH = 21


def _cs_z(x: np.ndarray, mask: np.ndarray, clip: float) -> np.ndarray:
    """Cross-sectional z-score along the last axis over ``mask``; 0 where masked."""
    m = mask & np.isfinite(x)
    cnt = m.sum(axis=-1, keepdims=True)
    xm = np.where(m, x, 0.0)
    mean = xm.sum(axis=-1, keepdims=True) / np.maximum(cnt, 1)
    var = (np.where(m, x - mean, 0.0) ** 2).sum(axis=-1, keepdims=True) / np.maximum(cnt - 1, 1)
    z = (x - mean) / np.sqrt(np.maximum(var, 1e-12))
    return np.where(m, np.clip(z, -clip, clip), 0.0)


@dataclass
class AlphaBank:
    dates: pd.DatetimeIndex  # month-ends
    tickers: list[str]
    seq: np.ndarray  # (M, N, L, 2) float32
    static: np.ndarray  # (M, N, F) float32
    mask: np.ndarray  # (M, N) investable with complete features
    target: np.ndarray  # (M, N) float32, 0 where no label
    mask_y: np.ndarray  # (M, N) label available
    label_end: pd.DatetimeIndex  # (M,)

    @property
    def n_static(self) -> int:
        return self.static.shape[-1]

    def rows_for(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return self.dates.get_indexer(dates)


def build_alpha_bank(panel: Panel, signals: dict[str, pd.DataFrame], betas: dict[str, pd.DataFrame], ac: AlphaModelConfig) -> AlphaBank:
    me = panel.month_ends
    tick = panel.tickers
    L, clip = ac.seq_len_months, ac.feature_clip
    mret = monthly_returns(panel.returns, me).reindex(columns=tick)
    R = mret.to_numpy(dtype=float)  # (M_all, N)
    elig = panel.eligible.reindex(index=me, columns=tick).fillna(False).to_numpy(bool)

    vols = {w: panel.returns.rolling(w, min_periods=w).std().reindex(me)[tick].to_numpy() for w in VOL_WINDOWS}
    vol_m = vols[252] * math.sqrt(DAYS_PER_MONTH)  # trailing monthly vol for the time-series channel

    # Channel 0: cross-sectional z of each month's return (computed per month, all listed stocks)
    cs = _cs_z(R, np.isfinite(R), clip)
    ts = np.clip(R / np.where(vol_m > 0, vol_m, np.nan), -clip, clip)

    sectors = sorted(panel.universe["sector"].unique())
    onehot = np.stack([(panel.universe["sector"].reindex(tick) == s).to_numpy(float) for s in sectors], axis=1)

    M, N = len(me), len(tick)
    static_parts = []
    finite = np.ones((M, N), bool)
    for w in VOL_WINDOWS:
        lv = np.log(vols[w] * math.sqrt(DAYS_PER_YEAR))
        finite &= np.isfinite(lv)
        static_parts.append(_cs_z(lv, elig, clip))
    for f in STYLE_FACTORS:
        b = betas[f].reindex(index=me, columns=tick).to_numpy(float)
        finite &= np.isfinite(b)
        static_parts.append(_cs_z(b, elig, clip))
    for k in SIGNAL_NAMES:
        z = signals[k].reindex(index=me, columns=tick).to_numpy(float)
        finite &= np.isfinite(z)
        static_parts.append(np.nan_to_num(np.clip(z, -clip, clip)))
    static = np.concatenate([np.stack(static_parts, axis=-1), np.broadcast_to(onehot, (M, N, len(sectors)))], axis=-1)

    seq = np.zeros((M, N, L, 2), np.float32)
    seq_ok = np.zeros((M, N), bool)
    for t in range(L - 1, M):
        win = R[t - L + 1 : t + 1]  # (L, N): months t-L+1 .. t, all dated <= t
        ok = np.isfinite(win).all(axis=0) & np.isfinite(ts[t - L + 1 : t + 1]).all(axis=0)
        seq[t, :, :, 0] = np.nan_to_num(cs[t - L + 1 : t + 1]).T
        seq[t, :, :, 1] = np.nan_to_num(ts[t - L + 1 : t + 1]).T
        seq_ok[t] = ok

    mask = elig & finite & seq_ok
    fwd = forward_returns(mret).to_numpy(dtype=float)
    mask_y = mask & np.isfinite(fwd)
    target = _cs_z(fwd, mask_y, ac.target_clip).astype(np.float32)

    label_end = list(me[1:]) + [panel.returns.index[-1]]
    keep = mask.sum(axis=1) >= MIN_STOCKS
    return AlphaBank(
        dates=me[keep], tickers=tick, seq=seq[keep], static=static[keep].astype(np.float32), mask=mask[keep],
        target=target[keep], mask_y=mask_y[keep], label_end=pd.DatetimeIndex(label_end)[keep],
    )


class MonthDataset(Dataset):
    """Each item is one month's cross-section (all stocks), ready for the AlphaFormer."""

    def __init__(self, bank: AlphaBank, rows: np.ndarray, with_target: bool = True):
        self.bank, self.rows, self.with_target = bank, np.asarray(rows), with_target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        k = int(self.rows[i])
        b = self.bank
        item = {
            "seq": torch.from_numpy(b.seq[k]),
            "static": torch.from_numpy(b.static[k]),
            "mask": torch.from_numpy(b.mask[k]),
        }
        if self.with_target:
            item["target"] = torch.from_numpy(b.target[k])
            item["mask_y"] = torch.from_numpy(b.mask_y[k])
        return item
