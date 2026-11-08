"""Kenneth French's 30 industry portfolios and monthly factors, 1926 to today.

Why this data
-------------
A long-only backtest on today's large caps is flattered by survivorship: every
stock in it is one that survived and grew. Industry portfolios hold every
listed firm in the industry at each date, including the ones that later
failed, so a 1960 investor could really have held them. They also give a
century of history, enough to test a long-term strategy across many regimes.

Timing conventions (monthly, index ``t`` = a month-end)
------------------------------------------------------
* ``returns[t]``  : value-weighted total return of each industry over month t.
* ``cap_start[t]``: industry market cap at the start of month t (number of firms
  x average firm size in the file's month-t row; the file value-weights month t
  with these caps). It is measured at the end of month t-1.
* A decision taken at the end of month t is held through month t+1. The market
  weights known at that decision are ``cap_start[t+1]`` (measured at the end of
  month t), exposed as ``cap_decision[t]``.
* ``bm[t]``: industry book-to-market known at the end of month t. French forms
  year Y's value at the end of June of Y; it is used from the end of July of Y.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..logging_utils import get_logger

log = get_logger(__name__)

SECTION_MONTHLY_VW = "Average Value Weighted Returns -- Monthly"
SECTION_FIRMS = "Number of Firms in Portfolios"
SECTION_SIZE = "Average Firm Size"
SECTION_BM = "Value-Weighted Average of BE/ME"
MISSING = (-99.99, -999.0)


@dataclass(frozen=True)
class IndustryPanel:
    """Everything the wealth study reads, aligned on month-ends."""

    dates: pd.DatetimeIndex          # month-ends
    names: list[str]                 # industry short names, in file order
    returns: np.ndarray              # [T, N] decimal total returns over month t
    cap_start: np.ndarray            # [T, N] market cap at the start of month t ($m)
    bm: np.ndarray                   # [T, N] book-to-market known at the end of month t (nan if none yet)
    rf: np.ndarray                   # [T] one-month T-bill return over month t
    mkt_rf: np.ndarray               # [T] French market excess return over month t
    smb: np.ndarray                  # [T]
    hml: np.ndarray                  # [T]

    @property
    def n_months(self) -> int:
        return len(self.dates)

    @property
    def n_assets(self) -> int:
        return len(self.names)

    def cap_decision(self) -> np.ndarray:
        """[T, N] market caps known at the end of month t (the next month's start caps).

        The last month has no next row; its caps are rolled forward by its own returns."""
        out = np.empty_like(self.cap_start)
        out[:-1] = self.cap_start[1:]
        out[-1] = self.cap_start[-1] * (1.0 + self.returns[-1])
        return out

    def market_weights(self) -> np.ndarray:
        """[T, N] cap weights known at each decision (end of month t)."""
        caps = self.cap_decision()
        return caps / caps.sum(axis=1, keepdims=True)

    def index_of(self, month: str) -> int:
        """Position of the month-end of 'YYYY-MM'."""
        ts = pd.Timestamp(month) + pd.offsets.MonthEnd(0)
        pos = self.dates.get_indexer([ts])[0]
        if pos < 0:
            raise KeyError(f"{month} is outside the panel {self.dates[0]:%Y-%m}..{self.dates[-1]:%Y-%m}")
        return int(pos)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _fetch_zip(url: str) -> str:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.read(name).decode("latin-1")


def parse_sections(text: str) -> dict[str, pd.DataFrame]:
    """Split a French CSV into its tables. Monthly rows are YYYYMM, annual rows YYYY.

    A table starts at a header line beginning with ',' and is keyed by the nearest
    non-empty text line above it (the factor file's monthly table has no title of its own)."""
    lines = text.splitlines()
    sections: dict[str, pd.DataFrame] = {}
    title = "untitled"
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(","):
            header = [h.strip() for h in line.split(",")[1:]]
            rows, j = [], i + 1
            while j < len(lines) and re.match(r"^\s*\d{4,6}\s*,", lines[j]):
                parts = [p.strip() for p in lines[j].split(",")]
                rows.append([parts[0]] + [float(x) for x in parts[1:len(header) + 1]])
                j += 1
            if rows:
                key, n = title, 2
                while key in sections:
                    key, n = f"{title} ({n})", n + 1
                df = pd.DataFrame(rows, columns=["key", *header]).set_index("key")
                sections[key] = df.replace(list(MISSING), np.nan)
            i = j
            continue
        if line.strip():
            title = line.strip()
        i += 1
    return sections


def _monthly_index(keys: pd.Index) -> pd.DatetimeIndex:
    return pd.to_datetime(keys.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)


def build_panel(industry_text: str, factor_text: str) -> IndustryPanel:
    ind = parse_sections(industry_text)
    for need in (SECTION_MONTHLY_VW, SECTION_FIRMS, SECTION_SIZE, SECTION_BM):
        if need not in ind:
            raise ValueError(f"industry file has no section {need!r}; found {list(ind)}")
    ret = ind[SECTION_MONTHLY_VW] / 100.0
    ret.index = _monthly_index(ret.index)
    firms = ind[SECTION_FIRMS]
    firms.index = _monthly_index(firms.index)
    size = ind[SECTION_SIZE]
    size.index = _monthly_index(size.index)
    bm_annual = ind[SECTION_BM]
    bm_annual.index = bm_annual.index.astype(int)
    names = list(ret.columns)
    if list(firms.columns) != names or list(size.columns) != names or list(bm_annual.columns) != names:
        raise ValueError("industry sections do not share the same columns")

    fac_sections = parse_sections(factor_text)
    monthly_key = next(k for k, v in fac_sections.items() if len(v.index[0]) == 6)
    fac = fac_sections[monthly_key] / 100.0
    fac.index = _monthly_index(fac.index)

    dates = ret.index.intersection(firms.index).intersection(size.index).intersection(fac.index)
    ret, firms, size, fac = ret.loc[dates], firms.loc[dates], size.loc[dates], fac.loc[dates]
    if ret.isna().any().any():
        bad = ret.columns[ret.isna().any()].tolist()
        raise ValueError(f"missing monthly returns in {bad}; the study needs a complete panel")

    # Book-to-market for year Y is formed at the end of June of Y; use it from the end of July of Y.
    bm = np.full(ret.shape, np.nan)
    for k, d in enumerate(dates):
        year = d.year if d.month >= 7 else d.year - 1
        if year in bm_annual.index:
            bm[k] = bm_annual.loc[year].to_numpy(dtype=float)

    return IndustryPanel(
        dates=pd.DatetimeIndex(dates),
        names=names,
        returns=ret.to_numpy(dtype=float),
        cap_start=(firms * size).to_numpy(dtype=float),
        bm=bm,
        rf=fac["RF"].to_numpy(dtype=float),
        mkt_rf=fac["Mkt-RF"].to_numpy(dtype=float),
        smb=fac["SMB"].to_numpy(dtype=float),
        hml=fac["HML"].to_numpy(dtype=float),
    )


# --------------------------------------------------------------------------- #
# Download and cache
# --------------------------------------------------------------------------- #
def load_panel(base_url: str, industry_file: str, factor_file: str, cache_dir: Path, refresh: bool = False) -> IndustryPanel:
    """Download (or read the cached copy of) both files and build the panel."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = {}
    for name in (industry_file, factor_file):
        cache = cache_dir / name.replace(".zip", ".txt")
        if cache.exists() and not refresh:
            texts[name] = cache.read_text(encoding="latin-1")
        else:
            log.info("Downloading %s", base_url + name)
            texts[name] = _fetch_zip(base_url + name)
            cache.write_text(texts[name], encoding="latin-1")
    panel = build_panel(texts[industry_file], texts[factor_file])
    log.info("Industry panel: %d industries, %s to %s (%d months)", panel.n_assets,
             f"{panel.dates[0]:%Y-%m}", f"{panel.dates[-1]:%Y-%m}", panel.n_months)
    return panel
