"""The monthly panel for the pension study. Everything is public and fetched on demand; nothing is committed.

    equity  the US stock market's total return for the month (Kenneth French: Mkt-RF + RF)
    yield   the 10-year Treasury constant-maturity yield at the end of the month
            (FRED DGS10, daily, from January 1962; before that FRED GS10, the monthly average,
            which is smoother than a month-end reading: those nine years only ever feed the
            scenario generator's early fits)
    infl    the month's change in the US consumer price index (FRED CPIAUCNS, not seasonally adjusted;
            the one month that was never published, October 2025, is interpolated)

Rows are month-ends from April 1953, when the 10-year series begins.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
FIRST = "1953-04"
DAILY_FROM = "1962-01"
FILES = {"DGS10": "dgs10_daily.csv", "GS10": "gs10_monthly.csv", "CPIAUCNS": "cpi_monthly.csv", "french": "french_factors.csv"}


def download(cache_dir: Path, refresh: bool = False) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: cache_dir / file for name, file in FILES.items()}
    for name, path in paths.items():
        if path.exists() and not refresh:
            continue
        if name == "french":
            resp = requests.get(FRENCH, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                path.write_text(zf.read(member).decode("latin-1"))
        else:
            resp = requests.get(FRED.format(id=name), timeout=60)
            resp.raise_for_status()
            path.write_text(resp.text)
    return paths


def _fred(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def _french_market(path: Path) -> pd.Series:
    """Monthly market total return, in decimals, from the monthly block of the factor file."""
    rows = {}
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and re.fullmatch(r"\d{6}", parts[0]):
            rows[pd.Timestamp(year=int(parts[0][:4]), month=int(parts[0][4:]), day=1) + pd.offsets.MonthEnd(0)] = (float(parts[1]) + float(parts[4])) / 100.0
    return pd.Series(rows).sort_index()


def build(paths: dict[str, Path]) -> pd.DataFrame:
    daily = _fred(paths["DGS10"]).resample("ME").last().dropna() / 100.0
    monthly = _fred(paths["GS10"]) / 100.0
    monthly.index = monthly.index + pd.offsets.MonthEnd(0)
    yld = pd.concat([monthly[monthly.index < DAILY_FROM], daily[daily.index >= DAILY_FROM]])
    cpi = _fred(paths["CPIAUCNS"])
    cpi.index = cpi.index + pd.offsets.MonthEnd(0)
    # A month the statistical agency never published (October 2025) is bridged: the index is interpolated
    # geometrically between its neighbours, one missing month at a time and never more.
    full = pd.date_range(cpi.index[0], cpi.index[-1], freq="ME")
    cpi = np.exp(np.log(cpi.reindex(full)).interpolate(limit=1, limit_area="inside"))
    panel = pd.DataFrame({"equity": _french_market(paths["french"]), "yield": yld, "infl": cpi.pct_change(), "cpi": cpi}).dropna()
    panel = panel[panel.index >= FIRST]
    # A month belongs to the panel only once it is complete in all three sources.
    gaps = panel.index.to_series().diff().dropna().dt.days.max()
    if gaps > 31:
        raise ValueError("the panel has a missing month")
    return panel


def load(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    return build(download(cache_dir, refresh))


def summary(panel: pd.DataFrame) -> dict:
    return {
        "first_month": panel.index[0].strftime("%Y-%m"),
        "last_month": panel.index[-1].strftime("%Y-%m"),
        "months": int(len(panel)),
        "equity_mean_annual": float(panel["equity"].mean() * 12),
        "equity_vol_annual": float(panel["equity"].std() * 12 ** 0.5),
        "yield_min": float(panel["yield"].min()),
        "yield_max": float(panel["yield"].max()),
        "inflation_mean_annual": float(panel["infl"].mean() * 12),
        "sources": {"equity": "Kenneth French data library, F-F_Research_Data_Factors (Mkt-RF + RF)",
                    "yield": "FRED DGS10 (month-end, from 1962-01) and GS10 (monthly average, before)",
                    "inflation": "FRED CPIAUCNS"},
    }
