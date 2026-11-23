"""Public series for the institutional toolkit, fetched from FRED on demand and cached (never committed)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"

# US Treasury constant-maturity yields (par, bond-equivalent, percent), maturity in years.
TREASURY = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0, "DGS3": 3.0,
            "DGS5": 5.0, "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0}
FX = "DEXCAUS"   # Canadian dollars per US dollar, daily noon/close


def fred_series(series_id: str, cache_dir: Path, refresh: bool = False) -> pd.Series:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{series_id}.csv"
    if refresh or not path.exists():
        resp = requests.get(FRED.format(id=series_id), timeout=60)
        resp.raise_for_status()
        path.write_text(resp.text)
    df = pd.read_csv(path)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna().rename(series_id)


def treasury_daily(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Daily yields in decimals; columns are maturities in years. Missing maturities stay NaN."""
    cols = {m: fred_series(sid, cache_dir, refresh) / 100.0 for sid, m in TREASURY.items()}
    return pd.DataFrame(cols).sort_index(axis=1)


def month_end(df: pd.DataFrame | pd.Series):
    return df.resample("ME").last()
