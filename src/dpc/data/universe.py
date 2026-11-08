"""The stock universe: tickers, names, country and sector labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("ticker", "name", "country", "sector")


def load_universe(path: Path) -> pd.DataFrame:
    """Read the universe CSV and validate it.

    The sector labels are *current* GICS sectors, so they are not
    point-in-time (a known limitation documented in the README).
    """
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    if df["ticker"].duplicated().any():
        dupes = df.loc[df["ticker"].duplicated(), "ticker"].tolist()
        raise ValueError(f"Duplicate tickers in universe: {dupes}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Universe file has blank cells")
    return df.set_index("ticker").sort_index()


def sector_dummies(universe: pd.DataFrame) -> pd.DataFrame:
    """One-hot sector matrix (ticker x sector), columns sorted alphabetically."""
    return pd.get_dummies(universe["sector"]).astype(float).sort_index(axis=1)
