"""Download and cache raw data: Yahoo Finance prices and Ken French factors.

Everything lands in ``data/raw`` as Parquet. Re-running is free: the cache is
used unless ``refresh=True``. Raw files are never edited by hand; all cleaning
happens downstream in :mod:`dpc.data.panel` so it is visible and testable.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from dpc.config import Config
from dpc.logging_utils import get_logger

log = get_logger(__name__)

PRICE_FIELDS = ("Adj Close", "Close", "Volume", "Dividends", "Stock Splits")
HTTP_TIMEOUT_SECONDS = 60


def _price_cache(cfg: Config) -> Path:
    return cfg.paths.data_raw / "prices_long.parquet"


def _ff_cache(cfg: Config) -> Path:
    return cfg.paths.data_raw / "ff_daily.parquet"


def download_prices(cfg: Config, tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV + corporate actions in long format (date, ticker, field columns)."""
    cache = _price_cache(cfg)
    if cache.exists() and not refresh:
        log.info("Using cached prices: %s", cache)
        return pd.read_parquet(cache)

    import yfinance as yf  # imported lazily: only needed when downloading

    end = cfg.data.end_date or (date.today() + timedelta(days=1)).isoformat()  # yf end is exclusive
    log.info("Downloading %d tickers from Yahoo Finance (%s -> %s)", len(tickers), cfg.data.start_date, end)
    wide = yf.download(
        tickers,
        start=cfg.data.start_date,
        end=end,
        auto_adjust=False,  # keep both raw Close and Adj Close so splits can be audited
        actions=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if wide is None or wide.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    long = wide[list(PRICE_FIELDS)].stack(level="Ticker", future_stack=True).reset_index()
    long.columns = ["date", "ticker", *PRICE_FIELDS]
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    long = long.dropna(subset=["Adj Close", "Close"], how="all")
    failed = sorted(set(tickers) - set(long["ticker"].unique()))
    if failed:
        raise RuntimeError(f"No price data for: {failed}. Check the tickers in the universe file.")
    long.to_parquet(cache, index=False)
    log.info("Cached %d rows to %s", len(long), cache)
    return long


def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV (header prose, then 'YYYYMMDD,f1,f2,...')."""
    lines = text.splitlines()
    header_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith(",") or ln.lower().startswith(",mkt"))
    header = [h.strip() for h in lines[header_idx].split(",")]
    rows = []
    for ln in lines[header_idx + 1 :]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(header) or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:  # first non-data line after the data block ends the table
                break
            continue
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date", *header[1:]])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100.0  # percent -> decimal
    return df


def _fetch_french_zip(url: str) -> str:
    resp = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        return zf.read(name).decode("latin-1")


def download_fama_french(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    """Daily Fama-French 5 factors + momentum + risk-free rate (decimals)."""
    cache = _ff_cache(cfg)
    if cache.exists() and not refresh:
        log.info("Using cached Fama-French factors: %s", cache)
        return pd.read_parquet(cache)
    log.info("Downloading Fama-French factors from %s", cfg.data.ff_base_url)
    five = _parse_french_csv(_fetch_french_zip(cfg.data.ff_base_url + cfg.data.ff_five_factor_file))
    mom = _parse_french_csv(_fetch_french_zip(cfg.data.ff_base_url + cfg.data.ff_momentum_file))
    mom.columns = ["Mom"]
    ff = five.join(mom, how="inner")
    expected = {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"}
    if set(ff.columns) != expected:
        raise RuntimeError(f"Unexpected Fama-French columns: {list(ff.columns)}")
    ff = ff.loc[cfg.data.start_date :]
    ff.index.name = "date"
    ff.to_parquet(cache)
    log.info("Cached Fama-French factors %s -> %s", ff.index.min().date(), ff.index.max().date())
    return ff
