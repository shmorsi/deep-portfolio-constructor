"""Market indices to compare against: the S&P 500 (dividends reinvested) and the NASDAQ Composite.

Added after the pre-registered test, as context. The protocol's benchmark is the
whole US market built from the 30 industries, and its verdict does not change.

Sources, downloaded once and cached under data/raw/benchmarks/ (never committed):

* Yahoo Finance daily closes: ``^SP500TR`` (S&P 500 total return, from January
  1988), ``^GSPC`` (S&P 500 price, from 1927) and ``^IXIC`` (NASDAQ Composite
  price, from February 1971).
* Robert Shiller's monthly S&P 500 dividends (annual dividends, from S&P's
  four-quarter totals), through the datasets/s-and-p-500 mirror of his data.

S&P 500 total return, monthly: from February 1988 the official ``^SP500TR``
month-end to month-end; before that, the month-end ``^GSPC`` price return plus
one twelfth of Shiller's dividend yield (annual dividends over that month's
average price). The two overlap from 1988 to mid-2023, which checks the splice.

NASDAQ Composite: price only. No free source has its dividends before 2003, so
it is understated by its dividend yield; this is said wherever it is shown.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..logging_utils import get_logger

log = get_logger(__name__)

TICKERS = {"sp500_tr": "^SP500TR", "sp500_price": "^GSPC", "nasdaq_price": "^IXIC"}
SHILLER_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
SPLICE = "1988-01"          # last month built from price plus dividends; ^SP500TR from the month after
NASDAQ_FIRST = "1971-03"    # first full month of the NASDAQ Composite (it starts on 5 February 1971)
LABELS = {"sp500": "S&P 500 (dividends reinvested)", "nasdaq": "NASDAQ Composite (price only)"}


def download(cache_dir: Path, refresh: bool = False) -> dict[str, Path]:
    """Daily closes from Yahoo Finance and Shiller's monthly file, cached as CSV."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: cache_dir / f"{name}_daily.csv" for name in TICKERS}
    paths["shiller"] = cache_dir / "shiller_sp500_monthly.csv"
    missing = [n for n in TICKERS if refresh or not paths[n].exists()]
    if missing:
        import yfinance as yf  # only needed to refresh the cache

        for name in missing:
            hist = yf.Ticker(TICKERS[name]).history(period="max", interval="1d", auto_adjust=False)
            if hist.empty:
                raise RuntimeError(f"Yahoo Finance returned no data for {TICKERS[name]}")
            closes = hist["Close"].rename("close")
            closes.index = pd.DatetimeIndex(closes.index.date, name="date")
            closes.to_csv(paths[name])
            log.info("Downloaded %s: %d days, %s to %s", TICKERS[name], len(closes), closes.index[0].date(), closes.index[-1].date())
    if refresh or not paths["shiller"].exists():
        resp = requests.get(SHILLER_URL, timeout=60)
        resp.raise_for_status()
        paths["shiller"].write_text(resp.text)
    return paths


def month_end_closes(path: Path) -> pd.Series:
    daily = pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    return daily.resample("ME").last().dropna()


def shiller_dividend_yield(path: Path) -> pd.Series:
    """Annual dividends over the month's average price, indexed by month-end (zero where the mirror has no dividend)."""
    s = pd.read_csv(io.StringIO(path.read_text()), parse_dates=["Date"]).set_index("Date")
    s.index = s.index + pd.offsets.MonthEnd(0)
    return (s["Dividend"] / s["SP500"]).where(s["Dividend"] > 0)


def sp500_total_return(paths: dict[str, Path]) -> tuple[pd.Series, dict[str, float]]:
    """Monthly S&P 500 total returns and a check of the splice on the months both sources cover."""
    price = month_end_closes(paths["sp500_price"]).pct_change()
    official = month_end_closes(paths["sp500_tr"]).pct_change()
    built = (price + shiller_dividend_yield(paths["shiller"]).reindex(price.index) / 12).dropna()
    cut = pd.Timestamp(SPLICE) + pd.offsets.MonthEnd(0)
    tr = pd.concat([built[built.index <= cut], official[official.index > cut].dropna()])
    both = pd.concat([built, official], axis=1, keys=["built", "official"]).dropna()
    gap = np.log1p(both["built"]) - np.log1p(both["official"])
    check = {"overlap_months": int(len(both)), "first": f"{both.index[0]:%Y-%m}", "last": f"{both.index[-1]:%Y-%m}",
             "annual_log_gap": float(12 * gap.mean()), "monthly_tracking_sd": float(gap.std()),
             "correlation": float(both["built"].corr(both["official"]))}
    return tr.rename("sp500"), check


def nasdaq_price_return(paths: dict[str, Path]) -> pd.Series:
    r = month_end_closes(paths["nasdaq_price"]).pct_change()
    return r[r.index >= pd.Timestamp(NASDAQ_FIRST) + pd.offsets.MonthEnd(0)].dropna().rename("nasdaq")


def build(cache_dir: Path, dates: pd.DatetimeIndex, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Monthly benchmark returns on the study's out-of-sample months (NaN before an index exists)."""
    paths = download(cache_dir, refresh)
    sp, check = sp500_total_return(paths)
    nq = nasdaq_price_return(paths)
    out = pd.DataFrame({"sp500": sp, "nasdaq": nq}).reindex(dates)
    missing = out["sp500"].isna().sum()
    if missing:
        raise ValueError(f"S&P 500 total return is missing for {missing} out-of-sample months")
    return out, {"sp500_splice": check, "sources": {**TICKERS, "shiller": SHILLER_URL}}
