"""Build the clean, aligned research panel from raw data.

The panel is the single input every later stage reads. Conventions:

* Calendar: trading days of the benchmark (SPY, NYSE calendar) that also have
  Fama-French factors. Canadian names are used via their US listings, so all
  prices are in USD on the same calendar.
* Returns: simple total returns from dividend- and split-adjusted closes.
  A missing price inside a stock's listed life is forward-filled for at most
  ``max_stale_run_days`` days, so the move is booked on the next valid day
  (total return is preserved; no artificial zero-then-jump is lost).
* Rebalance dates: the last trading day of each month.
* Eligibility at a rebalance date uses data up to and including that date only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dpc.config import Config
from dpc.data.download import download_fama_french, download_prices
from dpc.data.quality import QualityReport, check_quality, quality_markdown
from dpc.data.universe import load_universe
from dpc.logging_utils import get_logger

log = get_logger(__name__)

PANEL_FILES = ("returns", "prices", "ff", "benchmark", "eligible", "universe")


@dataclass
class Panel:
    returns: pd.DataFrame  # date x ticker, simple daily total returns (NaN = not listed)
    prices: pd.DataFrame  # date x ticker, adjusted close
    ff: pd.DataFrame  # date x {Mkt-RF, SMB, HML, RMW, CMA, RF, Mom}
    benchmark: pd.Series  # benchmark daily total return
    eligible: pd.DataFrame  # month_end x ticker, bool
    universe: pd.DataFrame  # ticker -> name, country, sector

    @property
    def month_ends(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.eligible.index)

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.returns.to_parquet(folder / "returns.parquet")
        self.prices.to_parquet(folder / "prices.parquet")
        self.ff.to_parquet(folder / "ff.parquet")
        self.benchmark.to_frame("benchmark").to_parquet(folder / "benchmark.parquet")
        self.eligible.to_parquet(folder / "eligible.parquet")
        self.universe.to_parquet(folder / "universe.parquet")

    @classmethod
    def load(cls, folder: Path) -> "Panel":
        missing = [f for f in PANEL_FILES if not (folder / f"{f}.parquet").exists()]
        if missing:
            raise FileNotFoundError(f"Panel files missing in {folder}: {missing}. Run `python -m dpc.cli data` first.")
        return cls(
            returns=pd.read_parquet(folder / "returns.parquet"),
            prices=pd.read_parquet(folder / "prices.parquet"),
            ff=pd.read_parquet(folder / "ff.parquet"),
            benchmark=pd.read_parquet(folder / "benchmark.parquet")["benchmark"],
            eligible=pd.read_parquet(folder / "eligible.parquet"),
            universe=pd.read_parquet(folder / "universe.parquet"),
        )


def month_end_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of every calendar month in ``calendar``."""
    s = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(s.groupby(calendar.to_period("M")).max().values)


def clean_returns(adj_close: pd.DataFrame, max_fill: int) -> pd.DataFrame:
    """Daily returns with short gaps bridged and nothing invented outside listing."""
    listed = adj_close.ffill().notna() & adj_close.bfill().notna()  # between first and last valid price
    filled = adj_close.ffill(limit=max_fill).where(listed)
    rets = filled.pct_change(fill_method=None)
    return rets.where(listed)


def eligibility(returns: pd.DataFrame, month_ends: pd.DatetimeIndex, min_history: int, max_missing: int) -> pd.DataFrame:
    """A stock is investable at month-end t if, in the ``min_history`` days up to
    and including t, it has at most ``max_missing`` missing returns and a valid
    return on t itself. Uses only information available at t."""
    valid = returns.notna().astype(float)
    count = valid.rolling(min_history, min_periods=min_history).sum()
    ok = (count >= min_history - max_missing) & returns.notna()
    return ok.reindex(month_ends).fillna(False).astype(bool)


def build_panel(cfg: Config, refresh: bool = False) -> tuple[Panel, QualityReport]:
    universe = load_universe(cfg.paths.universe_file)
    tickers = list(universe.index)
    bench = cfg.data.benchmark_ticker
    long = download_prices(cfg, sorted(set(tickers) | {bench}), refresh=refresh)
    ff = download_fama_french(cfg, refresh=refresh)

    bench_dates = pd.DatetimeIndex(sorted(long.loc[long["ticker"] == bench, "date"].unique()))
    calendar = bench_dates.intersection(ff.index)
    last_ff, last_px = ff.index.max(), bench_dates.max()
    if last_ff < last_px:
        log.warning(
            "Fama-French data ends %s but prices run to %s: the panel is truncated to %s "
            "(the French library is published with a lag).",
            last_ff.date(), last_px.date(), last_ff.date(),
        )

    report = check_quality(long[long["ticker"].isin(tickers)], calendar, cfg.data.quality)
    if report.failed_tickers:
        log.warning("Dropping tickers that failed quality checks: %s", report.failed_tickers)
    keep = [t for t in tickers if t not in report.failed_tickers]

    adj = long.pivot_table(index="date", columns="ticker", values="Adj Close", aggfunc="last")
    adj = adj.reindex(calendar)
    prices = adj[keep]
    returns = clean_returns(prices, cfg.data.quality.max_stale_run_days)
    benchmark = clean_returns(adj[[bench]], cfg.data.quality.max_stale_run_days)[bench]

    me = month_end_dates(calendar)
    # If the data stops mid-month, its last day is not a real month-end: drop it.
    if calendar[-1] < calendar[-1] + pd.offsets.BMonthEnd(0):
        me = me[:-1]
    elig = eligibility(returns, me, cfg.data.universe.min_history_days, cfg.data.universe.max_missing_in_window)

    panel = Panel(
        returns=returns,
        prices=prices,
        ff=ff.reindex(calendar),
        benchmark=benchmark,
        eligible=elig,
        universe=universe.loc[keep],
    )
    return panel, report


def run_stage1(cfg: Config, refresh: bool = False) -> Panel:
    panel, report = build_panel(cfg, refresh=refresh)
    panel.save(cfg.paths.data_processed)
    (cfg.paths.reports / "data_quality.md").write_text(quality_markdown(report), encoding="utf-8")
    report.summary.to_csv(cfg.paths.reports / "data_quality_summary.csv")
    n_elig = panel.eligible.sum(axis=1)
    log.info(
        "Panel: %d days (%s -> %s), %d tickers, eligible per month-end: min %d / median %d / max %d",
        len(panel.returns), panel.returns.index.min().date(), panel.returns.index.max().date(),
        panel.returns.shape[1], n_elig[n_elig > 0].min(), int(n_elig.median()), n_elig.max(),
    )
    return panel
