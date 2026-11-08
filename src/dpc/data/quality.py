"""Data-quality checks on raw Yahoo prices.

What we check, and why it matters for a backtest:

* **Missing values** after a stock lists: gaps silently turn into zero returns
  or lost returns depending on how you compute them.
* **Stock splits**: an unadjusted 2-for-1 split looks like a -50% crash. Note
  that Yahoo's ``Close`` is *already split-adjusted* (only dividends are left
  out), and ``Adj Close`` adjusts for both. So on every recorded split date we
  verify that both series are smooth; a failed back-adjustment would show up as
  a move of roughly ``1/ratio - 1`` in either series.
* **Outliers**: daily moves above a threshold are flagged for review. For
  large caps most are genuine (e.g. Citigroup +58% on 24 Nov 2008), so we do
  *not* clip P&L returns; robust scaling is applied only to model inputs.
* **Stale prices**: runs of identical closes suggest a dead or broken feed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dpc.config import QualityConfig


@dataclass
class QualityReport:
    summary: pd.DataFrame  # one row per ticker
    issues: pd.DataFrame  # one row per flagged (ticker, date, kind)

    @property
    def failed_tickers(self) -> list[str]:
        return sorted(self.summary.index[~self.summary["passed"]])


def _max_run(mask: pd.Series) -> int:
    """Length of the longest run of True values."""
    if not mask.any():
        return 0
    groups = (~mask).cumsum()
    return int(mask.groupby(groups).sum().max())


def check_quality(long: pd.DataFrame, calendar: pd.DatetimeIndex, qc: QualityConfig) -> QualityReport:
    """Run all checks. ``long`` has columns date, ticker, Adj Close, Close, Stock Splits.

    ``Close`` is Yahoo's split-adjusted (not dividend-adjusted) close.
    """
    summaries: list[dict] = []
    issues: list[tuple] = []
    for ticker, g in long.groupby("ticker", sort=True):
        g = g.set_index("date").sort_index()
        g = g[~g.index.duplicated(keep="last")]
        listed = g["Adj Close"].dropna()
        first, last = listed.index.min(), listed.index.max()
        cal = calendar[(calendar >= first) & (calendar <= last)]
        adj = g["Adj Close"].reindex(cal)
        raw = g["Close"].reindex(cal)
        missing = adj.isna()
        adj_ret = adj.ffill().pct_change(fill_method=None)
        raw_ret = raw.ffill().pct_change(fill_method=None)

        # Outliers
        out = adj_ret[adj_ret.abs() > qc.outlier_abs_daily_return]
        issues += [(ticker, d, "outlier_return", float(v)) for d, v in out.items()]

        # Stale prices: identical raw closes on consecutive trading days
        stale_run = _max_run((raw.diff() == 0) & raw.notna())
        if stale_run > qc.max_stale_run_days:
            issues.append((ticker, last, "stale_run_days", float(stale_run)))

        # Splits: both Yahoo series must be back-adjusted (smooth) on the split day
        splits = g["Stock Splits"].reindex(cal).fillna(0.0)
        split_ok = True
        for d, ratio in splits[splits > 0].items():
            a = adj_ret.get(d, np.nan)
            r = raw_ret.get(d, np.nan)
            smooth_adj = np.isfinite(a) and abs(a) < qc.split_day_abs_return_max
            smooth_close = np.isfinite(r) and abs(r) < qc.split_day_abs_return_max
            if not (smooth_adj and smooth_close):
                split_ok = False
                issues.append((ticker, d, "split_inconsistent", float(ratio)))
            else:
                issues.append((ticker, d, "split_verified", float(ratio)))

        missing_frac = float(missing.mean())
        if missing.any():
            issues += [(ticker, d, "missing_price", np.nan) for d in missing[missing].index]
        passed = split_ok and missing_frac <= qc.max_missing_frac and stale_run <= qc.max_stale_run_days
        summaries.append(
            {
                "ticker": ticker,
                "first_date": first.date(),
                "last_date": last.date(),
                "n_days": int(len(cal)),
                "missing_frac": missing_frac,
                "max_stale_run": stale_run,
                "n_outliers": int(len(out)),
                "n_splits": int((splits > 0).sum()),
                "max_abs_return": float(adj_ret.abs().max()),
                "passed": passed,
            }
        )
    summary = pd.DataFrame(summaries).set_index("ticker")
    issues_df = pd.DataFrame(issues, columns=["ticker", "date", "kind", "value"])
    return QualityReport(summary=summary, issues=issues_df)


def quality_markdown(report: QualityReport) -> str:
    """Human-readable quality report for ``reports/data_quality.md``."""
    s, iss = report.summary, report.issues
    counts = iss["kind"].value_counts()
    lines = [
        "# Data quality report",
        "",
        f"* Tickers checked: **{len(s)}**, passed: **{int(s['passed'].sum())}**, "
        f"failed: **{len(report.failed_tickers)}** {report.failed_tickers or ''}",
        f"* Splits verified: **{int(counts.get('split_verified', 0))}**, inconsistent: "
        f"**{int(counts.get('split_inconsistent', 0))}**",
        f"* Missing price-days after listing: **{int(counts.get('missing_price', 0))}**",
        f"* Outlier daily returns flagged (kept, not clipped): **{int(counts.get('outlier_return', 0))}**",
        f"* Stale-price runs flagged: **{int(counts.get('stale_run_days', 0))}**",
        "",
        "## Late listings (enter the investable set after their first year of data)",
        "",
    ]
    first_cal = s["first_date"].min()
    late = s[s["first_date"] > first_cal].sort_values("first_date")
    lines += [f"* {t}: {d}" for t, d in late["first_date"].items()] or ["* none"]
    lines += ["", "## Flagged outliers", "", "| ticker | date | return |", "|---|---|---|"]
    out = iss[iss["kind"] == "outlier_return"].sort_values("date")
    lines += [f"| {r.ticker} | {pd.Timestamp(r.date).date()} | {r.value:+.1%} |" for r in out.itertuples()]
    lines += ["", "## Split events", "", "| ticker | date | ratio | status |", "|---|---|---|---|"]
    sp = iss[iss["kind"].str.startswith("split")].sort_values(["ticker", "date"])
    lines += [f"| {r.ticker} | {pd.Timestamp(r.date).date()} | {r.value:g} | {r.kind} |" for r in sp.itertuples()]
    lines += ["", "## Per-ticker summary", "", s.to_markdown(floatfmt=".4f")]
    return "\n".join(lines) + "\n"
