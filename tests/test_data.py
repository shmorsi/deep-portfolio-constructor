"""Stage 1: data cleaning, quality checks and point-in-time eligibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dpc.config import QualityConfig
from dpc.data.download import _parse_french_csv
from dpc.data.panel import clean_returns, eligibility, month_end_dates
from dpc.data.quality import check_quality
from dpc.data.universe import load_universe

QC = QualityConfig(outlier_abs_daily_return=0.4, split_day_abs_return_max=0.25, max_missing_frac=0.02, max_stale_run_days=5)


def _cal(n: int = 300) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _long(cal: pd.DatetimeIndex, adj: np.ndarray, close: np.ndarray, splits: np.ndarray, ticker: str = "X") -> pd.DataFrame:
    return pd.DataFrame({"date": cal, "ticker": ticker, "Adj Close": adj, "Close": close, "Stock Splits": splits})


# ----------------------------------------------------------------- cleaning --
def test_clean_returns_bridges_gap_and_preserves_total_return() -> None:
    cal = _cal(6)
    px = pd.DataFrame({"A": [100.0, 101.0, np.nan, np.nan, 110.0, 111.0]}, index=cal)
    r = clean_returns(px, max_fill=5)["A"]
    # total compounded return equals price ratio: nothing lost in the gap
    assert np.isclose((1 + r.dropna()).prod(), 111.0 / 100.0)
    assert r.iloc[2] == 0.0 and r.iloc[3] == 0.0
    assert np.isclose(r.iloc[4], 110 / 101 - 1)


def test_clean_returns_invents_nothing_outside_listing() -> None:
    cal = _cal(6)
    px = pd.DataFrame({"A": [np.nan, np.nan, 50.0, 51.0, 52.0, np.nan]}, index=cal)
    r = clean_returns(px, max_fill=5)["A"]
    assert r.iloc[:3].isna().all()  # before listing (and first day has no return)
    assert np.isnan(r.iloc[5])  # after delisting: no forward-filled zero return


def test_month_end_dates() -> None:
    cal = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29", "2024-03-01"])
    assert list(month_end_dates(cal)) == list(pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-01"]))


def test_eligibility_requires_history_and_is_point_in_time() -> None:
    cal = _cal(400)
    rets = pd.DataFrame({"OLD": 0.001, "NEW": np.nan}, index=cal)
    rets.loc[cal[200]:, "NEW"] = 0.001
    me = month_end_dates(cal)
    el = eligibility(rets, me, min_history=252, max_missing=0)
    assert el["OLD"].iloc[-1] and not el["NEW"].iloc[-1]  # NEW has < 252 days
    # Truncation invariance: deleting the future must not change past decisions
    cut = me[8]
    el_trunc = eligibility(rets.loc[:cut], me[me <= cut], min_history=252, max_missing=0)
    pd.testing.assert_frame_equal(el.loc[:cut], el_trunc)


# ----------------------------------------------------------------- quality ---
def test_quality_passes_clean_split() -> None:
    cal = _cal()
    adj = 100 * np.cumprod(np.full(len(cal), 1.0005))
    splits = np.zeros(len(cal))
    splits[150] = 2.0  # Yahoo already back-adjusted both series -> smooth
    rep = check_quality(_long(cal, adj, adj, splits), cal, QC)
    assert rep.summary.loc["X", "passed"]
    assert (rep.issues["kind"] == "split_verified").sum() == 1


def test_quality_catches_unadjusted_split() -> None:
    cal = _cal()
    adj = 100 * np.cumprod(np.full(len(cal), 1.0005))
    adj[150:] /= 2.0  # a 2:1 split that was NOT back-adjusted looks like -50%
    splits = np.zeros(len(cal))
    splits[150] = 2.0
    rep = check_quality(_long(cal, adj, adj, splits), cal, QC)
    assert not rep.summary.loc["X", "passed"]
    assert "split_inconsistent" in set(rep.issues["kind"])


def test_quality_flags_outliers_stale_and_missing() -> None:
    cal = _cal()
    adj = 100 * np.cumprod(np.full(len(cal), 1.0005))
    adj[100:] *= 1.6  # +60% jump -> outlier (kept)
    adj[200:210] = adj[199]  # 10 identical closes -> stale
    adj[250] = np.nan  # a missing day
    rep = check_quality(_long(cal, adj, adj, np.zeros(len(cal))), cal, QC)
    kinds = set(rep.issues["kind"])
    assert {"outlier_return", "stale_run_days", "missing_price"} <= kinds
    assert not rep.summary.loc["X", "passed"]  # stale run > 5 days fails


# ----------------------------------------------------------------- parsing ---
def test_parse_french_csv() -> None:
    text = (
        "This file was created by CMPT_ME_BEME_RETS using the 202509 CRSP database.\n"
        "\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "20240102,  -1.00,   0.50,  0.10,  0.021\n"
        "20240103,   2.00,  -0.25,  0.00,  0.021\n"
        "\n"
        " Annual Factors: January-December\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "2024,  10.0, 1.0, 1.0, 5.0\n"
    )
    df = _parse_french_csv(text)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 2  # annual block ignored
    assert np.isclose(df.iloc[0]["Mkt-RF"], -0.01)  # percent -> decimal


def test_universe_rejects_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "u.csv"
    p.write_text("ticker,name,country,sector\nA,a,US,Energy\nA,b,US,Energy\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_universe(p)


# ----------------------------------------------------------- real panel ------
def test_real_panel_integrity(panel) -> None:
    r = panel.returns
    assert r.index.is_monotonic_increasing and r.index.is_unique
    assert set(panel.universe.index) == set(r.columns)
    # every eligible (date, stock) has a valid return that day
    el = panel.eligible
    assert not (el & r.reindex(el.index).isna()).any().any()
    # no gaps inside a listing
    inside = r.ffill().notna() & r.bfill().notna()
    assert not (inside & r.isna()).iloc[1:].any().any()
    # factors are complete and in decimals (daily |Mkt-RF| < 25%)
    assert panel.ff.notna().all().all()
    assert panel.ff["Mkt-RF"].abs().max() < 0.25
    assert len(panel.universe) >= 90
