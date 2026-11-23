"""Brinson-Fachler performance attribution, linked over time with Carino's method.

For each period and each sector j, with portfolio and benchmark weights wp, wb, sector returns
rp, rb and the benchmark's total return Rb:

    allocation   (wp - wb)(rb - Rb)      overweighting sectors that beat the benchmark
    selection    wb (rp - rb)            picking better holdings inside a sector
    interaction  (wp - wb)(rp - rb)

The three add up, over sectors, to Rp - Rb exactly. Single-period effects do not add up over
time because returns compound; Carino (1999) scales each period's effects by

    k_t = [ln(1+Rp_t) - ln(1+Rb_t)] / (Rp_t - Rb_t),   K = [ln(1+Rp) - ln(1+Rb)] / (Rp - Rb)

so that the linked effects add up to the cumulative active return Rp - Rb. Where a portfolio
holds nothing in a sector its sector return is undefined; it is set to the benchmark's, which
puts the whole effect in allocation.

The same calculation is written twice: in pandas, and in SQL run by DuckDB, as a performance
reporting team would hold it in a warehouse. The tests check that both agree to 1e-12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EFFECTS = ["allocation", "selection", "interaction"]


def holdings_frame(dates, names, sectors: dict[str, str], wp: np.ndarray, wb: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
    """Long table: one row per (date, asset) with its sector, both weights and its return."""
    T, N = returns.shape
    return pd.DataFrame({
        "date": np.repeat(pd.DatetimeIndex(dates).values, N),
        "asset": np.tile(np.asarray(names), T),
        "sector": np.tile(np.asarray([sectors[n] for n in names]), T),
        "wp": np.asarray(wp, float).ravel(),
        "wb": np.asarray(wb, float).ravel(),
        "r": np.asarray(returns, float).ravel(),
    })


def _sector_table(h: pd.DataFrame) -> pd.DataFrame:
    g = h.assign(wpr=h.wp * h.r, wbr=h.wb * h.r).groupby(["date", "sector"], sort=True)
    s = g[["wp", "wb", "wpr", "wbr"]].sum().reset_index()
    s["rb"] = np.where(s.wb > 0, s.wbr / s.wb.where(s.wb > 0, 1.0), 0.0)
    s["rp"] = np.where(s.wp > 0, s.wpr / s.wp.where(s.wp > 0, 1.0), s.rb)
    tot = s.groupby("date")[["wpr", "wbr"]].sum().rename(columns={"wpr": "Rp", "wbr": "Rb"})
    return s.merge(tot, left_on="date", right_index=True)


def brinson_fachler(h: pd.DataFrame) -> pd.DataFrame:
    """Per (date, sector) effects, plus each date's Rp and Rb."""
    s = _sector_table(h)
    s["allocation"] = (s.wp - s.wb) * (s.rb - s.Rb)
    s["selection"] = s.wb * (s.rp - s.rb)
    s["interaction"] = (s.wp - s.wb) * (s.rp - s.rb)
    return s[["date", "sector", "wp", "wb", "rp", "rb", "Rp", "Rb", *EFFECTS]]


def _carino_k(rp, rb):
    rp, rb = np.asarray(rp, float), np.asarray(rb, float)
    d = rp - rb
    near = np.abs(d) < 1e-12
    safe = np.where(near, 1.0, d)
    return np.where(near, 1.0 / (1.0 + rp), (np.log1p(rp) - np.log1p(rb)) / safe)


def carino_link(effects: pd.DataFrame) -> pd.DataFrame:
    """Linked effects by sector over the whole span; they add up to the cumulative active return."""
    per = effects.groupby("date")[["Rp", "Rb"]].first()
    Rp = float(np.prod(1 + per.Rp) - 1)
    Rb = float(np.prod(1 + per.Rb) - 1)
    K = float(_carino_k(Rp, Rb))
    k = pd.Series(_carino_k(per.Rp, per.Rb), index=per.index, name="k")
    e = effects.merge(k, left_on="date", right_index=True)
    linked = e[EFFECTS].mul(e.k / K, axis=0).groupby(e.sector).sum()
    linked["total"] = linked.sum(axis=1)
    linked.attrs.update({"Rp": Rp, "Rb": Rb, "active": Rp - Rb})
    return linked


SQL = """
WITH sec AS (
    SELECT date, sector,
           SUM(wp) AS wp, SUM(wb) AS wb,
           SUM(wp * r) AS wpr, SUM(wb * r) AS wbr
    FROM holdings
    GROUP BY date, sector
), sec_r AS (
    SELECT *,
           CASE WHEN wb > 0 THEN wbr / wb ELSE 0 END AS rb
    FROM sec
), sec_rp AS (
    SELECT *, CASE WHEN wp > 0 THEN wpr / wp ELSE rb END AS rp FROM sec_r
), tot AS (
    SELECT date, SUM(wpr) AS Rp, SUM(wbr) AS Rb FROM sec GROUP BY date
), eff AS (
    SELECT s.date, s.sector, t.Rp, t.Rb,
           (s.wp - s.wb) * (s.rb - t.Rb) AS allocation,
           s.wb * (s.rp - s.rb)          AS selection,
           (s.wp - s.wb) * (s.rp - s.rb) AS interaction
    FROM sec_rp s JOIN tot t USING (date)
), span AS (
    SELECT EXP(SUM(LN(1 + Rp))) - 1 AS Rp, EXP(SUM(LN(1 + Rb))) - 1 AS Rb FROM tot
), k AS (
    SELECT date,
           CASE WHEN ABS(Rp - Rb) < 1e-12 THEN 1 / (1 + Rp)
                ELSE (LN(1 + Rp) - LN(1 + Rb)) / (Rp - Rb) END AS k
    FROM tot
), big_k AS (
    SELECT CASE WHEN ABS(Rp - Rb) < 1e-12 THEN 1 / (1 + Rp)
                ELSE (LN(1 + Rp) - LN(1 + Rb)) / (Rp - Rb) END AS K
    FROM span
)
SELECT e.sector,
       SUM(e.allocation  * k.k / big_k.K) AS allocation,
       SUM(e.selection   * k.k / big_k.K) AS selection,
       SUM(e.interaction * k.k / big_k.K) AS interaction
FROM eff e JOIN k USING (date) CROSS JOIN big_k
GROUP BY e.sector
ORDER BY e.sector
"""


def carino_link_sql(h: pd.DataFrame) -> pd.DataFrame:
    """The same linked attribution, computed by DuckDB from the long holdings table."""
    import duckdb

    con = duckdb.connect()
    con.register("holdings", h)
    out = con.execute(SQL).df().set_index("sector")
    con.close()
    out["total"] = out[EFFECTS].sum(axis=1)
    return out
