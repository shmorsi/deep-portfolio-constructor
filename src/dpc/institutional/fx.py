"""How much of a US equity holding should a Canadian investor hedge back to Canadian dollars?

In Canadian dollars, an unhedged US holding earns (1 + r_usd)(1 + s) - 1, where s is the change
in Canadian dollars per US dollar. A forward hedge on a fraction h of the position removes h s
(and adds the interest-rate differential, which is carry, not risk, and is ignored here). The
hedge ratio that minimises the variance of the Canadian-dollar return is

    h* = cov(r_unhedged, s) / var(s).

For a Canadian investor the US dollar has tended to rise when stocks fall (the Canadian dollar
is a commodity, risk-on currency), so leaving the dollar unhedged has *lowered* risk. That is the
known result this measures on the data; it is why many Canadian plans hedge US equity only partly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cad_returns(r_usd: pd.Series, fx_level: pd.Series) -> pd.DataFrame:
    s = fx_level.pct_change()
    df = pd.concat([r_usd.rename("r_usd"), s.rename("s")], axis=1).dropna()
    df["unhedged"] = (1 + df.r_usd) * (1 + df.s) - 1
    return df


def hedged(df: pd.DataFrame, h: float) -> pd.Series:
    return df.unhedged - h * df.s


def min_variance_ratio(df: pd.DataFrame) -> float:
    c = np.cov(df.unhedged, df.s, ddof=1)
    return float(c[0, 1] / c[1, 1])


def rolling_ratio(df: pd.DataFrame, window: int = 120) -> pd.Series:
    cov = df.unhedged.rolling(window).cov(df.s)
    return (cov / df.s.rolling(window).var()).dropna()


def table(df: pd.DataFrame, ratios=(0.0, 0.5, 1.0)) -> list[dict]:
    hs = list(ratios) + [min_variance_ratio(df)]
    rows = []
    for h in hs:
        r = hedged(df, h)
        dd = (1 + r).cumprod()
        rows.append({"hedge_ratio": float(h), "vol_ann": float(r.std(ddof=1) * np.sqrt(12)),
                     "worst_month": float(r.min()), "max_drawdown": float((dd / dd.cummax() - 1).min()),
                     "corr_with_usd_equity": float(np.corrcoef(r, df.r_usd)[0, 1])})
    return rows
