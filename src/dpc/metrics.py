"""Performance metrics. All inputs are *daily excess* returns unless noted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dpc.stats import factor_regression

ALPHA_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def to_monthly(daily: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compound daily returns into calendar-month returns."""
    out = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    out.index = out.index.to_timestamp(how="end").normalize()
    return out


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def performance_metrics(
    excess: pd.Series,
    ff: pd.DataFrame,
    turnover: pd.Series | None,
    costs: pd.Series | None,
    ann_days: int,
    nw_lags: int,
) -> dict[str, float]:
    r = excess.dropna()
    if len(r) < 2:
        return {}
    years = len(r) / ann_days
    ann_ret = float((1 + r).prod() ** (1 / years) - 1)
    ann_vol = float(r.std() * np.sqrt(ann_days))
    downside = float(np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(ann_days))
    mean_ann = float(r.mean() * ann_days)
    monthly = to_monthly(r)
    ff_m = to_monthly(ff[ALPHA_FACTORS].reindex(r.index))
    reg = factor_regression(monthly, ff_m, nw_lags)
    out = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": mean_ann / ann_vol if ann_vol > 0 else np.nan,
        "sortino": mean_ann / downside if downside > 0 else np.nan,
        "max_drawdown": max_drawdown(r),
        "hit_rate_monthly": float((monthly > 0).mean()),
        "alpha_ann_ff6": reg["alpha"] * 12,
        "alpha_t_ff6": reg["alpha_t"],
        "beta_mkt": reg["beta_Mkt-RF"],
        "r2_ff6": reg["r2"],
    }
    if turnover is not None and len(turnover):
        t = turnover.iloc[1:] if len(turnover) > 1 else turnover  # exclude initial build
        out["turnover_ann"] = float(t.mean() * 12)
    if costs is not None and len(costs):
        out["cost_drag_ann"] = float(costs.sum() / years)
    return out


def metrics_table(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows).T
