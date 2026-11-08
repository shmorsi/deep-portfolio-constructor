"""Point-in-time factor exposures (betas) and sector dummies.

We have no free point-in-time market-cap or book-to-market history, so
"size" and "value" exposure is measured the returns-based way: each stock's
rolling regression beta to the Fama-French SMB and HML factor returns (plus
market, profitability, investment and momentum). A portfolio whose weighted
betas are ~0 has, by construction, ~0 sensitivity to those factor returns,
which is exactly what "neutral to market, size and value" should mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dpc.config import ExposuresConfig
from dpc.data.panel import Panel

STYLE_FACTORS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")


def rolling_betas(panel: Panel, ec: ExposuresConfig, factors: tuple[str, ...] = STYLE_FACTORS) -> dict[str, pd.DataFrame]:
    """Multivariate OLS betas of excess stock returns on factors, at each month-end,
    using the trailing ``beta_window_days`` days up to and including that date."""
    excess = panel.returns.sub(panel.ff["RF"], axis=0)
    F = panel.ff[list(factors)]
    out = {f: pd.DataFrame(np.nan, index=panel.month_ends, columns=panel.tickers) for f in factors}
    idx = panel.returns.index
    for t in panel.month_ends:
        end = idx.get_loc(t) + 1
        start = max(0, end - ec.beta_window_days)
        Y = excess.iloc[start:end]
        X = F.iloc[start:end].to_numpy()
        X1 = np.column_stack([np.ones(len(X)), X])
        for tk in Y.columns:
            y = Y[tk].to_numpy()
            ok = np.isfinite(y)
            if ok.sum() < ec.beta_min_obs:
                continue
            coef, *_ = np.linalg.lstsq(X1[ok], y[ok], rcond=None)
            for j, f in enumerate(factors):
                out[f].at[t, tk] = coef[j + 1]
    return out


def save_betas(betas: dict[str, pd.DataFrame], folder) -> None:
    for f, df in betas.items():
        df.to_parquet(folder / f"beta_{f}.parquet")


def load_betas(folder, factors: tuple[str, ...] = STYLE_FACTORS) -> dict[str, pd.DataFrame]:
    return {f: pd.read_parquet(folder / f"beta_{f}.parquet") for f in factors}
