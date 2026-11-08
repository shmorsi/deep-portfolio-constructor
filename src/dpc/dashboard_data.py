"""Data access for the Streamlit dashboard (kept separate so it is testable)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dpc.backtest import BacktestResult
from dpc.config import Config
from dpc.data.panel import Panel
from dpc.evaluation import LABELS, load_all_results
from dpc.exposures import load_betas


@dataclass
class DashboardData:
    cfg: Config
    panel: Panel
    results: dict[str, BacktestResult]
    betas: dict[str, pd.DataFrame]
    performance: pd.DataFrame | None
    decomposition: pd.DataFrame | None

    def risk_decomp(self, strategy: str) -> pd.DataFrame | None:
        p = self.cfg.paths.results / f"{strategy}__risk_decomp.parquet"
        return pd.read_parquet(p) if p.exists() else None

    def weights_at(self, strategy: str, date: pd.Timestamp) -> pd.DataFrame:
        w = self.results[strategy].weights.loc[date]
        w = w[w.abs() > self.cfg.backtest.min_weight_print]
        df = pd.DataFrame({"weight": w})
        uni = self.panel.universe
        df["name"] = uni.loc[df.index, "name"]
        df["sector"] = uni.loc[df.index, "sector"]
        df["country"] = uni.loc[df.index, "country"]
        return df.sort_values("weight", ascending=False)

    def sector_exposure(self, strategy: str, date: pd.Timestamp) -> pd.Series:
        w = self.results[strategy].weights.loc[date]
        return w.groupby(self.panel.universe["sector"]).sum().sort_values()

    def factor_exposure(self, strategy: str, date: pd.Timestamp) -> pd.Series:
        w = self.results[strategy].weights.loc[date]
        return pd.Series({f: float((self.betas[f].loc[date].fillna(0) * w).sum()) for f in self.cfg.optimizer.neutral_factors})

    def cumulative(self, strategies: list[str]) -> pd.DataFrame:
        rf = self.panel.ff["RF"]
        return pd.DataFrame({LABELS.get(s, s): (1 + self.results[s].excess_returns(rf)).cumprod() for s in strategies})


def load_dashboard_data(cfg: Config) -> DashboardData:
    panel = Panel.load(cfg.paths.data_processed)
    rep = cfg.paths.reports
    perf = pd.read_csv(rep / "stage6_performance.csv", index_col=0) if (rep / "stage6_performance.csv").exists() else None
    dec = pd.read_csv(rep / "stage6_risk_decomposition.csv", index_col=0) if (rep / "stage6_risk_decomposition.csv").exists() else None
    return DashboardData(cfg, panel, load_all_results(cfg), load_betas(cfg.paths.data_processed), perf, dec)


__all__ = ["DashboardData", "load_dashboard_data", "np"]
