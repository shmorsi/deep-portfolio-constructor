"""Data access for the Streamlit dashboard (kept separate so it is testable).

The dashboard runs straight from a fresh clone: everything it needs is
committed under ``reports/`` (backtest results, evaluation tables) plus a small
``reports/dashboard_bundle/`` of DERIVED inputs (factor betas at rebalance
dates, the risk-free rate, sector labels). No price download, no PyTorch.
If the full processed panel exists (after ``python -m dpc.cli data``), the
bundle can be regenerated from it with ``export_bundle``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dpc.backtest import BacktestResult
from dpc.config import Config
from dpc.strategies import LABELS, load_all_results

BUNDLE_DIR = "dashboard_bundle"


@dataclass
class DashboardData:
    cfg: Config
    universe: pd.DataFrame  # ticker -> name, country, sector
    rf: pd.Series  # daily risk-free rate
    betas: dict[str, pd.DataFrame]  # neutralised factor -> (rebalance date x ticker)
    results: dict[str, BacktestResult]
    performance: pd.DataFrame | None
    decomposition: pd.DataFrame | None

    def risk_decomp(self, strategy: str) -> pd.DataFrame | None:
        p = self.cfg.paths.results / f"{strategy}__risk_decomp.parquet"
        return pd.read_parquet(p) if p.exists() else None

    def weights_at(self, strategy: str, date: pd.Timestamp) -> pd.DataFrame:
        w = self.results[strategy].weights.loc[date]
        w = w[w.abs() > self.cfg.backtest.min_weight_print]
        df = pd.DataFrame({"weight": w})
        df["name"] = self.universe.loc[df.index, "name"]
        df["sector"] = self.universe.loc[df.index, "sector"]
        df["country"] = self.universe.loc[df.index, "country"]
        return df.sort_values("weight", ascending=False)

    def sector_exposure(self, strategy: str, date: pd.Timestamp) -> pd.Series:
        w = self.results[strategy].weights.loc[date]
        return w.groupby(self.universe["sector"].reindex(w.index)).sum().sort_values()

    def factor_exposure(self, strategy: str, date: pd.Timestamp) -> pd.Series:
        w = self.results[strategy].weights.loc[date]
        return pd.Series({f: float((self.betas[f].loc[date].reindex(w.index).fillna(0) * w).sum())
                          for f in self.cfg.optimizer.neutral_factors})

    def cumulative(self, strategies: list[str]) -> pd.DataFrame:
        return pd.DataFrame({LABELS.get(s, s): (1 + self.results[s].excess_returns(self.rf)).cumprod() for s in strategies})


def export_bundle(cfg: Config) -> Path:
    """Write the dashboard's derived inputs from the full processed panel."""
    from dpc.data.panel import Panel
    from dpc.exposures import load_betas

    panel = Panel.load(cfg.paths.data_processed)
    out = cfg.paths.reports / BUNDLE_DIR
    out.mkdir(parents=True, exist_ok=True)
    rebal = panel.month_ends[panel.month_ends >= pd.Timestamp(cfg.backtest.oos_start)]
    betas = load_betas(cfg.paths.data_processed)
    for f in cfg.optimizer.neutral_factors:
        betas[f].reindex(rebal).round(6).to_parquet(out / f"beta_{f}.parquet")
    panel.ff["RF"].loc[rebal[0]:].to_frame("RF").to_parquet(out / "rf.parquet")
    panel.universe.to_parquet(out / "universe.parquet")
    return out


def load_dashboard_data(cfg: Config) -> DashboardData:
    rep = cfg.paths.reports
    b = rep / BUNDLE_DIR
    if not (b / "rf.parquet").exists():
        export_bundle(cfg)  # needs the processed panel; the committed bundle makes this unnecessary
    perf = pd.read_csv(rep / "stage6_performance.csv", index_col=0) if (rep / "stage6_performance.csv").exists() else None
    dec = pd.read_csv(rep / "stage6_risk_decomposition.csv", index_col=0) if (rep / "stage6_risk_decomposition.csv").exists() else None
    return DashboardData(
        cfg=cfg,
        universe=pd.read_parquet(b / "universe.parquet"),
        rf=pd.read_parquet(b / "rf.parquet")["RF"],
        betas={f: pd.read_parquet(b / f"beta_{f}.parquet") for f in cfg.optimizer.neutral_factors},
        results=load_all_results(cfg.paths.results),
        performance=perf,
        decomposition=dec,
    )
