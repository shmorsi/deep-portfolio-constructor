"""One walk-forward backtest engine for every strategy.

Timeline (no look-ahead by construction):

    month-end t, after the close:  strategy sees data dated <= t only
                                   -> target weights; trade at the close of t
                                   -> pay  cost = c * sum|w_target - w_drifted|
    days t+1 ... next month-end:   weights drift with prices (buy-and-hold)

Strategies receive a :class:`PointInTimeView`, which exposes the panel only
up to the decision date, so a strategy *cannot* read the future through it.
The test-suite adds a second guard: truncating the data after date T must
leave every weight decided at or before T unchanged.

Return conventions
------------------
* Long-only strategies (fully invested) earn total returns; metrics subtract
  the risk-free rate.
* Dollar-neutral long-short strategies: the cash collateral earns the
  risk-free rate, so ``sum_i w_i r_i`` *is* the excess return. Metrics use it
  as is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from dpc.data.panel import Panel


class LookAheadError(RuntimeError):
    """Raised when a strategy is handed, or returns, information from the future."""


@dataclass
class PointInTimeView:
    """What a strategy is allowed to see at decision date ``date``."""

    date: pd.Timestamp
    _panel: Panel
    _signals: dict[str, pd.DataFrame]
    _betas: dict[str, pd.DataFrame]

    @property
    def returns(self) -> pd.DataFrame:
        """Daily returns dated <= decision date."""
        return self._panel.returns.loc[: self.date]

    @property
    def rf(self) -> pd.Series:
        return self._panel.ff["RF"].loc[: self.date]

    @property
    def eligible(self) -> pd.Series:
        return self._panel.eligible.loc[self.date]

    @property
    def investable(self) -> list[str]:
        """Eligible stocks that also have every signal and beta defined today."""
        ok = self.eligible.copy()
        for df in (*self._signals.values(), *self._betas.values()):
            ok &= df.loc[self.date].notna()
        return list(ok.index[ok])

    def signal(self, name: str) -> pd.Series:
        return self._signals[name].loc[self.date]

    def beta(self, name: str) -> pd.Series:
        return self._betas[name].loc[self.date]

    @property
    def sectors(self) -> pd.Series:
        return self._panel.universe["sector"]

    @property
    def tickers(self) -> list[str]:
        return self._panel.tickers


WeightFn = Callable[[PointInTimeView, pd.Series], pd.Series]
"""(view, current drifted weights) -> target weights indexed by ticker."""


@dataclass
class BacktestResult:
    name: str
    long_only: bool
    weights: pd.DataFrame  # rebalance date x ticker: post-trade target weights
    daily_gross: pd.Series  # before costs
    daily_net: pd.Series  # after costs
    turnover: pd.Series  # per rebalance: sum |w_target - w_drifted|
    costs: pd.Series  # per rebalance, as a fraction of NAV
    info: dict = field(default_factory=dict)

    def excess_returns(self, rf: pd.Series) -> pd.Series:
        r = self.daily_net
        return r - rf.reindex(r.index).fillna(0.0) if self.long_only else r

    def to_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        daily = pd.DataFrame({"gross": self.daily_gross, "net": self.daily_net})
        reb = pd.DataFrame({"turnover": self.turnover, "cost": self.costs})
        return daily, reb


def rebalance_dates(panel: Panel, start: str) -> pd.DatetimeIndex:
    me = panel.month_ends
    return me[me >= pd.Timestamp(start)]


def run_backtest(
    name: str,
    panel: Panel,
    weight_fn: WeightFn,
    signals: dict[str, pd.DataFrame],
    betas: dict[str, pd.DataFrame],
    start: str,
    cost_bps: float,
    long_only: bool,
    end: pd.Timestamp | None = None,
) -> BacktestResult:
    """Simulate a monthly-rebalanced strategy with daily drift and costs."""
    tickers = panel.tickers
    rets = panel.returns
    rdates = rebalance_dates(panel, start)
    if end is not None:
        rdates = rdates[rdates <= end]
        rets = rets.loc[:end]
    days = rets.index[rets.index >= rdates[0]]
    R = rets.loc[days].to_numpy(dtype=float)
    reb_set = set(rdates)
    c = cost_bps / 1e4

    w = np.zeros(len(tickers))
    gross, net = np.zeros(len(days)), np.zeros(len(days))
    W, turns, costs = {}, {}, {}
    for i, d in enumerate(days):
        if i > 0:  # earn the day's return on yesterday's post-trade weights
            r = np.nan_to_num(R[i], nan=0.0)
            if np.any(w[np.isnan(R[i])] != 0):
                raise LookAheadError(f"{name}: holding a stock with no return on {d.date()}")
            pr = float(w @ r)
            gross[i] = net[i] = pr
            w = w * (1 + r) / (1 + pr)  # buy-and-hold drift
        if d in reb_set:
            view = PointInTimeView(d, panel, signals, betas)
            target = weight_fn(view, pd.Series(w, index=tickers)).reindex(tickers).fillna(0.0)
            tw = target.to_numpy(dtype=float)
            if not np.all(np.isfinite(tw)):
                raise ValueError(f"{name}: non-finite weights on {d.date()}")
            not_listed = ~panel.eligible.loc[d].to_numpy()
            if np.any(np.abs(tw[not_listed]) > 1e-10):
                raise LookAheadError(f"{name}: weight on a non-investable stock on {d.date()}")
            turnover = float(np.abs(tw - w).sum())
            cost = c * turnover
            net[i] -= cost  # trade at today's close; cost hits today's NAV
            W[d], turns[d], costs[d] = tw, turnover, cost
            w = tw
    weights = pd.DataFrame.from_dict(W, orient="index", columns=tickers)
    return BacktestResult(
        name=name,
        long_only=long_only,
        weights=weights,
        daily_gross=pd.Series(gross, index=days),
        daily_net=pd.Series(net, index=days),
        turnover=pd.Series(turns),
        costs=pd.Series(costs),
    )


def save_result(res: BacktestResult, folder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    daily, reb = res.to_frames()
    res.weights.to_parquet(folder / f"{res.name}__weights.parquet")
    daily.to_parquet(folder / f"{res.name}__daily.parquet")
    reb.to_parquet(folder / f"{res.name}__rebal.parquet")
    pd.Series({"long_only": res.long_only, **{k: str(v) for k, v in res.info.items()}}).to_json(
        folder / f"{res.name}__meta.json"
    )


def load_result(name: str, folder) -> BacktestResult:
    import json

    daily = pd.read_parquet(folder / f"{name}__daily.parquet")
    reb = pd.read_parquet(folder / f"{name}__rebal.parquet")
    meta = json.loads((folder / f"{name}__meta.json").read_text())
    return BacktestResult(
        name=name,
        long_only=bool(meta.pop("long_only")),
        weights=pd.read_parquet(folder / f"{name}__weights.parquet"),
        daily_gross=daily["gross"],
        daily_net=daily["net"],
        turnover=reb["turnover"],
        costs=reb["cost"],
        info=meta,
    )
