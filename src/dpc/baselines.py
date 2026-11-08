"""Stage 3: classical baselines on the shared walk-forward engine.

Long-only, fully invested (the classic allocation methods):
    equal_weight, min_variance (sample cov), mean_variance_lw,
    risk_parity (equal risk contribution, LW cov), hrp (Lopez de Prado).

Market-neutral long-short (the fair comparison for the deep models):
    ls_naive       signal-weighted, dollar-neutral, no risk model, no constraints
    ls_classical   the shared constrained optimiser with Ledoit-Wolf covariance
                   and Grinold expected returns ("predict-then-optimise")
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from dpc.backtest import BacktestResult, PointInTimeView, WeightFn, run_backtest, save_result
from dpc.config import Config
from dpc.covariance import DAYS_PER_MONTH, ledoit_wolf_cov, sample_cov
from dpc.data.panel import Panel
from dpc.exposures import STYLE_FACTORS, load_betas, rolling_betas, save_betas
from dpc.logging_utils import get_logger
from dpc.metrics import metrics_table, performance_metrics
from dpc.optimizer import LongShortOptimizer, OptimizerInputs
from dpc.signals import SIGNAL_NAMES

log = get_logger(__name__)

from dpc.strategies import LONG_ONLY  # noqa: E402

LONG_SHORT_CLASSICAL = ["ls_naive", "ls_classical"]


# --------------------------------------------------------------------------- #
# Shared inputs
# --------------------------------------------------------------------------- #
def window_returns(view: PointInTimeView, tickers: list[str], window: int) -> pd.DataFrame:
    return view.returns[tickers].iloc[-window:]


def grinold_mu(view: PointInTimeView, tickers: list[str], ic: float, vol_window: int, signal: str = "combined") -> np.ndarray:
    """alpha_i = IC * sigma_i * z_i (monthly). Converts a score into a return forecast."""
    vol = view.returns[tickers].iloc[-vol_window:].std().to_numpy() * np.sqrt(DAYS_PER_MONTH)
    z = view.signal(signal)[tickers].to_numpy()
    return ic * vol * z


def sector_matrix(view: PointInTimeView) -> tuple[np.ndarray, list[str]]:
    sec = view.sectors.reindex(view.tickers)
    names = sorted(sec.unique())
    S = np.stack([(sec == s).to_numpy(dtype=float) for s in names])
    return S, names


def neutral_matrix(view: PointInTimeView, factors: list[str]) -> np.ndarray:
    return np.stack([view.beta(f).reindex(view.tickers).fillna(0.0).to_numpy() for f in factors])


def embed(sub: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    """Place a (k,) or (k,k) array for the investable subset into full-universe shape."""
    if sub.ndim == 1:
        out = np.zeros(n)
        out[idx] = sub
    else:
        out = np.zeros((n, n))
        out[np.ix_(idx, idx)] = sub
    return out


def long_short_inputs(
    view: PointInTimeView, w0: pd.Series, mu_full: np.ndarray, B: np.ndarray, s: np.ndarray, cfg: Config
) -> OptimizerInputs:
    oc = cfg.optimizer
    inv = set(view.investable)
    ub = np.array([oc.max_weight if t in inv else 0.0 for t in view.tickers])
    S, _ = sector_matrix(view)
    X = neutral_matrix(view, list(oc.neutral_factors))
    w0v = w0.reindex(view.tickers).fillna(0.0).to_numpy()
    first = np.abs(w0v).sum() < 1e-12
    tl = oc.gross_leverage if first else oc.turnover_limit  # initial build is exempt
    return OptimizerInputs(mu=mu_full, B=B, s=s, w0=w0v, ub=ub, S=S, X=X, turnover_limit=tl)


# --------------------------------------------------------------------------- #
# Long-only allocators
# --------------------------------------------------------------------------- #
def _long_only_qp(cov: np.ndarray, mu: np.ndarray | None, gamma: float, cap: float) -> np.ndarray:
    n = cov.shape[0]
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
    w = cp.Variable(n)
    risk = cp.sum_squares(L.T @ w)
    obj = cp.Minimize(risk) if mu is None else cp.Maximize(mu @ w - gamma * risk)
    cp.Problem(obj, [cp.sum(w) == 1, w >= 0, w <= cap]).solve(solver=cp.CLARABEL)
    out = np.clip(np.asarray(w.value).ravel(), 0, None)
    return out / out.sum()


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Equal risk contribution via the convex Spinu (2013) formulation:
    min 0.5 y'Sigma y - (1/n) sum log y ;  w = y / sum(y)."""
    n = cov.shape[0]
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
    y = cp.Variable(n, pos=True)
    cp.Problem(cp.Minimize(0.5 * cp.sum_squares(L.T @ y) - cp.sum(cp.log(y)) / n)).solve(solver=cp.CLARABEL)
    yv = np.asarray(y.value).ravel()
    return yv / yv.sum()


def hrp_weights(cov: np.ndarray, method: str) -> np.ndarray:
    """Hierarchical risk parity (Lopez de Prado, 2016).

    1. cluster stocks by correlation distance  d = sqrt((1 - rho) / 2)
    2. order them so similar stocks sit next to each other (quasi-diagonalise)
    3. split the ordered list in halves recursively, allocating between the
       halves in inverse proportion to each half's (inverse-variance) risk.
    No matrix inversion, so it is robust to a noisy covariance.
    """
    sd = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.clip(np.nan_to_num(cov / np.outer(sd, sd), nan=0.0), -1, 1)  # zero-variance -> uncorrelated
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), method=method))
    w = np.ones(len(cov))

    def cluster_var(items: np.ndarray) -> float:
        c = cov[np.ix_(items, items)]
        ivp = 1 / np.diag(c)
        ivp /= ivp.sum()
        return float(ivp @ c @ ivp)

    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        left, right = items[: len(items) // 2], items[len(items) // 2 :]
        vl, vr = cluster_var(left), cluster_var(right)
        alpha = 1 - vl / (vl + vr)
        w[left] *= alpha
        w[right] *= 1 - alpha
        stack += [left, right]
    return w / w.sum()


def make_long_only(name: str, cfg: Config) -> WeightFn:
    bc, cap = cfg.backtest, cfg.baselines.long_only_max_weight

    def fn(view: PointInTimeView, w0: pd.Series) -> pd.Series:
        tick = view.investable
        if name == "equal_weight":
            return pd.Series(1 / len(tick), index=tick)
        rw = window_returns(view, tick, bc.cov_window_days)
        if name == "min_variance":
            w = _long_only_qp(sample_cov(rw), None, 0.0, cap)
        elif name == "mean_variance_lw":
            mu = grinold_mu(view, tick, cfg.alpha.assumed_ic, bc.cov_window_days)
            w = _long_only_qp(ledoit_wolf_cov(rw), mu, cfg.optimizer.risk_aversion, cap)
        elif name == "risk_parity":
            w = risk_parity_weights(ledoit_wolf_cov(rw))
        elif name == "hrp":
            w = hrp_weights(sample_cov(rw), cfg.baselines.hrp_linkage)
        else:  # pragma: no cover
            raise ValueError(name)
        return pd.Series(w, index=tick)

    return fn


# --------------------------------------------------------------------------- #
# Long-short
# --------------------------------------------------------------------------- #
def make_ls_naive(cfg: Config) -> WeightFn:
    gross = cfg.optimizer.gross_leverage

    def fn(view: PointInTimeView, w0: pd.Series) -> pd.Series:
        z = view.signal("combined")[view.investable]
        z = z - z.mean()
        return z / z.abs().sum() * gross

    return fn


def make_ls_classical(cfg: Config, n_assets: int, n_sectors: int) -> tuple[WeightFn, LongShortOptimizer]:
    """Grinold alpha + Ledoit-Wolf covariance into the shared optimiser."""
    oc, bc = cfg.optimizer, cfg.backtest
    opt = LongShortOptimizer(n_assets, n_assets, n_sectors, len(oc.neutral_factors), oc, bc.transaction_cost_bps)

    def fn(view: PointInTimeView, w0: pd.Series) -> pd.Series:
        tick = view.investable
        idx = np.array([view.tickers.index(t) for t in tick])
        rw = window_returns(view, tick, bc.cov_window_days)
        L = np.linalg.cholesky(ledoit_wolf_cov(rw) + 1e-10 * np.eye(len(tick)))
        B = embed(L, idx, n_assets)
        mu = embed(grinold_mu(view, tick, cfg.alpha.assumed_ic, bc.cov_window_days), idx, n_assets)
        x = long_short_inputs(view, w0, mu, B, np.zeros(n_assets), cfg)
        return pd.Series(opt.solve(x), index=view.tickers)

    return fn, opt


# --------------------------------------------------------------------------- #
# Stage runner
# --------------------------------------------------------------------------- #
def load_signals(cfg: Config) -> dict[str, pd.DataFrame]:
    d = cfg.paths.data_processed
    return {k: pd.read_parquet(d / f"signal_{k}.parquet") for k in (*SIGNAL_NAMES, "combined")}


def get_betas(cfg: Config, panel: Panel) -> dict[str, pd.DataFrame]:
    d = cfg.paths.data_processed
    try:
        return load_betas(d)
    except FileNotFoundError:
        log.info("Estimating rolling factor betas (%d-day window)...", cfg.exposures.beta_window_days)
        betas = rolling_betas(panel, cfg.exposures)
        save_betas(betas, d)
        return betas


def summarize(results: list[BacktestResult], panel: Panel, cfg: Config) -> pd.DataFrame:
    rows = {}
    for r in results:
        rows[r.name] = performance_metrics(
            r.excess_returns(panel.ff["RF"]), panel.ff, r.turnover, r.costs,
            cfg.backtest.annualization_days, cfg.signals.newey_west_lags,
        )
    return metrics_table(rows)


def run_stage3(cfg: Config) -> pd.DataFrame:
    panel = Panel.load(cfg.paths.data_processed)
    signals = load_signals(cfg)
    betas = get_betas(cfg, panel)
    bc = cfg.backtest
    n_sectors = panel.universe["sector"].nunique()
    results = []
    for name in LONG_ONLY:
        log.info("Backtesting %s ...", name)
        res = run_backtest(name, panel, make_long_only(name, cfg), signals, betas, bc.oos_start, bc.transaction_cost_bps, True)
        results.append(res)
    log.info("Backtesting ls_naive ...")
    results.append(run_backtest("ls_naive", panel, make_ls_naive(cfg), signals, betas, bc.oos_start, bc.transaction_cost_bps, False))
    log.info("Backtesting ls_classical ...")
    fn, opt = make_ls_classical(cfg, len(panel.tickers), n_sectors)
    res = run_backtest("ls_classical", panel, fn, signals, betas, bc.oos_start, bc.transaction_cost_bps, False)
    res.info["turnover_slack_months"] = int(np.sum(np.array(opt.slack_used) > 1e-6))
    results.append(res)
    for r in results:
        save_result(r, cfg.paths.results)
    table = summarize(results, panel, cfg)
    table.to_csv(cfg.paths.reports / "stage3_baselines.csv")
    log.info("Stage 3 results (net of %g bps):\n%s", bc.transaction_cost_bps, table.round(3).to_string())
    return table


__all__ = ["run_stage3", "STYLE_FACTORS", "LONG_ONLY", "LONG_SHORT_CLASSICAL"]
