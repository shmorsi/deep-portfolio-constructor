"""Stage 4: walk-forward training and out-of-sample evaluation of the deep risk model.

Walk-forward protocol (no look-ahead)
-------------------------------------
* Refit once a year, at the first rebalance date of each year, on an
  expanding window.
* A training sample with origin o has a target window o+1 .. o+H. It may be
  used at refit date R only if o + H <= R, i.e. its label was fully observed
  by R. This is asserted, not assumed.
* Validation = the most recent ``val_fraction`` of eligible origins, with an
  embargo of H days between the last training origin and the first
  validation origin (overlapping targets would otherwise leak).
* The model fitted at R forecasts every month-end from R until the next refit.

Comparison: Gaussian NLL of the next 21 daily returns and the realised
volatility of the global minimum-variance portfolio, against the sample
covariance, Ledoit-Wolf and the trailing-volatility diagonal (the network's
starting point, an ablation).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from dpc.backtest import LookAheadError, PointInTimeView, run_backtest, save_result
from dpc.config import Config, RiskModelConfig
from dpc.covariance import ledoit_wolf_cov, sample_cov
from dpc.data.panel import Panel
from dpc.deep.features import CONTEXT_FACTORS, SampleBank, build_sample_bank
from dpc.deep.risk_model import DeepRiskModel, factor_gaussian_nll, to_return_space
from dpc.logging_utils import get_logger
from dpc.reproducibility import make_torch_generator, set_global_seed

log = get_logger(__name__)

DTYPE = torch.float32


# --------------------------------------------------------------------------- #
# Walk-forward bookkeeping
# --------------------------------------------------------------------------- #
def refit_dates(rebalance: pd.DatetimeIndex, every_months: int) -> pd.DatetimeIndex:
    return rebalance[::every_months]


def model_for_date(refits: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    """The latest refit date <= ``date`` (the only model allowed to forecast at ``date``)."""
    past = refits[refits <= date]
    if len(past) == 0:
        raise LookAheadError(f"No model trained on or before {date.date()}")
    return past[-1]


def train_val_split(bank: SampleBank, refit: pd.Timestamp, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    iR = bank.calendar.get_loc(refit)
    H = bank.horizon
    usable = np.where(bank.origin_idx + H <= iR)[0]
    if len(usable) < 20:
        raise ValueError(f"Too little history before {refit.date()} ({len(usable)} samples)")
    n_val = max(1, int(math.ceil(len(usable) * val_fraction)))
    val = usable[-n_val:]
    first_val_pos = bank.origin_idx[val[0]]
    train = usable[bank.origin_idx[usable] + H <= first_val_pos]  # embargo
    # Explicit look-ahead assertions
    if bank.origin_idx[np.concatenate([train, val])].max() + H > iR:
        raise LookAheadError("training label extends beyond the refit date")
    if bank.origin_idx[train].max() + H > bank.origin_idx[val].min():
        raise LookAheadError("training labels overlap the validation window")
    return train, val


def batch_tensors(bank: SampleBank, idx: np.ndarray, mask: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "x": torch.as_tensor(bank.x[idx], dtype=DTYPE),
        "f": torch.as_tensor(bank.f[idx], dtype=DTYPE),
        "static": torch.as_tensor(bank.static[idx], dtype=DTYPE),
        "mask": torch.as_tensor(mask, dtype=torch.bool),
        "sigma": torch.as_tensor(bank.sigma[idx], dtype=torch.float64),
        "target": torch.as_tensor(bank.target[idx], dtype=torch.float64),
    }


def _nll(model: DeepRiskModel, t: dict[str, torch.Tensor]) -> torch.Tensor:
    B, psi, _ = model(t["x"], t["f"], t["static"], t["mask"])
    return factor_gaussian_nll(B.double(), psi.double(), t["sigma"], t["target"], t["mask"])


def new_model(bank: SampleBank, rc: RiskModelConfig) -> DeepRiskModel:
    return DeepRiskModel(len(CONTEXT_FACTORS), bank.static.shape[-1], rc)


@dataclass
class FitResult:
    model: DeepRiskModel
    best_val: float
    epochs: int
    history: list[tuple[float, float]] = field(default_factory=list)


def fit_risk_model(
    bank: SampleBank, train: np.ndarray, val: np.ndarray, rc: RiskModelConfig, seed: int, stream: int,
    init_state: dict | None = None,
) -> FitResult:
    set_global_seed(seed)
    model = new_model(bank, rc)
    if init_state is not None:  # warm start from the previous refit
        model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=rc.learning_rate, weight_decay=rc.weight_decay)
    gen = make_torch_generator(seed, stream)
    train_mask = bank.mask_in & bank.mask_out
    vt = batch_tensors(bank, val, train_mask[val])
    best, best_state, bad, hist = float("inf"), None, 0, []
    for epoch in range(rc.max_epochs):
        model.train()
        perm = train[torch.randperm(len(train), generator=gen).numpy()]
        losses = []
        for i in range(0, len(perm), rc.batch_size):
            b = perm[i : i + rc.batch_size]
            loss = _nll(model, batch_tensors(bank, b, train_mask[b]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rc.grad_clip)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            v = _nll(model, vt).item()
        hist.append((float(np.mean(losses)), v))
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= rc.early_stopping_patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, best, len(hist), hist)


@torch.no_grad()
def predict(model: DeepRiskModel, bank: SampleBank, idx: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = batch_tensors(bank, idx, mask)
    B, psi, h = model(t["x"], t["f"], t["static"], t["mask"])
    return B.double().numpy(), psi.double().numpy(), h.double().numpy()


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def gaussian_nll_full(cov: np.ndarray, r: np.ndarray) -> float:
    """Mean per-stock-per-day NLL of rows of r (H,n) under N(0, cov)."""
    n = cov.shape[0]
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
    logdet = 2 * np.log(np.diag(L)).sum()
    sol = np.linalg.solve(L, r.T)  # (n,H)
    quad = (sol**2).sum(axis=0)
    return float(0.5 * (n * np.log(2 * np.pi) + logdet + quad).mean() / n)


def gmv_weights(cov: np.ndarray) -> np.ndarray:
    ones = np.ones(cov.shape[0])
    x = np.linalg.solve(cov, ones)
    return x / x.sum()


def factor_cov(B: np.ndarray, psi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    D = np.diag(sigma)
    return D @ (B @ B.T + np.diag(psi)) @ D


# --------------------------------------------------------------------------- #
# Stage runner
# --------------------------------------------------------------------------- #
@dataclass
class RiskForecasts:
    """Out-of-sample monthly-horizon forecasts at each rebalance date."""

    dates: pd.DatetimeIndex
    B: np.ndarray  # (D, N, K) return-space loadings, monthly horizon
    s: np.ndarray  # (D, N) idiosyncratic vol, monthly horizon
    tickers: list[str]

    def save(self, path) -> None:
        np.savez_compressed(path, dates=self.dates.values.astype("datetime64[ns]").astype(np.int64), B=self.B, s=self.s,
                            tickers=np.array(self.tickers))

    @classmethod
    def load(cls, path) -> "RiskForecasts":
        z = np.load(path, allow_pickle=False)
        return cls(pd.DatetimeIndex(z["dates"].astype("datetime64[ns]")), z["B"], z["s"], [str(t) for t in z["tickers"]])

    def at(self, date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        k = self.dates.get_loc(date)
        return self.B[k], self.s[k]


def walk_forward_risk(
    cfg: Config, panel: Panel, bank: SampleBank, rc: RiskModelConfig, seed: int, tag: str, save_models: bool = True
) -> tuple[RiskForecasts, pd.DataFrame, dict[pd.Timestamp, float]]:
    from dpc.backtest import rebalance_dates

    rebal = rebalance_dates(panel, cfg.backtest.oos_start)
    refits = refit_dates(rebal, rc.refit_every_months)
    me_idx = np.array([bank.index_of(d) for d in rebal])
    D, N, K = len(rebal), len(bank.tickers), rc.n_factors
    B_out, s_out = np.zeros((D, N, K)), np.zeros((D, N))
    raw = {"B": np.zeros((D, N, K)), "psi": np.ones((D, N))}
    log_rows, val_nll = [], {}
    prev_state = None
    for j, R in enumerate(refits):
        train, val = train_val_split(bank, R, rc.val_fraction)
        fit = fit_risk_model(bank, train, val, rc, seed, stream=R.year, init_state=prev_state if rc.warm_start else None)
        prev_state = copy.deepcopy(fit.model.state_dict())
        val_nll[R] = fit.best_val
        log.info("[%s] refit %s: %d train / %d val samples, %d epochs, best val NLL %.4f",
                 tag, R.date(), len(train), len(val), fit.epochs, fit.best_val)
        log_rows.append({"refit": R, "n_train": len(train), "n_val": len(val), "epochs": fit.epochs, "val_nll": fit.best_val})
        if save_models:
            torch.save({"state": fit.model.state_dict(), "rc": rc.__dict__}, cfg.paths.models / f"risk_{tag}_{R.date()}.pt")
        nxt = refits[j + 1] if j + 1 < len(refits) else pd.Timestamp.max
        use = np.where((rebal >= R) & (rebal < nxt))[0]
        for u in use:  # point-in-time check: model R forecasts only dates >= R
            assert model_for_date(refits, rebal[u]) == R
        idx = me_idx[use]
        elig = panel.eligible.loc[rebal[use]].to_numpy()
        mask = bank.mask_in[idx] & elig
        Bz, psi, _ = predict(fit.model, bank, idx, mask)
        Bh, sh = to_return_space(torch.as_tensor(Bz), torch.as_tensor(psi), torch.as_tensor(bank.sigma[idx]), rc.horizon_days)
        B_out[use], s_out[use] = Bh.numpy() * mask[..., None], sh.numpy() * mask
        raw["B"][use], raw["psi"][use] = Bz, psi
    fc = RiskForecasts(rebal, B_out, s_out, bank.tickers)
    fc.raw = raw  # type: ignore[attr-defined]  (standardised outputs, for evaluation)
    return fc, pd.DataFrame(log_rows), val_nll


def evaluate_risk(cfg: Config, panel: Panel, bank: SampleBank, fc: RiskForecasts) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample NLL and GMV realised vol for deep vs classical covariances."""
    win = cfg.backtest.cov_window_days
    rows, gmv = [], {k: [] for k in ("sample", "ledoit_wolf", "trailing_vol_diag", "deep")}
    for k, d in enumerate(fc.dates):
        i = bank.index_of(d)
        if not bank.mask_out[i].any():
            continue  # last month: no realised target yet
        m = panel.eligible.loc[d].to_numpy() & bank.mask_in[i] & bank.mask_out[i]
        hist = panel.returns.loc[:d].iloc[-win:].iloc[:, m]
        r = bank.target[i][:, m]  # (H, n) realised next-month daily returns
        covs = {
            "sample": sample_cov(hist, 1),
            "ledoit_wolf": ledoit_wolf_cov(hist, 1),
            "trailing_vol_diag": np.diag(bank.sigma[i][m] ** 2),
            "deep": factor_cov(fc.raw["B"][k][m], fc.raw["psi"][k][m], bank.sigma[i][m]),  # type: ignore[attr-defined]
        }
        row = {"date": d, "n": int(m.sum())}
        for name, c in covs.items():
            row[f"nll_{name}"] = gaussian_nll_full(c, r)
            gmv[name].append(pd.Series(r @ gmv_weights(c), index=bank.calendar[bank.origin_idx[i] + 1 : bank.origin_idx[i] + 1 + bank.horizon]))
        rows.append(row)
    per_month = pd.DataFrame(rows).set_index("date")
    ann = np.sqrt(cfg.backtest.annualization_days)
    summary = pd.DataFrame(
        {
            name: {
                "mean_oos_nll": per_month[f"nll_{name}"].mean(),
                "median_oos_nll": per_month[f"nll_{name}"].median(),
                "share_months_best_nll": float((per_month[[c for c in per_month if c.startswith("nll_")]].idxmin(axis=1) == f"nll_{name}").mean()),
                "gmv_realised_vol_ann": float(pd.concat(gmv[name]).std() * ann),
            }
            for name in gmv
        }
    ).T
    return summary, per_month


def make_ls_deep_twostage(cfg: Config, fc: RiskForecasts, n_sectors: int):
    """Grinold alpha (same as ls_classical) + deep covariance -> shared optimiser."""
    from dpc.baselines import embed, grinold_mu, long_short_inputs
    from dpc.optimizer import LongShortOptimizer

    n = len(fc.tickers)
    opt = LongShortOptimizer(n, fc.B.shape[-1], n_sectors, len(cfg.optimizer.neutral_factors), cfg.optimizer, cfg.backtest.transaction_cost_bps)

    def fn(view: PointInTimeView, w0: pd.Series) -> pd.Series:
        tick = view.investable
        idx = np.array([view.tickers.index(t) for t in tick])
        B, s = fc.at(view.date)
        keep = np.zeros(n, bool)
        keep[idx] = True
        mu = embed(grinold_mu(view, tick, cfg.alpha.assumed_ic, cfg.backtest.cov_window_days), idx, n)
        x = long_short_inputs(view, w0, mu, B * keep[:, None], s * keep, cfg)
        return pd.Series(opt.solve(x), index=view.tickers)

    return fn, opt


def run_stage4(cfg: Config) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    from dpc.baselines import get_betas, load_signals
    from dpc.plotting import CATEGORICAL, apply_style, save

    panel = Panel.load(cfg.paths.data_processed)
    rc = cfg.risk_model
    bank = build_sample_bank(panel, rc.seq_len_days, rc.horizon_days, rc.sample_stride_days)
    log.info("Sample bank: %d origins, %d stocks, seq %d, horizon %d", len(bank.origins), len(bank.tickers), rc.seq_len_days, rc.horizon_days)
    fc, fit_log, _ = walk_forward_risk(cfg, panel, bank, rc, cfg.project.seed, tag="main")
    fc.save(cfg.paths.data_processed / "deep_risk_forecasts.npz")
    np.savez_compressed(cfg.paths.data_processed / "deep_risk_raw.npz", **fc.raw)  # type: ignore[attr-defined]
    fit_log.to_csv(cfg.paths.reports / "stage4_refits.csv", index=False)
    summary, per_month = evaluate_risk(cfg, panel, bank, fc)
    summary.to_csv(cfg.paths.reports / "stage4_risk_model.csv")
    per_month.to_csv(cfg.paths.reports / "stage4_risk_model_monthly.csv")
    log.info("Stage 4 out-of-sample risk-model comparison:\n%s", summary.round(4).to_string())

    # Two-stage deep strategy (deep covariance, classical alpha)
    signals, betas = load_signals(cfg), get_betas(cfg, panel)
    fn, opt = make_ls_deep_twostage(cfg, fc, panel.universe["sector"].nunique())
    res = run_backtest("ls_deep_twostage", panel, fn, signals, betas, cfg.backtest.oos_start, cfg.backtest.transaction_cost_bps, False)
    res.info["turnover_slack_months"] = int(np.sum(np.array(opt.slack_used) > 1e-6))
    save_result(res, cfg.paths.results)

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"sample": CATEGORICAL[1], "ledoit_wolf": CATEGORICAL[2], "trailing_vol_diag": CATEGORICAL[3], "deep": CATEGORICAL[0]}
    for name, col in colors.items():
        s = per_month[f"nll_{name}"].rolling(12).mean()
        ax.plot(s.index, s.values, color=col, label=name, lw=2 if name == "deep" else 1.5)
    ax.set_title("Out-of-sample covariance NLL (12-month rolling mean, lower is better)")
    ax.set_ylabel("NLL per stock-day")
    ax.legend(loc="upper right", ncols=2)
    save(fig, cfg.paths.figures / "stage4_oos_nll.png")
    return summary
