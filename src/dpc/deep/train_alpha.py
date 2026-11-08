"""Stage 4b: walk-forward training of the AlphaFormer ("deep analyst views").

Protocol (identical in spirit to the risk model):
* yearly refits on an expanding window, from the first out-of-sample month;
* a month is a training example only if its label (next month's return)
  ended on or before the refit date -- asserted, not assumed;
* validation = the last ``val_months`` labelled months; early stopping on the
  validation information coefficient (IC);
* an ensemble of seeds, warm-started from the previous year's members;
* the model fitted at refit R scores every month-end from R to the next refit.

Evaluation: out-of-sample rank IC of the deep score versus the three classical
signals and their composite, with Newey-West t-statistics, on the same months.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dpc.backtest import LookAheadError, PointInTimeView, rebalance_dates, run_backtest, save_result
from dpc.config import AlphaModelConfig, Config
from dpc.data.panel import Panel
from dpc.deep.alpha_data import AlphaBank, MonthDataset, build_alpha_bank
from dpc.deep.alpha_model import AlphaEnsemble, AlphaFormer, information_coefficient
from dpc.deep.trainer import fit, resolve_device
from dpc.deep.train_risk import refit_dates
from dpc.logging_utils import get_logger
from dpc.reproducibility import make_torch_generator, set_global_seed

log = get_logger(__name__)

SEQ_CHANNELS = 2
SIGNAL_KEY = "deep_alpha"


def ic_step(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss = negative mean IC over the months in the batch; metric = IC per month."""
    score, _ = model(batch["seq"], batch["static"], batch["mask"])
    ic = information_coefficient(score, batch["target"], batch["mask_y"])
    return -ic.mean(), ic


def new_alpha_model(bank: AlphaBank, ac: AlphaModelConfig) -> AlphaFormer:
    return AlphaFormer(SEQ_CHANNELS, bank.n_static, ac)


def usable_rows(bank: AlphaBank, refit: pd.Timestamp) -> np.ndarray:
    rows = np.where((bank.label_end <= refit) & (bank.mask_y.sum(axis=1) >= 20))[0]
    if len(rows) and bank.label_end[rows].max() > refit:  # explicit look-ahead assertion
        raise LookAheadError("AlphaFormer training label extends beyond the refit date")
    return rows


def fit_member(bank: AlphaBank, train: np.ndarray, val: np.ndarray, ac: AlphaModelConfig, seed: int,
               stream: int, init_state: dict | None, device: torch.device):
    set_global_seed(seed)
    model = new_alpha_model(bank, ac)
    if init_state is not None:
        model.load_state_dict(init_state)
    gen = make_torch_generator(seed, stream)
    train_loader = DataLoader(MonthDataset(bank, train), batch_size=ac.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(MonthDataset(bank, val), batch_size=len(val), shuffle=False)
    return fit(model, train_loader, val_loader, ic_step, learning_rate=ac.learning_rate,
               weight_decay=ac.weight_decay, max_epochs=ac.max_epochs, patience=ac.early_stopping_patience,
               grad_clip=ac.grad_clip, device=device)


@torch.no_grad()
def predict(model: AlphaEnsemble, bank: AlphaBank, rows: np.ndarray, device: torch.device) -> np.ndarray:
    ds = MonthDataset(bank, rows, with_target=False)
    batch = {k: torch.stack([ds[i][k] for i in range(len(ds))]).to(device) for k in ("seq", "static", "mask")}
    score, _ = model.to(device)(batch["seq"], batch["static"], batch["mask"])
    out = score.cpu().numpy().astype(float)
    out[~bank.mask[rows]] = np.nan
    return out


def walk_forward_alpha(cfg: Config, panel: Panel, bank: AlphaBank, ac: AlphaModelConfig, seed: int, tag: str,
                       save_models: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = resolve_device(cfg.project.device)
    rebal = rebalance_dates(panel, cfg.backtest.oos_start)
    refits = refit_dates(rebal, ac.refit_every_months)
    scores = pd.DataFrame(np.nan, index=rebal, columns=bank.tickers)
    prev: list[dict | None] = [None] * ac.ensemble_size
    rows_log = []
    for j, R in enumerate(refits):
        usable = usable_rows(bank, R)
        val, train = usable[-ac.val_months:], usable[: -ac.val_months]
        members, metrics = [], []
        for e in range(ac.ensemble_size):
            member_seed = int(np.random.SeedSequence([seed, e]).generate_state(1)[0] % (2**31))
            res = fit_member(bank, train, val, ac, member_seed, stream=R.year, device=device,
                             init_state=prev[e] if ac.warm_start else None)
            members.append(res.model)
            metrics.append(res.best_metric)
            prev[e] = copy.deepcopy(res.model.state_dict())
        ens = AlphaEnsemble(members).eval()
        if save_models:
            torch.save({"states": [m.state_dict() for m in members]}, cfg.paths.models / f"alpha_{tag}_{R.date()}.pt")
        nxt = refits[j + 1] if j + 1 < len(refits) else pd.Timestamp.max
        use = rebal[(rebal >= R) & (rebal < nxt)]
        rows = bank.rows_for(use)
        ok = rows >= 0
        scores.loc[use[ok]] = predict(ens, bank, rows[ok], device)
        rows_log.append({"refit": R, "n_train": len(train), "n_val": len(val), "val_ic_members": np.round(metrics, 4).tolist(),
                         "val_ic_mean": float(np.mean(metrics))})
        log.info("[%s] AlphaFormer refit %s: %d train / %d val months, val IC per member %s",
                 tag, R.date(), len(train), len(val), np.round(metrics, 3).tolist())
    return scores, pd.DataFrame(rows_log)


def load_alpha_ensemble(cfg: Config, bank: AlphaBank, refit: pd.Timestamp, tag: str = "main") -> AlphaEnsemble:
    ck = torch.load(cfg.paths.models / f"alpha_{tag}_{refit.date()}.pt", weights_only=False)
    members = []
    for state in ck["states"]:
        m = new_alpha_model(bank, cfg.alpha_model)
        m.load_state_dict(state)
        members.append(m)
    return AlphaEnsemble(members).eval()


def make_ls_deep_alpha(cfg: Config, fc, n_sectors: int):
    """Deep views (AlphaFormer, via Grinold's IC x vol x score) + deep risk -> shared optimiser."""
    from dpc.baselines import embed, grinold_mu, long_short_inputs
    from dpc.optimizer import LongShortOptimizer

    n = len(fc.tickers)
    opt = LongShortOptimizer(n, fc.B.shape[-1], n_sectors, len(cfg.optimizer.neutral_factors), cfg.optimizer,
                             cfg.backtest.transaction_cost_bps)

    def fn(view: PointInTimeView, w0: pd.Series) -> pd.Series:
        tick = view.investable
        idx = np.array([view.tickers.index(t) for t in tick])
        B, s = fc.at(view.date)
        keep = np.zeros(n, bool)
        keep[idx] = True
        mu = embed(grinold_mu(view, tick, cfg.alpha.assumed_ic, cfg.backtest.cov_window_days, signal=SIGNAL_KEY), idx, n)
        x = long_short_inputs(view, w0, mu, B * keep[:, None], s * keep, cfg)
        return pd.Series(opt.solve(x), index=view.tickers)

    return fn, opt


def run_stage4b(cfg: Config) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    from dpc.baselines import get_betas, load_signals
    from dpc.deep.train_risk import RiskForecasts
    from dpc.plotting import BASELINE, CATEGORICAL, apply_style, save
    from dpc.signals import forward_returns, ic_table, information_coefficients, monthly_returns

    panel = Panel.load(cfg.paths.data_processed)
    signals, betas = load_signals(cfg), get_betas(cfg, panel)
    ac = cfg.alpha_model
    bank = build_alpha_bank(panel, signals, betas, ac)
    log.info("AlphaFormer bank: %d months x %d stocks, %d static features, device %s",
             len(bank.dates), len(bank.tickers), bank.n_static, resolve_device(cfg.project.device))
    scores, refit_log = walk_forward_alpha(cfg, panel, bank, ac, cfg.project.seed, tag="main")
    scores.to_parquet(cfg.paths.data_processed / f"signal_{SIGNAL_KEY}.parquet")
    refit_log.to_csv(cfg.paths.reports / "stage4b_alpha_refits.csv", index=False)

    # Out-of-sample IC: deep views vs classical signals, same months
    fwd = forward_returns(monthly_returns(panel.returns, panel.month_ends))
    oos = scores.index[scores.notna().any(axis=1)]
    compare = {k: signals[k].reindex(oos) for k in ("momentum", "reversal", "low_vol", "combined")}
    compare[SIGNAL_KEY] = scores.reindex(oos)
    table = ic_table(compare, fwd.reindex(oos), cfg.signals.newey_west_lags)
    table.to_csv(cfg.paths.reports / "stage4b_alpha_ic.csv")
    log.info("Out-of-sample IC, deep views vs classical:\n%s", table.round(4).to_string())

    # Two-stage strategy: deep views + deep risk
    fc = RiskForecasts.load(cfg.paths.data_processed / "deep_risk_forecasts.npz")
    sig_all = {**signals, SIGNAL_KEY: scores.reindex(panel.month_ends)}
    fn, opt = make_ls_deep_alpha(cfg, fc, panel.universe["sector"].nunique())
    res = run_backtest("ls_deep_alpha", panel, fn, sig_all, betas, cfg.backtest.oos_start, cfg.backtest.transaction_cost_bps, False)
    res.info["turnover_slack_months"] = int(np.sum(np.array(opt.slack_used) > 1e-6))
    save_result(res, cfg.paths.results)

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = [SIGNAL_KEY, "combined", "momentum", "low_vol"]
    colors = {SIGNAL_KEY: CATEGORICAL[0], "combined": CATEGORICAL[2], "momentum": CATEGORICAL[1], "low_vol": CATEGORICAL[3]}
    for k in order:
        ic = information_coefficients(compare[k], fwd.reindex(oos)).cumsum()
        ax.plot(ic.index, ic.values, color=colors[k], lw=2.2 if k == SIGNAL_KEY else 1.4,
                label="AlphaFormer (deep)" if k == SIGNAL_KEY else k)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_title("Out-of-sample cumulative rank IC: deep views vs classical signals")
    ax.set_ylabel("Sum of monthly rank IC")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, cfg.paths.figures / "stage4b_alpha_ic.png")
    return table
