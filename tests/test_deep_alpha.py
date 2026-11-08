"""AlphaFormer, the shared Trainer, and the first-principles PyTorch optimiser."""

from __future__ import annotations

import dataclasses

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from dpc.backtest import LookAheadError
from dpc.deep.alpha_data import MonthDataset, build_alpha_bank
from dpc.deep.alpha_model import AlphaEnsemble, AlphaFormer, information_coefficient, masked_zscore
from dpc.deep.torch_qp import neutral_mean_variance, scale_to_gross
from dpc.deep.train_alpha import ic_step, usable_rows
from dpc.deep.trainer import fit, resolve_device
from helpers import toy_panel, toy_signals_betas


def _ac(cfg, **kw):
    return dataclasses.replace(cfg.alpha_model, d_model=16, n_heads=4, dropout=0.0, **kw)


# ------------------------------------------------------------------ model ----
def test_alphaformer_shapes_masking_and_permutation_equivariance(cfg) -> None:
    torch.manual_seed(0)
    net = AlphaFormer(2, 7, _ac(cfg)).eval()
    seq, static = torch.randn(2, 9, 12, 2), torch.randn(2, 9, 7)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 3] = False
    s, h = net(seq, static, mask)
    assert s.shape == (2, 9) and h.shape == (2, 9, 16)
    assert s[1, 3] == 0 and torch.all(h[1, 3] == 0)
    perm = torch.randperm(9)
    s2, _ = net(seq[:, perm], static[:, perm], mask[:, perm])
    assert torch.allclose(s[:, perm], s2, atol=1e-5)  # relabelling stocks relabels forecasts


def test_information_coefficient_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(3, 40)), rng.normal(size=(3, 40))
    mask = np.ones((3, 40), bool)
    mask[2, :10] = False
    ic = information_coefficient(torch.tensor(x), torch.tensor(y), torch.tensor(mask)).numpy()
    for r in range(3):
        m = mask[r]
        assert np.isclose(ic[r], np.corrcoef(x[r, m], y[r, m])[0, 1], atol=1e-6)
    assert torch.allclose(information_coefficient(torch.tensor(y), torch.tensor(y), torch.tensor(mask)), torch.ones(3, dtype=torch.float64))


def test_masked_zscore() -> None:
    x = torch.randn(2, 30, dtype=torch.float64) * 5 + 3
    mask = torch.ones(2, 30, dtype=torch.bool)
    mask[0, :5] = False
    z = masked_zscore(x, mask)
    for r in range(2):
        zr = z[r][mask[r]]
        assert abs(zr.mean()) < 1e-10 and abs(zr.std() - 1) < 1e-6
    assert torch.all(z[0, :5] == 0)


def test_ensemble_output_is_standardised(cfg) -> None:
    torch.manual_seed(1)
    ens = AlphaEnsemble([AlphaFormer(2, 5, _ac(cfg)) for _ in range(3)]).eval()
    mask = torch.ones(1, 20, dtype=torch.bool)
    with torch.no_grad():
        s, _ = ens(torch.randn(1, 20, 12, 2), torch.randn(1, 20, 5), mask)
    assert abs(float(s.mean())) < 1e-5 and abs(float(s.std()) - 1) < 1e-4


# ------------------------------------------------------------------- data ----
def test_alpha_bank_is_point_in_time_and_labels_are_honest(cfg) -> None:
    p = toy_panel(n_days=900, n_stocks=24)
    s, b = toy_signals_betas(p)
    ac = _ac(cfg)
    bank = build_alpha_bank(p, s, b, ac)
    assert len(bank.dates) > 0
    T = bank.dates[len(bank.dates) // 2]
    p2 = toy_panel(n_days=900, n_stocks=24)
    p2.returns.loc[p2.returns.index > T] = np.random.default_rng(7).normal(0.02, 0.2, (int((p2.returns.index > T).sum()), 24))
    bank2 = build_alpha_bank(p2, s, b, ac)
    k1, k2 = bank.rows_for(bank.dates[bank.dates <= T]), bank2.rows_for(bank.dates[bank.dates <= T])
    assert np.array_equal(bank.seq[k1], bank2.seq[k2]) and np.array_equal(bank.static[k1], bank2.static[k2])
    assert np.array_equal(bank.mask[k1], bank2.mask[k2])
    # every label ends after its decision date
    assert (bank.label_end > bank.dates).all()


def test_usable_rows_never_include_unfinished_labels(cfg) -> None:
    p = toy_panel(n_days=900, n_stocks=24)
    s, b = toy_signals_betas(p)
    bank = build_alpha_bank(p, s, b, _ac(cfg))
    R = bank.dates[-3]
    rows = usable_rows(bank, R)
    assert (bank.label_end[rows] <= R).all()
    assert R not in set(bank.dates[rows])  # its own label (next month) is not finished yet


# ---------------------------------------------------------------- trainer ----
def _synthetic_bank(cfg, n_months: int = 60, n: int = 30, seed: int = 0):
    """A month bank whose target is a known function of one static feature."""
    from dpc.deep.alpha_data import AlphaBank

    rng = np.random.default_rng(seed)
    static = rng.normal(size=(n_months, n, 5)).astype(np.float32)
    target = (static[..., 0] + 0.5 * rng.normal(size=(n_months, n))).astype(np.float32)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    return AlphaBank(dates, [f"S{i}" for i in range(n)], rng.normal(size=(n_months, n, 12, 2)).astype(np.float32), static,
                     np.ones((n_months, n), bool), target, np.ones((n_months, n), bool), dates + pd.offsets.MonthEnd(1))


def test_trainer_learns_a_learnable_signal_and_is_reproducible(cfg) -> None:
    bank = _synthetic_bank(cfg)
    ac = _ac(cfg)

    def run() -> tuple[float, torch.Tensor]:
        torch.manual_seed(3)
        model = AlphaFormer(2, 5, ac)
        g = torch.Generator().manual_seed(3)
        tl = DataLoader(MonthDataset(bank, np.arange(48)), batch_size=8, shuffle=True, generator=g)
        vl = DataLoader(MonthDataset(bank, np.arange(48, 60)), batch_size=12)
        res = fit(model, tl, vl, ic_step, learning_rate=3e-3, weight_decay=0.0, max_epochs=30, patience=10,
                  grad_clip=1.0, device=resolve_device("cpu"))
        return res.best_metric, torch.cat([p.detach().flatten() for p in res.model.parameters()])

    m1, w1 = run()
    m2, w2 = run()
    assert m1 > 0.6  # true IC of the synthetic signal is about 0.89
    assert torch.equal(w1, w2)  # same seed, same weights, bit for bit


def test_resolve_device() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")


# ------------------------------------------------- first-principles optimiser --
def _problem(seed: int = 0, n: int = 8):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, n))
    sigma = G @ G.T / n + np.eye(n) * 0.1
    A = np.vstack([np.ones(n), rng.normal(1.0, 0.3, n)])  # dollar-neutral + beta-neutral
    mu = rng.normal(0, 0.02, n)
    return mu, sigma, A


def test_kkt_layer_matches_cvxpy_and_satisfies_constraints() -> None:
    mu, sigma, A = _problem()
    gamma = 5.0
    w_t = neutral_mean_variance(torch.tensor(mu), torch.tensor(sigma), torch.tensor(A), gamma).numpy()
    w = cp.Variable(len(mu))
    cp.Problem(cp.Maximize(mu @ w - gamma * cp.quad_form(w, sigma)), [A @ w == 0]).solve()
    assert np.allclose(w_t, w.value, atol=1e-6)
    assert np.allclose(A @ w_t, 0, atol=1e-12)


def test_kkt_layer_gradients_pass_gradcheck() -> None:
    mu, sigma, A = _problem(1)
    mu_t = torch.tensor(mu, requires_grad=True)
    sig_t = torch.tensor(sigma, requires_grad=True)
    f = lambda m, s: neutral_mean_variance(m, 0.5 * (s + s.T), torch.tensor(A), 5.0)  # noqa: E731
    assert torch.autograd.gradcheck(f, (mu_t, sig_t), eps=1e-6, atol=1e-6)


def test_kkt_layer_is_linear_in_mu_and_gross_scaling_keeps_neutrality() -> None:
    mu, sigma, A = _problem(2)
    t = lambda x: torch.tensor(x)  # noqa: E731
    w1 = neutral_mean_variance(t(mu), t(sigma), t(A), 5.0)
    w2 = neutral_mean_variance(t(2 * mu), t(sigma), t(A), 5.0)
    assert torch.allclose(w2, 2 * w1)
    wg = scale_to_gross(w1, 2.0)
    assert torch.isclose(wg.abs().sum(), torch.tensor(2.0, dtype=torch.float64))
    assert torch.allclose(t(A) @ wg, torch.zeros(2, dtype=torch.float64), atol=1e-12)


def test_usable_rows_raises_if_labels_leak(cfg) -> None:
    bank = _synthetic_bank(cfg)
    bad = dataclasses.replace(bank, label_end=bank.dates)  # labels "end" on the decision date
    rows = usable_rows(bad, bank.dates[10])
    assert (bad.label_end[rows] <= bank.dates[10]).all()
    _ = LookAheadError, pytest
