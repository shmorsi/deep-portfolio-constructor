"""Stages 4-5: the deep covariance model and the differentiable optimiser layer."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
from cvxpylayers.torch import CvxpyLayer

from dpc.backtest import LookAheadError
from dpc.deep.e2e import AlphaNet, month_utility
from dpc.deep.features import build_sample_bank
from dpc.deep.risk_model import DeepRiskModel, factor_gaussian_nll, to_return_space
from dpc.deep.train_risk import factor_cov, gaussian_nll_full, model_for_date, train_val_split
from dpc.optimizer import build_long_short_problem
from helpers import toy_panel


def _small_rc(cfg, **kw):
    return dataclasses.replace(cfg.risk_model, seq_len_days=20, patch_len_days=5, d_model=16, n_heads=4, n_factors=3, dropout=0.0, **kw)


def test_woodbury_nll_matches_dense_gaussian() -> None:
    rng = np.random.default_rng(0)
    b, N, K, H = 2, 7, 3, 5
    B = torch.as_tensor(rng.normal(0, 0.5, (b, N, K)))
    psi = torch.as_tensor(rng.uniform(0.2, 1.5, (b, N)))
    sig = torch.as_tensor(rng.uniform(0.01, 0.03, (b, N)))
    tgt = torch.as_tensor(rng.normal(0, 0.02, (b, H, N)))
    mask = torch.ones(b, N, dtype=torch.bool)
    mask[1, :2] = False  # masked stocks must contribute nothing
    got = factor_gaussian_nll(B * mask.unsqueeze(-1), torch.where(mask, psi, 1.0), sig, tgt, mask).item()
    tot, cnt = 0.0, 0
    for i in range(b):
        m = mask[i].numpy()
        cov = factor_cov(B[i].numpy()[m], psi[i].numpy()[m], sig[i].numpy()[m])
        tot += gaussian_nll_full(cov, tgt[i].numpy()[:, m]) * m.sum() * H
        cnt += m.sum() * H
    assert np.isclose(got, tot / cnt, rtol=1e-8)


def test_model_starts_at_trailing_vol_prior_and_is_positive_definite(cfg) -> None:
    rc = _small_rc(cfg)
    net = DeepRiskModel(6, 4, rc).eval()
    x, f = torch.randn(2, 20, 9), torch.randn(2, 20, 6)
    st, mask = torch.randn(2, 9, 4), torch.ones(2, 9, dtype=torch.bool)
    B, psi, _ = net(x, f, st, mask)
    assert B.abs().max() < 0.05  # ~no correlation at initialisation
    assert torch.allclose(psi, torch.ones_like(psi), atol=1e-4)  # idio variance = 1 (in vol units)
    Bh, sh = to_return_space(B.double(), psi.double(), torch.full((2, 9), 0.02, dtype=torch.float64), 21)
    cov = Bh[0] @ Bh[0].T + torch.diag(sh[0] ** 2)
    assert torch.linalg.eigvalsh(cov).min() > 0


def test_cross_sectional_attention_is_permutation_equivariant(cfg) -> None:
    """Relabelling the stocks must just relabel the forecasts (no order dependence)."""
    torch.manual_seed(0)
    net = DeepRiskModel(6, 4, _small_rc(cfg)).eval()
    for p in net.head_b.parameters():
        torch.nn.init.normal_(p, std=0.3)
    x, f, st = torch.randn(1, 20, 9), torch.randn(1, 20, 6), torch.randn(1, 9, 4)
    mask = torch.ones(1, 9, dtype=torch.bool)
    mask[0, 4] = False
    perm = torch.randperm(9)
    B1, psi1, _ = net(x, f, st, mask)
    B2, psi2, _ = net(x[:, :, perm], f, st[:, perm], mask[:, perm])
    assert torch.allclose(B1[:, perm], B2, atol=1e-5)
    assert torch.allclose(psi1[:, perm], psi2, atol=1e-5)


def test_training_split_never_sees_labels_after_refit(cfg) -> None:
    p = toy_panel(n_days=700)
    rc = _small_rc(cfg)
    bank = build_sample_bank(p, rc.seq_len_days, rc.horizon_days, rc.sample_stride_days)
    R = p.month_ends[-6]
    train, val = train_val_split(bank, R, 0.2)
    iR = bank.calendar.get_loc(R)
    assert bank.origin_idx[np.r_[train, val]].max() + bank.horizon <= iR
    assert bank.origin_idx[train].max() + bank.horizon <= bank.origin_idx[val].min()  # embargo


def test_sample_bank_is_point_in_time(cfg) -> None:
    p = toy_panel(n_days=500)
    rc = _small_rc(cfg)
    a = build_sample_bank(p, rc.seq_len_days, rc.horizon_days, rc.sample_stride_days)
    T = p.returns.index[350]
    p.returns.loc[p.returns.index > T] = 0.5
    b = build_sample_bank(p, rc.seq_len_days, rc.horizon_days, rc.sample_stride_days)
    k = a.origins <= T
    assert np.array_equal(a.x[k], b.x[k]) and np.array_equal(a.static[k], b.static[k])


def test_model_for_date() -> None:
    refits = pd.DatetimeIndex(["2010-01-29", "2011-01-31"])
    assert model_for_date(refits, pd.Timestamp("2010-06-30")) == refits[0]
    with pytest.raises(LookAheadError):
        model_for_date(refits, pd.Timestamp("2009-12-31"))


def test_alpha_net_starts_at_classical_composite() -> None:
    net = AlphaNet(d_embed=8, hidden=16, residual_scale=1.0).double()
    z, zc, h = torch.randn(3, 10, 3, dtype=torch.float64), torch.randn(3, 10, dtype=torch.float64), torch.randn(3, 10, 8, dtype=torch.float64)
    assert torch.equal(net(z, zc, h), zc)


def test_month_utility() -> None:
    w = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    w0 = torch.zeros_like(w)
    r = torch.tensor([[[0.02, 0.0], [0.0, -0.02]]], dtype=torch.float64)
    u = month_utility(w, w0, r, cost=0.001, lam=10.0)
    # P&L 0.01 + 0.01, cost 0.001 * 1.0, variance penalty 10 * (0.01^2 * 2)
    assert torch.isclose(u, torch.tensor([0.02 - 0.001 - 10 * 2e-4], dtype=torch.float64)).all()


def test_differentiable_optimizer_layer_constraints_and_gradients(cfg) -> None:
    N, K, NS, NF = 12, 2, 3, 3
    oc = cfg.optimizer
    lsp = build_long_short_problem(N, K, NS, NF, oc, 10.0)
    layer = CvxpyLayer(lsp.problem, parameters=lsp.parameters, variables=[lsp.w])
    g = torch.Generator().manual_seed(0)
    d = torch.float64
    mu = (torch.randn(1, N, generator=g, dtype=d) * 0.02).requires_grad_()
    B = torch.randn(1, N, K, generator=g, dtype=d) * 0.03
    s = torch.full((1, N), 0.05, dtype=d)
    w0 = torch.zeros(1, N, dtype=d)
    ub = torch.full((1, N), 0.2, dtype=d)
    S = torch.zeros(1, NS, N, dtype=d)
    S[0, torch.arange(N) % NS, torch.arange(N)] = 1
    X = torch.randn(1, NF, N, generator=g, dtype=d) * 0.3 + 1
    tl = torch.full((1,), 2.0, dtype=d)
    (w,) = layer(mu, B, s, w0, ub, S, X, tl, solver_args={"eps": 1e-8})
    tol = 1e-4
    assert abs(w.sum().item()) < tol
    assert (w.abs() <= 0.2 + tol).all()
    assert (torch.einsum("sn,n->s", S[0], w[0]).abs() <= oc.sector_limit + tol).all()
    assert (torch.einsum("fn,n->f", X[0], w[0]).abs() <= oc.factor_limit + tol).all()
    # Gradient check: autograd through the KKT system vs central finite differences
    c = torch.randn(N, generator=g, dtype=d)
    (w[0] @ c).backward()
    direction = torch.randn(1, N, generator=g, dtype=d)
    eps = 1e-4

    def f(m: torch.Tensor) -> float:
        with torch.no_grad():
            (ww,) = layer(m, B, s, w0, ub, S, X, tl, solver_args={"eps": 1e-10})
        return float(ww[0] @ c)

    fd = (f(mu.detach() + eps * direction) - f(mu.detach() - eps * direction)) / (2 * eps)
    ad = float((mu.grad * direction).sum())
    assert mu.grad.abs().sum() > 0
    assert abs(ad - fd) <= 1e-2 * max(1.0, abs(fd))
