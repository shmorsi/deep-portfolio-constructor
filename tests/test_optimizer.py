"""The shared long-short optimiser satisfies every constraint; allocators behave."""

from __future__ import annotations

import numpy as np
import pytest

from dpc.baselines import hrp_weights, risk_parity_weights
from dpc.optimizer import LongShortOptimizer, OptimizerInputs

N, K, NS, NF = 30, 4, 3, 3
TOL = 1e-5


def _inputs(seed: int, w0_scale: float = 0.0, turnover_limit: float = 2.0, n_blocked: int = 3) -> OptimizerInputs:
    rng = np.random.default_rng(seed)
    ub = np.full(N, 0.10)
    ub[:n_blocked] = 0.0  # not investable
    S = np.zeros((NS, N))
    S[np.arange(N) % NS, np.arange(N)] = 1.0
    w0 = rng.normal(size=N) * w0_scale
    w0[:n_blocked] = 0
    w0 -= w0.mean() * (w0 != 0)
    return OptimizerInputs(
        mu=rng.normal(0, 0.01, N),
        B=rng.normal(0, 0.03, (N, K)),
        s=rng.uniform(0.03, 0.08, N),
        w0=w0,
        ub=ub,
        S=S,
        X=rng.normal(1.0, 0.4, (NF, N)),
        turnover_limit=turnover_limit,
    )


def _opt(cfg) -> LongShortOptimizer:
    return LongShortOptimizer(N, K, NS, NF, cfg.optimizer, cfg.backtest.transaction_cost_bps)


@pytest.mark.parametrize("seed", range(5))
def test_all_constraints_hold(cfg, seed: int) -> None:
    oc = cfg.optimizer
    x = _inputs(seed)
    w = _opt(cfg).solve(x)
    assert abs(w.sum()) < TOL  # dollar neutral
    assert np.all(np.abs(w) <= x.ub + TOL)  # position caps, blocked names at 0
    assert np.all(w[:3] == 0)
    assert np.abs(w).sum() <= oc.gross_leverage + TOL  # leverage
    assert np.all(np.abs(x.S @ w) <= oc.sector_limit + TOL)  # sector neutral
    assert np.all(np.abs(x.X @ w) <= oc.factor_limit + TOL)  # beta/size/value neutral
    assert np.abs(w - x.w0).sum() <= x.turnover_limit + TOL  # turnover


def test_turnover_limit_binds_and_slack_unused_when_feasible(cfg) -> None:
    opt = _opt(cfg)
    x0 = _inputs(0)
    w_prev = opt.solve(x0)
    x1 = _inputs(1)
    x1.w0, x1.turnover_limit = w_prev, 0.3
    w = opt.solve(x1)
    assert np.abs(w - w_prev).sum() <= 0.3 + TOL
    assert opt.slack_used[-1] < 1e-6


def test_elastic_turnover_keeps_problem_feasible(cfg) -> None:
    """A starting book that violates neutrality and a tiny turnover limit is still solvable."""
    opt = _opt(cfg)
    x = _inputs(2, w0_scale=0.05, turnover_limit=1e-4)
    w = opt.solve(x)
    assert np.all(np.abs(x.X @ w) <= cfg.optimizer.factor_limit + TOL)  # hard constraints win
    assert opt.slack_used[-1] > 0


def test_risk_aversion_reduces_risk(cfg) -> None:
    import dataclasses

    x = _inputs(3)
    risks = []
    for gamma in (0.5, 50.0):
        oc = dataclasses.replace(cfg.optimizer, risk_aversion=gamma)
        w = LongShortOptimizer(N, K, NS, NF, oc, 10.0).solve(x)
        risks.append(np.sum((x.B.T @ w) ** 2) + np.sum((x.s * w) ** 2))
    assert risks[1] < risks[0]


def test_hrp_on_diagonal_cov_is_inverse_variance() -> None:
    var = np.array([0.01, 0.04, 0.02, 0.09])
    w = hrp_weights(np.diag(var), "single")
    np.testing.assert_allclose(w, (1 / var) / (1 / var).sum(), rtol=1e-6)


def test_risk_parity_equalises_risk_contributions() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    cov = A @ A.T / 8 + np.eye(8) * 0.1
    w = risk_parity_weights(cov)
    rc = w * (cov @ w)
    assert np.allclose(rc / rc.sum(), 1 / 8, atol=1e-4)
    assert np.isclose(w.sum(), 1) and np.all(w > 0)
