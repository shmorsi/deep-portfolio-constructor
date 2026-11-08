"""The long-short portfolio optimiser shared by every market-neutral strategy.

The SAME convex problem is used by the classical two-stage strategy (solved
with cvxpy), the deep two-stage strategy and the end-to-end model (wrapped in
a cvxpylayers ``CvxpyLayer`` so gradients flow through it). Only the inputs
(expected returns ``mu`` and risk model ``B, s``) differ between strategies,
which makes the comparison fair.

    maximise    mu'w  - gamma * ( ||B'w||^2 + ||s * w||^2 )          risk: Sigma = BB' + diag(s^2)
                      - kappa * ||w - w0||_1                            trading-cost penalty
                      - M * u                                           elastic turnover slack
    subject to  sum(w) = 0                                              dollar neutral
                |w_i| <= ub_i                                           position cap (0 if not investable)
                ||w||_1 <= gross                                        leverage
                |S'w| <= sector_limit                                   sector neutral (+-5%)
                |X'w| <= factor_limit                                   beta / size / value neutral
                ||w - w0||_1 <= turnover_limit + u,  u >= 0             turnover limit

The turnover constraint is *elastic*: the slack ``u`` carries a large
penalty, so the limit binds whenever it can, but the problem is always
feasible (after a month of price drift, the old portfolio may be further than
``turnover_limit`` from the neutral set). Every slack use is logged.
DPP-compliant, so cvxpylayers can differentiate it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from dpc.config import OptimizerConfig

# Penalty per unit of turnover beyond the limit. Large relative to any
# monthly alpha (~1e-2) so the limit binds whenever feasible.
TURNOVER_SLACK_PENALTY = 10.0


@dataclass
class LongShortProblem:
    problem: cp.Problem
    w: cp.Variable
    slack: cp.Variable
    mu: cp.Parameter
    B: cp.Parameter
    s: cp.Parameter
    w0: cp.Parameter
    ub: cp.Parameter
    S: cp.Parameter
    X: cp.Parameter
    turnover_limit: cp.Parameter

    @property
    def parameters(self) -> list[cp.Parameter]:
        """Order used by the cvxpylayers layer."""
        return [self.mu, self.B, self.s, self.w0, self.ub, self.S, self.X, self.turnover_limit]


def build_long_short_problem(
    n_assets: int, n_factors: int, n_sectors: int, n_neutral: int, oc: OptimizerConfig, cost_bps: float
) -> LongShortProblem:
    mu = cp.Parameter(n_assets, name="mu")
    B = cp.Parameter((n_assets, n_factors), name="B")
    s = cp.Parameter(n_assets, nonneg=True, name="s")
    w0 = cp.Parameter(n_assets, name="w0")
    ub = cp.Parameter(n_assets, nonneg=True, name="ub")
    S = cp.Parameter((n_sectors, n_assets), name="S")
    X = cp.Parameter((n_neutral, n_assets), name="X")
    tl = cp.Parameter(nonneg=True, name="turnover_limit")
    w = cp.Variable(n_assets, name="w")
    u = cp.Variable(nonneg=True, name="turnover_slack")

    kappa = oc.turnover_penalty_mult * cost_bps / 1e4
    trade = cp.norm1(w - w0)
    risk = cp.sum_squares(B.T @ w) + cp.sum_squares(cp.multiply(s, w))
    objective = cp.Maximize(mu @ w - oc.risk_aversion * risk - kappa * trade - TURNOVER_SLACK_PENALTY * u)
    constraints = [
        cp.sum(w) == 0,
        cp.abs(w) <= ub,
        cp.norm1(w) <= oc.gross_leverage,
        cp.abs(S @ w) <= oc.sector_limit,
        cp.abs(X @ w) <= oc.factor_limit,
        trade <= tl + u,
    ]
    prob = cp.Problem(objective, constraints)
    if not prob.is_dpp():  # pragma: no cover - structural guarantee
        raise RuntimeError("Long-short problem is not DPP; cvxpylayers cannot differentiate it.")
    return LongShortProblem(prob, w, u, mu, B, s, w0, ub, S, X, tl)


@dataclass
class OptimizerInputs:
    """Numpy inputs for one rebalance (all arrays over the full ticker list)."""

    mu: np.ndarray  # (N,) expected monthly return
    B: np.ndarray  # (N, K) risk factor loadings, Sigma = BB' + diag(s^2)
    s: np.ndarray  # (N,) idiosyncratic vol
    w0: np.ndarray  # (N,) current (drifted) weights
    ub: np.ndarray  # (N,) position cap, 0 for non-investable
    S: np.ndarray  # (n_sectors, N) sector membership
    X: np.ndarray  # (n_neutral, N) factor betas to neutralise
    turnover_limit: float


class LongShortOptimizer:
    """Solves the problem with cvxpy for one rebalance at a time (classical use)."""

    def __init__(self, n_assets: int, n_factors: int, n_sectors: int, n_neutral: int, oc: OptimizerConfig, cost_bps: float):
        self.oc = oc
        self.lsp = build_long_short_problem(n_assets, n_factors, n_sectors, n_neutral, oc, cost_bps)
        self.slack_used: list[float] = []

    def solve(self, x: OptimizerInputs) -> np.ndarray:
        p = self.lsp
        p.mu.value, p.B.value, p.s.value = x.mu, x.B, x.s
        p.w0.value, p.ub.value, p.S.value, p.X.value = x.w0, x.ub, x.S, x.X
        p.turnover_limit.value = x.turnover_limit
        p.problem.solve(solver=cp.CLARABEL)
        if p.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or p.w.value is None:
            raise RuntimeError(f"Optimiser failed: {p.problem.status}")
        self.slack_used.append(float(p.slack.value))
        w = np.asarray(p.w.value, dtype=float)
        w[np.abs(w) < 1e-9] = 0.0
        return w
