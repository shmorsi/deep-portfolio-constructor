"""A differentiable portfolio optimiser from first principles, in pure PyTorch.

The production model uses cvxpylayers because the real problem has
inequalities (position caps, leverage, turnover, neutrality bands). But the
core idea fits in a few lines when every constraint is an equality:

    maximise   mu'w - gamma * w' Sigma w
    subject to A w = 0            (dollar-, sector-, beta-neutral: rows of A)

The optimum satisfies the KKT conditions, which here are LINEAR:

    [ 2 gamma Sigma   A' ] [ w      ]   [ mu ]
    [ A               0  ] [ lambda ] = [ 0  ]

So the optimal portfolio is w*(mu, Sigma) = (first block of) K^{-1} [mu; 0].
Because ``torch.linalg.solve`` is differentiable, autograd gives dw*/dmu and
dw*/dSigma for free. This is the implicit function theorem in action: the
gradient of a solution is the solution of a linear system with the same
matrix K. cvxpylayers does exactly this for the general (inequality) case,
by differentiating the KKT conditions of a conic program.

Two facts this makes visible:
* w* is linear in mu: doubling every expected return doubles the bets.
* Every row of A is satisfied exactly, so the portfolio cannot carry the
  exposures A rules out, however mu is learned.
"""

from __future__ import annotations

import torch


def neutral_mean_variance(mu: torch.Tensor, sigma: torch.Tensor, A: torch.Tensor, gamma: float) -> torch.Tensor:
    """Solve the equality-constrained mean-variance problem; fully differentiable.

    Shapes: mu (..., N), sigma (..., N, N) positive definite, A (..., m, N) full row rank.
    Returns w (..., N) with A w = 0.
    """
    n, m = mu.shape[-1], A.shape[-2]
    top = torch.cat([2.0 * gamma * sigma, A.transpose(-1, -2)], dim=-1)  # (..., N, N+m)
    bottom = torch.cat([A, torch.zeros(*A.shape[:-2], m, m, dtype=A.dtype, device=A.device)], dim=-1)
    K = torch.cat([top, bottom], dim=-2)  # (..., N+m, N+m)
    rhs = torch.cat([mu, torch.zeros(*mu.shape[:-1], m, dtype=mu.dtype, device=mu.device)], dim=-1)
    sol = torch.linalg.solve(K, rhs.unsqueeze(-1)).squeeze(-1)
    return sol[..., :n]


def scale_to_gross(w: torch.Tensor, gross: float, eps: float = 1e-12) -> torch.Tensor:
    """Rescale so that sum |w| = gross (keeps every equality constraint, since A(cw) = c Aw = 0)."""
    return w * (gross / (w.abs().sum(-1, keepdim=True) + eps))
