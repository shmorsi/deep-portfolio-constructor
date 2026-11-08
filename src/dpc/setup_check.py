"""Stage 0 smoke test: is the research toolchain installed and reproducible?

Run with::

    python -m dpc.setup_check

It (1) loads and validates ``config.yaml``, (2) creates the folder layout,
(3) seeds everything, (4) checks that every library imports, (5) solves a tiny
*differentiable* portfolio problem with cvxpylayers and compares its gradient
to the closed-form answer, and (6) writes a run manifest to ``reports/``.

The cvxpylayers check is not a toy for its own sake: Stage 5 trains a network
*through* an optimizer, which only works if the optimizer's gradients are
right. We verify that here, on a problem whose answer we know by hand.
"""

from __future__ import annotations

import sys

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

from dpc.config import Config, ensure_directories, load_config
from dpc.logging_utils import get_logger, setup_logging
from dpc.reproducibility import package_versions, run_manifest, set_global_seed, write_run_manifest

log = get_logger(__name__)

# Size of the toy problem in the gradient check (not a research setting).
TOY_N_ASSETS = 5
# Tolerance for comparing the solver's Jacobian with the exact one. The
# default conic solver (SCS) is first-order, so we allow ~1e-4 slack.
JACOBIAN_ATOL = 1e-4


def dollar_neutral_projection_layer(n_assets: int) -> CvxpyLayer:
    """A differentiable layer that maps raw scores ``s`` to weights ``w``.

    Solves::

        minimise    ||w - s||^2
        subject to  sum(w) = 0          (dollar-neutral: longs fund shorts)

    The exact solution is ``w = s - mean(s)``, so dw/ds = I - (1/n) * 11'.
    This is the simplest member of the family of problems used in Stage 5.
    """
    s = cp.Parameter(n_assets)
    w = cp.Variable(n_assets)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w - s)), [cp.sum(w) == 0])
    assert problem.is_dpp(), "Problem must follow DPP rules to be differentiable."
    return CvxpyLayer(problem, parameters=[s], variables=[w])


def check_differentiable_optimizer(seed: int, n_assets: int = TOY_N_ASSETS) -> float:
    """Return the max abs error between cvxpylayers' Jacobian and the exact one."""
    layer = dollar_neutral_projection_layer(n_assets)
    gen = torch.Generator().manual_seed(seed)
    scores = torch.randn(n_assets, dtype=torch.float64, generator=gen)

    def forward(x: torch.Tensor) -> torch.Tensor:
        (w,) = layer(x)
        return w

    jac = torch.autograd.functional.jacobian(forward, scores)
    exact = torch.eye(n_assets, dtype=torch.float64) - torch.full(
        (n_assets, n_assets), 1.0 / n_assets, dtype=torch.float64
    )
    weights = forward(scores).detach()
    if abs(float(weights.sum())) > JACOBIAN_ATOL:
        raise RuntimeError(f"Dollar-neutral constraint violated: sum(w) = {float(weights.sum()):.2e}")
    return float((jac - exact).abs().max())


def main(cfg: Config | None = None) -> int:
    cfg = cfg or load_config()
    setup_logging(cfg.logging.level)
    set_global_seed(cfg.project.seed, cfg.project.deterministic_torch, cfg.project.torch_num_threads)
    ensure_directories(cfg)
    log.info("Config loaded from %s (seed=%d)", cfg.root / "config.yaml", cfg.project.seed)

    versions = package_versions()
    missing = [name for name, v in versions.items() if v is None]
    for name, v in versions.items():
        log.info("  %-13s %s", name, v or "NOT INSTALLED")
    if missing:
        log.error("Missing packages: %s — run `pip install -r requirements.txt`", missing)
        return 1

    log.info("cvxpy solvers available: %s", ", ".join(sorted(cp.installed_solvers())))
    err = check_differentiable_optimizer(cfg.project.seed)
    log.info("cvxpylayers Jacobian vs closed form: max abs error = %.2e (tol %.0e)", err, JACOBIAN_ATOL)
    if err > JACOBIAN_ATOL:
        log.error("Differentiable optimizer gradient check FAILED.")
        return 1

    # Reproducibility check: same seed -> identical numbers.
    a = np.random.default_rng(cfg.project.seed).standard_normal(3)
    b = np.random.default_rng(cfg.project.seed).standard_normal(3)
    if not np.array_equal(a, b):
        log.error("Seeding is not reproducible.")
        return 1

    manifest_path = cfg.paths.reports / "run_manifest.json"
    write_run_manifest(run_manifest(cfg.project.seed, cfg.root), manifest_path)
    log.info("Run manifest written to %s", manifest_path)
    log.info("Stage 0 setup check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
