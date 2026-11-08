"""The differentiable optimizer must return correct gradients (Stage 5 depends on it)."""

from __future__ import annotations

import pytest

from dpc.setup_check import JACOBIAN_ATOL, check_differentiable_optimizer


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cvxpylayers_jacobian_matches_closed_form(seed: int) -> None:
    assert check_differentiable_optimizer(seed) < JACOBIAN_ATOL
