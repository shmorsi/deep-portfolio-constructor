"""Same seed must give the same numbers; different seeds must not."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from dpc.reproducibility import make_rng, make_torch_generator, package_versions, set_global_seed


def _draw_everything() -> tuple[float, np.ndarray, torch.Tensor, torch.Tensor]:
    """Draw from every global RNG and initialise a small network."""
    net = torch.nn.LSTM(input_size=3, hidden_size=4)
    return (
        random.random(),
        np.random.rand(3),
        torch.randn(3),
        torch.cat([p.detach().flatten() for p in net.parameters()]),
    )


def test_global_seed_reproduces_all_libraries() -> None:
    set_global_seed(123, deterministic_torch=True)
    first = _draw_everything()
    set_global_seed(123, deterministic_torch=True)
    second = _draw_everything()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[3], second[3])  # identical network initialisation


def test_different_seeds_differ() -> None:
    set_global_seed(1)
    a = _draw_everything()
    set_global_seed(2)
    b = _draw_everything()
    assert not np.array_equal(a[1], b[1])
    assert not torch.equal(a[3], b[3])


def test_training_step_is_bitwise_reproducible() -> None:
    """A full forward/backward/optimiser step gives identical weights twice."""

    def one_step() -> torch.Tensor:
        set_global_seed(7, deterministic_torch=True, torch_num_threads=1)
        net = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        x, y = torch.randn(32, 5), torch.randn(32, 1)
        loss = torch.nn.functional.mse_loss(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return torch.cat([p.detach().flatten() for p in net.parameters()])

    assert torch.equal(one_step(), one_step())


def test_streams_are_independent_and_reproducible() -> None:
    a0 = make_rng(42, stream=0).standard_normal(5)
    a0_again = make_rng(42, stream=0).standard_normal(5)
    a1 = make_rng(42, stream=1).standard_normal(5)
    np.testing.assert_array_equal(a0, a0_again)
    assert not np.allclose(a0, a1)

    g0 = torch.randn(5, generator=make_torch_generator(42, 0))
    g0_again = torch.randn(5, generator=make_torch_generator(42, 0))
    g1 = torch.randn(5, generator=make_torch_generator(42, 1))
    assert torch.equal(g0, g0_again)
    assert not torch.equal(g0, g1)


@pytest.mark.parametrize("pkg", ["numpy", "pandas", "torch", "cvxpy", "cvxpylayers", "scikit-learn"])
def test_core_packages_installed(pkg: str) -> None:
    assert package_versions((pkg,))[pkg] is not None
