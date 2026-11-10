"""Investment policies for the plan. Each maps the state the plan can see to three weights
(stocks, 10-year bonds, matching bonds) that are never negative and add to one: no shorting, no leverage.

The state, in order (plan.STATE): funded ratio, yield, 12-month yield change, 12-month stock return,
inflation over the last published 12 months, expected inflation, share of the horizon elapsed, and the
three weights held now.
"""

from __future__ import annotations

import torch
from torch import nn


class Fixed:
    """The same weights every month (rebalanced back to them)."""

    def __init__(self, stocks: float, bonds: float, matching: float):
        self.w = torch.tensor([stocks, bonds, matching], dtype=torch.float32)
        if abs(float(self.w.sum()) - 1.0) > 1e-6 or float(self.w.min()) < 0:
            raise ValueError("weights must be non-negative and add to one")

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return self.w.expand(state.shape[0], 3)


class Glide:
    """A glide path: hold more stocks while the plan is short, and move into matching bonds as it fills.

    Stocks are `high` at a funded ratio of `low_fr` or less, `low` at `high_fr` or more, and on the
    straight line between. The rest is in matching bonds. This is the rule many plans actually run.
    """

    def __init__(self, low_fr: float = 0.80, high_fr: float = 1.10, high: float = 0.70, low: float = 0.20):
        self.low_fr, self.high_fr, self.high, self.low = low_fr, high_fr, high, low

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        along = torch.clamp((state[:, 0] - self.low_fr) / (self.high_fr - self.low_fr), 0.0, 1.0)
        stocks = self.high - (self.high - self.low) * along
        return torch.stack([stocks, torch.zeros_like(stocks), 1.0 - stocks], dim=1)


class Learned(nn.Module):
    """A small network that starts as the glide path and learns where to depart from it.

    The network's output is added to the logarithm of the glide path's weights and passed through a
    softmax. With its last layer at zero (how it starts) it is the glide path with a 2% floor under
    every asset, so an asset the glide path never holds stays within reach. Whatever it learns stays
    long-only and fully invested by construction.
    """

    # Fixed centring and scaling of the ten state variables (not fitted to any data).
    CENTRE = torch.tensor([1.0, 0.05, 0.0, 0.10, 0.03, 0.03, 0.5, 1 / 3, 1 / 3, 1 / 3])
    SCALE = torch.tensor([0.2, 0.03, 0.015, 0.2, 0.03, 0.02, 0.5, 0.5, 0.5, 0.5])

    def __init__(self, hidden: int = 32, anchor: Glide | None = None):
        super().__init__()
        self.anchor = anchor or Glide()
        self.net = nn.Sequential(nn.Linear(10, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        base = torch.log(self.anchor(state) + 0.02)
        return torch.softmax(base + self.net((state - self.CENTRE) / self.SCALE), dim=1)


class Ensemble:
    """The average of several networks' weights (one per training seed)."""

    def __init__(self, members: list[Learned]):
        self.members = members

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(state) for m in self.members], dim=0).mean(dim=0)


def baselines() -> dict[str, object]:
    """The four standard policies the learned one is compared with."""
    return {
        "60/40": Fixed(0.60, 0.40, 0.0),
        "Liability-driven 40/60": Fixed(0.40, 0.0, 0.60),
        "Fully matched": Fixed(0.0, 0.0, 1.0),
        "Glide path": Glide(),
    }
