"""Training the learned policy: gradient descent through the plan simulator.

The network's weights decide the plan's holdings; the holdings decide its assets; the assets
decide what the sponsor has to pay in. All of that is differentiable, so the score (average cost
plus the average cost of the worst 5%) can be pushed down directly. The network only ever sees
scenarios drawn from a generator fitted on the data it is allowed to know.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import scenarios as sc
from .plan import Plan, exogenous, run
from .policy import Learned


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 300
    batch: int = 1024
    n_train: int = 8192
    n_val: int = 4096
    lr: float = 1e-2
    check_every: int = 30
    hidden: int = 32


def score_t(cost: torch.Tensor, tail: float = 0.05) -> torch.Tensor:
    """Average cost plus the average of the worst `tail` share, as a differentiable tensor."""
    k = max(1, int(torch.ceil(torch.tensor(tail * cost.numel())).item()))
    return cost.mean() + torch.topk(cost, k).values.mean()


def train(model: sc.Model, plan: Plan, seed: int, cfg: TrainConfig = TrainConfig(), log=None) -> tuple[Learned, list[dict]]:
    """Train one network. Scenarios for training and for validation come from `model` with different seeds.

    Returns the network at its best validation score, and the validation history.
    """
    torch.manual_seed(seed)
    train_set = exogenous(sc.simulate(model, cfg.n_train, plan.horizon, seed=1000 + seed), plan)
    val_set = exogenous(sc.simulate(model, cfg.n_val, plan.horizon, seed=5000 + seed), plan)
    net = Learned(hidden=cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    best, best_state = float("inf"), None

    def validate(step: int) -> None:
        nonlocal best, best_state
        with torch.no_grad():
            val = float(score_t(run(net, val_set, plan)["cost"]))
        history.append({"seed": seed, "step": step, "validation_score": val})
        if log:
            log(f"seed {seed} step {step:4d}  validation score {val:.4f}")
        if val < best - 1e-6:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}

    validate(0)
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, len(train_set), (cfg.batch,), generator=gen)
        loss = score_t(run(net, train_set.take(idx), plan)["cost"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % cfg.check_every == 0:
            validate(step)
    net.load_state_dict(best_state)
    net.eval()
    return net, history
