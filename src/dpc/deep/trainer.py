"""A small, reusable PyTorch training loop.

One loop for the deep models that train on mini-batches of cross-sections:
AdamW, gradient clipping, early stopping on a validation metric, best-weights
restore, and explicit device placement (CUDA, Apple MPS or CPU). Everything a
reviewer looks for in a PyTorch project, with nothing hidden in a framework.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

Batch = dict[str, torch.Tensor]
StepFn = Callable[[nn.Module, Batch], tuple[torch.Tensor, torch.Tensor]]
"""(model, batch) -> (loss to minimise, metric per sample to maximise)."""


def resolve_device(name: str) -> torch.device:
    """'auto' picks CUDA, then Apple MPS, then CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def to_device(batch: Batch, device: torch.device) -> Batch:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@dataclass
class FitResult:
    model: nn.Module
    best_metric: float
    epochs: int
    history: list[tuple[float, float]] = field(default_factory=list)  # (train loss, val metric) per epoch


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, step: StepFn, device: torch.device) -> float:
    model.eval()
    metrics = []
    for batch in loader:
        _, m = step(model, to_device(batch, device))
        metrics.append(m.detach().cpu())
    return float(torch.cat(metrics).mean())


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    step: StepFn,
    *,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    grad_clip: float,
    device: torch.device,
) -> FitResult:
    """Train until the validation metric stops improving; return the best weights."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best = evaluate(model, val_loader, step, device)  # the starting point is a valid answer too
    best_state = copy.deepcopy(model.state_dict())
    history: list[tuple[float, float]] = [(float("nan"), best)]
    bad = 0
    for _ in range(max_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            loss, _ = step(model, to_device(batch, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(loss.item())
        v = evaluate(model, val_loader, step, device)
        history.append((sum(losses) / max(len(losses), 1), v))
        if v > best + 1e-6:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, best, len(history) - 1, history)
