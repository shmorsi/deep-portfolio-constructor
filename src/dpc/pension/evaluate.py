"""How a policy is judged: what the plan costs its sponsor, and how bad that can get.

The cost of a scenario is every contribution the sponsor makes over the horizon plus the shortfall
still left at the end, in start-of-plan money, as a share of the starting liability. A surplus at
the end earns nothing: the sponsor cannot take it back, which is how most plans work.

The score a policy is trained and compared on is the average cost plus the average cost in the
worst 5% of scenarios (the expected shortfall, or CVaR, at 5%). Lower is better.
"""

from __future__ import annotations

import numpy as np

TAIL = 0.05


def cvar(cost: np.ndarray, tail: float = TAIL) -> float:
    """Average cost in the worst `tail` share of scenarios."""
    k = max(1, int(np.ceil(tail * len(cost))))
    return float(np.sort(cost)[-k:].mean())


def score(cost: np.ndarray) -> float:
    return float(cost.mean()) + cvar(cost)


def summarise(out: dict[str, np.ndarray]) -> dict[str, float]:
    """One policy's results over a scenario set (arrays as returned by plan.run, converted to numpy)."""
    cost, funded, w = out["cost"], out["funded"], out["weights"]
    return {
        "mean_cost": float(cost.mean()),
        "cvar_cost": cvar(cost),
        "score": score(cost),
        "funded_end_share": float((funded[:, -1] >= 1.0).mean()),
        "funded_end_median": float(np.median(funded[:, -1])),
        "funded_low_mean": float(funded.min(axis=1).mean()),
        "funded_low_p05": float(np.quantile(funded.min(axis=1), 0.05)),
        "stocks_mean": float(w[:, :, 0].mean()),
        "bonds_mean": float(w[:, :, 1].mean()),
        "matching_mean": float(w[:, :, 2].mean()),
        "left_mean": float(out["left"].mean()),
    }


def paired(cost_a: np.ndarray, cost_b: np.ndarray, draws: int, seed: int) -> dict[str, float | str]:
    """Score of a minus score of b on the same scenarios, with a 95% bootstrap interval over scenarios."""
    rng = np.random.default_rng(seed)
    n = len(cost_a)
    diffs = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        diffs[i] = score(cost_a[idx]) - score(cost_b[idx])
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    return {"difference": score(cost_a) - score(cost_b), "lo": lo, "hi": hi, "verdict": verdict(lo, hi)}


def paired_mean_blocks(cost_a: np.ndarray, cost_b: np.ndarray, block: int, draws: int, seed: int) -> dict[str, float | str]:
    """Mean cost of a minus mean cost of b over consecutive start years, with a moving-block bootstrap interval.

    For the walk on real history, where neighbouring 20-year periods share most of their months:
    blocks of `block` consecutive start years are resampled together.
    """
    rng = np.random.default_rng(seed)
    d = cost_a - cost_b
    n = len(d)
    block = min(block, n)
    k = int(np.ceil(n / block))
    means = np.empty(draws)
    for i in range(draws):
        idx = (rng.integers(0, n - block + 1, k)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[i] = d[idx].mean()
    lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {"difference": float(d.mean()), "lo": lo, "hi": hi, "verdict": verdict(lo, hi)}


def verdict(lo: float, hi: float) -> str:
    """The pre-registered reading of an interval for (cost of a) minus (cost of b)."""
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "no detectable difference"


def fan(funded: np.ndarray, every: int = 12) -> dict[str, list[float]]:
    """Percentiles of the funded ratio at each year-end, for the fan chart."""
    cols = funded[:, ::every]
    return {f"p{q:02d}": [round(float(v), 4) for v in np.quantile(cols, q / 100.0, axis=0)] for q in (5, 25, 50, 75, 95)}
