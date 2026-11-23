"""Cox-Ross-Rubinstein binomial trees: European and American options.

u = e^{sigma sqrt dt}, d = 1/u, risk-neutral p = (e^{(r-q)dt} - d)/(u - d). Backward induction;
an American option takes the larger of continuation and immediate exercise at every node.
The early-exercise premium of an American put is the difference from the European price.
"""

from __future__ import annotations

import numpy as np


def crr(S, K, T, r, sigma, steps=500, q=0.0, kind="put", american=True) -> float:
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    s = S * u ** (steps - j) * d ** j
    pay = (lambda x: np.maximum(x - K, 0.0)) if kind == "call" else (lambda x: np.maximum(K - x, 0.0))
    v = pay(s)
    for n in range(steps - 1, -1, -1):
        v = disc * (p * v[:-1] + (1 - p) * v[1:])
        if american:
            s = S * u ** (n - np.arange(n + 1)) * d ** np.arange(n + 1)
            v = np.maximum(v, pay(s))
    return float(v[0])
