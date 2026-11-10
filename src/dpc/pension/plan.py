"""The plan simulator: assets, benefits, contributions and the funded ratio, month by month.

Everything the policy cannot influence (asset returns, the liability, the benefits, inflation,
the things it may look at) is computed once per scenario set (`exogenous`). The part the policy
does influence (the assets, and so the contributions the sponsor has to make) is simulated in
PyTorch, so a policy with parameters can be trained through it (`run`).

One month, in order:
  1. the policy sees the funded ratio and the recent past, and sets the weights;
  2. trading to those weights costs `cost_bps` of every dollar traded;
  3. the month's returns arrive; the benefit is paid out of the assets;
  4. the liability is revalued at the new yield and indexed to the month's inflation;
  5. if the plan is short, the sponsor pays in 1/`amort_months` of the shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .fixed_income import par_bond_return
from .liability import Schedule, paths
from .scenarios import BURN

ASSETS = ("stocks", "10-year bonds", "matching bonds")
STATE = ("funded ratio", "yield", "yield change, 12 months", "stock return, 12 months", "inflation, last published 12 months",
         "expected inflation (last ten years)", "time elapsed", "weight in stocks", "weight in 10-year bonds", "weight in matching bonds")
YEARS_BACK = 10   # expected inflation is the average of the last ten years, to the last published month


@dataclass(frozen=True)
class Plan:
    funded0: float = 0.90        # assets over liabilities at the start
    amort_months: int = 120      # a shortfall is paid off over this many months
    cost_bps: float = 10.0       # cost of trading, per dollar traded
    horizon: int = 240           # months the study follows the plan
    schedule: Schedule = field(default_factory=Schedule)


@dataclass
class Exo:
    """What the world does to the plan, per scenario (S scenarios, T months)."""
    returns: torch.Tensor     # (S, T, 3) returns of the three assets over each month
    liability: torch.Tensor   # (S, T+1)
    benefit: torch.Tensor     # (S, T)
    index: torch.Tensor       # (S, T+1) cumulative inflation
    feats: torch.Tensor       # (S, T, 5) what the policy may look at, known at each decision

    def __len__(self) -> int:
        return self.returns.shape[0]

    def take(self, idx: torch.Tensor) -> "Exo":
        return Exo(self.returns[idx], self.liability[idx], self.benefit[idx], self.index[idx], self.feats[idx])


def exogenous(scen: dict[str, np.ndarray], plan: Plan) -> Exo:
    """Turn a scenario set (see scenarios.py) into what the plan needs. Uses nothing from after each decision."""
    T = plan.horizon
    y_all, eq_all, pi_all = scen["yield"], scen["equity"], scen["infl"]
    if y_all.shape[1] != BURN + T + 1:
        raise ValueError("scenario length does not match the plan's horizon")
    y, eq, pi = y_all[:, BURN:], eq_all[:, BURN:], pi_all[:, BURN:]
    # Consumer prices come out mid-month, so at a month-end the latest published inflation is the month before's.
    # Expected inflation at time t: the average, annualised, of the 120 published months ending at t-1.
    back = YEARS_BACK * 12
    csum = np.concatenate([np.zeros((pi_all.shape[0], 1)), np.cumsum(pi_all, axis=1)], axis=1)   # csum[:, j] = inflation over months ending at t = -BURN+1 .. -BURN+j
    expected = np.stack([(csum[:, BURN + t - 1] - csum[:, BURN + t - 1 - back]) / back * 12.0 for t in range(T + 1)], axis=1)
    liab = paths(plan.schedule, y, pi, expected)
    returns = np.stack([eq, par_bond_return(y[:, :-1], y[:, 1:]), liab["matching"]], axis=2)
    # What the policy may look at when it decides at time t = 0 .. T-1 (column BURN + t of the yield array):
    #   the yield now, its change over 12 months, the stock market's return over the last 12 months,
    #   inflation over the 12 months to the last published month, and expected inflation.
    feats = np.empty((y.shape[0], T, 5))
    for t in range(T):
        c = BURN + t
        feats[:, t, 0] = y_all[:, c]
        feats[:, t, 1] = y_all[:, c] - y_all[:, c - 12]
        feats[:, t, 2] = np.prod(1.0 + eq_all[:, c - 12:c], axis=1) - 1.0     # months ending at t-11 .. t
        feats[:, t, 3] = pi_all[:, c - 13:c - 1].mean(axis=1) * 12.0          # months ending at t-12 .. t-1
        feats[:, t, 4] = expected[:, t]
    as_t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return Exo(as_t(returns), as_t(liab["liability"]), as_t(liab["benefit"]), as_t(liab["index"]), as_t(feats))


def run(policy, exo: Exo, plan: Plan) -> dict[str, torch.Tensor]:
    """Follow the plan under `policy`. Returns, per scenario:

      cost     what the sponsor pays over the horizon: every contribution plus the shortfall left at the end,
               in start-of-plan money, as a share of the starting liability
      funded   (S, T+1) the funded ratio
      contrib  (S, T)   contributions, in start-of-plan money, as a share of the starting liability
      weights  (S, T, 3)
    """
    S, T = len(exo), plan.horizon
    L, R = exo.liability, exo.returns
    rate = plan.cost_bps / 1e4
    assets = plan.funded0 * L[:, 0]
    drift = None
    funded, contribs, weights = [assets / L[:, 0]], [], []
    for t in range(T):
        fr = assets / L[:, t]
        elapsed = torch.full((S,), t / T)
        held = drift if drift is not None else torch.zeros(S, 3)
        state = torch.cat([fr[:, None], exo.feats[:, t], elapsed[:, None], held], dim=1)
        w = policy(state)
        if drift is not None:                                   # the plan starts already invested: no cost at inception
            assets = assets * (1.0 - rate * (w - drift).abs().sum(dim=1))
        growth = 1.0 + (w * R[:, t]).sum(dim=1)
        after = assets * growth - exo.benefit[:, t]
        drift = w * (1.0 + R[:, t]) / growth[:, None]
        pay = torch.relu(L[:, t + 1] - after) / plan.amort_months
        assets = torch.clamp(after + pay, min=1e-6)
        contribs.append(pay / exo.index[:, t + 1])
        funded.append(assets / L[:, t + 1])
        weights.append(w)
    contrib = torch.stack(contribs, dim=1) / L[:, :1]
    left = torch.relu(L[:, T] - assets) / exo.index[:, T] / L[:, 0]
    return {"cost": contrib.sum(dim=1) + left, "funded": torch.stack(funded, dim=1), "contrib": contrib,
            "weights": torch.stack(weights, dim=1), "left": left}
