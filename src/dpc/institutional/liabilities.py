"""A defined-benefit plan's promise, member by member, with mortality.

Mortality follows Gompertz-Makeham, mu(x) = A + B c^x, with the parameters of the Standard
Ultimate Survival Model of Dickson, Hardy and Waters, *Actuarial Mathematics for Life Contingent
Risks* (A = 0.00022, B = 2.7e-6, c = 1.124): a textbook table, not a Canadian pensioner table
(CPM2014). Survival from age x for t years has the closed form

    t_p_x = exp(-A t - B c^x (c^t - 1) / ln c).

The plan is a closed cohort: active members, each with an accrued pension payable from 65, and
retirees already being paid. Pensions rise by a fixed cost-of-living adjustment each year.
Expected cash flows are benefit x probability of being alive x indexation.

Two valuations, as a Canadian plan files them:

* **Solvency** (wind-up): discount on the government zero curve. What it would cost to settle
  the promise today; it moves with every change in yields.
* **Going concern**: discount at the expected return on the plan's assets (a smooth, assumed
  rate). Usually lower, and much less volatile, which is why sponsors prefer it and regulators
  look at both.

Longevity risk is a multiplier on the force of mortality: 0.8 means 20% lower death rates at
every age, roughly two more years of life at 65.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUSM = {"A": 0.00022, "B": 2.7e-6, "c": 1.124}


def survival(x: float | np.ndarray, t: np.ndarray, mult: float = 1.0, A=SUSM["A"], B=SUSM["B"], c=SUSM["c"]) -> np.ndarray:
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    return np.exp(-mult * (A * t + B * c ** x * (c ** t - 1.0) / np.log(c)))


def life_expectancy(x: float, mult: float = 1.0, horizon: int = 120) -> float:
    """Curtate expectation of life: sum of t_p_x for t = 1, 2, ..."""
    t = np.arange(1, horizon + 1)
    return float(survival(x, t, mult).sum())


@dataclass(frozen=True)
class Cohort:
    ages: np.ndarray          # each member group's age today
    pensions: np.ndarray      # annual pension (accrued, for actives) per group, in today's money
    retirement_age: float = 65.0
    cola: float = 0.02        # yearly indexation, applied from today

    def cash_flows(self, mult: float = 1.0, horizon: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Expected payments at the end of each year 1..horizon (paid annually in arrear, for clarity)."""
        t = np.arange(1, horizon + 1, dtype=float)
        cf = np.zeros_like(t)
        for x, p in zip(self.ages, self.pensions):
            alive = survival(x, t, mult)
            paying = (x + t) >= self.retirement_age + 1e-9
            cf += p * alive * paying * (1 + self.cola) ** t
        return t, cf


def example_cohort() -> Cohort:
    """A mature plan: actives aged 30 to 64, retirees 65 to 95, with more money owed to the old."""
    act = np.arange(30, 65, 5, dtype=float)
    ret = np.arange(65, 100, 5, dtype=float)
    ages = np.concatenate([act, ret])
    # accrued pensions grow with service for actives; retirees' pensions are level and the groups thin out with age
    pa = 6.0 + 0.9 * (act - 30)
    pr = 40.0 * np.exp(-0.06 * (ret - 65))
    return Cohort(ages, np.concatenate([pa, pr]))


def value_flat(times: np.ndarray, cash: np.ndarray, rate: float) -> float:
    return float(np.sum(cash / (1 + rate) ** times))


def valuations(cohort: Cohort, zero, going_concern_rate: float = 0.06, mult: float = 1.0) -> dict:
    """Solvency (on the zero curve) and going-concern (at a flat assumed return) values, and durations."""
    from .curve import key_rate_durations

    t, cf = cohort.cash_flows(mult)
    sol = key_rate_durations(cf, t, zero)
    gc = value_flat(t, cf, going_concern_rate)
    gc_dur = float(np.sum(t * cf / (1 + going_concern_rate) ** (t + 1)) / gc)   # modified duration
    return {"solvency": sol["pv"], "going_concern": gc, "solvency_duration": sol["effective_duration"],
            "going_concern_duration": gc_dur, "krd": sol["krd"], "cash_flows": cf, "times": t}
