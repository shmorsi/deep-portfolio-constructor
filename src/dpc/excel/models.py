"""The two spreadsheet models, written once in Python so the workbooks can be checked against them.

Every cell formula in `build.py` has a line here that computes the same thing; `verify.py` has
LibreOffice recalculate each workbook from its formulas and compares the results to this module.

**Pension funding model** (annual, end-of-year cash flows). The liability rolls forward at the
discount rate, grows by the normal cost and falls by benefit payments; assets earn the expected
return on the policy mix and receive the normal cost plus a special payment that amortises any
shortfall over a fixed number of years (re-set every year, as a fresh-start rule does).

**LBO** (five years). A company bought at an EBITDA multiple with senior debt (mandatory
amortisation and a cash sweep) and a subordinated private-credit tranche (cash coupon, bullet).
Interest is charged on opening balances, which keeps the workbook free of circular references.
Returns to equity (MOIC, IRR) and the lender's view (leverage and interest cover by year).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PensionInputs:
    liability: float = 100.0
    funded_ratio: float = 0.90
    discount_rate: float = 0.045
    benefits_pct: float = 0.06
    normal_cost_pct: float = 0.02
    equity_share: float = 0.50
    equity_return: float = 0.07
    bond_return: float = 0.045
    amortisation_years: int = 10
    duration: float = 14.0
    convexity: float = 280.0
    years: int = 20


def pension(p: PensionInputs = PensionInputs()) -> dict[str, np.ndarray]:
    ret = p.equity_share * p.equity_return + (1 - p.equity_share) * p.bond_return
    n = p.years
    L, A = np.zeros(n + 1), np.zeros(n + 1)
    nc, ben, sp = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    L[0], A[0] = p.liability, p.liability * p.funded_ratio
    r, m = p.discount_rate, p.amortisation_years
    annuity = (1 - (1 + r) ** -m) / r
    for t in range(1, n + 1):
        nc[t] = p.normal_cost_pct * L[t - 1]
        ben[t] = p.benefits_pct * L[t - 1]
        sp[t] = max(L[t - 1] - A[t - 1], 0.0) / annuity
        L[t] = L[t - 1] * (1 + r) + nc[t] - ben[t]
        A[t] = A[t - 1] * (1 + ret) + nc[t] + sp[t] - ben[t]
    return {"liability": L, "assets": A, "normal_cost": nc, "benefits": ben, "special": sp,
            "funded_ratio": A / L, "expected_return": np.array(ret)}


def rate_shock(p: PensionInputs, dy: float, bond_duration: float = 8.0) -> dict[str, float]:
    """Funded ratio after an immediate parallel move: liability by duration and convexity, bonds by duration."""
    L0, A0 = p.liability, p.liability * p.funded_ratio
    L1 = L0 * (1 - p.duration * dy + 0.5 * p.convexity * dy ** 2)
    A1 = A0 * (p.equity_share + (1 - p.equity_share) * (1 - bond_duration * dy))
    return {"liability": L1, "assets": A1, "funded_ratio": A1 / L1}


@dataclass(frozen=True)
class LBOInputs:
    revenue0: float = 500.0
    margin0: float = 0.20
    margin_step: float = 0.005
    growth: float = 0.06
    da_pct: float = 0.03
    capex_pct: float = 0.04
    nwc_pct: float = 0.10          # of the change in revenue
    tax: float = 0.25
    entry_multiple: float = 10.0
    exit_multiple: float = 10.0
    fees_pct: float = 0.02
    senior_x: float = 4.0
    senior_rate: float = 0.075
    senior_amort: float = 0.05     # of the original senior balance, each year
    sweep: float = 0.50            # share of free cash flow after mandatory payments
    sub_x: float = 1.5
    sub_rate: float = 0.11
    min_cash: float = 10.0
    years: int = 5
    grid: tuple = field(default=(8.0, 9.0, 10.0, 11.0, 12.0))


def lbo(p: LBOInputs = LBOInputs()) -> dict:
    n = p.years
    ebitda0 = p.revenue0 * p.margin0
    ev0 = p.entry_multiple * ebitda0
    senior0, sub0 = p.senior_x * ebitda0, p.sub_x * ebitda0
    equity0 = ev0 + p.fees_pct * ev0 + p.min_cash - senior0 - sub0
    rev, ebitda, fcf, senior, sub, cash, interest = (np.zeros(n + 1) for _ in range(7))
    rev[0], ebitda[0], senior[0], sub[0], cash[0] = p.revenue0, ebitda0, senior0, sub0, p.min_cash
    mand, swp = np.zeros(n + 1), np.zeros(n + 1)
    for t in range(1, n + 1):
        rev[t] = rev[t - 1] * (1 + p.growth)
        ebitda[t] = rev[t] * (p.margin0 + p.margin_step * t)
        da = p.da_pct * rev[t]
        interest[t] = senior[t - 1] * p.senior_rate + sub[t - 1] * p.sub_rate
        taxes = max(ebitda[t] - da - interest[t], 0.0) * p.tax
        fcf[t] = ebitda[t] - interest[t] - taxes - p.capex_pct * rev[t] - p.nwc_pct * (rev[t] - rev[t - 1])
        mand[t] = min(p.senior_amort * senior0, senior[t - 1])
        swp[t] = min(max(fcf[t] - mand[t], 0.0) * p.sweep, senior[t - 1] - mand[t])
        senior[t] = senior[t - 1] - mand[t] - swp[t]
        sub[t] = sub[t - 1]
        cash[t] = cash[t - 1] + fcf[t] - mand[t] - swp[t]
    net_debt = senior[n] + sub[n] - cash[n]
    exit_equity = p.exit_multiple * ebitda[n] - net_debt
    moic = exit_equity / equity0
    grid = {(e, x): (x * ebitda[n] - net_debt) / (e * ebitda0 * (1 + p.fees_pct) + p.min_cash - senior0 - sub0)
            for e in p.grid for x in p.grid}
    return {"revenue": rev, "ebitda": ebitda, "fcf": fcf, "senior": senior, "sub": sub, "cash": cash, "interest": interest,
            "mandatory": mand, "sweep": swp, "equity0": equity0, "exit_equity": exit_equity, "moic": moic,
            "irr": moic ** (1 / n) - 1, "leverage": (senior + sub) / np.where(ebitda > 0, ebitda, np.nan),
            "coverage": np.where(interest > 0, ebitda / np.where(interest > 0, interest, 1), np.nan), "grid_moic": grid}
