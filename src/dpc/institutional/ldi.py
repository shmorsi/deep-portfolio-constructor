"""Liability-driven investing with a swap overlay, and the liquidity risk it creates.

A plan hedges the interest-rate risk of its promise by matching the liability's DV01 (the
change in value for a 1 bp move). Physical bonds cover part of it; receive-fixed interest-rate
swaps cover the rest without tying up capital, which leaves money in return-seeking assets.
That is leverage: the swap's losses when yields rise must be paid in cash, as collateral, within
days.

September 2022 in the UK is the case study: long gilt yields rose by more than a percentage
point within days, swap and repo margin calls forced pension LDI funds to sell gilts into a
falling market, and the Bank of England had to intervene. The hedge had done its job on
funding (the liability fell as much as the hedge lost); what failed was liquidity.

This module measures both sides for one plan, at each hedge ratio:

* **Funding risk:** the funded ratio after a parallel yield move (hedging stabilises it).
* **Liquidity risk:** the margin call on the swaps, and the largest yield rise the plan's
  unencumbered liquid assets can meet (hedging with swaps shrinks it).

Parallel shifts, first-order (DV01) plus convexity for the liability, a single swap tenor:
stated simplifications.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plan:
    liability: float = 100.0
    liability_duration: float = 14.0
    liability_convexity: float = 280.0
    funded_ratio: float = 0.95
    bond_share: float = 0.30          # physical long bonds, as a share of assets
    bond_duration: float = 16.0
    cash_share: float = 0.05          # cash and T-bills: the first collateral
    return_seeking_share: float = 0.65
    haircut_bonds: float = 0.03       # long bonds pledged as collateral lose this much value
    swap_duration: float = 18.0       # receive-fixed swap DV01 per unit notional, in years


def overlay(plan: Plan, hedge_ratio: float) -> dict:
    """Swap notional needed to hedge `hedge_ratio` of the liability DV01, and the resulting leverage."""
    A = plan.funded_ratio * plan.liability
    dv01_L = plan.liability * plan.liability_duration * 1e-4
    dv01_bonds = A * plan.bond_share * plan.bond_duration * 1e-4
    need = max(hedge_ratio * dv01_L - dv01_bonds, 0.0)
    notional = need / (plan.swap_duration * 1e-4)
    physical_hedge = min(dv01_bonds / dv01_L, hedge_ratio) if dv01_L else 0.0
    return {"assets": A, "dv01_liability": dv01_L, "dv01_bonds": dv01_bonds, "swap_notional": notional,
            "swap_dv01": need, "leverage": (A + notional) / A, "physical_hedge_ratio": physical_hedge,
            "achieved_hedge_ratio": (dv01_bonds + need) / dv01_L}


def shock(plan: Plan, hedge_ratio: float, dy: float, equity: float = 0.0) -> dict:
    """Funded ratio and collateral position after a parallel yield move dy (decimal) and an equity move."""
    o = overlay(plan, hedge_ratio)
    A = o["assets"]
    L1 = plan.liability * (1 - plan.liability_duration * dy + 0.5 * plan.liability_convexity * dy ** 2)
    bonds = A * plan.bond_share * (1 - plan.bond_duration * dy)
    cash = A * plan.cash_share
    rs = A * plan.return_seeking_share * (1 + equity)
    swap_pnl = -o["swap_notional"] * plan.swap_duration * dy
    A1 = bonds + cash + rs + swap_pnl
    margin_call = max(-swap_pnl, 0.0)
    collateral = cash + bonds * (1 - plan.haircut_bonds)
    return {"funded_ratio": A1 / L1, "change_pts": 100 * (A1 / L1 - plan.funded_ratio),
            "margin_call": margin_call, "collateral": collateral,
            "coverage": collateral / margin_call if margin_call > 0 else float("inf"),
            "forced_sale_of_return_seeking": max(margin_call - collateral, 0.0)}


def max_survivable_rise(plan: Plan, hedge_ratio: float) -> float:
    """The largest yield rise (decimal) that cash plus pledgeable bonds can meet, solved exactly.

    collateral(dy) = A c + A s_b (1 - D_b dy)(1 - h) must cover N D_s dy.
    """
    o = overlay(plan, hedge_ratio)
    if o["swap_notional"] <= 0:
        return float("inf")
    A = o["assets"]
    hc = 1 - plan.haircut_bonds
    num = A * plan.cash_share + A * plan.bond_share * hc
    den = o["swap_notional"] * plan.swap_duration + A * plan.bond_share * hc * plan.bond_duration
    return num / den


def worst_rises(daily_yield, horizons=(1, 5, 20)) -> dict:
    """Largest historical rise in a yield over h business days (decimal), with its end date."""
    out = {}
    y = daily_yield.dropna()
    for h in horizons:
        d = y.diff(h)
        out[h] = {"rise": float(d.max()), "date": f"{d.idxmax():%Y-%m-%d}"}
    return out
