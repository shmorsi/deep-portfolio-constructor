"""Derivatives and rates: textbook values (Hull, Options, Futures and Other Derivatives) and identities."""

from __future__ import annotations

import numpy as np
import pytest

from dpc.derivatives import black, hedging, montecarlo, rates, sabr, trees


def test_black_scholes_hull_example_and_parity():
    # Hull, Example 15.6: S 42, K 40, r 10%, sigma 20%, T 0.5 -> call 4.76, put 0.81
    c = black.bsm(42, 40, 0.5, 0.10, 0.20)
    p = black.bsm(42, 40, 0.5, 0.10, 0.20, kind="put")
    assert c == pytest.approx(4.76, abs=0.005) and p == pytest.approx(0.81, abs=0.005)
    assert c - p == pytest.approx(42 - 40 * np.exp(-0.05), abs=1e-12)          # put-call parity


def test_greeks_match_finite_differences():
    S, K, T, r, s, q = 100.0, 95.0, 0.75, 0.03, 0.25, 0.01
    g = black.greeks(S, K, T, r, s, q)
    f = lambda **kw: float(black.bsm(**{"S": S, "K": K, "T": T, "r": r, "sigma": s, "q": q, **kw}))
    h = 1e-4
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h ** 2, rel=1e-3)
    assert g["vega"] == pytest.approx((f(sigma=s + h) - f(sigma=s - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-5)


def test_implied_vol_roundtrip_and_bounds():
    for s in (0.05, 0.2, 0.8):
        price = float(black.bsm(100, 110, 1.0, 0.02, s, kind="put"))
        assert black.implied_vol(price, 100, 110, 1.0, 0.02, kind="put") == pytest.approx(s, abs=1e-8)
    assert np.isnan(black.implied_vol(0.0, 100, 50, 1.0, 0.0))


def test_black76_and_bachelier():
    F, K, T, r = 0.03, 0.03, 2.0, 0.02
    assert float(black.black76(F, K, T, r, 0.2)) == pytest.approx(float(black.black76(F, K, T, r, 0.2, kind="put")))
    # ATM Bachelier: df * sigma * sqrt(T / 2 pi)
    assert float(black.bachelier(F, K, T, r, 0.01)) == pytest.approx(np.exp(-r * T) * 0.01 * np.sqrt(T / (2 * np.pi)))


def test_american_put_tree():
    # Hull, Example 21.1 setting: S 50, K 50, r 10%, sigma 40%, T 5 months; converged American put about 4.28
    am = trees.crr(50, 50, 5 / 12, 0.10, 0.40, steps=2000)
    eu = trees.crr(50, 50, 5 / 12, 0.10, 0.40, steps=2000, american=False)
    assert am == pytest.approx(4.28, abs=0.02)
    assert eu == pytest.approx(float(black.bsm(50, 50, 5 / 12, 0.10, 0.40, kind="put")), abs=0.01)
    assert am > eu


def test_asian_control_variate_reduces_error():
    out = montecarlo.asian_call(100, 100, 1.0, 0.03, 0.25, n=12, paths=40_000)
    assert out["control_se"] < 0.2 * out["plain_se"]
    assert out["control"] == pytest.approx(out["plain"], abs=4 * out["plain_se"])
    mc_geo = montecarlo.geometric_asian_call(100, 100, 1.0, 0.03, 0.25, 12)
    assert mc_geo < float(black.bsm(100, 100, 1.0, 0.03, 0.25))        # averaging lowers volatility


def test_sabr_recovers_its_parameters():
    F, T = 0.03, 5.0
    K = F * np.array([0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5])
    true = dict(alpha=0.035, beta=0.5, rho=-0.3, nu=0.45)
    vols = sabr.implied_vol(F, K, T, **true)
    fit = sabr.calibrate(F, T, K, vols)
    assert fit["rmse"] < 1e-6
    assert fit["rho"] == pytest.approx(true["rho"], abs=1e-3) and fit["nu"] == pytest.approx(true["nu"], abs=1e-3)


def test_bootstrap_reprices_par_bonds_and_swaps():
    m = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    y = np.array([0.043, 0.042, 0.040, 0.039, 0.039, 0.040, 0.041, 0.045, 0.046])
    c = rates.bootstrap_par(m, y)
    for mat, yy in zip(m, y):
        assert rates.bond_price(c, yy, mat) == pytest.approx(100.0, abs=1e-9)
        # a swap paying semiannually on the same curve has the par yield as its par rate
        assert rates.swap(c, mat)["par_rate"] * 1 == pytest.approx(yy, abs=1e-12)
    s = rates.swap(c, 10, fixed=0.041, notional=1e6)
    assert s["pv"] == pytest.approx(0.0, abs=1e-6) and s["dv01"] > 0
    assert s["dv01"] == pytest.approx(1e6 * 1e-4 * 8.2, rel=0.1)              # roughly notional x duration x 1 bp
    assert c.forward(9.0, 10.0) > 0


def test_hedging_error_shrinks_with_rebalancing_and_pays_the_vol_gap():
    a = hedging.simulate_short_call(rebalances=16, paths=20_000)
    b = hedging.simulate_short_call(rebalances=256, paths=20_000)
    assert b["std"] < 0.35 * a["std"]                         # about 1/4 for 16x the rebalances
    assert abs(b["mean"]) < 0.02
    rich = hedging.simulate_short_call(sigma_implied=0.25, sigma_real=0.15, rebalances=252)
    assert rich["mean"] > 0.5                                 # sold 25% vol, 15% happened
