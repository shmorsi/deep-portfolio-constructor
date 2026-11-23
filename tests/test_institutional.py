"""Tests for the institutional toolkit: every identity the modules rely on, on synthetic data (no download)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dpc.institutional import attribution, black_litterman as bl, curve, fx, ldi, liabilities, privates
from dpc.institutional.risk_model import FactorRiskModel, risk_decomposition

RNG = np.random.default_rng(7)


# ----------------------------------------------------------------- risk model
def _factor_data(T=400, N=12, K=3):
    f = RNG.normal(0, 0.04, (T, K))
    B = RNG.normal(1, 0.3, (N, K))
    spec = RNG.uniform(0.01, 0.03, N)
    e = RNG.normal(0, 1, (T, N)) * spec
    return f, B, spec, f @ B.T + e


def test_factor_model_recovers_loadings_and_specific_risk():
    f, B, spec, r = _factor_data()
    m = FactorRiskModel.fit(r, f, [f"a{i}" for i in range(12)], ["f1", "f2", "f3"])
    assert np.allclose(m.loadings, B, atol=0.1)
    assert np.allclose(np.sqrt(m.specific_var), spec, rtol=0.15)
    S = m.cov
    assert np.allclose(S, S.T) and np.all(np.linalg.eigvalsh(S) > 0)


def test_risk_contributions_add_up():
    f, _, _, r = _factor_data()
    m = FactorRiskModel.fit(r, f, [f"a{i}" for i in range(12)], ["f1", "f2", "f3"])
    w = RNG.dirichlet(np.ones(12))
    b = np.full(12, 1 / 12)
    d = risk_decomposition(m, w, b)
    assert d["contribution"].sum() == pytest.approx(d["sigma"], rel=1e-10)
    assert d["factor_share"] + d["specific_share"] == pytest.approx(1.0, abs=1e-10)
    assert sum(d["factor_variance_by_factor"].values()) == pytest.approx(d["factor_share"], abs=1e-10)


# ----------------------------------------------------------- Black-Litterman
def test_no_view_returns_the_benchmark():
    A = RNG.normal(size=(6, 6))
    S = A @ A.T / 100 + np.eye(6) * 0.001
    b = RNG.dirichlet(np.ones(6))
    pi = bl.implied_returns(S, b, 2.5)
    assert np.allclose(bl.unconstrained_weights(pi, S, 2.5), b)
    # a view equal to what the prior already implies leaves the prior unchanged
    P = bl.relative_view(list("abcdef"), ["a"], ["b"])[None, :]
    mu, _ = bl.posterior(pi, S, P, P @ pi, bl.he_litterman_omega(P, S, 0.05), 0.05)
    assert np.allclose(mu, pi)


def test_bullish_view_raises_the_winner_and_he_litterman_halfway():
    S = np.diag([0.04, 0.04, 0.04]) / 12
    b = np.array([1 / 3] * 3)
    pi = bl.implied_returns(S, b, 2.5)
    P = np.array([[1.0, 0.0, 0.0]])
    q = pi[0] + 0.01
    mu, _ = bl.posterior(pi, S, P, [q], bl.he_litterman_omega(P, S, 0.05), 0.05)
    # with Omega = P tau Sigma P' the posterior for a single absolute view lands halfway
    assert mu[0] == pytest.approx(pi[0] + 0.005)
    assert np.allclose(mu[1:], pi[1:])


# --------------------------------------------------------------- optimiser
def test_te_budget_is_binding_and_long_only():
    from dpc.institutional.active import optimise

    A = RNG.normal(size=(8, 8))
    S = A @ A.T / 400 + np.eye(8) * 0.0005
    b = RNG.dirichlet(np.ones(8) * 5)
    mu = RNG.normal(0.005, 0.003, 8)
    te = 0.01
    w = optimise(mu, S, b, te, 0.10)
    assert w.sum() == pytest.approx(1.0) and np.all(w >= -1e-9) and np.all(w <= b + 0.10 + 1e-6)
    assert np.sqrt((w - b) @ S @ (w - b)) == pytest.approx(te, rel=1e-3)
    assert np.allclose(optimise(mu, S, b, 1e-7, 0.10), b, atol=1e-4)   # no risk budget, no bet


# -------------------------------------------------------------- attribution
def _holdings(T=24, N=10):
    names = [f"i{k}" for k in range(N)]
    sectors = {n: ("A", "B", "C")[k % 3] for k, n in enumerate(names)}
    wp = RNG.dirichlet(np.ones(N), T)
    wp[:, 0] = 0.0                                   # a holding the portfolio never owns
    wp = wp / wp.sum(axis=1, keepdims=True)
    wb = RNG.dirichlet(np.ones(N), T)
    r = RNG.normal(0.01, 0.05, (T, N))
    dates = pd.date_range("2020-01-31", periods=T, freq="ME")
    return attribution.holdings_frame(dates, names, sectors, wp, wb, r), wp, wb, r


def test_brinson_adds_up_each_period():
    h, wp, wb, r = _holdings()
    e = attribution.brinson_fachler(h)
    per = e.groupby("date")[attribution.EFFECTS].sum().sum(axis=1)
    active = (wp * r).sum(axis=1) - (wb * r).sum(axis=1)
    assert np.allclose(per.to_numpy(), active)


def test_carino_linking_adds_up_to_cumulative_active_return():
    h, wp, wb, r = _holdings()
    linked = attribution.carino_link(attribution.brinson_fachler(h))
    Rp = np.prod(1 + (wp * r).sum(axis=1)) - 1
    Rb = np.prod(1 + (wb * r).sum(axis=1)) - 1
    assert linked["total"].sum() == pytest.approx(Rp - Rb, abs=1e-12)


def test_sql_matches_pandas():
    pytest.importorskip("duckdb")
    h, *_ = _holdings()
    a = attribution.carino_link(attribution.brinson_fachler(h))
    b = attribution.carino_link_sql(h)
    assert np.allclose(a.sort_index().to_numpy(), b.sort_index()[a.columns].to_numpy(), atol=1e-12)


# -------------------------------------------------------------------- curve
def test_nss_recovers_a_known_curve_and_krds_sum_to_duration():
    true = curve.NSS(np.array([0.045, -0.02, 0.01, 0.015]), 1.5, 10.0, 0.0)
    m = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    fit = curve.NSS.fit(m, true(m))
    assert fit.rmse < 2e-5
    t = np.arange(1, 41, dtype=float)
    cf = np.full_like(t, 5.0)
    k = curve.key_rate_durations(cf, t, fit)
    assert k["sum_krd"] == pytest.approx(k["effective_duration"], rel=1e-6)
    # a zero-coupon bond's duration is its maturity on a continuously compounded curve
    z = curve.key_rate_durations(np.array([100.0]), np.array([10.0]), fit)
    assert z["effective_duration"] == pytest.approx(10.0, rel=1e-6)


def test_pca_finds_a_level_factor():
    T, mats = 300, np.arange(1, 31)
    level = RNG.normal(0, 0.002, T)
    slope = RNG.normal(0, 0.0007, T)
    X = level[:, None] + slope[:, None] * (mats - 15) / 15 + RNG.normal(0, 5e-5, (T, 30))
    p = curve.pca(pd.DataFrame(X, columns=mats))
    assert p["explained"][0] > 0.8 and p["cumulative"] > 0.99
    assert np.all(p["loadings"][0] > 0)


# --------------------------------------------------------------- liabilities
def test_survival_and_life_expectancy():
    t = np.arange(0, 60)
    s = liabilities.survival(65, t)
    assert s[0] == 1.0 and np.all(np.diff(s) < 0)
    e65 = liabilities.life_expectancy(65)
    assert 20 < e65 < 24                              # Gompertz approximation: about 22.5 years
    assert liabilities.life_expectancy(65, mult=0.8) > e65 + 1


def test_lower_mortality_and_lower_rates_raise_the_liability():
    c = liabilities.example_cohort()
    flat = lambda r: (lambda t: np.full_like(np.asarray(t, float), r))
    base = liabilities.valuations(c, flat(0.04))
    assert liabilities.valuations(c, flat(0.03))["solvency"] > base["solvency"]
    assert liabilities.valuations(c, flat(0.04), mult=0.8)["solvency"] > base["solvency"]
    assert base["going_concern"] < base["solvency"]                         # 6% > 4%
    assert sum(base["krd"].values()) == pytest.approx(base["solvency_duration"], rel=1e-6)


# ---------------------------------------------------------------------- LDI
def test_full_hedge_stabilises_funding_but_creates_margin_calls():
    p = ldi.Plan()
    small = [ldi.shock(p, h, -0.01)["change_pts"] for h in (0.0, 0.5, 1.0)]
    assert abs(small[2]) < abs(small[1]) < abs(small[0])
    assert ldi.shock(p, 0.0, 0.01)["margin_call"] == 0.0 or ldi.overlay(p, 0.0)["swap_notional"] == 0.0
    assert ldi.shock(p, 1.0, 0.01)["margin_call"] > ldi.shock(p, 0.5, 0.01)["margin_call"]
    assert ldi.max_survivable_rise(p, 1.0) < ldi.max_survivable_rise(p, 0.6)


def test_survivable_rise_is_where_coverage_is_one():
    p = ldi.Plan()
    dy = ldi.max_survivable_rise(p, 0.9)
    assert ldi.shock(p, 0.9, dy)["coverage"] == pytest.approx(1.0, rel=1e-9)
    assert ldi.overlay(p, 0.9)["achieved_hedge_ratio"] == pytest.approx(0.9)


# ----------------------------------------------------------------------- FX
def test_min_variance_hedge_ratio_minimises_variance():
    n = 600
    s = RNG.normal(0, 0.025, n)
    r = -0.4 * s + RNG.normal(0.007, 0.04, n)
    lvl = pd.Series(np.cumprod(1 + np.r_[0, s]), index=pd.date_range("1975-01-31", periods=n + 1, freq="ME"))
    df = fx.cad_returns(pd.Series(np.r_[np.nan, r], index=lvl.index), lvl)
    h = fx.min_variance_ratio(df)
    v = lambda x: fx.hedged(df, x).var()
    assert v(h) <= min(v(h - 0.05), v(h + 0.05), v(0.0), v(1.0))


# ------------------------------------------------------------------ privates
def test_fund_returns_all_capital_and_irr_matches_growth():
    fm = privates.FundModel()
    sim = fm.simulate(1.0)
    assert sim["nav"][-1] == pytest.approx(0.0, abs=1e-12)          # fully distributed at the end of its life
    assert sim["paid_in"] < 1.0                                      # some commitment is never called
    tvpi, irr = privates.tvpi_irr(sim)
    assert tvpi > 1.0 and irr == pytest.approx(fm.growth, abs=1e-6)  # NAV compounds at G, so IRR = G


def test_pacing_reaches_its_target_and_the_denominator_effect():
    rate = privates.rate_for_target(0.25, years=40)
    path = privates.pace(rate, years=40)
    assert path["share"][-5:].mean() == pytest.approx(0.25, abs=0.005)
    crash = np.full(40, 0.06)
    crash[25] = -0.30
    shocked = privates.pace(rate, years=40, public_return=crash)
    assert shocked["share"][25] > path["share"][25] + 0.04


def test_geltner_recovers_volatility():
    true = RNG.normal(0.02, 0.08, 4000)
    out = privates.desmoothing_check(true, 0.6)
    assert out["vol_observed"] < 0.6 * out["vol_true"]
    assert out["vol_desmoothed"] == pytest.approx(out["vol_true"], rel=0.05)
    assert out["phi_hat"] == pytest.approx(0.6, abs=0.03)


# ------------------------------------------------------------- VaR backtest
def test_kupiec_and_christoffersen():
    from dpc.institutional.var_backtest import christoffersen, kupiec

    ok = np.zeros(1000, bool)
    ok[::100] = True                                   # exactly 1% and evenly spread
    k = kupiec(ok, 0.01)
    assert k["exceptions"] == 10 and k["lr"] == pytest.approx(0.0, abs=1e-9) and k["p_value"] == pytest.approx(1.0)
    bad = np.zeros(1000, bool)
    bad[:40] = True                                    # 4% and all in a row
    assert kupiec(bad, 0.01)["p_value"] < 1e-6
    assert christoffersen(bad)["p_value"] < 1e-6
    assert christoffersen(ok)["p_value"] > 0.05


def test_var_on_normal_returns_has_about_one_percent_exceptions():
    from dpc.institutional.var_backtest import backtest

    r = RNG.normal(0.01, 0.04, 6000)
    out = backtest(r, 0.99, 120)
    assert out["normal"]["rate"] == pytest.approx(0.01, abs=0.004)
    assert out["historical"]["mean_es"] > out["historical"]["mean_var"]
