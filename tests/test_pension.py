"""Tests for the pension study (src/dpc/pension). None of them needs the network or the real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from dpc.pension import evaluate as ev, scenarios as sc
from dpc.pension.fixed_income import convexity, duration, par_bond_return, present_value
from dpc.pension.liability import Schedule, paths
from dpc.pension.plan import Plan, exogenous, run
from dpc.pension.policy import Ensemble, Fixed, Glide, Learned, baselines
from dpc.pension.train import TrainConfig, score_t, train

T = 36  # a short horizon keeps the tests fast; the mathematics does not depend on it


def make_panel(months: int = 600, seed: int = 0) -> pd.DataFrame:
    """A synthetic monthly panel with the columns data.py produces."""
    rng = np.random.default_rng(seed)
    y = np.empty(months)
    y[0] = 0.05
    for t in range(1, months):
        y[t] = max(0.002, 0.0005 + 0.99 * y[t - 1] + 0.0025 * rng.standard_normal())
    infl = 0.0025 + 0.002 * rng.standard_normal(months)
    idx = pd.date_range("1953-04-30", periods=months, freq="ME")
    return pd.DataFrame({"equity": 0.008 + 0.045 * rng.standard_normal(months), "yield": y, "infl": infl, "cpi": 100 * np.cumprod(1 + infl)}, index=idx)


def flat_scenarios(n: int, y: np.ndarray, infl: float = 0.0, equity: float = 0.0, past_infl: float | None = None) -> dict[str, np.ndarray]:
    """Scenarios with a given yield path (after a flat run-up), constant stock returns, and constant inflation
    (`past_infl` before the plan starts, `infl` after; the same by default)."""
    total = sc.BURN + T
    full = np.concatenate([np.full((n, sc.BURN), y[0]), np.broadcast_to(y, (n, T + 1))], axis=1)
    pi = np.full((n, total), infl)
    pi[:, :sc.BURN] = infl if past_infl is None else past_infl
    return {"yield": full, "equity": np.full((n, total), equity), "infl": pi}


@pytest.fixture(scope="module")
def plan() -> Plan:
    return Plan(horizon=T)


# ---------------------------------------------------------------- bond mathematics

def test_a_par_bond_earns_its_coupon_when_the_yield_does_not_move():
    assert par_bond_return(np.array([0.06]), np.array([0.06]))[0] == pytest.approx(0.06 / 12, abs=1e-12)


def test_a_bond_loses_about_duration_times_the_rise_in_yield():
    up = par_bond_return(np.array([0.05]), np.array([0.0510]))[0] - 0.05 / 12
    # a 10-year par bond at 5% has a modified duration near 7.8 years
    assert up == pytest.approx(-7.75 * 0.0010, abs=2e-4)
    assert par_bond_return(np.array([0.05]), np.array([0.04]))[0] > par_bond_return(np.array([0.05]), np.array([0.05]))[0]


def test_duration_and_convexity_match_finite_differences():
    cash, y, h = Schedule().cash(), 0.04, 1e-5
    pv = present_value(cash, y)
    slope = (present_value(cash, y + h) - present_value(cash, y - h)) / (2 * h)
    curve = (present_value(cash, y + h) - 2 * pv + present_value(cash, y - h)) / h ** 2
    assert duration(cash, y) == pytest.approx(-slope / pv, rel=1e-6)
    assert convexity(cash, y) == pytest.approx(curve / pv, rel=1e-4)
    assert 12 < duration(cash, y) < 18   # a typical pension liability


def test_the_closed_form_liability_value_matches_discounting_each_payment():
    s = Schedule()
    k = np.arange(1, s.months + 1)
    for y, pe in ((0.01, 0.0), (0.04, 0.02), (0.09, 0.06), (0.02, 0.05)):   # the last has a negative real yield
        grown = s.cash() * (1 + pe / 12) ** k          # each payment in the money of the month it is paid
        assert float(s.unit_value(y, pe)) == pytest.approx(present_value(grown, y), rel=1e-10)
        later = s.cash()[37:] * (1 + pe / 12) ** np.arange(1, s.months - 36)
        assert float(s.unit_value(y, pe, elapsed=37)) == pytest.approx(present_value(later, y), rel=1e-10)
    assert float(s.unit_value(0.03, 0.03 + 12 * (1 / s.g - 1) * (1 + 0.03 / 12))) == pytest.approx(s.months * 1 / s.g, rel=1e-6)   # ratio of one


# ---------------------------------------------------------------- the liability

def test_liability_rolls_forward_at_the_yield_when_nothing_changes():
    y = np.full((1, T + 1), 0.05)
    out = paths(Schedule(), y, np.zeros((1, T)), np.zeros((1, T + 1)))
    L, B = out["liability"][0], out["benefit"][0]
    assert L[0] == pytest.approx(100.0)
    assert np.allclose(L[1:], L[:-1] * (1 + 0.05 / 12) - B, rtol=1e-10)
    assert np.allclose(out["matching"], 0.05 / 12, atol=1e-12)


def test_expected_inflation_is_already_in_the_price_and_a_surprise_is_not():
    y = np.full((1, T + 1), 0.05)
    exp = np.full((1, T + 1), 0.024)
    as_expected = paths(Schedule(), y, np.full((1, T), 0.002), exp)        # 0.2% a month is the 2.4% a year that was expected
    L, B, R = as_expected["liability"][0], as_expected["benefit"][0], as_expected["matching"][0]
    assert np.allclose(L[1:], L[:-1] * (1 + R) - B, rtol=1e-10)            # matched assets keep up with the promise exactly
    hot = paths(Schedule(), y, np.full((1, T), 0.006), exp)                # inflation runs above what was expected
    assert np.allclose(hot["matching"], as_expected["matching"])           # the bonds pay what they pay
    assert (hot["liability"][0, 1:] > as_expected["liability"][0, 1:]).all()   # and the promise has grown past them
    assert paths(Schedule(), y, np.zeros((1, T)), exp)["liability"][0, 0] == pytest.approx(100.0)


# ---------------------------------------------------------------- the plan

def test_a_fully_matched_fully_funded_plan_stays_funded_whatever_yields_do(plan):
    rng = np.random.default_rng(1)
    y = np.clip(0.05 + np.cumsum(0.004 * rng.standard_normal(T + 1)), 0.005, 0.15)
    exo = exogenous(flat_scenarios(4, y), plan)
    full = Plan(funded0=1.0, horizon=T)
    out = run(Fixed(0.0, 0.0, 1.0), exo, full)
    assert torch.allclose(out["funded"], torch.ones_like(out["funded"]), atol=2e-4)
    assert float(out["cost"].max()) < 1e-3


def test_inflation_that_was_expected_does_not_hurt_a_matched_plan(plan):
    rng = np.random.default_rng(6)
    y = np.clip(0.05 + np.cumsum(0.004 * rng.standard_normal(T + 1)), 0.005, 0.15)
    exo = exogenous(flat_scenarios(2, y, infl=0.003), plan)                 # 3.6% a year, before and after the start
    out = run(Fixed(0.0, 0.0, 1.0), exo, Plan(funded0=1.0, horizon=T))
    assert torch.allclose(out["funded"], torch.ones_like(out["funded"]), atol=3e-4)


def test_an_inflation_surprise_erodes_a_matched_plan_and_the_sponsor_pays(plan):
    y = np.full(T + 1, 0.05)
    exo = exogenous(flat_scenarios(2, y, infl=0.006, past_infl=0.002), plan)   # prices rise three times as fast as they used to
    out = run(Fixed(0.0, 0.0, 1.0), exo, Plan(funded0=1.0, horizon=T))
    assert float(out["funded"][:, -1].max()) < 1.0
    assert float(out["cost"].min()) > 0.0


def test_the_cost_is_the_contributions_plus_what_is_left(plan):
    exo = exogenous(sc.simulate(sc.fit(make_panel()), 64, T, seed=3), plan)
    out = run(Glide(), exo, plan)
    assert torch.allclose(out["cost"], out["contrib"].sum(dim=1) + out["left"], atol=1e-6)
    assert float(out["contrib"].min()) >= 0.0


def test_a_plan_that_is_never_short_costs_nothing(plan):
    exo = exogenous(flat_scenarios(2, np.full(T + 1, 0.05)), plan)
    out = run(Fixed(0.0, 0.0, 1.0), exo, Plan(funded0=2.0, horizon=T))
    assert float(out["cost"].abs().max()) == 0.0


def test_trading_costs_are_charged_only_when_the_weights_move(plan):
    exo = exogenous(sc.simulate(sc.fit(make_panel()), 32, T, seed=4), plan)
    free = run(Glide(), exo, Plan(horizon=T, cost_bps=0.0))
    paid = run(Glide(), exo, Plan(horizon=T, cost_bps=50.0))
    assert float((paid["cost"] - free["cost"]).min()) >= -1e-6
    assert float((paid["cost"] - free["cost"]).mean()) > 0.0


# ---------------------------------------------------------------- policies

def test_every_policy_is_long_only_and_fully_invested():
    state = torch.rand(200, 10) * torch.tensor([1.5, 0.15, 0.04, 0.6, 0.1, 0.08, 1.0, 1.0, 1.0, 1.0]) + torch.tensor([0.3, 0, -0.02, -0.3, 0, 0, 0, 0, 0, 0])
    net = Learned()
    torch.nn.init.normal_(net.net[-1].weight, std=0.5)
    for policy in [*baselines().values(), net, Ensemble([net, Learned()])]:
        w = policy(state).detach()
        assert w.shape == (200, 3)
        assert float(w.min()) >= 0.0
        assert torch.allclose(w.sum(dim=1), torch.ones(200), atol=1e-5)


def test_the_glide_path_holds_fewer_stocks_as_the_plan_fills():
    state = torch.zeros(4, 10)
    state[:, 0] = torch.tensor([0.6, 0.8, 0.95, 1.3])
    stocks = Glide()(state)[:, 0]
    assert stocks.tolist() == pytest.approx([0.70, 0.70, 0.45, 0.20], abs=1e-6)


def test_the_learned_policy_starts_next_to_the_glide_path():
    state = torch.rand(100, 10)
    state[:, 0] += 0.4
    assert float((Learned()(state) - Glide()(state)).abs().max()) < 0.04


def test_a_decision_never_depends_on_what_comes_after_it(plan):
    """Change every number after month 20: the weights chosen up to month 20 must not move."""
    scen = sc.simulate(sc.fit(make_panel()), 16, T, seed=5)
    later = {k: v.copy() for k, v in scen.items()}
    cut = sc.BURN + 20
    later["equity"][:, cut:] += 0.07
    later["infl"][:, cut:] += 0.01
    later["yield"][:, cut + 1:] += 0.02
    net = Learned()
    torch.nn.init.normal_(net.net[-1].weight, std=0.5)
    with torch.no_grad():
        a, b = run(net, exogenous(scen, plan), plan), run(net, exogenous(later, plan), plan)
    assert torch.allclose(a["weights"][:, :21], b["weights"][:, :21], atol=1e-6)
    assert not torch.allclose(a["weights"][:, 22:], b["weights"][:, 22:], atol=1e-6)


# ---------------------------------------------------------------- scenarios

def test_the_generator_is_reproducible_and_keeps_yields_above_the_floor():
    model = sc.fit(make_panel())
    a, b, c = sc.simulate(model, 50, T, seed=9), sc.simulate(model, 50, T, seed=9), sc.simulate(model, 50, T, seed=10)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["equity"], c["equity"])
    assert a["yield"].shape == (50, sc.BURN + T + 1) and a["equity"].shape == (50, sc.BURN + T)
    assert float(a["yield"].min()) >= sc.Y_FLOOR


def test_the_ar1_fit_recovers_a_known_process():
    rng = np.random.default_rng(2)
    x = np.empty(20000)
    x[0] = 0.0
    for t in range(1, len(x)):
        x[t] = 0.002 + 0.9 * x[t - 1] + 0.01 * rng.standard_normal()
    a, phi, resid = sc._ar1(x)
    assert phi == pytest.approx(0.9, abs=0.01) and a == pytest.approx(0.002, abs=0.0005)
    assert resid.std() == pytest.approx(0.01, rel=0.05)


def test_the_model_risk_generator_gives_the_yield_no_pull_to_an_average():
    panel = make_panel()
    walk = sc.fit(panel, random_walk=True)
    assert walk.phi_y == 1.0 and walk.a_y == 0.0 and np.isnan(walk.mean_yield)
    assert np.allclose(walk.shocks[:, 1], np.diff(panel["yield"].to_numpy()))
    assert np.array_equal(walk.shocks[:, 0], sc.fit(panel).shocks[:, 0])   # the same stock returns either way


def test_scenarios_only_rearrange_shocks_that_happened():
    model = sc.fit(make_panel())
    drawn = sc.simulate(model, 20, T, seed=1)["equity"]
    assert np.isin(drawn, model.shocks[:, 0]).all()


def test_history_is_read_back_exactly():
    panel = make_panel()
    got = sc.from_history(panel, 300, T)
    assert np.array_equal(got["yield"][0], panel["yield"].to_numpy()[300 - sc.BURN:300 + T + 1])
    assert np.array_equal(got["equity"][0], panel["equity"].to_numpy()[300 - sc.BURN + 1:300 + T + 1])
    with pytest.raises(ValueError):
        sc.from_history(panel, 5, T)


# ---------------------------------------------------------------- scoring

def test_the_tail_cost_is_the_average_of_the_worst_five_percent():
    cost = np.arange(100, dtype=float)
    assert ev.cvar(cost) == pytest.approx(np.mean([95, 96, 97, 98, 99]))
    assert ev.score(cost) == pytest.approx(cost.mean() + 97.0)
    assert float(score_t(torch.tensor(cost, dtype=torch.float32))) == pytest.approx(ev.score(cost), rel=1e-6)


def test_the_verdict_follows_the_interval():
    rng = np.random.default_rng(3)
    base = rng.random(2000)
    assert ev.paired(base, base, 200, seed=1)["verdict"] == "no detectable difference"
    assert ev.paired(base - 0.2, base, 200, seed=1)["verdict"] == "better"
    assert ev.paired(base + 0.2, base, 200, seed=1)["verdict"] == "worse"
    assert ev.paired_mean_blocks(base[:30] - 0.2, base[:30], 10, 200, seed=1)["verdict"] == "better"


def test_training_runs_and_does_not_make_the_validation_score_worse(plan):
    net, history = train(sc.fit(make_panel()), plan, seed=1, cfg=TrainConfig(steps=6, batch=64, n_train=256, n_val=256, check_every=3))
    scores = [h["validation_score"] for h in history]
    assert len(scores) == 3 and all(np.isfinite(scores))
    assert isinstance(net, Learned)
    # the network returned is the best one seen, so it is never worse than where it started
    assert min(scores) <= scores[0] + 1e-9


# ---------------------------------------------------------------- data

def test_the_panel_uses_month_end_yields_from_1962_and_monthly_averages_before(tmp_path):
    from dpc.pension import data

    days = pd.date_range("1961-12-01", "1962-03-31", freq="B")
    (tmp_path / "dgs10_daily.csv").write_text("observation_date,DGS10\n" + "\n".join(f"{d:%Y-%m-%d},{4 + d.day / 100:.2f}" for d in days))
    months = pd.date_range("1953-04-01", "1962-03-01", freq="MS")
    (tmp_path / "gs10_monthly.csv").write_text("observation_date,GS10\n" + "\n".join(f"{d:%Y-%m-%d},3.00" for d in months))
    cpi_months = pd.date_range("1953-03-01", "1962-03-01", freq="MS")
    (tmp_path / "cpi_monthly.csv").write_text("observation_date,CPIAUCNS\n" + "\n".join(f"{d:%Y-%m-%d},{26 * 1.002 ** i:.4f}" for i, d in enumerate(cpi_months)))
    (tmp_path / "french_factors.csv").write_text("header\n,Mkt-RF,SMB,HML,RF\n" + "\n".join(f"{d:%Y%m},1.00,0.00,0.00,0.20" for d in months) + "\n\nAnnual\n1953,10,0,0,2\n")
    panel = data.build({"DGS10": tmp_path / "dgs10_daily.csv", "GS10": tmp_path / "gs10_monthly.csv", "CPIAUCNS": tmp_path / "cpi_monthly.csv", "french": tmp_path / "french_factors.csv"})
    assert panel.index[0] == pd.Timestamp("1953-04-30") and panel.index[-1] == pd.Timestamp("1962-03-31")
    assert panel.loc["1961-12-31", "yield"] == pytest.approx(0.03)                      # the monthly average
    assert panel.loc["1962-01-31", "yield"] == pytest.approx((4 + 31 / 100) / 100)      # the last trading day of January 1962
    assert panel["equity"].iloc[0] == pytest.approx(0.012)
    assert panel["infl"].iloc[5] == pytest.approx(0.002, abs=1e-4)   # the fake index is written to four decimals


def test_anatomy_buckets_average_within_each_range():
    from dpc.pension.anatomy import bucket_means

    key = np.array([0.5, 0.75, 0.78, 0.95, 2.0])
    w = np.array([[1, 0, 0], [0.5, 0, 0.5], [0.3, 0, 0.7], [0, 1, 0], [0, 0, 1]], dtype=float)
    out = bucket_means(key, (0.70, 0.80, 0.90), {"w": w})
    assert out["share_of_months"] == pytest.approx([0.2, 0.4, 0.0, 0.4])
    assert out["w"][0] == pytest.approx([1, 0, 0])
    assert out["w"][1] == pytest.approx([0.4, 0, 0.6])
    assert out["w"][2] is None                       # nothing fell in 80% to 90%
    assert out["w"][3] == pytest.approx([0, 0.5, 0.5])
    kept = bucket_means(key, (0.70, 0.80, 0.90), {"w": w}, keep=key < 1.0)
    assert kept["share_of_months"] == pytest.approx([0.25, 0.5, 0.0, 0.25])


def test_anatomy_describes_the_glide_path_as_itself(plan):
    """An untrained network is the glide path with a 2% floor, so its description must sit on the glide path's."""
    from dpc.pension.anatomy import describe
    from dpc.pension.policy import Ensemble, Learned

    exo = exogenous(sc.simulate(sc.fit(make_panel()), 40, plan.horizon, seed=5), plan)
    d = describe(Ensemble([Learned()]), exo, plan)
    for learned, glide in zip(d["by_funded"]["learned"], d["by_funded"]["glide"]):
        if learned is not None:
            assert sum(learned) == pytest.approx(1.0, abs=1e-6) and sum(glide) == pytest.approx(1.0, abs=1e-6)
            assert learned[0] == pytest.approx(glide[0], abs=0.03)
    assert sum(d["by_funded"]["share_of_months"]) == pytest.approx(1.0)
    assert d["curve"]["glide"][0] == pytest.approx([0.7, 0.0, 0.3], abs=1e-6)


def test_the_tunable_glide_path_starts_as_the_protocols_glide_path():
    from dpc.pension.anatomy import TunedGlide

    rule = TunedGlide()
    n = {k: float(v.detach()) for k, v in rule.numbers().items()}
    assert n == pytest.approx({"slide_from": 0.80, "slide_to": 1.10, "stocks_when_short": 0.70, "stocks_when_funded": 0.20}, abs=1e-6)
    state = torch.zeros(7, 10)
    state[:, 0] = torch.tensor([0.5, 0.8, 0.9, 1.0, 1.1, 1.3, 2.0])
    with torch.no_grad():
        assert torch.allclose(rule(state), Glide()(state), atol=1e-6)


def test_tuning_the_glide_path_does_not_make_its_validation_score_worse(plan):
    from dpc.pension.anatomy import tune_glide

    model = sc.fit(make_panel())
    cfg = TrainConfig(steps=6, batch=64, n_train=128, n_val=128, check_every=3)
    rule = tune_glide(model, plan, seed=1, cfg=cfg)
    val = exogenous(sc.simulate(model, cfg.n_val, plan.horizon, seed=5001), plan)
    with torch.no_grad():
        assert float(score_t(run(rule, val, plan)["cost"])) <= float(score_t(run(Glide(), val, plan)["cost"])) + 1e-6
        w = rule(torch.rand(20, 10) * 2)
    assert float(w.min()) >= 0 and torch.allclose(w.sum(dim=1), torch.ones(20), atol=1e-6)
