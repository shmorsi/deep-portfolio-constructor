"""The risk study (docs/risk_protocol.md): realized volatility, the classical forecasters, the scores,
the portfolio rule and the decision rule, on made-up data with known answers."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dpc.risk import forecasters as fc
from dpc.risk import portfolio as pf
from dpc.risk import report, run
from dpc.risk import scores as sc
from dpc.risk.data import EWMA_DECAY, MIN_DAYS, MONTH_DAYS, build, ewma_month_end, monthly_rv
from dpc.wealth.data import IndustryPanel

ROOT = Path(__file__).resolve().parents[1]


def synthetic(T: int = 420, N: int = 5, seed: int = 0) -> tuple[IndustryPanel, pd.DataFrame]:
    """Daily returns with slowly moving volatility, and the monthly panel they compound to."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1930-01-31", periods=T, freq="ME")
    days = pd.bdate_range(dates[0].replace(day=1), dates[-1])
    month_of_day = days.to_period("M")
    log_vol = np.empty((T, N))
    log_vol[0] = np.log(0.01) + rng.normal(0, 0.3, N)
    for t in range(1, T):
        log_vol[t] = np.log(0.01) + 0.92 * (log_vol[t - 1] - np.log(0.01)) + rng.normal(0, 0.18, N) + rng.normal(0, 0.1)
    position = pd.Series(np.arange(T), index=dates.to_period("M"))
    vol = np.exp(log_vol)[position.reindex(month_of_day).to_numpy()]
    common = rng.normal(0, 0.004, len(days))[:, None]
    daily = pd.DataFrame(vol * rng.normal(0, 1, (len(days), N)) + common + 0.0003, index=days, columns=[f"I{i}" for i in range(N)])
    returns = ((1.0 + daily).groupby(month_of_day).prod() - 1.0).reindex(dates.to_period("M")).to_numpy()
    caps = np.empty((T, N))
    caps[0] = rng.uniform(50, 500, N)
    for t in range(1, T):
        caps[t] = caps[t - 1] * (1 + returns[t - 1])
    panel = IndustryPanel(dates=dates, names=list(daily.columns), returns=returns, cap_start=caps, bm=rng.uniform(0.3, 1.5, (T, N)),
                          rf=np.full(T, 0.003), mkt_rf=returns.mean(axis=1) - 0.003, smb=rng.normal(0, 0.02, T), hml=rng.normal(0, 0.02, T))
    return panel, daily


@pytest.fixture(scope="module")
def world():
    panel, daily = synthetic()
    return panel, build(panel, daily)


def test_realized_volatility_is_the_root_sum_of_squares_and_needs_fifteen_days():
    dates = pd.date_range("2000-01-31", periods=2, freq="ME")
    jan = pd.bdate_range("2000-01-03", "2000-01-31")
    feb = pd.bdate_range("2000-02-01", "2000-02-18")[:MIN_DAYS - 1]             # one day short
    daily = pd.DataFrame({"A": np.r_[np.full(len(jan), 0.01), np.full(len(feb), 0.02)]}, index=jan.append(feb))
    rv = monthly_rv(daily, dates)
    assert rv[0, 0] == pytest.approx(0.01 * np.sqrt(len(jan)))
    assert np.isnan(rv[1, 0])


def test_ewma_is_the_riskmetrics_recursion_at_the_last_day_of_the_month(world):
    panel, rp = world
    r = rp.daily["I0"].to_numpy()
    var = r[0] ** 2
    by_month = {}
    for day, x in zip(rp.daily.index, r):
        if day != rp.daily.index[0]:
            var = EWMA_DECAY * var + (1 - EWMA_DECAY) * x ** 2
        by_month[day.to_period("M")] = var
    t = 200
    assert rp.ewma[t, 0] == pytest.approx(np.sqrt(MONTH_DAYS * by_month[panel.dates[t].to_period("M")]))


def test_no_classical_forecast_uses_anything_after_its_month(world):
    panel, rp = world
    t0 = 300
    later = rp.daily.copy()
    later.loc[later.index > panel.dates[t0]] *= 3.0                               # a different future
    rp2 = build(panel, later)
    before, after = fc.point_forecasts(rp), fc.point_forecasts(rp2)
    for name in fc.CLASSICAL:
        np.testing.assert_array_equal(before[name][:t0 + 1], after[name][:t0 + 1])
        np.testing.assert_array_equal(fc.quantiles_from_errors(before[name], rp.rv)[:t0 + 1], fc.quantiles_from_errors(after[name], rp2.rv)[:t0 + 1])
        assert not np.array_equal(before[name][t0 + 1:], after[name][t0 + 1:])


def test_har_is_least_squares_on_earlier_months_only(world):
    _, rp = world
    t, i = 250, 2
    log_rv = np.log(rp.rv[:, i])
    x = fc.har_features(np.log(rp.rv))[:, i]
    rows = [s for s in range(t) if np.isfinite(x[s]).all()]                        # pairs (x_s, log rv_{s+1}) with s + 1 <= t
    beta = np.linalg.lstsq(x[rows], log_rv[np.asarray(rows) + 1], rcond=None)[0]
    assert fc.har(rp.rv)[t, i] == pytest.approx(float(np.exp(x[t] @ beta)), rel=1e-9)
    assert np.isnan(fc.har(rp.rv)[:max(fc.HAR_LAGS) - 1]).all()


def test_quantiles_are_the_point_times_its_last_120_error_quantiles(world):
    _, rp = world
    point = fc.trailing_mean(rp.rv)
    q = fc.quantiles_from_errors(point, rp.rv)
    t, i = 260, 1
    errors = np.log(rp.rv[t - fc.ERROR_WINDOW + 1:t + 1, i]) - np.log(point[t - fc.ERROR_WINDOW:t, i])   # forecasts of months t-119 .. t
    np.testing.assert_allclose(q[t, i], point[t, i] * np.exp(np.quantile(errors, fc.LEVELS)))
    assert np.isnan(q[fc.MEAN_WINDOW - 1 + fc.ERROR_WINDOW - 1]).all() and np.isfinite(q[fc.MEAN_WINDOW - 1 + fc.ERROR_WINDOW]).all()
    assert (np.diff(q[t], axis=1) >= 0).all()


def test_scores_on_a_worked_example():
    y = np.array([[2.0, 1.0]])
    q = np.tile(np.linspace(0.6, 1.4, 9), (1, 2, 1)) * np.array([2.0, 2.0])[None, :, None]     # medians 2 and 2; bands [1.2, 2.8]
    s = sc.month_sums(y, q)
    d1, d2, lv = 2.0 - q[0, 0], 1.0 - q[0, 1], np.asarray(fc.LEVELS)
    pin = np.maximum(lv * d1, (lv - 1) * d1).sum() + np.maximum(lv * d2, (lv - 1) * d2).sum()
    assert s["wql_num"][0] == pytest.approx(2 * pin / 9)
    p = sc.pooled(s)
    assert p["wql"] == pytest.approx(2 * pin / 9 / 3.0) and p["mae"] == pytest.approx(0.5) and p["cover80"] == pytest.approx(0.5)
    assert p["qlike"] == pytest.approx((0.0 + (0.25 - np.log(0.25) - 1.0)) / 2)
    perfect = sc.pooled(sc.month_sums(y, np.repeat(y[:, :, None], 9, axis=2)))
    assert perfect["wql"] == 0 and perfect["mae"] == 0 and perfect["qlike"] == pytest.approx(0)


def test_a_ratio_to_itself_is_one_and_better_needs_the_whole_interval_below_one():
    rng = np.random.default_rng(1)
    a = rng.uniform(1, 2, 240)
    idx = sc.block_indices(240, 12, 300, seed=42)
    assert idx.shape == (300, 240) and (np.diff(idx[:, :12], axis=1) == 1).all()
    np.testing.assert_array_equal(idx, sc.block_indices(240, 12, 300, seed=42))
    assert sc.ratio_ci(a, a, idx) == (1.0, 1.0, 1.0)
    ratio, lo, hi = sc.ratio_ci(0.9 * a, a, idx)
    assert (ratio, lo, hi) == pytest.approx((0.9, 0.9, 0.9)) and sc.verdict_ratio(lo, hi) == "better"
    assert sc.verdict_ratio(0.95, 1.02) == "no detectable difference" and sc.verdict_ratio(1.01, 1.2) == "worse"


def test_minimum_variance_respects_the_mandate_and_leans_to_the_calm_industries():
    n = 8
    solver = pf.MinVariance(n, cap=0.2)
    equal = solver.solve(np.full(n, 0.05), np.eye(n))
    np.testing.assert_allclose(equal, np.full(n, 1 / n), atol=1e-5)
    sigma = np.array([0.02, 0.03, 0.05, 0.05, 0.06, 0.08, 0.10, 0.12])
    w = solver.solve(sigma, np.eye(n))
    assert w.sum() == pytest.approx(1) and (w >= 0).all() and w.max() <= 0.2 + 1e-6
    assert w[0] == pytest.approx(0.2, abs=1e-5) and (np.diff(w) <= 1e-6).all()           # the cap binds on the calmest; weights fall as risk rises
    np.testing.assert_allclose(solver.solve(10 * sigma, np.eye(n)), w, atol=1e-5)          # the level of risk does not matter, only its shape
    np.testing.assert_allclose(pf.inverse_volatility(sigma), (1 / sigma) / (1 / sigma).sum())


def test_realized_variance_follows_weights_that_drift_inside_the_month():
    days = pd.bdate_range("2001-03-01", "2001-03-30")
    r = np.column_stack([np.full(len(days), 0.01), np.full(len(days), -0.02)])
    daily = pd.DataFrame(r, index=days, columns=["A", "B"])
    dates = pd.DatetimeIndex([pd.Timestamp("2001-03-31")])
    assert pf.realized_variance(daily, dates, np.array([[1.0, 0.0]]))[0] == pytest.approx(len(days) * 0.01 ** 2)
    value, total, want = np.array([0.5, 0.5]), 1.0, 0.0
    for _ in days:
        value = value * (1 + np.array([0.01, -0.02]))
        want += (value.sum() / total - 1) ** 2
        total = value.sum()
    assert pf.realized_variance(daily, dates, np.array([[0.5, 0.5]]))[0] == pytest.approx(want)


def test_the_risk_verdict_needs_both_the_t_and_the_interval():
    rng = np.random.default_rng(3)
    base = np.exp(rng.normal(-6, 0.3, 480))
    lower = pf.compare_variance(base * np.exp(-0.2 + rng.normal(0, 0.05, 480)), base, 12, 12, 400, 42)
    assert lower["verdict"] == "lower risk" and lower["t_nw"] < -2 and lower["ci_high"] < 0
    assert lower["volatility_change"] == pytest.approx(np.exp(lower["diff_log_variance"] / 2) - 1)
    assert pf.compare_variance(base * np.exp(rng.normal(0, 0.3, 480)), base, 12, 12, 400, 42)["verdict"] == "no detectable difference"
    assert pf.compare_variance(base * np.exp(0.2 + rng.normal(0, 0.05, 480)), base, 12, 12, 400, 42)["verdict"] == "higher risk"


def test_forecast_files_round_trip_and_refuse_other_months_or_crossing_quantiles(tmp_path, world):
    _, rp = world
    idx = fc.test_origins(rp, "1950-01")
    origins = run.origin_labels(rp, idx)
    q = fc.quantiles_from_errors(fc.har(rp.rv), rp.rv)[idx]
    fc.save_forecast(tmp_path / "har.npz", origins, rp.names, q, {"kind": "classical"})
    back = fc.load_forecast(tmp_path / "har.npz", origins, rp.names)
    np.testing.assert_allclose(back.q, q, rtol=1e-6)
    assert back.meta == {"kind": "classical"} and back.median.shape == (len(idx), len(rp.names))
    with pytest.raises(ValueError, match="months or industries"):
        fc.load_forecast(tmp_path / "har.npz", origins[1:], rp.names)
    fc.save_forecast(tmp_path / "bad.npz", origins, rp.names, q[:, :, ::-1], {})
    with pytest.raises(ValueError, match="cross"):
        fc.load_forecast(tmp_path / "bad.npz", origins, rp.names)


def _forecasts(rp, idx, with_model: bool) -> dict[str, fc.Forecast]:
    origins = run.origin_labels(rp, idx)
    out = {name: fc.Forecast(name, origins, rp.names, fc.quantiles_from_errors(point, rp.rv)[idx], {"kind": "classical"})
           for name, point in fc.point_forecasts(rp).items()}
    if with_model:                                                                  # a stand-in for the model: HAR with noise, so it cannot beat HAR
        rng = np.random.default_rng(9)
        noisy = out["har"].q * np.exp(rng.normal(0, 0.4, out["har"].q.shape[:2]))[:, :, None]
        refits = [{"refit": "1949-12", "kept_step": 0, "val_wql": {"0": 0.3}}, {"refit": "1954-12", "kept_step": 50, "val_wql": {"0": 0.3, "50": 0.29}}]
        out["tsfm_zero_cpu_s0"] = fc.Forecast("tsfm_zero_cpu_s0", origins, rp.names, noisy, {"kind": "tsfm", "variant": "zero"})
        out["tsfm_ft_cpu_s0"] = fc.Forecast("tsfm_ft_cpu_s0", origins, rp.names, noisy, {"kind": "tsfm", "variant": "ft", "refits": refits, "tsfm_commit": "abc1234"})
    return out


@pytest.fixture(scope="module")
def study(cfg, world):
    panel, rp = world
    small = cfg.replace(wealth=dataclasses.replace(cfg.wealth, oos_start="1950-01",
                                                   evaluation=dataclasses.replace(cfg.wealth.evaluation, bootstrap_draws=200)))
    idx = fc.test_origins(rp, "1950-01")
    summary, runs = run.evaluate(small, panel, rp, _forecasts(rp, idx, with_model=True))
    return summary, runs, idx


def test_the_study_scores_every_forecaster_and_applies_the_decision_rule(study, world):
    summary, runs, idx = study
    _, rp = world
    assert summary["window"] == {"first_month": "1950-01", "last_month": f"{rp.dates[-1]:%Y-%m}", "months": len(idx), "industries": 5}
    assert set(summary["accuracy"]) == {"last", "mean60", "ewma", "har", "tsfm_zero_cpu_s0", "tsfm_ft_cpu_s0"}
    assert summary["accuracy"]["har"]["vs_har"]["wql"]["ratio"] == 1.0
    noisy = summary["accuracy"]["tsfm_ft_cpu_s0"]["vs_har"]["wql"]
    assert noisy["ratio"] > 1 and noisy["verdict"] == "worse"
    cpu, t4 = summary["models"]["cpu_s0"], summary["models"]["patch_s2"]
    assert cpu["status"] == "run" and cpu["earns_its_place"] is False and cpu["refits"] == 2 and cpu["refits_kept_pretrained"] == 1
    assert t4["status"] == "not run" and "earns_its_place" not in t4
    for rule in run.RULES:
        P = summary["portfolios"][rule]
        assert set(P) == set(summary["accuracy"]) | {"oracle"} and "vs_har" not in P["har"]
        assert all(len(runs[f"{rule}_{name}"]) == len(idx) for name in P)
    oracle = summary["portfolios"]["min_variance"]["oracle"]["vs_har"]
    assert oracle["diff_log_variance"] < 0                                         # knowing next month's volatility cannot hurt this rule here
    assert summary["references"]["market"]["realized_volatility"] > 0
    assert sum(len(v) for v in summary["decades_wql_vs_har"].values()) == len(summary["decades_wql_vs_har"]) * 6


def test_every_forecaster_would_earn_its_place_only_through_both_tests(study):
    summary, _, _ = study
    s = json.loads(json.dumps(summary))
    s["accuracy"]["tsfm_ft_cpu_s0"]["vs_har"]["wql"]["verdict"] = "better"
    s["models"]["cpu_s0"]["earns_its_place"] = False                                # better forecasts alone are not enough
    assert "**No** for the CPU model" in report.model_sentence(s, "cpu_s0")
    assert "has not been run" in report.model_sentence(s, "patch_s2")


def test_documents_are_rendered_from_the_summary_alone(study):
    summary, _, _ = study
    s = json.loads(json.dumps(summary))                                             # what a reader of summary.json gets
    block = report.readme_section(s, "abc1234")
    assert block.startswith(report.START) and block.endswith(report.END)
    assert f"{s['window']['months']} months" in block and "HAR regression" in block and "Oracle (not investable)" in block
    md = report.results_md(s, "abc1234")
    assert "## Inverse-volatility portfolio" in md and "## Quantile loss by decade" in md and "commit abc1234" in md
    site = report.export(s, "abc1234")
    assert site["stats"]["statusPatchS2"]["display"] == "not run" and site["stats"]["earnsCpuS0"]["display"] == "no"
    assert site["stats"]["wqlFtCpuS0"]["display"].endswith("×") and all(v["source"].startswith("deep-portfolio-constructor") for v in site["stats"].values())


def test_committed_documents_match_the_committed_summary():
    """Once the study has been run: results.md, the site export and the README block are exactly what summary.json renders."""
    out = ROOT / "reports" / "risk"
    if not (out / "summary.json").is_file():
        pytest.skip("the risk study has not been run yet")
    s = json.loads((out / "summary.json").read_text())
    site = json.loads((out / "site_risk.json").read_text())
    commit = site["commit"]
    assert site == json.loads(json.dumps(report.export(s, commit), ensure_ascii=False))
    assert (out / "results.md").read_text() == report.results_md(s, commit) + "\n"
    readme = (ROOT / "README.md").read_text()
    assert report.START + readme.split(report.START, 1)[1].split(report.END, 1)[0] + report.END == report.readme_section(s, commit)


def _script():
    spec = importlib.util.spec_from_file_location("risk_tsfm_forecast", ROOT / "scripts" / "risk_tsfm_forecast.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                                                 # numpy only: tsfm and torch are imported inside main()
    return module


def test_the_forecast_script_keeps_the_protocol_and_this_package_in_step(world):
    script = _script()
    _, rp = world
    dates = [f"{d:%Y-%m}" for d in rp.dates]
    assert script.LEVELS == fc.LEVELS and script.OOS_START == "1960-01"
    assert script.PROTOCOL_SHA256 == {name: info["sha256"] for name, info in run.MODELS.items()}
    protocol = (ROOT / "docs" / "risk_protocol.md").read_text()
    assert all(digest in protocol for digest in script.PROTOCOL_SHA256.values())
    script.OOS_START = "1950-01"
    idx = script.origin_indices(dates)
    np.testing.assert_array_equal(idx, fc.test_origins(rp, "1950-01"))
    refits = script.refit_indices(dates, idx)
    assert [dates[t] for t in refits] == ["1949-12", "1954-12", "1959-12"]             # the panel's last month, 1964-12, is scored but decides nothing
    hist = script.history(np.where(np.arange(len(dates))[:, None] < 3, np.nan, rp.rv), 10)
    assert all(len(h) == 8 and np.isfinite(h).all() for h in hist)                  # months 3..10: nothing after the decision, nothing before the first observation
    y = rp.rv[idx + 1]
    q = fc.quantiles_from_errors(fc.har(rp.rv), rp.rv)[idx]
    assert script.pooled_wql(y, q) == pytest.approx(sc.pooled(sc.month_sums(y, q))["wql"])
    assert re.search(r"at most 300 steps, validated every 25", protocol) and (script.STEPS, script.EVAL_EVERY, script.BATCH, script.LEARNING_RATE) == (300, 25, 64, 1e-4)
