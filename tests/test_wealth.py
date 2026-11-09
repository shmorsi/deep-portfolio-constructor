"""Tests for the long-only wealth study: data parsing, timing, look-ahead, simulator, model, baselines."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from dpc.wealth.baselines import growth_optimizer, min_variance, momentum_tilt, risk_parity, tilt_bounds, _cov
from dpc.wealth.data import IndustryPanel, parse_sections
from dpc.wealth.engine import run
from dpc.wealth.features import build_features
from dpc.wealth.model import DeepGrowthNet, crra, simulate
from dpc.wealth.train import TrainSettings, Variant, refit_points, Tensors, train_member


def synthetic_panel(T: int = 240, N: int = 6, seed: int = 0) -> IndustryPanel:
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.008, 0.045, T)
    returns = mkt[:, None] * rng.uniform(0.6, 1.4, N) + rng.normal(0.0, 0.03, (T, N))
    caps = np.empty((T, N))
    caps[0] = rng.uniform(50, 500, N)
    for t in range(1, T):
        caps[t] = caps[t - 1] * (1 + returns[t - 1])
    dates = pd.date_range("1950-01-31", periods=T, freq="ME")
    return IndustryPanel(dates=dates, names=[f"I{i}" for i in range(N)], returns=returns, cap_start=caps,
                         bm=rng.uniform(0.3, 1.5, (T, N)), rf=np.full(T, 0.003), mkt_rf=mkt - 0.003,
                         smb=rng.normal(0, 0.02, T), hml=rng.normal(0, 0.02, T))


def test_parse_sections_reads_titled_and_untitled_tables():
    text = ("This file was created\n\n  Average Value Weighted Returns -- Monthly\n,A,B\n192607,  1.0, -2.0\n192608, -99.99, 3.5\n\n"
            "Some note\n\n,X,Y\n192607, 0.1, 0.2\n\n  Annual Factors\n,X,Y\n  1927, 1.0, 2.0\n")
    s = parse_sections(text)
    vw = s["Average Value Weighted Returns -- Monthly"]
    assert list(vw.columns) == ["A", "B"] and np.isnan(vw.loc["192608", "A"]) and vw.loc["192608", "B"] == 3.5
    assert s["Some note"].loc["192607", "Y"] == 0.2
    assert s["Annual Factors"].index.tolist() == ["1927"]


def test_market_weights_use_caps_known_at_the_decision():
    p = synthetic_panel()
    w = p.market_weights()
    np.testing.assert_allclose(w.sum(1), 1.0)
    np.testing.assert_allclose(w[:-1], p.cap_start[1:] / p.cap_start[1:].sum(1, keepdims=True))


def test_features_are_point_in_time():
    p = synthetic_panel(T=200)
    full = build_features(p)
    for cut in (90, 140, 199):
        short = IndustryPanel(**{f.name: (getattr(p, f.name)[:cut + 1] if f.name not in ("names",) else p.names)
                                 for f in dataclasses.fields(p)})
        part = build_features(short)
        np.testing.assert_allclose(part.assets[cut], full.assets[cut], atol=1e-10)
        np.testing.assert_allclose(part.context[cut], full.context[cut], atol=1e-10)


def test_buy_and_hold_drifts_and_trades_once():
    p = synthetic_panel()
    w0 = np.full(p.n_assets, 1 / p.n_assets)
    res = run(p, lambda t, w: w0 if w.sum() == 0 else w, 10, 100, cost_bps=25.0, name="hold")
    assert res.turnover[0] == pytest.approx(1.0) and np.allclose(res.turnover[1:], 0.0, atol=1e-12)
    value = w0 * np.prod(1 + p.returns[10:101], axis=0)                 # each holding left alone
    assert res.wealth()[-1] == pytest.approx((1 - 0.0025) * value.sum(), rel=1e-10)


class _Fixed(torch.nn.Module):
    """A stand-in policy with a fixed target and trade fraction."""

    def __init__(self, target: np.ndarray, kappa: float) -> None:
        super().__init__()
        self.target = torch.tensor(target, dtype=torch.float64)
        self.kappa = kappa

    def forward(self, x, ctx, w_now, w_mkt):
        b = w_now.shape[0]
        return self.target.expand(b, -1), torch.full((b,), self.kappa, dtype=torch.float64)


def test_torch_simulator_matches_the_engine():
    p = synthetic_panel()
    target, kappa, cost = np.linspace(1, 2, p.n_assets) / np.linspace(1, 2, p.n_assets).sum(), 0.3, 0.001
    f64 = torch.float64
    R, W = torch.tensor(p.returns, dtype=f64), torch.tensor(p.market_weights(), dtype=f64)
    X, C = torch.zeros(p.n_months, p.n_assets, 1, dtype=f64), torch.zeros(p.n_months, 1, dtype=f64)
    logs, _, _ = simulate(_Fixed(target, kappa), X, C, R, W, torch.tensor([20]), 50, cost)
    w_start = p.market_weights()[20]
    res = run(p, lambda t, w: w + kappa * (target - w), 21, 70, cost * 1e4, "fixed", initial=w_start)
    np.testing.assert_allclose(np.exp(logs[0].numpy()) - 1, res.net, rtol=1e-10, atol=1e-12)


def test_untrained_network_holds_the_market_and_buys_fully_from_cash():
    torch.manual_seed(0)
    m = DeepGrowthNet(n_features=4, n_context=3, d_model=16, n_heads=4, n_layers=1, dropout=0.0, max_tilt=3.0, init_trade=0.25).eval()
    w_mkt = torch.softmax(torch.randn(2, 7), -1)
    target, kappa = m(torch.randn(2, 7, 4), torch.randn(2, 3), w_mkt.clone(), w_mkt)
    torch.testing.assert_close(target, w_mkt)
    assert torch.allclose(kappa, torch.full((2,), 0.25), atol=1e-6)
    _, k_cash = m(torch.randn(2, 7, 4), torch.randn(2, 3), torch.zeros(2, 7), w_mkt)
    assert torch.allclose(k_cash, torch.ones(2))


def test_tilts_stay_inside_the_mandate_and_weights_are_long_only():
    torch.manual_seed(1)
    m = DeepGrowthNet(n_features=4, n_context=3, d_model=16, n_heads=4, n_layers=1, dropout=0.0, max_tilt=3.0, init_trade=0.25)
    torch.nn.init.normal_(m.tilt_head.weight, std=5.0)
    w_mkt = torch.softmax(torch.randn(4, 9), -1)
    target, _ = m(torch.randn(4, 9, 4), torch.randn(4, 3), w_mkt, w_mkt)
    assert (target >= 0).all() and torch.allclose(target.sum(-1), torch.ones(4))
    ratio = target / w_mkt
    assert ratio.max() / ratio.min() <= 9.0 + 1e-4             # relative tilts bounded by max_tilt both ways


def test_crra_is_log_growth_for_unit_risk_aversion():
    x = torch.tensor([-0.1, 0.0, 0.05])
    assert torch.equal(crra(x, 1.0), x)
    assert (crra(x, 3.0) <= x + 1e-9).all()                     # more risk-averse utility is lower, never higher


def test_training_labels_never_reach_validation(monkeypatch):
    p = synthetic_panel(T=300)
    feats = build_features(p)
    s = TrainSettings(d_model=8, n_heads=2, n_layers=1, dropout=0.0, init_trade=0.25, learning_rate=1e-3, weight_decay=0.0,
                      episode_months=24, batch_episodes=4, max_steps=2, eval_every=1, patience=1, val_years=5,
                      refit_every_years=5, warm_start=True, grad_clip=1.0, risk_aversion=1.0, max_tilt=3.0, cost_bps=10.0)
    t = Tensors.build(p, feats)
    seen = []
    import dpc.wealth.train as tr
    real = tr.simulate

    def spy(model, X, C, R, W, starts, length, cost, **kw):
        seen.append((int(starts.max()) + length, model.training))
        return real(model, X, C, R, W, starts, length, cost, **kw)

    monkeypatch.setattr(tr, "simulate", spy)
    refit = refit_points(p, "1970-01", 5)[0]
    train_member(t, refit, 0, s, Variant("deep"), None)
    v0 = refit - 60
    assert all(last <= v0 for last, training in seen if training)   # the last label month of every training episode
    assert any(not training for _, training in seen)


def test_baselines_respect_their_constraints():
    p = synthetic_panel(T=200)
    t = 150
    w_mv = min_variance(p, 1, 0.3)(t, np.zeros(p.n_assets))
    assert w_mv.min() >= -1e-9 and w_mv.max() <= 0.3 + 1e-6
    w_rp = risk_parity(p, 1)(t, np.zeros(p.n_assets))
    rc = w_rp * (_cov(p, t) @ w_rp)
    assert rc.max() / rc.min() < 1.01                             # equal risk from every industry
    lo, hi = tilt_bounds(p.market_weights()[t], 3.0)
    w_g = growth_optimizer(p, 1, 3.0, 0.5)(t, np.zeros(p.n_assets))
    assert (w_g >= lo - 1e-7).all() and (w_g <= hi + 1e-7).all()
    w_m = momentum_tilt(p, 3.0)(t, np.zeros(p.n_assets))
    assert abs(w_m.sum() - 1) < 1e-12 and (w_m > 0).all()


def _returns_frame(net: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("1960-01-31", periods=len(net), freq="ME")
    return pd.DataFrame({"net": net, "gross": net, "turnover": np.zeros(len(net))}, index=idx)


def test_verdict_rule_follows_the_protocol():
    from dpc.wealth.evaluate import compare

    rng = np.random.default_rng(3)
    base = rng.normal(0.008, 0.04, 600)
    noise = rng.normal(0.0, 0.01, 600)
    args = dict(lags=12, block=12, draws=1000, seed=0)
    better = compare(_returns_frame(base + 0.004 + noise), _returns_frame(base), **args)
    worse = compare(_returns_frame(base - 0.004 + noise), _returns_frame(base), **args)
    tie = compare(_returns_frame(base + noise - noise.mean()), _returns_frame(base), **args)
    assert better["verdict"] == "better" and better["t_nw"] >= 2 and better["ci_low"] > 0
    assert worse["verdict"] == "worse" and worse["t_nw"] <= -2 and worse["ci_high"] < 0
    assert tie["verdict"] == "no detectable difference"
    assert better["diff_log_growth"] == pytest.approx(12 * np.mean(np.log1p(base + 0.004 + noise) - np.log1p(base)))


def test_repricing_at_the_run_cost_reproduces_the_net_returns():
    from dpc.wealth.evaluate import reprice
    from dpc.wealth.engine import run as run_engine

    p = synthetic_panel()
    rng = np.random.default_rng(5)
    res = run_engine(p, lambda t, w: rng.dirichlet(np.ones(p.n_assets)), 30, 120, cost_bps=25.0, name="random")
    df = pd.DataFrame({"net": res.net, "gross": res.gross, "turnover": res.turnover}, index=res.dates)
    np.testing.assert_allclose(reprice(df, 25.0)["net"], res.net, rtol=1e-12, atol=1e-14)
    assert (reprice(df, 0.0)["net"] >= res.net - 1e-15).all()          # free trading never earns less


def test_results_table_lists_every_strategy_and_comparison():
    from dpc.wealth.report import results_markdown

    perf = {n: {"terminal_wealth": 2.0, "cagr": 0.1, "volatility": 0.15, "sharpe": 0.5, "max_drawdown": -0.3, "worst_year": -0.2,
                "turnover_per_year": 0.1, "active_share": 0.2, "alpha_annual": 0.0, "alpha_t": 0.0, "beat_market_rolling": 0.5}
            for n in ("market", "deep")}
    comp = {"diff_log_growth": 0.01, "t_nw": 2.5, "ci_low": 0.001, "ci_high": 0.02, "verdict": "better"}
    site = {"commit": "abc1234", "oos": {"start": "1960-01", "end": "2026-09", "months": 801}, "cost_bps": 10, "performance": perf,
            "labels": {"market": "Market (hold the index)", "deep": "Deep Growth Network"}, "comparisons": {"deep - market": comp},
            "decades": {"1960s": {"market": 0.08, "deep": 0.09}}, "costs": {"deep@0bps": {"terminal_wealth": 2.1, "cagr": 0.11, "vs_market_log_growth": 0.01}},
            "seeds": {}, "oracle_terminal_wealth": 1e8, "trade_fraction": None, "data_check": {"correlation": 0.9998, "mean_abs_diff_monthly": 0.0004},
            "tests": 1, "params": 10}
    text = results_markdown(site)
    for needle in ("| Market (hold the index) |", "| Deep Growth Network |", "Deep Growth Network minus the market", "**better**", "1960s", "abc1234"):
        assert needle in text
