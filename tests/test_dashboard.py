"""The dashboard data layer works on the real results (skipped until they exist)."""

from __future__ import annotations

import numpy as np
import pytest


def test_dashboard_data_consistent_with_constraints(cfg, panel) -> None:
    from dpc.dashboard_data import load_dashboard_data

    try:
        d = load_dashboard_data(cfg)
    except FileNotFoundError:
        pytest.skip("results not built")
    if "ls_classical" not in d.results:
        pytest.skip("results not built")
    date = d.results["ls_classical"].weights.index[-1]
    w = d.weights_at("ls_classical", date)
    assert abs(w["weight"].sum()) < 1e-4  # dollar neutral
    assert (w["weight"].abs() <= cfg.optimizer.max_weight + 1e-5).all()
    assert (d.sector_exposure("ls_classical", date).abs() <= cfg.optimizer.sector_limit + 1e-5).all()
    assert (d.factor_exposure("ls_classical", date).abs() <= cfg.optimizer.factor_limit + 1e-5).all()
    cum = d.cumulative(["ls_classical"])
    assert np.isfinite(cum.to_numpy()).all()


def test_all_saved_long_short_weights_respect_hard_constraints(cfg, panel) -> None:
    """Every rebalance of every long-short strategy satisfies the hard limits."""
    from dpc.backtest import load_result

    for name in ("ls_classical", "ls_deep_twostage", "e2e"):
        try:
            r = load_result(name, cfg.paths.results)
        except FileNotFoundError:
            continue
        W = r.weights.to_numpy()
        tol = 1e-5
        assert np.abs(W.sum(axis=1)).max() < tol
        assert np.abs(W).max() <= cfg.optimizer.max_weight + tol
        assert np.abs(W).sum(axis=1).max() <= cfg.optimizer.gross_leverage + tol
        sec = panel.universe["sector"].reindex(r.weights.columns)
        S = np.stack([(sec == s).to_numpy(float) for s in sorted(sec.unique())])
        assert np.abs(W @ S.T).max() <= cfg.optimizer.sector_limit + tol
