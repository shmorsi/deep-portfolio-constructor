"""The anomaly study's statistics on synthetic data with known answers (no download)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dpc.anomalies import ZOO, study

RNG = np.random.default_rng(11)


def test_zoo_is_well_formed():
    assert len({a.key for a in ZOO}) == len(ZOO) == 15
    assert all(a.long in ("hi", "lo") and 1960 < a.year < 2026 for a in ZOO)


def test_newey_west_t_on_iid_and_autocorrelated_series():
    x = RNG.normal(0.01, 0.05, 5000)
    m, t = study.nw_t(x, 0)
    assert t == pytest.approx(m / (x.std(ddof=0) / np.sqrt(len(x))), rel=1e-6)
    ar = np.zeros(5000)
    e = RNG.normal(0, 1, 5000)
    for i in range(1, 5000):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    assert abs(study.nw_t(ar + 0.05, 12)[1]) < abs(study.nw_t(ar + 0.05, 0)[1])     # HAC widens the error


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert study.benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8


def test_sharpe_bootstrap_detects_a_real_difference():
    n = 600
    a = RNG.normal(0.01, 0.03, n)
    b = a * 0.5 + RNG.normal(0.0, 0.03, n)
    out = study.sharpe_diff_bootstrap(a, b, draws=500)
    assert out["verdict"] == "better"
    same = study.sharpe_diff_bootstrap(a, a, draws=200)
    assert same["difference"] == 0.0 and same["verdict"] == "no detectable difference"


def test_decay_regression_recovers_a_known_drop():
    idx = pd.date_range("1963-07-31", "2026-09-30", freq="ME")
    data = {}
    for a in ZOO:
        mu = np.where(idx.year <= a.year, 0.006, 0.003)              # returns halve after publication
        data[a.key] = mu + RNG.normal(0, 0.002, len(idx))
    _, pooled = study.decay(pd.DataFrame(data, index=idx))
    assert pooled["post_coef_ann"] == pytest.approx(-0.036, abs=0.004)
    assert pooled["implied_decay"] == pytest.approx(0.5, abs=0.08)
