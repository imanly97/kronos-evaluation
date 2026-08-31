"""Metric + test-statistic sanity checks (PLAN §6, §8)."""
from __future__ import annotations

import numpy as np
import pytest

from src import metrics, recalibrate


def test_qlike_zero_at_perfect_forecast():
    y = np.array([0.01, 0.02, 0.03, 0.015])
    assert metrics.qlike(y, y) == pytest.approx(0.0, abs=1e-9)


def test_qlike_positive_and_penalises_underforecast_more():
    y = np.full(1, 0.02)
    over = metrics.qlike(np.array([0.04]), y)[0]     # forecast 2x
    under = metrics.qlike(np.array([0.01]), y)[0]    # forecast 0.5x
    assert over > 0 and under > 0
    assert under > over                              # QLIKE asymmetry


def test_mincer_zarnowitz_recovers_a_known_line():
    rng = np.random.default_rng(0)
    f = np.exp(rng.normal(0, 0.4, 500))
    y = np.exp(0.1 + 1.0 * np.log(f) + rng.normal(0, 0.05, 500))   # a=0.1, b=1 in logs
    mz = metrics.mincer_zarnowitz(f, y, logs=True)
    assert mz["intercept"] == pytest.approx(0.1, abs=0.03)
    assert mz["slope"] == pytest.approx(1.0, abs=0.03)


def test_diebold_mariano_detects_a_real_difference():
    rng = np.random.default_rng(1)
    n = 400
    la = rng.normal(1.0, 0.3, n)      # model A: mean loss 1.0
    lb = rng.normal(0.7, 0.3, n)      # model B: mean loss 0.7 -> A is worse
    r = metrics.diebold_mariano(la, lb)
    assert r["dm_stat"] > 0 and r["p_value"] < 0.01


def test_diebold_mariano_null_when_equal():
    rng = np.random.default_rng(2)
    x = rng.normal(1.0, 0.3, 400)
    r = metrics.diebold_mariano(x + rng.normal(0, 1e-6, 400), x)
    assert r["p_value"] > 0.05


def test_pit_exact_uniform_for_a_calibrated_forecast():
    rng = np.random.default_rng(3)
    n, S = 3000, 200
    y = rng.normal(0.0, 0.02, n)
    samples = rng.normal(0.0, 0.02, (n, S))     # same distribution -> calibrated
    p = metrics.pit_exact(y, samples)
    ks = metrics.pit_ks(p)
    assert ks["ks_p"] > 0.05
    assert ks["mean"] == pytest.approx(0.5, abs=0.03)


def test_pit_exact_detects_overconfidence():
    rng = np.random.default_rng(9)
    n, S = 3000, 200
    y = rng.normal(0.0, 0.03, n)                 # truth wider than forecast
    samples = rng.normal(0.0, 0.02, (n, S))      # overconfident
    p = metrics.pit_exact(y, samples)
    # too much mass in the tails -> U-shaped -> std above uniform's 0.289
    assert metrics.pit_ks(p)["std"] > 0.31


def test_crps_lower_for_the_better_forecast():
    rng = np.random.default_rng(5)
    n, S = 2000, 200
    y = rng.normal(0, 0.02, n)
    good = rng.normal(0, 0.02, (n, S))
    bad = rng.normal(0.01, 0.05, (n, S))
    assert metrics.crps_sample(y, good).mean() < metrics.crps_sample(y, bad).mean()


def test_pit_5knot_is_roughly_centered():
    rng = np.random.default_rng(3)
    n = 4000
    y = rng.normal(0.0, 0.02, n)
    from scipy.stats import norm
    taus = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    qgrid = np.tile(norm.ppf(taus, 0.0, 0.02), (n, 1))
    ks = metrics.pit_ks(metrics.pit(y, qgrid, taus))
    assert ks["mean"] == pytest.approx(0.5, abs=0.03)   # 5-knot interp: centered but not KS-uniform


def test_recalibrate_is_causal_and_fixes_bias():
    rng = np.random.default_rng(4)
    import pandas as pd
    n = 300
    truth = 0.02 * np.exp(rng.normal(0, 0.3, n))
    fc = 0.7 * truth * np.exp(rng.normal(0, 0.1, n))
    origins = pd.date_range("2024-01-01", periods=n)
    rc = recalibrate.recalibrate(origins.values, fc, truth, min_train=60)
    assert np.allclose(rc[:60], fc[:60])                       # warm-up = passthrough
    assert abs(np.mean(rc[60:] - truth[60:])) < abs(np.mean(fc - truth))
