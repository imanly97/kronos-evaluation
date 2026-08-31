"""Baseline forecaster tests (PLAN §6.1) — walk-forward, causal, sane scale."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import baselines


@pytest.fixture
def const_vol_series():
    """250 sessions of RV ~ 0.02 with mild noise; matching return series."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-02", periods=250)
    rv = pd.Series(0.02 * np.exp(rng.normal(0, 0.15, 250)), index=idx)
    ret = pd.Series(rng.normal(0, 0.02, 250), index=idx)
    return rv, ret


def test_all_baselines_recover_the_level(const_vol_series):
    rv, ret = const_vol_series
    for name, fn in baselines.DEFAULT.items():
        f = fn(rv, ret, H=1)
        assert 0.01 < f < 0.04, f"{name} forecast {f:.4f} off-scale"


def test_horizon_scaling_is_monotone(const_vol_series):
    rv, ret = const_vol_series
    for name, fn in baselines.DEFAULT.items():
        f1, f5 = fn(rv, ret, 1), fn(rv, ret, 5)
        assert f5 > f1, f"{name}: H=5 forecast not larger than H=1"


def test_rw_is_last_rv_scaled():
    idx = pd.bdate_range("2024-01-02", periods=30)
    rv = pd.Series(np.linspace(0.01, 0.03, 30), index=idx)
    ret = pd.Series(0.0, index=idx)
    assert baselines.rw(rv, ret, 4) == pytest.approx(0.03 * 2.0)


def test_walk_forward_is_strictly_causal(const_vol_series):
    rv, ret = const_vol_series
    origins = list(rv.index[100:110])

    calls = {}
    orig_ewma = baselines.ewma

    def spy(rv_h, rt_h, H, **kw):
        calls["last"] = rv_h.index[-1]
        return orig_ewma(rv_h, rt_h, H, **kw)

    baselines.walk_forward(rv, ret, [origins[3]], H=1,
                           models={"ewma94": lambda a, b, H: spy(a, b, H, lam=0.94)})
    assert calls["last"] <= origins[3]


def test_walk_forward_shape(const_vol_series):
    rv, ret = const_vol_series
    origins = list(rv.index[120:140])
    out = baselines.walk_forward(rv, ret, origins, H=1, models=baselines.FAST)
    assert list(out.columns) == list(baselines.FAST)
    assert len(out) == 20
    assert out.notna().all().all()
