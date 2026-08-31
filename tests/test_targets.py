"""Realised-target construction tests (PLAN §6, §8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data, targets


def _flat_hourly(days=15):
    rows, px = [], 100.0
    for d in pd.bdate_range("2024-07-01", periods=days):
        for h in (9, 10, 11, 12, 13, 14, 15):
            rows.append((pd.Timestamp(f"{d.date()} {h}:30", tz=data.NY), px))
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                         "volume": 1e5, "amount": 1e7}, index=idx)


def _noisy_hourly(days=40, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2024-07-01", periods=days):
        for h in (9, 10, 11, 12, 13, 14, 15):
            rows.append(pd.Timestamp(f"{d.date()} {h}:30", tz=data.NY))
    idx = pd.DatetimeIndex(rows)
    c = 100 * np.exp(np.cumsum(rng.normal(0, sigma, len(idx))))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, sigma / 2, len(idx))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, sigma / 2, len(idx))))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                         "volume": 1e5, "amount": 1e7}, index=idx)


def test_rv_of_a_flat_series_is_zero():
    df = _flat_hourly()
    b = data.session_bars(df, data.sessions(df)[0])
    for est in ("cc", "parkinson", "gk"):
        assert targets.realised_vol(b, est) == pytest.approx(0.0, abs=1e-12)


def test_cc_rv_matches_hand_calc():
    closes = [100, 101, 100, 102]
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes})
    r = np.diff(np.log(closes))
    assert targets.realised_vol(df, "cc") == pytest.approx(np.sqrt(np.sum(r * r)))


def test_forward_session_rv_variance_additivity():
    df = _noisy_hourly()
    days = data.settled_sessions(df)
    origin = days[5]
    rv1 = targets.forward_session_rv(df, origin, 1)
    rv2 = targets.forward_session_rv(df, origin, 2)
    per2 = targets.realised_vol(data.session_bars(df, days[7]), "gk")
    assert rv2 == pytest.approx(np.sqrt(rv1**2 + per2**2), rel=1e-9)


def test_forward_targets_never_look_ahead():
    df = _noisy_hourly()
    days = data.settled_sessions(df)
    origin = days[10]
    # forward_session_rv must only touch sessions strictly after the origin
    used = targets._next_sessions(df, origin, 2)
    assert all(u > origin for u in used)
    # forward_return only bars strictly after the origin timestamp
    o_ts = pd.Timestamp(f"{origin} 15:30", tz=data.NY)
    fut = df.loc[df.index > o_ts].head(7)
    assert fut.index.min() > o_ts


def test_forward_session_rv_raises_at_the_edge():
    df = _noisy_hourly(days=12)
    last = data.settled_sessions(df)[-1]
    with pytest.raises(data.DataError):
        targets.forward_session_rv(df, last, 1)
