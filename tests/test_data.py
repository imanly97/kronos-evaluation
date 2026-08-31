"""No-lookahead + data-hygiene tests (PLAN §8).

The slicing tests are synthetic and always run — they guard the one guarantee
the whole study rests on: a forecast never sees a bar at or after its origin.
The real-data checks skip if the cache is empty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data


def _synthetic_daily(n=300, start="2023-01-02"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1e6, "amount": 1e8}, index=idx)
    return df


def _synthetic_hourly(days=20):
    rows = []
    for d in pd.bdate_range("2024-07-01", periods=days):
        for h in (9, 10, 11, 12, 13, 14, 15):
            rows.append(pd.Timestamp(f"{d.date()} {h}:30", tz=data.NY))
    idx = pd.DatetimeIndex(rows)
    px = 100 + np.arange(len(idx)) * 0.01
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                         "volume": 1e5, "amount": 1e7}, index=idx)


# --------------------------------------------------------------------------- #
# the no-lookahead guarantee
# --------------------------------------------------------------------------- #
def test_context_before_is_strictly_causal():
    df = _synthetic_daily()
    for origin in ("2023-06-15", "2023-09-01", "2024-01-10"):
        ctx = data.context_before(df, origin, 40)
        assert ctx.index.max() < pd.Timestamp(origin)
        assert len(ctx) == 40
        assert ctx.index.is_monotonic_increasing


def test_context_before_tz_aware():
    df = _synthetic_hourly()
    origin = pd.Timestamp("2024-07-10 12:30", tz=data.NY)
    ctx = data.context_before(df, origin, 10)
    assert ctx.index.max() < origin
    assert len(ctx) == 10
    # a bar exactly at the origin must be excluded
    assert origin not in ctx.index


def test_context_before_raises_when_no_history():
    df = _synthetic_daily()
    with pytest.raises(data.DataError):
        data.context_before(df, "2022-01-01", 10)


def test_settled_sessions_drops_partial_days():
    df = _synthetic_hourly(days=10)
    # amputate the last day to 2 bars (a live/partial session)
    last = df.index[-1].date()
    keep = [ts for ts in df.index if ts.date() != last or ts.hour <= 10]
    df2 = df.loc[keep]
    settled = data.settled_sessions(df2, min_bars=4)
    assert last not in settled
    assert len(settled) == 9


def test_session_bars_and_sessions_roundtrip():
    df = _synthetic_hourly(days=5)
    days = data.sessions(df)
    assert len(days) == 5
    for d in days:
        assert len(data.session_bars(df, d)) == 7


# --------------------------------------------------------------------------- #
# real-data hygiene (skip if not cached)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ticker", ["NVDA", "AAPL"])
def test_cached_daily_is_clean(ticker):
    df = data._read(ticker, "daily")
    if df is None:
        pytest.skip("daily cache empty — run `python -m src.data refresh`")
    assert list(df.columns) == data.COLS
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    assert not df.isna().any().any()
    assert (df[data.OHLC] > 0).all().all()
    assert (df["high"] >= df["low"]).all()


def test_nvda_2024_split_is_continuous():
    df = data._read("NVDA", "daily")
    if df is None:
        pytest.skip("daily cache empty")
    seg = df.loc["2024-06-05":"2024-06-12", "close"]
    step = seg.pct_change().abs().max()
    assert step < 0.15, f"NVDA 10:1 split not adjusted — {step:.0%} jump in the window"
