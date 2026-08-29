"""Offline tests for the deterministic parts of the pipeline.

    .venv/bin/python -m pytest -q

The Kronos and LLM paths are exercised in notebooks/walkthrough.ipynb (they need
weights / an API key); these tests stay hermetic.
"""
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("NARB_LOG_DIR", "/tmp/narb_test_log")

from src import store
from src.divergence import classify_divergence
from src.narrative import NarrativeScore


class _FakeForecast:
    """Minimal stand-in for KronosForecast for divergence tests."""
    def __init__(self, ret_dist):
        self._r = np.asarray(ret_dist)
        self.ticker, self.asof = "TEST", "2026-02-02"

    @property
    def ret_dist(self):
        return self._r

    median_ret = property(lambda s: float(np.median(s._r)))
    std_ret = property(lambda s: float(np.std(s._r)))
    p_up = property(lambda s: float(np.mean(s._r > 0)))
    strength = property(lambda s: s.median_ret / s.std_ret if s.std_ret else 0.0)

    def ret_quantiles(self, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
        return {q: float(np.quantile(self._r, q)) for q in qs}

    def prob_direction(self, d):
        if d > 0:
            return float(np.mean(self._r > 0))
        if d < 0:
            return float(np.mean(self._r < 0))
        return float(np.mean(np.abs(self._r) < self.std_ret))


def _score(direction, conviction):
    return NarrativeScore(ticker="TEST", asof="2026-02-02", direction=direction,
                          conviction=conviction, rationale="x")


def test_directional_divergence():
    fc = _FakeForecast(np.full(200, 0.02))          # structure firmly up
    div = classify_divergence(_score("bear", 8), fc)  # narrative firmly down
    assert div.is_directional
    assert div.divergence_type in ("directional", "directional+magnitude")


def test_low_conviction_is_not_directional():
    fc = _FakeForecast(np.full(200, 0.02))
    div = classify_divergence(_score("bear", 2), fc)
    assert not div.is_directional


def test_magnitude_divergence():
    # structure: +1% mean, tight — almost never negative
    rng = np.random.default_rng(0)
    fc = _FakeForecast(rng.normal(0.01, 0.003, 5000))
    div = classify_divergence(_score("bear", 9), fc)
    assert div.is_magnitude
    assert div.p_narr_direction < 0.30


def test_aligned():
    fc = _FakeForecast(np.full(200, 0.02))
    div = classify_divergence(_score("bull", 8), fc)
    assert div.divergence_type == "aligned"
    assert not div.diverges


def test_hash_chain_detects_tampering(tmp_path, monkeypatch):
    p = tmp_path / "signals.jsonl"
    monkeypatch.setattr(store, "SIGNALS_JSONL", p)
    store.append_signal({"asof": "2026-02-02", "ticker": "AAA", "kronos_q50": 0.01})
    store.append_signal({"asof": "2026-02-02", "ticker": "BBB", "kronos_q50": 0.02})
    assert store.verify_chain(p)

    lines = p.read_text().splitlines()
    lines[0] = lines[0].replace('"kronos_q50":0.01', '"kronos_q50":0.99') \
        if '"kronos_q50":0.01' in lines[0] else lines[0].replace("0.01", "0.99")
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(store.ImmutableViolation):
        store.verify_chain(p)


def test_immutability(tmp_path, monkeypatch):
    p = tmp_path / "signals.jsonl"
    monkeypatch.setattr(store, "SIGNALS_JSONL", p)
    store.append_signal({"asof": "2026-02-02", "ticker": "AAA", "kronos_q50": 0.01})
    with pytest.raises(store.ImmutableViolation):
        store.append_signal({"asof": "2026-02-02", "ticker": "AAA", "kronos_q50": 0.02})


def test_narrative_signal_sign():
    assert _score("bull", 7).signal == 7
    assert _score("bear", 7).signal == -7
    assert _score("neutral", 7).signal == 0


def test_data_no_lookahead_slice():
    """load_context must never return a bar dated >= asof (uses cache if present)."""
    from src.data import load_context, DataError
    try:
        pl = load_context("AAPL", date(2025, 5, 29))
    except DataError:
        pytest.skip("no network and no cache for AAPL")
    assert pl.df.index.max() < pd.Timestamp("2025-05-29")
