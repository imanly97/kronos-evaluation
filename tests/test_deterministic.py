"""Offline tests for the deterministic spine — no network, no LLM, no Kronos.

Covers the load-bearing invariants: the hash chain detects tampering, locked rows
are immutable, the price loader never looks ahead, setup labels stay in the
controlled vocabulary, and the grade-quantile interpolation is monotone + clamped.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import store
from src.config import SETUPS


# --------------------------------------------------------------------------- #
# store: hash chain + immutability
# --------------------------------------------------------------------------- #
@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "LOG_DIR", tmp_path)
    monkeypatch.setattr(store, "_path", lambda t: tmp_path / f"{t}.jsonl")
    return tmp_path


def _fc_row(ticker="AAPL", asof="2025-05-01", **kw):
    base = dict(asof=asof, ticker=ticker, price_source="cache", context_to="2025-04-30",
                prev_close=100.0, q05=-0.02, q25=-0.005, q50=0.004, q75=0.012, q95=0.03,
                p_up=0.7, std=0.015, strength=0.6, strength_z=None)
    base.update(kw)
    return base


def test_chain_verifies_and_detects_tampering(tmp_store):
    store.append("forecasts", _fc_row("AAPL"))
    store.append("forecasts", _fc_row("MSFT"))
    assert store.verify_chain("forecasts")

    path = tmp_store / "forecasts.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row["q50"] = 0.999                      # tamper with a locked value
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(store.ImmutableViolation):
        store.verify_chain("forecasts")


def test_locked_row_is_immutable(tmp_store):
    store.append("forecasts", _fc_row("AAPL", q50=0.004))
    with pytest.raises(store.ImmutableViolation):
        store.append("forecasts", _fc_row("AAPL", q50=0.5))


def test_replace_supersedes_but_keeps_chain(tmp_store):
    store.append("forecasts", _fc_row("AAPL", q50=0.004))
    store.append("forecasts", _fc_row("AAPL", q50=0.010), allow_replace=True)
    assert store.verify_chain("forecasts")
    df = store.load("forecasts")
    assert len(df) == 1 and df.iloc[0]["q50"] == pytest.approx(0.010)


def test_load_takes_last_row_per_key(tmp_store):
    store.append("forecasts", _fc_row("AAPL"))
    store.append("forecasts", _fc_row("MSFT"))
    assert set(store.load("forecasts")["ticker"]) == {"AAPL", "MSFT"}


# --------------------------------------------------------------------------- #
# data: strict no-lookahead
# --------------------------------------------------------------------------- #
def test_load_context_never_looks_ahead():
    from src.data import load_context

    asof = "2025-05-15"
    pl = load_context("AAPL", asof)          # uses the on-disk cache
    assert pl.df.index.max() < pd.Timestamp(asof)
    assert pl.df.index.is_monotonic_increasing


def test_setup_features_are_a_pure_function_of_the_window():
    from src.data import load_context
    from src.setups import setup_features

    pl = load_context("MSFT", "2025-05-15")
    a = setup_features(pl.df, ticker="MSFT")
    b = setup_features(pl.df.copy(), ticker="MSFT")
    assert a.as_row() == b.as_row()


# --------------------------------------------------------------------------- #
# setups: controlled vocabulary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ticker", ["AAPL", "NVDA", "JPM", "XOM"])
def test_classify_stays_in_vocabulary(ticker):
    from src.setups import describe_setup

    assert describe_setup(ticker, "2025-05-15").label in SETUPS


# --------------------------------------------------------------------------- #
# grade: quantile interpolation
# --------------------------------------------------------------------------- #
def test_actual_quantile_is_monotone_and_clamped():
    from src.grade import actual_quantile

    row = {"q05": -0.03, "q25": -0.01, "q50": 0.0, "q75": 0.012, "q95": 0.035}
    grid = np.linspace(-0.10, 0.10, 40)
    qs = [actual_quantile(row, r) for r in grid]
    assert all(0.001 <= q <= 0.999 for q in qs)
    assert all(b >= a - 1e-9 for a, b in zip(qs, qs[1:]))          # non-decreasing
    assert actual_quantile(row, 0.0) == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# features: ranking + strength-z fallback
# --------------------------------------------------------------------------- #
def test_rank_table_orders_by_abs_strength_and_falls_back_to_cross_sectional():
    from src.features import rank_table

    fc = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "q50": [0.02, -0.001, 0.01], "q05": [-0.01, -0.02, -0.03],
        "q95": [0.05, 0.02, 0.04], "p_up": [0.9, 0.5, 0.7],
        "std": [0.02, 0.02, 0.02], "strength": [1.0, -0.05, 0.5],
    })
    out = rank_table(fc)                     # no strength_hist -> cross-sectional
    assert list(out["ticker"]) == ["A", "C", "B"]
    assert (out["strength_z_basis"] == "cross-sectional").all()
