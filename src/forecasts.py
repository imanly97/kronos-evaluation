"""Deterministic Kronos batch over the universe → the immutable forecast cache.

This is a cron job, not an agent (PLAN §4). For each (asof, ticker) it runs the
dispersion-preserving sampler once, reduces the path distribution to the row the
rest of the desk consumes — return quantiles, P(up), dispersion, strength — and
locks it into `store.forecasts`.

  run_asof(asof)                  one trading day, the whole universe
  run_window(start, end)          every NYSE session in [start, end]
  strength_history(ticker, asof)  that name's past strength series (for the z-score)

Idempotent: an (asof, ticker) already in the log is skipped unless --replace.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date

import pandas as pd

from .config import (
    QUANTILES,
    REPLAY_END,
    REPLAY_START,
    SAMPLE_COUNT,
    UNIVERSE,
)
from .data import DataError, _coerce_date, load_context
from .kronos_infer import forecast
from .store import ImmutableViolation, append, load, verify_chain

_QCOLS = {q: f"q{int(q * 100):02d}" for q in QUANTILES}


def _sessions(start: date, end: date, ref: str = "AAPL") -> list[date]:
    """Actual trading sessions in [start, end], read off a liquid name's history."""
    pl = load_context(ref, end + pd.Timedelta(days=5), force=False)
    idx = pd.DatetimeIndex(pl.df.index)
    days = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    return [d.date() for d in days]


def forecast_row(ticker: str, asof) -> dict:
    """Run Kronos once and reduce to a `store.forecasts` row (no write)."""
    pl = load_context(ticker, asof)
    fc = forecast(pl, sample_count=SAMPLE_COUNT)
    rq = fc.ret_quantiles(QUANTILES)
    return {
        "asof": str(_coerce_date(asof)),
        "ticker": ticker,
        "price_source": pl.source,
        "context_to": str(pl.last_context_date.date()),
        "prev_close": round(fc.prev_close, 4),
        **{col: round(rq[q], 6) for q, col in _QCOLS.items()},
        "p_up": round(fc.p_up, 4),
        "std": round(fc.std_ret, 6),
        "strength": round(fc.strength, 4),
        "strength_z": None,     # filled cross-sectionally at triage time
    }


def run_asof(asof, tickers: list[str] | None = None, *, replace: bool = False,
             verbose: bool = True) -> list[dict]:
    tickers = tickers or UNIVERSE
    done = set(zip(load("forecasts")["asof"].astype(str), load("forecasts")["ticker"])) \
        if not load("forecasts").empty else set()
    out = []
    for t in tickers:
        if not replace and (str(_coerce_date(asof)), t) in done:
            continue
        t0 = time.time()
        try:
            row = forecast_row(t, asof)
        except DataError as exc:
            if verbose:
                print(f"  {t:6} {asof}  SKIP — {exc}", file=sys.stderr)
            continue
        try:
            append("forecasts", row, allow_replace=replace)
        except ImmutableViolation:
            continue
        out.append(row)
        if verbose:
            print(f"  {t:6} {asof}  q50 {row['q50']*100:+5.2f}%  P(up) {row['p_up']:.2f}  "
                  f"strength {row['strength']:+.2f}  [{time.time()-t0:.1f}s]")
    return out


def run_window(start: str = REPLAY_START, end: str = REPLAY_END,
               tickers: list[str] | None = None, *, replace: bool = False) -> int:
    s, e = _coerce_date(start), _coerce_date(end)
    sessions = _sessions(s, e)
    print(f"forecast cache: {len(sessions)} sessions {sessions[0]}..{sessions[-1]}, "
          f"{len(tickers or UNIVERSE)} tickers")
    n = 0
    for i, d in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}] {d}")
        n += len(run_asof(d, tickers, replace=replace))
    verify_chain("forecasts")
    print(f"done — {n} new forecast rows; chain OK")
    return n


def strength_history(ticker: str, asof, *, min_obs: int = 1) -> pd.Series:
    """That name's `strength` for every locked forecast strictly before `asof`."""
    f = load("forecasts")
    if f.empty:
        return pd.Series(dtype=float)
    cut = str(_coerce_date(asof))
    h = f[(f["ticker"] == ticker) & (f["asof"].astype(str) < cut)]
    return h.sort_values("asof")["strength"].astype(float).reset_index(drop=True)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the Kronos forecast cache.")
    ap.add_argument("--asof", help="single session (default: the replay window)")
    ap.add_argument("--start", default=REPLAY_START)
    ap.add_argument("--end", default=REPLAY_END)
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args(argv)

    if args.asof:
        rows = run_asof(args.asof, args.tickers, replace=args.replace)
        print(f"{len(rows)} rows locked for {args.asof}")
    else:
        run_window(args.start, args.end, args.tickers, replace=args.replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
