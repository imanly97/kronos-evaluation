"""Deterministic post-close grading of the forecast cache (PLAN §4, §6).

The morning after a session settles: for every `forecasts` row with no `actuals`
row yet, pull the realised close and record

  ret              close-to-close return on `asof`
  actual_quantile  where that return lands in Kronos's own distribution
                   (piecewise-linear across the five stored quantiles, clamped)
  inside_envelope  q05 <= ret <= q95
  dir_match        sign(ret) == sign(q50)

Append-only. No LLM. This is a cron job.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .config import QUANTILES
from .data import DataError, _coerce_date, load_actual
from .store import ImmutableViolation, append, load, sync_db, verify_chain

_QCOLS = [f"q{int(q * 100):02d}" for q in QUANTILES]


def _sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def actual_quantile(row: dict, ret: float) -> float:
    """Interpolate `ret`'s quantile from the five stored return quantiles."""
    qs = np.array(QUANTILES, dtype=float)
    xs = np.array([row[c] for c in _QCOLS], dtype=float)
    order = np.argsort(xs)
    xs, qs = xs[order], qs[order]
    if ret <= xs[0]:
        # linear extrapolation on the lower tail, clamped
        slope = (qs[1] - qs[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
        return float(np.clip(qs[0] + slope * (ret - xs[0]), 0.001, 0.999))
    if ret >= xs[-1]:
        slope = (qs[-1] - qs[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 0.0
        return float(np.clip(qs[-1] + slope * (ret - xs[-1]), 0.001, 0.999))
    return float(np.interp(ret, xs, qs))


def pending() -> list[dict]:
    f = load("forecasts")
    if f.empty:
        return []
    a = load("actuals")
    done = set() if a.empty else set(zip(a["asof"].astype(str), a["ticker"]))
    return [r for r in f.to_dict("records") if (str(r["asof"]), r["ticker"]) not in done]


def grade_one(row: dict) -> dict | None:
    try:
        # settled historical bar — use the cache; only refresh if it's genuinely behind
        a = load_actual(row["ticker"], row["asof"], force=False)
    except DataError as exc:
        print(f"  {row['ticker']:6} {row['asof']}  SKIP — {exc}", file=sys.stderr)
        return None
    ret = a.ret
    q = actual_quantile(row, ret)
    return {
        "asof": str(row["asof"]), "ticker": row["ticker"],
        "price_source": a.source,
        "prev_close": round(a.prev_close, 4), "actual_close": round(a.actual_close, 4),
        "ret": round(ret, 6),
        "actual_quantile": round(q, 4),
        "inside_envelope": bool(row["q05"] <= ret <= row["q95"]),
        "dir_match": bool(_sign(ret) == _sign(row["q50"]) and _sign(ret) != 0),
    }


def run(asof: str | None = None, *, replace: bool = False) -> list[dict]:
    todo = pending()
    if asof:
        todo = [r for r in todo if str(r["asof"]) == str(_coerce_date(asof))]
    graded = []
    for r in todo:
        res = grade_one(r)
        if res is None:
            continue
        try:
            append("actuals", res, allow_replace=replace)
            graded.append(res)
        except ImmutableViolation as exc:
            print(f"  {exc}", file=sys.stderr)
    if graded:
        verify_chain("actuals")
        sync_db()
    return graded


def report() -> str:
    from .store import joined

    j = joined()
    g = j.dropna(subset=["ret"]) if "ret" in j.columns else j.iloc[0:0]
    if g.empty:
        return "no graded forecasts yet"
    lines = [
        f"graded forecast-days: {len(g)}  "
        f"({g['ticker'].nunique()} tickers, {g['asof'].nunique()} dates)",
        f"  envelope coverage (target ~90%):  {g['inside_envelope'].mean():.0%}",
        f"  direction hit rate:               {g['dir_match'].mean():.0%}",
        f"  mean actual_quantile (target ~.5): {g['actual_quantile'].mean():.2f}",
        f"  median |actual_quantile - .5|:     {(g['actual_quantile'] - .5).abs().median():.2f}",
    ]
    if "flagged" in g.columns and g["flagged"].any():
        fl, un = g[g["flagged"]], g[~g["flagged"]]
        lines.append(f"  flagged vs unflagged |ret|:        "
                     f"{fl['ret'].abs().mean()*100:.2f}%  vs  {un['ret'].abs().mean()*100:.2f}%")
    return "\n".join(lines)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grade the forecast cache against the close.")
    ap.add_argument("--asof")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    if args.report:
        print(report())
        return 0
    graded = run(args.asof, replace=args.replace)
    print(f"graded {len(graded)} forecast-days")
    for r in graded:
        print(f"  {r['ticker']:6} {r['asof']}  ret {r['ret']*100:+5.2f}%  "
              f"q={r['actual_quantile']:.2f}  inside={r['inside_envelope']!s:5} "
              f"dir={r['dir_match']}")
    print("\n" + report())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
