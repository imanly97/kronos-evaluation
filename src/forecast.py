"""Run Kronos over an origin grid → the immutable forecast store (PLAN §6, §9).

The only expensive part of the study. For each (origin, name) we run the
dispersion-preserving sampler once, reduce the sample-path distribution to the
quantities the metrics need, and append to a parquet store. Resumable: an
(origin, ticker) already present is skipped.

Runs **unbatched** — profiling (2026-08) showed the MPS backend degrades ~6x per
name with any batch dim and OOMs past batch 8. At L=250 / S=50, one forecast is
~2.6s; the hourly-H1 dev grid (~11k forecasts) is ~8h.

  run(freq, H, split=…)   walk the grid, write store/{freq}_H{H}.parquet

Each row: origin, ticker, context_to, n_paths, and the model's reduction —
  kronos_rv_{gk,park,cc}      median over paths of that path's forward RV
  kronos_rv_gk_mean           mean over paths
  kronos_path_rv_std          dispersion of per-path RV (a signal itself)
  kronos_ret_med, kronos_p_up cumulative-return median / P(up) over the horizon
  kronos_q{05,25,50,75,95}    quantiles of the path cumulative returns  (for PIT)
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DEV_END,
    DEV_START,
    DEV_UNIVERSE,
    FORECAST_STORE,
    LOCKBOX_END,
    LOCKBOX_START,
    LOOKBACK_DAILY,
    LOOKBACK_HOURLY,
    SAMPLE_COUNT,
    UNIVERSE,
)
from .data import DataError, context_before, get_daily, get_hourly, session_bars, settled_sessions
from .kronos import PRICE_COLS, load_predictor, sample_paths
from .targets import _GK_C

_QS = (0.05, 0.25, 0.50, 0.75, 0.95)


# --------------------------------------------------------------------------- #
# path-distribution reduction
# --------------------------------------------------------------------------- #
def _gk_rv_per_path(paths: np.ndarray) -> np.ndarray:
    o, h, l, c = (paths[:, :, PRICE_COLS.index(x)] for x in ("open", "high", "low", "close"))
    hl = np.log(np.clip(h / l, 1e-9, None)) ** 2
    co = np.log(np.clip(c / o, 1e-9, None)) ** 2
    return np.sqrt(np.clip(0.5 * hl - _GK_C * co, 0.0, None).sum(axis=1))


def _park_rv_per_path(paths: np.ndarray) -> np.ndarray:
    h, l = paths[:, :, PRICE_COLS.index("high")], paths[:, :, PRICE_COLS.index("low")]
    return np.sqrt((np.log(np.clip(h / l, 1e-9, None)) ** 2 / (4 * np.log(2))).sum(axis=1))


def _cc_rv_per_path(paths: np.ndarray, prev_close: float) -> np.ndarray:
    c = paths[:, :, PRICE_COLS.index("close")]
    seq = np.concatenate([np.full((c.shape[0], 1), prev_close), c], axis=1)
    r = np.diff(np.log(np.clip(seq, 1e-9, None)), axis=1)
    return np.sqrt((r * r).sum(axis=1))


def _reduce(paths: np.ndarray, prev_close: float) -> dict:
    close = paths[:, :, PRICE_COLS.index("close")]
    cum_ret = np.log(np.clip(close[:, -1], 1e-9, None) / prev_close)
    gk = _gk_rv_per_path(paths)
    out = {
        "kronos_rv_gk": float(np.median(gk)),
        "kronos_rv_gk_mean": float(np.mean(gk)),
        "kronos_rv_park": float(np.median(_park_rv_per_path(paths))),
        "kronos_rv_cc": float(np.median(_cc_rv_per_path(paths, prev_close))),
        "kronos_path_rv_std": float(np.std(gk)),
        "kronos_ret_med": float(np.median(cum_ret)),
        "kronos_p_up": float(np.mean(cum_ret > 0)),
    }
    for q in _QS:
        out[f"kronos_q{int(q*100):02d}"] = float(np.quantile(cum_ret, q))       # return quantiles
        out[f"kronos_rv_q{int(q*100):02d}"] = float(np.quantile(gk, q))         # path-RV quantiles
    # full sorted per-path arrays -> exact PIT / CRPS at eval time
    out["ret_paths"] = np.sort(cum_ret).astype(np.float32).tolist()
    out["rv_paths"] = np.sort(gk).astype(np.float32).tolist()
    return out


def _forecast_one(context: pd.DataFrame, y_index: pd.DatetimeIndex,
                  sample_count: int, predictor) -> dict:
    paths = sample_paths(context, y_index, sample_count=sample_count, predictor=predictor)
    return _reduce(paths, float(context["close"].iloc[-1]))


# --------------------------------------------------------------------------- #
# grids
# --------------------------------------------------------------------------- #
_WINDOW = {"dev": (DEV_START, DEV_END), "lockbox": (LOCKBOX_START, LOCKBOX_END)}


def common_sessions(universe: list[str], freq: str) -> list[date]:
    per = []
    for t in universe:
        if freq == "hourly":
            per.append(set(settled_sessions(get_hourly(t))))
        else:
            per.append({d.date() for d in get_daily(t).index})
    return sorted(set.intersection(*per))


def origin_grid(freq: str, split: str, *, every: int = 1,
                universe: list[str] | None = None) -> list[date]:
    universe = universe or UNIVERSE
    lo, hi = (pd.Timestamp(x).date() for x in _WINDOW[split])
    return [d for d in common_sessions(universe, freq) if lo <= d <= hi][::every]


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def _store_path(freq: str, H: int) -> Path:
    return FORECAST_STORE / f"{freq}_H{H}.parquet"


def load_store(freq: str, H: int) -> pd.DataFrame:
    p = _store_path(freq, H)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _flush(freq: str, H: int, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.concat([load_store(freq, H), pd.DataFrame(rows)], ignore_index=True)
    df.drop_duplicates(["origin", "ticker"], keep="last").to_parquet(_store_path(freq, H))


def run(freq: str, H: int, *, split: str = "dev", every: int = 1,
        universe: list[str] | None = None, sample_count: int = SAMPLE_COUNT,
        limit: int | None = None) -> pd.DataFrame:
    universe = universe or UNIVERSE
    lookback = LOOKBACK_HOURLY if freq == "hourly" else LOOKBACK_DAILY
    getter = get_hourly if freq == "hourly" else get_daily
    data = {t: getter(t) for t in universe}
    sess = common_sessions(universe, freq)
    grid = origin_grid(freq, split, every=every, universe=universe)
    if limit:
        grid = grid[:limit]

    store = load_store(freq, H)
    done = set(zip(store["origin"].astype(str), store["ticker"])) if not store.empty else set()
    predictor = load_predictor()

    print(f"{freq} H={H} {split}: {len(grid)} origins x {len(universe)} names "
          f"(lookback {lookback}, {sample_count} paths)")
    buf, t_start, n_written = [], time.time(), 0
    for gi, D in enumerate(grid):
        i = sess.index(D)
        if i + H >= len(sess):
            continue
        tgt_days = sess[i + 1: i + 1 + H]
        y_index = _target_index(freq, data[universe[0]], tgt_days)
        origin_ts = (session_bars(data[universe[0]], tgt_days[0]).index[0]
                     if freq == "hourly" else pd.Timestamp(tgt_days[0]))

        t0, n_origin = time.time(), 0
        for t in universe:
            if (D.isoformat(), t) in done:
                continue
            try:
                ctx = context_before(data[t], origin_ts, lookback)
            except DataError:
                continue
            if len(ctx) != lookback:
                continue
            try:
                r = _forecast_one(ctx, y_index, sample_count, predictor)
            except Exception as exc:  # noqa: BLE001 — log + skip, don't sink the run
                print(f"    !! {t} {D}: {exc}")
                continue
            buf.append({"origin": D.isoformat(), "ticker": t, "freq": freq, "H": H,
                        "gen_ts": _utc(), "n_paths": sample_count, "lookback": lookback,
                        "context_to": ctx.index[-1].isoformat(), **r})
            n_written += 1
            n_origin += 1
        if (gi + 1) % 3 == 0 or gi == len(grid) - 1:
            _flush(freq, H, buf); buf.clear()
            eta = (time.time() - t_start) / (gi + 1) * (len(grid) - gi - 1) / 3600
            print(f"  [{gi+1}/{len(grid)}] {D}  {n_origin} names  "
                  f"{time.time()-t0:.0f}s  {n_written} rows total  eta {eta:.1f}h", flush=True)
    _flush(freq, H, buf)
    return load_store(freq, H)


def _target_index(freq: str, ref_df: pd.DataFrame, tgt_days: list[date]) -> pd.DatetimeIndex:
    if freq == "hourly":
        idx = []
        for d in tgt_days:
            idx += list(session_bars(ref_df, d).index)
        return pd.DatetimeIndex(idx)
    return pd.DatetimeIndex([pd.Timestamp(d) for d in tgt_days])


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run Kronos over the origin grid.")
    ap.add_argument("--freq", choices=["daily", "hourly"], required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--split", choices=["dev", "lockbox"], default="dev")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dev-universe", action="store_true")
    args = ap.parse_args(argv)
    run(args.freq, args.H, split=args.split, every=args.every,
        universe=DEV_UNIVERSE if args.dev_universe else UNIVERSE,
        sample_count=args.samples, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
