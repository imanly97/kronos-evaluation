"""Regenerate a handful of Kronos hourly forecasts KEEPING the full sample paths,
so the findings notebook can draw fan charts / trajectory plots.

    .venv/bin/python research/probe_hourly_rv/trajectories.py

Writes data/trajectories.pkl: a list of dicts, one per (ticker, origin):
  ctx_close   last 60 context closes (index = timestamps)
  paths       (n_samples, HORIZON) forecast close prices
  actual      (HORIZON,) realised close prices
  origin, ticker, realized_rv, kronos_rv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/isaac/Desktop/foundation_model_desk")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor_kronos"))

from src.kronos_infer import _auto_regressive_paths, load_predictor  # noqa: E402
from model.kronos import calc_time_stamps  # noqa: E402

DATA = Path(__file__).parent / "data"
LOOKBACK, HORIZON, SAMPLES = 400, 6, 200
FEAT = ["open", "high", "low", "close", "volume", "amount"]

# picked from probe_results: a calm and a turbulent origin for a few names
PICKS = [
    ("NVDA", "2025-01-27 10:30"),   # DeepSeek selloff week
    ("NVDA", "2025-06-10 12:30"),   # quiet
    ("TSLA", "2025-03-10 11:30"),   # turbulent
    ("AAPL", "2025-04-07 10:30"),   # tariff crash
    ("XOM",  "2025-05-19 13:30"),   # quiet
    ("AMD",  "2025-02-05 09:30"),   # earnings-week
]


def one(pred, df: pd.DataFrame, origin: pd.Timestamp) -> dict | None:
    i = df.index.get_indexer([origin], method="nearest")[0]
    if i < LOOKBACK or i + HORIZON >= len(df):
        return None
    ctx = df.iloc[i - LOOKBACK:i]
    fut = df.iloc[i:i + HORIZON]
    x = ctx[FEAT].to_numpy(np.float32)
    xm, xs = x.mean(0), x.std(0)
    xn = np.clip((x - xm) / (xs + 1e-5), -pred.clip, pred.clip)
    xst = calc_time_stamps(pd.Series(ctx.index.tz_localize(None))).values.astype(np.float32)
    yst = calc_time_stamps(pd.Series(fut.index.tz_localize(None))).values.astype(np.float32)
    raw = _auto_regressive_paths(pred, xn[None], xst[None], yst[None], HORIZON,
                                 T=1.0, top_k=0, top_p=0.9, sample_count=SAMPLES)[0]
    paths = raw * (xs + 1e-5) + xm
    close = paths[:, :, FEAT.index("close")]
    last = float(ctx["close"].iloc[-1])
    seq = np.concatenate([np.full((close.shape[0], 1), last), close], axis=1)
    rv = float(np.median(np.diff(np.log(np.clip(seq, 1e-6, None)), axis=1).std(axis=1)))
    real_rv = float(np.log(fut["close"]).diff().dropna().std())
    return {
        "ticker": df.attrs.get("ticker", "?"), "origin": df.index[i],
        "ctx_close": ctx["close"].iloc[-60:],
        "paths": close, "actual": fut["close"].to_numpy(),
        "y_index": pd.DatetimeIndex(fut.index),
        "kronos_rv": rv, "realized_rv": real_rv,
    }


def main() -> None:
    pred = load_predictor()
    out = []
    for tk, ts in PICKS:
        df = pd.read_pickle(DATA / "hourly_cache" / f"{tk}.pkl")
        df.attrs["ticker"] = tk
        r = one(pred, df, pd.Timestamp(ts, tz="America/New_York"))
        if r:
            out.append(r)
            print(f"  {tk} {r['origin']}  kronos_rv {r['kronos_rv']:.4f}  realized {r['realized_rv']:.4f}")
    with open(DATA / "trajectories.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print(f"wrote {DATA / 'trajectories.pkl'}  ({len(out)} forecasts)")


if __name__ == "__main__":
    main()
