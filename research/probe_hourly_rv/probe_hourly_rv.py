"""GO/NO-GO probe: does Kronos-small have realised-volatility skill on hourly
equity bars, vs EWMA and HAR-RV?

Self-contained (does not touch src/ schema). Pulls hourly bars via the Yahoo
chart API, runs Kronos over a grid of forecast origins, extracts the model's
implied next-session RV from the sample-path dispersion, and scores it against
realised RV and two classical baselines.

    .venv/bin/python .../scratchpad/probe_hourly_rv.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from curl_cffi import requests as cr

ROOT = Path("/Users/isaac/Desktop/foundation_model_desk")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor_kronos"))

from src.kronos_infer import _auto_regressive_paths, load_predictor  # noqa: E402
from model.kronos import calc_time_stamps  # noqa: E402

NAMES = ["AAPL", "MSFT", "NVDA", "AMD", "JPM", "XOM", "NFLX", "TSLA"]
LOOKBACK = 400          # hourly bars of context (< 512)
HORIZON = 6             # forecast the next ~1 session
SAMPLES = 120
N_ORIGINS = 120         # forecast origins per name, evenly spaced
OOS_START = "2024-08-01"   # after Kronos's ~June-2024 cutoff
CACHE = Path(__file__).parent / "hourly_cache"
CACHE.mkdir(exist_ok=True)

FEAT = ["open", "high", "low", "close", "volume", "amount"]


def get_hourly(ticker: str) -> pd.DataFrame:
    p = CACHE / f"{ticker}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           "?range=730d&interval=1h&includePrePost=false")
    r = cr.get(url, impersonate="chrome124", timeout=30)
    res = r.json()["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York")
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"],
                       "close": q["close"], "volume": q["volume"]}, index=ts).dropna()
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df.to_pickle(p)
    return df


def kronos_rv(pred, ctx: pd.DataFrame, y_index: pd.DatetimeIndex) -> float:
    """Model-implied std of the next HORIZON hourly log-returns."""
    x = ctx[FEAT].to_numpy(np.float32)
    xm, xs = x.mean(0), x.std(0)
    xn = np.clip((x - xm) / (xs + 1e-5), -pred.clip, pred.clip)
    xst = calc_time_stamps(pd.Series(ctx.index.tz_localize(None))).values.astype(np.float32)
    yst = calc_time_stamps(pd.Series(y_index.tz_localize(None))).values.astype(np.float32)
    raw = _auto_regressive_paths(pred, xn[None], xst[None], yst[None], HORIZON,
                                 T=1.0, top_k=0, top_p=0.9, sample_count=SAMPLES)[0]
    paths = raw * (xs + 1e-5) + xm                      # (samples, HORIZON, feat)
    close = paths[:, :, FEAT.index("close")]
    last = float(ctx["close"].iloc[-1])
    seq = np.concatenate([np.full((close.shape[0], 1), last), close], axis=1)
    rets = np.diff(np.log(np.clip(seq, 1e-6, None)), axis=1)   # (samples, HORIZON)
    per_path_rv = rets.std(axis=1)
    return float(np.median(per_path_rv))


def ewma_rv(rets: pd.Series, lam: float = 0.94) -> float:
    v = 0.0
    for r in rets.to_numpy():
        v = lam * v + (1 - lam) * r * r
    return float(np.sqrt(v))


def har_rv(rets: pd.Series, bars_day: int = 7) -> float:
    """HAR-RV on hourly data: regress next-day RV on (1h, 1d, 1w) RV, walk-forward-ish.
    Simplified: use the fitted-on-history closed form via lstsq."""
    rv_d = rets.pow(2).rolling(bars_day).sum().pow(0.5)
    rv_w = rets.pow(2).rolling(bars_day * 5).sum().pow(0.5) / np.sqrt(5)
    rv_h = rets.abs()
    tgt = rv_d.shift(-bars_day)
    d = pd.DataFrame({"h": rv_h, "d": rv_d, "w": rv_w, "y": tgt}).dropna()
    if len(d) < 200:
        return float(rv_d.iloc[-1] / np.sqrt(bars_day))
    A = np.c_[np.ones(len(d)), d[["h", "d", "w"]].to_numpy()]
    coef, *_ = np.linalg.lstsq(A, d["y"].to_numpy(), rcond=None)
    x = np.array([1.0, rv_h.iloc[-1], rv_d.iloc[-1], rv_w.iloc[-1]])
    return float(max(np.dot(coef, x), 1e-6) / np.sqrt(bars_day))


def qlike(f: np.ndarray, r: np.ndarray) -> float:
    f2, r2 = f**2 + 1e-12, r**2 + 1e-12
    return float(np.mean(r2 / f2 - np.log(r2 / f2) - 1))


def main() -> None:
    pred = load_predictor()
    rows = []
    for tk in NAMES:
        df = get_hourly(tk)
        df = df[df.index >= pd.Timestamp(OOS_START, tz="America/New_York")]
        logret = np.log(df["close"]).diff()
        origins = np.linspace(LOOKBACK + 40, len(df) - HORIZON - 1, N_ORIGINS, dtype=int)
        t0 = time.time()
        for i in origins:
            ctx = df.iloc[i - LOOKBACK:i]
            fut = df.iloc[i:i + HORIZON]
            y_index = pd.DatetimeIndex(fut.index)
            realized = float(np.log(fut["close"]).diff().dropna().std())
            if not np.isfinite(realized) or realized == 0:
                continue
            k = kronos_rv(pred, ctx, y_index)
            hist = logret.iloc[i - LOOKBACK:i].dropna()
            rows.append({
                "ticker": tk, "origin": df.index[i],
                "realized": realized,
                "kronos": k,
                "ewma": ewma_rv(hist),
                "har": har_rv(hist),
                "naive": float(hist.iloc[-HORIZON:].std()),
            })
        print(f"  {tk}: {len(origins)} origins [{time.time()-t0:.0f}s]")

    d = pd.DataFrame(rows)
    d.to_pickle(Path(__file__).parent / "probe_results.pkl")
    print(f"\n{len(d)} forecast points, {d['ticker'].nunique()} names, "
          f"{d['origin'].min()} .. {d['origin'].max()}\n")

    def score(col):
        f, r = d[col].to_numpy(), d["realized"].to_numpy()
        lr = np.corrcoef(np.log(f + 1e-9), np.log(r + 1e-9))[0, 1]
        return {"corr_logRV": round(lr, 3), "spearman": round(d[col].corr(d["realized"], "spearman"), 3),
                "QLIKE": round(qlike(f, r), 4), "RMSE": round(float(np.sqrt(np.mean((f - r)**2))), 5),
                "bias": round(float(np.mean(f - r)), 5)}

    tab = pd.DataFrame({c: score(c) for c in ["kronos", "ewma", "har", "naive"]}).T
    print(tab.to_string())
    print("\n>>> GO if kronos's corr/QLIKE is at least competitive with ewma/har.")
    # incremental: does kronos add info over ewma?
    from numpy.linalg import lstsq
    X = np.c_[np.ones(len(d)), np.log(d["ewma"]), np.log(d["kronos"])]
    y = np.log(d["realized"].to_numpy() + 1e-9)
    coef, *_ = lstsq(X, y, rcond=None)
    yhat = X @ coef
    r2_both = 1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2)
    Xe = np.c_[np.ones(len(d)), np.log(d["ewma"])]
    ce, *_ = lstsq(Xe, y, rcond=None)
    r2_ewma = 1 - np.sum((y - Xe @ ce)**2) / np.sum((y - y.mean())**2)
    print(f"\nlog-RV R^2: ewma alone {r2_ewma:.3f}  |  ewma + kronos {r2_both:.3f}  "
          f"(kronos coef {coef[2]:+.3f})")


if __name__ == "__main__":
    main()
