"""The evaluation harness (PLAN §6, §8).

Joins the Kronos forecast store → realised targets → walk-forward baselines into
one tidy panel, then produces the `(task × cut)` tables:

  assemble(freq, H)            build/cache the panel (forecasts + realised + baselines + tags)
  vol_scorecard(panel)         per-model QLIKE / MSE / MZ / bias
  dm_vs(panel, ref)            Diebold–Mariano of every model against `ref`
  incremental_r2(panel)        does Kronos add log-RV R² over EWMA? (+ bootstrap CI)
  mcs(panel)                   Hansen–Lunde–Nason Model Confidence Set on QLIKE
  calibration(panel)           Kronos return-interval PIT + coverage; RV-interval coverage
  by_cut(panel, col, fn)       any metric within vol-quartile / sector / year buckets
  report(freq, H)              run the lot
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .baselines import DEFAULT as BASELINE_FNS
from .baselines import walk_forward
from .config import LOCKBOX_END, LOCKBOX_START, SECTOR
from .data import get_daily, get_hourly
from .forecast import load_store
from .metrics import (
    coverage,
    diebold_mariano,
    mincer_zarnowitz,
    mse_log,
    pit,
    pit_ks,
    qlike,
)
from .recalibrate import recalibrate
from .targets import (
    daily_return_series,
    daily_rv_series,
    forward_daily_return,
    forward_daily_rv,
    forward_session_rv,
    session_forward_return,
    session_return_series,
    session_rv_series,
)

MODELS = ["kronos", "kronos_recal", "ewma94", "rolling22", "har", "garch", "gjr", "rw"]
_RET_TAUS = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
PANEL_DIR = None  # set from config at import


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def _panel_path(freq, H, estimator):
    from .config import FORECAST_STORE
    return FORECAST_STORE / f"panel_{freq}_H{H}_{estimator}.parquet"


def assemble(freq: str, H: int, *, estimator: str = "gk", rebuild: bool = False) -> pd.DataFrame:
    p = _panel_path(freq, H, estimator)
    if p.exists() and not rebuild:
        return pd.read_parquet(p)

    fc = load_store(freq, H)
    if fc.empty:
        raise RuntimeError(f"forecast store {freq} H={H} is empty — run src.forecast")
    fc = fc.copy()
    fc["origin"] = pd.to_datetime(fc["origin"])
    tickers = sorted(fc["ticker"].unique())
    getter = get_hourly if freq == "hourly" else get_daily

    rows = []
    for tk in tickers:
        bars = getter(tk)
        sub = fc[fc["ticker"] == tk].sort_values("origin")
        # realised targets
        rlz_rv, rlz_ret = [], []
        for o in sub["origin"]:
            od = o.date()
            try:
                if freq == "hourly":
                    rlz_rv.append(forward_session_rv(bars, od, H, estimator))
                    rlz_ret.append(session_forward_return(bars, od, H))
                else:
                    rlz_rv.append(forward_daily_rv(bars, od, H, estimator))
                    rlz_ret.append(forward_daily_return(bars, od, H))
            except Exception:  # noqa: BLE001
                rlz_rv.append(np.nan); rlz_ret.append(np.nan)
        sub = sub.assign(realised=rlz_rv, realised_ret=rlz_ret)

        # walk-forward baselines over this name's own RV/return history
        if freq == "hourly":
            rv_s = session_rv_series(bars, estimator)
            rt_s = session_return_series(bars)
        else:
            rv_s = daily_rv_series(bars, estimator)
            rt_s = daily_return_series(bars)
        rv_s.index = pd.to_datetime(rv_s.index)
        rt_s.index = pd.to_datetime(rt_s.index)
        bl = walk_forward(rv_s, rt_s, list(sub["origin"]), H, models=BASELINE_FNS)
        bl = bl.rename_axis("origin").reset_index()
        bl["origin"] = pd.to_datetime(bl["origin"])
        sub = sub.merge(bl, on="origin", how="left")
        rows.append(sub)

    panel = pd.concat(rows, ignore_index=True)
    panel["kronos"] = panel["kronos_rv_gk"] if estimator == "gk" else \
        panel[f"kronos_rv_{estimator}"] if f"kronos_rv_{estimator}" in panel else panel["kronos_rv_gk"]
    panel["kronos_recal"] = np.nan
    for tk, g in panel.groupby("ticker"):
        m = g.dropna(subset=["kronos", "realised"])
        if len(m) > 60:
            rc = recalibrate(m["origin"].values, m["kronos"].values, m["realised"].values)
            panel.loc[m.index, "kronos_recal"] = rc

    # tags
    panel["year"] = panel["origin"].dt.year
    panel["sector"] = panel["ticker"].map(SECTOR)
    panel["in_lockbox"] = (panel["origin"] >= pd.Timestamp(LOCKBOX_START)) & \
                          (panel["origin"] <= pd.Timestamp(LOCKBOX_END))
    panel["vol_q"] = pd.qcut(panel["realised"].rank(method="first"), 4,
                             labels=["calm", "low", "elevated", "turbulent"])
    panel.to_parquet(p)
    return panel


# --------------------------------------------------------------------------- #
# metric tables
# --------------------------------------------------------------------------- #
def _clean(panel, cols):
    return panel.dropna(subset=["realised", *cols])


def vol_scorecard(panel: pd.DataFrame, models: list[str] = MODELS) -> pd.DataFrame:
    out = {}
    for m in models:
        d = _clean(panel, [m])
        y, f = d["realised"].to_numpy(), d[m].to_numpy()
        mz = mincer_zarnowitz(f, y)
        out[m] = {"n": len(d), "QLIKE": float(np.mean(qlike(f, y))),
                  "RMSE_logRV": float(np.sqrt(np.mean(mse_log(f, y)))),
                  "MZ_slope": mz["slope"], "MZ_r2": mz["r2"],
                  "bias": float(np.mean(f - y))}
    return pd.DataFrame(out).T


def dm_vs(panel: pd.DataFrame, ref: str = "ewma94",
          models: list[str] | None = None, *, loss=qlike) -> pd.DataFrame:
    models = models or [m for m in MODELS if m != ref]
    out = {}
    for m in models:
        d = _clean(panel, [m, ref])
        y = d["realised"].to_numpy()
        r = diebold_mariano(loss(d[m].to_numpy(), y), loss(d[ref].to_numpy(), y))
        out[m] = {"n": r["n"], "mean_loss_diff": r["mean_diff"],
                  "DM_stat": r["dm_stat"], "p_value": r["p_value"],
                  "verdict": ("worse than " + ref if r["dm_stat"] > 0 and r["p_value"] < .1
                              else "better than " + ref if r["dm_stat"] < 0 and r["p_value"] < .1
                              else "tie")}
    return pd.DataFrame(out).T


def incremental_r2(panel: pd.DataFrame, add: str = "kronos", base: str = "ewma94",
                   *, boot: int = 2000, seed: int = 0) -> dict:
    d = _clean(panel, [add, base])
    y = np.log(d["realised"].to_numpy())
    lb, la = np.log(d[base].to_numpy()), np.log(d[add].to_numpy())

    def r2(X):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ coef
        return 1 - np.sum(e**2) / np.sum((y - y.mean())**2), coef

    n = len(d)
    Xb = np.c_[np.ones(n), lb]
    Xba = np.c_[np.ones(n), lb, la]
    r2b, _ = r2(Xb)
    r2ba, coef = r2(Xba)
    rng = np.random.default_rng(seed)
    gains = []
    idx = np.arange(n)
    for _ in range(boot):
        s = rng.choice(idx, n, replace=True)
        ys, Xbs, Xbas = y[s], Xb[s], Xba[s]

        def _r2(X, yv):
            c, *_ = np.linalg.lstsq(X, yv, rcond=None)
            e = yv - X @ c
            return 1 - np.sum(e**2) / np.sum((yv - yv.mean())**2)
        gains.append(_r2(Xbas, ys) - _r2(Xbs, ys))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {"n": n, f"r2_{base}": round(r2b, 4), f"r2_{base}+{add}": round(r2ba, 4),
            "gain": round(r2ba - r2b, 4), "gain_ci95": [round(lo, 4), round(hi, 4)],
            f"{add}_coef": round(float(coef[2]), 3)}


def mcs(panel: pd.DataFrame, models: list[str] = MODELS, *, loss=qlike,
        alpha: float = 0.10) -> dict:
    from .metrics import model_confidence_set

    d = _clean(panel, models)
    y = d["realised"].to_numpy()
    L = pd.DataFrame({m: loss(d[m].to_numpy(), y) for m in models})
    return model_confidence_set(L, alpha=alpha)


def calibration(panel: pd.DataFrame) -> dict:
    from .metrics import crps_sample, pit_exact

    d = panel.dropna(subset=["realised_ret", "kronos_q05", "kronos_q95"])
    out = {
        "return_cov_90": round(coverage(d["realised_ret"], d["kronos_q05"], d["kronos_q95"]), 3),
        "return_cov_50": round(coverage(d["realised_ret"], d["kronos_q25"], d["kronos_q75"]), 3),
    }
    if "ret_paths" in panel.columns:
        dp = panel.dropna(subset=["realised_ret"])
        dp = dp[dp["ret_paths"].notna()]
        p = pit_exact(dp["realised_ret"].to_numpy(), list(dp["ret_paths"]))
        out["return_PIT"] = pit_ks(p)
        out["return_CRPS"] = float(np.mean(crps_sample(dp["realised_ret"].to_numpy(),
                                                       list(dp["ret_paths"]))))
    else:                                       # fallback: 5-knot interpolation
        qcols = ["kronos_q05", "kronos_q25", "kronos_q50", "kronos_q75", "kronos_q95"]
        out["return_PIT"] = pit_ks(pit(d["realised_ret"].to_numpy(), d[qcols].to_numpy(), _RET_TAUS))
    if "kronos_rv_q05" in panel.columns:
        dr = panel.dropna(subset=["realised", "kronos_rv_q05", "kronos_rv_q95"])
        out["rv_cov_90"] = round(coverage(dr["realised"], dr["kronos_rv_q05"], dr["kronos_rv_q95"]), 3)
        if "rv_paths" in panel.columns:
            drp = dr[dr["rv_paths"].notna()]
            out["rv_PIT"] = pit_ks(pit_exact(drp["realised"].to_numpy(), list(drp["rv_paths"])))
    return out


def by_cut(panel: pd.DataFrame, col: str, models: list[str] = MODELS,
           *, metric=qlike) -> pd.DataFrame:
    rows = {}
    for key, g in panel.groupby(col, observed=True):
        r = {}
        for m in models:
            d = _clean(g, [m])
            r[m] = float(np.mean(metric(d[m].to_numpy(), d["realised"].to_numpy()))) if len(d) else np.nan
        r["n"] = len(g)
        rows[key] = r
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def report(freq: str, H: int, *, estimator: str = "gk", lockbox: bool = False) -> None:
    panel = assemble(freq, H, estimator=estimator)
    panel = panel[panel["in_lockbox"]] if lockbox else panel[~panel["in_lockbox"]]
    tag = "LOCKBOX" if lockbox else "DEV"
    print(f"\n{'='*70}\nKronos vol eval — {freq} H={H} ({estimator})  [{tag}]  "
          f"n={len(panel)}  {panel['ticker'].nunique()} names  "
          f"{panel['origin'].min().date()}..{panel['origin'].max().date()}\n{'='*70}")

    print("\n[scorecard]");            print(vol_scorecard(panel).round(4).to_string())
    print("\n[Diebold–Mariano vs EWMA (QLIKE)]"); print(dm_vs(panel).round(4).to_string())
    print("\n[incremental info]");     print(incremental_r2(panel))
    try:
        print("\n[Model Confidence Set (QLIKE, 90%)]"); print(mcs(panel))
    except Exception as e:  # noqa: BLE001
        print("  MCS failed:", e)
    print("\n[calibration]");          print(calibration(panel))
    print("\n[QLIKE by realised-vol quartile]"); print(by_cut(panel, "vol_q").round(4).to_string())
    print("\n[QLIKE by sector]");      print(by_cut(panel, "sector").round(4).to_string())


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kronos evaluation harness.")
    ap.add_argument("--freq", choices=["daily", "hourly"], default="hourly")
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--estimator", default="gk")
    ap.add_argument("--lockbox", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    if args.rebuild:
        assemble(args.freq, args.H, estimator=args.estimator, rebuild=True)
    report(args.freq, args.H, estimator=args.estimator, lockbox=args.lockbox)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
