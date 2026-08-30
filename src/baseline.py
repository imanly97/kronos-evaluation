"""Kronos baseline — how good is the raw forecast, before any agent touches it?

Pure deterministic analysis of the forecast cache against realised closes. No
LLM, no triage. Answers: is the distribution calibrated, does the median call
direction, is there a bias, and — the one that matters for triage — does the
model's own strength signal rank-order the eventful days?

**Contamination:** Kronos-small's pretraining data ends ~June 2024 (Shi et al.,
arXiv:2508.02739); the replay window is May–Aug 2025, ~11 months out of sample.
The forecasts are genuine predictions, not lookups of known prices.

    .venv/bin/python -m src.baseline            # full report
    .venv/bin/python -m src.baseline --json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.store import joined, load

QLEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]


def _frame() -> pd.DataFrame:
    j = joined()
    if j.empty or "ret" not in j.columns:
        return pd.DataFrame()
    j = j.dropna(subset=["ret", "actual_quantile"]).copy()
    j["asof"] = pd.to_datetime(j["asof"])
    j["signed_err"] = j["ret"] - j["q50"]
    j["abs_err"] = j["signed_err"].abs()
    j["abs_ret"] = j["ret"].abs()
    j["width_90"] = j["q95"] - j["q05"]
    j["strength_abs"] = j["strength"].abs()
    return j


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def calibration(j: pd.DataFrame) -> dict:
    """PIT: if the forecast is calibrated, actual_quantile ~ Uniform(0,1)."""
    q = j["actual_quantile"].to_numpy()
    # KS distance to uniform
    qs = np.sort(q)
    n = len(qs)
    d = np.max(np.abs(np.arange(1, n + 1) / n - qs))
    edges = np.linspace(0, 1, 11)
    hist = np.histogram(q, bins=edges)[0] / n
    return {
        "n": n,
        "pit_mean": round(float(q.mean()), 3),          # 0.5 = unbiased
        "pit_std": round(float(q.std()), 3),            # 0.29 = uniform
        "ks_vs_uniform": round(float(d), 3),            # lower is better
        "pit_deciles": [round(float(x), 3) for x in hist],   # flat = calibrated
        "frac_in_05_95": round(float(np.mean((q > 0.05) & (q < 0.95))), 3),   # target 0.90
        "frac_in_25_75": round(float(np.mean((q > 0.25) & (q < 0.75))), 3),   # target 0.50
    }


def direction(j: pd.DataFrame) -> dict:
    up_actual = j["ret"] > 0
    med_call = j["q50"] > 0
    pup_call = j["p_up"] > 0.5
    base = float(up_actual.mean())
    return {
        "up_base_rate": round(base, 3),
        "median_sign_hit": round(float((med_call == up_actual).mean()), 3),
        "p_up_gt_half_hit": round(float((pup_call == up_actual).mean()), 3),
        "hit_vs_always_up": round(float((med_call == up_actual).mean()) - max(base, 1 - base), 3),
    }


def bias(j: pd.DataFrame) -> dict:
    return {
        "mean_signed_err": round(float(j["signed_err"].mean()), 5),   # >0: model too low
        "median_signed_err": round(float(j["signed_err"].median()), 5),
        "mean_abs_err": round(float(j["abs_err"].mean()), 5),
        "mean_forecast_median": round(float(j["q50"].mean()), 5),
        "mean_actual_ret": round(float(j["ret"].mean()), 5),
        "mean_90pct_width": round(float(j["width_90"].mean()), 4),
    }


def skill(j: pd.DataFrame) -> dict:
    """Does the model's own strength signal rank-order the eventful days?"""
    from scipy.stats import spearmanr

    rho_ev, p_ev = spearmanr(j["strength_abs"], j["abs_ret"])
    rho_disp, _ = spearmanr(j["std"], j["abs_ret"])
    # top strength decile vs bottom decile, by |actual return|
    hi = j[j["strength_abs"] >= j["strength_abs"].quantile(0.9)]
    lo = j[j["strength_abs"] <= j["strength_abs"].quantile(0.1)]
    return {
        "spearman_strength_vs_abs_ret": round(float(rho_ev), 3),
        "spearman_pvalue": round(float(p_ev), 4),
        "spearman_dispersion_vs_abs_ret": round(float(rho_disp), 3),
        "abs_ret_top_strength_decile": round(float(hi["abs_ret"].mean()), 4),
        "abs_ret_bottom_strength_decile": round(float(lo["abs_ret"].mean()), 4),
        "eventful_decile_ratio": round(
            float(hi["abs_ret"].mean() / max(lo["abs_ret"].mean(), 1e-9)), 2),
    }


def by_week(j: pd.DataFrame) -> pd.DataFrame:
    g = j.set_index("asof").groupby(pd.Grouper(freq="W"))
    return pd.DataFrame({
        "n": g.size(),
        "cover_90": g.apply(lambda x: ((x["actual_quantile"] > 0.05) &
                                       (x["actual_quantile"] < 0.95)).mean()),
        "dir_hit": g.apply(lambda x: ((x["q50"] > 0) == (x["ret"] > 0)).mean()),
        "mean_signed_err": g["signed_err"].mean(),
    }).dropna()


def by_ticker(j: pd.DataFrame) -> pd.DataFrame:
    g = j.groupby("ticker")
    return pd.DataFrame({
        "n": g.size(),
        "cover_90": g.apply(lambda x: ((x["actual_quantile"] > 0.05) &
                                       (x["actual_quantile"] < 0.95)).mean()),
        "dir_hit": g.apply(lambda x: ((x["q50"] > 0) == (x["ret"] > 0)).mean()),
        "mean_signed_err": g["signed_err"].mean(),
        "mean_abs_err": g["abs_err"].mean(),
    }).sort_values("mean_abs_err", ascending=False)


def report() -> dict:
    j = _frame()
    if j.empty:
        return {"error": "no graded forecasts — run src.forecasts then src.grade"}
    return {
        "window": [str(j["asof"].min().date()), str(j["asof"].max().date())],
        "forecast_days": len(j), "tickers": int(j["ticker"].nunique()),
        "sessions": int(j["asof"].nunique()),
        "calibration": calibration(j),
        "direction": direction(j),
        "bias": bias(j),
        "skill": skill(j),
    }


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kronos baseline analysis.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--by-ticker", action="store_true")
    ap.add_argument("--by-week", action="store_true")
    args = ap.parse_args(argv)

    rep = report()
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0

    print(f"Kronos-small baseline — {rep['window'][0]} .. {rep['window'][1]}  "
          f"({rep['forecast_days']} forecast-days, {rep['sessions']} sessions, "
          f"{rep['tickers']} tickers)\n")
    for section in ("calibration", "direction", "bias", "skill"):
        print(f"[{section}]")
        for k, v in rep[section].items():
            print(f"  {k:32} {v}")
        print()
    if args.by_ticker:
        print(by_ticker(_frame()).round(3).to_string())
    if args.by_week:
        print(by_week(_frame()).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
