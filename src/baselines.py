"""Classical volatility-forecast baselines (PLAN §6.1) — all walk-forward.

Each baseline answers the same question Kronos does: **given a per-session
realised-vol history (and return history) up to session T, forecast the RV
realised over sessions T+1 … T+H**, in the same units as the target
(`sqrt` of summed session variance).

  rw            random walk: last session's RV, scaled to H
  rolling(k)    sqrt of the mean of the last k session variances, scaled to H
  ewma(lam)     RiskMetrics EWMA variance recursion
  har           HAR-RV (Corsi 2009), direct-H log regression, expanding window
  garch         GARCH(1,1) on session returns (`arch`), expanding window
  gjr           GJR-GARCH(1,1,1) — asymmetric

`walk_forward()` runs a set of them over a list of origins for one name.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# O(1) baselines
# --------------------------------------------------------------------------- #
def rw(rv_hist: pd.Series, ret_hist: pd.Series, H: int) -> float:
    return float(rv_hist.iloc[-1] * np.sqrt(H))


def rolling(rv_hist: pd.Series, ret_hist: pd.Series, H: int, *, k: int = 22) -> float:
    v = rv_hist.iloc[-k:].to_numpy(float) ** 2
    return float(np.sqrt(np.nanmean(v) * H))


def ewma(rv_hist: pd.Series, ret_hist: pd.Series, H: int, *, lam: float = 0.94) -> float:
    v = 0.0
    for r in rv_hist.to_numpy(float) ** 2:
        v = lam * v + (1 - lam) * r
    return float(np.sqrt(v * H))


# --------------------------------------------------------------------------- #
# HAR-RV (Corsi 2009), direct-H horizon, fit in logs on an expanding window
# --------------------------------------------------------------------------- #
def _har_design(rv: np.ndarray) -> np.ndarray:
    """[1, log RV^(d), log RV^(w), log RV^(m)] at each t (needs >= 22 history)."""
    d = rv
    w = pd.Series(rv).rolling(5).mean().to_numpy()
    m = pd.Series(rv).rolling(22).mean().to_numpy()
    return np.column_stack([np.ones_like(d), np.log(d), np.log(w), np.log(m)])


def har(rv_hist: pd.Series, ret_hist: pd.Series, H: int) -> float:
    rv = rv_hist.to_numpy(float)
    rv = np.clip(rv, 1e-8, None)
    if len(rv) < 60:
        return rolling(rv_hist, ret_hist, H)
    X = _har_design(rv)
    # target: log of the RV realised over the next H sessions (variance-summed)
    fwd = np.array([np.sqrt(np.sum(rv[t + 1: t + 1 + H] ** 2)) if t + H < len(rv) else np.nan
                    for t in range(len(rv))])
    ok = np.isfinite(X).all(1) & np.isfinite(fwd) & (fwd > 0)
    if ok.sum() < 40:
        return rolling(rv_hist, ret_hist, H)
    beta, *_ = np.linalg.lstsq(X[ok], np.log(fwd[ok]), rcond=None)
    return float(np.exp(X[-1] @ beta))


# --------------------------------------------------------------------------- #
# GARCH family (arch), on session returns, expanding window
# --------------------------------------------------------------------------- #
def _garch(ret_hist: pd.Series, H: int, *, kind: str) -> float:
    from arch import arch_model

    r = ret_hist.dropna().to_numpy(float) * 100.0     # arch likes ~unit-scale
    if len(r) < 100:
        return float(np.std(r / 100.0) * np.sqrt(H))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = 1 if kind == "gjr" else 0
        am = arch_model(r, vol="GARCH", p=1, o=o, q=1, dist="normal", mean="Zero")
        res = am.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=H, reindex=False)
        var_path = fc.variance.to_numpy().ravel()[:H] / (100.0 ** 2)
    return float(np.sqrt(np.sum(var_path)))


def garch(rv_hist: pd.Series, ret_hist: pd.Series, H: int) -> float:
    return _garch(ret_hist, H, kind="garch")


def gjr(rv_hist: pd.Series, ret_hist: pd.Series, H: int) -> float:
    return _garch(ret_hist, H, kind="gjr")


# --------------------------------------------------------------------------- #
# walk-forward driver
# --------------------------------------------------------------------------- #
DEFAULT = {
    "rw": rw,
    "rolling22": lambda rv, rt, H: rolling(rv, rt, H, k=22),
    "ewma94": lambda rv, rt, H: ewma(rv, rt, H, lam=0.94),
    "har": har,
    "garch": garch,
    "gjr": gjr,
}
FAST = {k: DEFAULT[k] for k in ("rw", "rolling22", "ewma94")}   # no per-origin fit


def walk_forward(
    rv_series: pd.Series,
    ret_series: pd.Series,
    origins: list,
    H: int,
    *,
    models: dict | None = None,
    min_history: int = 60,
) -> pd.DataFrame:
    """For each origin, forecast RV(origin+1 … origin+H) with every model.

    `rv_series` / `ret_series` are indexed by session date. Every forecast at
    origin T uses only `series.loc[:T]` — strictly causal.
    """
    models = models or DEFAULT
    rv_series = rv_series.sort_index()
    ret_series = ret_series.sort_index()
    rows = []
    for o in origins:
        rv_h = rv_series.loc[:o]
        rt_h = ret_series.loc[:o]
        if len(rv_h) < min_history:
            continue
        row = {"origin": o}
        for name, fn in models.items():
            try:
                row[name] = fn(rv_h, rt_h, H)
            except Exception:  # noqa: BLE001 — a single fit failure shouldn't sink the run
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("origin")
