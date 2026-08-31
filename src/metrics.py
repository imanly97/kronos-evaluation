"""Forecast-evaluation metrics and tests (PLAN §6, §8).

Loss functions
  qlike(f, y)        QLIKE — robust to noise in the vol proxy. **primary.**
  mse_log(f, y)      squared error on log RV
  pinball(y, q, tau) quantile (check) loss

Tests
  mincer_zarnowitz(f, y)   y = a + b f ; HAC t-tests of a=0, b=1
  diebold_mariano(la, lb)  equal-predictive-accuracy test, Newey-West SEs
  pit(y, q_grid, taus)     probability-integral transform for calibration
  coverage(y, lo, hi)      empirical interval coverage
  model_confidence_set(losses)   Hansen–Lunde–Nason MCS (via `arch`)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def qlike(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE, comparing variances (f, y are vols)."""
    f2 = np.clip(np.asarray(f, float) ** 2, 1e-12, None)
    y2 = np.clip(np.asarray(y, float) ** 2, 1e-12, None)
    return y2 / f2 - np.log(y2 / f2) - 1.0


def mse_log(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.log(np.clip(f, 1e-9, None)) - np.log(np.clip(y, 1e-9, None))) ** 2


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = np.asarray(y, float) - np.asarray(q, float)
    return np.maximum(tau * d, (tau - 1) * d)


# --------------------------------------------------------------------------- #
# Mincer–Zarnowitz
# --------------------------------------------------------------------------- #
def mincer_zarnowitz(f: np.ndarray, y: np.ndarray, *, logs: bool = True,
                     hac_lags: int | None = None) -> dict:
    import statsmodels.api as sm

    f = np.asarray(f, float); y = np.asarray(y, float)
    m = np.isfinite(f) & np.isfinite(y) & (f > 0) & (y > 0)
    F = np.log(f[m]) if logs else f[m]
    Y = np.log(y[m]) if logs else y[m]
    X = sm.add_constant(F)
    lags = hac_lags if hac_lags is not None else int(4 * (len(Y) / 100) ** (2 / 9))
    res = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})
    a, b = res.params
    # joint test a=0, b=1
    R, r = np.array([[1, 0], [0, 1]]), np.array([0, 1])
    wald = res.f_test((R, r))
    return {
        "n": int(m.sum()), "intercept": float(a), "slope": float(b),
        "r2": float(res.rsquared),
        "t_intercept_eq_0": float(res.tvalues[0]),
        "t_slope_eq_1": float((b - 1) / res.bse[1]),
        "joint_p": float(np.asarray(wald.pvalue).item()),
    }


# --------------------------------------------------------------------------- #
# Diebold–Mariano
# --------------------------------------------------------------------------- #
def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, *,
                    hac_lags: int | None = None) -> dict:
    """H0: equal expected loss. Positive stat => model A worse (higher loss)."""
    import statsmodels.api as sm

    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return {"n": len(d), "dm_stat": np.nan, "p_value": np.nan, "mean_diff": np.nan}
    lags = hac_lags if hac_lags is not None else int(4 * (len(d) / 100) ** (2 / 9))
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC",
                                         cov_kwds={"maxlags": max(lags, 1)})
    return {
        "n": len(d), "mean_diff": float(d.mean()),
        "dm_stat": float(res.tvalues[0]), "p_value": float(res.pvalues[0]),
    }


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def pit(y: np.ndarray, q_grid: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Interpolate each y's quantile level from a per-obs quantile grid.

    q_grid: (n_obs, n_taus) forecast quantiles; taus: the levels. Returns the
    PIT value in [0,1] for each obs (linear interp, clamped in the tails).
    """
    y = np.asarray(y, float)
    taus = np.asarray(taus, float)
    out = np.empty(len(y))
    for i, (yi, qi) in enumerate(zip(y, np.asarray(q_grid, float))):
        order = np.argsort(qi)
        qs, ts = qi[order], taus[order]
        if yi <= qs[0]:
            slope = (ts[1] - ts[0]) / (qs[1] - qs[0]) if qs[1] != qs[0] else 0.0
            out[i] = np.clip(ts[0] + slope * (yi - qs[0]), 1e-4, 1 - 1e-4)
        elif yi >= qs[-1]:
            slope = (ts[-1] - ts[-2]) / (qs[-1] - qs[-2]) if qs[-1] != qs[-2] else 0.0
            out[i] = np.clip(ts[-1] + slope * (yi - qs[-1]), 1e-4, 1 - 1e-4)
        else:
            out[i] = np.interp(yi, qs, ts)
    return out


def pit_exact(y: np.ndarray, sample_arrays) -> np.ndarray:
    """Rank PIT: for each obs, (fraction of samples < y) with a mid-rank tie
    correction and jitter so a calibrated forecast gives Uniform(0,1)."""
    y = np.asarray(y, float)
    out = np.empty(len(y))
    for i, (yi, s) in enumerate(zip(y, sample_arrays)):
        s = np.sort(np.asarray(s, float))
        n = len(s)
        below = np.searchsorted(s, yi, side="left")
        equal = np.searchsorted(s, yi, side="right") - below
        out[i] = (below + 0.5 * (equal + 1)) / (n + 1)
    return np.clip(out, 1e-6, 1 - 1e-6)


def crps_sample(y: np.ndarray, sample_arrays) -> np.ndarray:
    """CRPS estimated from samples: E|X-y| - 0.5 E|X-X'| (per observation)."""
    y = np.asarray(y, float)
    out = np.empty(len(y))
    for i, (yi, s) in enumerate(zip(y, sample_arrays)):
        s = np.asarray(s, float)
        term1 = np.mean(np.abs(s - yi))
        term2 = np.mean(np.abs(s[:, None] - s[None, :]))
        out[i] = term1 - 0.5 * term2
    return out


def pit_ks(pit_vals: np.ndarray) -> dict:
    from scipy.stats import kstest

    p = np.asarray(pit_vals, float)
    p = p[np.isfinite(p)]
    ks = kstest(p, "uniform")
    return {"n": len(p), "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "mean": float(p.mean()), "std": float(p.std())}


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))


# --------------------------------------------------------------------------- #
# Model Confidence Set
# --------------------------------------------------------------------------- #
def model_confidence_set(losses: pd.DataFrame, *, alpha: float = 0.10,
                         reps: int = 1000, block: int = 10) -> dict:
    """losses: (n_obs, n_models) per-observation loss. Returns the models in the
    (1-alpha) MCS and their p-values (Hansen–Lunde–Nason, via `arch`)."""
    from arch.bootstrap import MCS

    L = losses.dropna()
    mcs = MCS(L, size=alpha, reps=reps, block_size=block, method="R")
    mcs.compute()
    return {
        "included": list(mcs.included),
        "excluded": list(mcs.excluded),
        "pvalues": mcs.pvalues["Pvalue"].to_dict(),
    }
