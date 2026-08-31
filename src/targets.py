"""Realised forecast targets, built from *actual* bars only (PLAN §6).

Everything here is ground truth — what a forecast is graded against. Model output
lives in `forecast.py`; keeping them apart makes the no-lookahead boundary easy
to audit.

Volatility estimators (per-bar, then aggregated over a window):
  cc      close-to-close: std of log returns. Simple, noisy.
  parkinson   range: (ln H/L)^2 / (4 ln 2). ~5x more efficient than cc.
  gk      Garman-Klass: uses O,H,L,C. Most efficient of the three. **primary.**

Targets:
  realised_vol(bars, estimator)          scalar RV over a window of bars
  forward_session_rv(hourly, day, H)     RV realised over the next H sessions (hourly study)
  forward_daily_rv(daily, day, H)        RV realised over the next H daily bars (daily study)
  forward_return(df, origin, H)          cumulative log return over the horizon
  forward_direction(df, origin, H)       sign of that return  (+1 / -1 / 0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BARS_PER_SESSION
from .data import DataError, session_bars, settled_sessions

_GK_C = 2 * np.log(2) - 1


def _bar_gk(o, h, l, c) -> np.ndarray:
    hl = np.log(h / l) ** 2
    co = np.log(c / o) ** 2
    return 0.5 * hl - _GK_C * co


def _bar_parkinson(h, l) -> np.ndarray:
    return np.log(h / l) ** 2 / (4 * np.log(2))


def realised_vol(bars: pd.DataFrame, estimator: str = "gk") -> float:
    """RV over a window of bars — sqrt of the summed per-bar variance.

    Units: same as a per-bar return std, aggregated over the window (i.e. NOT
    annualised). Compare like-for-like against a forecast on the same window.
    """
    o, h, l, c = (bars[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    if estimator == "cc":
        r = np.diff(np.log(c))
        return float(np.sqrt(np.sum(r * r))) if len(r) else float("nan")
    if estimator == "parkinson":
        v = _bar_parkinson(h, l)
    elif estimator == "gk":
        v = _bar_gk(o, h, l, c)
    else:
        raise ValueError(f"unknown estimator {estimator!r}")
    v = v[np.isfinite(v)]
    v = np.clip(v, 0.0, None)          # a malformed bar contributes zero variance
    return float(np.sqrt(np.sum(v))) if len(v) else float("nan")


# --------------------------------------------------------------------------- #
# forward (realised) targets — hourly study
# --------------------------------------------------------------------------- #
def _next_sessions(hourly: pd.DataFrame, day, H: int) -> list:
    days = settled_sessions(hourly)
    d = pd.Timestamp(day).date()
    if d not in days:
        raise DataError(f"{day} is not a settled session")
    i = days.index(d)
    if i + H >= len(days):
        raise DataError(f"not enough sessions after {day} for H={H}")
    return days[i + 1: i + 1 + H]


def forward_session_rv(hourly: pd.DataFrame, origin_day, H: int,
                       estimator: str = "gk") -> float:
    """RV realised over the H sessions *after* `origin_day`, from intraday bars.

    Origin = the close of `origin_day`; target = sessions origin+1 … origin+H.
    Intraday only — the overnight jump between sessions is excluded (PLAN §6.1).
    """
    tgt = _next_sessions(hourly, origin_day, H)
    var = 0.0
    for s in tgt:
        b = session_bars(hourly, s)
        if len(b) < 3:
            raise DataError(f"session {s} has only {len(b)} bars")
        rv = realised_vol(b, estimator)
        var += rv * rv
    return float(np.sqrt(var))


def forward_return(df: pd.DataFrame, origin, H_bars: int) -> float:
    """Cumulative log return of the H_bars bars strictly after `origin`."""
    fut = df.loc[df.index > pd.Timestamp(origin, tz=df.index.tz)].head(H_bars)
    if len(fut) < H_bars:
        raise DataError(f"only {len(fut)} bars after {origin}, need {H_bars}")
    c = fut["close"].to_numpy(float)
    return float(np.log(c[-1] / c[0]))


def session_forward_return(hourly: pd.DataFrame, origin_day, H: int) -> float:
    """Close-to-close log return from origin_day's close to session origin+H's close."""
    days = settled_sessions(hourly)
    d = pd.Timestamp(origin_day).date()
    i = days.index(d)
    if i + H >= len(days):
        raise DataError(f"not enough sessions after {origin_day}")
    c0 = float(session_bars(hourly, days[i])["close"].iloc[-1])
    c1 = float(session_bars(hourly, days[i + H])["close"].iloc[-1])
    return float(np.log(c1 / c0))


# --------------------------------------------------------------------------- #
# forward (realised) targets — daily study
# --------------------------------------------------------------------------- #
def forward_daily_rv(daily: pd.DataFrame, origin_day, H: int,
                     estimator: str = "gk") -> float:
    """RV over the H daily bars *after* `origin_day` (daily study)."""
    o = pd.Timestamp(origin_day).normalize()
    fut = daily.loc[daily.index > o].head(H)
    if len(fut) < H:
        raise DataError(f"only {len(fut)} daily bars after {origin_day}, need {H}")
    return realised_vol(fut, estimator)


def forward_daily_return(daily: pd.DataFrame, origin_day, H: int) -> float:
    o = pd.Timestamp(origin_day).normalize()
    c0 = float(daily.loc[daily.index <= o, "close"].iloc[-1])
    fut = daily.loc[daily.index > o, "close"].head(H)
    if len(fut) < H:
        raise DataError(f"only {len(fut)} daily bars after {origin_day}")
    return float(np.log(fut.iloc[-1] / c0))


def direction(x: float) -> int:
    return int(np.sign(x))
