"""Expanding-window affine recalibration of a vol forecast (PLAN §6.1, §7).

The probe found Kronos ranks RV well but is biased low on the *level*
(Mincer–Zarnowitz slope ~0.8). A cheap fix: at each origin t, fit
`log RV = a + b·log(forecast)` on everything strictly before t, then apply it to
the current forecast. Honest — never uses future data.

  recalibrate(origins, forecast, realised, *, min_train=60) -> recalibrated series
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def recalibrate(origins: pd.Series | np.ndarray,
                forecast: pd.Series | np.ndarray,
                realised: pd.Series | np.ndarray,
                *, min_train: int = 60, logs: bool = True) -> np.ndarray:
    """Walk-forward affine correction. Returns an array aligned to `forecast`;
    the first `min_train` entries fall back to the raw forecast."""
    f = np.asarray(forecast, float)
    y = np.asarray(realised, float)
    order = np.argsort(np.asarray(origins))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    fo, yo = f[order], y[order]
    out = fo.copy()
    for k in range(len(fo)):
        if k < min_train:
            continue
        tr = slice(0, k)
        ft, yt = fo[tr], yo[tr]
        m = np.isfinite(ft) & np.isfinite(yt) & (ft > 0) & (yt > 0)
        if m.sum() < min_train // 2:
            continue
        if logs:
            A = np.c_[np.ones(m.sum()), np.log(ft[m])]
            beta, *_ = np.linalg.lstsq(A, np.log(yt[m]), rcond=None)
            out[k] = float(np.exp(beta[0] + beta[1] * np.log(max(fo[k], 1e-9))))
        else:
            A = np.c_[np.ones(m.sum()), ft[m]]
            beta, *_ = np.linalg.lstsq(A, yt[m], rcond=None)
            out[k] = float(max(beta[0] + beta[1] * fo[k], 1e-9))
    return out[inv]
