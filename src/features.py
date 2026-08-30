"""Cross-sectional + per-name-historical features over a day's forecasts.

Three things the triage agent leans on (PLAN §5.1):

  rank_table(fc)          — median / P(up) / dispersion / strength, ranked, with
                            each name's strength z-scored against its own trailing
                            history when the forecast cache has enough of it.
  strength_z(hist, x)     — the z-score itself, guarded for short history.
  correlated_cluster(...) — are the names that trade with this one all pointing
                            the same way (macro / sector), or is this idiosyncratic?

Pure functions of a forecast table + no-lookahead price history. No LLM.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import (
    CLUSTER_CORR_MIN,
    CLUSTER_CORR_WINDOW,
    DEV_UNIVERSE,
    STRENGTH_Z_WINDOW,
    UNIVERSE,
)
from .data import DataError, load_context

STRENGTH_Z_MIN_OBS = 20     # below this, no per-name z (fall back to cross-sectional)


def strength_z(hist: pd.Series | np.ndarray, current: float) -> tuple[float, str]:
    """z of `current` strength against trailing history.

    Returns (z, basis) where basis is "per-name" or "insufficient".
    """
    h = pd.Series(hist, dtype=float).dropna()
    h = h.iloc[-STRENGTH_Z_WINDOW:]
    if len(h) < STRENGTH_Z_MIN_OBS:
        return float("nan"), "insufficient"
    sd = float(h.std(ddof=1))
    if sd == 0:
        return 0.0, "per-name"
    return (current - float(h.mean())) / sd, "per-name"


def rank_table(fc: pd.DataFrame, *, strength_hist: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Rank one asof's forecasts by |strength|; attach strength_z.

    `fc` needs columns: ticker, q50, q05, q95, p_up, std, strength.
    `strength_hist[ticker]` (optional) is that name's trailing strength series
    *excluding* today; when absent or too short, strength_z is the cross-sectional
    z of today's |strength| and `strength_z_basis` says so.
    """
    d = fc.copy()
    d["dispersion"] = d["std"]
    d["strength_abs"] = d["strength"].abs()

    xs_mean, xs_sd = d["strength_abs"].mean(), d["strength_abs"].std(ddof=0) or 1.0
    zs, bases = [], []
    for _, row in d.iterrows():
        if strength_hist and row["ticker"] in strength_hist:
            z, basis = strength_z(strength_hist[row["ticker"]], row["strength_abs"])
        else:
            z, basis = float("nan"), "insufficient"
        if basis == "insufficient":
            z = (row["strength_abs"] - xs_mean) / xs_sd
            basis = "cross-sectional"
        zs.append(z)
        bases.append(basis)
    d["strength_z"] = zs
    d["strength_z_basis"] = bases

    d = d.sort_values("strength_abs", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    cols = ["rank", "ticker", "q50", "p_up", "dispersion", "strength",
            "strength_abs", "strength_z", "strength_z_basis", "q05", "q95"]
    return d[[c for c in cols if c in d.columns]]


def _returns_matrix(tickers: list[str], asof: str, window: int) -> pd.DataFrame:
    """Trailing daily returns, one column per ticker, bars strictly before asof."""
    cols = {}
    for t in tickers:
        try:
            pl = load_context(t, asof)
        except DataError:
            continue
        cols[t] = pl.df["close"].pct_change().dropna().iloc[-window:]
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).dropna(how="any")


def correlated_cluster(
    ticker: str,
    asof: str,
    fc: pd.DataFrame,
    *,
    universe: list[str] | None = None,
    window: int = CLUSTER_CORR_WINDOW,
    min_corr: float = CLUSTER_CORR_MIN,
) -> dict:
    """Peers that co-move with `ticker`, and whether their forecasts agree in sign.

    Returns {peers, mean_abs_corr, same_sign_frac, read} where read is one of
    "idiosyncratic" | "cluster-aligned" | "cluster-divergent".
    """
    universe = universe or UNIVERSE
    peers_pool = [t for t in universe if t != ticker]
    rmat = _returns_matrix([ticker] + peers_pool, asof, window)
    if rmat.empty or ticker not in rmat.columns:
        return {"peers": [], "mean_abs_corr": float("nan"),
                "same_sign_frac": float("nan"), "read": "idiosyncratic"}

    corr = rmat.corr()[ticker].drop(ticker)
    peers = corr[corr.abs() >= min_corr].sort_values(ascending=False)
    if peers.empty:
        return {"peers": [], "mean_abs_corr": 0.0,
                "same_sign_frac": float("nan"), "read": "idiosyncratic"}

    med = fc.set_index("ticker")["q50"]
    my_sign = np.sign(med.get(ticker, 0.0))
    peer_signs = [np.sign(med[p]) for p in peers.index if p in med.index]
    same = float(np.mean([s == my_sign and s != 0 for s in peer_signs])) if peer_signs else float("nan")

    if np.isnan(same):
        read = "idiosyncratic"
    elif same >= 0.6:
        read = "cluster-aligned"
    else:
        read = "cluster-divergent"
    return {
        "peers": list(peers.index),
        "peer_corr": {p: round(float(peers[p]), 2) for p in peers.index},
        "mean_abs_corr": round(float(peers.abs().mean()), 2),
        "same_sign_frac": None if np.isnan(same) else round(same, 2),
        "read": read,
    }


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect cluster read for a ticker/date.")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--universe", nargs="+", default=DEV_UNIVERSE)
    args = ap.parse_args(argv)
    # a stub forecast table (median 0) just to exercise the correlation half
    fc = pd.DataFrame({"ticker": args.universe, "q50": 0.0})
    out = correlated_cluster(args.ticker, args.asof, fc, universe=args.universe)
    for k, v in out.items():
        print(f"  {k:16} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
