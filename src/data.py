"""Daily price loader with a strict no-lookahead rule.

Contract
--------
`load_context(ticker, asof)` returns the daily bars Kronos is allowed to see when
generating a signal *for* `asof`: every bar is strictly BEFORE `asof`. The first
bar Kronos predicts is dated `asof` itself.

`load_actual(ticker, asof)` returns the realised close ON `asof` (and the prior
close), used the next day by the evaluator. It never feeds the model.

Sources: yfinance (primary) -> Stooq CSV (fallback) -> on-disk cache (last resort).
Whichever source answered is recorded so the blog can report it honestly.
"""
from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR, LOOKBACK_BARS, UNIVERSE

OHLC = ["open", "high", "low", "close"]
COLS = OHLC + ["volume", "amount"]


class DataError(RuntimeError):
    """Raised when no source can satisfy the no-lookahead contract."""


@dataclass
class PriceLoad:
    ticker: str
    asof: date
    df: pd.DataFrame          # index = tz-naive DatetimeIndex, columns = COLS
    source: str               # "yfinance" | "stooq" | "cache"
    refreshed: bool

    @property
    def last_context_date(self) -> pd.Timestamp:
        return self.df.index[-1]


# --------------------------------------------------------------------------- #
# source fetchers — each returns a full-history OHLCV frame or raises
# --------------------------------------------------------------------------- #
def _from_yahoo(ticker: str) -> pd.DataFrame:
    """Yahoo's public chart API via curl_cffi (browser TLS impersonation).

    Returns split+dividend adjusted OHLCV: we scale OHLC by adjclose/close so the
    series Kronos tokenizes has no artificial jumps.
    """
    from curl_cffi import requests as cr

    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        "?range=5y&interval=1d&events=div%2Csplit"
    )
    r = cr.get(url, impersonate="chrome124", timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"]
    if not res:
        raise DataError(f"yahoo returned no result for {ticker}")
    res = res[0]
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"],
         "close": q["close"], "volume": q["volume"]},
        index=ts,
    )
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is not None:
        factor = pd.Series(adj, index=ts) / df["close"]
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * factor
    df = df.dropna(how="any")
    if df.empty:
        raise DataError(f"yahoo frame empty after cleaning for {ticker}")
    return df


def _from_stooq(ticker: str) -> pd.DataFrame:
    """Stooq daily CSV. Split-adjusted only; used when Yahoo is unreachable."""
    from curl_cffi import requests as cr

    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    r = cr.get(url, impersonate="chrome124", timeout=30)
    r.raise_for_status()
    body = r.text
    if not body.startswith("Date"):
        raise DataError(f"stooq returned no CSV for {ticker}: {body[:80]!r}")
    raw = pd.read_csv(io.StringIO(body))
    raw.columns = [c.lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date")[["open", "high", "low", "close", "volume"]]
    raw.index = raw.index.normalize()
    raw.index.name = None
    return raw


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.csv"


def _read_cache(ticker: str) -> pd.DataFrame | None:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    return df


def _write_cache(ticker: str, df: pd.DataFrame, source: str) -> None:
    out = df.copy()
    out["_source"] = source
    out.to_csv(_cache_path(ticker))


def _add_amount(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[OHLC].mean(axis=1)
    return df


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _refresh(ticker: str) -> tuple[pd.DataFrame, str]:
    errors = []
    for name, fn in (("yahoo", _from_yahoo), ("stooq", _from_stooq)):
        try:
            df = fn(ticker)
            _write_cache(ticker, df, name)
            return df, name
        except Exception as exc:  # noqa: BLE001 — fall through to next source
            errors.append(f"{name}: {exc}")
    cached = _read_cache(ticker)
    if cached is not None:
        return cached.drop(columns=[c for c in cached.columns if c.startswith("_")]), "cache"
    raise DataError(f"all sources failed for {ticker}:\n  " + "\n  ".join(errors))


def _needs_refresh(ticker: str, need_through: date, force: bool) -> bool:
    if force:
        return True
    cached = _read_cache(ticker)
    if cached is None or cached.empty:
        return True
    return cached.index.max().date() < need_through


def load_context(ticker: str, asof: date, *, force: bool = False) -> PriceLoad:
    """Bars strictly before `asof`, most recent `LOOKBACK_BARS` of them."""
    asof = _coerce_date(asof)
    # For signal generation we only need history through the trading day before
    # asof; asking for `asof - 1` avoids a pointless refresh when asof is today
    # and today's bar does not exist yet.
    need_through = asof - timedelta(days=1)
    refreshed = _needs_refresh(ticker, need_through, force)
    if refreshed:
        full, source = _refresh(ticker)
    else:
        full = _read_cache(ticker)
        full = full.drop(columns=[c for c in full.columns if c.startswith("_")])
        source = "cache"

    ctx = full[full.index < pd.Timestamp(asof)].tail(LOOKBACK_BARS)
    if len(ctx) < LOOKBACK_BARS // 2:
        raise DataError(
            f"{ticker}: only {len(ctx)} bars before {asof} (need ~{LOOKBACK_BARS})"
        )
    if ctx.index.max() >= pd.Timestamp(asof):
        raise DataError(f"{ticker}: lookahead — last bar {ctx.index.max()} >= {asof}")

    return PriceLoad(ticker, asof, _add_amount(ctx[["open", "high", "low", "close", "volume"]]),
                     source, refreshed)


@dataclass
class ActualLoad:
    ticker: str
    asof: date
    prev_close: float
    actual_close: float
    source: str

    @property
    def ret(self) -> float:
        return self.actual_close / self.prev_close - 1.0


def load_actual(ticker: str, asof: date, *, force: bool = True) -> ActualLoad:
    """Realised close on `asof` and the prior trading day's close."""
    asof = _coerce_date(asof)
    if _needs_refresh(ticker, asof, force):
        full, source = _refresh(ticker)
    else:
        full = _read_cache(ticker)
        full = full.drop(columns=[c for c in full.columns if c.startswith("_")])
        source = "cache"

    upto = full[full.index <= pd.Timestamp(asof)]
    if upto.empty or upto.index.max() != pd.Timestamp(asof):
        raise DataError(f"{ticker}: no settled bar for {asof} yet (last {upto.index.max() if not upto.empty else 'n/a'})")
    if len(upto) < 2:
        raise DataError(f"{ticker}: need a prior close before {asof}")
    return ActualLoad(ticker, asof, float(upto["close"].iloc[-2]),
                      float(upto["close"].iloc[-1]), source)


def _coerce_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d), "%Y-%m-%d").date()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Load / refresh daily prices.")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--tickers", nargs="+", default=UNIVERSE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--actual", action="store_true", help="show realised close on asof")
    args = ap.parse_args(argv)

    asof = _coerce_date(args.asof)
    rows = []
    for t in args.tickers:
        try:
            if args.actual:
                a = load_actual(t, asof, force=args.force)
                rows.append((t, "actual", f"{a.prev_close:.2f}", f"{a.actual_close:.2f}",
                             f"{a.ret * 100:+.2f}%", a.source))
            else:
                pl = load_context(t, asof, force=args.force)
                rows.append((t, f"{len(pl.df)} bars", str(pl.df.index.min().date()),
                             str(pl.last_context_date.date()), "", pl.source))
        except DataError as exc:
            rows.append((t, "ERROR", str(exc)[:60], "", "", ""))

    w = [max(len(str(r[i])) for r in rows + [("ticker", "info", "from", "to", "ret", "src")])
         for i in range(6)]
    hdr = ("ticker", "info", "from", "to", "ret", "src")
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
