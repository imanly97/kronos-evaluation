"""Price data — daily and hourly — with a strict no-lookahead contract.

Fetch: Yahoo chart API via curl_cffi (browser-TLS impersonation; the plain UA
gets 429'd), Stooq CSV fallback for daily, on-disk cache.

  get_daily(ticker)   -> full split+dividend-adjusted OHLCV(+amount), tz-naive
  get_hourly(ticker)  -> regular-session hourly OHLCV(+amount), split-adjusted,
                         tz = America/New_York
  refresh_universe(freq)  -> bulk pull for W1

No-lookahead helpers:
  context_before(df, origin, n)   -> the n bars with timestamp strictly < origin
  session_bars(df, day)           -> the intraday bars dated `day`
  sessions(df)                    -> the trading days present in an hourly frame
"""
from __future__ import annotations

import argparse
import io
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DAILY_CACHE,
    HOURLY_CACHE,
    UNIVERSE,
)

OHLC = ["open", "high", "low", "close"]
COLS = OHLC + ["volume", "amount"]
NY = "America/New_York"


class DataError(RuntimeError):
    """No source could satisfy the request."""


# --------------------------------------------------------------------------- #
# fetchers
# --------------------------------------------------------------------------- #
def _chart(ticker: str, rng: str, interval: str, *, prepost: bool = False) -> dict:
    from curl_cffi import requests as cr

    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range={rng}&interval={interval}&events=div%2Csplit"
           f"&includePrePost={'true' if prepost else 'false'}")
    for attempt in range(3):
        try:
            r = cr.get(url, impersonate="chrome124", timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"]
            if not res:
                raise DataError(f"yahoo: no result for {ticker}")
            return res[0]
        except DataError:
            raise
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                raise DataError(f"yahoo fetch failed for {ticker}: {exc}") from exc
            time.sleep(2)
    raise DataError("unreachable")


def _split_factor(res: dict, index: pd.DatetimeIndex) -> pd.Series:
    """Cumulative split adjustment: divide pre-split prices so the series is
    continuous through each split (most recent segment unscaled)."""
    splits = (res.get("events", {}) or {}).get("splits", {}) or {}
    factor = pd.Series(1.0, index=index)
    for ev in splits.values():
        ratio = ev["numerator"] / ev["denominator"]
        split_ts = pd.Timestamp(ev["date"], unit="s", tz="UTC")
        factor.loc[factor.index < split_ts.tz_convert(factor.index.tz or "UTC")
                   if factor.index.tz else split_ts.tz_localize(None)] /= ratio
    return factor


def _from_yahoo_daily(ticker: str) -> pd.DataFrame:
    res = _chart(ticker, "10y", "1d")
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"],
                       "close": q["close"], "volume": q["volume"]}, index=ts)
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is not None:                       # split + dividend adjust OHLC
        f = pd.Series(adj, index=ts) / df["close"]
        for c in OHLC:
            df[c] = df[c] * f
    return _finalize(df, ticker)


def _from_stooq_daily(ticker: str) -> pd.DataFrame:
    from curl_cffi import requests as cr

    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    r = cr.get(url, impersonate="chrome124", timeout=30)
    r.raise_for_status()
    if not r.text.startswith("Date"):
        raise DataError(f"stooq: no CSV for {ticker}")
    raw = pd.read_csv(io.StringIO(r.text))
    raw.columns = [c.lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date")[["open", "high", "low", "close", "volume"]]
    raw.index = raw.index.normalize()
    raw.index.name = None
    return _finalize(raw, ticker)


def _from_yahoo_hourly(ticker: str) -> pd.DataFrame:
    res = _chart(ticker, "730d", "1h")
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(NY)
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"],
                       "close": q["close"], "volume": q["volume"]}, index=ts).dropna()
    # Yahoo's intraday feed is already SPLIT-adjusted (verified on NVDA's 2024-06
    # 10:1 — the pre-split bars come back ~$120, not ~$1200). It is NOT dividend
    # adjusted; for these large-caps the yield is <2%/yr so RV is unaffected.
    _assert_no_split_jump(df, res, ticker)
    # regular-session hourly bars land at :30 (09:30 … 15:30). Anything else is
    # Yahoo's live/partial stub for the current bar — drop it.
    df = df[df.index.minute == 30]
    return _finalize(df, ticker, normalize_index=False)


def _assert_no_split_jump(df: pd.DataFrame, res: dict, ticker: str) -> None:
    """Guard against Yahoo's intraday feed silently changing its adjustment
    convention: no >40% session-to-session close gap within a week of a split."""
    splits = (res.get("events", {}) or {}).get("splits", {}) or {}
    if not splits:
        return
    daily_last = df["close"].groupby(df.index.date).last()
    gap = np.log(daily_last).diff().abs()
    gidx = pd.DatetimeIndex(gap.index)
    for ev in splits.values():
        sd = pd.Timestamp(ev["date"], unit="s", tz="UTC").tz_convert(NY).normalize().tz_localize(None)
        near = gap[(gidx >= sd - pd.Timedelta(days=5)) & (gidx <= sd + pd.Timedelta(days=2))]
        if len(near) and near.max() > 0.40:
            raise DataError(f"{ticker}: {near.max():.0%} close gap near the "
                            f"{sd.date()} split — Yahoo intraday adjustment changed")


def _finalize(df: pd.DataFrame, ticker: str, *, normalize_index: bool = True) -> pd.DataFrame:
    df = df.dropna(how="any").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.empty:
        raise DataError(f"{ticker}: empty after cleaning")
    df["amount"] = df["volume"] * df[OHLC].mean(axis=1)
    return df[COLS]


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def _path(ticker: str, freq: str) -> Path:
    base = DAILY_CACHE if freq == "daily" else HOURLY_CACHE
    return base / f"{ticker.upper()}.parquet"


def _read(ticker: str, freq: str) -> pd.DataFrame | None:
    p = _path(ticker, freq)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:  # noqa: BLE001 — corrupt cache -> refetch
        return None


def _write(ticker: str, freq: str, df: pd.DataFrame) -> None:
    df.to_parquet(_path(ticker, freq))


def get_daily(ticker: str, *, force: bool = False) -> pd.DataFrame:
    if not force:
        cached = _read(ticker, "daily")
        if cached is not None and not cached.empty:
            return cached
    errs = []
    for fn in (_from_yahoo_daily, _from_stooq_daily):
        try:
            df = fn(ticker)
            _write(ticker, "daily", df)
            return df
        except Exception as exc:  # noqa: BLE001
            errs.append(str(exc))
    cached = _read(ticker, "daily")
    if cached is not None:
        return cached
    raise DataError(f"{ticker} daily: all sources failed — " + " | ".join(errs))


def get_hourly(ticker: str, *, force: bool = False) -> pd.DataFrame:
    if not force:
        cached = _read(ticker, "hourly")
        if cached is not None and not cached.empty:
            return cached
    df = _from_yahoo_hourly(ticker)
    _write(ticker, "hourly", df)
    return df


def refresh_universe(freq: str = "daily", tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or UNIVERSE
    getter = get_daily if freq == "daily" else get_hourly
    rows = []
    for t in tickers:
        try:
            df = getter(t, force=True)
            rows.append((t, len(df), df.index.min(), df.index.max()))
            print(f"  {t:6} {len(df):>6} bars  {df.index.min().date()} .. {df.index.max().date()}")
        except DataError as exc:
            rows.append((t, 0, None, None))
            print(f"  {t:6} FAILED — {exc}")
        time.sleep(0.2)
    return pd.DataFrame(rows, columns=["ticker", "bars", "from", "to"])


# --------------------------------------------------------------------------- #
# no-lookahead slicing
# --------------------------------------------------------------------------- #
def _coerce_ts(x, tz=None) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if tz is not None:
        t = t.tz_localize(tz) if t.tz is None else t.tz_convert(tz)
    return t


def context_before(df: pd.DataFrame, origin, n: int) -> pd.DataFrame:
    """The `n` most-recent bars with timestamp **strictly before** `origin`."""
    o = _coerce_ts(origin, df.index.tz)
    ctx = df.loc[df.index < o].tail(n)
    if ctx.empty:
        raise DataError(f"no bars before {origin}")
    if ctx.index.max() >= o:                  # belt-and-braces
        raise DataError(f"lookahead: last ctx bar {ctx.index.max()} >= {origin}")
    return ctx


def sessions(df_hourly: pd.DataFrame) -> list[date]:
    return sorted({ts.date() for ts in df_hourly.index})


def session_bars(df_hourly: pd.DataFrame, day) -> pd.DataFrame:
    d = pd.Timestamp(day).date()
    return df_hourly.loc[[ts.date() == d for ts in df_hourly.index]]


def settled_sessions(df_hourly: pd.DataFrame, *, min_bars: int = 4) -> list[date]:
    """Trading days with at least `min_bars` bars — i.e. complete, gradeable
    sessions (drops the live/partial current day and any data gap)."""
    n = pd.Series(1, index=df_hourly.index).groupby(df_hourly.index.date).sum()
    return sorted(d for d, c in n.items() if c >= min_bars)


def _coerce_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch / inspect the price cache.")
    ap.add_argument("cmd", choices=["refresh", "show"])
    ap.add_argument("--freq", choices=["daily", "hourly"], default="daily")
    ap.add_argument("--tickers", nargs="+")
    args = ap.parse_args(argv)

    if args.cmd == "refresh":
        refresh_universe(args.freq, args.tickers)
    else:
        for t in (args.tickers or UNIVERSE):
            df = (_read(t, args.freq))
            if df is None:
                print(f"{t}: not cached")
            else:
                print(f"{t}: {len(df)} {args.freq} bars  {df.index.min()} .. {df.index.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
