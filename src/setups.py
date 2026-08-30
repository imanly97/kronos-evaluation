"""Technical-feature extraction + a controlled-vocabulary setup label.

`setup_features(df)` is a pure function of a no-lookahead OHLCV window: it returns
the handful of deterministic numbers the triage agent is allowed to talk about
(trend slope, distance from the N-day high/low, range position, realised-vol
ratio, recent gap, stretch from the moving average).

`classify(feats)` maps those numbers onto exactly one label from `config.SETUPS`
via a fixed priority ladder — the shared key between a morning brief and a desk
memory entry (PLAN §3 D10, §5.1).

Everything here is deterministic. No LLM, no network.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import ANALOG_LOOKBACK, ANALOG_MIN, SETUPS
from .data import DataError, load_context

# --- windows / thresholds (module-level so the eval harness can cite them) --- #
TREND_WIN = 20          # bars for the trend-slope regression
CHANNEL_WIN = 63        # bars for the high/low channel and range position
RVOL_SHORT = 5
RVOL_LONG = 63
MA_WIN = 50
GAP_LOOKBACK = 3        # a gap this recent still counts as "post-gap"

BREAKOUT_RANGE_POS = 0.92     # within the top 8% of the channel
BREAKDOWN_RANGE_POS = 0.08
VOL_EXPANSION_RATIO = 1.8     # short rvol this many x the long rvol
GAP_MIN = 0.04               # 4% overnight gap
STRETCH_MIN = 0.10           # 10% from the 50-day MA
TREND_SLOPE_MIN = 0.0015     # per-bar log-return slope ~ 0.15%/day
TREND_R2_MIN = 0.45          # a "clean" trend


@dataclass
class SetupFeatures:
    ticker: str
    asof: str
    context_to: str
    close: float
    trend_slope: float      # per-bar slope of log(close) over TREND_WIN
    trend_r2: float
    dist_from_high: float    # (close - channel_high) / close   (<= 0)
    dist_from_low: float     # (close - channel_low)  / close   (>= 0)
    range_pos: float         # 0 at channel low, 1 at channel high
    rvol_short: float        # annualised realised vol, short window
    rvol_long: float
    rvol_ratio: float
    last_gap: float          # largest |overnight gap| within GAP_LOOKBACK bars
    gap_age: int             # bars since that gap (0 = most recent bar); -1 if none
    stretch_from_ma: float   # (close - SMA) / SMA

    def as_row(self) -> dict:
        return asdict(self)


def _slope_r2(y: np.ndarray) -> tuple[float, float]:
    """OLS slope of y on 0..n-1, plus R²."""
    n = len(y)
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    yc = y - y.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return 0.0, 0.0
    slope = float((x * yc).sum() / denom)
    fit = slope * x
    ss_res = float(((yc - fit) ** 2).sum())
    ss_tot = float((yc ** 2).sum())
    r2 = 0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
    return slope, r2


def setup_features(df: pd.DataFrame, *, ticker: str = "", asof: str = "") -> SetupFeatures:
    """Compute the deterministic feature set from a no-lookahead OHLCV window."""
    if len(df) < CHANNEL_WIN + 1:
        raise ValueError(f"need >= {CHANNEL_WIN + 1} bars, got {len(df)}")
    close = df["close"].astype(float)
    high, low, open_ = df["high"].astype(float), df["low"].astype(float), df["open"].astype(float)

    logc = np.log(close.to_numpy())
    slope, r2 = _slope_r2(logc[-TREND_WIN:])

    chan = df.iloc[-CHANNEL_WIN:]
    ch_hi, ch_lo = float(chan["high"].max()), float(chan["low"].min())
    c = float(close.iloc[-1])
    rng = max(ch_hi - ch_lo, 1e-9)
    range_pos = (c - ch_lo) / rng

    rets = close.pct_change().dropna().to_numpy()
    ann = np.sqrt(252.0)
    rvol_s = float(np.std(rets[-RVOL_SHORT:]) * ann) if len(rets) >= RVOL_SHORT else float("nan")
    rvol_l = float(np.std(rets[-RVOL_LONG:]) * ann) if len(rets) >= RVOL_LONG else float("nan")
    rvol_ratio = rvol_s / rvol_l if rvol_l and not np.isnan(rvol_l) else float("nan")

    gaps = (open_.to_numpy()[1:] / close.to_numpy()[:-1] - 1.0)
    recent = gaps[-GAP_LOOKBACK:]
    if len(recent):
        j = int(np.argmax(np.abs(recent)))
        last_gap = float(recent[j])
        gap_age = len(recent) - 1 - j
    else:
        last_gap, gap_age = 0.0, -1

    sma = float(close.iloc[-MA_WIN:].mean())
    stretch = (c - sma) / sma if sma else 0.0

    return SetupFeatures(
        ticker=ticker, asof=str(asof), context_to=str(df.index[-1].date()),
        close=c, trend_slope=slope, trend_r2=r2,
        dist_from_high=(c - ch_hi) / c, dist_from_low=(c - ch_lo) / c,
        range_pos=float(range_pos),
        rvol_short=rvol_s, rvol_long=rvol_l, rvol_ratio=float(rvol_ratio),
        last_gap=last_gap, gap_age=gap_age, stretch_from_ma=float(stretch),
    )


def classify(f: SetupFeatures) -> str:
    """Priority ladder → exactly one label from config.SETUPS."""
    up = f.trend_slope > TREND_SLOPE_MIN
    down = f.trend_slope < -TREND_SLOPE_MIN
    clean = f.trend_r2 >= TREND_R2_MIN

    if not np.isnan(f.rvol_ratio) and f.rvol_ratio >= VOL_EXPANSION_RATIO:
        return "vol-expansion"
    if abs(f.last_gap) >= GAP_MIN and f.gap_age >= 0:
        return "post-gap-drift"
    if f.range_pos >= BREAKOUT_RANGE_POS and not down:
        return "momentum-breakout"
    if f.range_pos <= BREAKDOWN_RANGE_POS and not up:
        return "momentum-breakdown"
    if abs(f.stretch_from_ma) >= STRETCH_MIN and not clean:
        return "mean-reversion-candidate"
    if (up or down) and clean:
        return "trend-continuation"
    if abs(f.trend_slope) < TREND_SLOPE_MIN and 0.2 <= f.range_pos <= 0.8:
        return "range-bound"
    return "quiet"


@dataclass
class Setup:
    features: SetupFeatures
    label: str

    def as_row(self) -> dict:
        return {**self.features.as_row(), "setup": self.label}


def describe_setup(ticker: str, asof: str) -> Setup:
    """Load the no-lookahead window for (ticker, asof) and label it."""
    pl = load_context(ticker, asof)
    f = setup_features(pl.df, ticker=ticker, asof=asof)
    return Setup(features=f, label=classify(f))


assert set(SETUPS)  # keep the import meaningful; classify() must return one of these


@dataclass
class AnalogResult:
    ticker: str
    setup: str
    asof: str
    n: int
    base_rate_up: float      # fraction of analogs whose next-day close-to-close was > 0
    mean_ret: float          # mean next-day return across analogs
    mean_abs_ret: float
    dates: list[str]
    sufficient: bool         # n >= ANALOG_MIN

    def line(self) -> str:
        if not self.sufficient:
            return f"{self.n} prior {self.setup} instances — too few to quote a base rate"
        return (f"{self.n} prior {self.setup} instances: next-day up {self.base_rate_up:.0%}, "
                f"mean {self.mean_ret*100:+.2f}% (|move| {self.mean_abs_ret*100:.2f}%)")


ANALOG_MIN_GAP = 5     # bars to skip after a hit, so analogs are ~independent


def find_analogs(ticker: str, setup: str, asof: str, *,
                 lookback: int = ANALOG_LOOKBACK, min_gap: int = ANALOG_MIN_GAP) -> AnalogResult:
    """Past dates where `ticker` *entered* `setup` (strictly before `asof`) and the
    realised next-day close-to-close move that followed each.

    Deterministic: walks the name's own price history, re-labelling each bar with
    the same `classify()` ladder triage uses today. After each hit it skips
    `min_gap` bars so a multi-week stretch in one setup counts once, not daily.
    """
    from .data import _coerce_date, _read_cache

    asof_ts = pd.Timestamp(_coerce_date(asof))
    hist = _read_cache(ticker)
    if hist is None:
        return AnalogResult(ticker, setup, str(asof), 0, float("nan"),
                            float("nan"), float("nan"), [], False)
    hist = hist.drop(columns=[c for c in hist.columns if c.startswith("_")])
    hist = hist[hist.index < asof_ts].iloc[-(lookback + CHANNEL_WIN + 1):]
    if len(hist) < CHANNEL_WIN + TREND_WIN + 5:
        return AnalogResult(ticker, setup, str(asof), 0, float("nan"),
                            float("nan"), float("nan"), [], False)

    closes = hist["close"].astype(float)
    hits, rets = [], []
    # need one bar after each candidate to score the forward move
    i = CHANNEL_WIN + 1
    while i < len(hist) - 1:
        try:
            f = setup_features(hist.iloc[:i], ticker=ticker)
        except ValueError:
            i += 1
            continue
        if classify(f) != setup:
            i += 1
            continue
        fwd = closes.iloc[i] / closes.iloc[i - 1] - 1.0
        hits.append(str(hist.index[i - 1].date()))
        rets.append(float(fwd))
        i += min_gap

    n = len(rets)
    if n == 0:
        return AnalogResult(ticker, setup, str(asof), 0, float("nan"),
                            float("nan"), float("nan"), [], False)
    arr = np.array(rets)
    return AnalogResult(
        ticker, setup, str(asof), n,
        float(np.mean(arr > 0)), float(arr.mean()), float(np.abs(arr).mean()),
        hits[-12:], n >= ANALOG_MIN,
    )


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Label a ticker's setup on a date.")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--tickers", nargs="+", required=True)
    args = ap.parse_args(argv)
    for t in args.tickers:
        s = describe_setup(t, args.asof)
        ff = s.features
        print(f"{t:6} {s.label:24}  slope {ff.trend_slope:+.4f} r2 {ff.trend_r2:.2f}  "
              f"rangepos {ff.range_pos:.2f}  rvolx {ff.rvol_ratio:.2f}  "
              f"gap {ff.last_gap:+.3f}@{ff.gap_age}  stretch {ff.stretch_from_ma:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
