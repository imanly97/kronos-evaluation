"""Single source of truth: paths, universe, splits, model params.

Everything downstream imports from here (PLAN §4, §5, §9).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))     # weights stay in-project
KRONOS_SRC = ROOT / "vendor_kronos"

CACHE_DIR = ROOT / "cache"
DAILY_CACHE = CACHE_DIR / "daily"
HOURLY_CACHE = CACHE_DIR / "hourly"
FORECAST_STORE = ROOT / "store"          # immutable forecast + metric outputs
NB_OUT = ROOT / "notebooks" / "_out"
for _d in (DAILY_CACHE, HOURLY_CACHE, FORECAST_STORE, NB_OUT):
    _d.mkdir(parents=True, exist_ok=True)

# --- universe (LOCKED once chosen in W1) ---------------------------------
# ~30 US large-caps: top market cap, options-liquid, >=5y history, sector-spread.
UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "AMD", "NFLX",
    "ORCL", "CRM", "ADBE", "CSCO", "QCOM", "TXN", "INTC",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    "UNH", "JNJ", "LLY", "MRK", "XOM", "CVX", "CAT",
]
DEV_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMD", "JPM", "XOM", "NFLX", "TSLA"]  # probe set

# --- splits (PLAN §5) ---------------------------------------------------
# Kronos-small pretraining ends ~2024-06-30 (arXiv:2508.02739).
CONTAMINATED_END = "2024-06-30"
DEV_START, DEV_END = "2024-07-01", "2025-12-31"
LOCKBOX_START, LOCKBOX_END = "2026-01-01", "2026-08-31"

DATA_START = "2018-01-01"        # daily history floor
HOURLY_START = "2023-10-01"      # Yahoo's ~730d intraday limit

# --- Kronos -----------------------------------------------------------
KRONOS_MODEL = os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-small")
KRONOS_TOKENIZER = os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
KRONOS_DEVICE = os.getenv("KRONOS_DEVICE")        # None -> auto (mps / cpu)
MAX_CONTEXT = 512                                 # Kronos-small / base window

# context bars fed to the sampler (< MAX_CONTEXT)
LOOKBACK_DAILY = 250
LOOKBACK_HOURLY = 400

SAMPLE_COUNT = 200                                # sampled paths per forecast
T = 1.0
TOP_P = 0.9
TOP_K = 0
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# --- forecast targets (PLAN §6) --------------------------------------
HORIZONS_DAILY = [1, 5]                           # sessions
HORIZONS_HOURLY = [1, 2]                          # sessions (~7, ~14 bars)
BARS_PER_SESSION = 7                              # Yahoo regular-session hourly bars

# --- back-compat shims for src/data.py (pending W1 hourly rework) --------
CACHE_DIR = DAILY_CACHE
LOOKBACK_BARS = LOOKBACK_DAILY
