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

# --- universe (LOCKED 2026-08-31) --------------------------------------
# 30 US large-caps. Screen (research/universe_screen.py): trailing-year median
# dollar volume >= $300M/day AND worst day >= $50M, continuous history from 2018,
# then hand-balanced for sector spread and volatility dispersion.
UNIVERSE: list[str] = [
    # tech / semis / software
    "AAPL", "MSFT", "NVDA", "AMD", "MU", "CRM",
    # communication services
    "GOOGL", "META", "NFLX",
    # consumer discretionary
    "AMZN", "TSLA", "HD", "MCD",
    # consumer staples
    "WMT", "COST", "PG",
    # financials
    "JPM", "BAC", "GS", "V",
    # health care
    "UNH", "LLY", "JNJ", "MRK",
    # energy
    "XOM", "CVX",
    # industrials
    "CAT", "BA",
    # materials / utilities
    "LIN", "NEE",
]
DEV_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMD", "JPM", "XOM", "NFLX", "TSLA"]  # probe set

SECTOR: dict[str, str] = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AMD": "Tech", "MU": "Tech", "CRM": "Tech",
    "GOOGL": "Comm", "META": "Comm", "NFLX": "Comm",
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "MCD": "ConsDisc",
    "WMT": "ConsStap", "COST": "ConsStap", "PG": "ConsStap",
    "JPM": "Fin", "BAC": "Fin", "GS": "Fin", "V": "Fin",
    "UNH": "Health", "LLY": "Health", "JNJ": "Health", "MRK": "Health",
    "XOM": "Energy", "CVX": "Energy",
    "CAT": "Indu", "BA": "Indu",
    "LIN": "Materials", "NEE": "Utilities",
}

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

# context bars fed to the sampler (< MAX_CONTEXT).
# 250 chosen for compute: MPS falls off a memory cliff above ~250 and with any
# batch dim (profiled 2026-08). L=400 sensitivity check runs on a name subset.
LOOKBACK_DAILY = 250
LOOKBACK_HOURLY = 250

SAMPLE_COUNT = 50                                 # sampled paths per forecast
SAMPLE_COUNT_REF = 200                            # for the S-sensitivity check
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
