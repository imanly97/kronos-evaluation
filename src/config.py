"""Single source of truth for paths, the universe, and model/loop params.

Everything downstream imports from here so the knobs live in one file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))   # weights stay in-project
KRONOS_SRC = ROOT / "vendor_kronos"

CACHE_DIR = ROOT / "cache" / "prices"
# LOG_DIR / MEMORY_DIR are overridable so notebooks can use a scratch location.
LOG_DIR = Path(os.getenv("FMD_LOG_DIR", ROOT / "log"))
MEMORY_DIR = Path(os.getenv("FMD_MEMORY_DIR", ROOT / "memory"))
DB_PATH = LOG_DIR / "desk.db" if os.getenv("FMD_LOG_DIR") else ROOT / "desk.db"
NB_OUT = ROOT / "notebooks" / "_out"

for _d in (CACHE_DIR, LOG_DIR, MEMORY_DIR, NB_OUT):
    _d.mkdir(parents=True, exist_ok=True)

# --- universe (LOCKED 2026-08-29) ----------------------------------------
# ~40 deeply-liquid US large-caps across sectors, so triage is a real filter.
UNIVERSE: list[str] = [
    # tech / comm
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL", "CRM",
    "AMD", "NFLX", "ADBE", "INTC", "QCOM", "CSCO", "TXN",
    # consumer
    "TSLA", "HD", "MCD", "NKE", "COST", "WMT", "DIS", "SBUX",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    # health
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE",
    # energy / industrial
    "XOM", "CVX", "CAT",
]
DEV_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMD", "NFLX", "JPM", "XOM", "TSLA"]

# --- replay window (LOCKED) --------------------------------------------
REPLAY_START = "2025-05-01"
REPLAY_END = "2025-08-01"

# --- Kronos --------------------------------------------------------------
KRONOS_MODEL = os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-small")
KRONOS_TOKENIZER = os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
KRONOS_DEVICE = os.getenv("KRONOS_DEVICE")            # None -> auto (mps/cpu)
MAX_CONTEXT = 512
# v1 finding: long daily lookbacks destabilise the AR sampler for names near the
# high/low of the window (median -3 to -8%, -30% tails). 120 bars keeps the
# normalisation local. See PLAN §3 D2.
LOOKBACK_BARS = 120
PRED_LEN = 3
GRADE_BAR = 1                # 1-indexed; the bar dated `asof`
SAMPLE_COUNT = 200
T = 1.0
TOP_P = 0.9
TOP_K = 0
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# --- features ---------------------------------------------------------
STRENGTH_Z_WINDOW = 252      # trailing bars for "is today unusual for THIS name"
CLUSTER_CORR_WINDOW = 63     # bars for pairwise return correlation
CLUSTER_CORR_MIN = 0.55      # |corr| above this = "correlated" for the cluster read
ANALOG_LOOKBACK = 504        # bars searched for past instances of a setup
ANALOG_MIN = 4               # need at least this many analogs to quote a base rate

# --- setup vocabulary (LOCKED, PLAN §3 D10) --------------------------------
SETUPS = [
    "momentum-breakout",       # near/through an N-day high, strong recent trend
    "momentum-breakdown",      # near/through an N-day low, strong down trend
    "trend-continuation",      # established trend, not at an extreme
    "range-bound",             # low trend slope, mid-range
    "post-gap-drift",          # a gap in the last few bars, still digesting it
    "vol-expansion",           # realised vol well above its own trailing level
    "mean-reversion-candidate",# stretched from a moving average, trend fading
    "quiet",                   # nothing notable
]

# --- triage -----------------------------------------------------------
TRIAGE_MIN = 3
TRIAGE_MAX = 7
TRIAGE_MAX_CANDIDATES = 15    # agent investigates at most this many before deciding
CRITIC_MAX_ROUNDS = 3

# --- post-mortem ----------------------------------------------------
POSTMORTEM_TAIL_Q = 0.10      # |actual_quantile - 0.5| beyond this = a "surprise" to explain

# --- LLM ------------------------------------------------------------
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
