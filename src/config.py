"""Single source of truth for paths, the ticker universe, and model params.

Everything downstream imports from here so the pilot's knobs live in one file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# keep HF model weights inside the project, out of git
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
# vendored Kronos source (cloned by scripts/setup.sh)
KRONOS_SRC = ROOT / "vendor_kronos"

CACHE_DIR = ROOT / "cache" / "prices"
# LOG_DIR is overridable so the walkthrough notebook can use a scratch log
# instead of touching the real immutable one.
LOG_DIR = Path(os.getenv("NARB_LOG_DIR", ROOT / "log"))
DB_PATH = LOG_DIR / "narb.db" if os.getenv("NARB_LOG_DIR") else ROOT / "narb.db"
CASE_DIR = ROOT / "case_studies"
NB_OUT = ROOT / "notebooks" / "_out"

for _d in (CACHE_DIR, LOG_DIR, NB_OUT):
    _d.mkdir(parents=True, exist_ok=True)

# --- universe (LOCKED 2026-08-29) ------------------------------------------
# Mega-cap, deeply liquid, news-heavy, clean daily bars. 12 names.
UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "TSLA", "AMD", "NFLX", "AVGO", "JPM", "XOM",
]

# --- Kronos ---------------------------------------------------------------
KRONOS_MODEL = os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-small")
KRONOS_TOKENIZER = os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
KRONOS_DEVICE = os.getenv("KRONOS_DEVICE")  # None -> auto-detect
MAX_CONTEXT = 512          # Kronos-small / base context window
# Daily bars fed as context. NOT the full 512: empirically, long daily lookbacks
# destabilise the autoregressive sampler for names trading near the high/low of
# the window — the normalised last price reads as a multi-sigma extreme and the
# model reverts hard (medians of -3 to -8% and -30%+ tails). 120 bars (~6 months)
# keeps normalisation local and the sampler well-behaved. See PLAN §16.
LOOKBACK_BARS = 120
PRED_LEN = 3               # forecast horizon in bars; we grade bar 1
GRADE_BAR = 1             # 1-indexed: the bar dated `asof`
SAMPLE_COUNT = 200         # sampled paths per ticker per night
T = 1.0                    # sampling temperature
TOP_P = 0.9
TOP_K = 0
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# --- LLM ----------------------------------------------------------------
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

# --- divergence thresholds (v1, LOCKED 2026-08-29) --------------------------
# See PLAN.md §6. Directional: narrative and structure disagree on sign, with
# at least mild narrative conviction. Magnitude: narrative is loud but the move
# it implies sits in a low-probability region of Kronos's own distribution.
DIV_DIRECTIONAL_MIN_CONVICTION = 4
DIV_MAGNITUDE_MIN_CONVICTION = 7
DIV_MAGNITUDE_MAX_PROB = 0.30      # P(return in narrative's direction) under Kronos

# --- knowledge-cutoff honesty flag ----------------------------------------
# Case studies dated on/before this are "contaminated": the LLM may have the
# realised outcome in its training data, so they are illustrative, not evidence.
# Live signals dated after it are the actual test. See PLAN.md §5a.
LLM_KNOWLEDGE_CUTOFF = "2026-01-31"
