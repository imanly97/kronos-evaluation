"""Immutable signal log + queryable mirror.

The source of truth is an append-only JSONL file (`log/signals.jsonl`). Each line
carries a `row_hash` chained off the previous line's hash, so any later edit to a
historical row breaks the chain and is detectable. Git history of that file is the
external tamper-evidence (see PLAN §8, §10).

`narb.db` is a convenience SQLite mirror, rebuilt from the JSONL on demand — never
authoritative, always disposable.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DB_PATH, LOG_DIR

SIGNALS_JSONL = LOG_DIR / "signals.jsonl"
ACTUALS_JSONL = LOG_DIR / "actuals.jsonl"

SIGNAL_KEYS = [
    "asof", "ticker", "gen_ts", "llm_contaminated",
    "price_source", "context_from", "context_to", "prev_close",
    "kronos_q05", "kronos_q25", "kronos_q50", "kronos_q75", "kronos_q95",
    "kronos_p_up", "kronos_std", "kronos_strength", "kronos_dir",
    "narr_dir", "narr_conviction", "narr_rationale", "narr_headline_count",
    "narr_signal", "divergence_type", "divergence_verdict", "brief",
]
ACTUAL_KEYS = [
    "asof", "ticker", "eval_ts", "price_source",
    "prev_close", "actual_close", "ret",
    "inside_envelope", "dir_match_struct", "dir_match_narr", "divergence_outcome",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def _hash(prev_hash: str, payload: str) -> str:
    return hashlib.sha256((prev_hash + payload).encode()).hexdigest()


def _last_hash(path: Path) -> str:
    if not path.exists():
        return "genesis"
    last = None
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return "genesis"
    return json.loads(last)["row_hash"]


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    out = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.add((r["asof"], r["ticker"]))
    return out


class ImmutableViolation(RuntimeError):
    """Raised on an attempt to re-write a signal that is already locked."""


def append_signal(record: dict, *, allow_replace: bool = False) -> dict:
    """Lock one (asof, ticker) signal. Refuses to overwrite an existing one."""
    key = (str(record["asof"]), record["ticker"])
    if not allow_replace and key in _existing_keys(SIGNALS_JSONL):
        raise ImmutableViolation(f"signal for {key} already locked; not overwriting")

    row = {k: record.get(k) for k in SIGNAL_KEYS}
    row.setdefault("gen_ts", _utc_now())
    prev = _last_hash(SIGNALS_JSONL)
    payload = _canonical(row)
    row_full = {**row, "prev_hash": prev, "row_hash": _hash(prev, payload)}
    with SIGNALS_JSONL.open("a") as fh:
        fh.write(json.dumps(row_full, default=str) + "\n")
    return row_full


def append_actual(record: dict, *, allow_replace: bool = False) -> dict:
    key = (str(record["asof"]), record["ticker"])
    if not allow_replace and key in _existing_keys(ACTUALS_JSONL):
        raise ImmutableViolation(f"actual for {key} already recorded")
    row = {k: record.get(k) for k in ACTUAL_KEYS}
    row.setdefault("eval_ts", _utc_now())
    prev = _last_hash(ACTUALS_JSONL)
    payload = _canonical(row)
    row_full = {**row, "prev_hash": prev, "row_hash": _hash(prev, payload)}
    with ACTUALS_JSONL.open("a") as fh:
        fh.write(json.dumps(row_full, default=str) + "\n")
    return row_full


def verify_chain(path: Path) -> bool:
    """True iff every row_hash matches the recomputed chain."""
    if not path.exists():
        return True
    prev = "genesis"
    keys = SIGNAL_KEYS if path == SIGNALS_JSONL else ACTUAL_KEYS
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            payload = _canonical({k: r.get(k) for k in keys})
            if r["prev_hash"] != prev or r["row_hash"] != _hash(prev, payload):
                raise ImmutableViolation(f"{path.name}: chain broken at line {i + 1}")
            prev = r["row_hash"]
    return True


# --------------------------------------------------------------------------- #
# frames + SQLite mirror
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_json(path, lines=True)
    # The JSONL keeps every physical line (the hash chain covers all of them);
    # a `--replace` re-lock appends a superseding row, so the queryable view
    # takes the last row per (asof, ticker).
    if {"asof", "ticker"}.issubset(df.columns):
        df = (df.drop_duplicates(subset=["asof", "ticker"], keep="last")
                .reset_index(drop=True))
    return df


def load_signals() -> pd.DataFrame:
    return _read_jsonl(SIGNALS_JSONL)


def load_actuals() -> pd.DataFrame:
    return _read_jsonl(ACTUALS_JSONL)


def joined() -> pd.DataFrame:
    s, a = load_signals(), load_actuals()
    if s.empty:
        return s
    if a.empty:
        return s
    drop = [c for c in ("prev_close", "price_source", "prev_hash", "row_hash", "eval_ts")
            if c in a.columns]
    return s.merge(a.drop(columns=drop), on=["asof", "ticker"], how="left",
                   suffixes=("", "_actual"))


def sync_db(db_path: Path = DB_PATH) -> Path:
    """Rebuild narb.db from the JSONL logs."""
    verify_chain(SIGNALS_JSONL)
    verify_chain(ACTUALS_JSONL)
    con = sqlite3.connect(db_path)
    try:
        s, a = load_signals(), load_actuals()
        if not s.empty:
            s.to_sql("signals", con, if_exists="replace", index=False)
        if not a.empty:
            a.to_sql("actuals", con, if_exists="replace", index=False)
    finally:
        con.close()
    return db_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Signal log utilities.")
    ap.add_argument("cmd", choices=["verify", "sync", "show"])
    args = ap.parse_args()
    if args.cmd == "verify":
        verify_chain(SIGNALS_JSONL)
        verify_chain(ACTUALS_JSONL)
        print("chain OK")
    elif args.cmd == "sync":
        print("wrote", sync_db())
    elif args.cmd == "show":
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(joined())
