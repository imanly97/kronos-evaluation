"""Immutable desk log + queryable mirror.

Three append-only JSONL tables live in `log/`:

  forecasts.jsonl   one row per (asof, ticker)   — the deterministic Kronos output
  watchlist.jsonl   one row per (asof, ticker)   — what triage flagged + the brief
  actuals.jsonl     one row per (asof, ticker)   — next-day grade of a forecast

Each line carries a `row_hash` chained off the previous line's hash, so any later
edit to a historical row breaks the chain and is detectable. Git history of these
files is the external tamper-evidence (PLAN §8, §11).

`desk.db` is a convenience SQLite mirror, rebuilt from the JSONL on demand — never
authoritative, always disposable.

Desk *memory* (episodes + derived stats) is a separate store — see `memory.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DB_PATH, LOG_DIR

# --------------------------------------------------------------------------- #
# schema — column order is the canonical-hash order, do not reorder in place
# --------------------------------------------------------------------------- #
SCHEMA: dict[str, list[str]] = {
    "forecasts": [
        "asof", "ticker", "gen_ts", "price_source", "context_to", "prev_close",
        "q05", "q25", "q50", "q75", "q95",
        "p_up", "std", "strength", "strength_z",
    ],
    "watchlist": [
        "asof", "ticker", "run_ts", "rank", "setup", "reason",
        "analog_base_rate", "analog_n", "cluster_read", "memory_line",
        "caveat", "brief_md", "groundedness", "critic_rounds", "critic_ok",
    ],
    "actuals": [
        "asof", "ticker", "eval_ts", "price_source",
        "prev_close", "actual_close", "ret",
        "actual_quantile", "inside_envelope", "dir_match",
    ],
}
KEY = ("asof", "ticker")
_TS_FIELD = {"forecasts": "gen_ts", "watchlist": "run_ts", "actuals": "eval_ts"}


class ImmutableViolation(RuntimeError):
    """Raised on an attempt to overwrite a row that is already locked."""


def _path(table: str) -> Path:
    if table not in SCHEMA:
        raise KeyError(f"unknown table {table!r} (have {list(SCHEMA)})")
    return LOG_DIR / f"{table}.jsonl"


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
    return "genesis" if last is None else json.loads(last)["row_hash"]


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
            out.add((str(r["asof"]), r["ticker"]))
    return out


# --------------------------------------------------------------------------- #
# append + verify
# --------------------------------------------------------------------------- #
def append(table: str, record: dict, *, allow_replace: bool = False) -> dict:
    """Lock one (asof, ticker) row into `table`. Refuses to overwrite by default.

    A `--replace` re-lock appends a *superseding* physical line (the hash chain
    still covers it); the queryable views take the last row per key.
    """
    path = _path(table)
    cols = SCHEMA[table]
    key = (str(record["asof"]), record["ticker"])
    if not allow_replace and key in _existing_keys(path):
        raise ImmutableViolation(f"{table} row for {key} already locked; not overwriting")

    row = {k: record.get(k) for k in cols}
    row[KEY[0]] = str(row[KEY[0]])
    row.setdefault(_TS_FIELD[table], _utc_now())
    if row.get(_TS_FIELD[table]) is None:
        row[_TS_FIELD[table]] = _utc_now()

    prev = _last_hash(path)
    row_full = {**row, "prev_hash": prev, "row_hash": _hash(prev, _canonical(row))}
    with path.open("a") as fh:
        fh.write(json.dumps(row_full, default=str) + "\n")
    return row_full


def verify_chain(table: str) -> bool:
    """True iff every row_hash matches the recomputed chain. Raises on a break."""
    path = _path(table)
    if not path.exists():
        return True
    cols = SCHEMA[table]
    prev = "genesis"
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            payload = _canonical({k: r.get(k) for k in cols})
            if r["prev_hash"] != prev or r["row_hash"] != _hash(prev, payload):
                raise ImmutableViolation(f"{path.name}: chain broken at line {i + 1}")
            prev = r["row_hash"]
    return True


def verify_all() -> bool:
    for t in SCHEMA:
        verify_chain(t)
    return True


# --------------------------------------------------------------------------- #
# frames + SQLite mirror
# --------------------------------------------------------------------------- #
def load(table: str) -> pd.DataFrame:
    """Queryable view of `table`: last physical row per (asof, ticker)."""
    path = _path(table)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=SCHEMA[table])
    df = pd.read_json(path, lines=True, dtype={"asof": str})
    df = (df.drop_duplicates(subset=list(KEY), keep="last").reset_index(drop=True))
    return df


def joined() -> pd.DataFrame:
    """forecasts ⟕ actuals ⟕ watchlist on (asof, ticker)."""
    f = load("forecasts")
    if f.empty:
        return f
    a = load("actuals")
    if not a.empty:
        drop = [c for c in ("gen_ts", "run_ts", "eval_ts", "price_source",
                            "prev_close", "prev_hash", "row_hash")
                if c in a.columns]
        f = f.merge(a.drop(columns=drop), on=list(KEY), how="left", suffixes=("", "_act"))
    w = load("watchlist")
    if not w.empty:
        keep = list(KEY) + ["rank", "setup", "reason", "caveat"]
        f = f.merge(w[[c for c in keep if c in w.columns]], on=list(KEY), how="left")
        f["flagged"] = f["rank"].notna()
    else:
        f["flagged"] = False
    return f


def sync_db(db_path: Path = DB_PATH) -> Path:
    """Rebuild desk.db from the JSONL logs (verifies every chain first)."""
    verify_all()
    con = sqlite3.connect(db_path)
    try:
        for t in SCHEMA:
            df = load(t)
            if not df.empty:
                df.to_sql(t, con, if_exists="replace", index=False)
    finally:
        con.close()
    return db_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Desk log utilities.")
    ap.add_argument("cmd", choices=["verify", "sync", "show"])
    ap.add_argument("--table", choices=list(SCHEMA), default="forecasts")
    args = ap.parse_args(argv)

    if args.cmd == "verify":
        verify_all()
        print("all chains OK")
    elif args.cmd == "sync":
        print("wrote", sync_db())
    elif args.cmd == "show":
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(load(args.table))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
