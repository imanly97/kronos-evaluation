"""Desk memory — append-only episodes + a derived per-(ticker, setup) stat table.

`memory/episodes.jsonl`  one line per graded, classified watchlist name (PLAN §5.4),
                         hash-chained like the `log/` tables.
`memory/stats.json`      derived, rebuildable: {"<ticker>|<setup>": {n, hit_rate,
                         mean_signed_err, mean_abs_err}}.

`recall(ticker, setup)` is deterministic key-based lookup — this is the edge that
closes the loop (triage reads it back the next morning).

The post-mortem agent is what *writes* episodes; see `postmortem.py`. Until that
lands, `recall` simply reports an empty track record, which triage handles.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import MEMORY_DIR

EPISODES = MEMORY_DIR / "episodes.jsonl"
STATS = MEMORY_DIR / "stats.json"

EPISODE_KEYS = [
    "asof", "ticker", "setup", "forecast_median", "actual",
    "actual_quantile", "classification", "note",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)


def _hash(prev: str, payload: str) -> str:
    return hashlib.sha256((prev + payload).encode()).hexdigest()


def _last_hash() -> str:
    if not EPISODES.exists():
        return "genesis"
    last = None
    for line in EPISODES.read_text().splitlines():
        if line.strip():
            last = line
    return "genesis" if last is None else json.loads(last)["row_hash"]


def append_episode(rec: dict) -> dict:
    row = {k: rec.get(k) for k in EPISODE_KEYS}
    row["asof"] = str(row["asof"])
    row["logged_ts"] = _utc_now()
    prev = _last_hash()
    row_full = {**row, "prev_hash": prev, "row_hash": _hash(prev, _canonical(row))}
    with EPISODES.open("a") as fh:
        fh.write(json.dumps(row_full, default=str) + "\n")
    return row_full


def _load_episodes() -> list[dict]:
    if not EPISODES.exists() or EPISODES.stat().st_size == 0:
        return []
    return [json.loads(l) for l in EPISODES.read_text().splitlines() if l.strip()]


def verify_chain() -> bool:
    prev = "genesis"
    for i, line in enumerate(l for l in
                             (EPISODES.read_text().splitlines() if EPISODES.exists() else [])
                             if l.strip()):
        r = json.loads(line)
        payload = _canonical({k: r.get(k) for k in EPISODE_KEYS} | {"logged_ts": r.get("logged_ts")})
        if r["prev_hash"] != prev or r["row_hash"] != _hash(prev, payload):
            raise RuntimeError(f"episodes.jsonl: chain broken at line {i + 1}")
        prev = r["row_hash"]
    return True


def rebuild_stats() -> dict:
    """Derive per-(ticker, setup) rolling stats from the episode log."""
    agg: dict[str, dict] = {}
    for e in _load_episodes():
        key = f"{e['ticker']}|{e['setup']}"
        a = agg.setdefault(key, {"n": 0, "_hits": 0, "_signed": 0.0, "_abs": 0.0})
        a["n"] += 1
        fm, act = e.get("forecast_median"), e.get("actual")
        if fm is not None and act is not None:
            err = act - fm
            a["_signed"] += err
            a["_abs"] += abs(err)
            if (fm > 0) == (act > 0) and act != 0:
                a["_hits"] += 1
    out = {}
    for key, a in agg.items():
        n = a["n"]
        out[key] = {
            "n": n,
            "hit_rate": round(a["_hits"] / n, 3) if n else None,
            "mean_signed_err": round(a["_signed"] / n, 5) if n else None,
            "mean_abs_err": round(a["_abs"] / n, 5) if n else None,
        }
    STATS.write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def _stats() -> dict:
    if STATS.exists():
        return json.loads(STATS.read_text())
    return {}


@dataclass
class Recall:
    ticker: str
    setup: str
    n: int
    hit_rate: float | None
    mean_signed_err: float | None
    mean_abs_err: float | None
    episodes: list[dict] = field(default_factory=list)

    def line(self) -> str:
        if self.n == 0:
            return f"no prior desk episodes for {self.ticker} in a {self.setup} setup"
        hr = f"{self.hit_rate:.0%}" if self.hit_rate is not None else "n/a"
        return (f"desk has {self.n} prior {self.setup} episode(s) for {self.ticker}: "
                f"direction hit {hr}, mean signed error {self.mean_signed_err:+.2%}")


def recall(ticker: str, setup: str, *, k: int = 5) -> Recall:
    key = f"{ticker}|{setup}"
    s = _stats().get(key, {})
    eps = [e for e in _load_episodes() if e["ticker"] == ticker and e["setup"] == setup]
    eps.sort(key=lambda e: str(e["asof"]))
    return Recall(
        ticker=ticker, setup=setup,
        n=s.get("n", 0), hit_rate=s.get("hit_rate"),
        mean_signed_err=s.get("mean_signed_err"), mean_abs_err=s.get("mean_abs_err"),
        episodes=eps[-k:],
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Desk memory utilities.")
    ap.add_argument("cmd", choices=["rebuild", "verify", "recall"])
    ap.add_argument("--ticker")
    ap.add_argument("--setup")
    args = ap.parse_args()
    if args.cmd == "rebuild":
        print(json.dumps(rebuild_stats(), indent=2))
    elif args.cmd == "verify":
        verify_chain()
        print("episodes chain OK")
    else:
        print(recall(args.ticker, args.setup).line())
