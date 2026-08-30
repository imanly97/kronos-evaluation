"""One entry point for the loop — pre-market, post-close, or a full replay.

    # historical replay of the whole loop over the locked window
    python -m src.run_daily replay --start 2025-05-01 --end 2025-08-01

    # a single live session (two cron jobs)
    python -m src.run_daily premarket --asof 2025-05-29     # forecasts + triage
    python -m src.run_daily postclose --asof 2025-05-28     # grade + post-mortem

`premarket` is deterministic-forecast + triage agent. `postclose` grades the
forecast that has now settled and runs the post-mortem, which writes the desk
memory that the next `premarket` reads back. In `replay` the two run back to back
per session because the day has already settled.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date

from . import forecasts, grade, postmortem, triage
from .config import DEV_UNIVERSE, REPLAY_END, REPLAY_START, UNIVERSE
from .data import _coerce_date
from .store import sync_db


def _sessions(start: str, end: str) -> list[date]:
    return forecasts._sessions(_coerce_date(start), _coerce_date(end))


def premarket(asof: str, universe: list[str], *, skip_forecast: bool = False) -> dict:
    t0 = time.time()
    if not skip_forecast:
        forecasts.run_asof(asof, universe, verbose=False)
    st = triage.run(asof, universe=universe)
    print(f"  premarket {asof}: kept {st['kept']}  [{time.time() - t0:.0f}s]")
    return st


def postclose(asof: str, universe: list[str]) -> dict:
    t0 = time.time()
    graded = grade.run(asof)
    st = postmortem.run(asof, universe=universe)
    print(f"  postclose {asof}: graded {len(graded)}, "
          f"{len(st.get('done', []))} episodes  [{time.time() - t0:.0f}s]")
    return st


def replay(start: str, end: str, universe: list[str], *, skip_forecast: bool = False) -> None:
    sessions = _sessions(start, end)
    print(f"replay {sessions[0]}..{sessions[-1]} — {len(sessions)} sessions, "
          f"{len(universe)} names")
    for i, d in enumerate(sessions, 1):
        asof = d.isoformat()
        print(f"[{i}/{len(sessions)}] {asof}")
        try:
            premarket(asof, universe, skip_forecast=skip_forecast)
            postclose(asof, universe)
        except Exception as exc:  # noqa: BLE001 — one bad session shouldn't sink the replay
            print(f"  !! {asof} failed: {exc}", file=sys.stderr)
    sync_db()
    print("\n" + grade.report())


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the forecast-desk loop.")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_pre = sub.add_parser("premarket")
    p_pre.add_argument("--asof", help="session date; default = today")
    p_post = sub.add_parser("postclose")
    p_post.add_argument("--asof", help="session date; default = the last settled session")
    p_rep = sub.add_parser("replay")
    p_rep.add_argument("--start", default=REPLAY_START)
    p_rep.add_argument("--end", default=REPLAY_END)

    for p in (p_pre, p_post, p_rep):
        p.add_argument("--dev", action="store_true", help="use the 8-name dev universe")
        p.add_argument("--skip-forecast", action="store_true",
                       help="assume the forecast cache is already built")
        p.add_argument("--no-memory", action="store_true",
                       help="ablation: triage ignores desk memory (PLAN §7)")

    args = ap.parse_args(argv)
    universe = DEV_UNIVERSE if getattr(args, "dev", False) else UNIVERSE
    if getattr(args, "no_memory", False):
        triage.USE_MEMORY = False
        print("  [ablation] desk memory OFF for triage")

    if args.mode == "premarket":
        asof = args.asof or date.today().isoformat()
        premarket(asof, universe, skip_forecast=args.skip_forecast)
    elif args.mode == "postclose":
        asof = args.asof or _last_settled_session()
        postclose(asof, universe)
    else:
        replay(args.start, args.end, universe, skip_forecast=args.skip_forecast)
    return 0


def _last_settled_session(ref: str = "AAPL") -> str:
    """The most recent trading day with a settled bar in the price cache."""
    from .data import _read_cache

    df = _read_cache(ref)
    if df is None or df.empty:
        return date.today().isoformat()
    return df.index.max().date().isoformat()


if __name__ == "__main__":
    raise SystemExit(_main())
