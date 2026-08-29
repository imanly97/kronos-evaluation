"""Scheduler entrypoint.

Two modes, matching the two launchd / GitHub Actions jobs:

  python -m src.run_nightly signal      # ~08:45 ET — generate + lock today's signals
  python -m src.run_nightly evaluate    # ~09:45 ET next day — grade yesterday's

`signal` also opportunistically grades any still-pending prior signals, so a
missed evaluate run is self-healing.
"""
from __future__ import annotations

import sys
from datetime import date

from . import evaluate, graph
from .store import verify_chain
from .store import SIGNALS_JSONL, ACTUALS_JSONL


def _run_signal() -> int:
    verify_chain(SIGNALS_JSONL)
    asof = str(date.today())
    print(f"=== SIGNAL run  asof={asof} ===", flush=True)
    results = graph.run_universe(asof)
    ok = sum(1 for r in results if r.get("persisted"))
    print(f"locked {ok}/{len(results)} signals")
    print("\n".join(graph._summary_row(s) for s in results))

    # self-heal: grade anything still pending from earlier days
    pend = [p for p in evaluate.pending() if str(p["asof"]) < asof]
    if pend:
        print(f"\n=== catch-up EVALUATE ({len(pend)} pending) ===")
        evaluate.run()
    print("\n" + evaluate.report())
    return 0 if ok else 1


def _run_evaluate() -> int:
    verify_chain(ACTUALS_JSONL)
    print(f"=== EVALUATE run  {date.today()} ===", flush=True)
    res = evaluate.run()
    print(f"graded {sum(1 for s in res if s.get('persisted'))}/{len(res)}")
    print("\n" + evaluate.report())
    return 0


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    mode = argv[0] if argv else "signal"
    if mode == "signal":
        return _run_signal()
    if mode == "evaluate":
        return _run_evaluate()
    print(f"unknown mode {mode!r}; use 'signal' or 'evaluate'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
