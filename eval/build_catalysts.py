"""Build eval/catalysts.json — the curated scheduled-event calendar the
post-mortem agent checks against (PLAN §5.3, §7 catalyst-recall test).

Earnings dates: the Nasdaq earnings calendar (date-indexed, reliable), walked day
by day across the replay window and filtered to our universe.
Macro dates: FOMC decision days, CPI prints and BLS jobs reports, hand-entered
from the public 2025 schedules.

The file is committed so the eval is reproducible without a live call.

    .venv/bin/python -m eval.build_catalysts
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta

from curl_cffi import requests as cr

from src.config import REPLAY_END, REPLAY_START, ROOT, UNIVERSE

OUT = ROOT / "eval" / "catalysts.json"
UNIV = set(UNIVERSE)

# known reporters the Nasdaq feed dropped on the day (verified against company IR)
CURATED_EXTRA = {
    "AAPL": ["2025-05-01", "2025-07-31"],
}

MACRO = [
    {"date": "2025-05-02", "kind": "jobs"},
    {"date": "2025-05-07", "kind": "fomc"},
    {"date": "2025-05-13", "kind": "cpi"},
    {"date": "2025-06-06", "kind": "jobs"},
    {"date": "2025-06-11", "kind": "cpi"},
    {"date": "2025-06-18", "kind": "fomc"},
    {"date": "2025-07-03", "kind": "jobs"},
    {"date": "2025-07-15", "kind": "cpi"},
    {"date": "2025-07-30", "kind": "fomc"},
]


def _nasdaq_earnings(day: date) -> list[str]:
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={day.isoformat()}"
    for attempt in range(3):
        try:
            r = cr.get(url, impersonate="chrome124",
                       headers={"Accept": "application/json"}, timeout=25)
            r.raise_for_status()
            rows = (r.json().get("data") or {}).get("rows") or []
            return [x["symbol"] for x in rows if x.get("symbol") in UNIV]
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"  {day}: failed ({exc})")
                return []
            time.sleep(2)
    return []


def main() -> None:
    lo = date.fromisoformat(REPLAY_START)
    hi = date.fromisoformat(REPLAY_END) + timedelta(days=4)

    earnings: dict[str, list[str]] = {}
    d = lo
    while d <= hi:
        if d.weekday() < 5:
            for sym in _nasdaq_earnings(d):
                earnings.setdefault(sym, []).append(d.isoformat())
            time.sleep(0.4)
        d += timedelta(days=1)

    for sym, dts in CURATED_EXTRA.items():
        earnings.setdefault(sym, [])
        earnings[sym].extend(d for d in dts if d not in earnings[sym])
    earnings = {k: sorted(v) for k, v in sorted(earnings.items())}
    OUT.write_text(json.dumps(
        {"window": [REPLAY_START, REPLAY_END], "source": "nasdaq calendar + curated macro",
         "earnings": earnings, "macro": MACRO},
        indent=2, sort_keys=True,
    ))
    print(f"\n{len(earnings)} universe names with an in-window print:")
    for k, v in earnings.items():
        print(f"  {k:6} {v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
