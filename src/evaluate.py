"""Post-close evaluator (second LangGraph).

Runs the morning after a session: for every locked signal that has no `actuals`
row yet, pull the settled close, grade it, and append (never update) the result.

Grading (PLAN §7):
  - inside_envelope : did the actual close-to-close return land in Kronos's 5-95 band?
  - dir_match_struct: sign(actual) == sign(median r)?
  - dir_match_narr  : sign(actual) == narrative direction?
  - divergence_outcome (only meaningful on divergence rows):
        which side's direction matched the realised sign.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from .data import DataError, load_actual
from .store import (
    ImmutableViolation,
    append_actual,
    load_actuals,
    load_signals,
    sync_db,
)


def _sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


class EState(TypedDict, total=False):
    asof: str
    ticker: str
    signal: dict
    actual: Any
    result: dict
    error: str
    persisted: bool


def n_load_actual(state: EState) -> EState:
    try:
        a = load_actual(state["ticker"], state["asof"])
        return {**state, "actual": a}
    except DataError as exc:
        return {**state, "error": f"load_actual: {exc}"}


def n_grade(state: EState) -> EState:
    s, a = state["signal"], state["actual"]
    ret = a.ret
    inside = bool(s["kronos_q05"] <= ret <= s["kronos_q95"])
    dm_struct = _sign(ret) == _sign(s["kronos_q50"]) and _sign(ret) != 0
    dm_narr = _sign(ret) == int(s["narr_dir"]) and int(s["narr_dir"]) != 0

    dtype = s.get("divergence_type", "aligned")
    if dtype == "aligned":
        outcome = "n/a"
    else:
        narr_ok = _sign(ret) == int(s["narr_dir"])
        struct_ok = _sign(ret) == _sign(s["kronos_q50"])
        outcome = ({(True, True): "both", (True, False): "narrative",
                    (False, True): "structure", (False, False): "neither"}
                   [(narr_ok, struct_ok)])

    result = {
        "asof": state["asof"], "ticker": state["ticker"],
        "price_source": a.source,
        "prev_close": a.prev_close, "actual_close": a.actual_close, "ret": ret,
        "inside_envelope": inside,
        "dir_match_struct": dm_struct, "dir_match_narr": dm_narr,
        "divergence_outcome": outcome,
    }
    return {**state, "result": result}


def n_persist(state: EState) -> EState:
    try:
        append_actual(state["result"])
        return {**state, "persisted": True}
    except ImmutableViolation as exc:
        return {**state, "error": str(exc), "persisted": False}


def n_skip(state: EState) -> EState:
    print(f"  [{state['ticker']} {state['asof']}] skipped — {state.get('error')}",
          file=sys.stderr)
    return state


def _route(state: EState) -> str:
    return "skip" if state.get("error") else "grade"


def build_eval_graph():
    g = StateGraph(EState)
    g.add_node("load_actual", n_load_actual)
    g.add_node("grade", n_grade)
    g.add_node("persist", n_persist)
    g.add_node("skip", n_skip)
    g.add_edge(START, "load_actual")
    g.add_conditional_edges("load_actual", _route, {"grade": "grade", "skip": "skip"})
    g.add_edge("grade", "persist")
    g.add_edge("persist", END)
    g.add_edge("skip", END)
    return g.compile()


_GRAPH = None


def pending() -> list[dict]:
    sig = load_signals()
    if sig.empty:
        return []
    act = load_actuals()
    done = set() if act.empty else set(zip(act["asof"].astype(str), act["ticker"]))
    return [r for r in sig.to_dict("records")
            if (str(r["asof"]), r["ticker"]) not in done]


def run(asof: str | None = None) -> list[EState]:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_eval_graph()
    todo = pending()
    if asof:
        todo = [r for r in todo if str(r["asof"]) == str(asof)]
    out = []
    for r in todo:
        out.append(_GRAPH.invoke({"asof": str(r["asof"]), "ticker": r["ticker"], "signal": r}))
    sync_db()
    return out


def report() -> str:
    from .store import joined

    j = joined()
    if j.empty or "ret" not in j.columns:
        return "no graded signals yet"
    g = j.dropna(subset=["ret"])
    if g.empty:
        return "no graded signals yet"

    lines = [f"graded signal-days: {len(g)}  ({g['ticker'].nunique()} tickers, "
             f"{g['asof'].nunique()} dates)"]
    lines.append(f"  envelope coverage (target ~90%): {g['inside_envelope'].mean():.0%}")
    lines.append(f"  structure direction hit:         {g['dir_match_struct'].mean():.0%}")
    lines.append(f"  narrative direction hit:         {g['dir_match_narr'].mean():.0%}")

    dv = g[g["divergence_type"] != "aligned"]
    if not dv.empty:
        oc = dv["divergence_outcome"].value_counts()
        lines.append(f"  divergence rows: {len(dv)}")
        for k, v in oc.items():
            lines.append(f"    {k:10} {v}")
        wins_n = (dv["divergence_outcome"] == "narrative").sum()
        wins_s = (dv["divergence_outcome"] == "structure").sum()
        decisive = wins_n + wins_s
        if decisive:
            lo, hi = _wilson(wins_n, decisive)
            lines.append(f"  narrative win rate on decisive divergences: "
                         f"{wins_n}/{decisive} = {wins_n/decisive:.0%}  "
                         f"(95% CI {lo:.0%}-{hi:.0%})")
    else:
        lines.append("  no divergence rows yet")
    return "\n".join(lines)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grade locked signals against the close.")
    ap.add_argument("--asof", help="grade only this date (default: all pending)")
    ap.add_argument("--report", action="store_true", help="just print the running report")
    args = ap.parse_args(argv)

    if args.report:
        print(report())
        return 0

    res = run(args.asof)
    graded = [s for s in res if s.get("persisted")]
    print(f"graded {len(graded)}/{len(res)} pending signal-days")
    for s in graded:
        r = s["result"]
        print(f"  {r['ticker']:6} {r['asof']}  ret {r['ret']*100:+5.2f}%  "
              f"inside={r['inside_envelope']!s:5}  "
              f"struct={r['dir_match_struct']!s:5} narr={r['dir_match_narr']!s:5}  "
              f"{r['divergence_outcome']}")
    print("\n" + report())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
