"""The nightly signal graph (LangGraph).

Per ticker:

    load_prices ─(ok)─> run_kronos ─> run_headlines ─> score_narrative
         │                                                   │
      (data error)                                            v
         └──────────────> handle_error <──────── assess_divergence
                              │                              │
                             END                          persist ─> END

Design principle (PLAN §5): an agent only where there is real reasoning
(`run_headlines`, and the LLM brief inside `assess_divergence`); everything else
is deterministic and labelled as such.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import date, datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import PRED_LEN, QUANTILES, SAMPLE_COUNT, UNIVERSE
from .data import DataError, load_context
from .divergence import assess
from .headlines_agent import get_headlines
from .kronos_infer import forecast
from .narrative import score_narrative
from .store import ImmutableViolation, append_signal, sync_db


class GState(TypedDict, total=False):
    ticker: str
    asof: str
    seed: int | None
    with_brief: bool
    allow_replace: bool

    price: Any
    forecast: Any
    headlines: dict
    narrative: Any
    divergence: Any

    error: str
    record: dict
    persisted: bool


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
def n_load_prices(state: GState) -> GState:
    try:
        pl = load_context(state["ticker"], state["asof"])
        return {**state, "price": pl}
    except DataError as exc:
        return {**state, "error": f"load_prices: {exc}"}


def n_run_kronos(state: GState) -> GState:
    fc = forecast(state["price"], sample_count=SAMPLE_COUNT, pred_len=PRED_LEN,
                  seed=state.get("seed"))
    return {**state, "forecast": fc}


def n_run_headlines(state: GState) -> GState:
    hs = get_headlines(state["ticker"], state["asof"])
    return {**state, "headlines": hs}


def n_score_narrative(state: GState) -> GState:
    hs = state["headlines"]
    sc = score_narrative(state["ticker"], hs.get("headlines", []), state["asof"])
    return {**state, "narrative": sc}


def n_assess_divergence(state: GState) -> GState:
    div = assess(state["narrative"], state["forecast"],
                 brief=state.get("with_brief", True))
    return {**state, "divergence": div}


def n_persist(state: GState) -> GState:
    fc, sc, div = state["forecast"], state["narrative"], state["divergence"]
    pl = state["price"]
    rq = fc.ret_quantiles(QUANTILES)
    record = {
        "asof": state["asof"], "ticker": state["ticker"],
        "llm_contaminated": sc.llm_contaminated,
        "price_source": pl.source,
        "context_from": str(pl.df.index.min().date()),
        "context_to": str(pl.last_context_date.date()),
        "prev_close": fc.prev_close,
        "kronos_q05": rq[0.05], "kronos_q25": rq[0.25], "kronos_q50": rq[0.50],
        "kronos_q75": rq[0.75], "kronos_q95": rq[0.95],
        "kronos_p_up": fc.p_up, "kronos_std": fc.std_ret,
        "kronos_strength": fc.strength, "kronos_dir": div.struct_dir,
        "narr_dir": sc.dir_int, "narr_conviction": sc.conviction,
        "narr_rationale": sc.rationale, "narr_headline_count": sc.headline_count,
        "narr_signal": sc.signal,
        "divergence_type": div.divergence_type,
        "divergence_verdict": "pending",
        "brief": div.brief,
    }
    try:
        append_signal(record, allow_replace=state.get("allow_replace", False))
        persisted = True
    except ImmutableViolation as exc:
        return {**state, "record": record, "persisted": False,
                "error": f"persist: {exc}"}
    return {**state, "record": record, "persisted": persisted}


def n_handle_error(state: GState) -> GState:
    print(f"  [{state['ticker']}] SKIPPED — {state.get('error')}", file=sys.stderr)
    return state


def _route_after_prices(state: GState) -> str:
    return "handle_error" if state.get("error") else "run_kronos"


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(GState)
    g.add_node("load_prices", n_load_prices)
    g.add_node("run_kronos", n_run_kronos)
    g.add_node("run_headlines", n_run_headlines)
    g.add_node("score_narrative", n_score_narrative)
    g.add_node("assess_divergence", n_assess_divergence)
    g.add_node("persist", n_persist)
    g.add_node("handle_error", n_handle_error)

    g.add_edge(START, "load_prices")
    g.add_conditional_edges("load_prices", _route_after_prices,
                            {"run_kronos": "run_kronos", "handle_error": "handle_error"})
    g.add_edge("run_kronos", "run_headlines")
    g.add_edge("run_headlines", "score_narrative")
    g.add_edge("score_narrative", "assess_divergence")
    g.add_edge("assess_divergence", "persist")
    g.add_edge("persist", END)
    g.add_edge("handle_error", END)
    return g.compile()


_GRAPH = None


def run_ticker(ticker: str, asof: str | date, *, with_brief: bool = True,
               allow_replace: bool = False, seed: int | None = None) -> GState:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH.invoke({
        "ticker": ticker, "asof": str(asof), "with_brief": with_brief,
        "allow_replace": allow_replace, "seed": seed,
    })


def run_universe(asof: str | date, tickers: list[str] | None = None, **kw) -> list[GState]:
    tickers = tickers or UNIVERSE
    out = []
    for t in tickers:
        print(f"[{t}] …", flush=True)
        try:
            out.append(run_ticker(t, asof, **kw))
        except Exception:  # noqa: BLE001 — one ticker must not kill the run
            traceback.print_exc()
            out.append({"ticker": t, "asof": str(asof), "error": "unhandled exception"})
    sync_db()
    return out


def _summary_row(s: GState) -> str:
    if s.get("error") and not s.get("persisted"):
        return f"{s['ticker']:6}  ERROR  {s.get('error','')[:70]}"
    d = s["divergence"]
    n = s["narrative"]
    return (f"{s['ticker']:6}  struct {d.struct_median*100:+5.2f}%  "
            f"P(up) {s['forecast'].p_up:4.0%}   "
            f"narr {n.direction:>7}/{n.conviction}   "
            f"=> {d.divergence_type}")


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the nightly signal graph.")
    ap.add_argument("--asof", default=str(date.today()))
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--no-brief", action="store_true")
    ap.add_argument("--replace", action="store_true", help="overwrite an existing lock")
    ap.add_argument("--draw", action="store_true", help="print the graph as mermaid")
    args = ap.parse_args(argv)

    if args.draw:
        print(build_graph().get_graph().draw_mermaid())
        return 0

    results = run_universe(args.asof, args.tickers, with_brief=not args.no_brief,
                           allow_replace=args.replace)
    print("\n" + "=" * 72)
    for s in results:
        print(_summary_row(s))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
