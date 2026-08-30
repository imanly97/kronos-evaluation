"""Post-mortem agent — the post-close half of the loop (PLAN §5.3).

For a graded session, look at the watchlist names and the day's biggest
surprises (actual landed in Kronos's tail) and classify *why* each happened:

    within-expected-dispersion | catalyst:<type> | regime-move | data-issue | unexplained

Tools (deterministic):
  quantile_of_actual  where the close landed in the forecast distribution
  sector_move         did the correlated cluster move together (regime vs idio)
  known_catalyst      earnings-calendar / scheduled-macro lookup (curated, not sentiment)
  data_sanity         split / stale-price / volume checks

Writes one `memory.episodes` line per name and rebuilds `memory/stats.json`.
This is what closes the loop — triage reads those stats back the next morning.

    from src.postmortem import run
    run("2025-05-05")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from functools import lru_cache

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from .config import CLUSTER_CORR_MIN, CLUSTER_CORR_WINDOW, POSTMORTEM_TAIL_Q, ROOT, UNIVERSE
from .data import _coerce_date, _read_cache
from .llm import complete_json, have_key
from .memory import append_episode, rebuild_stats
from .store import load

CATALYSTS_PATH = ROOT / "eval" / "catalysts.json"
CLASSES = ["within-expected-dispersion", "regime-move", "data-issue", "unexplained"]
MACRO_WINDOW_DAYS = 0     # macro prints move the tape same-day
EARNINGS_WINDOW_DAYS = 1  # an after-hours print shows in the next session's bar


@lru_cache(maxsize=64)
def _prices(ticker: str) -> pd.DataFrame | None:
    df = _read_cache(ticker)
    if df is None:
        return None
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def _ret_on(ticker: str, day: pd.Timestamp) -> float | None:
    df = _prices(ticker)
    if df is None or day not in df.index:
        return None
    i = df.index.get_loc(day)
    if i < 1:
        return None
    return float(df["close"].iloc[i] / df["close"].iloc[i - 1] - 1.0)


# --------------------------------------------------------------------------- #
# deterministic tools
# --------------------------------------------------------------------------- #
def quantile_of_actual(asof: str, ticker: str) -> float | None:
    a = load("actuals")
    row = a[(a["asof"].astype(str) == str(asof)) & (a["ticker"] == ticker)]
    return None if row.empty else float(row["actual_quantile"].iloc[0])


def _peers(ticker: str, day: pd.Timestamp, universe: list[str],
           *, window: int = CLUSTER_CORR_WINDOW, min_corr: float = CLUSTER_CORR_MIN) -> list[str]:
    """Names whose trailing-`window` returns (strictly before `day`) correlate with `ticker`."""
    cols = {}
    for t in [ticker] + [u for u in universe if u != ticker]:
        df = _prices(t)
        if df is None:
            continue
        s = df[df.index < day]["close"].pct_change().dropna().iloc[-window:]
        if len(s) >= window // 2:
            cols[t] = s
    if ticker not in cols or len(cols) < 2:
        return []
    mat = pd.DataFrame(cols).dropna(how="any")
    if ticker not in mat.columns or len(mat) < window // 2:
        return []
    corr = mat.corr()[ticker].drop(ticker)
    return list(corr[corr.abs() >= min_corr].index)


def sector_move(ticker: str, asof: str, *, universe: list[str] | None = None) -> dict:
    """Return of `ticker` vs its correlated cluster on `asof` (regime vs idio). Cache-only."""
    universe = universe or UNIVERSE
    day = pd.Timestamp(_coerce_date(asof))
    peers = _peers(ticker, day, universe)

    my = _ret_on(ticker, day)
    peer_rets = [r for r in (_ret_on(p, day) for p in peers) if r is not None]
    if my is None or len(peer_rets) < 3:
        return {"name_ret": None if my is None else round(my, 4), "cluster_ret": None,
                "peers_n": len(peer_rets), "moved_together": None, "read": "unknown"}
    cl_ret = float(np.mean(peer_rets))
    together = bool(np.sign(my) == np.sign(cl_ret) and abs(cl_ret) >= 0.005
                    and abs(my - cl_ret) <= max(0.01, 1.5 * abs(cl_ret)))
    return {
        "name_ret": round(my, 4), "cluster_ret": round(cl_ret, 4),
        "peers_n": len(peer_rets), "moved_together": together,
        "read": "regime" if together else "idiosyncratic",
    }


def _load_catalysts() -> dict:
    if CATALYSTS_PATH.exists():
        return json.loads(CATALYSTS_PATH.read_text())
    return {"earnings": {}, "macro": []}


def known_catalyst(ticker: str, asof: str) -> dict:
    """Scheduled events near `asof`, split into the name's OWN event (earnings) and
    market-wide events (macro). Curated calendar — not sentiment."""
    cat = _load_catalysts()
    day = _coerce_date(asof)
    earnings = [{"type": "earnings", "date": d}
                for d in cat.get("earnings", {}).get(ticker, [])
                if abs((_coerce_date(d) - day).days) <= EARNINGS_WINDOW_DAYS]
    macro = [{"type": f"macro:{ev['kind']}", "date": ev["date"]}
             for ev in cat.get("macro", [])
             if abs((_coerce_date(ev["date"]) - day).days) <= MACRO_WINDOW_DAYS]
    return {"earnings": earnings, "macro": macro,
            "has_earnings": bool(earnings), "has_macro": bool(macro)}


def data_sanity(ticker: str, asof: str) -> dict:
    """Cheap integrity checks on the bar we just graded."""
    day = pd.Timestamp(_coerce_date(asof))
    df = _read_cache(ticker)
    if df is None or day not in df.index:
        return {"ok": False, "issues": ["no settled bar in cache"]}
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    i = df.index.get_loc(day)
    issues = []
    row, prev = df.iloc[i], df.iloc[i - 1]
    gap = row["open"] / prev["close"] - 1.0
    if abs(gap) > 0.20:
        issues.append(f"overnight gap {gap:+.0%} — possible split/adjustment artefact")
    if row["high"] < row["low"] or row["close"] <= 0:
        issues.append("impossible OHLC")
    vol_med = float(df["volume"].iloc[max(0, i - 21):i].median())
    if vol_med and row["volume"] < 0.1 * vol_med:
        issues.append("volume < 10% of trailing median — thin/holiday bar")
    return {"ok": not issues, "issues": issues, "gap": round(float(gap), 4)}


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
class PState(TypedDict, total=False):
    asof: str
    universe: list[str]
    queue: list[str]
    scope_reason: dict[str, str]
    done: list[dict]
    trace: list[str]


def _in_scope(asof: str, universe: list[str]) -> dict[str, str]:
    j = load("actuals")
    j = j[j["asof"].astype(str) == str(asof)]
    scope: dict[str, str] = {}
    w = load("watchlist")
    if not w.empty:
        for t in w[w["asof"].astype(str) == str(asof)]["ticker"]:
            scope[t] = "watchlist"
    for _, r in j.iterrows():
        q = r["actual_quantile"]
        if q is not None and (q <= POSTMORTEM_TAIL_Q or q >= 1 - POSTMORTEM_TAIL_Q):
            scope.setdefault(r["ticker"], f"tail surprise (q={q:.2f})")
    return scope


def n_scope(state: PState) -> PState:
    universe = state.get("universe") or UNIVERSE
    scope = _in_scope(state["asof"], universe)
    return {**state, "universe": universe, "queue": list(scope),
            "scope_reason": scope, "done": [],
            "trace": [f"{len(scope)} names in scope: "
                      + ", ".join(f"{k} ({v})" for k, v in scope.items())]}


def n_examine(state: PState) -> PState:
    queue = list(state["queue"])
    ticker = queue.pop(0)
    asof = state["asof"]

    a = load("actuals")
    arow = a[(a["asof"].astype(str) == str(asof)) & (a["ticker"] == ticker)]
    f = load("forecasts")
    frow = f[(f["asof"].astype(str) == str(asof)) & (f["ticker"] == ticker)]
    if arow.empty or frow.empty:
        return {**state, "queue": queue,
                "trace": state["trace"] + [f"{ticker}: no graded forecast, skipped"]}
    arow, frow = arow.iloc[0], frow.iloc[0]

    q = float(arow["actual_quantile"])
    sm = sector_move(ticker, asof, universe=state["universe"])
    cat = known_catalyst(ticker, asof)
    ds = data_sanity(ticker, asof)

    # setup label for this name/day (the memory key) — from the watchlist if flagged
    w = load("watchlist")
    wrow = w[(w["asof"].astype(str) == str(asof)) & (w["ticker"] == ticker)]
    if not wrow.empty:
        setup = wrow.iloc[0]["setup"]
    else:
        from .setups import describe_setup
        setup = describe_setup(ticker, asof).label

    dossier = {
        "ticker": ticker, "asof": asof, "setup": setup,
        "forecast_median": round(float(frow["q50"]), 5),
        "actual_ret": round(float(arow["ret"]), 5),
        "actual_quantile": round(q, 4),
        "inside_envelope": bool(arow["inside_envelope"]),
        "sector_move": sm, "catalyst": cat, "data_sanity": ds,
    }
    cls, note = _classify(dossier)
    episode = {
        "asof": asof, "ticker": ticker, "setup": setup,
        "forecast_median": dossier["forecast_median"], "actual": dossier["actual_ret"],
        "actual_quantile": dossier["actual_quantile"],
        "classification": cls, "note": note,
    }
    append_episode(episode)
    return {
        **state, "queue": queue,
        "done": state["done"] + [episode],
        "trace": state["trace"] + [
            f"{ticker}: q={q:.2f} {sm['read']} "
            f"earn={cat['has_earnings']} macro={cat['has_macro']} -> {cls} — {note}"],
    }


def _classify(d: dict) -> tuple[str, str]:
    """Deterministic guardrails first; LLM only for the genuinely ambiguous middle.

    Order matters: a name's OWN earnings print outranks everything; a market-wide
    macro print only *explains* a move that was actually regime-wide.
    """
    q = d["actual_quantile"]
    cat, sm = d["catalyst"], d["sector_move"]
    regime = sm["read"] == "regime"

    if not d["data_sanity"]["ok"]:
        return "data-issue", "; ".join(d["data_sanity"]["issues"])
    if cat["has_earnings"]:
        ev = cat["earnings"][0]
        return "catalyst:earnings", f"earnings {ev['date']}, close at q{q:.2f}"
    tail = q <= POSTMORTEM_TAIL_Q or q >= 1 - POSTMORTEM_TAIL_Q
    if not tail:
        return "within-expected-dispersion", f"close landed at q{q:.2f}, mid-distribution"
    if regime and cat["has_macro"]:
        ev = cat["macro"][0]
        return "regime-move", (f"{ev['type']} {ev['date']}; cluster {sm['cluster_ret']:+.2%} "
                               f"vs name {sm['name_ret']:+.2%}")
    if regime:
        return "regime-move", (f"cluster moved {sm['cluster_ret']:+.2%}, "
                               f"name {sm['name_ret']:+.2%} (no scheduled catalyst)")
    if not have_key():
        return "unexplained", f"tail outcome q{q:.2f}, idiosyncratic, no scheduled catalyst (mock)"

    macro_hint = (f" A market-wide {cat['macro'][0]['type']} print landed on "
                  f"{cat['macro'][0]['date']} but this name did not move with its cluster."
                  if cat["has_macro"] else "")
    try:
        out = complete_json(
            "You are a post-mortem analyst on a forecasting desk. A forecast's "
            "actual close landed deep in the distribution's tail "
            f"(q{q:.2f}), the name did NOT move with its correlated cluster, and "
            "there is no scheduled earnings date for it. This is a genuine miss "
            "that needs a reason. Choose one label: 'regime-move' (cluster "
            "evidence is borderline but suggestive of a broad move), "
            "'catalyst:unlisted' (a real, nameable company event our calendar "
            "missed — you may use historical knowledge, but only name one you are "
            "confident occurred), or 'unexplained' (no identifiable cause). Do NOT "
            "answer 'within-expected-dispersion' — a tail outcome is by definition "
            "not that." + macro_hint,
            f"DOSSIER:\n{json.dumps(d, indent=2, default=str)}\n\n"
            'Return JSON: {"classification": "<label>", "note": "<max 20 words>"}',
            max_tokens=600,
        )
    except ValueError:
        return "unexplained", f"idiosyncratic tail (q{q:.2f}); classifier returned no usable label"
    cls = str(out.get("classification", "unexplained"))
    if cls == "within-expected-dispersion" or (cls not in CLASSES and not cls.startswith("catalyst:")):
        cls = "unexplained"
    return cls, str(out.get("note", "")).strip()


def n_finish(state: PState) -> PState:
    stats = rebuild_stats()
    return {**state, "trace": state["trace"] + [
        f"wrote {len(state['done'])} episodes; stats table now {len(stats)} (ticker,setup) keys"]}


def _route(state: PState) -> str:
    return "examine" if state["queue"] else "finish"


def build_graph():
    g = StateGraph(PState)
    g.add_node("scope", n_scope)
    g.add_node("examine", n_examine)
    g.add_node("finish", n_finish)
    g.add_edge(START, "scope")
    g.add_conditional_edges("scope", _route, {"examine": "examine", "finish": "finish"})
    g.add_conditional_edges("examine", _route, {"examine": "examine", "finish": "finish"})
    g.add_edge("finish", END)
    return g.compile()


_GRAPH = None


def run(asof: str, *, universe: list[str] | None = None) -> PState:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    init: PState = {"asof": str(_coerce_date(asof))}
    if universe:
        init["universe"] = universe
    return _GRAPH.invoke(init, config={"recursion_limit": 150})


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the post-mortem agent for a session.")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--universe", nargs="+")
    args = ap.parse_args(argv)
    st = run(args.asof, universe=args.universe)
    print("\n".join(f"  {t}" for t in st["trace"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
