"""Headline sub-agent (LangGraph).

A small state machine, not a linear script, because there is genuine branching:
source selection depends on whether `asof` is historical or live, and a
"no material news" path has to short-circuit cleanly to a neutral narrative.

    fetch ──> dedupe ──> assess_materiality ──> finalize
                             │
                   (no material headlines)
                             └────────────────> finalize (no_news=True)

Everything here is PRICE-BLIND: we never fetch, pass, or store a quote.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import ANTHROPIC_MODEL, CASE_DIR

HEADLINE_DIR = CASE_DIR / "headlines"
HEADLINE_DIR.mkdir(parents=True, exist_ok=True)


class HState(TypedDict, total=False):
    ticker: str
    asof: str
    mode: str                 # "curated" | "live"
    raw: list[dict]
    headlines: list[dict]     # final, deduped, material
    sources_used: list[str]
    no_news: bool
    notes: list[str]


# --------------------------------------------------------------------------- #
# fetchers
# --------------------------------------------------------------------------- #
def _fetch_curated(ticker: str, asof: str) -> tuple[list[dict], str] | None:
    """Manually curated + cited headlines: case_studies/headlines/<TICKER>_<asof>.json."""
    p = HEADLINE_DIR / f"{ticker.upper()}_{asof}.json"
    if not p.exists():
        return None
    items = json.loads(p.read_text())
    for it in items:
        it.setdefault("source", "curated")
    return items, f"curated:{p.name}"


def _fetch_yf_news(ticker: str, asof: str) -> tuple[list[dict], str] | None:
    """Live headlines from Yahoo via yfinance. Current-only; best-effort."""
    try:
        import yfinance as yf

        news = yf.Ticker(ticker).news or []
    except Exception:  # noqa: BLE001
        return None
    out = []
    for n in news:
        c = n.get("content", n)
        title = c.get("title") or ""
        if not title:
            continue
        pub = c.get("pubDate") or c.get("providerPublishTime") or ""
        out.append({
            "title": title,
            "summary": c.get("summary") or c.get("description") or "",
            "source": (c.get("provider") or {}).get("displayName", "Yahoo")
            if isinstance(c.get("provider"), dict) else "Yahoo",
            "published": str(pub),
            "url": (c.get("canonicalUrl") or {}).get("url", "") if isinstance(
                c.get("canonicalUrl"), dict) else c.get("link", ""),
        })
    return (out, "yfinance.news") if out else ([], "yfinance.news(empty)")


LIVE_FETCHERS = [_fetch_yf_news]


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
def _is_historical(asof: str) -> bool:
    return datetime.strptime(asof, "%Y-%m-%d").date() < date.today()


def n_fetch(state: HState) -> HState:
    ticker, asof = state["ticker"], state["asof"]
    raw, used, notes = [], [], []

    curated = _fetch_curated(ticker, asof)
    if curated:
        raw.extend(curated[0])
        used.append(curated[1])
        notes.append(f"loaded {len(curated[0])} curated headlines")

    mode = "curated" if curated else "live"
    if not curated and _is_historical(asof):
        notes.append("historical asof with no curated file — narrative will be neutral")
    elif not _is_historical(asof):
        for fn in LIVE_FETCHERS:
            got = fn(ticker, asof)
            if got is None:
                continue
            items, tag = got
            raw.extend(items)
            used.append(tag)
            notes.append(f"{tag}: {len(items)} items")
        mode = "live" if not curated else "curated+live"

    return {**state, "raw": raw, "sources_used": used, "mode": mode, "notes": notes}


def n_dedupe(state: HState) -> HState:
    raw = state.get("raw", [])
    kept: list[dict] = []
    for h in raw:
        t = h.get("title", "").strip().lower()
        if not t:
            continue
        if any(SequenceMatcher(None, t, k["title"].strip().lower()).ratio() > 0.82
               for k in kept):
            continue
        kept.append(h)
    notes = state.get("notes", []) + [f"dedupe: {len(raw)} -> {len(kept)}"]
    return {**state, "raw": kept, "notes": notes}


MATERIALITY_SYSTEM = """You filter news headlines for market materiality for a \
single stock. Keep only headlines that a portfolio manager would consider \
potentially price-relevant for the next session: earnings/guidance, regulatory \
or legal action, M&A, major product/customer/supply news, analyst actions with \
a real thesis, macro items that hit this name specifically. Drop recycled \
analysis, listicles, and generic market recaps. Never use or mention price. \
Return the `filter` tool once."""

MATERIALITY_TOOL = {
    "name": "filter",
    "description": "Return which headlines are material.",
    "input_schema": {
        "type": "object",
        "properties": {
            "keep_indices": {"type": "array", "items": {"type": "integer"},
                             "description": "1-based indices of material headlines"},
            "reason": {"type": "string"},
        },
        "required": ["keep_indices", "reason"],
    },
}


def n_assess_materiality(state: HState) -> HState:
    raw = state.get("raw", [])
    notes = list(state.get("notes", []))
    if not raw:
        return {**state, "headlines": [], "notes": notes + ["materiality: nothing to assess"]}

    from .llm import client as _client, have_key

    if not have_key():
        # mock: keep everything, note it
        return {**state, "headlines": raw,
                "notes": notes + ["materiality: MOCK (no API key) — kept all"]}

    listing = "\n".join(f"[{i}] {h.get('title','')} — {(h.get('summary') or '')[:160]}"
                        for i, h in enumerate(raw, 1))
    client = _client()
    resp = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=512, system=MATERIALITY_SYSTEM,
        tools=[MATERIALITY_TOOL], tool_choice={"type": "tool", "name": "filter"},
        messages=[{"role": "user",
                   "content": f"Ticker {state['ticker']}, date {state['asof']}.\n{listing}"}],
    )
    blk = next(b for b in resp.content if getattr(b, "type", None) == "tool_use")
    keep = {i - 1 for i in blk.input.get("keep_indices", [])}
    material = [h for i, h in enumerate(raw) if i in keep]
    return {**state, "headlines": material,
            "notes": notes + [f"materiality: kept {len(material)}/{len(raw)} — {blk.input.get('reason','')}"]}


def n_finalize(state: HState) -> HState:
    hs = state.get("headlines", state.get("raw", []))
    no_news = len(hs) == 0
    notes = state.get("notes", []) + [
        "FINAL: no material news — neutral narrative" if no_news
        else f"FINAL: {len(hs)} material headlines"
    ]
    return {**state, "headlines": hs, "no_news": no_news, "notes": notes}


def _route_after_materiality(state: HState) -> str:
    return "finalize"


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
def build_headlines_graph():
    g = StateGraph(HState)
    g.add_node("fetch", n_fetch)
    g.add_node("dedupe", n_dedupe)
    g.add_node("assess_materiality", n_assess_materiality)
    g.add_node("finalize", n_finalize)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "dedupe")
    g.add_edge("dedupe", "assess_materiality")
    g.add_conditional_edges("assess_materiality", _route_after_materiality,
                            {"finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def get_headlines(ticker: str, asof: str | date) -> HState:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_headlines_graph()
    return _GRAPH.invoke({"ticker": ticker, "asof": str(asof)})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--asof", required=True)
    args = ap.parse_args()
    res = get_headlines(args.ticker, args.asof)
    print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=2, default=str))
