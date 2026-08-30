"""Triage agent — the pre-market half of the loop (PLAN §5.1).

A LangGraph graph with real state and branching: from one day's ~40 locked
forecasts, rank them, then walk the top candidates one at a time — describe the
setup, pull analogs, read the correlated cluster, recall desk memory — and decide
keep or drop. Every kept name gets a brief that the groundedness critic
(`critic.py`) signs off before it lands.

Output: rows in `store.watchlist` + a PM-readable `log/brief_<asof>.md`.

    from src.triage import run
    state = run("2025-05-29")
"""
from __future__ import annotations

import argparse
import os
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .config import TRIAGE_MAX, TRIAGE_MAX_CANDIDATES, TRIAGE_MIN, UNIVERSE
from .critic import MAX_ROUNDS, review
from .features import correlated_cluster, rank_table
from .forecasts import strength_history
from .llm import complete, complete_json, have_key
from .memory import recall
from .setups import describe_setup, find_analogs
from .store import append, load

_FCOLS = ["ticker", "q50", "q05", "q95", "p_up", "std", "strength"]

# the ablation switch (PLAN §7): FMD_TRIAGE_MEMORY=0 makes recall a no-op so the
# same replay can be run memory-on vs memory-off and the outputs compared.
USE_MEMORY = os.getenv("FMD_TRIAGE_MEMORY", "1") != "0"


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class TState(TypedDict, total=False):
    asof: str
    universe: list[str]
    ranked: list[dict]
    queue: list[str]
    considered: list[str]
    dossiers: dict[str, dict]
    kept: list[str]
    dropped: dict[str, str]
    briefs: dict[str, dict]
    trace: list[str]


class _NoMemory:
    n = 0
    hit_rate = None

    @staticmethod
    def line() -> str:
        return "desk memory disabled for this run (ablation)"


_NO_MEMORY = _NoMemory()


def _forecast_table(asof: str, universe: list[str]) -> pd.DataFrame:
    f = load("forecasts")
    f = f[(f["asof"].astype(str) == str(asof)) & (f["ticker"].isin(universe))]
    if f.empty:
        raise RuntimeError(f"no forecasts in the cache for {asof}; run src.forecasts first")
    return f[_FCOLS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
def n_rank(state: TState) -> TState:
    universe = state.get("universe") or UNIVERSE
    fc = _forecast_table(state["asof"], universe)
    hist = {t: strength_history(t, state["asof"]) for t in fc["ticker"]}
    ranked = rank_table(fc, strength_hist=hist)
    top = ranked.head(TRIAGE_MAX_CANDIDATES)["ticker"].tolist()
    return {
        **state,
        "universe": universe,
        "ranked": ranked.to_dict("records"),
        "queue": top,
        "considered": [], "dossiers": {}, "kept": [], "dropped": {}, "briefs": {},
        "trace": [f"ranked {len(ranked)} forecasts; {len(top)} candidates queued"],
    }


def n_investigate(state: TState) -> TState:
    queue = list(state["queue"])
    ticker = queue.pop(0)
    asof = state["asof"]

    setup = describe_setup(ticker, asof)
    analog = find_analogs(ticker, setup.label, asof)
    fc_df = pd.DataFrame(state["ranked"])
    cluster = correlated_cluster(ticker, asof, fc_df[["ticker", "q50"]],
                                 universe=state["universe"])
    mem = recall(ticker, setup.label) if USE_MEMORY else _NO_MEMORY
    rank_row = next(r for r in state["ranked"] if r["ticker"] == ticker)

    dossier = {
        "ticker": ticker,
        "rank": int(rank_row["rank"]),
        "forecast": {
            "median_ret": round(rank_row["q50"], 5),
            "p_up": round(rank_row["p_up"], 3),
            "dispersion": round(rank_row["dispersion"], 5),
            "q05": round(rank_row["q05"], 5),
            "q95": round(rank_row["q95"], 5),
            "strength": round(rank_row["strength"], 3),
            "strength_z": round(rank_row["strength_z"], 2),
            "strength_z_basis": rank_row["strength_z_basis"],
        },
        "setup": setup.label,
        "setup_features": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in setup.features.as_row().items()
                           if k not in ("ticker", "asof")},
        "analogs": {
            "n": analog.n, "base_rate_up": _r(analog.base_rate_up),
            "mean_ret": _r(analog.mean_ret), "mean_abs_ret": _r(analog.mean_abs_ret),
            "sufficient": analog.sufficient, "line": analog.line(),
        },
        "cluster": cluster,
        "memory": {"n": mem.n, "hit_rate": mem.hit_rate, "line": mem.line()},
    }
    dossiers = {**state["dossiers"], ticker: dossier}
    return {
        **state, "queue": queue,
        "considered": state["considered"] + [ticker],
        "dossiers": dossiers,
        "trace": state["trace"] + [f"investigated {ticker}: {setup.label}, "
                                   f"rank {dossier['rank']}, {analog.line()}"],
    }


def _r(x):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), 5)


_DECIDE_SYS = (
    "You triage a daily forecast universe for a research desk. For the candidate "
    "below decide whether it is worth a human analyst's morning — something is "
    "unusual, actionable, or informative — or whether it is routine and can be "
    "dropped. You are selecting and describing, NOT predicting the market. "
    f"Keep between {TRIAGE_MIN} and {TRIAGE_MAX} names for the day; you have kept "
    "{n_kept} so far with {n_left} candidates still to see. Favour: high strength "
    "or strength_z, a decisive setup, an informative analog base rate, an "
    "idiosyncratic move against its cluster, or a relevant desk-memory record. "
    "Drop: quiet setups, weak strength, nothing distinctive."
)


def n_decide(state: TState) -> TState:
    ticker = state["considered"][-1]
    dossier = state["dossiers"][ticker]
    n_left = len(state["queue"])
    if not have_key():
        keep = (dossier["setup"] != "quiet" and abs(dossier["forecast"]["strength"]) >= 0.6
                and len(state["kept"]) < TRIAGE_MAX)
        reason = "mock decision (no API key): strength/setup gate"
    else:
        out = complete_json(
            _DECIDE_SYS.format(n_kept=len(state["kept"]), n_left=n_left),
            f"CANDIDATE DOSSIER:\n{_fmt(dossier)}\n\n"
            'Return JSON: {"keep": true|false, "reason": "<one sentence>"}',
            max_tokens=300,
        )
        keep = bool(out.get("keep"))
        reason = str(out.get("reason", "")).strip()

    if keep and len(state["kept"]) >= TRIAGE_MAX:
        keep, reason = False, f"day full ({TRIAGE_MAX} kept); {reason}"

    if keep:
        return {**state, "kept": state["kept"] + [ticker],
                "trace": state["trace"] + [f"KEEP {ticker} — {reason}"]}
    return {**state, "dropped": {**state["dropped"], ticker: reason},
            "trace": state["trace"] + [f"drop {ticker} — {reason}"]}


_BRIEF_SYS = (
    "Write a short morning brief (120-180 words) for one flagged name, for a "
    "portfolio manager. Start directly with the prose — no title, no header line, "
    "no bold name (the desk adds those). Cover, in prose: why it was flagged; the "
    "setup in plain words; the analog base rate (only if the sample is "
    "sufficient); the correlated-cluster read (macro/sector vs idiosyncratic); "
    "the desk-memory track record (or that there is none yet); and end with ONE "
    "explicit sentence starting 'Caveat:' with the main confidence limitation. "
    "Every number you cite MUST come from the dossier. No price targets, no "
    "certainty language, no market call — you are describing, not predicting."
)
_BRIEF_MAX_TOKENS = 900


def n_brief(state: TState) -> TState:
    ticker = state["kept"][-1]
    dossier = state["dossiers"][ticker]
    values = _flat_values(dossier)

    if not have_key():
        brief_md = _mock_brief(dossier)
    else:
        brief_md = complete(_BRIEF_SYS, f"DOSSIER:\n{_fmt(dossier)}",
                            max_tokens=_BRIEF_MAX_TOKENS)

    rounds = 0
    rep = review(brief_md, values, setup=dossier["setup"])
    while not rep.ok and rounds < MAX_ROUNDS and have_key():
        rounds += 1
        brief_md = complete(
            _BRIEF_SYS,
            f"DOSSIER:\n{_fmt(dossier)}\n\nYOUR PREVIOUS DRAFT:\n{brief_md}\n\n"
            f"The groundedness critic raised:\n{rep.feedback()}\n\nRewrite the brief.",
            max_tokens=_BRIEF_MAX_TOKENS,
        )
        rep = review(brief_md, values, setup=dossier["setup"])

    brief = {
        "asof": state["asof"], "ticker": ticker,
        "rank": dossier["rank"], "setup": dossier["setup"],
        "reason": next((t[len(f"KEEP {ticker} — "):] for t in reversed(state["trace"])
                        if t.startswith(f"KEEP {ticker} — ")), ""),
        "analog_base_rate": dossier["analogs"]["base_rate_up"] if dossier["analogs"]["sufficient"] else None,
        "analog_n": dossier["analogs"]["n"],
        "cluster_read": dossier["cluster"]["read"],
        "memory_line": dossier["memory"]["line"],
        "caveat": _last_sentence(brief_md),
        "brief_md": brief_md,
        "groundedness": round(rep.groundedness, 3),
        "critic_rounds": rounds,
        "critic_ok": rep.ok,
    }
    return {**state, "briefs": {**state["briefs"], ticker: brief},
            "trace": state["trace"] + [
                f"brief {ticker}: groundedness {rep.groundedness:.0%}, "
                f"{rounds} critic round(s), {'clean' if rep.ok else 'residual issues'}"]}


def n_persist(state: TState) -> TState:
    lines = [f"# Morning brief — {state['asof']}", "",
             f"Triage kept **{len(state['kept'])}** of {len(state['considered'])} "
             f"candidates examined (universe {len(state['universe'])}).", ""]
    for t in state["kept"]:
        b = state["briefs"][t]
        try:
            append("watchlist", {
                "asof": b["asof"], "ticker": t, "rank": b["rank"], "setup": b["setup"],
                "reason": b["reason"], "analog_base_rate": b["analog_base_rate"],
                "analog_n": b["analog_n"], "cluster_read": b["cluster_read"],
                "memory_line": b["memory_line"], "caveat": b["caveat"],
                "brief_md": b["brief_md"], "groundedness": b["groundedness"],
                "critic_rounds": b["critic_rounds"], "critic_ok": b["critic_ok"],
            })
        except Exception as exc:  # noqa: BLE001
            state["trace"].append(f"watchlist write skipped for {t}: {exc}")
        lines += [f"## {t}  ·  {b['setup']}  ·  rank {b['rank']}", "", b["brief_md"], "",
                  f"_groundedness {b['groundedness']:.0%} over {b['critic_rounds']} "
                  f"critic round(s)_", "", "---", ""]

    if state["dropped"]:
        lines.append("### Dropped")
        for t, why in state["dropped"].items():
            lines.append(f"- **{t}** — {why}")
    from .config import LOG_DIR
    out = LOG_DIR / f"brief_{state['asof']}.md"
    out.write_text("\n".join(lines))
    return {**state, "trace": state["trace"] + [f"wrote {out}"]}


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
def _route_after_decide(state: TState) -> str:
    return "brief" if (state["kept"] and state["kept"][-1] == state["considered"][-1]) else "next"


def _route_next(state: TState) -> str:
    if not state["queue"] or len(state["kept"]) >= TRIAGE_MAX:
        return "persist"
    return "investigate"


def build_graph():
    g = StateGraph(TState)
    g.add_node("rank", n_rank)
    g.add_node("investigate", n_investigate)
    g.add_node("decide", n_decide)
    g.add_node("brief", n_brief)
    g.add_node("persist", n_persist)
    g.add_edge(START, "rank")
    g.add_edge("rank", "investigate")
    g.add_edge("investigate", "decide")
    g.add_conditional_edges("decide", _route_after_decide,
                            {"brief": "brief", "next": "investigate_or_persist"})
    # a tiny shim so both post-decide paths hit the same router
    g.add_node("investigate_or_persist", lambda s: s)
    g.add_conditional_edges("investigate_or_persist", _route_next,
                            {"investigate": "investigate", "persist": "persist"})
    g.add_conditional_edges("brief", _route_next,
                            {"investigate": "investigate", "persist": "persist"})
    g.add_edge("persist", END)
    return g.compile()


_GRAPH = None


def run(asof: str, *, universe: list[str] | None = None, recursion_limit: int = 120) -> TState:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    init: TState = {"asof": str(asof)}
    if universe:
        init["universe"] = universe
    return _GRAPH.invoke(init, config={"recursion_limit": recursion_limit})


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _fmt(d: dict) -> str:
    import json
    return json.dumps(d, indent=2, default=str)


def _flat_values(dossier: dict) -> dict[str, float]:
    v: dict[str, float] = {}
    for k, x in dossier["forecast"].items():
        if isinstance(x, (int, float)):
            v[f"forecast.{k}"] = x
    for k, x in dossier["setup_features"].items():
        if isinstance(x, (int, float)):
            v[f"feat.{k}"] = x
    a = dossier["analogs"]
    for k in ("n", "base_rate_up", "mean_ret", "mean_abs_ret"):
        if isinstance(a.get(k), (int, float)):
            v[f"analog.{k}"] = a[k]
    c = dossier["cluster"]
    if isinstance(c.get("mean_abs_corr"), (int, float)):
        v["cluster.mean_abs_corr"] = c["mean_abs_corr"]
    if isinstance(c.get("same_sign_frac"), (int, float)):
        v["cluster.same_sign_frac"] = c["same_sign_frac"]
    v["cluster.n_peers"] = len(c.get("peers", []))
    if isinstance(dossier["memory"].get("hit_rate"), (int, float)):
        v["memory.hit_rate"] = dossier["memory"]["hit_rate"]
    v["memory.n"] = dossier["memory"]["n"]
    return v


def _last_sentence(text: str) -> str:
    flat = text.replace("\n", " ")
    i = flat.lower().rfind("caveat:")
    if i != -1:
        return flat[i:].strip()
    parts = [p.strip() for p in flat.split(".") if p.strip()]
    return (parts[-1] + ".") if parts else ""


def _mock_brief(d: dict) -> str:
    f = d["forecast"]
    return (
        f"{d['ticker']} flagged at rank {d['rank']} on a {d['setup']} setup. "
        f"Kronos median {f['median_ret']*100:+.2f}% with P(up) {f['p_up']:.0%} and "
        f"dispersion {f['dispersion']*100:.2f}%; strength {f['strength']:+.2f}. "
        f"{d['analogs']['line']}. Cluster read: {d['cluster']['read']}. "
        f"{d['memory']['line']}. "
        f"Caveat: mock brief with no LLM; treat as a placeholder with low confidence."
    )


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the triage agent for one session.")
    ap.add_argument("--asof", required=True)
    ap.add_argument("--universe", nargs="+")
    args = ap.parse_args(argv)
    st = run(args.asof, universe=args.universe)
    print("\n".join(f"  {t}" for t in st["trace"]))
    print(f"\nkept: {st['kept']}")
    from .config import LOG_DIR
    print(f"brief: {LOG_DIR / f'brief_{args.asof}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
