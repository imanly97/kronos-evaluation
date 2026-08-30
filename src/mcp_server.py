"""MCP server exposing the forecast desk to any MCP client (PLAN §5, §13).

Tools:
  todays_watchlist(asof?)        the flagged names + briefs for a session
  why_flagged(ticker, asof)      the full reason one name made the watchlist
  forecast_row(ticker, asof)     the deterministic Kronos row for a name/day
  desk_memory(ticker, setup)     the desk's prior track record for that situation
  loop_status()                  coverage / hit-rate summary across the log

Run:  python -m src.mcp_server        (stdio transport)
"""
from __future__ import annotations

from datetime import date

from mcp.server.fastmcp import FastMCP

from .grade import report as grade_report
from .memory import recall
from .store import load

mcp = FastMCP("forecast-desk")


def _latest_asof(table: str) -> str | None:
    df = load(table)
    return None if df.empty else str(df["asof"].max())


@mcp.tool()
def todays_watchlist(asof: str | None = None) -> dict:
    """The triage watchlist for `asof` (default: the most recent run): each
    flagged name with its setup, rank, one-line reason, and the full brief."""
    w = load("watchlist")
    if w.empty:
        return {"asof": None, "names": [], "note": "no watchlist logged yet"}
    asof = asof or str(w["asof"].max())
    rows = w[w["asof"].astype(str) == str(asof)].sort_values("rank")
    return {
        "asof": asof,
        "names": [
            {"ticker": r["ticker"], "rank": int(r["rank"]), "setup": r["setup"],
             "reason": r["reason"], "cluster_read": r["cluster_read"],
             "analog_base_rate": r["analog_base_rate"], "caveat": r["caveat"],
             "brief": r["brief_md"]}
            for _, r in rows.iterrows()
        ],
    }


@mcp.tool()
def why_flagged(ticker: str, asof: str | None = None) -> dict:
    """Why `ticker` made the watchlist on `asof`: the reason line, the analog
    base rate, the correlated-cluster read, the desk-memory line, and the brief."""
    w = load("watchlist")
    asof = asof or _latest_asof("watchlist")
    row = w[(w["asof"].astype(str) == str(asof)) & (w["ticker"] == ticker)]
    if row.empty:
        return {"ticker": ticker, "asof": asof, "flagged": False,
                "note": "not on the watchlist for that session"}
    r = row.iloc[0]
    return {
        "ticker": ticker, "asof": str(asof), "flagged": True,
        "rank": int(r["rank"]), "setup": r["setup"], "reason": r["reason"],
        "analog_base_rate": r["analog_base_rate"], "analog_n": r["analog_n"],
        "cluster_read": r["cluster_read"], "memory_line": r["memory_line"],
        "caveat": r["caveat"], "brief": r["brief_md"],
    }


@mcp.tool()
def forecast_row(ticker: str, asof: str | None = None) -> dict:
    """The deterministic Kronos forecast row for `ticker` / `asof`: return
    quantiles, P(up), dispersion, strength. No lookahead."""
    f = load("forecasts")
    asof = asof or _latest_asof("forecasts")
    row = f[(f["asof"].astype(str) == str(asof)) & (f["ticker"] == ticker)]
    if row.empty:
        return {"ticker": ticker, "asof": asof, "note": "not in the forecast cache"}
    r = row.iloc[0]
    return {k: (float(r[k]) if k in ("q05", "q25", "q50", "q75", "q95", "p_up",
                                     "std", "strength", "prev_close") else r[k])
            for k in f.columns if k not in ("prev_hash", "row_hash")}


@mcp.tool()
def desk_memory(ticker: str, setup: str) -> dict:
    """The desk's prior episodes and hit rate for `ticker` in a `setup` situation."""
    rc = recall(ticker, setup)
    return {"ticker": ticker, "setup": setup, "n": rc.n, "hit_rate": rc.hit_rate,
            "mean_signed_err": rc.mean_signed_err, "mean_abs_err": rc.mean_abs_err,
            "line": rc.line(), "episodes": rc.episodes}


@mcp.tool()
def loop_status() -> dict:
    """Coverage and hit-rate summary across every graded forecast in the log."""
    f, w, a = load("forecasts"), load("watchlist"), load("actuals")
    return {
        "forecasts": len(f), "watchlist": len(w), "graded": len(a),
        "forecast_dates": None if f.empty else [str(f["asof"].min()), str(f["asof"].max())],
        "report": grade_report(),
    }


if __name__ == "__main__":
    mcp.run()
