"""MCP server exposing the pilot's two capabilities as tools.

Any MCP client (Claude Desktop, an IDE, a PM's assistant) can then call:

  - get_kronos_distribution(ticker, asof?)  -> price-only forecast distribution
  - get_headlines(ticker, asof?)            -> price-blind overnight news bundle
  - get_signal(ticker, asof)                -> a locked signal row, if it exists

Run:  python -m src.mcp_server        (stdio transport)
"""
from __future__ import annotations

from datetime import date

from mcp.server.fastmcp import FastMCP

from .config import PRED_LEN, QUANTILES, SAMPLE_COUNT
from .data import load_context
from .headlines_agent import get_headlines as _get_headlines
from .kronos_infer import forecast
from .narrative import score_narrative
from .store import load_signals

mcp = FastMCP("narrative-arb")


@mcp.tool()
def get_kronos_distribution(ticker: str, asof: str | None = None,
                            samples: int = SAMPLE_COUNT) -> dict:
    """Kronos's price-only forecast for `ticker` as of `asof` (default: today).

    Returns the graded-bar close-to-close return distribution: median, P(up),
    dispersion, and 5/25/50/75/95 quantiles. Kronos sees only price/volume
    history strictly before `asof` — no news, no lookahead.
    """
    asof = asof or str(date.today())
    pl = load_context(ticker, asof)
    fc = forecast(pl, sample_count=samples, pred_len=PRED_LEN)
    rq = fc.ret_quantiles(QUANTILES)
    return {
        "ticker": ticker, "asof": asof,
        "context_through": str(pl.last_context_date.date()),
        "price_source": pl.source,
        "prev_close": round(fc.prev_close, 4),
        "median_return": fc.median_ret,
        "p_up": fc.p_up,
        "std_return": fc.std_ret,
        "strength_sd": fc.strength,
        "return_quantiles": {f"q{int(q*100):02d}": rq[q] for q in QUANTILES},
        "n_samples": samples,
    }


@mcp.tool()
def get_headlines(ticker: str, asof: str | None = None) -> dict:
    """Overnight, price-blind news bundle for `ticker` as of `asof`.

    Uses curated+cited headlines for historical dates, a live fetch otherwise.
    Returns the deduped, materiality-filtered set and the agent's trace notes.
    """
    asof = asof or str(date.today())
    hs = _get_headlines(ticker, asof)
    return {
        "ticker": ticker, "asof": asof,
        "mode": hs.get("mode"),
        "no_news": hs.get("no_news"),
        "sources_used": hs.get("sources_used", []),
        "notes": hs.get("notes", []),
        "headlines": hs.get("headlines", []),
    }


@mcp.tool()
def get_narrative_score(ticker: str, asof: str | None = None) -> dict:
    """Run the price-blind narrative scorer for `ticker` as of `asof`."""
    asof = asof or str(date.today())
    hs = _get_headlines(ticker, asof)
    sc = score_narrative(ticker, hs.get("headlines", []), asof)
    return sc.to_dict()


@mcp.tool()
def get_signal(ticker: str, asof: str) -> dict:
    """Return the locked signal row for (ticker, asof), or an empty dict."""
    df = load_signals()
    if df.empty:
        return {}
    hit = df[(df["ticker"] == ticker) & (df["asof"].astype(str) == asof)]
    return {} if hit.empty else hit.iloc[0].to_dict()


if __name__ == "__main__":
    mcp.run()
