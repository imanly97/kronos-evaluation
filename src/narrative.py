"""Price-blind narrative scorer.

Given only overnight headlines for a ticker, emit a structured view:
direction in {bear, neutral, bull}, conviction 1-10, and a rationale. The model
never sees a price, a chart, or a return. What it *does* see is logged verbatim
alongside the score so the blog can show exactly what the signal was built from.

Honesty caveat (PLAN §5a): for historical `asof` dates the LLM may carry the
realised outcome in its training data. We do not try to scrub that — it is
undetectable — we flag the row `llm_contaminated=True` and lean on the live run.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date

from .config import ANTHROPIC_MODEL, LLM_KNOWLEDGE_CUTOFF

DIR_TO_INT = {"bear": -1, "neutral": 0, "bull": 1}
INT_TO_DIR = {v: k for k, v in DIR_TO_INT.items()}

SYSTEM = """You are a sell-side-style news analyst. You will be given a set of \
overnight news headlines and short summaries about a single US-listed stock, \
plus the trading date they precede.

Your job: judge what the NEWS FLOW alone implies for that stock's next regular \
session, close-to-close.

Hard rules:
- You are given NO price data and must not use, recall, or guess any. Do not \
reference the current price, recent returns, chart levels, or "the stock is \
up/down". Reason only from the news content.
- If the headlines carry no material, price-relevant information, say so and \
return direction "neutral" with low conviction.
- Distinguish a genuine catalyst (earnings, guidance, regulatory action, M&A, \
major product/customer news) from noise (routine coverage, recycled analysis).

Return your answer by calling the `score` tool exactly once."""

SCORE_TOOL = {
    "name": "score",
    "description": "Record the narrative view for this ticker/date.",
    "input_schema": {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bear", "neutral", "bull"]},
            "conviction": {
                "type": "integer", "minimum": 1, "maximum": 10,
                "description": "1 = barely any signal, 10 = unambiguous hard catalyst",
            },
            "rationale": {
                "type": "string",
                "description": "2-4 sentences. Name the specific headline(s) driving "
                               "the call and any offsetting factor. No price talk.",
            },
            "key_drivers": {
                "type": "array", "items": {"type": "string"},
                "description": "Short phrases: the concrete news items that moved the needle.",
            },
            "material_news": {"type": "boolean"},
        },
        "required": ["direction", "conviction", "rationale", "key_drivers", "material_news"],
    },
}


@dataclass
class NarrativeScore:
    ticker: str
    asof: str
    direction: str
    conviction: int
    rationale: str
    key_drivers: list[str] = field(default_factory=list)
    material_news: bool = True
    headline_count: int = 0
    llm_contaminated: bool = False
    model: str = ANTHROPIC_MODEL
    mocked: bool = False

    @property
    def dir_int(self) -> int:
        return DIR_TO_INT[self.direction]

    @property
    def signal(self) -> int:
        """S_N = direction x conviction  (PLAN §6)."""
        return self.dir_int * self.conviction

    def to_dict(self) -> dict:
        return asdict(self)


def _contaminated(asof: str | date) -> bool:
    return str(asof) <= LLM_KNOWLEDGE_CUTOFF


def _format_headlines(headlines: list[dict]) -> str:
    lines = []
    for i, h in enumerate(headlines, 1):
        src = h.get("source", "?")
        when = h.get("published", "")
        title = h.get("title", "").strip()
        summ = (h.get("summary") or "").strip()
        lines.append(f"[{i}] ({src}, {when}) {title}\n    {summ}".rstrip())
    return "\n".join(lines) if lines else "(no headlines)"


def _mock_score(ticker: str, asof: str, headlines: list[dict]) -> NarrativeScore:
    """Deterministic stand-in when ANTHROPIC_API_KEY is absent.

    Keyword heuristic only — enough to exercise the pipeline and the notebook.
    """
    blob = " ".join((h.get("title", "") + " " + (h.get("summary") or "")).lower()
                     for h in headlines)
    bull = sum(w in blob for w in ("beat", "raises guidance", "surge", "record",
                                   "upgrade", "approval", "wins", "strong demand"))
    bear = sum(w in blob for w in ("miss", "cuts guidance", "downgrade", "probe",
                                   "lawsuit", "recall", "warns", "slump", "ban"))
    if not headlines:
        direction, conviction = "neutral", 1
    elif bull == bear:
        direction, conviction = "neutral", 3
    elif bull > bear:
        direction, conviction = "bull", min(9, 4 + bull - bear)
    else:
        direction, conviction = "bear", min(9, 4 + bear - bull)
    return NarrativeScore(
        ticker=ticker, asof=str(asof), direction=direction, conviction=conviction,
        rationale=f"[MOCK] keyword tally bull={bull} bear={bear} over {len(headlines)} "
                  "headlines. Set ANTHROPIC_API_KEY for a real score.",
        key_drivers=[h.get("title", "")[:80] for h in headlines[:3]],
        material_news=bool(headlines) and (bull + bear) > 0,
        headline_count=len(headlines),
        llm_contaminated=_contaminated(asof),
        mocked=True,
    )


def score_narrative(
    ticker: str,
    headlines: list[dict],
    asof: str | date,
    *,
    force_mock: bool = False,
) -> NarrativeScore:
    asof = str(asof)
    from .llm import client as _client, have_key

    if force_mock or not have_key():
        return _mock_score(ticker, asof, headlines)

    client = _client()
    user = (
        f"Ticker: {ticker}\nUpcoming trading date: {asof}\n\n"
        f"Overnight headlines:\n{_format_headlines(headlines)}"
    )
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM,
        tools=[SCORE_TOOL],
        tool_choice={"type": "tool", "name": "score"},
        messages=[{"role": "user", "content": user}],
    )
    block = next(b for b in resp.content if getattr(b, "type", None) == "tool_use")
    d = block.input
    return NarrativeScore(
        ticker=ticker, asof=asof,
        direction=d["direction"], conviction=int(d["conviction"]),
        rationale=d["rationale"], key_drivers=list(d.get("key_drivers", [])),
        material_news=bool(d["material_news"]),
        headline_count=len(headlines),
        llm_contaminated=_contaminated(asof),
        model=ANTHROPIC_MODEL,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--headlines", help="path to a JSON list of {title,source,summary}")
    args = ap.parse_args()
    hs = json.loads(open(args.headlines).read()) if args.headlines else []
    sc = score_narrative(args.ticker, hs, args.asof)
    print(json.dumps(sc.to_dict(), indent=2))
