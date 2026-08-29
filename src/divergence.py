"""Divergence adjudicator.

The *classification* is deterministic and grounded entirely in the two signals'
numbers (PLAN §6). The *brief* is an LLM call that must stay inside those numbers
— it explains the disagreement to a PM, it does not add a new opinion.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .config import (
    ANTHROPIC_MODEL,
    DIV_DIRECTIONAL_MIN_CONVICTION,
    DIV_MAGNITUDE_MAX_PROB,
    DIV_MAGNITUDE_MIN_CONVICTION,
)

if TYPE_CHECKING:
    from .kronos_infer import KronosForecast
    from .narrative import NarrativeScore


@dataclass
class DivergenceResult:
    ticker: str
    asof: str
    narr_signal: int              # S_N = dir x conviction
    narr_dir: int
    struct_dir: int               # sign(median r)
    struct_median: float
    struct_strength: float
    struct_p_up: float
    p_narr_direction: float        # P(return in narrative's sign) under Kronos
    is_directional: bool
    is_magnitude: bool
    divergence_type: str          # aligned | directional | magnitude | directional+magnitude
    brief: str = ""

    @property
    def diverges(self) -> bool:
        return self.is_directional or self.is_magnitude

    def to_dict(self) -> dict:
        return asdict(self)


def classify_divergence(narrative: "NarrativeScore", fc: "KronosForecast") -> DivergenceResult:
    n_dir = narrative.dir_int
    s_median = fc.median_ret
    s_dir = 0 if s_median == 0 else (1 if s_median > 0 else -1)

    p_narr = fc.prob_direction(n_dir) if n_dir != 0 else fc.prob_direction(0)

    is_dir = (
        n_dir != 0
        and s_dir != 0
        and n_dir != s_dir
        and narrative.conviction >= DIV_DIRECTIONAL_MIN_CONVICTION
    )
    is_mag = (
        n_dir != 0
        and narrative.conviction >= DIV_MAGNITUDE_MIN_CONVICTION
        and p_narr < DIV_MAGNITUDE_MAX_PROB
    )

    if is_dir and is_mag:
        dtype = "directional+magnitude"
    elif is_dir:
        dtype = "directional"
    elif is_mag:
        dtype = "magnitude"
    else:
        dtype = "aligned"

    return DivergenceResult(
        ticker=fc.ticker, asof=str(fc.asof),
        narr_signal=narrative.signal, narr_dir=n_dir,
        struct_dir=s_dir, struct_median=s_median,
        struct_strength=fc.strength, struct_p_up=fc.p_up,
        p_narr_direction=p_narr,
        is_directional=is_dir, is_magnitude=is_mag,
        divergence_type=dtype,
    )


BRIEF_SYSTEM = """You write a one-paragraph pre-market brief for a portfolio \
manager. You are given two independent signals on one stock for today's session:

1. STRUCTURE — a price-only probabilistic forecast (it has seen only historical \
price/volume, nothing else).
2. NARRATIVE — a price-blind read of overnight news.

Write <=120 words. State what each signal says, where they agree or disagree, \
and what would resolve it at the close. Ground every claim in the numbers given. \
Do not add your own market view, do not give a recommendation, do not mention \
data the brief was not given. Plain, direct sentences."""


def write_brief(div: DivergenceResult, narrative: "NarrativeScore",
                fc: "KronosForecast") -> str:
    facts = (
        f"Ticker {div.ticker}, session {div.asof}.\n"
        f"STRUCTURE (Kronos, {len(fc.ret_dist)} sampled paths): "
        f"median close-to-close {div.struct_median * 100:+.2f}%, "
        f"P(up) {fc.p_up:.0%}, strength {div.struct_strength:+.2f} sd, "
        f"5-95% band [{fc.ret_quantiles()[0.05] * 100:+.2f}%, "
        f"{fc.ret_quantiles()[0.95] * 100:+.2f}%].\n"
        f"NARRATIVE: direction {narrative.direction}, conviction "
        f"{narrative.conviction}/10 over {narrative.headline_count} headlines. "
        f"Rationale: {narrative.rationale}\n"
        f"Key drivers: {', '.join(narrative.key_drivers) or 'n/a'}\n"
        f"Divergence classification: {div.divergence_type}. "
        f"P(move in narrative's direction) under structure: {div.p_narr_direction:.0%}."
    )

    from .llm import client as _client, have_key

    if not have_key():
        agree = "disagree" if div.diverges else "broadly agree"
        return (
            f"[MOCK BRIEF] Structure and narrative {agree} on {div.ticker} for "
            f"{div.asof}. Structure: median {div.struct_median * 100:+.2f}%, "
            f"P(up) {fc.p_up:.0%}. Narrative: {narrative.direction} at "
            f"{narrative.conviction}/10. Type: {div.divergence_type}. "
            f"Structure assigns {div.p_narr_direction:.0%} probability to the "
            f"narrative's direction. Resolves at today's close."
        )

    resp = _client().messages.create(
        model=ANTHROPIC_MODEL, max_tokens=400, system=BRIEF_SYSTEM,
        messages=[{"role": "user", "content": facts}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


def assess(narrative: "NarrativeScore", fc: "KronosForecast", *, brief: bool = True
           ) -> DivergenceResult:
    div = classify_divergence(narrative, fc)
    if brief:
        div.brief = write_brief(div, narrative, fc)
    return div
