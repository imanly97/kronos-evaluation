"""Groundedness critic for triage briefs (PLAN §5.2).

Two passes over a draft brief:

  1. deterministic — pull every number out of the prose and check each one sits
     within tolerance of a value the deterministic core actually computed for
     this name. This is the backbone of the "100% of numeric claims traceable"
     eval, so it must be reproducible.
  2. LLM — read the brief against the same computed values and flag overclaims,
     a direction that contradicts the forecast, or a missing confidence caveat.

`review()` returns a `CriticReport`; the triage graph loops draft→review until it
passes or `CRITIC_MAX_ROUNDS` is hit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import CRITIC_MAX_ROUNDS
from .llm import complete_json, have_key

# a number, optionally signed, optionally a percentage, optionally an 'x' multiple
_NUM_RE = re.compile(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*(%|x|bps)?", re.IGNORECASE)
_REL_TOL = 0.15      # 15% relative, or...
_ABS_TOL = 0.15      # ...0.15 absolute (percentage points / raw), whichever is looser
_IGNORE = {0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 50.0, 100.0}   # structural / vocabulary numbers


@dataclass
class CriticReport:
    ok: bool
    n_numbers: int
    n_grounded: int
    ungrounded: list[str] = field(default_factory=list)
    llm_issues: list[str] = field(default_factory=list)
    missing_caveat: bool = False

    @property
    def groundedness(self) -> float:
        return 1.0 if self.n_numbers == 0 else self.n_grounded / self.n_numbers

    def feedback(self) -> str:
        bits = []
        if self.ungrounded:
            bits.append("Numbers in the brief that match no computed value "
                        f"(remove or correct them): {'; '.join(self.ungrounded)}")
        if self.missing_caveat:
            bits.append("Add one explicit one-line confidence caveat.")
        for i in self.llm_issues:
            bits.append(i)
        return "\n".join(f"- {b}" for b in bits)


def _candidates(value: float) -> set[float]:
    """The numeric forms a brief might legitimately use for one computed value."""
    out = {value, -value}
    out |= {value * 100, -value * 100}       # 0.0146 -> 1.46 (%)
    out |= {round(value, 2), round(value * 100, 2)}
    return out


def _grounded(num: float, unit: str, allowed: list[float]) -> bool:
    targets = {num}
    if unit == "%":
        targets.add(num / 100.0)
    for t in targets:
        for a in allowed:
            if abs(t - a) <= max(_ABS_TOL, _REL_TOL * abs(a)):
                return True
            if abs(t) > 1 and abs(a) > 1 and abs(t - a) <= _REL_TOL * abs(a):
                return True
    return False


def check_numbers(brief_md: str, values: dict[str, float]) -> tuple[int, int, list[str]]:
    allowed: list[float] = []
    for v in values.values():
        if v is None:
            continue
        try:
            allowed.extend(_candidates(float(v)))
        except (TypeError, ValueError):
            continue

    n_total = n_ok = 0
    bad: list[str] = []
    for m in _NUM_RE.finditer(brief_md):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        try:
            num = float(raw)
        except ValueError:
            continue
        if unit in ("", None) and num in _IGNORE:
            continue
        n_total += 1
        if _grounded(num, unit, allowed):
            n_ok += 1
        else:
            ctx = brief_md[max(0, m.start() - 30):m.end() + 15].replace("\n", " ")
            bad.append(f"…{ctx.strip()}…")
    return n_total, n_ok, bad


_CAVEAT_HINTS = ("caveat", "confidence", "uncertain", "low convict", "tentative",
                 "wide dispersion", "small sample", "thin", "speculative")


def _has_caveat(brief_md: str) -> bool:
    low = brief_md.lower()
    return any(h in low for h in _CAVEAT_HINTS)


def _llm_pass(brief_md: str, values: dict[str, float], setup: str) -> list[str]:
    if not have_key():
        return []
    facts = "\n".join(f"  {k} = {v}" for k, v in values.items() if v is not None)
    system = (
        "You are a groundedness critic on a research desk. You are given the "
        "computed facts for one stock and a draft morning brief about it. "
        "Flag ONLY: (a) a claim that contradicts the computed facts, (b) an "
        "overclaim the facts do not support (certainty language, a causal story, "
        "a price target), (c) a stated direction opposite to the forecast median. "
        "Do NOT nitpick style. If the brief is clean, return an empty list."
    )
    user = (f"SETUP: {setup}\n\nCOMPUTED FACTS:\n{facts}\n\nDRAFT BRIEF:\n{brief_md}\n\n"
            'Return JSON: {"issues": ["short issue", ...]}')
    try:
        out = complete_json(system, user, max_tokens=500)
        return [str(i) for i in out.get("issues", [])][:5]
    except Exception:  # noqa: BLE001 — critic must never crash triage
        return []


def review(brief_md: str, values: dict[str, float], *, setup: str = "",
           require_caveat: bool = True) -> CriticReport:
    n_total, n_ok, bad = check_numbers(brief_md, values)
    missing_caveat = require_caveat and not _has_caveat(brief_md)
    llm_issues = _llm_pass(brief_md, values, setup)
    ok = (not bad) and (not missing_caveat) and (not llm_issues)
    return CriticReport(
        ok=ok, n_numbers=n_total, n_grounded=n_ok, ungrounded=bad,
        llm_issues=llm_issues, missing_caveat=missing_caveat,
    )


MAX_ROUNDS = CRITIC_MAX_ROUNDS
