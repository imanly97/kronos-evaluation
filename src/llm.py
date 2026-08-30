"""Shared Anthropic client + a couple of thin call helpers.

Centralised so every call site picks up the same config — notably the
`anthropic-workspace-id` header, which identity-linked API keys require — and so
the agents have one place to get a parsed-JSON or plain-text completion.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from .config import ANTHROPIC_MODEL

MODEL = ANTHROPIC_MODEL


def have_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def client():
    import anthropic

    headers = {}
    ws = os.getenv("ANTHROPIC_WORKSPACE_ID")
    if ws:
        headers["anthropic-workspace-id"] = ws
    return anthropic.Anthropic(default_headers=headers or None)


def complete(system: str, user: str, *, max_tokens: int = 1024,
             model: str | None = None) -> str:
    """Single-turn completion → the assistant's text."""
    msg = client().messages.create(
        model=model or MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def complete_json(system: str, user: str, *, max_tokens: int = 1024,
                  model: str | None = None, retries: int = 1):
    """Completion parsed as JSON. Retries once on a parse failure with a nudge."""
    sys_j = system + "\n\nRespond with a single valid JSON value and nothing else."
    last_err = None
    for attempt in range(retries + 1):
        txt = complete(sys_j, user, max_tokens=max_tokens, model=model)
        txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(txt)
        except json.JSONDecodeError as exc:
            last_err = exc
            m = _JSON_RE.search(txt)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            user = (user + f"\n\nYour previous reply did not parse as JSON "
                    f"({exc}). Return only the JSON value.")
    raise ValueError(f"LLM did not return parseable JSON after {retries + 1} tries: {last_err}")
