"""Shared Anthropic client factory.

Centralised so every call site picks up the same config — notably the
`anthropic-workspace-id` header, which identity-linked API keys require.
"""
from __future__ import annotations

import os
from functools import lru_cache


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
