"""Usage and cost of an LLM call.

`generate()` returns Completion(text, usage) so that the cost travels along with
the response. The orchestrator (council/debate) only looks at the text via
Agent.run; the Agent accumulates the Usage records for the cost report (see
costs.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usage:
    """Cost/tokens of ONE call. None fields = backend didn't report."""

    backend: str
    model: str
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclass
class Completion:
    """A backend's response: text + usage/cost (+ reasoning, if the model exposes it)."""

    text: str
    usage: Usage
    reasoning: str | None = None
