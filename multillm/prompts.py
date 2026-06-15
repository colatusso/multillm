"""System prompt composition: LLM voice + role (function) instruction.

This is the heart of "per llm x per function": an agent's final prompt is born
from two independent layers -- what the LLM IS (base_prompt) and what it DOES
here (role). Either one may be absent.
"""
from __future__ import annotations


def compose_system(base_prompt: str = "", role_prompt: str = "", *, sep: str = "\n\n") -> str:
    """Joins the LLM's voice (base) with the role prompt.

    Both optional: an agent can run with only the LLM's voice (no role), only the
    role, or both. Empty/blank parts are ignored.

    >>> compose_system("voice", "role")
    'voice\\n\\nrole'
    >>> compose_system("", "role")
    'role'
    """
    parts = [p.strip() for p in (base_prompt, role_prompt) if p and p.strip()]
    return sep.join(parts)
