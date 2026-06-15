"""ARCH 2 - Roles: sequential debate divergent -> contrarian -> synthesizer (M rounds).

Expensive (serial), but refines ONE chain via critique. Use only on hard tasks.
"""
from __future__ import annotations

from typing import Callable

from .agent import Agent

Verifier = Callable[[str], bool]


async def debate(
    query: str,
    divergent: Agent,
    contrarian: Agent,
    synthesizer: Agent,
    *,
    rounds: int = 2,
    verifier: Verifier | None = None,
) -> tuple[str, bool | None]:
    """Debate that refines a draft via adversarial critique.

    1) divergent generates distinct approaches.
    2) for `rounds`: contrarian attacks the current draft; synthesizer integrates the
       critique into a new draft. The context chains (each round sees the previous draft).

    Returns (final_draft, ok), where ok = verifier(draft) or None if no verifier.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    ideas = await divergent.run(
        f"Problem:\n{query}\n\nPropose 3 quite different approaches, with pros and cons."
    )
    draft = "(no draft yet)"
    for _ in range(rounds):
        crit = await contrarian.run(
            f"Problem:\n{query}\n\nIdeas:\n{ideas}\n\nCurrent draft:\n{draft}\n\n"
            "Attack it: point out holes, breaking cases, and wrong assumptions. Do not soften it."
        )
        draft = await synthesizer.run(
            f"Problem:\n{query}\n\nIdeas:\n{ideas}\n\nCritiques:\n{crit}\n\nPrevious draft:\n{draft}\n\n"
            "Integrate everything into a final, coherent and correct answer."
        )
    ok = verifier(draft) if verifier else None
    return draft, ok
