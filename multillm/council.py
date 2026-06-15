"""ARCH 1 - Council: N proposers in parallel + selector (verifier OR judge).

Cheap: it SELECTS, it does not synthesize. Latency ~= 1 call (parallel), cost N x.
To RAISE above the best answer (synthesize + rank), see mixture.py.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .agent import Agent

Verifier = Callable[[str], bool]


@dataclass
class Candidate:
    """Answer from a proposer, with where it came from (for ranking and tracking)."""

    agent: str                  # agent name (e.g. "qwen:solver")
    model: str                  # effective model id
    text: str
    reasoning: str | None = None
    error: str | None = None    # filled in if the call failed

    @property
    def ok(self) -> bool:
        """Usable answer: no error and non-empty text."""
        return self.error is None and bool(self.text.strip())


async def gather_all(query: str, proposers: Sequence[Agent], *, on_done=None) -> list[Candidate]:
    """Run ALL proposers in parallel and return one Candidate per proposer.

    Never discards anything: a failure becomes Candidate.error, an empty answer
    becomes ok=False. The consumer decides what to do (mixture splits into
    candidates/dropped).

    on_done(done, total, candidate): optional callback invoked when EACH proposer
    finishes (for live progress). The result keeps the original order of the proposers;
    only the callback fires in completion order.
    """
    if not proposers:
        raise ValueError("need at least 1 proposer")

    total = len(proposers)
    done = 0

    async def _one(p: Agent) -> Candidate:
        try:
            c = await p.run_completion(query)
            cand = Candidate(agent=p.name, model=c.usage.model, text=c.text, reasoning=c.reasoning)
        except Exception as e:  # caught here so as not to take down the other proposers
            cand = Candidate(agent=p.name, model=getattr(p.backend, "model", "?"),
                             text="", error=str(e)[:200])
        nonlocal done
        done += 1
        if on_done is not None:
            on_done(done, total, cand)
        return cand

    return list(await asyncio.gather(*(_one(p) for p in proposers)))


async def gather_candidates(query: str, proposers: Sequence[Agent]) -> list[Candidate]:
    """Only the usable candidates (ok). Raises if no proposer answered."""
    allc = await gather_all(query, proposers)
    cands = [c for c in allc if c.ok]
    if not cands:
        errs = [f"{c.agent}: {c.error or 'empty answer'}" for c in allc]
        raise RuntimeError(f"all proposers failed: {errs}")
    return cands


async def council(
    query: str,
    proposers: Sequence[Agent],
    *,
    verifier: Verifier | None = None,
    judge: Agent | None = None,
) -> tuple[str, str]:
    """Run the proposers in parallel and SELECT one answer (does not rewrite).

    - verifier (verifiable domain): the 1st candidate that passes; otherwise the first
      (how="verifier-fallback"). Grounded and cheap selection.
    - judge (non-verifiable): the judge agent returns only the index of the best.
    - no selector: returns the first candidate.

    Returns (text, how). `verifier` takes priority over `judge`.
    """
    cands = await gather_candidates(query, proposers)

    if verifier is not None:
        for c in cands:
            if verifier(c.text):
                return c.text, "verifier"
        return cands[0].text, "verifier-fallback"

    if judge is not None:
        listing = "\n\n".join(f"[{i}]\n{c.text}" for i, c in enumerate(cands))
        raw = await judge.run(
            f"Question:\n{query}\n\nCandidates:\n{listing}\n\nReply with only the number of the best one."
        )
        m = re.search(r"\d+", raw or "")
        idx = int(m.group()) if m else 0
        idx = max(0, min(idx, len(cands) - 1))
        return cands[idx].text, f"judge->{idx}"

    return cands[0].text, "no-selector"
