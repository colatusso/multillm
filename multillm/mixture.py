"""Mixture-of-Agents: N proposers propose, 1 synthesizer JUDGES and RAISES.

The synthesizer (judge) reads all the answers and returns a structured JSON:
  ranking       -> indices from best->worst (basis for tracking)
  observations  -> what it noticed (correct points, errors, why it ranked this way) — meta, goes to stderr
  answer        -> final STANDALONE answer for the human (does NOT mention the candidates
                   or the process; it is the clean answer with what the judge deemed correct)

Proposers that failed/came back empty do not enter the synthesis, but stay in `dropped`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from .agent import Agent
from .council import Candidate, gather_all

_SYNTH_USER = (
    "Original question:\n{query}\n\n"
    "Below are {n} answers from different models to the SAME question:\n\n{listing}\n\n"
    "Reply ONLY with a valid JSON object (nothing outside it, no ``` fences), "
    "with exactly these keys:\n"
    '- "ranking": list of the answer indices, from BEST to worst (e.g. [2,0,1]).\n'
    '- "observations": what you noticed — correct points, errors, and why you ranked it this way.\n'
    '- "answer": the final answer for the user. It must NOT mention the answers '
    "you received or the judging process — it is a STANDALONE answer, complete and clear, "
    "with what you deemed correct (including what only you know). Human-friendly, no fluff."
)


def _extract_json(raw: str) -> dict | None:
    """Tries to find a JSON object in the text: raw, inside ```...```, or the 1st {...}."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _clean_ranking(value, n: int) -> list[int] | None:
    if not isinstance(value, list):
        return None
    seen: set[int] = set()
    out: list[int] = []
    for v in value:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
    return out or None


def _parse_judge(raw: str, n: int) -> tuple[list[int] | None, str, str]:
    """(ranking, observations, answer). If there is no JSON, the raw text becomes the answer."""
    obj = _extract_json(raw)
    if obj is None:
        return None, "", raw.strip()
    ranking = _clean_ranking(obj.get("ranking"), n)
    obs = str(obj.get("observations") or "").strip()
    resp = str(obj.get("answer") or "").strip() or raw.strip()
    return ranking, obs, resp


@dataclass
class MixtureResult:
    final: str                                  # judge's "answer" field (standalone)
    ranking: list[int] | None                   # indices from best->worst
    observations: str = ""                      # judge's "observations" field (meta)
    candidates: list[Candidate] = field(default_factory=list)
    dropped: list[Candidate] = field(default_factory=list)

    def ranked(self) -> list[Candidate]:
        return [self.candidates[i] for i in (self.ranking or []) if 0 <= i < len(self.candidates)]

    def winner(self) -> Candidate | None:
        r = self.ranked()
        return r[0] if r else None


async def mixture(query: str, proposers: Sequence[Agent], synthesizer: Agent,
                  *, on_proposer=None, on_judge=None) -> MixtureResult:
    """Run the proposers; the judge returns JSON (ranking/observations/answer).

    Live progress (optional): on_proposer(done, total, candidate) on each
    proposer completed; on_judge() when the judge starts evaluating.
    Each agent's cost stays in agent.usages, as always.
    """
    allc = await gather_all(query, proposers, on_done=on_proposer)
    cands = [c for c in allc if c.ok]
    dropped = [c for c in allc if not c.ok]
    if not cands:
        errs = [f"{c.agent}: {c.error or 'empty answer'}" for c in allc]
        raise RuntimeError(f"all proposers failed: {errs}")

    if on_judge is not None:
        on_judge()
    # BLIND judging: the judge sees only the index, NEVER the model name -- this
    # avoids self-preference / label bias (an LLM judge favoring its own family).
    # index -> model is remapped deterministically afterwards from `cands` order
    # (ranking, tracking and the cost report all read the model from there).
    listing = "\n\n".join(f"[{i}]\n{c.text}" for i, c in enumerate(cands))
    raw = await synthesizer.run(_SYNTH_USER.format(query=query, n=len(cands), listing=listing))
    ranking, observations, final = _parse_judge(raw, len(cands))
    return MixtureResult(final=final, ranking=ranking, observations=observations,
                         candidates=cands, dropped=dropped)
