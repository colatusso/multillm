"""Builds teams of agents from config blocks (yaml): `team`, `debate`.

A "ref" is just a dict {llm: <name>, role: <optional name>}. This way the whole
team (who proposes, who judges, who debates) lives in the yaml -- no model
hardcoded in the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agent import Agent, build_agent
from .config import Config


def agent_from_ref(cfg: Config, ref: dict) -> Agent:
    """Builds an Agent from {llm: ..., role: ... (optional)}."""
    if "llm" not in ref:
        raise KeyError(f"agent ref missing 'llm': {ref}")
    role = cfg.role(ref["role"]) if ref.get("role") else None
    return build_agent(cfg.llm(ref["llm"]), role)


@dataclass
class Team:
    proposers: list[Agent] = field(default_factory=list)
    synthesizer: Agent | None = None

    @property
    def agents(self) -> list[Agent]:
        """All agents on the team (for the cost report)."""
        return self.proposers + ([self.synthesizer] if self.synthesizer else [])


def build_team(cfg: Config, name: str = "team") -> Team:
    """Reads the `team` block from the yaml and builds proposers + synthesizer."""
    spec = cfg.raw.get(name)
    if not spec:
        raise KeyError(f"block '{name}' not found in config (yaml)")
    proposers = [agent_from_ref(cfg, r) for r in spec.get("proposers", [])]
    s = spec.get("synthesizer")
    return Team(proposers=proposers, synthesizer=agent_from_ref(cfg, s) if s else None)
