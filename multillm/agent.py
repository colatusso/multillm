"""Agent = backend + composed system (LLM voice + role) + accumulated cost."""
from __future__ import annotations

from dataclasses import dataclass, field

from .backends import Backend, build_backend
from .config import LLMSpec, RoleSpec
from .prompts import compose_system
from .usage import Completion, Usage


@dataclass
class Agent:
    name: str
    backend: Backend
    system: str
    temperature: float | None = None
    usages: list[Usage] = field(default_factory=list)

    async def run_completion(self, user: str) -> Completion:
        """Generates, accumulates the cost and returns the full Completion (text+reasoning+usage)."""
        c = await self.backend.generate(system=self.system, user=user, temperature=self.temperature)
        self.usages.append(c.usage)
        return c

    async def run(self, user: str) -> str:
        """Like run_completion, but returns only the TEXT (council/debate use this)."""
        return (await self.run_completion(user)).text

    @property
    def cost_usd(self) -> float:
        return sum(u.cost_usd or 0.0 for u in self.usages)


def build_agent(llm: LLMSpec, role: RoleSpec | None = None, *, name: str | None = None) -> Agent:
    """Builds an agent combining the LLM's voice with a role (optional).

    system      = base_prompt(LLM) + prompt(role)
    temperature = the role's, if any; otherwise the LLM's
    name        = "<llm>:<role>" (or just "<llm>" without a role), unless overridden.
    """
    backend = build_backend(llm)
    system = compose_system(llm.base_prompt, role.prompt if role else "")
    temperature = role.temperature if role and role.temperature is not None else llm.temperature
    label = name or (f"{llm.name}:{role.name}" if role else llm.name)
    return Agent(name=label, backend=backend, system=system, temperature=temperature)
