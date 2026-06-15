"""Backend contract: takes (system, user) and returns text."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BackendError(RuntimeError):
    """Failure generating a response in a backend (network, subprocess, auth, etc.)."""


class Backend(ABC):
    """Generates a text response from a system + user prompt.

    Uniform interface by design: extra capabilities (tools/MCP in the Claude
    backend, temperature in OpenRouter) live in the backend's CONFIG, not in the
    signature -- the orchestrator (council/debate) doesn't need to know the difference.

    Built from an LLMSpec via `from_spec` (Open/Closed: a new backend
    = a new class + 1 line in the registry, without touching the others).
    """

    @classmethod
    @abstractmethod
    def from_spec(cls, spec) -> "Backend":
        ...

    @abstractmethod
    async def generate(self, *, system: str, user: str, temperature: float | None = None) -> str:
        ...
