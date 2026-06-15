"""Backend registry.

Adding a new backend = implement Backend, import it here and add 1 line to the
BACKENDS dict. Nothing else in the code needs to change (Open/Closed).
"""
from __future__ import annotations

from .base import Backend, BackendError
from .claude import ClaudeBackend
from .openrouter import OpenRouterBackend

BACKENDS: dict[str, type[Backend]] = {
    "claude": ClaudeBackend,
    "openrouter": OpenRouterBackend,
}


def register_backend(name: str, cls: type[Backend]) -> None:
    """Registers/overrides a backend (also used by tests)."""
    BACKENDS[name] = cls


def build_backend(spec) -> Backend:
    """Builds the backend from an LLMSpec via its `from_spec`."""
    try:
        cls = BACKENDS[spec.backend]
    except KeyError:
        raise BackendError(
            f"unknown backend: '{spec.backend}' (known: {list(BACKENDS)})"
        ) from None
    return cls.from_spec(spec)


__all__ = [
    "Backend",
    "BackendError",
    "BACKENDS",
    "register_backend",
    "build_backend",
    "ClaudeBackend",
    "OpenRouterBackend",
]
