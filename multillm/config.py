"""Declarative config (YAML) -> typed specs.

The YAML separates reusable blocks:
  llms:  each model's voice/backend (base_prompt + backend options).
  roles: the instruction for each brainstorm role/function.

Unrecognized fields of an LLM fall into `options` and are forwarded to the
backend (e.g. tools, permission_mode, mcp_config). This way, adding a new flag
doesn't require touching the parser or an if/elif -- the backend decides what to
read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Fields read directly into LLMSpec; the rest of the YAML dict becomes `options`.
_LLM_FIELDS = {"backend", "model", "base_prompt", "temperature"}


@dataclass
class RoleSpec:
    """A brainstorm role/function. `temperature` (if set) overrides the LLM's."""

    name: str
    prompt: str = ""
    temperature: float | None = None


@dataclass
class LLMSpec:
    """A model's voice/backend. `options` holds backend-specific flags."""

    name: str
    backend: str
    model: str = ""
    base_prompt: str = ""
    temperature: float | None = None
    options: dict = field(default_factory=dict)


@dataclass
class Config:
    llms: dict[str, LLMSpec] = field(default_factory=dict)
    roles: dict[str, RoleSpec] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def llm(self, name: str) -> LLMSpec:
        try:
            return self.llms[name]
        except KeyError:
            raise KeyError(f"LLM '{name}' not defined (llms: {list(self.llms)})") from None

    def role(self, name: str) -> RoleSpec:
        try:
            return self.roles[name]
        except KeyError:
            raise KeyError(f"role '{name}' not defined (roles: {list(self.roles)})") from None


def _llm_from_dict(name: str, spec: dict) -> LLMSpec:
    if "backend" not in spec:
        raise ValueError(f"LLM '{name}' missing required field 'backend'")
    known = {k: v for k, v in spec.items() if k in _LLM_FIELDS}
    options = {k: v for k, v in spec.items() if k not in _LLM_FIELDS}
    return LLMSpec(name=name, options=options, **known)


def parse_config(data: dict | None) -> Config:
    data = data or {}
    llms = {n: _llm_from_dict(n, s or {}) for n, s in (data.get("llms") or {}).items()}
    roles = {n: RoleSpec(name=n, **(s or {})) for n, s in (data.get("roles") or {}).items()}
    return Config(llms=llms, roles=roles, raw=data)


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return parse_config(data)
