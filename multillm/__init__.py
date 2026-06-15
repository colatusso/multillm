"""multillm - multi-LLM reasoning with pluggable backends.

Two architectures:
  - council (ARCH 1): N proposers in parallel + selector (verifier OR judge).
  - debate  (ARCH 2): roles in sequence (divergent -> contrarian -> synthesizer).

Backends per LLM:
  - "claude"     : Claude Code headless (`claude -p`), FULL agent (tools/MCP).
  - "openrouter" : OpenAI-compatible HTTP (Qwen, DeepSeek, etc.).

Each agent's system prompt composes the LLM's voice (base_prompt) with the
role/function instruction (role). The cost of each call is tracked per agent --
see costs.format_cost_report.
"""
from .agent import Agent, build_agent
from .config import Config, LLMSpec, RoleSpec, load_config, parse_config
from .costs import AgentCost, format_cost_report, per_agent_costs, total_cost_usd
from .council import Candidate, council, gather_candidates
from .env import load_env
from .mixture import MixtureResult, mixture
from .prompts import compose_system
from .roles import debate
from .team import Team, agent_from_ref, build_team
from .tracking import format_stats, load_stats, record_result, save_stats
from .usage import Completion, Usage
from .verifier import extract_code, run_python

__all__ = [
    "Agent",
    "build_agent",
    "Config",
    "LLMSpec",
    "RoleSpec",
    "load_config",
    "parse_config",
    "load_env",
    "council",
    "gather_candidates",
    "Candidate",
    "mixture",
    "MixtureResult",
    "debate",
    "Team",
    "build_team",
    "agent_from_ref",
    "record_result",
    "load_stats",
    "save_stats",
    "format_stats",
    "compose_system",
    "Usage",
    "Completion",
    "AgentCost",
    "per_agent_costs",
    "total_cost_usd",
    "format_cost_report",
    "extract_code",
    "run_python",
]
