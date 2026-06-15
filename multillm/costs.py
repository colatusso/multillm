"""Per-agent cost aggregation.

Each Agent accumulates its Usage records in `agent.usages`. Here we sum per
agent and overall. Uses duck typing (agent.name / agent.usages) to avoid
coupling to Agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class AgentCost:
    name: str
    backend: str
    model: str
    calls: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    costed: bool  # did any Usage carry a cost? (False = backend didn't report)


def per_agent_costs(agents: Iterable["Agent"]) -> list[AgentCost]:
    out: list[AgentCost] = []
    for a in agents:
        usages = getattr(a, "usages", [])
        if not usages:
            continue
        first = usages[0]
        out.append(
            AgentCost(
                name=a.name,
                backend=first.backend,
                model=first.model,
                calls=len(usages),
                cost_usd=sum(u.cost_usd or 0.0 for u in usages),
                input_tokens=sum(u.input_tokens or 0 for u in usages),
                output_tokens=sum(u.output_tokens or 0 for u in usages),
                costed=any(u.cost_usd is not None for u in usages),
            )
        )
    return out


def total_cost_usd(agents: Iterable["Agent"]) -> float:
    return sum(u.cost_usd or 0.0 for a in agents for u in getattr(a, "usages", []))


def format_cost_report(agents: Iterable["Agent"]) -> str:
    agents = list(agents)
    lines = per_agent_costs(agents)
    if not lines:
        return "Costs: (no calls recorded)"
    w = max(len(l.name) for l in lines)
    rows = ["Costs per agent:"]
    for l in lines:
        cost = f"${l.cost_usd:.4f}" if l.costed else "(no cost)"
        rows.append(
            f"  {l.name:<{w}}  {l.backend:<10} {l.calls:>2} call(s)  {cost:>11}  "
            f"in={l.input_tokens} out={l.output_tokens}"
        )
    rows.append(f"  {'TOTAL':<{w}}  {'':<10} {'':>2}           ${total_cost_usd(agents):.4f}")
    return "\n".join(rows)
