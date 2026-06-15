from multillm.agent import Agent
from multillm.costs import format_cost_report, per_agent_costs, total_cost_usd
from multillm.usage import Usage


def _agent(name, costs, backend="openrouter"):
    a = Agent(name, backend=None, system="s")
    for c in costs:
        a.usages.append(Usage(backend=backend, model="m", cost_usd=c, input_tokens=10, output_tokens=5))
    return a


def test_total_and_per_agent_aggregation():
    a = _agent("a", [0.01, 0.02])
    b = _agent("b", [0.005])
    agents = [a, b]
    assert round(total_cost_usd(agents), 4) == 0.035
    lines = per_agent_costs(agents)
    la = next(l for l in lines if l.name == "a")
    assert la.calls == 2 and round(la.cost_usd, 4) == 0.03
    assert (la.input_tokens, la.output_tokens) == (20, 10)
    assert la.costed is True


def test_uncosted_agent_is_marked():
    a = Agent("x", backend=None, system="s")
    a.usages.append(Usage(backend="claude", model="sonnet", cost_usd=None, input_tokens=3, output_tokens=4))
    assert per_agent_costs([a])[0].costed is False
    assert total_cost_usd([a]) == 0.0
    assert "(no cost)" in format_cost_report([a])


def test_unused_agents_skipped():
    a = Agent("unused", backend=None, system="s")   # no usages
    assert per_agent_costs([a]) == []
    assert "(no calls recorded)" in format_cost_report([a])
