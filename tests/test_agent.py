from multillm.agent import build_agent
from multillm.config import LLMSpec, RoleSpec


def test_composes_system_and_temp_override():
    llm = LLMSpec(name="c", backend="fake", model="m", base_prompt="voz", temperature=0.5)
    role = RoleSpec(name="div", prompt="diverge", temperature=1.0)
    a = build_agent(llm, role)
    assert a.system == "voz\n\ndiverge"
    assert a.temperature == 1.0          # role overrides LLM
    assert a.name == "c:div"


def test_no_role_uses_llm_only():
    llm = LLMSpec(name="c", backend="fake", base_prompt="voz", temperature=0.3)
    a = build_agent(llm)
    assert a.system == "voz"
    assert a.temperature == 0.3
    assert a.name == "c"


async def test_run_returns_text_and_accumulates_usage():
    llm = LLMSpec(name="c", backend="fake", base_prompt="voz",
                  options={"reply": "oi", "cost": 0.01})
    a = build_agent(llm)
    out = await a.run("question")
    assert out == "oi"
    assert len(a.usages) == 1
    assert a.cost_usd == 0.01
    assert a.usages[0].backend == "fake"
