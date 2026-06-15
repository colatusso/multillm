import pytest

from multillm.config import parse_config
from multillm.team import agent_from_ref, build_team


def _cfg():
    return parse_config({
        "llms": {
            "sonnet": {"backend": "fake", "model": "sonnet", "base_prompt": "voice-s"},
            "opus": {"backend": "fake", "model": "opus", "base_prompt": "voice-o"},
        },
        "roles": {"solver": {"prompt": "solve"}, "judge": {"prompt": "revise"}},
        "team": {
            "proposers": [{"llm": "sonnet", "role": "solver"}, {"llm": "opus", "role": "solver"}],
            "synthesizer": {"llm": "opus", "role": "judge"},
        },
    })


def test_build_team_from_config():
    team = build_team(_cfg())
    assert [p.name for p in team.proposers] == ["sonnet:solver", "opus:solver"]
    assert team.synthesizer.name == "opus:judge"
    assert len(team.agents) == 3          # proposers + synthesizer


def test_agent_from_ref_without_role():
    a = agent_from_ref(_cfg(), {"llm": "opus"})
    assert a.name == "opus"
    assert a.system == "voice-o"             # only the LLM's voice, no role


def test_build_team_missing_block_raises():
    with pytest.raises(KeyError):
        build_team(parse_config({"llms": {}, "roles": {}}))


def test_agent_from_ref_requires_llm():
    with pytest.raises(KeyError):
        agent_from_ref(_cfg(), {"role": "solver"})
