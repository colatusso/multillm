import pytest

from multillm.__main__ import _adhoc_agent, _adhoc_team, _split_spec
from multillm.config import parse_config


@pytest.fixture(autouse=True)
def _or_key(monkeypatch):
    # OpenRouterBackend refuses an empty key; a dummy is enough (it only instantiates
    # the client here, no network call happens in these tests).
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def _cfg():
    return parse_config({
        "roles": {"solver": {"prompt": "solve"}, "judge": {"prompt": "revise"},
                  "generator": {"prompt": "invent"}},
        "team": {"synthesizer": {"llm": "opus", "role": "judge"}},
        "llms": {"opus": {"backend": "fake", "model": "opus", "base_prompt": "voice"}},
    })


def test_split_spec_default_role():
    assert _split_spec("qwen/qwen3-max", "solver") == ("qwen/qwen3-max", "solver")


def test_split_spec_explicit_role():
    assert _split_spec("qwen/qwen3-max@generator", "solver") == ("qwen/qwen3-max", "generator")


def test_split_spec_keeps_openrouter_variant_colon():
    # ':free' / ':nitro' belong to the id -- must NOT be parsed as the role
    assert _split_spec("deepseek/deepseek-r1:free", "solver") == ("deepseek/deepseek-r1:free", "solver")
    assert _split_spec("x/y:nitro@judge", "solver") == ("x/y:nitro", "judge")


def test_adhoc_agent_uses_id_as_model_and_role_prompt():
    a = _adhoc_agent(_cfg(), "qwen/qwen3-max", "solver", "high")
    assert a.name == "qwen/qwen3-max:solver"
    assert a.backend.model == "qwen/qwen3-max"       # id used as-is
    assert a.backend.reasoning == {"effort": "high"}  # effort applied (str -> {"effort": ...})
    assert a.system == "solve"                        # role prompt from yaml, no llm voice


class _Args:
    def __init__(self, **kw):
        self.models = kw.get("models", "")
        self.role = kw.get("role", "solver")
        self.judge = kw.get("judge", "")
        self.effort = kw.get("effort", "high")


def test_adhoc_team_ignores_yaml_proposers_judge_from_yaml():
    team = _adhoc_team(_cfg(), _Args(models="qwen/qwen3-max,deepseek/deepseek-r1@generator"))
    assert [p.name for p in team.proposers] == ["qwen/qwen3-max:solver", "deepseek/deepseek-r1:generator"]
    assert team.synthesizer.name == "opus:judge"      # judge falls back to the yaml synthesizer


def test_adhoc_team_explicit_judge():
    team = _adhoc_team(_cfg(), _Args(models="qwen/qwen3-max", judge="x/y-judge"))
    assert team.synthesizer.name == "x/y-judge:judge"  # --judge overrides, default role 'judge'


def test_adhoc_team_blank_models_yields_no_proposers():
    team = _adhoc_team(_cfg(), _Args(models=" , , "))   # trailing/blank entries are filtered
    assert team.proposers == []


def test_adhoc_agent_unknown_role_raises():
    with pytest.raises(KeyError):                        # clear error, caught cleanly in main()
        _adhoc_agent(_cfg(), "qwen/q@nope", "solver", "high")
