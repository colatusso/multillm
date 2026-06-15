import pytest

from multillm.config import LLMSpec, parse_config


def test_splits_known_and_options():
    cfg = parse_config({
        "llms": {
            "c": {"backend": "claude", "model": "sonnet", "base_prompt": "x",
                  "tools": ["Read"], "permission_mode": "acceptEdits"},
        },
        "roles": {"solver": {"prompt": "solve"}},  # solver stays per the glossary
    })
    llm = cfg.llm("c")
    assert isinstance(llm, LLMSpec)
    assert (llm.backend, llm.model, llm.base_prompt) == ("claude", "sonnet", "x")
    assert llm.options == {"tools": ["Read"], "permission_mode": "acceptEdits"}
    assert cfg.role("solver").prompt == "solve"


def test_role_temperature_and_missing():
    cfg = parse_config({"roles": {"div": {"prompt": "p", "temperature": 1.0}}})
    assert cfg.role("div").temperature == 1.0
    with pytest.raises(KeyError):
        cfg.llm("nope")
    with pytest.raises(KeyError):
        cfg.role("nope")


def test_llm_requires_backend():
    with pytest.raises(ValueError):
        parse_config({"llms": {"x": {"model": "sonnet"}}})


def test_empty_config():
    cfg = parse_config(None)
    assert cfg.llms == {} and cfg.roles == {}
