from types import SimpleNamespace

import pytest

from multillm.backends.openrouter import OpenRouterBackend
from multillm.config import LLMSpec


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # from_spec needs a non-empty key


def _spec(**options):
    return LLMSpec(name="m", backend="openrouter", model="vendor/model", options=options)


def test_provider_routing_read_from_options():
    b = OpenRouterBackend.from_spec(_spec(provider={"sort": "price"}))
    assert b.provider == {"sort": "price"}        # forwarded to the request body in generate()


def test_provider_defaults_to_none():
    b = OpenRouterBackend.from_spec(_spec(reasoning={"effort": "high"}))
    assert b.provider is None                      # absent -> OpenRouter's default routing (back-compat)


def test_reasoning_string_is_normalized():
    b = OpenRouterBackend.from_spec(_spec(reasoning="high"))
    assert b.reasoning == {"effort": "high"}       # "high" -> {"effort": "high"}


def test_reasoning_and_provider_together():
    b = OpenRouterBackend.from_spec(_spec(reasoning="high", provider={"order": ["DeepSeek"]},
                                          max_tokens=12345))
    assert b.reasoning == {"effort": "high"}
    assert b.provider == {"order": ["DeepSeek"]}
    assert b.max_tokens == 12345


def test_empty_api_key_raises():
    from multillm.backends.base import BackendError
    with pytest.raises(BackendError):
        OpenRouterBackend(model="x", api_key="")


# --- generate(): prove provider/reasoning are forwarded in extra_body (no network) ---
class _FakeMsg:
    content = "ok"
    reasoning = None


class _FakeUsage:
    cost = 0.01
    prompt_tokens = 1
    completion_tokens = 2


class _FakeResp:
    choices = [type("C", (), {"message": _FakeMsg()})()]
    usage = _FakeUsage()


class _FakeClient:
    def __init__(self):
        self.captured = {}

        async def create(**kw):                  # instance attribute -> no `self` bind
            self.captured.update(kw)
            return _FakeResp()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


async def test_generate_forwards_provider_and_reasoning():
    b = OpenRouterBackend.from_spec(_spec(provider={"sort": "price"}, reasoning="high"))
    b._client = _FakeClient()
    await b.generate(system="s", user="u")
    eb = b._client.captured["extra_body"]
    assert eb["provider"] == {"sort": "price"}
    assert eb["reasoning"] == {"effort": "high"}
    assert eb["usage"] == {"include": True}


async def test_generate_omits_provider_when_none():
    b = OpenRouterBackend.from_spec(_spec())          # no provider in options
    b._client = _FakeClient()
    await b.generate(system="s", user="u")
    assert "provider" not in b._client.captured["extra_body"]   # back-compat: body untouched
