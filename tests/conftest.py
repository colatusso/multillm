"""Test fixtures: fake backends (no network/subprocess).

FakeBackend is deterministic and records each call in `.calls`, returning
Completion (text + Usage) -- only the external API (LLM) is fake; the
council/debate/costs logic runs for real.
"""
from __future__ import annotations

import pytest

from multillm.backends import Backend, register_backend
from multillm.usage import Completion, Usage


class FakeBackend(Backend):
    def __init__(self, *, reply="answer", script=None, model="fake-model", cost=None):
        self.model = model
        self._reply = reply
        self._script = list(script) if script else None
        self._cost = cost
        self.calls: list[dict] = []

    @classmethod
    def from_spec(cls, spec):
        o = spec.options
        return cls(
            reply=o.get("reply", "answer"),
            script=o.get("script"),
            model=spec.model or "fake-model",
            cost=o.get("cost"),
        )

    async def generate(self, *, system, user, temperature=None):
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if self._script is not None:
            text = self._script[min(len(self.calls) - 1, len(self._script) - 1)]
        else:
            text = self._reply
        usage = Usage(backend="fake", model=self.model, cost_usd=self._cost,
                      input_tokens=1, output_tokens=1)
        return Completion(text=text, usage=usage)


class FailingBackend(Backend):
    @classmethod
    def from_spec(cls, spec):
        return cls()

    async def generate(self, *, system, user, temperature=None):
        raise RuntimeError("simulated failure")


register_backend("fake", FakeBackend)


@pytest.fixture
def Fake():
    return FakeBackend


@pytest.fixture
def Failing():
    return FailingBackend
