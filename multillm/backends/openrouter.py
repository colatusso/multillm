"""OpenRouter backend (OpenAI-compatible HTTP): Qwen, DeepSeek, etc."""
from __future__ import annotations

import os

from openai import AsyncOpenAI

from ..usage import Completion, Usage
from .base import Backend, BackendError

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterBackend(Backend):
    """Text-gen via OpenAI-compatible API. `temperature` controls diversity.

    - Cost: requests `usage: {include: true}` -> `usage.cost` (USD) with no extra call.
    - Reasoning: if `reasoning` is set (e.g. {"effort": "high"}), it goes in the body and
      the returned `message.reasoning` is captured. Models without support ignore it.
      Note: reasoning consumes tokens -- you may need a larger `max_tokens`.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = OPENROUTER_BASE,
        default_temperature: float = 0.7,
        max_tokens: int = 32000,  # high: reasoning (effort high) + long response must not overflow -> empty content
        reasoning: dict | None = None,
    ):
        if not api_key:
            raise BackendError("OpenRouter: empty api_key (set OPENROUTER_API_KEY)")
        self.model = model
        self.default_temperature = default_temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @classmethod
    def from_spec(cls, spec) -> "OpenRouterBackend":
        o = spec.options
        reasoning = o.get("reasoning")
        if isinstance(reasoning, str):            # "high" -> {"effort": "high"}
            reasoning = {"effort": reasoning}
        return cls(
            spec.model,
            api_key=os.environ.get(o.get("api_key_env", "OPENROUTER_API_KEY"), ""),
            base_url=o.get("base_url", OPENROUTER_BASE),
            default_temperature=spec.temperature if spec.temperature is not None else 0.7,
            max_tokens=o.get("max_tokens", 32000),
            reasoning=reasoning,
        )

    @staticmethod
    def _cost_from_usage(usage) -> float | None:
        if usage is None:
            return None
        cost = getattr(usage, "cost", None)
        if cost is None:
            cost = (getattr(usage, "model_extra", None) or {}).get("cost")
        return cost

    @staticmethod
    def _reasoning_from_message(msg) -> str | None:
        r = getattr(msg, "reasoning", None)
        if r is None:
            extra = getattr(msg, "model_extra", None) or {}
            r = extra.get("reasoning") or extra.get("reasoning_content")
        return r or None

    async def generate(self, *, system: str, user: str, temperature: float | None = None) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        extra_body = {"usage": {"include": True}}
        if self.reasoning:
            extra_body["reasoning"] = self.reasoning
        try:
            r = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.default_temperature if temperature is None else temperature,
                max_tokens=self.max_tokens,
                extra_body=extra_body,
            )
        except Exception as e:  # normalize any SDK error as BackendError
            raise BackendError(f"OpenRouter failed ({self.model}): {e}") from e

        msg = r.choices[0].message
        u = r.usage
        usage = Usage(
            backend="openrouter",
            model=self.model,
            cost_usd=self._cost_from_usage(u),
            input_tokens=getattr(u, "prompt_tokens", None) if u else None,
            output_tokens=getattr(u, "completion_tokens", None) if u else None,
        )
        return Completion(text=msg.content or "", usage=usage, reasoning=self._reasoning_from_message(msg))
