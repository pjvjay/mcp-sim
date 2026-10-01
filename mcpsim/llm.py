"""The one LLM boundary (DESIGN §6).

Everything that talks to a model goes through the :class:`LLM` protocol so tests can substitute
a scripted fake (``tests/fake_llm.py``). :class:`AnthropicLLM` is the real implementation:
retries with exponential backoff on 429/5xx (max 5 attempts), records usage per model and
estimates cost from :data:`RATE_TABLE`. Nothing outside this module imports ``anthropic``.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import anthropic
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MAX_TOKENS = 4096

# USD per million tokens (input, output). These are estimates and the report says so.
RATE_TABLE: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (3.0, 15.0),
    "claude-opus-5-5": (15.0, 75.0),
    "claude-fable-5-1": (15.0, 75.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
KNOWN_MODELS: tuple[str, ...] = (
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-fable-5-1",
    "claude-haiku-4-5-20251001",
)


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """A provider-neutral assistant turn: Anthropic-shaped content blocks as plain dicts."""

    model_config = ConfigDict(extra="forbid")

    content: list[dict[str, Any]]
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str = ""

    def text(self) -> str:
        """Concatenated ``text`` blocks."""
        return "\n".join(
            str(block.get("text", "")) for block in self.content if block.get("type") == "text"
        )

    def tool_uses(self) -> list[dict[str, Any]]:
        return [block for block in self.content if block.get("type") == "tool_use"]


class LLM(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResponse: ...


def rate_for(model: str) -> tuple[float, float] | None:
    """Longest-prefix lookup in :data:`RATE_TABLE` (so dated ids like ``-20251001`` resolve)."""
    best: tuple[float, float] | None = None
    best_len = -1
    for prefix, rates in RATE_TABLE.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best, best_len = rates, len(prefix)
    return best


def estimate_cost_usd(model: str, usage: Usage) -> float:
    """Cost estimate in USD; 0.0 for a model missing from the rate table."""
    rates = rate_for(model)
    if rates is None:
        return 0.0
    rate_in, rate_out = rates
    return (usage.input_tokens * rate_in + usage.output_tokens * rate_out) / 1_000_000


def total_cost_usd(usage_by_model: dict[str, Usage]) -> float:
    return round(sum(estimate_cost_usd(m, u) for m, u in usage_by_model.items()), 6)


SleepFn = Callable[[float], Awaitable[None]]


def is_retryable(exc: BaseException) -> bool:
    """429, any 5xx, or a connection-level failure before a status was received."""
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return isinstance(exc, anthropic.APIConnectionError)


class AnthropicLLM:
    """Anthropic Messages API behind the :class:`LLM` protocol."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        *,
        api_key: str | None = None,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        sleep: SleepFn = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        # Our own backoff governs; disable the SDK's built-in retries on a client we create.
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._jitter = jitter
        self.usage: dict[str, Usage] = {}
        self.calls = 0
        self.retries = 0

    def _record(self, model: str, usage: Usage) -> None:
        self.usage[model] = self.usage.get(model, Usage()) + usage

    def cost_usd(self) -> float:
        return total_cost_usd(self.usage)

    def _delay(self, attempt: int) -> float:
        # attempt is 1-based: 1s, 2s, 4s, 8s ... capped, plus up to one second of jitter.
        return min(self._base_delay * (2 ** (attempt - 1)), self._max_delay) + self._jitter()

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        attempt = 0
        while True:
            attempt += 1
            self.calls += 1
            try:
                message = await self._client.messages.create(**kwargs)
            except Exception as exc:
                if attempt >= self._max_attempts or not is_retryable(exc):
                    raise
                self.retries += 1
                await self._sleep(self._delay(attempt))
                continue
            usage = Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            )
            self._record(model, usage)
            content = [
                block.model_dump(mode="json", exclude_none=True) for block in message.content
            ]
            return LLMResponse(
                content=content,
                stop_reason=message.stop_reason,
                usage=usage,
                model=message.model,
            )
