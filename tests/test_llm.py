from __future__ import annotations

from typing import Any

import anthropic
import httpx
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock
from anthropic.types import Usage as AnthropicUsage

from mcpsim.llm import (
    AnthropicLLM,
    LLMResponse,
    Usage,
    estimate_cost_usd,
    is_retryable,
    rate_for,
    total_cost_usd,
)
from tests.fake_llm import ScriptedLLM, structured_response, text_response, tool_use_response


def test_usage_add_and_total() -> None:
    total = Usage(input_tokens=10, output_tokens=5) + Usage(input_tokens=1, output_tokens=2)
    assert (total.input_tokens, total.output_tokens, total.total_tokens) == (11, 7, 18)


def test_rate_for_prefix_match() -> None:
    assert rate_for("claude-sonnet-5-5") == (3.0, 15.0)
    assert rate_for("claude-opus-5-5") == (15.0, 75.0)
    assert rate_for("claude-fable-5-1") == (15.0, 75.0)
    assert rate_for("claude-haiku-4-5-20251001") == (1.0, 5.0)
    assert rate_for("gpt-9") is None


def test_estimate_cost() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    assert estimate_cost_usd("claude-sonnet-5-5", usage) == pytest.approx(3.0 + 1.5)
    assert estimate_cost_usd("unknown-model", usage) == 0.0
    assert total_cost_usd({"claude-sonnet-5-5": usage, "claude-haiku-4-5-20251001": usage}) == (
        pytest.approx(4.5 + 1.5)
    )


def test_llm_response_helpers() -> None:
    r = tool_use_response("lookup", {"slug": "penne"}, text="Looking it up")
    assert r.text() == "Looking it up"
    assert r.tool_uses() == [
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"slug": "penne"}}
    ]
    assert r.stop_reason == "tool_use"
    assert text_response("done").stop_reason == "end_turn"
    assert structured_response("emit_plan", {"paths": []}).tool_uses()[0]["name"] == "emit_plan"


async def test_scripted_llm_records_calls_and_exhausts() -> None:
    llm = ScriptedLLM([text_response("one")])
    llm.push(text_response("two"))
    first = await llm.complete(model="m", system="s", messages=[{"role": "user", "content": "hi"}])
    assert first.text() == "one"
    assert llm.calls[0]["model"] == "m" and llm.calls[0]["tool_choice"] is None
    assert llm.remaining == 1
    await llm.complete(model="m", system="s", messages=[])
    with pytest.raises(RuntimeError, match="no scripted response left for call #3"):
        await llm.complete(model="m", system="s", messages=[])


# --- AnthropicLLM retry behaviour with a stub client (no network) --------------------------------


def _status_error(status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    if status == 429:
        return anthropic.RateLimitError("rate limited", response=response, body=None)
    if status >= 500:
        return anthropic.InternalServerError("server error", response=response, body=None)
    return anthropic.BadRequestError("bad request", response=response, body=None)


def _message(text: str = "ok", tool: bool = False) -> Message:
    content: list[Any] = [TextBlock(type="text", text=text)]
    if tool:
        content.append(ToolUseBlock(type="tool_use", id="toolu_1", name="t", input={"a": 1}))
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-sonnet-5-5",
        content=content,
        stop_reason="tool_use" if tool else "end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=100, output_tokens=20),
    )


class _StubMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.kwargs: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        self.kwargs.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, Message)
        return outcome


class _StubClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.messages = _StubMessages(outcomes)


def _llm(outcomes: list[Any], **kwargs: Any) -> tuple[AnthropicLLM, list[float], _StubClient]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = _StubClient(outcomes)
    llm = AnthropicLLM(client, sleep=fake_sleep, jitter=lambda: 0.0, **kwargs)  # type: ignore[arg-type]
    return llm, sleeps, client


def test_is_retryable() -> None:
    assert is_retryable(_status_error(429)) is True
    assert is_retryable(_status_error(503)) is True
    assert is_retryable(_status_error(400)) is False
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    assert is_retryable(anthropic.APIConnectionError(request=request)) is True
    assert is_retryable(ValueError("x")) is False


async def test_retries_on_429_and_5xx_then_succeeds() -> None:
    llm, sleeps, client = _llm(
        [_status_error(429), _status_error(502), _message("hello", tool=True)]
    )
    response = await llm.complete(
        model="claude-sonnet-5-5",
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "t", "description": "", "input_schema": {"type": "object"}}],
        tool_choice={"type": "auto"},
        max_tokens=99,
    )
    assert isinstance(response, LLMResponse)
    assert response.text() == "hello"
    assert response.tool_uses() == [
        {"type": "tool_use", "id": "toolu_1", "name": "t", "input": {"a": 1}}
    ]
    assert response.stop_reason == "tool_use"
    assert response.usage == Usage(input_tokens=100, output_tokens=20)
    assert sleeps == [1.0, 2.0]
    assert (llm.calls, llm.retries) == (3, 2)
    assert llm.usage == {"claude-sonnet-5-5": Usage(input_tokens=100, output_tokens=20)}
    assert llm.cost_usd() == pytest.approx((100 * 3 + 20 * 15) / 1_000_000)
    sent = client.messages.kwargs[-1]
    assert sent["max_tokens"] == 99 and sent["tool_choice"] == {"type": "auto"}
    assert sent["tools"][0]["name"] == "t" and sent["system"] == "sys"


async def test_non_retryable_error_raises_immediately() -> None:
    llm, sleeps, _ = _llm([_status_error(400), _message()])
    with pytest.raises(anthropic.BadRequestError):
        await llm.complete(model="m", system="s", messages=[])
    assert sleeps == [] and llm.calls == 1


async def test_gives_up_after_max_attempts() -> None:
    llm, sleeps, _ = _llm([_status_error(429)] * 5 + [_message()], max_attempts=5)
    with pytest.raises(anthropic.RateLimitError):
        await llm.complete(model="m", system="s", messages=[])
    assert llm.calls == 5 and llm.retries == 4
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


async def test_backoff_is_capped() -> None:
    llm, sleeps, _ = _llm([_status_error(500)] * 4 + [_message()], max_delay=3.0)
    await llm.complete(model="m", system="s", messages=[])
    assert sleeps == [1.0, 2.0, 3.0, 3.0]


def test_max_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError):
        AnthropicLLM(_StubClient([]), max_attempts=0)  # type: ignore[arg-type]


# --- forced tool choice -> auto + instruction (Claude 5.5 / Fable 5.1 reject "tool"/"any") ----


def test_translate_tool_choice_table() -> None:
    from mcpsim.llm import translate_tool_choice

    choice, note = translate_tool_choice({"type": "tool", "name": "emit_execution_plan"})
    assert choice == {"type": "auto"}
    assert note is not None and "`emit_execution_plan`" in note and "exactly once" in note
    choice, note = translate_tool_choice({"type": "any"})
    assert choice == {"type": "auto"}
    assert note is not None and "exactly one of the provided tools" in note
    assert translate_tool_choice({"type": "auto"}) == ({"type": "auto"}, None)
    assert translate_tool_choice({"type": "none"}) == ({"type": "none"}, None)


@pytest.mark.asyncio
async def test_forced_tool_choice_is_sent_as_auto_with_the_instruction() -> None:
    llm, _, client = _llm([_message(tool=True)])
    await llm.complete(
        model="claude-opus-5-5", system="You plan.", messages=[{"role": "user", "content": "go"}],
        tools=[{"name": "emit", "description": "d", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "emit"},
    )
    sent = client.messages.kwargs[0]
    assert sent["tool_choice"] == {"type": "auto"}
    assert sent["system"].startswith("You plan.")
    assert "Respond ONLY by calling the tool named `emit` exactly once" in sent["system"]
    assert llm.forced_choice_translations == 1


@pytest.mark.asyncio
async def test_auto_tool_choice_is_passed_through_unchanged() -> None:
    llm, _, client = _llm([_message("ok")])
    await llm.complete(
        model="claude-sonnet-5-5", system="S", messages=[{"role": "user", "content": "go"}],
        tools=[{"name": "emit", "description": "d", "input_schema": {"type": "object"}}],
        tool_choice={"type": "auto"},
    )
    sent = client.messages.kwargs[0]
    assert sent["tool_choice"] == {"type": "auto"} and sent["system"] == "S"
    assert llm.forced_choice_translations == 0
