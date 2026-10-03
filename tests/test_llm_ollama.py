"""OllamaLLM against ``httpx.MockTransport`` standing in for a local Ollama server.

Every assertion here is on the exact request body shapes and reply mappings in the
docs/LOCAL_MODELS.md table; no network, no model.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from mcpsim.llm import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_NUM_CTX,
    MAX_OLLAMA_DEADLINE_S,
    OLLAMA_DEADLINE_ENV,
    OLLAMA_HOST_ENV,
    OLLAMA_KEEP_ALIVE_ENV,
    OLLAMA_NUM_CTX_ENV,
    LLMResponse,
    OllamaError,
    OllamaLLM,
    Usage,
    estimate_cost_usd,
    forced_history_as_text,
    forced_tool,
    is_retryable_ollama,
    ollama_deadline_s,
    ollama_host,
    ollama_keep_alive,
    ollama_num_ctx,
    to_ollama_messages,
    to_ollama_tools,
)

HOST = "http://ollama.test:11434"

LOOKUP_TOOL: dict[str, Any] = {
    "name": "lookup",
    "description": "Price of a product by slug.",
    "input_schema": {
        "type": "object",
        "properties": {"slug": {"type": "string"}},
        "required": ["slug"],
    },
}
PLAN_TOOL: dict[str, Any] = {
    "name": "emit_plan",
    "description": "Return the execution plan.",
    "input_schema": {
        "type": "object",
        "properties": {"paths": {"type": "array", "items": {"type": "object"}}},
        "required": ["paths"],
        "$defs": {"Step": {"type": "object"}},
    },
}


def reply(
    content: str = "",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    done_reason: str = "stop",
    prompt_eval_count: int = 120,
    eval_count: int = 30,
    model: str = "command-r7b",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "model": model,
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


Outcome = dict[str, Any] | int | Exception


class Server:
    """A scripted Ollama: one outcome per request (a reply dict, a status code or an error)."""

    def __init__(self, outcomes: list[Outcome]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            body = {"error": "model 'nope' not found"} if outcome == 404 else {"error": "boom"}
            return httpx.Response(outcome, json=body)
        return httpx.Response(200, json=outcome)

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def make(outcomes: list[Outcome], **kwargs: Any) -> tuple[OllamaLLM, Server, list[float]]:
    server = Server(outcomes)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    llm = OllamaLLM(
        HOST,
        client=client,
        num_ctx=8192,
        sleep=fake_sleep,
        jitter=lambda: 0.0,
        **kwargs,
    )
    return llm, server, sleeps


async def test_tools_call_request_body() -> None:
    llm, server, _ = make([reply("Looking.")])
    response = await llm.complete(
        model="ollama:command-r7b",
        system="You are the agent.",
        messages=[{"role": "user", "content": "Price of penne?"}],
        tools=[LOOKUP_TOOL],
        tool_choice={"type": "auto"},
        max_tokens=512,
    )
    assert str(server.requests[0].url) == f"{HOST}/api/chat"
    body = server.body()
    assert body == {
        "model": "command-r7b",
        "messages": [
            {"role": "system", "content": "You are the agent."},
            {"role": "user", "content": "Price of penne?"},
        ],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 512},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Price of a product by slug.",
                    "parameters": LOOKUP_TOOL["input_schema"],
                },
            }
        ],
    }
    assert "format" not in body
    assert response.content == [{"type": "text", "text": "Looking."}]
    assert response.stop_reason == "end_turn"


async def test_forced_tool_choice_becomes_format_and_wraps_the_reply() -> None:
    payload = {"paths": [{"id": "happy"}]}
    llm, server, _ = make([reply(json.dumps(payload))])
    response = await llm.complete(
        model="ollama:command-r7b",
        system="Plan it.",
        messages=[{"role": "user", "content": "scenario + catalog"}],
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": "emit_plan"},
        max_tokens=2048,
    )
    body = server.body()
    assert body["format"] == PLAN_TOOL["input_schema"]
    assert "tools" not in body
    assert body["options"]["num_predict"] == 2048
    system = body["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith("Plan it.")
    assert "`emit_plan` schema" in system["content"]
    assert "Return the execution plan." in system["content"]
    assert response.content == [
        {"type": "tool_use", "id": "call_1", "name": "emit_plan", "input": payload}
    ]
    assert response.stop_reason == "tool_use"
    assert response.tool_uses()[0]["input"] == payload


async def test_forced_reply_that_is_not_json_stays_text() -> None:
    llm, _, _ = make([reply("not json at all")])
    response = await llm.complete(
        model="ollama:command-r7b",
        system="",
        messages=[{"role": "user", "content": "x"}],
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": "emit_plan"},
    )
    assert response.content == [{"type": "text", "text": "not json at all"}]
    assert response.stop_reason == "end_turn"


def test_forced_tool_must_be_in_tools() -> None:
    with pytest.raises(ValueError, match="not in tools"):
        forced_tool([LOOKUP_TOOL], {"type": "tool", "name": "emit_plan"})
    assert forced_tool([LOOKUP_TOOL], {"type": "auto"}) is None
    assert forced_tool(None, None) is None


async def test_tool_result_round_trip() -> None:
    llm, server, _ = make([reply("Penne is $2.49.")])
    messages = [
        {"role": "user", "content": "Price of penne?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me look."},
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"slug": "penne"}},
                {"type": "tool_use", "id": "call_2", "name": "list_items", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": '{"price": 2.49}'},
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": [{"type": "text", "text": "no such page"}],
                    "is_error": True,
                },
            ],
        },
    ]
    await llm.complete(
        model="ollama:command-r7b", system="sys", messages=messages, tools=[LOOKUP_TOOL]
    )
    assert server.body()["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Price of penne?"},
        {
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [
                {"function": {"name": "lookup", "arguments": {"slug": "penne"}}},
                {"function": {"name": "list_items", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_name": "lookup", "content": '{"price": 2.49}'},
        {"role": "tool", "tool_name": "list_items", "content": "ERROR: no such page"},
    ]


def test_to_ollama_messages_keeps_text_after_tool_results_and_skips_empty_system() -> None:
    out = to_ollama_messages(
        "",
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "lookup", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                    {"type": "text", "text": "and now?"},
                ],
            },
        ],
    )
    assert out == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "lookup", "arguments": {}}}],
        },
        {"role": "tool", "tool_name": "lookup", "content": "ok"},
        {"role": "user", "content": "and now?"},
    ]


def test_to_ollama_tools_passes_schema_through() -> None:
    out = to_ollama_tools([LOOKUP_TOOL, {"name": "bare"}])
    assert out[0]["function"]["parameters"] is LOOKUP_TOOL["input_schema"]
    assert out[1] == {
        "type": "function",
        "function": {"name": "bare", "description": "", "parameters": {"type": "object"}},
    }


async def test_tool_calls_map_to_tool_use_blocks_with_synthetic_ids() -> None:
    llm, _, _ = make(
        [
            reply(
                "",
                tool_calls=[
                    {"function": {"name": "lookup", "arguments": {"slug": "penne"}}},
                    {"function": {"name": "list_items", "arguments": '{"limit": 2}'}},
                ],
                prompt_eval_count=1451,
                eval_count=918,
                model="command-r7b:latest",
            ),
            reply("", tool_calls=[{"function": {"name": "lookup", "arguments": {"slug": "x"}}}]),
        ]
    )
    first = await llm.complete(
        model="ollama:command-r7b", system="s", messages=[], tools=[LOOKUP_TOOL]
    )
    assert isinstance(first, LLMResponse)
    assert first.content == [
        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"slug": "penne"}},
        {"type": "tool_use", "id": "call_2", "name": "list_items", "input": {"limit": 2}},
    ]
    assert first.stop_reason == "tool_use"
    assert first.usage == Usage(input_tokens=1451, output_tokens=918)
    assert first.model == "command-r7b:latest"
    second = await llm.complete(
        model="ollama:command-r7b", system="s", messages=[], tools=[LOOKUP_TOOL]
    )
    assert second.tool_uses()[0]["id"] == "call_3"  # ids stay unique across the conversation
    assert llm.usage == {"ollama:command-r7b": Usage(input_tokens=1451 + 120, output_tokens=948)}
    assert llm.cost_usd() == 0.0
    assert estimate_cost_usd("ollama:command-r7b", first.usage) == 0.0
    assert llm.calls == 2 and llm.retries == 0


async def test_done_reason_length_is_max_tokens() -> None:
    llm, _, _ = make([reply("half an ans", done_reason="length")])
    response = await llm.complete(model="ollama:llama3.2:3b", system="s", messages=[])
    assert response.stop_reason == "max_tokens"
    assert response.content == [{"type": "text", "text": "half an ans"}]


async def test_forced_reply_truncated_is_max_tokens_not_tool_use() -> None:
    llm, _, _ = make([reply('{"paths": [', done_reason="length")])
    response = await llm.complete(
        model="ollama:command-r7b",
        system="",
        messages=[],
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": "emit_plan"},
    )
    assert response.stop_reason == "max_tokens"
    assert response.tool_uses() == []


async def test_404_names_ollama_pull() -> None:
    llm, _, sleeps = make([404])
    with pytest.raises(OllamaError, match=r"ollama pull nope") as exc_info:
        await llm.complete(model="ollama:nope", system="s", messages=[])
    assert "model 'nope' not found" in str(exc_info.value) and HOST in str(exc_info.value)
    assert llm.calls == 1 and sleeps == []


async def test_400_is_not_retried() -> None:
    llm, _, sleeps = make([400])
    with pytest.raises(OllamaError, match=r"rejected the request \(400\): boom"):
        await llm.complete(model="ollama:command-r7b", system="s", messages=[])
    assert llm.calls == 1 and sleeps == []


async def test_retry_on_503_then_success() -> None:
    llm, server, sleeps = make([503, reply("after retry")])
    response = await llm.complete(model="ollama:command-r7b", system="s", messages=[])
    assert response.text() == "after retry"
    assert sleeps == [1.0]
    assert (llm.calls, llm.retries) == (2, 1)
    assert server.body(0) == server.body(1)  # the same body is resent


async def test_connection_errors_retry_then_become_user_facing() -> None:
    boom: Callable[[], Exception] = lambda: httpx.ConnectError("refused")  # noqa: E731
    llm, _, sleeps = make([boom(), boom(), boom()])
    with pytest.raises(OllamaError, match="cannot reach Ollama at .*ollama serve"):
        await llm.complete(model="ollama:command-r7b", system="s", messages=[])
    assert llm.calls == 3 and llm.retries == 2
    assert sleeps == [1.0, 2.0]


async def test_gives_up_on_5xx_after_max_attempts() -> None:
    llm, _, sleeps = make([500, 502, 503], max_attempts=3)
    with pytest.raises(OllamaError, match="failed with 503 after 3 attempt"):
        await llm.complete(model="ollama:command-r7b", system="s", messages=[])
    assert sleeps == [1.0, 2.0]


async def test_backoff_is_capped() -> None:
    llm, _, sleeps = make([500, 500, 500, reply("ok")], max_attempts=4, max_delay=1.5)
    await llm.complete(model="ollama:command-r7b", system="s", messages=[])
    assert sleeps == [1.0, 1.5, 1.5]


def test_is_retryable_ollama() -> None:
    request = httpx.Request("POST", f"{HOST}/api/chat")
    for status, expected in ((500, True), (503, True), (404, False), (400, False)):
        exc = httpx.HTTPStatusError(
            "x", request=request, response=httpx.Response(status, request=request)
        )
        assert is_retryable_ollama(exc) is expected, status
    assert is_retryable_ollama(httpx.ConnectError("refused")) is True
    # A read timeout is the model being slow: retrying spends the deadline on the same prompt.
    assert is_retryable_ollama(httpx.ReadTimeout("slow")) is False
    assert is_retryable_ollama(ValueError("x")) is False


def test_env_helpers() -> None:
    assert ollama_host({}) == DEFAULT_OLLAMA_HOST
    assert ollama_host({OLLAMA_HOST_ENV: "http://box:11434/"}) == "http://box:11434"
    assert ollama_host({OLLAMA_HOST_ENV: "box:11434"}) == "http://box:11434"
    assert ollama_num_ctx({}) == DEFAULT_OLLAMA_NUM_CTX == 8192
    assert ollama_num_ctx({OLLAMA_NUM_CTX_ENV: "32768"}) == 32768
    with pytest.raises(ValueError, match=OLLAMA_NUM_CTX_ENV):
        ollama_num_ctx({OLLAMA_NUM_CTX_ENV: "lots"})
    with pytest.raises(ValueError, match=OLLAMA_NUM_CTX_ENV):
        ollama_num_ctx({OLLAMA_NUM_CTX_ENV: "0"})


def test_constructor_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OLLAMA_HOST_ENV, "http://box:11434")
    monkeypatch.setenv(OLLAMA_NUM_CTX_ENV, "4096")
    llm = OllamaLLM(client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)))  # type: ignore[arg-type,return-value]
    assert (llm.host, llm.num_ctx) == ("http://box:11434", 4096)
    with pytest.raises(ValueError):
        OllamaLLM(HOST, max_attempts=0)


# --- deadline and keep_alive (docs/LOCAL_MODELS.md "Speed") ------------------------------------


def test_deadline_defaults_to_and_never_exceeds_ten_minutes() -> None:
    assert MAX_OLLAMA_DEADLINE_S == 600.0
    assert ollama_deadline_s({}) == 600.0
    assert ollama_deadline_s({OLLAMA_DEADLINE_ENV: "120"}) == 120.0
    assert ollama_deadline_s({OLLAMA_DEADLINE_ENV: "600"}) == 600.0
    for bad in ("601", "3600", "0", "-5", "ten"):
        with pytest.raises(ValueError, match=OLLAMA_DEADLINE_ENV):
            ollama_deadline_s({OLLAMA_DEADLINE_ENV: bad})
    with pytest.raises(ValueError, match="deadline"):
        OllamaLLM(HOST, deadline=601)


async def test_a_call_past_its_deadline_is_cancelled_with_a_clear_error() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200, json=reply("late"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    llm = OllamaLLM(HOST, client=client, num_ctx=8192, deadline=0.2)
    started = time.monotonic()
    with pytest.raises(OllamaError, match=r"command-r7b did not answer within 0\.2 s"):
        await llm.complete(model="ollama:command-r7b", system="s",
                           messages=[{"role": "user", "content": "hi"}])
    assert time.monotonic() - started < 5
    assert llm.calls == 1 and llm.retries == 0


async def test_a_read_timeout_is_the_deadline_and_is_not_retried() -> None:
    llm, server, sleeps = make([httpx.ReadTimeout("slow model"), reply("never sent")])
    with pytest.raises(OllamaError, match="did not answer within 600 s"):
        await llm.complete(model="ollama:command-r7b", system="s", messages=[])
    assert len(server.requests) == 1 and sleeps == []
    assert not is_retryable_ollama(httpx.ReadTimeout("slow"))
    assert is_retryable_ollama(httpx.ConnectError("down"))


async def test_keep_alive_is_sent_and_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OLLAMA_KEEP_ALIVE_ENV, raising=False)
    assert ollama_keep_alive() == "30m"
    monkeypatch.setenv(OLLAMA_KEEP_ALIVE_ENV, "2h")
    llm, server, _ = make([reply("ok")])
    await llm.complete(model="ollama:command-r7b", system="s", messages=[])
    assert server.body()["keep_alive"] == "2h"


def test_forced_history_is_replayed_as_the_json_the_model_wrote() -> None:
    """In a ``format`` conversation an earlier answer goes back as its compact JSON text, so
    the next call extends what the server cached instead of a re-rendered tool call."""
    payload = {"paths": [{"id": "1", "kind": "happy", "steps": []}]}
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "plan it"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_1", "name": "emit_plan", "input": payload}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "accepted"},
            {"type": "text", "text": "Now one more path."}]},
    ]
    replayed = forced_history_as_text(messages, "emit_plan")
    assert replayed[1]["content"] == [
        {"type": "text", "text": '{"paths":[{"id":"1","kind":"happy","steps":[]}]}'}]
    llm = OllamaLLM(HOST, num_ctx=8192)
    body = llm.build_request(model="command-r7b", system="sys", messages=messages,
                             tools=[PLAN_TOOL], tool_choice={"type": "tool", "name": "emit_plan"},
                             max_tokens=100)
    assert body["messages"][1:] == [
        {"role": "user", "content": "plan it"},
        {"role": "assistant", "content": '{"paths":[{"id":"1","kind":"happy","steps":[]}]}'},
        {"role": "user", "content": "accepted\nNow one more path."},
    ]
    # An unforced conversation keeps real tool calls (the agent loop needs them).
    body = llm.build_request(model="command-r7b", system="sys", messages=messages,
                             tools=[PLAN_TOOL], tool_choice={"type": "auto"}, max_tokens=100)
    assert body["messages"][2]["tool_calls"][0]["function"]["name"] == "emit_plan"
