"""GeminiLLM against ``httpx.MockTransport`` standing in for Gemini's OpenAI-compatible endpoint.

No network, no key: assertions are on the request bodies sent and the replies mapped back.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from mcpsim.llm import (
    DEFAULT_GEMINI_BASE_URL,
    GEMINI,
    GEMINI_API_KEY_ENV,
    GEMINI_BASE_URL_ENV,
    GEMINI_REASONING_EFFORT_ENV,
    GeminiError,
    GeminiLLM,
    clear_llm_cache,
    gemini_arguments,
    gemini_base_url,
    gemini_reasoning_effort,
    is_retryable_gemini,
    make_llm,
    parse_model_spec,
    to_openai_messages,
    to_openai_tool_choice,
)

BASE = "https://gemini.test/v1beta/openai"
MODEL = "gemini:gemini-3-flash-preview"

LOOKUP_TOOL: dict[str, Any] = {
    "name": "lookup",
    "description": "Price of a product by slug.",
    "input_schema": {
        "type": "object",
        "properties": {"slug": {"type": "string"}},
        "required": ["slug"],
    },
}
VERDICT_TOOL: dict[str, Any] = {
    "name": "verdict",
    "description": "Record a verdict.",
    "input_schema": {"type": "object", "properties": {"passed": {"type": "boolean"}}},
}
SIGNATURE = {"google": {"thought_signature": "sig-abc"}}


def completion(message: dict[str, Any], finish: str = "stop", **usage: int) -> dict[str, Any]:
    return {
        "model": "gemini-3-flash-preview",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def client_for(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[GeminiLLM, list[float]]:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    llm = GeminiLLM(
        "test-key",
        base_url=BASE,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=sleep,
        jitter=lambda: 0.0,
        **kwargs,
    )
    return llm, slept


def test_gemini_is_a_provider() -> None:
    assert parse_model_spec(MODEL) == (GEMINI, "gemini-3-flash-preview")


def test_tool_call_round_trip_keeps_the_thought_signature() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{BASE}/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(200, json=completion(
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "c1", "type": "function", "extra_content": SIGNATURE,
                    "function": {"name": "lookup", "arguments": '{"slug": "penne"}'},
                }]},
                finish="tool_calls",
                prompt_tokens=20, completion_tokens=4, total_tokens=60,
            ))
        return httpx.Response(200, json=completion({"role": "assistant", "content": "$2.49"}))

    llm, _ = client_for(handler)

    async def run() -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Penne?"}]
        first = await llm.complete(model=MODEL, system="Use tools.", messages=messages,
                                   tools=[LOOKUP_TOOL], max_tokens=256)
        assert first.stop_reason == "tool_use"
        (use,) = first.tool_uses()
        assert (use["id"], use["name"], use["input"]) == ("c1", "lookup", {"slug": "penne"})
        assert use["provider_extra"] == SIGNATURE
        # thinking tokens (total - prompt) count as output
        assert (first.usage.input_tokens, first.usage.output_tokens) == (20, 40)
        messages += [
            {"role": "assistant", "content": list(first.content)},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": '{"price": 2.49}'}
            ]},
        ]
        second = await llm.complete(model=MODEL, system="Use tools.", messages=messages,
                                    tools=[LOOKUP_TOOL])
        assert (second.stop_reason, second.text()) == ("end_turn", "$2.49")

    asyncio.run(run())
    body = seen[0]
    assert body["model"] == "gemini-3-flash-preview"
    assert body["max_tokens"] == 256
    assert body["reasoning_effort"] == "low"
    assert "temperature" not in body
    assert body["messages"][0] == {"role": "system", "content": "Use tools."}
    assert body["tools"] == [{"type": "function", "function": {
        "name": "lookup", "description": "Price of a product by slug.",
        "parameters": LOOKUP_TOOL["input_schema"]}}]
    assert "tool_choice" not in body
    replay = seen[1]["messages"]
    assert replay[2] == {"role": "assistant", "content": None, "tool_calls": [{
        "id": "c1", "type": "function", "extra_content": SIGNATURE,
        "function": {"name": "lookup", "arguments": '{"slug": "penne"}'}}]}
    assert replay[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"price": 2.49}'}
    assert llm.usage[MODEL].input_tokens == 30


def test_forced_tool_choice_and_temperature() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion(
            {"role": "assistant", "tool_calls": [{"id": "v", "type": "function", "function": {
                "name": "verdict", "arguments": '{"passed": true}'}}]},
            finish="tool_calls",
        ))

    llm, _ = client_for(handler, reasoning_effort=None)
    result = asyncio.run(llm.complete(
        model=MODEL, system="", messages=[{"role": "user", "content": "ok?"}],
        tools=[VERDICT_TOOL], tool_choice={"type": "tool", "name": "verdict"}, temperature=0.2,
    ))
    assert result.tool_uses()[0]["input"] == {"passed": True}
    assert "provider_extra" not in result.tool_uses()[0]
    body = seen[0]
    assert body["tool_choice"] == {"type": "function", "function": {"name": "verdict"}}
    assert body["temperature"] == 0.2
    assert "reasoning_effort" not in body
    assert body["messages"] == [{"role": "user", "content": "ok?"}]
    with pytest.raises(ValueError, match="not in tools"):
        llm.build_request(model="m", system="", messages=[], tools=[VERDICT_TOOL],
                          tool_choice={"type": "tool", "name": "nope"}, max_tokens=10)


def test_length_finish_is_max_tokens() -> None:
    reply = completion({"role": "assistant", "content": "cut"}, finish="length")
    assert GeminiLLM("k").parse_response(reply, model="m").stop_reason == "max_tokens"


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ({"type": "tool", "name": "x"}, {"type": "function", "function": {"name": "x"}}),
        ({"type": "any"}, "required"),
        ({"type": "auto"}, "auto"),
        ({"type": "none"}, "none"),
        ({"type": "weird"}, None),
        (None, None),
    ],
)
def test_to_openai_tool_choice(choice: dict[str, Any] | None, expected: Any) -> None:
    assert to_openai_tool_choice(choice) == expected


def test_to_openai_messages_marks_errors_and_keeps_text_after_results() -> None:
    out = to_openai_messages("", [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "boom", "is_error": True},
        {"type": "text", "text": "and then?"},
    ]}])
    assert out == [
        {"role": "tool", "tool_call_id": "a", "content": "ERROR: boom"},
        {"role": "user", "content": "and then?"},
    ]


def test_429_and_503_are_retried_honouring_retry_after() -> None:
    replies = [
        httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow"}}),
        httpx.Response(503, json=[{"error": {"message": "high demand"}}]),
        httpx.Response(200, json=completion({"role": "assistant", "content": "hi"})),
    ]
    llm, slept = client_for(lambda request: replies.pop(0))
    result = asyncio.run(llm.complete(model=MODEL, system="", messages=[]))
    assert result.text() == "hi"
    assert (llm.calls, llm.retries) == (3, 2)
    assert slept == [7.0, 4.0]  # Retry-After, then base_delay * 2 for the second attempt


def test_retries_give_up_with_the_status() -> None:
    llm, _ = client_for(lambda request: httpx.Response(503, text="busy"), max_attempts=2)
    with pytest.raises(GeminiError, match="503 after 2 attempt"):
        asyncio.run(llm.complete(model=MODEL, system="", messages=[]))


@pytest.mark.parametrize(
    ("status", "needle"),
    [(400, "rejected the request (400): bad schema"), (403, "rejected the key"),
     (404, "no model 'gemini-3-flash-preview'")],
)
def test_4xx_fails_at_once(status: int, needle: str) -> None:
    llm, _ = client_for(
        lambda request: httpx.Response(status, json=[{"error": {"message": "bad schema"}}])
    )
    with pytest.raises(GeminiError, match=needle.replace("(", r"\(").replace(")", r"\)")):
        asyncio.run(llm.complete(model=MODEL, system="", messages=[]))
    assert llm.calls == 1


def test_is_retryable_gemini() -> None:
    request = httpx.Request("POST", BASE)
    for status, expected in ((429, True), (500, True), (503, True), (400, False), (404, False)):
        exc = httpx.HTTPStatusError("x", request=request,
                                    response=httpx.Response(status, request=request))
        assert is_retryable_gemini(exc) is expected
    assert is_retryable_gemini(httpx.ConnectError("down"))
    assert not is_retryable_gemini(ValueError("nope"))


def test_env_helpers() -> None:
    assert gemini_base_url({}) == DEFAULT_GEMINI_BASE_URL
    assert gemini_base_url({GEMINI_BASE_URL_ENV: "https://x/v1/"}) == "https://x/v1"
    assert gemini_reasoning_effort({}) == "low"
    assert gemini_reasoning_effort({GEMINI_REASONING_EFFORT_ENV: "medium"}) == "medium"
    assert gemini_reasoning_effort({GEMINI_REASONING_EFFORT_ENV: ""}) is None


def test_make_llm_needs_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_llm_cache()
    monkeypatch.delenv(GEMINI_API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=f"{GEMINI_API_KEY_ENV} is not set.*judge"):
        make_llm("gemini", purpose="judge")
    monkeypatch.setenv(GEMINI_API_KEY_ENV, "k")
    assert isinstance(make_llm("gemini"), GeminiLLM)
    clear_llm_cache()


# --- negative cases: malformed replies, broken transport, bad configuration --------------------


def _call(arguments: Any, *, call_id: str | None = "c1", name: str = "lookup") -> dict[str, Any]:
    call: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if call_id is not None:
        call["id"] = call_id
    return call


def test_calls_without_ids_get_ids_that_never_repeat_across_turns() -> None:
    llm = GeminiLLM("k")
    reply = completion({"role": "assistant", "tool_calls": [_call("{}", call_id=None)]},
                       finish="tool_calls")
    first = llm.parse_response(reply, model="m").tool_uses()[0]["id"]
    second = llm.parse_response(reply, model="m").tool_uses()[0]["id"]
    assert first != second and first.startswith("gemini_call_")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [({"slug": "penne"}, {"slug": "penne"}), ('{"slug": "penne"}', {"slug": "penne"}),
     ("", {}), ("   ", {}), (None, {})],
)
def test_gemini_arguments_accepts_objects_and_empty(raw: Any, expected: dict[str, Any]) -> None:
    assert gemini_arguments(raw, "lookup") == expected


@pytest.mark.parametrize(
    ("raw", "needle"),
    [('{"slug": "penne"', "not valid JSON"), ("[1, 2]", "not a JSON object"),
     ('"penne"', "not a JSON object"), (42, "not a JSON object")],
)
def test_gemini_arguments_rejects_anything_else(raw: Any, needle: str) -> None:
    with pytest.raises(GeminiError, match=f"arguments for 'lookup' that are {needle}"):
        gemini_arguments(raw, "lookup")


def test_malformed_tool_arguments_fail_the_call_without_a_retry() -> None:
    reply = completion({"role": "assistant", "tool_calls": [_call('{"slug": ')]},
                       finish="tool_calls")
    llm, slept = client_for(lambda request: httpx.Response(200, json=reply))
    with pytest.raises(GeminiError, match="not valid JSON"):
        asyncio.run(llm.complete(model=MODEL, system="", messages=[], tools=[LOOKUP_TOOL]))
    assert (llm.calls, llm.retries, slept) == (1, 0, [])
    assert MODEL not in llm.usage


@pytest.mark.parametrize(
    ("response", "needle"),
    [(httpx.Response(200, text="<html>busy</html>"), "not JSON (HTTP 200): <html>busy</html>"),
     (httpx.Response(200, json=[{"choices": []}]), "non-object reply")],
)
def test_a_200_that_is_not_a_json_object_is_a_gemini_error(
    response: httpx.Response, needle: str
) -> None:
    llm, _ = client_for(lambda request: response)
    with pytest.raises(GeminiError, match=needle.replace("(", r"\(").replace(")", r"\)")):
        asyncio.run(llm.complete(model=MODEL, system="", messages=[]))


def test_connection_errors_are_retried_then_named() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    llm, slept = client_for(handler, max_attempts=3)
    with pytest.raises(GeminiError, match=r"cannot reach Gemini at https://gemini.test/v1beta/"
                                          r"openai after 3 attempt\(s\) \(ConnectError"):
        asyncio.run(llm.complete(model=MODEL, system="", messages=[]))
    assert (llm.calls, llm.retries, slept) == (3, 2, [2.0, 4.0])


def test_a_connection_error_then_an_answer_succeeds() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ReadError("reset", request=request)
        return httpx.Response(200, json=completion({"role": "assistant", "content": "ok"}))

    llm, _ = client_for(handler)
    assert asyncio.run(llm.complete(model=MODEL, system="", messages=[])).text() == "ok"
    assert (llm.calls, llm.retries) == (2, 1)


def test_an_unexpected_exception_is_not_retried() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise ValueError("bug in the transport")

    llm, slept = client_for(handler)
    with pytest.raises(ValueError, match="bug in the transport"):
        asyncio.run(llm.complete(model=MODEL, system="", messages=[]))
    assert (llm.calls, slept) == (1, [])


@pytest.mark.parametrize(
    ("header", "expected_sleep"),
    [("Wed, 21 Oct 2026 07:28:00 GMT", 2.0),  # an HTTP date is not seconds: back off instead
     ("3600", 60.0),                           # capped at max_delay
     ("", 2.0)],
)
def test_retry_after_that_is_not_usable_falls_back_or_is_capped(
    header: str, expected_sleep: float
) -> None:
    replies = [httpx.Response(429, headers={"retry-after": header}),
               httpx.Response(200, json=completion({"role": "assistant", "content": "ok"}))]
    llm, slept = client_for(lambda request: replies.pop(0))
    asyncio.run(llm.complete(model=MODEL, system="", messages=[]))
    assert slept == [expected_sleep]


def test_a_4xx_with_a_plain_text_body_still_names_the_detail() -> None:
    llm, _ = client_for(lambda request: httpx.Response(400, text="Bad Request: schema"))
    with pytest.raises(GeminiError, match=r"rejected the request \(400\): Bad Request: schema"):
        asyncio.run(llm.complete(model=MODEL, system="", messages=[]))


def test_401_names_the_key_variable() -> None:
    llm, _ = client_for(
        lambda request: httpx.Response(401, json={"error": {"message": "API key not valid"}})
    )
    with pytest.raises(GeminiError, match=f"rejected the key in {GEMINI_API_KEY_ENV} \\(401\\)"):
        asyncio.run(llm.complete(model=MODEL, system="", messages=[]))


def test_complete_without_a_key_makes_no_request() -> None:
    seen: list[httpx.Request] = []
    llm = GeminiLLM(
        "", base_url=BASE,
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: seen.append(r))),  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(RuntimeError, match=f"{GEMINI_API_KEY_ENV} is not set"):
        asyncio.run(llm.complete(model=MODEL, system="", messages=[]))
    assert seen == []


def test_max_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        GeminiLLM("k", max_attempts=0)


def test_cost_is_reported_as_zero() -> None:
    assert GeminiLLM("k").cost_usd() == 0.0


def test_an_empty_reply_is_an_empty_end_turn() -> None:
    llm = GeminiLLM("k")
    for data in ({}, {"choices": []}, {"choices": [{"message": None}]}):
        result = llm.parse_response(data, model="m")
        assert (result.content, result.stop_reason) == ([], "end_turn")
        assert (result.usage.input_tokens, result.usage.output_tokens) == (0, 0)
        assert result.model == "m"


def test_usage_accumulates_per_model() -> None:
    llm, _ = client_for(lambda request: httpx.Response(200, json=completion(
        {"role": "assistant", "content": "ok"}, prompt_tokens=7, completion_tokens=3,
        total_tokens=10)))
    for _ in range(2):
        asyncio.run(llm.complete(model=MODEL, system="", messages=[]))
    assert (llm.usage[MODEL].input_tokens, llm.usage[MODEL].output_tokens) == (14, 6)


def test_reasoning_effort_and_base_url_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(GEMINI_REASONING_EFFORT_ENV, "medium")
    monkeypatch.setenv(GEMINI_BASE_URL_ENV, "https://proxy.test/v1/")
    llm = GeminiLLM("k")
    assert llm.base_url == "https://proxy.test/v1"
    body = llm.build_request(model="m", system="", messages=[], tools=None, tool_choice=None,
                             max_tokens=10)
    assert body["reasoning_effort"] == "medium" and "tools" not in body
    monkeypatch.setenv(GEMINI_REASONING_EFFORT_ENV, "")
    assert "reasoning_effort" not in GeminiLLM("k").build_request(
        model="m", system="", messages=[], tools=None, tool_choice=None, max_tokens=10)


def test_to_openai_messages_covers_every_block_shape() -> None:
    out = to_openai_messages("sys", [
        {"role": "user", "content": "plain"},
        {"content": None},  # no role: a user message with empty text
        {"role": "assistant", "content": [
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "t1", "name": "lookup", "input": None},
            {"type": "thinking", "text": "ignored"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "Only text."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "before"},
            {"type": "tool_result", "tool_use_id": "t1",
             "content": [{"type": "text", "text": "ok"}]},
            {"type": "image", "source": "x"},
        ]},
    ])
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "plain"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Looking.", "tool_calls": [{
            "id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "assistant", "content": "Only text."},
        {"role": "user", "content": "before"},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
        {"role": "user", "content": '{"type": "image", "source": "x"}'},
    ]
