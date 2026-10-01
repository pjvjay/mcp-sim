"""A scripted :class:`mcpsim.llm.LLM` for tests: returns queued responses, records every call."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from mcpsim.llm import DEFAULT_MAX_TOKENS, LLMResponse, Usage


def text_response(
    text: str, *, stop_reason: str = "end_turn", usage: Usage | None = None, model: str = "fake"
) -> LLMResponse:
    return LLMResponse(
        content=[{"type": "text", "text": text}],
        stop_reason=stop_reason,
        usage=usage or Usage(input_tokens=10, output_tokens=5),
        model=model,
    )


def tool_use_response(
    name: str,
    arguments: dict[str, Any],
    *,
    tool_use_id: str = "toolu_1",
    text: str | None = None,
    usage: Usage | None = None,
    model: str = "fake",
) -> LLMResponse:
    """An assistant turn that calls one tool (optionally with text before it)."""
    content: list[dict[str, Any]] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    content.append({"type": "tool_use", "id": tool_use_id, "name": name, "input": arguments})
    return LLMResponse(
        content=content,
        stop_reason="tool_use",
        usage=usage or Usage(input_tokens=10, output_tokens=5),
        model=model,
    )


def structured_response(name: str, payload: dict[str, Any], *, model: str = "fake") -> LLMResponse:
    """What a forced ``tool_choice`` structured-output call returns (planner / judge)."""
    return tool_use_response(name, payload, tool_use_id="toolu_structured", model=model)


class ScriptedLLM:
    """Pops one queued :class:`LLMResponse` per ``complete`` call and records the call."""

    def __init__(self, responses: Iterable[LLMResponse] = ()) -> None:
        self.queue: deque[LLMResponse] = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def push(self, *responses: LLMResponse) -> None:
        self.queue.extend(responses)

    @property
    def remaining(self) -> int:
        return len(self.queue)

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
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "max_tokens": max_tokens,
            }
        )
        if not self.queue:
            raise RuntimeError(
                f"ScriptedLLM: no scripted response left for call #{len(self.calls)} "
                f"(model={model!r})"
            )
        return self.queue.popleft()
