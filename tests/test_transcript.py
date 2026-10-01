from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcpsim.llm import Usage
from mcpsim.mcpclient import ToolResult
from mcpsim.transcript import (
    AssistantEvent,
    EndEvent,
    ErrorEvent,
    FinalResultEvent,
    SystemEvent,
    ToolCallEvent,
    ToolResultEvent,
    Transcript,
    UsageEvent,
    UserEvent,
    parse_event,
)


def _transcript() -> Transcript:
    t = Transcript(scenario="fake-lookup", path_id="happy", mode="guided", index=0)
    t.add(
        SystemEvent(
            scenario="fake-lookup",
            path_id="happy",
            index=0,
            mode="guided",
            models={"agent": "claude-sonnet-5-5"},
            prompts={"agent": "You are...", "user": "You play..."},
        )
    )
    t.add(UserEvent(text="How much is penne?"))
    t.add(
        AssistantEvent(
            text="Let me look.",
            tool_uses=[{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}],
            stop_reason="tool_use",
        )
    )
    t.add(ToolCallEvent(name="lookup", arguments={"slug": "penne"}, tool_use_id="toolu_1"))
    result = ToolResult(
        name="lookup",
        is_error=False,
        structured={"slug": "penne", "price": 2.49},
        text='{"slug": "penne"}',
        ms=1.25,
        sha256="ab" * 32,
        chars=17,
    )
    t.add(ToolResultEvent.from_result(result, tool_use_id="toolu_1"))
    t.add(
        AssistantEvent(text='```json final_result\n{"slug": "penne"}\n```', stop_reason="end_turn")
    )
    t.add(FinalResultEvent(parsed={"slug": "penne"}, raw='{"slug": "penne"}'))
    t.add(ErrorEvent(message="just a note"))
    usage = {"claude-sonnet-5-5": Usage(input_tokens=100, output_tokens=20)}
    t.add(UsageEvent(per_model=usage, cost_usd=0.0006))
    t.add(EndEvent(outcome="completed", reason="final answer delivered"))
    t.final_result = {"slug": "penne"}
    t.outcome = "completed"
    t.reason = "final answer delivered"
    t.usage = usage
    t.cost_usd = 0.0006
    return t


def test_event_kinds_and_helpers() -> None:
    t = _transcript()
    assert t.kinds() == [
        "system",
        "user",
        "assistant",
        "tool_call",
        "tool_result",
        "assistant",
        "final_result",
        "error",
        "usage",
        "end",
    ]
    assert t.stem == "happy-guided-0"
    assert [c.name for c in t.tool_calls()] == ["lookup"]
    assert t.tool_results()[0].structured == {"slug": "penne", "price": 2.49}
    assert t.tool_results()[0].tool_use_id == "toolu_1"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    t = _transcript()
    path = t.write_jsonl(tmp_path / "transcripts" / "happy-guided-0.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    for line in lines:
        event = json.loads(line)
        assert "t" in event and "kind" in event
    back = Transcript.read_jsonl(path)
    assert back.scenario == "fake-lookup"
    assert (back.path_id, back.mode, back.index) == ("happy", "guided", 0)
    assert back.kinds() == t.kinds()
    assert back.final_result == {"slug": "penne"}
    assert back.outcome == "completed" and back.reason == "final answer delivered"
    assert back.usage == t.usage and back.cost_usd == pytest.approx(0.0006)
    assert back.model_dump() == t.model_dump()


def test_incomplete_transcript_reads_as_error() -> None:
    t = Transcript.from_events([SystemEvent(scenario="s", path_id="p", mode="free", index=2)])
    assert t.outcome == "error" and "no end event" in t.reason
    assert (t.scenario, t.path_id, t.mode, t.index) == ("s", "p", "free", 2)
    assert t.final_result is None and t.usage == {}


def test_bad_line_is_reported_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "user", "text": "ok", "t": "2026-01-01T00:00:00Z"}\n{nope\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2: bad transcript event"):
        Transcript.read_jsonl(path)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_event({"kind": "mystery", "t": "2026-01-01T00:00:00Z"})


def test_parse_event_discriminates_on_kind() -> None:
    e = parse_event({"kind": "end", "t": "2026-01-01T00:00:00Z", "outcome": "budget_exceeded"})
    assert isinstance(e, EndEvent) and e.outcome == "budget_exceeded" and e.reason == ""
    with pytest.raises(ValueError):
        parse_event({"kind": "end", "t": "2026-01-01T00:00:00Z", "outcome": "crashed"})
