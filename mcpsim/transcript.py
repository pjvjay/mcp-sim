"""Transcript model and JSONL format (DESIGN §5).

One event per line, every event with ``t`` (ISO time) and ``kind``. The ``system`` event also
carries the run identity so a file can be read back into a :class:`Transcript` on its own.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path as FsPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mcpsim.llm import Usage
from mcpsim.mcpclient import ToolResult

Outcome = Literal["completed", "budget_exceeded", "error"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: str = Field(default_factory=now_iso)


class SystemEvent(_EventBase):
    """Prompts, models and mode used for this run, plus the run identity."""

    kind: Literal["system"] = "system"
    scenario: str = ""
    path_id: str = ""
    index: int = 0
    mode: str = ""
    models: dict[str, str] = Field(default_factory=dict)
    prompts: dict[str, str] = Field(default_factory=dict)


class UserEvent(_EventBase):
    """Simulated-user text."""

    kind: Literal["user"] = "user"
    text: str


class AssistantEvent(_EventBase):
    """An agent turn: text and/or raw ``tool_use`` blocks."""

    kind: Literal["assistant"] = "assistant"
    text: str = ""
    tool_uses: list[dict[str, Any]] = Field(default_factory=list)
    stop_reason: str | None = None


class ToolCallEvent(_EventBase):
    kind: Literal["tool_call"] = "tool_call"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_use_id: str | None = None


class ToolResultEvent(_EventBase):
    kind: Literal["tool_result"] = "tool_result"
    name: str
    is_error: bool
    structured: dict[str, Any] | list[Any] | None = None
    text: str = ""
    sha256: str = ""
    chars: int = 0
    ms: float = 0.0
    tool_use_id: str | None = None

    @classmethod
    def from_result(cls, result: ToolResult, tool_use_id: str | None = None) -> ToolResultEvent:
        return cls(
            name=result.name,
            is_error=result.is_error,
            structured=result.structured,
            text=result.text,
            sha256=result.sha256,
            chars=result.chars,
            ms=result.ms,
            tool_use_id=tool_use_id,
        )


class FinalResultEvent(_EventBase):
    kind: Literal["final_result"] = "final_result"
    parsed: dict[str, Any] | None = None
    raw: str = ""


class UsageEvent(_EventBase):
    kind: Literal["usage"] = "usage"
    per_model: dict[str, Usage] = Field(default_factory=dict)
    cost_usd: float = 0.0
    estimate: bool = True


class ErrorEvent(_EventBase):
    """A non-fatal problem worth seeing in the transcript (e.g. a malformed final block)."""

    kind: Literal["error"] = "error"
    message: str


class EndEvent(_EventBase):
    kind: Literal["end"] = "end"
    outcome: Outcome
    reason: str = ""


Event = Annotated[
    SystemEvent
    | UserEvent
    | AssistantEvent
    | ToolCallEvent
    | ToolResultEvent
    | FinalResultEvent
    | UsageEvent
    | ErrorEvent
    | EndEvent,
    Field(discriminator="kind"),
]

_event_adapter: TypeAdapter[Any] = TypeAdapter(Event)


def parse_event(data: dict[str, Any]) -> Any:
    return _event_adapter.validate_python(data)


class Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str
    path_id: str
    mode: str
    index: int
    events: list[Event] = Field(default_factory=list)
    final_result: dict[str, Any] | None = None
    outcome: Outcome = "error"
    reason: str = ""
    usage: dict[str, Usage] = Field(default_factory=dict)
    cost_usd: float = 0.0

    @property
    def stem(self) -> str:
        """``<path>-<mode>-<i>``, the file stem used under ``transcripts/`` and ``verdicts/``."""
        return f"{self.path_id}-{self.mode}-{self.index}"

    def add(self, event: Any) -> None:
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def tool_calls(self) -> list[ToolCallEvent]:
        return [e for e in self.events if isinstance(e, ToolCallEvent)]

    def tool_results(self) -> list[ToolResultEvent]:
        return [e for e in self.events if isinstance(e, ToolResultEvent)]

    def to_jsonl(self) -> str:
        return "".join(e.model_dump_json() + "\n" for e in self.events)

    def write_jsonl(self, path: str | FsPath) -> FsPath:
        fs_path = FsPath(path)
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        fs_path.write_text(self.to_jsonl(), encoding="utf-8")
        return fs_path

    @classmethod
    def from_events(cls, events: list[Any]) -> Transcript:
        """Rebuild a transcript from its events; identity comes from the ``system`` event."""
        system = next((e for e in events if isinstance(e, SystemEvent)), None)
        final = next((e for e in events if isinstance(e, FinalResultEvent)), None)
        usage = next((e for e in events if isinstance(e, UsageEvent)), None)
        end = next((e for e in events if isinstance(e, EndEvent)), None)
        return cls(
            scenario=system.scenario if system else "",
            path_id=system.path_id if system else "",
            mode=system.mode if system else "",
            index=system.index if system else 0,
            events=events,
            final_result=final.parsed if final else None,
            outcome=end.outcome if end else "error",
            reason=end.reason if end else "transcript has no end event",
            usage=dict(usage.per_model) if usage else {},
            cost_usd=usage.cost_usd if usage else 0.0,
        )

    @classmethod
    def read_jsonl(cls, path: str | FsPath) -> Transcript:
        events: list[Any] = []
        for line_no, line in enumerate(
            FsPath(path).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                events.append(parse_event(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: bad transcript event: {exc}") from exc
        return cls.from_events(events)
