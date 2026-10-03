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
from mcpsim.scenario import Trigger

Outcome = Literal["completed", "budget_exceeded", "error"]
NO_EVIDENCE = "no evidence"
EVIDENCE_LIMIT = 300


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


class ToolsOfferedEvent(_EventBase):
    """The set of tools the agent is offered changed (DESIGN §2 "Tool scoping and disclosure").

    ``reason`` says why: ``initial:<mode>:<disclosure>`` on turn one,
    ``discover_tools:<query>`` when the agent asked for more, or whatever an observer passed to
    ``offer_tools``. The judge renders the running set after each change.
    """

    kind: Literal["tools_offered"] = "tools_offered"
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    reason: str = ""


class GoalEnabledEvent(_EventBase):
    """An observer added a goal mid-run (``enable_goal``); the agent reads it next turn.

    ``observer`` / ``condition`` name the report that enabled it (empty for a goal enabled by
    hand through :class:`~mcpsim.agent.LiveRun`); ``reason`` is ``observer:<obs>.<cond>``.
    """

    kind: Literal["goal_enabled"] = "goal_enabled"
    text: str
    reason: str = ""
    observer: str = ""
    condition: str = ""


class InformantReport(BaseModel):
    """One observer's answer to one condition at one trigger (DESIGN §2b).

    ``value`` is ``True`` / ``False`` / ``None`` (cannot tell from what it watches);
    ``evidence`` is a verbatim quote (at most :data:`EVIDENCE_LIMIT` characters) or
    ``"no evidence"``; ``confidence`` is 0–1 (code and group observers report 1.0);
    ``at_event`` is the index of the transcript event that triggered the report.
    """

    model_config = ConfigDict(extra="forbid")

    observer: str
    condition: str
    value: bool | None
    evidence: str = NO_EVIDENCE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    trigger: Trigger
    at_event: int = Field(default=-1, ge=-1)

    @property
    def key(self) -> str:
        return f"{self.observer}.{self.condition}"

    @property
    def shown(self) -> str:
        """``true`` / ``false`` / ``unknown``."""
        return "unknown" if self.value is None else str(self.value).lower()

    def line(self) -> str:
        """``shelf_auditor.direct_match = true — "match": "direct" …`` (the planner's format)."""
        return f"{self.key} = {self.shown} — {self.evidence}"


class InformantReportEvent(_EventBase):
    """A batch of informant reports at one trigger, recorded BEFORE any effect is applied.

    ``flags`` and ``failures`` are the ``flag`` / ``fail`` effects these reports triggered
    (``failures`` entries read ``<observer>.<condition> — <evidence>``), so a saved transcript
    carries them and :meth:`Transcript.from_events` rebuilds ``flags`` / ``hard_failures``.
    """

    kind: Literal["informant_report"] = "informant_report"
    trigger: Trigger
    reports: list[InformantReport] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


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
    | ToolsOfferedEvent
    | GoalEnabledEvent
    | InformantReportEvent
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
    # Observer effects (DESIGN §2b): ``flag`` names, and ``fail`` effects as
    # ``<observer>.<condition> — <evidence>``; any hard failure fails the run in the judge.
    flags: list[str] = Field(default_factory=list)
    hard_failures: list[str] = Field(default_factory=list)

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

    def informant_reports(self) -> list[InformantReport]:
        """Every report in the order it was recorded (across all triggers)."""
        return [r for e in self.events if isinstance(e, InformantReportEvent) for r in e.reports]

    def add_reports(
        self,
        trigger: Trigger,
        reports: list[InformantReport],
        *,
        flags: list[str] | None = None,
        failures: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> InformantReportEvent:
        """Record a batch of reports and the flag / fail / note effects they triggered."""
        event = InformantReportEvent(
            trigger=trigger,
            reports=list(reports),
            flags=list(flags or []),
            failures=list(failures or []),
            notes=list(notes or []),
        )
        self.add(event)
        for flag in event.flags:
            if flag not in self.flags:
                self.flags.append(flag)
        for failure in event.failures:
            if failure not in self.hard_failures:
                self.hard_failures.append(failure)
        return event

    def span(self) -> tuple[datetime, datetime] | None:
        """``(first event time, last event time)``, or ``None`` without two parseable times."""
        times: list[datetime] = []
        for e in self.events:
            try:
                times.append(datetime.fromisoformat(e.t))
            except (TypeError, ValueError):
                continue
        if len(times) < 2:
            return None
        return min(times), max(times)

    @property
    def duration_s(self) -> float:
        """Seconds from the first event to the last (0.0 when the times are missing)."""
        span = self.span()
        if span is None:
            return 0.0
        return round(max(0.0, (span[1] - span[0]).total_seconds()), 3)

    def tools_offered(self) -> list[str]:
        """The tools offered at the end of the run, in the order they were offered."""
        offered: list[str] = []
        for e in self.events:
            if isinstance(e, ToolsOfferedEvent):
                offered = [n for n in offered if n not in e.removed]
                offered.extend(n for n in e.added if n not in offered)
        return offered

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
        flags: list[str] = []
        failures: list[str] = []
        for e in events:
            if isinstance(e, InformantReportEvent):
                flags.extend(f for f in e.flags if f not in flags)
                failures.extend(f for f in e.failures if f not in failures)
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
            flags=flags,
            hard_failures=failures,
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
