"""Observers: the Informant-Report Method (DESIGN §2b).

Never ask the *subject* (the agent under test) to report its own status: its answers are shaped
by alignment training and meta-knowledge. Instead *observers* — informants, each with a social
identity or relationship to the subject (an independent auditor, a clerk, a customer) — watch
the conversation and the tool traffic and report, per declared **condition**, true / false /
unknown with a verbatim quote as evidence. Conditions are declared the way Sierra declares them,
as ``when(...)`` clauses whose **effects** enable a goal and its toolset, flag the run for the
judge, or fail it outright.

The declaration models live in :mod:`mcpsim.scenario` (they are part of the scenario file) and
are re-exported here; this module adds the Python DSL that builds the same models::

    from mcpsim.observers import observer

    auditor = observer(
        "shelf_auditor",
        identity="An independent auditor who trusts only what the store's records say.",
        watches=["tool_traffic", "final_answer"],
    )
    auditor.when("find_product has returned a DIRECT match for penne", id="direct_match").then(
        enable_tools=["get_product"], enable_goal="Quote the cheapest direct hit."
    )
    scenario = load_scenario("x.yaml").with_observers([auditor])

which mirrors the TS-style one-liner ``observer.when("See a chat with the word bear in it")
{ /* enable this goal and its toolset */ }``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mcpsim.llm import LLM, LLMResponse, Usage
from mcpsim.matcher import match
from mcpsim.scenario import (
    CHECK_SLICES,
    COUNT_OPERATORS,
    DEFAULT_TRIGGERS,
    DEFAULT_WATCHES,
    TRIGGERS,
    WATCHES,
    Check,
    Condition,
    Effect,
    Observer,
    ObserverKind,
    Scenario,
    Trigger,
    Watch,
    tool_result_slice,
)
from mcpsim.transcript import (
    EVIDENCE_LIMIT,
    NO_EVIDENCE,
    AssistantEvent,
    EndEvent,
    ErrorEvent,
    FinalResultEvent,
    GoalEnabledEvent,
    InformantReport,
    InformantReportEvent,
    SystemEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolsOfferedEvent,
    Transcript,
    UsageEvent,
    UserEvent,
)

__all__ = [
    "BUDGET_EXHAUSTED",
    "CHECK_SLICES",
    "COUNT_OPERATORS",
    "DEFAULT_OBSERVER_MAX_CALLS",
    "DEFAULT_TRIGGERS",
    "DEFAULT_WATCHES",
    "EVIDENCE_LIMIT",
    "METHOD",
    "NO_EVIDENCE",
    "OBSERVER_MAX_CALLS_ENV",
    "OMITTED",
    "REPORT_TOOL",
    "TRIGGERS",
    "WATCHES",
    "Check",
    "Condition",
    "ConditionBuilder",
    "Effect",
    "InformantReport",
    "InformantReportEvent",
    "ObservationLike",
    "Observer",
    "ObserverBuilder",
    "ObserverKind",
    "ObserverRunner",
    "ReportPayload",
    "Trigger",
    "TriggeredEffect",
    "Watch",
    "evaluate_check",
    "observer",
    "observer_max_calls",
    "observer_system_prompt",
    "report_tool",
    "render_slices",
]

OBSERVER_MAX_CALLS_ENV = "MCPSIM_OBSERVER_MAX_CALLS"
DEFAULT_OBSERVER_MAX_CALLS = 12
REPORT_TOOL = "report_conditions"
OBSERVER_MAX_TOKENS = 1024
TOOL_TEXT_LIMIT = 1200
BUDGET_EXHAUSTED = "observer budget exhausted"
OMITTED = "observer omitted this condition"
NO_MODEL = "no observer model available"
NOTHING_TO_WATCH = "(nothing to watch yet)"
METHOD = (
    "You are an informant. You report on another AI's work from what you can see; you never "
    "ask it and never take its own statements as proof of status. For each condition answer "
    "true, false or unknown and quote the exact text that proves it."
)
_WORD = re.compile(r"\S+")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_REGEX_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}


# --- the Python DSL ---------------------------------------------------------------------------


def _effect_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate effect keyword arguments eagerly so a typo fails at the call site."""
    return Effect.model_validate(kwargs).model_dump(mode="python", exclude_defaults=True)


class ConditionBuilder:
    """One ``when(...)`` clause under construction; every method returns ``self`` so effects
    chain (``.then(...).otherwise(...)``) and ``.when(...)`` opens the next clause."""

    def __init__(self, parent: ObserverBuilder, data: dict[str, Any]) -> None:
        self._parent = parent
        self._data = data

    def then(self, **effect: Any) -> ConditionBuilder:
        """The effect applied when the condition is reported true."""
        self._data["then"] = _effect_kwargs(effect)
        return self

    def otherwise(self, **effect: Any) -> ConditionBuilder:
        """The effect applied when the condition is reported false."""
        self._data["otherwise"] = _effect_kwargs(effect)
        return self

    def check(self, **check: Any) -> ConditionBuilder:
        """The code validator (``kind: code``): ``word_count=``, ``regex=``, ``tool_result=`` or
        ``tool_called=``."""
        self._data["check"] = Check.model_validate(check).model_dump(
            mode="python", exclude_none=True
        )
        return self

    def all_of(self, *refs: str) -> ConditionBuilder:
        """``kind: group``: every ``<observer>.<condition>`` (``!`` negates) must be true."""
        self._data["all_of"] = list(refs)
        return self

    def any_of(self, *refs: str) -> ConditionBuilder:
        """``kind: group``: at least one ``<observer>.<condition>`` must be true."""
        self._data["any_of"] = list(refs)
        return self

    def when(self, text: str, *, id: str, **fields: Any) -> ConditionBuilder:  # noqa: A002
        """Open the next condition on the same observer."""
        return self._parent.when(text, id=id, **fields)

    def build(self) -> Observer:
        return self._parent.build()

    @property
    def observer(self) -> ObserverBuilder:
        return self._parent


class ObserverBuilder:
    """An observer under construction; :meth:`build` validates it into an :class:`Observer`.

    ``Scenario.with_observers`` accepts builders directly (it calls ``build``).
    """

    def __init__(
        self,
        name: str,
        *,
        identity: str,
        kind: ObserverKind = "llm",
        watches: list[Watch] | None = None,
        on: list[Trigger] | None = None,
        model: str | None = None,
    ) -> None:
        self._data: dict[str, Any] = {
            "name": name,
            "identity": identity,
            "kind": kind,
            "watches": list(watches) if watches is not None else list(DEFAULT_WATCHES),
            "on": list(on) if on is not None else list(DEFAULT_TRIGGERS),
            "conditions": [],
        }
        if model is not None:
            self._data["model"] = model

    @property
    def name(self) -> str:
        return str(self._data["name"])

    def when(self, text: str, *, id: str, **fields: Any) -> ConditionBuilder:  # noqa: A002
        """Declare a condition: the natural-language conditional and its ``id``.

        Extra keyword arguments are the same fields the YAML takes (``check``, ``all_of``,
        ``any_of``, ``then``, ``otherwise``); the chained methods set them one at a time.
        """
        data: dict[str, Any] = {"id": id, "when": text, **fields}
        self._data["conditions"].append(data)
        return ConditionBuilder(self, data)

    def to_dict(self) -> dict[str, Any]:
        """The declaration as a scenario file would hold it (unvalidated)."""
        return {
            **{k: v for k, v in self._data.items() if k != "conditions"},
            "conditions": [dict(c) for c in self._data["conditions"]],
        }

    def build(self) -> Observer:
        return Observer.model_validate(self.to_dict())


def observer(
    name: str,
    *,
    identity: str,
    kind: ObserverKind = "llm",
    watches: list[Watch] | None = None,
    on: list[Trigger] | None = None,
    model: str | None = None,
) -> ObserverBuilder:
    """Start declaring an observer (see the module docstring for the shape)."""
    return ObserverBuilder(name, identity=identity, kind=kind, watches=watches, on=on, model=model)


# --- what observers see -----------------------------------------------------------------------


class ObservationLike(Protocol):
    """A scout observation as the ``scout`` slice and the code checks read it
    (:class:`mcpsim.scout.Observation` satisfies this)."""

    @property
    def kind(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> dict[str, Any]: ...

    @property
    def is_error(self) -> bool: ...

    @property
    def summary(self) -> str: ...

    @property
    def structured(self) -> dict[str, Any] | list[Any] | None: ...


def _compact(value: Any, limit: int = TOOL_TEXT_LIMIT) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clip(text: str, limit: int = TOOL_TEXT_LIMIT) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _scalar(value: Any) -> str:
    """A measured value in evidence: strings in single quotes, anything else compact JSON."""
    return f"'{value}'" if isinstance(value, str) else _compact(value)


def _evidence(text: str) -> str:
    text = " ".join(str(text).split())
    if not text:
        return NO_EVIDENCE
    return text if len(text) <= EVIDENCE_LIMIT else text[: EVIDENCE_LIMIT - 1] + "…"


def numbered_events(transcript: Transcript | None) -> list[tuple[int, Any]]:
    """``(number, event)`` for every event except ``usage``: the numbering the judge's
    transcript rendering uses, so evidence cites the same turns everywhere."""
    if transcript is None:
        return []
    numbered: list[tuple[int, Any]] = []
    for event in transcript.events:
        if isinstance(event, UsageEvent):
            continue
        numbered.append((len(numbered) + 1, event))
    return numbered


def _result_body(event: ToolResultEvent) -> str:
    if event.structured is not None:
        return _compact(event.structured)
    return _clip(event.text) or "(empty)"


def _render_event(number: int, event: Any) -> str | None:
    """One compact line per event for the ``all`` slice (``None`` for events it hides)."""
    n = f"[{number}]"
    if isinstance(event, SystemEvent):
        return f"{n} system: mode={event.mode or '?'}"
    if isinstance(event, UserEvent):
        return f"{n} user: {event.text}"
    if isinstance(event, AssistantEvent):
        names = [str(tu.get("name", "?")) for tu in event.tool_uses]
        line = f"{n} assistant: {event.text.strip() or '(no text)'}"
        return line + (f"\n    requests tool calls: {', '.join(names)}" if names else "")
    if isinstance(event, ToolCallEvent):
        return f"{n} tool_call {event.name} {_compact(event.arguments)}"
    if isinstance(event, ToolResultEvent):
        status = "ERROR" if event.is_error else "ok"
        return f"{n} tool_result {event.name} ({status}): {_result_body(event)}"
    if isinstance(event, FinalResultEvent):
        if event.parsed is not None:
            return f"{n} final_result: {_compact(event.parsed)}"
        return f"{n} final_result: (not parseable as JSON) {_clip(event.raw) or '(none)'}"
    if isinstance(event, ErrorEvent):
        return f"{n} error: {event.message}"
    if isinstance(event, ToolsOfferedEvent):
        change = []
        if event.added:
            change.append(f"added {', '.join(event.added)}")
        if event.removed:
            change.append(f"removed {', '.join(event.removed)}")
        detail = "; ".join(change) or "no change"
        return f"{n} tools offered: {detail}" + (f" ({event.reason})" if event.reason else "")
    if isinstance(event, GoalEnabledEvent):
        return f"{n} goal enabled: {event.text}" + (f" ({event.reason})" if event.reason else "")
    if isinstance(event, InformantReportEvent):
        lines = [f"{n} informant reports ({event.trigger}):"]
        lines += [f"    {r.line()}" for r in event.reports]
        return "\n".join(lines)
    if isinstance(event, EndEvent):
        return f"{n} end: outcome={event.outcome}" + (
            f" reason={event.reason}" if event.reason else ""
        )
    return None


def last_assistant_text(transcript: Transcript | None) -> str | None:
    if transcript is None:
        return None
    for event in reversed(transcript.events):
        if isinstance(event, AssistantEvent):
            return event.text
    return None


def has_final_answer(transcript: Transcript | None) -> bool:
    """A final answer exists once a ``final_result`` event was recorded (parseable or not)."""
    return transcript is not None and any(
        isinstance(e, FinalResultEvent) for e in transcript.events
    )


def final_answer_prose(text: str) -> str:
    """The final answer without its fenced blocks (what a word count should measure)."""
    return " ".join(_FENCE.sub(" ", text).split())


def render_observation(index: int, observation: ObservationLike) -> str:
    args = ", ".join(
        f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in observation.arguments.items()
    )
    head = f"{observation.name}({args})" if observation.kind == "tool" else observation.name
    status = "ERROR" if observation.is_error else "ok"
    body = observation.summary
    if observation.structured is not None:
        body = _compact(observation.structured)
    return f"[s{index}] {head} → {status}: {_clip(body) or '(empty)'}"


def render_slices(
    watches: Sequence[Watch],
    transcript: Transcript | None,
    scout: Sequence[ObservationLike] | None = None,
) -> str:
    """ONLY the watched slices, each under a heading, with the judge's turn numbers.

    ``conversation`` = user and assistant text turns; ``tool_traffic`` = tool calls and results
    (structured content as compact JSON, text truncated to :data:`TOOL_TEXT_LIMIT`);
    ``final_answer`` = the last assistant text and the parsed ``final_result``; ``scout`` = the
    scout's observations; ``all`` = everything. A slice with nothing in it yet says so.
    """
    numbered = numbered_events(transcript)
    wanted = set(watches)
    sections: list[str] = []

    def section(title: str, lines: list[str]) -> None:
        sections.append(f"## {title}\n" + ("\n".join(lines) if lines else NOTHING_TO_WATCH))

    if "all" in wanted:
        lines = [text for text in (_render_event(n, e) for n, e in numbered) if text is not None]
        section("Everything observed so far", lines)
    else:
        if "conversation" in wanted:
            lines = []
            for n, e in numbered:
                if isinstance(e, UserEvent):
                    lines.append(f"[{n}] user: {e.text}")
                elif isinstance(e, AssistantEvent):
                    lines.append(f"[{n}] assistant: {e.text.strip() or '(no text)'}")
            section("Conversation", lines)
        if "tool_traffic" in wanted:
            lines = []
            for n, e in numbered:
                if isinstance(e, ToolCallEvent | ToolResultEvent):
                    text = _render_event(n, e)
                    if text is not None:
                        lines.append(text)
            section("Tool traffic", lines)
        if "final_answer" in wanted:
            lines = []
            last: tuple[int, AssistantEvent] | None = None
            final: tuple[int, FinalResultEvent] | None = None
            for n, e in numbered:
                if isinstance(e, AssistantEvent):
                    last = (n, e)
                elif isinstance(e, FinalResultEvent):
                    final = (n, e)
            if last is not None and final is not None:
                lines.append(f"[{last[0]}] assistant: {last[1].text.strip() or '(no text)'}")
            if final is not None:
                text = _render_event(final[0], final[1])
                if text is not None:
                    lines.append(text)
            section("Final answer", lines)
    if "scout" in wanted or "all" in wanted:
        lines = [render_observation(i, o) for i, o in enumerate(scout or [], start=1)]
        section("Scout observations (read-only calls made before planning)", lines)
    return "\n\n".join(sections)


# --- code checks ------------------------------------------------------------------------------


def _compare(value: int, op: str, bound: int) -> bool:
    if op == "gt":
        return value > bound
    if op == "gte":
        return value >= bound
    if op == "lt":
        return value < bound
    if op == "lte":
        return value <= bound
    return value == bound


def _slice_text(
    of: str, transcript: Transcript | None, scout: Sequence[ObservationLike] | None
) -> tuple[list[tuple[str, str]] | None, str]:
    """``(items, empty_reason)``: the ``(where, text)`` pieces a check reads for slice ``of``.

    ``items`` is ``None`` when the slice does not exist yet (the check reports unknown);
    ``errors`` is the exception, an empty list, because "no error" is an answer.
    """
    numbered = numbered_events(transcript)
    if of == "final_answer":
        last = last_assistant_text(transcript)
        if last is None or not has_final_answer(transcript):
            return None, "no final answer yet"
        items = [("the final answer", final_answer_prose(last))]
        if transcript is not None and transcript.final_result is not None:
            items.append(("final_result", _compact(transcript.final_result, 1_000_000)))
        return items, ""
    if of == "last_assistant":
        last = last_assistant_text(transcript)
        if last is None:
            return None, "no assistant turn yet"
        return [("the last assistant turn", last)], ""
    if of == "conversation":
        items = [
            (f"turn {n}", e.text) for n, e in numbered if isinstance(e, UserEvent | AssistantEvent)
        ]
        return (items, "") if items else (None, "no conversation yet")
    if of == "errors":
        return [
            (f"error: {e.message}", e.message) for _, e in numbered if isinstance(e, ErrorEvent)
        ], ""
    tool = tool_result_slice(of)
    if tool is None:  # pragma: no cover - Check validation rejects other slices
        return None, f"unknown slice {of!r}"
    last_result = _last_tool_result(tool, transcript, scout)
    if last_result is None:
        return None, f"no result from {tool} yet"
    structured, text, _ = last_result
    body = _compact(structured, 1_000_000) if structured is not None else text
    return [(f"tool_result[{tool}]", body)], ""


def _last_tool_result(
    tool: str, transcript: Transcript | None, scout: Sequence[ObservationLike] | None
) -> tuple[dict[str, Any] | list[Any] | None, str, bool] | None:
    """``(structured, text, is_error)`` of the LAST result for ``tool``: the transcript's, else
    the scout's (plan time has no transcript); ``None`` when it was never called."""
    if transcript is not None:
        for event in reversed(transcript.events):
            if isinstance(event, ToolResultEvent) and event.name == tool:
                return event.structured, event.text, event.is_error
    for observation in reversed(list(scout or [])):
        if observation.kind == "tool" and observation.name == tool:
            return observation.structured, observation.summary, observation.is_error
    return None


def _tool_call_count(
    tool: str, transcript: Transcript | None, scout: Sequence[ObservationLike] | None
) -> int:
    count = sum(1 for c in transcript.tool_calls() if c.name == tool) if transcript else 0
    count += sum(1 for o in (scout or []) if o.kind == "tool" and o.name == tool)
    return count


def evaluate_check(
    check: Check,
    transcript: Transcript | None,
    scout: Sequence[ObservationLike] | None = None,
) -> tuple[bool | None, str]:
    """Run one ``kind: code`` validator: ``(value, evidence)``, evidence being the measured value
    (``"112 words"``, ``"matched 'bear' at turn 3"``, ``"find_product.match == 'direct'"``)."""
    if check.word_count is not None:
        spec = check.word_count
        op = next(k for k in spec if k in COUNT_OPERATORS)
        items, why = _slice_text(str(spec["of"]), transcript, scout)
        if items is None:
            return None, why
        if spec["of"] == "final_answer":
            items = items[:1]  # the prose, not the final_result JSON that follows it
        words = sum(len(_WORD.findall(text)) for _, text in items)
        return _compare(words, op, int(spec[op])), f"{words} words"
    if check.regex is not None:
        spec = check.regex
        flags = 0
        for ch in str(spec.get("flags", "")):
            flags |= _REGEX_FLAGS[ch]
        pattern = re.compile(str(spec["pattern"]), flags)
        items, why = _slice_text(str(spec["of"]), transcript, scout)
        if items is None:
            return None, why
        for where, text in items:
            m = pattern.search(text)
            if m:
                shown = f"matched {spec['pattern']!r} at {where}"
                if where.startswith("error: "):
                    shown = f"matched {spec['pattern']!r} in {where}"
                elif not where.startswith("turn "):
                    shown = f"matched {spec['pattern']!r} in {where}"
                return True, _evidence(shown)
        scope = f"{len(items)} error(s)" if spec["of"] == "errors" else str(spec["of"])
        return False, f"no match for {spec['pattern']!r} in {scope}"
    if check.tool_result is not None:
        tool = str(check.tool_result["tool"])
        where = check.tool_result["where"]
        last = _last_tool_result(tool, transcript, scout)
        if last is None:
            return None, f"no result from {tool} yet"
        structured, text, is_error = last
        if is_error:
            return False, _evidence(f"{tool} returned an error: {text or '(no message)'}")
        if structured is None:
            return None, f"{tool} returned text only, nothing structured to check"
        matches = match(where, structured)
        failed = [m for m in matches if not m.passed]
        if failed:
            first = failed[0]
            return False, _evidence(
                f"{tool}.{first.path} == {_scalar(first.actual)} "
                f"(expected {first.op} {_scalar(first.expected)})"
            )
        shown = ", ".join(f"{tool}.{mt.path} == {_scalar(mt.actual)}" for mt in matches)
        return True, _evidence(shown)
    tool = str(check.tool_called)
    count = _tool_call_count(tool, transcript, scout)
    if count:
        return True, f"{tool} called {count} time(s)"
    return False, f"{tool} not called"


# --- the runner -------------------------------------------------------------------------------


class ReportItem(BaseModel):
    """One condition as an LLM observer reports it through :data:`REPORT_TOOL`."""

    model_config = ConfigDict(extra="ignore")

    condition_id: str = Field(description="The id of the condition this report answers.")
    value: bool | None = Field(
        description="true when the condition holds, false when it does not, null when you "
        "cannot tell from what you can see."
    )
    evidence: str = Field(
        default=NO_EVIDENCE,
        description="The exact text (with its [turn] number) that proves the answer, or "
        "exactly 'no evidence'.",
    )
    confidence: float = Field(default=1.0, description="0.0 to 1.0.")


class ReportPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reports: list[ReportItem] = Field(default_factory=list)


def report_tool() -> dict[str, Any]:
    """The single forced tool an LLM observer answers with."""
    return {
        "name": REPORT_TOOL,
        "description": (
            "Report, for every condition listed in your instructions, whether it holds "
            "(true), does not hold (false) or cannot be told from what you watched (null), "
            "with a verbatim quote as evidence. Call it exactly once."
        ),
        "input_schema": ReportPayload.model_json_schema(),
    }


def observer_system_prompt(observer: Observer) -> str:
    """Identity, the method, and the conditions to report on."""
    conditions = "\n".join(f"- {c.id}: {c.when.strip()}" for c in observer.conditions)
    return "\n\n".join(
        [
            f"## Who you are\n{observer.identity.strip()}",
            f"## The method\n{METHOD}",
            "## Conditions to report on (answer every one by its id)\n" + conditions,
            "You see only the parts of the transcript you are allowed to watch, numbered "
            "[n]; quote evidence with that number. Answer unknown (null) rather than guess "
            f"when what you watch does not settle a condition. Respond ONLY by calling the "
            f"`{REPORT_TOOL}` tool once with one entry per condition.",
        ]
    )


def observer_max_calls(default: int = DEFAULT_OBSERVER_MAX_CALLS) -> int:
    """``MCPSIM_OBSERVER_MAX_CALLS`` (default 12): LLM observer calls allowed per run."""
    raw = os.environ.get(OBSERVER_MAX_CALLS_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{OBSERVER_MAX_CALLS_ENV} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{OBSERVER_MAX_CALLS_ENV} must be >= 0, got {value}")
    return value


@dataclass(frozen=True)
class TriggeredEffect:
    """An effect a report triggered: ``then`` of a true report, ``otherwise`` of a false one."""

    observer: str
    condition: str
    effect: Effect
    report: InformantReport

    @property
    def reason(self) -> str:
        return f"observer:{self.observer}.{self.condition}"

    @property
    def failure(self) -> str:
        """The ``hard_failures`` entry: ``<observer>.<condition> — <evidence>``."""
        return f"{self.observer}.{self.condition} — {self.report.evidence}"


@dataclass
class ObserverRunner:
    """Asks the scenario's observers for reports at a trigger (DESIGN §2b).

    ``llm`` serves ``kind: llm`` observers (each under ``models.model_for_observer``); with
    ``include_llm=False`` (the dry run) only code and group observers run. ``max_calls`` caps
    LLM observer calls per run (default from ``MCPSIM_OBSERVER_MAX_CALLS``); past it an LLM
    observer reports unknown with evidence :data:`BUDGET_EXHAUSTED`, never silently. Effects are
    never applied here: :meth:`effects` lists them and the caller (agent loop, dry run, scout)
    applies them after recording the reports.
    """

    scenario: Scenario
    llm: LLM | None = None
    include_llm: bool = True
    max_calls: int | None = None
    calls: int = 0
    usage: dict[str, Usage] = field(default_factory=dict)
    latest: dict[tuple[str, str], InformantReport] = field(default_factory=dict)
    _pending_usage: dict[str, Usage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_calls is None:
            self.max_calls = observer_max_calls()

    @property
    def observers(self) -> list[Observer]:
        return [o for o in self.scenario.observers if self.include_llm or o.kind != "llm"]

    def observers_for(self, trigger: Trigger) -> list[Observer]:
        return [o for o in self.observers if trigger in o.on]

    def fires_at(self, trigger: Trigger) -> bool:
        return bool(self.observers_for(trigger))

    def take_usage(self) -> dict[str, Usage]:
        """Usage accrued since the last call (the agent loop folds it into the run's)."""
        taken, self._pending_usage = self._pending_usage, {}
        return taken

    async def report(
        self,
        trigger: Trigger,
        transcript: Transcript | None = None,
        scout: Sequence[ObservationLike] | None = None,
    ) -> list[InformantReport]:
        """Every report of every observer whose ``on`` contains ``trigger``, in declaration
        order (groups see the reports made just before them)."""
        at_event = len(transcript.events) - 1 if transcript is not None else -1
        reports: list[InformantReport] = []
        for observer in self.observers_for(trigger):
            if observer.kind == "code":
                batch = self._code_reports(observer, trigger, at_event, transcript, scout)
            elif observer.kind == "group":
                batch = self._group_reports(observer, trigger, at_event)
            else:
                batch = await self._llm_reports(observer, trigger, at_event, transcript, scout)
            for r in batch:
                self.latest[(r.observer, r.condition)] = r
            reports.extend(batch)
        return reports

    def effects(self, reports: Sequence[InformantReport]) -> list[TriggeredEffect]:
        """``then`` for every true report, ``otherwise`` for every false one; unknown → nothing."""
        found: list[TriggeredEffect] = []
        for r in reports:
            if r.value is None:
                continue
            condition = self.scenario.observer(r.observer).condition(r.condition)
            effect = condition.then if r.value else condition.otherwise
            if not effect.is_empty():
                found.append(TriggeredEffect(r.observer, r.condition, effect, r))
        return found

    # -- code ---------------------------------------------------------------------------------

    def _code_reports(
        self,
        observer: Observer,
        trigger: Trigger,
        at_event: int,
        transcript: Transcript | None,
        scout: Sequence[ObservationLike] | None,
    ) -> list[InformantReport]:
        reports: list[InformantReport] = []
        for condition in observer.conditions:
            assert condition.check is not None  # Observer validation guarantees it
            value, evidence = evaluate_check(condition.check, transcript, scout)
            reports.append(
                InformantReport(
                    observer=observer.name,
                    condition=condition.id,
                    value=value,
                    evidence=_evidence(evidence),
                    confidence=1.0,
                    trigger=trigger,
                    at_event=at_event,
                )
            )
        return reports

    # -- group --------------------------------------------------------------------------------

    def _group_reports(
        self, observer: Observer, trigger: Trigger, at_event: int
    ) -> list[InformantReport]:
        reports: list[InformantReport] = []
        for condition in observer.conditions:
            value, evidence, confidence = self._combine(condition)
            reports.append(
                InformantReport(
                    observer=observer.name,
                    condition=condition.id,
                    value=value,
                    evidence=_evidence(evidence),
                    confidence=confidence,
                    trigger=trigger,
                    at_event=at_event,
                )
            )
        return reports

    def _term(self, observer: str, condition: str, negated: bool) -> tuple[bool | None, float]:
        latest = self.latest.get((observer, condition))
        if latest is None or latest.value is None:
            return None, latest.confidence if latest else 1.0
        return (not latest.value if negated else latest.value), latest.confidence

    def _combine(self, condition: Condition) -> tuple[bool | None, str, float]:
        """Three-valued ``all_of`` AND ``any_of``; unknown propagates unless decidable."""
        shown: list[str] = []
        confidences: list[float] = []

        def values(refs: list[str]) -> list[bool | None]:
            out: list[bool | None] = []
            for ref in refs:
                negated = ref.startswith("!")
                observer, _, cond = ref.lstrip("!").partition(".")
                value, confidence = self._term(observer, cond, negated)
                confidences.append(confidence)
                label = "unknown" if value is None else str(value).lower()
                shown.append(f"{ref} is {label}")
                out.append(value)
            return out

        parts: list[bool | None] = []
        if condition.all_of:
            every = values(condition.all_of)
            if any(v is False for v in every):
                parts.append(False)
            elif all(v is True for v in every):
                parts.append(True)
            else:
                parts.append(None)
        if condition.any_of:
            some = values(condition.any_of)
            if any(v is True for v in some):
                parts.append(True)
            elif all(v is False for v in some):
                parts.append(False)
            else:
                parts.append(None)
        if any(p is False for p in parts):
            value: bool | None = False
        elif all(p is True for p in parts):
            value = True
        else:
            value = None
        return value, ", ".join(shown), min(confidences) if confidences else 1.0

    # -- llm ----------------------------------------------------------------------------------

    def _unknown(
        self, observer: Observer, trigger: Trigger, at_event: int, evidence: str
    ) -> list[InformantReport]:
        return [
            InformantReport(
                observer=observer.name,
                condition=c.id,
                value=None,
                evidence=_evidence(evidence),
                confidence=0.0,
                trigger=trigger,
                at_event=at_event,
            )
            for c in observer.conditions
        ]

    async def _llm_reports(
        self,
        observer: Observer,
        trigger: Trigger,
        at_event: int,
        transcript: Transcript | None,
        scout: Sequence[ObservationLike] | None,
    ) -> list[InformantReport]:
        if self.llm is None:
            return self._unknown(observer, trigger, at_event, NO_MODEL)
        assert self.max_calls is not None
        if self.calls >= self.max_calls:
            return self._unknown(observer, trigger, at_event, BUDGET_EXHAUSTED)
        self.calls += 1
        model = self.scenario.models.model_for_observer(observer)
        try:
            response = await self.llm.complete(
                model=model,
                system=observer_system_prompt(observer),
                messages=[
                    {"role": "user", "content": render_slices(observer.watches, transcript, scout)}
                ],
                tools=[report_tool()],
                tool_choice={"type": "tool", "name": REPORT_TOOL},
                max_tokens=OBSERVER_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 - an observer must not crash the run
            text = str(exc).strip()
            return self._unknown(
                observer,
                trigger,
                at_event,
                f"observer call failed: {type(exc).__name__}" + (f": {text}" if text else ""),
            )
        self._record_usage(model, response.usage)
        return parse_reports(observer, response, trigger=trigger, at_event=at_event)

    def _record_usage(self, model: str, usage: Usage) -> None:
        self.usage[model] = self.usage.get(model, Usage()) + usage
        self._pending_usage[model] = self._pending_usage.get(model, Usage()) + usage


def parse_reports(
    observer: Observer, response: LLMResponse, *, trigger: Trigger, at_event: int
) -> list[InformantReport]:
    """The observer's reply → one report per declared condition.

    A condition the reply omits is unknown with evidence :data:`OMITTED`; a reply that did not
    call the tool, or whose payload is invalid, makes every condition unknown with the parse
    error as evidence. Evidence is clipped to :data:`EVIDENCE_LIMIT`, confidence to 0–1.
    """

    def unknown(evidence: str) -> list[InformantReport]:
        return [
            InformantReport(
                observer=observer.name,
                condition=c.id,
                value=None,
                evidence=_evidence(evidence),
                confidence=0.0,
                trigger=trigger,
                at_event=at_event,
            )
            for c in observer.conditions
        ]

    blocks = [b for b in response.tool_uses() if b.get("name") == REPORT_TOOL]
    if not blocks:
        return unknown(
            f"malformed observer reply: did not call {REPORT_TOOL} "
            f"(stop_reason={response.stop_reason})"
        )
    payload = blocks[0].get("input")
    if not isinstance(payload, dict):
        return unknown(f"malformed observer reply: {REPORT_TOOL} input is not an object")
    try:
        parsed = ReportPayload.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        return unknown(f"malformed observer reply: {problems}")
    by_id = {item.condition_id: item for item in parsed.reports}
    reports: list[InformantReport] = []
    for condition in observer.conditions:
        item = by_id.get(condition.id)
        if item is None:
            reports.append(
                InformantReport(
                    observer=observer.name,
                    condition=condition.id,
                    value=None,
                    evidence=OMITTED,
                    confidence=0.0,
                    trigger=trigger,
                    at_event=at_event,
                )
            )
            continue
        confidence = min(1.0, max(0.0, float(item.confidence)))
        reports.append(
            InformantReport(
                observer=observer.name,
                condition=condition.id,
                value=item.value,
                evidence=_evidence(item.evidence),
                confidence=confidence,
                trigger=trigger,
                at_event=at_event,
            )
        )
    return reports
