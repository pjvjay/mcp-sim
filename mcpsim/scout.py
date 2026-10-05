"""Scout: read-only observations before planning (DESIGN §2b "Scout → orchestrator").

The scout is the observers' hands. Before the planner runs it makes a bounded set of cheap,
read-only calls against the live server — never a write tool
(:func:`mcpsim.scoping.is_write_tool`), never a tool whose description claims a cost
(:func:`mcpsim.planner.is_expensive`), never more than ``max(2, budgets.max_tool_calls // 2)``
tool calls — in this order:

1. **Resources**: every static resource URI in the catalog (not templates), summarised: mime
   type, size, and for JSON the top-level type, keys, list lengths and the first items.
2. **Expected-outcome lookups**: every disclosed tool with a string property whose name is a
   top-level ``expected_outcome.json`` key with a plain string value is called with that value
   (``find_product(query="penne")``), other required arguments defaulted from the schema.
3. **Zero-argument read tools** among the disclosed set, most relevant first.

The *disclosed* set is what the agent would see on turn one (``all``: everything; otherwise the
explicit ``initial`` globs or :func:`mcpsim.scoping.initial_tools`). The scenario's observers
then report at the ``scout`` trigger from the observations alone (an observer at scout time
sees no conversation); their effects grow or shrink the disclosed set, enable goals and set
flags. Tools an effect enabled get one pass of steps 2–3 (no unbounded loops). The result is
saved as ``scout.json`` beside ``plan.json`` and is what the orchestrating planner reads instead
of raw tool output.
"""

from __future__ import annotations

import json
from pathlib import Path as FsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mcpsim.mcpclient import Catalog, Session, ToolInfo
from mcpsim.observers import ObserverRunner
from mcpsim.planner import dry_run_arguments, is_expensive
from mcpsim.scenario import Scenario
from mcpsim.scoping import (
    initial_tools,
    is_write_tool,
    rank_tools,
    scenario_terms,
)
from mcpsim.transcript import (
    Event,
    GoalEnabledEvent,
    InformantReport,
    InformantReportEvent,
    ToolsOfferedEvent,
)

SCOUT_FILE = "scout.json"
MIN_SCOUT_CALLS = 2
SUMMARY_LIMIT = 600
RESOURCE_BODY_LIMIT = 50_000
_KEYS_SHOWN = 12
_ITEMS_SHOWN = 3
_ITEM_CHARS = 120


class Observation(BaseModel):
    """One read-only call: a tool with its arguments, or a static resource read.

    ``summary`` is the compact description the planner sees (at most :data:`SUMMARY_LIMIT`
    characters); ``structured`` keeps a tool's structured content in full so code observers'
    ``tool_result`` checks and the planner's observation lines read real values.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool", "resource"]
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    is_error: bool = False
    summary: str = ""
    structured_keys: list[str] = Field(default_factory=list)
    structured: dict[str, Any] | list[Any] | None = None
    ms: float = 0.0
    chars: int = 0
    mime_type: str | None = None
    from_expected: list[str] = Field(default_factory=list)

    def call_label(self) -> str:
        """``find_product(query="penne")`` or the resource URI."""
        if self.kind != "tool":
            return self.name
        args = ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in self.arguments.items()
        )
        return f"{self.name}({args})"


class ScoutResult(BaseModel):
    """What the scout saw and what the observers made of it (``scout.json``)."""

    model_config = ConfigDict(extra="forbid")

    observations: list[Observation] = Field(default_factory=list)
    disclosed: list[str] = Field(default_factory=list)
    on_request: list[str] = Field(default_factory=list)
    reports: list[InformantReport] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    tool_calls: int = 0
    budget: int = 0
    planner_prompt_chars: int = 0

    def tool_observations(self) -> list[Observation]:
        return [o for o in self.observations if o.kind == "tool"]

    def lookups(self) -> list[Observation]:
        """The expected-outcome lookups that answered without an error (proven calls)."""
        return [o for o in self.tool_observations() if o.from_expected and not o.is_error]

    def save(self, path: str | FsPath) -> FsPath:
        fs_path = FsPath(path)
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        fs_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return fs_path

    @classmethod
    def load(cls, path: str | FsPath) -> ScoutResult:
        return cls.model_validate_json(FsPath(path).read_text(encoding="utf-8"))


def scout_budget(scenario: Scenario) -> int:
    """``max(2, budgets.max_tool_calls // 2)`` tool calls."""
    return max(MIN_SCOUT_CALLS, scenario.budgets.max_tool_calls // 2)


def should_scout(scenario: Scenario) -> bool:
    """The scout runs whenever disclosure is not ``all`` (there is something to disclose)."""
    return scenario.tools.disclosure != "all"


def initial_disclosed(scenario: Scenario, catalog: Catalog) -> list[str]:
    """The tools the agent would see on turn one, before any path or observer widens it."""
    policy = scenario.tools
    if policy.disclosure == "all":
        return catalog.tool_names()
    if policy.disclosure == "progressive" and policy.initial is not None:
        return [t.name for t in catalog.select(policy.initial)]
    return [t.name for t in initial_tools(scenario, catalog)]


# --- summaries --------------------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compact(value: Any, limit: int) -> str:
    return _clip(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str), limit)


def summarise_json(value: Any, limit: int = SUMMARY_LIMIT) -> str:
    """Top-level type, keys (first 12), list lengths, first 3 items (120 chars each)."""
    if isinstance(value, dict):
        keys = list(value)
        shown = ", ".join(str(k) for k in keys[:_KEYS_SHOWN])
        if len(keys) > _KEYS_SHOWN:
            shown += f", … ({len(keys) - _KEYS_SHOWN} more)"
        parts = [f"object with keys {shown}" if keys else "empty object"]
        for key in keys:
            inner = value[key]
            if isinstance(inner, list):
                parts.append(f"{key}: list of {len(inner)}")
                parts.extend(
                    f"{key}[{i}]={_compact(item, _ITEM_CHARS)}"
                    for i, item in enumerate(inner[:_ITEMS_SHOWN])
                )
            elif isinstance(inner, dict):
                parts.append(f"{key}={_compact(inner, _ITEM_CHARS)}")
            else:
                parts.append(f"{key}={_compact(inner, _ITEM_CHARS)}")
        return _clip("; ".join(parts), limit)
    if isinstance(value, list):
        parts = [f"list of {len(value)}"]
        parts.extend(
            f"[{i}]={_compact(item, _ITEM_CHARS)}" for i, item in enumerate(value[:_ITEMS_SHOWN])
        )
        return _clip("; ".join(parts), limit)
    return _compact(value, limit)


def structured_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(k) for k in value]
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [str(k) for k in value[0]]
    return []


# --- what to call -----------------------------------------------------------------------------


def lookup_arguments(
    tool: ToolInfo, spec: dict[str, Any] | None
) -> tuple[dict[str, Any], list[str]] | None:
    """Arguments for an expected-outcome lookup, or ``None`` when the tool has no string
    property pinned by a plain string value in ``expected_outcome.json`` (or a required
    argument has no default). The same filler the dry-run plan uses."""
    arguments, _, from_expected = dry_run_arguments(tool, spec)
    if arguments is None:
        return None
    strings = [name for name in from_expected if isinstance(arguments.get(name), str)]
    if not strings:
        return None
    return arguments, from_expected


def is_zero_argument(tool: ToolInfo) -> bool:
    required = (tool.input_schema or {}).get("required")
    return not (isinstance(required, list) and required)


# --- the scout --------------------------------------------------------------------------------


class _Scouting:
    def __init__(
        self,
        scenario: Scenario,
        catalog: Catalog,
        session: Session,
        observers: ObserverRunner | None,
        budget: int,
    ) -> None:
        self.scenario = scenario
        self.catalog = catalog
        self.session = session
        self.observers = observers
        self.budget = budget
        self.result = ScoutResult(disclosed=initial_disclosed(scenario, catalog), budget=budget)
        self.called: set[str] = set()

    @property
    def observations(self) -> list[Observation]:
        return self.result.observations

    def remaining(self) -> int:
        return self.budget - self.result.tool_calls

    async def read_resources(self) -> None:
        seen: set[str] = set()
        for res in self.catalog.resources[: self.budget]:
            if res.uri in seen:
                continue
            seen.add(res.uri)
            content = await self.session.read_resource(res.uri)
            structured: dict[str, Any] | list[Any] | None = None
            if content.chars <= RESOURCE_BODY_LIMIT:
                try:
                    parsed = json.loads(content.text) if content.text.strip() else None
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict | list):
                    structured = parsed
                    summary = summarise_json(parsed)
                else:
                    summary = _clip(content.text, SUMMARY_LIMIT) or "(empty)"
            else:
                summary = f"({content.chars} chars; body not read)"
            self.observations.append(
                Observation(
                    kind="resource",
                    name=res.uri,
                    summary=f"{content.mime_type or 'text'}, {content.chars} chars: {summary}",
                    structured_keys=structured_keys(structured),
                    structured=structured,
                    ms=content.ms,
                    chars=content.chars,
                    mime_type=content.mime_type,
                )
            )

    async def call(
        self, tool: ToolInfo, arguments: dict[str, Any], from_expected: list[str]
    ) -> None:
        result = await self.session.call_tool(tool.name, arguments)
        self.result.tool_calls += 1
        self.called.add(tool.name)
        if result.structured is not None:
            summary = summarise_json(result.structured)
        else:
            summary = _clip(result.text, SUMMARY_LIMIT) or (
                "(error, no message)" if result.is_error else "(empty)"
            )
        self.observations.append(
            Observation(
                kind="tool",
                name=tool.name,
                arguments=dict(arguments),
                is_error=result.is_error,
                summary=summary,
                structured_keys=structured_keys(result.structured),
                structured=result.structured,
                ms=result.ms,
                chars=result.chars,
                from_expected=list(from_expected),
            )
        )

    def _readable(self, names: list[str]) -> list[ToolInfo]:
        """The disclosed tools the scout may call: never a write tool, never one whose
        description claims a cost."""
        tools = [self.catalog.tool(n) for n in names if self.catalog.has_tool(n)]
        return [t for t in tools if not is_write_tool(t) and not is_expensive(t)]

    async def observe_tools(self, names: list[str]) -> None:
        """Steps 2 and 3 over ``names`` (disclosed, readable, not yet called), within budget."""
        tools = [t for t in self._readable(names) if t.name not in self.called]
        spec = self.scenario.expected_outcome.json
        for tool in tools:
            if self.remaining() <= 0:
                return
            found = lookup_arguments(tool, spec)
            if found is not None:
                await self.call(tool, found[0], found[1])
        terms = scenario_terms(self.scenario)
        zero = [t for t in tools if t.name not in self.called and is_zero_argument(t)]
        for tool, _ in rank_tools(terms, zero):
            if self.remaining() <= 0:
                return
            await self.call(tool, {}, [])

    async def report(self) -> list[str]:
        """Observers at the ``scout`` trigger; apply effects; returns the newly disclosed tools."""
        if self.observers is None or not self.observers.fires_at("scout"):
            return []
        reports = await self.observers.report("scout", None, self.observations)
        if not reports:
            return []
        effects = self.observers.effects(reports)
        flags = [e.effect.flag for e in effects if e.effect.flag]
        failures = [e.failure for e in effects if e.effect.fail]
        notes = [f"{e.observer}.{e.condition}: {e.effect.note}" for e in effects if e.effect.note]
        self.result.events.append(
            InformantReportEvent(
                trigger="scout", reports=reports, flags=flags, failures=failures, notes=notes
            )
        )
        self.result.reports.extend(reports)
        self.result.flags.extend(f for f in flags if f not in self.result.flags)
        self.result.failures.extend(f for f in failures if f not in self.result.failures)
        self.result.notes.extend(notes)
        newly: list[str] = []
        for e in effects:
            if e.effect.enable_tools:
                added = [
                    t.name
                    for t in self.catalog.select(e.effect.enable_tools)
                    if t.name not in self.result.disclosed
                ]
                self.result.disclosed.extend(added)
                newly.extend(added)
                self.result.events.append(ToolsOfferedEvent(added=added, reason=e.reason))
            if e.effect.disable_tools:
                removed = [
                    t.name
                    for t in self.catalog.select(e.effect.disable_tools)
                    if t.name in self.result.disclosed
                ]
                self.result.disclosed = [n for n in self.result.disclosed if n not in removed]
                newly = [n for n in newly if n not in removed]
                self.result.events.append(ToolsOfferedEvent(removed=removed, reason=e.reason))
            if e.effect.enable_goal:
                self.result.goals.append(e.effect.enable_goal)
                self.result.events.append(
                    GoalEnabledEvent(
                        text=e.effect.enable_goal,
                        reason=e.reason,
                        observer=e.observer,
                        condition=e.condition,
                    )
                )
        return newly

    async def run(self) -> ScoutResult:
        await self.read_resources()
        await self.observe_tools(list(self.result.disclosed))
        newly = await self.report()
        if newly:
            before = len(self.observations)
            await self.observe_tools(newly)  # one pass over the new tools only
            if len(self.observations) > before:
                await self.report()
        self.result.on_request = [
            n for n in self.catalog.tool_names() if n not in self.result.disclosed
        ]
        return self.result


async def scout(
    scenario: Scenario,
    catalog: Catalog,
    session: Session,
    *,
    observers: ObserverRunner | None = None,
    budget: int | None = None,
) -> ScoutResult:
    """Observe the server read-only, let the observers report, return the :class:`ScoutResult`.

    ``catalog`` is the *allowed* catalog. ``observers`` is the scenario's runner (``None`` runs
    no observer at all; the dry run passes one with ``include_llm=False``). ``budget`` overrides
    :func:`scout_budget`.
    """
    scouting = _Scouting(
        scenario, catalog, session, observers, scout_budget(scenario) if budget is None else budget
    )
    return await scouting.run()
