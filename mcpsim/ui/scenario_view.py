"""What the runner UI shows about a scenario, read through the scenario model (contract A).

A scenario file is loaded with :func:`mcpsim.scenario.load_scenario`, exactly as ``mcpsim run``
loads it, so the UI calls a file valid only when the runner would run it (an ``agent.skill``
whose ``env:`` variable is unset is an error here too). A run's ``scenario.json`` is validated
the way the runner reads it back for a re-judge. The view's v2 fields come from the validated
:class:`~mcpsim.scenario.Scenario`, defaults included (title, user instructions, expected
behaviour). Only a file or snapshot that does not validate is read as plain YAML, and then just
for display: its name, title, category and persona next to the error.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path as FsPath
from typing import Any

import yaml
from pydantic import ValidationError

from mcpsim.scenario import (
    DEFAULT_CATEGORY,
    Context,
    Scenario,
    ScenarioError,
    default_title,
    default_user_instructions,
    load_scenario,
)


@dataclass
class ScenarioContext:
    """``context`` as text: an absent field is ``""`` and every ``details`` value a string (a
    non-string value as compact JSON, as :meth:`mcpsim.scenario.Context.items` renders it)."""

    device: str = ""
    location: str = ""
    language: str = ""
    details: dict[str, str] = field(default_factory=dict)
    agent_visible: bool = False

    @classmethod
    def of(cls, context: Context) -> ScenarioContext:
        pairs = context.items()
        # items() lists the set fields among device / location / language first, then details.
        fixed = sum(1 for v in (context.device, context.location, context.language) if v)
        return cls(
            device=(context.device or "").strip(),
            location=(context.location or "").strip(),
            language=(context.language or "").strip(),
            details=dict(pairs[fixed:]),
            agent_visible=context.agent_visible,
        )


@dataclass
class ScenarioView:
    """Everything the runner shows about one scenario. ``error`` is set for a file that does
    not validate; the other fields are then a best-effort reading of the raw file."""

    name: str
    file: str
    title: str = ""
    category: str = DEFAULT_CATEGORY
    user_instructions: str = ""
    user_instructions_derived: bool = False
    context: ScenarioContext = field(default_factory=ScenarioContext)
    expected_behavior: list[str] = field(default_factory=list)
    expected_behavior_derived: bool = False
    agent_skill: str | None = None
    agent_notes: str | None = None
    # The SOP as resolved when the scenario was validated (skill_text is the procedure itself).
    agent_skill_name: str | None = None
    agent_skill_path: str | None = None
    agent_skill_text: str | None = None
    role: str = ""
    goal: str = ""
    instructions: list[str] = field(default_factory=list)
    expected_outcome_text: str | None = None
    expected_outcome_json: dict[str, Any] | None = None
    repeat: int | None = None
    judge_votes: int | None = None
    # The models the scenario file names itself (the "scenario file" layer of the precedence);
    # for a run's scenario.json, every model the run used.
    models: dict[str, str] = field(default_factory=dict)
    server: str = ""
    tools: dict[str, Any] = field(default_factory=dict)
    observers: list[str] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def search_text(self) -> str:
        return " ".join([self.name, self.title, self.category, self.user_instructions]).lower()


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def read_raw(path: FsPath) -> dict[str, Any] | None:
    """The file's top-level mapping, or ``None`` when it cannot be read or parsed."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def view_from_scenario(
    scenario: Scenario, *, file: str, raw: Mapping[str, Any] | None = None
) -> ScenarioView:
    """The view of a validated scenario.

    ``raw`` is the mapping the file holds: a v2 field the file leaves out is "derived" (the
    model filled in its default), and ``models`` is what the file names. Without ``raw`` (a
    run's ``scenario.json``, where the runner wrote every field) a value equal to its default
    counts as derived, and ``models`` is every model the run used.
    """
    behaviors = scenario.behaviors
    instructions = [item.strip() for item in scenario.instructions]
    user_instructions = scenario.simulated_user_instructions
    if raw is not None:
        derived_ui = "user_instructions" not in raw
        derived_eb = "expected_behavior" not in raw
        models = scenario.models.explicit()
    else:
        derived_ui = user_instructions == default_user_instructions(scenario.role, scenario.goal)
        derived_eb = behaviors == instructions
        dumped = scenario.models.model_dump(mode="json")
        models = {k: v for k, v in dumped.items() if isinstance(v, str) and v}
    outcome = scenario.expected_outcome
    agent = scenario.agent
    return ScenarioView(
        name=scenario.name,
        file=file,
        title=scenario.display_title,
        category=scenario.category,
        user_instructions=user_instructions,
        user_instructions_derived=derived_ui,
        context=ScenarioContext.of(scenario.context),
        expected_behavior=behaviors,
        expected_behavior_derived=derived_eb,
        agent_skill=agent.skill,
        agent_notes=agent.notes,
        agent_skill_name=agent.skill_name,
        agent_skill_path=agent.skill_path,
        agent_skill_text=agent.skill_text,
        role=scenario.role,
        goal=scenario.goal,
        instructions=instructions,
        expected_outcome_text=outcome.text,
        expected_outcome_json=dict(outcome.json) if outcome.json is not None else None,
        repeat=scenario.repeat,
        judge_votes=scenario.judge_votes,
        models=models,
        server=scenario.server.kind,
        tools=scenario.tools.model_dump(mode="json", exclude_defaults=True),
        observers=[o.name for o in scenario.observers],
    )


def view_from_invalid(
    raw: Mapping[str, Any], *, file: str, error: str, fallback_name: str
) -> ScenarioView:
    """A file (or snapshot) that does not validate: its error, plus what the raw mapping says
    about its name, title, category and persona, so the list can still place it."""
    name = _text(raw.get("name")) or fallback_name
    role, goal = _text(raw.get("role")), _text(raw.get("goal"))
    instructions = raw.get("instructions")
    return ScenarioView(
        name=name,
        file=file,
        title=_text(raw.get("title")) or default_title(name),
        category=_text(raw.get("category")) or DEFAULT_CATEGORY,
        user_instructions=_text(raw.get("user_instructions"))
        or (default_user_instructions(role, goal) if role and goal else ""),
        role=role,
        goal=goal,
        instructions=(
            [_text(i) for i in instructions if _text(i)] if isinstance(instructions, list) else []
        ),
        error=error,
    )


def load_view(
    path: FsPath, *, display_path: str | None = None
) -> tuple[ScenarioView, Scenario | None]:
    """``(view, scenario)`` for one scenario file, loaded with
    :func:`~mcpsim.scenario.load_scenario`; ``scenario`` is ``None`` when it does not load, and
    the view then carries the loader's error."""
    file = display_path or str(path)
    raw = read_raw(path)
    try:
        scenario = load_scenario(path)
    except (ScenarioError, ValueError, OSError) as exc:
        message = str(exc).strip() or type(exc).__name__
        view = view_from_invalid(raw or {}, file=file, error=message, fallback_name=path.stem)
        return view, None
    return view_from_scenario(scenario, file=file, raw=raw if raw is not None else {}), scenario


def snapshot_view(data: Mapping[str, Any], *, fallback_name: str) -> ScenarioView:
    """A run's ``scenario.json`` (the scenario as it ran: models and SOP resolved), validated
    the way the runner reads it back for ``mcpsim judge``."""
    try:
        scenario = Scenario.model_validate(dict(data))
    except (ValidationError, ValueError) as exc:
        return view_from_invalid(
            data, file="scenario.json", error=str(exc), fallback_name=fallback_name
        )
    return view_from_scenario(scenario, file="scenario.json")
