"""Scenario v2 fields for the runner UI, read with fallbacks (contract A).

This is the one place the UI reads a scenario. It loads the file with
:func:`mcpsim.scenario.load_scenario` and reads the v2 display fields (``category``, ``title``,
``user_instructions``, ``context``, ``expected_behavior``, ``agent``) from the validated model
when the model has them, else from the raw mapping, else from the documented defaults.

The scenario model on this branch predates v2 and forbids unknown keys, so a v2 file fails
``load_scenario``; :func:`load_scenario_file` then strips the v2 keys and validates the rest.
Once the model carries the v2 fields the first attempt succeeds and the fallback goes unused,
so switching to the real model means deleting the fallback, nothing else.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path as FsPath
from typing import Any

import yaml

from mcpsim import scenario as scenario_module
from mcpsim.scenario import MODEL_ROLES, Scenario, ScenarioError, load_scenario, parse_scenario

V2_KEYS: tuple[str, ...] = (
    "category",
    "title",
    "user_instructions",
    "context",
    "expected_behavior",
    "agent",
)
DEFAULT_CATEGORY = "Uncategorized"
CONTEXT_KEYS: tuple[str, ...] = ("device", "location", "language")


@dataclass
class ScenarioContext:
    device: str = ""
    location: str = ""
    language: str = ""
    details: dict[str, str] = field(default_factory=dict)
    agent_visible: bool = False


@dataclass
class ScenarioView:
    """Everything the runner shows about one scenario; ``error`` is set for a file that failed."""

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
    # Filled in a run's scenario.json once the runner has read the SOP (v2 model).
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
    models: dict[str, str] = field(default_factory=dict)
    server: str = ""
    tools: dict[str, Any] = field(default_factory=dict)
    observers: list[str] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def search_text(self) -> str:
        return " ".join([self.name, self.title, self.category, self.user_instructions]).lower()


def derive_title(name: str) -> str:
    """``cheapest-penne`` -> ``Cheapest penne`` (the scenario module's default when it has one)."""
    core = getattr(scenario_module, "default_title", None)
    if callable(core):
        return str(core(name))
    words = re.sub(r"[-_.]+", " ", name).strip()
    return words[:1].upper() + words[1:] if words else name


def derive_user_instructions(role: str, goal: str) -> str:
    """The second-person default built from ``role`` and ``goal`` (contract A); the scenario
    module's own default when it has one, else the same text."""
    core = getattr(scenario_module, "default_user_instructions", None)
    if callable(core):
        return str(core(role, goal))
    return (
        f"You are this person: {role.strip()}\n\nWhat you want from the assistant: {goal.strip()}"
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool | int | float):
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)


def _plain(value: Any) -> Any:
    """A pydantic model (the future v2 model) as plain data; anything else unchanged."""
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else value


def _field(model: Scenario | None, raw: dict[str, Any], key: str) -> Any:
    """The model's attribute when it has one (v2 model), else the raw mapping's key."""
    if model is not None and hasattr(model, key):
        value = _plain(getattr(model, key))
        if value is not None:
            return value
    return raw.get(key)


def read_context(value: Any) -> ScenarioContext:
    if not isinstance(value, dict):
        return ScenarioContext()
    details_raw = value.get("details")
    details: dict[str, str] = {}
    if isinstance(details_raw, dict):
        details = {str(k): _text(v) for k, v in details_raw.items()}
    return ScenarioContext(
        device=_text(value.get("device")),
        location=_text(value.get("location")),
        language=_text(value.get("language")),
        details=details,
        agent_visible=value.get("agent_visible") is True,
    )


def read_raw(path: FsPath) -> dict[str, Any] | None:
    """The file's top-level mapping, or ``None`` when it cannot be read or parsed."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_scenario_file(path: FsPath) -> tuple[Scenario | None, dict[str, Any], str | None]:
    """``(model, raw mapping, error)``; the model is ``None`` exactly when ``error`` is set."""
    raw = read_raw(path) or {}
    try:
        return load_scenario(path), raw, None
    except ScenarioError as exc:
        first_error = str(exc)
    present = [k for k in V2_KEYS if k in raw]
    if not present:
        return None, raw, first_error
    # Pre-v2 model: validate everything except the v2 display fields.
    stripped = {k: v for k, v in raw.items() if k not in V2_KEYS}
    try:
        return parse_scenario(stripped, source=str(path)), raw, None
    except ScenarioError as exc:
        return None, raw, str(exc)


def build_view(
    raw: dict[str, Any],
    *,
    file: str,
    model: Scenario | None = None,
    error: str | None = None,
    fallback_name: str = "",
) -> ScenarioView:
    """A :class:`ScenarioView` from a validated model and/or a raw mapping."""
    name = model.name if model is not None else _text(raw.get("name")) or fallback_name
    role = model.role if model is not None else _text(raw.get("role"))
    goal = model.goal if model is not None else _text(raw.get("goal"))
    raw_instructions = model.instructions if model is not None else raw.get("instructions")
    instructions = (
        [_text(i) for i in raw_instructions if _text(i)]
        if isinstance(raw_instructions, list)
        else []
    )

    title = _text(_field(model, raw, "title")) or derive_title(name)
    category = _text(_field(model, raw, "category")) or DEFAULT_CATEGORY

    # "Derived" means the file does not say it: the v2 model fills these defaults itself, so
    # whether the raw mapping has the key is what tells a default from a written value.
    user_instructions = _text(_field(model, raw, "user_instructions"))
    derived_ui = not _text(raw.get("user_instructions"))
    if not user_instructions:
        user_instructions = derive_user_instructions(role, goal)

    behavior_raw = _field(model, raw, "expected_behavior")
    behavior = (
        [_text(i) for i in behavior_raw if _text(i)] if isinstance(behavior_raw, list) else []
    )
    raw_behavior = raw.get("expected_behavior")
    derived_eb = not (isinstance(raw_behavior, list) and any(_text(i) for i in raw_behavior))
    if not behavior:
        behavior = list(instructions)

    agent = _field(model, raw, "agent")
    agent = agent if isinstance(agent, dict) else {}

    outcome = raw.get("expected_outcome") if model is None else _plain(model.expected_outcome)
    outcome = outcome if isinstance(outcome, dict) else {}
    outcome_json = outcome.get("json")

    models_raw = raw.get("models")
    models = (
        {k: _text(v) for k, v in models_raw.items() if k in MODEL_ROLES and _text(v)}
        if isinstance(models_raw, dict)
        else {}
    )

    server = ""
    server_raw = raw.get("server")
    if isinstance(server_raw, dict):
        server = "stdio" if "stdio" in server_raw else "http" if "http" in server_raw else ""

    tools_raw = raw.get("tools")
    observers_raw = raw.get("observers")
    observers: list[str] = []
    if isinstance(observers_raw, list):
        for entry in observers_raw:
            if isinstance(entry, dict):
                observers.append(_text(entry.get("name") or entry.get("use")))

    def _int(key: str) -> int | None:
        value = getattr(model, key, None) if model is not None else raw.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return ScenarioView(
        name=name,
        file=file,
        title=title,
        category=category,
        user_instructions=user_instructions,
        user_instructions_derived=derived_ui,
        context=read_context(_field(model, raw, "context")),
        expected_behavior=behavior,
        expected_behavior_derived=derived_eb,
        agent_skill=_text(agent.get("skill")) or None,
        agent_notes=_text(agent.get("notes")) or None,
        agent_skill_name=_text(agent.get("skill_name")) or None,
        agent_skill_path=_text(agent.get("skill_path")) or None,
        agent_skill_text=_text(agent.get("skill_text")) or None,
        role=role,
        goal=goal,
        instructions=instructions,
        expected_outcome_text=_text(outcome.get("text")) or None,
        expected_outcome_json=outcome_json if isinstance(outcome_json, dict) else None,
        repeat=_int("repeat"),
        judge_votes=_int("judge_votes"),
        models=models,
        server=server,
        tools=tools_raw if isinstance(tools_raw, dict) else {},
        observers=[o for o in observers if o],
        error=error,
    )


def scenario_view(path: FsPath, *, display_path: str | None = None) -> ScenarioView:
    """Load one scenario file into a view; a broken file yields a view with ``error`` set."""
    model, raw, error = load_scenario_file(path)
    return build_view(
        raw,
        file=display_path or str(path),
        model=model,
        error=error,
        fallback_name=path.stem,
    )
