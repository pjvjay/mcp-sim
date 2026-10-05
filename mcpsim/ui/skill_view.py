"""The simulate skill as the runner UI shows it, read through :mod:`mcpsim.skill` (contract C).

The UI never reads ``config.yaml`` or the role files itself. :class:`SkillSource` loads the
skill with :func:`mcpsim.skill.load_skill` (cached until one of its files changes, so an edit
shows on the next request), and everything shown is what the CLI resolves:
:meth:`~mcpsim.skill.Skill.describe` for the roles, their settings and the run defaults, and
:meth:`~mcpsim.skill.Skill.resolve` for one scenario's models and run settings with the layer
each came from. Model precedence, lowest to highest: built-in default < role frontmatter <
``config.yaml`` ``defaults`` < matching overrides < the scenario file < the run's own models
(the settings form, which the job passes as ``--models``).
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path as FsPath
from typing import Any

from mcpsim.llm import OLLAMA, parse_model_spec
from mcpsim.scenario import Scenario
from mcpsim.skill import ROLE_FILE_FOR, Skill, SkillError, load_skill

LOCAL_PLANNER = "planner-local"


class SkillSource:
    """The skill directory the UI serves, and the last version of it that loaded."""

    def __init__(self, skill: Skill) -> None:
        self.directory: FsPath = skill.path
        self._last = skill
        self._lock = threading.Lock()

    def load(self) -> tuple[Skill, str | None]:
        """``(skill, error)``: the skill as it is on disk now, or, when an edit broke it, the
        last version that loaded and the loader's error (runs started now would fail on it)."""
        try:
            skill = load_skill(self.directory)
        except SkillError as exc:
            with self._lock:
                return self._last, str(exc)
        with self._lock:
            self._last = skill
        return skill, None


def _row(
    role: str,
    spec: str,
    source: str,
    skill: Skill,
    file_role: str,
) -> dict[str, Any]:
    provider, model = parse_model_spec(spec)
    role_file = skill.roles[file_role]
    return {
        "role": role,
        "provider": provider,
        "model": model,
        "spec": f"{provider}:{model}",
        "source": source,
        "file": str(role_file.path),
        "temperature": role_file.temperature,
        "max_tokens": role_file.max_tokens,
        "extra": dict(role_file.extra),
    }


def role_rows(skill: Skill, described: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per role file, from :meth:`Skill.describe`: the model it resolves to (before any
    scenario) and where that came from. ``planner-local`` shows its own frontmatter: it serves
    the planner only when the planner resolves to an ``ollama:`` model."""
    rows: list[dict[str, Any]] = []
    for name, entry in described["roles"].items():
        model = entry.get("model")
        spec = model if isinstance(model, str) else str(entry["frontmatter_model"])
        source = str(entry.get("model_source") or entry["summary"])
        row = _row(name, spec, source, skill, name)
        row.update(
            summary=entry["summary"],
            description=entry["description"],
            prompts=list(entry["prompts"]),
        )
        rows.append(row)
    return rows


def scenario_settings(
    skill: Skill, scenario: Scenario, run_models: Mapping[str, str] | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """``(models, run)`` for one freshly loaded scenario, as ``mcpsim run --skill`` resolves
    them: ``models`` is ``{role: row}`` (the planner's file is ``planner-local.md`` when it
    resolves to Ollama), ``run`` is ``{key: {value, source}}``."""
    resolved = skill.resolve(scenario, model_overrides=run_models)
    local = parse_model_spec(resolved.models["planner"].value)[0] == OLLAMA
    models: dict[str, dict[str, Any]] = {}
    for role, setting in resolved.models.items():
        file_role = LOCAL_PLANNER if role == "planner" and local else ROLE_FILE_FOR[role]
        models[role] = _row(role, setting.value, setting.source, skill, file_role)
    run = {k: {"value": s.value, "source": s.source} for k, s in resolved.run.items()}
    return models, run


def overrides_json(skill: Skill) -> list[dict[str, Any]]:
    """``config.yaml``'s overrides with their unset keys left out."""
    out: list[dict[str, Any]] = []
    for override in skill.config.overrides:
        data = override.model_dump(mode="json")
        out.append(
            {
                "match": {k: v for k, v in data["match"].items() if v is not None},
                "models": dict(data["models"]),
                "run": {k: v for k, v in data["run"].items() if v is not None},
            }
        )
    return out
