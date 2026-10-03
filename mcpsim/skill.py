"""The simulate skill: every LLM role's model, settings and prompts, loaded at run time.

A skill directory (``skills/simulate/`` in the repository, packaged with mcpsim as well) holds:

* ``SKILL.md`` — how to run the flow end to end (frontmatter ``name`` and ``description``);
* ``config.yaml`` — per-role model defaults, run defaults, the scenario sources, ``runs_dir``
  and per-scenario overrides (:class:`SkillConfig`);
* ``roles/<role>.md`` for every role in :data:`ROLE_SPECS` — YAML frontmatter (``role``,
  ``provider``, ``model``, optional ``temperature`` / ``max_tokens``, role-specific keys such as
  the judge's ``votes``) and a body of prompt templates (:mod:`mcpsim.prompt_template`), one per
  ``{% prompt NAME %}`` marker.

:func:`load_skill` reads and validates all of it: unknown frontmatter keys, unknown prompts,
unknown placeholders and missing required ones are :class:`SkillError` naming the file and line.
The directory comes from ``--skill DIR``, else ``MCPSIM_SKILL``, else the packaged copy
(:func:`skill_dir`).

Model precedence for a role, lowest to highest (:meth:`Skill.resolve_models`): the built-in
default < the role file's frontmatter < ``config.yaml`` ``defaults`` < every matching
``config.yaml`` override, in file order < the scenario file's own ``models`` < the CLI's
``--models``. Run settings (``repeat``, ``modes``, ``judge_votes``, ``concurrency``) follow the
same ladder (:meth:`Skill.resolve_run`): built-in < the judge file's ``votes`` (for
``judge_votes``) < ``config.yaml`` ``run`` < matching overrides' ``run`` < the scenario file <
the CLI. :meth:`Skill.apply` returns the scenario with everything resolved, which is what the
runner writes to ``scenario.json``.
"""

from __future__ import annotations

import glob as globmod
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path as FsPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mcpsim.llm import ANTHROPIC, OLLAMA, PROVIDERS, parse_model_spec, rejects_sampling
from mcpsim.prompt_template import Template, TemplateError, Value, parse, split_parts
from mcpsim.scenario import DEFAULT_MODELS, MODEL_ROLES, Models, Scenario, ScenarioError

_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)

SKILL_ENV = "MCPSIM_SKILL"
SKILL_FILE = "SKILL.md"
CONFIG_FILE = "config.yaml"
DEFAULT_ROLES_DIR = "roles"
DEFAULT_RUNS_DIR = "runs"
SKILL_NAME = "simulate"
# Inside an installed wheel the skill lives at mcpsim/_skills/simulate (pyproject maps the
# repository's skills/ directory there); in a source checkout at <repo>/skills/simulate.
PACKAGED_DIR = FsPath(__file__).resolve().parent / "_skills" / SKILL_NAME
CHECKOUT_DIR = FsPath(__file__).resolve().parent.parent / "skills" / SKILL_NAME
SCENARIO_SUFFIXES = (".yaml", ".yml", ".json")

Mode = Literal["guided", "free"]
MODES: tuple[Mode, ...] = ("guided", "free")
RUN_KEYS: tuple[str, ...] = ("repeat", "modes", "judge_votes", "concurrency")
# What a scenario file sets itself (``Scenario`` fields) among the run settings.
SCENARIO_RUN_FIELDS: tuple[str, ...] = ("repeat", "judge_votes", "concurrency")
BUILTIN_RUN: dict[str, Any] = {
    "repeat": Scenario.model_fields["repeat"].default,
    "modes": list(MODES),
    "judge_votes": Scenario.model_fields["judge_votes"].default,
    "concurrency": Scenario.model_fields["concurrency"].default,
}

BUILTIN = "built-in default"
SCENARIO_LAYER = "scenario file"
CLI_LAYER = "command line"


class SkillError(ValueError):
    """The skill directory, its config.yaml or one of its role files is invalid."""


# --- role specifications ----------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSpec:
    """What one prompt of a role may read (``accepts``) and must insert (``requires``)."""

    accepts: tuple[str, ...]
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoleSpec:
    """A role file's prompts, the scenario model role it configures (``None`` for a role whose
    model is decided elsewhere) and its role-specific frontmatter keys (positive integers)."""

    prompts: dict[str, PromptSpec]
    model_role: str | None
    extra_keys: tuple[str, ...] = ()
    summary: str = ""


ROLE_SPECS: dict[str, RoleSpec] = {
    "planner": RoleSpec(
        prompts={
            "system": PromptSpec(
                accepts=(
                    "path_kinds", "reference_example", "reference_key", "discover_tool",
                    "example_step", "plan_tool", "catalog",
                ),
                requires=("catalog",),
            ),
            "user": PromptSpec(
                accepts=(
                    "name", "role", "goal", "instructions", "outcome_text", "outcome_json",
                    "has_scout", "reports", "goals", "observations", "observations_left_out",
                ),
                requires=("goal",),
            ),
            "reask": PromptSpec(accepts=("problems", "plan_tool"), requires=("problems",)),
        },
        model_role="planner",
        summary="hosted planner (any planner model that is not ollama:)",
    ),
    "planner-local": RoleSpec(
        prompts={
            "system": PromptSpec(accepts=("tools", "answer_fields"), requires=("tools",)),
            "user": PromptSpec(
                accepts=("request", "rules", "reports", "observations"), requires=("request",)
            ),
            "reask": PromptSpec(accepts=("problems",), requires=("problems",)),
            "policy_system": PromptSpec(accepts=("tools",), requires=("tools",)),
            "policy_user": PromptSpec(accepts=("rule", "none_answer"), requires=("rule",)),
        },
        model_role=None,
        extra_keys=("policy_max_tokens",),
        summary="local execution planner (used only when the planner is an ollama: model)",
    ),
    "agent": RoleSpec(
        prompts={
            "system": PromptSpec(
                accepts=(
                    "role", "goal", "instructions", "skill_name", "skill_text", "notes",
                    "context", "goals", "guided", "steps", "answer_fields", "final_result_name",
                ),
                requires=(
                    "goal", "instructions", "skill_text", "goals", "steps", "final_result_name",
                ),
            ),
            "goal_note": PromptSpec(accepts=("goals",), requires=("goals",)),
        },
        model_role="agent",
        summary="agent under test",
    ),
    "user": RoleSpec(
        prompts={
            "system": PromptSpec(
                accepts=("user_instructions", "context", "language", "final_result_name"),
                requires=("user_instructions",),
            ),
            "opening": PromptSpec(accepts=()),
            "silent_agent": PromptSpec(accepts=()),
            "fallback_reply": PromptSpec(accepts=()),
        },
        model_role="user",
        summary="simulated user",
    ),
    "observer": RoleSpec(
        prompts={
            "system": PromptSpec(
                accepts=("identity", "conditions", "report_tool"),
                requires=("identity", "conditions"),
            ),
            "user": PromptSpec(accepts=("watched",), requires=("watched",)),
        },
        model_role="observer",
        summary="LLM observers (an observer's own model field wins)",
    ),
    "judge": RoleSpec(
        prompts={
            "system": PromptSpec(accepts=("has_sop", "verdict_tool")),
            "user": PromptSpec(
                accepts=(
                    "name", "title", "category", "role", "goal", "user_instructions", "context",
                    "agent_saw_context", "instructions", "behaviors", "outcome_text",
                    "outcome_json", "skill_name", "skill_text", "notes", "path_id", "path_kind",
                    "path_title", "rationale", "steps", "checkpoints", "matches", "reports",
                    "observer_failures", "flags", "run_path", "run_mode", "run_index", "outcome",
                    "outcome_reason", "final_result", "transcript", "has_sop", "verdict_tool",
                ),
                requires=("behaviors", "transcript"),
            ),
        },
        model_role="judge",
        extra_keys=("votes",),
        summary="independent judge",
    ),
}
ROLE_NAMES: tuple[str, ...] = tuple(ROLE_SPECS)
COMMON_KEYS: tuple[str, ...] = (
    "role", "description", "provider", "model", "temperature", "max_tokens",
)
REQUIRED_KEYS: tuple[str, ...] = ("role", "provider", "model")
# The role file that configures each scenario model role.
ROLE_FILE_FOR: dict[str, str] = {
    spec.model_role: name for name, spec in ROLE_SPECS.items() if spec.model_role is not None
}


def canonical_spec(spec: str) -> str:
    """``anthropic:claude-x`` -> ``claude-x`` (a bare name is an Anthropic model), ``ollama:m``
    unchanged: one spelling per model, so comparisons and cost keys stay stable."""
    provider, name = parse_model_spec(spec)
    return name if provider == ANTHROPIC else f"{provider}:{name}"


# --- the role files ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    """One loaded role file: settings from its frontmatter, prompts from its body."""

    name: str
    path: FsPath
    provider: str
    model: str
    temperature: float | None
    max_tokens: int | None
    description: str
    extra: dict[str, int]
    prompts: dict[str, Template]

    @property
    def spec(self) -> str:
        """``provider:model`` in canonical form (:func:`canonical_spec`)."""
        return canonical_spec(f"{self.provider}:{self.model}")

    def render(self, prompt: str, values: Mapping[str, Value] | None = None, **more: Value) -> str:
        """The named prompt rendered with ``values`` (and keyword values)."""
        template = self.prompts.get(prompt)
        if template is None:
            raise KeyError(f"{self.path}: role {self.name!r} has no prompt {prompt!r}")
        merged: dict[str, Value] = {**(values or {}), **more}
        return template.render(merged)

    def tokens(self, default: int) -> int:
        """``max_tokens`` from the frontmatter, else ``default``."""
        return self.max_tokens if self.max_tokens is not None else default

    def setting(self, key: str, default: int) -> int:
        """A role-specific integer key (``votes``, ``policy_max_tokens``), else ``default``."""
        return self.extra.get(key, default)


def _split_frontmatter(text: str, origin: str) -> tuple[dict[str, Any], str, int]:
    """``(frontmatter, body, line number of the body's first line)``."""
    lines = text.lstrip("﻿").split("\n")
    if not lines or lines[0].strip() != "---":
        raise SkillError(f"{origin}: a role file starts with a '---' YAML frontmatter block")
    end = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end is None:
        raise SkillError(f"{origin}: the frontmatter opened on line 1 is never closed")
    try:
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        raise SkillError(f"{origin}: the frontmatter is not valid YAML: {exc}") from None
    if not isinstance(meta, dict):
        raise SkillError(f"{origin}: the frontmatter must be a mapping")
    return meta, "\n".join(lines[end + 1 :]), end + 2


def _positive_int(meta: dict[str, Any], key: str, origin: str) -> int | None:
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SkillError(f"{origin}: {key} must be a positive integer, got {value!r}")
    return value


def load_role(path: str | os.PathLike[str], name: str | None = None) -> Role:
    """Load and validate one role file; ``name`` defaults to the file's stem."""
    fs_path = FsPath(path)
    role_name = name or fs_path.stem
    origin = str(fs_path)
    spec = ROLE_SPECS.get(role_name)
    if spec is None:
        raise SkillError(f"{origin}: unknown role {role_name!r}; roles: {', '.join(ROLE_NAMES)}")
    try:
        text = fs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SkillError(f"{origin}: cannot read role file: {exc}") from None
    meta, body, first_line = _split_frontmatter(text, origin)
    allowed = (*COMMON_KEYS, *spec.extra_keys)
    unknown = sorted(str(k) for k in meta if k not in allowed)
    if unknown:
        raise SkillError(
            f"{origin}: unknown frontmatter key(s) {', '.join(unknown)}; "
            f"this role accepts: {', '.join(allowed)}"
        )
    missing = [k for k in REQUIRED_KEYS if meta.get(k) in (None, "")]
    if missing:
        raise SkillError(f"{origin}: frontmatter needs {', '.join(missing)}")
    if meta["role"] != role_name:
        raise SkillError(
            f"{origin}: frontmatter says role {meta['role']!r}, the file is {role_name!r}"
        )
    provider = str(meta["provider"]).strip().lower()
    if provider not in PROVIDERS:
        raise SkillError(
            f"{origin}: provider must be one of {', '.join(PROVIDERS)}, got {meta['provider']!r}"
        )
    model = str(meta["model"]).strip()
    if not model:
        raise SkillError(f"{origin}: model must not be blank")
    temperature = meta.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            raise SkillError(f"{origin}: temperature must be a number, got {temperature!r}")
        if not 0 <= float(temperature) <= 2:
            raise SkillError(f"{origin}: temperature must be between 0 and 2, got {temperature}")
        # Whether the model the role ends up on accepts it is checked once models are resolved
        # (Skill.check_sampling): config.yaml or a scenario may move the role to another model.
    max_tokens = _positive_int(meta, "max_tokens", origin)
    extra = {k: v for k in spec.extra_keys if (v := _positive_int(meta, k, origin)) is not None}
    description = meta.get("description") or ""
    if not isinstance(description, str):
        raise SkillError(f"{origin}: description must be text")
    try:
        sources = split_parts(body, origin=origin, first_line=first_line)
    except TemplateError as exc:
        raise SkillError(str(exc)) from None
    unknown_prompts = sorted(set(sources) - set(spec.prompts))
    if unknown_prompts:
        raise SkillError(
            f"{origin}: unknown prompt(s) {', '.join(unknown_prompts)}; role {role_name!r} has: "
            f"{', '.join(spec.prompts)}"
        )
    missing_prompts = [p for p in spec.prompts if p not in sources]
    if missing_prompts:
        raise SkillError(
            f"{origin}: missing prompt(s) {', '.join(missing_prompts)} "
            "(each starts with a line '{% prompt NAME %}')"
        )
    prompts: dict[str, Template] = {}
    for prompt_name, (source, line) in sources.items():
        where = f"{origin} (prompt {prompt_name})"
        if not _COMMENT.sub("", source).strip():
            raise SkillError(f"{where}: the prompt is empty")
        try:
            template = parse(source, origin=origin, first_line=line)
        except TemplateError as exc:
            raise SkillError(f"{exc} (prompt {prompt_name})") from None
        prompt_spec = spec.prompts[prompt_name]
        for unknown_name in sorted(template.names - set(prompt_spec.accepts)):
            accepts = ", ".join(prompt_spec.accepts) or "(no placeholders)"
            raise SkillError(
                f"{origin}, line {template.first_line(unknown_name)}: unknown placeholder "
                f"'{unknown_name}' in prompt {prompt_name}; it accepts: {accepts}"
            )
        for required in prompt_spec.requires:
            if required not in template.inserted:
                raise SkillError(
                    f"{where}: the prompt must insert {{{{ {required} }}}} "
                    f"(required placeholders: {', '.join(prompt_spec.requires)})"
                )
        prompts[prompt_name] = template
    return Role(
        name=role_name,
        path=fs_path,
        provider=provider,
        model=model,
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=max_tokens,
        description=" ".join(description.split()),
        extra=extra,
        prompts=prompts,
    )


# --- config.yaml ------------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(_Strict):
    """Run settings; ``None`` leaves the setting to the layer below."""

    repeat: int | None = Field(default=None, ge=1)
    modes: list[Mode] | None = None
    judge_votes: int | None = Field(default=None, ge=1)
    concurrency: int | None = Field(default=None, ge=1)

    @field_validator("modes")
    @classmethod
    def _modes(cls, value: list[Mode] | None) -> list[Mode] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("modes must list at least one of guided, free")
        if len(set(value)) != len(value):
            raise ValueError("modes lists a mode twice")
        return value

    def given(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in RUN_KEYS if getattr(self, k) is not None}


def _check_models(value: dict[str, str]) -> dict[str, str]:
    unknown = sorted(set(value) - set(MODEL_ROLES))
    if unknown:
        raise ValueError(
            f"unknown role(s) {', '.join(unknown)}; model roles: {', '.join(MODEL_ROLES)}"
        )
    out: dict[str, str] = {}
    for role, spec in value.items():
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError(f"{role}: a model is 'provider:model', got {spec!r}")
        out[role] = canonical_spec(spec)
    return out


class OverrideMatch(_Strict):
    """Which scenarios an override applies to: ``fnmatch`` globs, both must match when both
    are given."""

    name: str | None = None
    category: str | None = None

    @model_validator(mode="after")
    def _one(self) -> OverrideMatch:
        if self.name is None and self.category is None:
            raise ValueError("match needs name and/or category (fnmatch globs)")
        return self

    def matches(self, name: str, category: str) -> bool:
        if self.name is not None and not fnmatchcase(name, self.name):
            return False
        return self.category is None or fnmatchcase(category, self.category)

    def label(self) -> str:
        parts = [f"{k}={v}" for k, v in (("name", self.name), ("category", self.category)) if v]
        return ", ".join(parts)


class Override(_Strict):
    """``{match: {name|category: glob}, models: {role: spec}, run: {...}}``."""

    match: OverrideMatch
    models: dict[str, str] = Field(default_factory=dict)
    run: RunConfig = Field(default_factory=RunConfig)

    @field_validator("models")
    @classmethod
    def _models(cls, value: dict[str, str]) -> dict[str, str]:
        return _check_models(value)


class SkillConfig(_Strict):
    """``config.yaml``. Relative paths (``scenarios`` entries, ``runs_dir``) resolve against
    the directory mcpsim runs in; a ``scenarios`` entry may reference environment variables as
    ``$NAME``, ``${NAME}`` or ``${NAME:-default}``."""

    roles_dir: str = DEFAULT_ROLES_DIR
    defaults: dict[str, str] = Field(default_factory=dict)
    run: RunConfig = Field(default_factory=RunConfig)
    scenarios: list[str] = Field(default_factory=list)
    runs_dir: str = DEFAULT_RUNS_DIR
    overrides: list[Override] = Field(default_factory=list)

    @field_validator("defaults")
    @classmethod
    def _defaults(cls, value: dict[str, str]) -> dict[str, str]:
        return _check_models(value)

    @field_validator("scenarios")
    @classmethod
    def _entries(cls, value: list[str]) -> list[str]:
        for i, entry in enumerate(value):
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"scenarios[{i}] is blank")
        return value


def load_config(path: str | os.PathLike[str]) -> SkillConfig:
    fs_path = FsPath(path)
    try:
        raw = yaml.safe_load(fs_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise SkillError(f"{fs_path}: cannot read: {exc}") from None
    except yaml.YAMLError as exc:
        raise SkillError(f"{fs_path}: not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise SkillError(f"{fs_path}: the top level must be a mapping")
    try:
        return SkillConfig.model_validate(raw)
    except ValidationError as exc:
        lines = [f"{fs_path}: invalid config"]
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"  {loc}: {err['msg']}")
        raise SkillError("\n".join(lines)) from None


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def expand_env(text: str, env: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    """``(expanded, None)``, or ``(None, reason)`` when a referenced variable is unset and has
    no ``:-default``."""
    source = os.environ if env is None else env
    missing: list[str] = []

    def replace(m: re.Match[str]) -> str:
        name = m.group(1) or m.group(3)
        value = source.get(name, "")
        if value:
            return value
        if m.group(2) is not None:
            return m.group(2)
        missing.append(name)
        return ""

    expanded = _ENV_REF.sub(replace, text)
    if missing:
        return None, f"{', '.join(missing)} is not set"
    return expanded, None


# --- scenario sources -------------------------------------------------------------------------


@dataclass
class ScenarioEntry:
    """One scenario file the skill knows about; ``error`` when it does not load."""

    file: FsPath
    source: str
    name: str
    title: str = ""
    category: str = ""
    error: str | None = None
    scenario: Scenario | None = field(default=None, repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "category": self.category,
            "file": str(self.file),
            "source": self.source,
            "error": self.error,
        }


@dataclass
class ScenarioSource:
    """One ``scenarios`` entry: what it expanded to, its files, or why it was skipped."""

    entry: str
    path: str | None
    files: list[FsPath] = field(default_factory=list)
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "path": self.path,
            "files": [str(f) for f in self.files],
            "note": self.note,
        }


def scenario_files_in(path: FsPath) -> list[FsPath]:
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SCENARIO_SUFFIXES
    )


def expand_source(
    entry: str, *, base: FsPath | None = None, env: Mapping[str, str] | None = None
) -> ScenarioSource:
    """A directory (its ``*.yaml`` / ``*.yml`` / ``*.json`` files), a glob, or one file;
    relative to ``base`` (default: the current directory)."""
    expanded, reason = expand_env(entry.strip(), env)
    if expanded is None:
        return ScenarioSource(entry=entry, path=None, note=f"skipped: {reason}")
    path = FsPath(expanded).expanduser()
    if not path.is_absolute():
        path = (base or FsPath.cwd()) / path
    if any(ch in expanded for ch in "*?["):
        files = sorted(
            FsPath(p) for p in globmod.glob(str(path), recursive=True)
            if FsPath(p).is_file() and FsPath(p).suffix.lower() in SCENARIO_SUFFIXES
        )
        note = None if files else "no scenario file matches"
        return ScenarioSource(entry=entry, path=str(path), files=files, note=note)
    if path.is_dir():
        files = scenario_files_in(path)
        return ScenarioSource(
            entry=entry, path=str(path), files=files,
            note=None if files else "no scenario files (*.yaml, *.yml, *.json)",
        )
    if path.is_file():
        return ScenarioSource(entry=entry, path=str(path), files=[path])
    return ScenarioSource(entry=entry, path=str(path), note="not found")


def one_line(message: str) -> str:
    """A multi-line error (a scenario's validation report) on one line: ``first; second``."""
    return "; ".join(line.strip() for line in message.strip().splitlines() if line.strip())


def load_entries(files: Sequence[tuple[FsPath, str]]) -> list[ScenarioEntry]:
    """Load every ``(file, source)``; a file that does not load is an entry with ``error``.
    A scenario name defined twice is an error on the second file."""
    from mcpsim.scenario import load_scenario

    entries: list[ScenarioEntry] = []
    seen: dict[str, FsPath] = {}
    for file, source in files:
        try:
            scenario = load_scenario(file)
        except (ScenarioError, ValueError, OSError) as exc:
            first = str(exc).strip().splitlines()
            entries.append(
                ScenarioEntry(
                    file=file, source=source, name=file.stem,
                    error="\n".join(first) or type(exc).__name__,
                )
            )
            continue
        entry = ScenarioEntry(
            file=file,
            source=source,
            name=scenario.name,
            title=scenario.display_title,
            category=scenario.category,
            scenario=scenario,
        )
        if scenario.name in seen:
            entry.error = (
                f"scenario name {scenario.name!r} is also defined by {seen[scenario.name]}"
            )
            entry.scenario = None
        else:
            seen[scenario.name] = file
        entries.append(entry)
    return entries


def select_entries(
    entries: Sequence[ScenarioEntry],
    *,
    names: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
) -> list[ScenarioEntry]:
    """The entries whose name matches any ``names`` glob and whose category matches any
    ``categories`` glob (no globs: everything). An entry that failed to load has no category,
    so a category filter leaves it out."""
    selected: list[ScenarioEntry] = []
    for entry in entries:
        if names and not any(fnmatchcase(entry.name, g) for g in names):
            continue
        if categories and (
            entry.error is not None or not any(fnmatchcase(entry.category, g) for g in categories)
        ):
            continue
        selected.append(entry)
    return selected


# --- the skill --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """A resolved value and the layer it came from."""

    value: Any
    source: str


@dataclass
class Resolved:
    """Everything resolved for one scenario: models and run settings with their sources."""

    models: dict[str, Setting]
    run: dict[str, Setting]

    @property
    def modes(self) -> list[Mode]:
        return list(self.run["modes"].value)

    def to_json(self) -> dict[str, Any]:
        return {
            "models": {r: {"model": s.value, "source": s.source} for r, s in self.models.items()},
            "run": {k: {"value": s.value, "source": s.source} for k, s in self.run.items()},
        }


@dataclass
class Skill:
    """A loaded skill directory (see the module docstring)."""

    path: FsPath
    name: str
    description: str
    body: str
    config: SkillConfig
    roles: dict[str, Role]

    def role(self, name: str) -> Role:
        try:
            return self.roles[name]
        except KeyError:
            raise KeyError(f"skill {self.path} has no role {name!r}") from None

    @property
    def config_path(self) -> FsPath:
        return self.path / CONFIG_FILE

    def runs_dir(self, base: FsPath | None = None) -> FsPath:
        """``runs_dir`` (relative to ``base``, default the current directory)."""
        expanded, reason = expand_env(self.config.runs_dir)
        if expanded is None:
            raise SkillError(f"{self.config_path}: runs_dir: {reason}")
        path = FsPath(expanded).expanduser()
        return path if path.is_absolute() else (base or FsPath.cwd()) / path

    # -- precedence ---------------------------------------------------------------------------

    def matching_overrides(self, name: str, category: str) -> list[tuple[int, Override]]:
        return [
            (i, o) for i, o in enumerate(self.config.overrides, start=1)
            if o.match.matches(name, category)
        ]

    def resolve_models(
        self,
        name: str,
        category: str,
        scenario_models: Mapping[str, str] | None = None,
        cli: Mapping[str, str] | None = None,
    ) -> dict[str, Setting]:
        """``{role: Setting(spec, source)}`` for every model role, by the precedence ladder."""
        resolved: dict[str, Setting] = {}
        for role in MODEL_ROLES:
            setting = Setting(canonical_spec(DEFAULT_MODELS[role]), BUILTIN)
            role_file = self.roles.get(ROLE_FILE_FOR[role])
            if role_file is not None:
                setting = Setting(role_file.spec, f"roles/{role_file.path.name}")
            if role in self.config.defaults:
                setting = Setting(self.config.defaults[role], f"{CONFIG_FILE} defaults")
            for index, override in self.matching_overrides(name, category):
                if role in override.models:
                    setting = Setting(
                        override.models[role],
                        f"{CONFIG_FILE} overrides[{index}] ({override.match.label()})",
                    )
            if scenario_models and role in scenario_models:
                setting = Setting(canonical_spec(scenario_models[role]), SCENARIO_LAYER)
            if cli and role in cli:
                setting = Setting(canonical_spec(cli[role]), CLI_LAYER)
            resolved[role] = setting
        return resolved

    def resolve_run(
        self,
        name: str,
        category: str,
        scenario_run: Mapping[str, Any] | None = None,
        cli: Mapping[str, Any] | None = None,
    ) -> dict[str, Setting]:
        """``{key: Setting}`` for ``repeat``, ``modes``, ``judge_votes`` and ``concurrency``."""
        resolved = {k: Setting(v, BUILTIN) for k, v in BUILTIN_RUN.items()}
        judge = self.roles.get("judge")
        if judge is not None and "votes" in judge.extra:
            resolved["judge_votes"] = Setting(judge.extra["votes"], f"roles/{judge.path.name}")
        for key, value in self.config.run.given().items():
            resolved[key] = Setting(value, f"{CONFIG_FILE} run")
        for index, override in self.matching_overrides(name, category):
            for key, value in override.run.given().items():
                resolved[key] = Setting(
                    value, f"{CONFIG_FILE} overrides[{index}] ({override.match.label()})"
                )
        for key, value in (scenario_run or {}).items():
            if key in RUN_KEYS and value is not None:
                resolved[key] = Setting(value, SCENARIO_LAYER)
        for key, value in (cli or {}).items():
            if key in RUN_KEYS and value is not None:
                resolved[key] = Setting(value, CLI_LAYER)
        resolved["modes"] = Setting(list(resolved["modes"].value), resolved["modes"].source)
        return resolved

    def resolve(
        self,
        scenario: Scenario,
        *,
        model_overrides: Mapping[str, str] | None = None,
        run_overrides: Mapping[str, Any] | None = None,
    ) -> Resolved:
        """Resolve a freshly loaded scenario (its ``models`` / run fields as the file wrote
        them) against this skill and the CLI's overrides."""
        unknown = sorted(set(model_overrides or {}) - set(MODEL_ROLES))
        if unknown:
            raise ValueError(
                f"unknown model role(s) {', '.join(unknown)}; expected one of "
                f"{', '.join(MODEL_ROLES)}"
            )
        explicit_run = {
            k: getattr(scenario, k) for k in SCENARIO_RUN_FIELDS if k in scenario.model_fields_set
        }
        return Resolved(
            models=self.resolve_models(
                scenario.name, scenario.category, scenario.models.explicit(), model_overrides
            ),
            run=self.resolve_run(scenario.name, scenario.category, explicit_run, run_overrides),
        )

    def apply(
        self,
        scenario: Scenario,
        *,
        model_overrides: Mapping[str, str] | None = None,
        run_overrides: Mapping[str, Any] | None = None,
        allow_same_judge: bool = False,
    ) -> tuple[Scenario, Resolved]:
        """The scenario with every model and run setting resolved (what ``scenario.json``
        records), plus the resolution with its sources. Raises :class:`SkillError` when a role
        sets a temperature its resolved model rejects."""
        resolved = self.resolve(
            scenario, model_overrides=model_overrides, run_overrides=run_overrides
        )
        models = Models.model_validate(
            {
                **{role: s.value for role, s in resolved.models.items()},
                "allow_same_judge": scenario.models.allow_same_judge or allow_same_judge,
            }
        )
        updated = scenario.model_copy(
            update={
                "models": models,
                "repeat": resolved.run["repeat"].value,
                "judge_votes": resolved.run["judge_votes"].value,
                "concurrency": resolved.run["concurrency"].value,
            }
        )
        self.check_sampling(updated)
        return updated, resolved

    def check_sampling(self, scenario: Scenario) -> None:
        """A role whose frontmatter sets ``temperature`` must resolve to a model that accepts
        it (Claude Opus 5.5, Sonnet 5.5, Fable and Opus 4.7+ answer 400)."""
        local = parse_model_spec(scenario.models.planner)[0] == OLLAMA
        calls: dict[str, list[str]] = {
            "planner": [] if local else [scenario.models.planner],
            "planner-local": [scenario.models.planner] if local else [],
            "agent": [scenario.models.agent],
            "user": [scenario.models.user],
            "observer": [
                scenario.models.model_for_observer(o) for o in scenario.observers if o.kind == "llm"
            ],
            "judge": [scenario.models.judge],
        }
        for role_name, role in self.roles.items():
            if role.temperature is None:
                continue
            for spec in calls.get(role_name, []):
                provider, model = parse_model_spec(spec)
                if provider == ANTHROPIC and rejects_sampling(model):
                    raise SkillError(
                        f"{role.path}: temperature {role.temperature} is set, but the {role_name} "
                        f"model for scenario {scenario.name!r} is {model}, which rejects it "
                        "(the API answers 400); remove temperature or choose another model"
                    )

    # -- scenarios ----------------------------------------------------------------------------

    def sources(self, base: FsPath | None = None) -> list[ScenarioSource]:
        return [expand_source(e, base=base) for e in self.config.scenarios]

    def entries(self, base: FsPath | None = None) -> list[ScenarioEntry]:
        """Every configured scenario file, loaded (or with its load error)."""
        files: list[tuple[FsPath, str]] = []
        seen: set[FsPath] = set()
        for source in self.sources(base):
            for file in source.files:
                resolved = file.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append((file, source.entry))
        return load_entries(files)

    # -- description --------------------------------------------------------------------------

    def describe(
        self, *, scenario: Scenario | None = None, base: FsPath | None = None
    ) -> dict[str, Any]:
        """What ``mcpsim config --json`` prints (and the UI's ``/api/config`` can serve): the
        skill, its roles with their settings and prompts, the run defaults, the scenario
        sources, ``runs_dir``, the overrides and, with ``scenario``, that scenario's
        resolution."""
        name = scenario.name if scenario is not None else ""
        category = scenario.category if scenario is not None else ""
        models = (
            self.resolve(scenario).models
            if scenario is not None
            else self.resolve_models(name, category)
        )
        run = (
            self.resolve(scenario).run
            if scenario is not None
            else self.resolve_run(name, category)
        )
        roles: dict[str, Any] = {}
        for role_name, role in self.roles.items():
            spec = ROLE_SPECS[role_name]
            entry: dict[str, Any] = {
                "file": str(role.path),
                "summary": spec.summary,
                "description": role.description,
                "frontmatter_model": role.spec,
                "temperature": role.temperature,
                "max_tokens": role.max_tokens,
                "prompts": {
                    p: {
                        "accepts": list(spec.prompts[p].accepts),
                        "requires": list(spec.prompts[p].requires),
                    }
                    for p in role.prompts
                },
                **role.extra,
            }
            if spec.model_role is not None:
                entry["model"] = models[spec.model_role].value
                entry["model_source"] = models[spec.model_role].source
            roles[role_name] = entry
        sources = self.sources(base)
        return {
            "skill": {"name": self.name, "path": str(self.path), "description": self.description},
            "config": str(self.config_path),
            "roles": roles,
            "run": {k: {"value": s.value, "source": s.source} for k, s in run.items()},
            "scenarios": [s.to_json() for s in sources],
            "runs_dir": str(self.runs_dir(base)),
            "overrides": [o.model_dump(mode="json") for o in self.config.overrides],
            **({"scenario": name} if scenario is not None else {}),
        }


def skill_dir(explicit: str | os.PathLike[str] | None = None) -> FsPath:
    """``--skill DIR``, else ``MCPSIM_SKILL``, else the packaged skill (the source checkout's
    ``skills/simulate`` when mcpsim runs from one)."""
    if explicit is not None and str(explicit).strip():
        return FsPath(explicit).expanduser().resolve()
    env = os.environ.get(SKILL_ENV, "").strip()
    if env:
        return FsPath(env).expanduser().resolve()
    for candidate in (PACKAGED_DIR, CHECKOUT_DIR):
        if (candidate / SKILL_FILE).is_file():
            return candidate
    raise SkillError(
        f"no packaged {SKILL_NAME} skill found (looked in {PACKAGED_DIR} and {CHECKOUT_DIR}); "
        f"pass --skill DIR or set {SKILL_ENV}"
    )


def _signature(path: FsPath) -> tuple[tuple[str, int, int], ...]:
    """Every file a load reads, with its mtime and size: the cache key of :func:`load_skill`."""
    files = [path / SKILL_FILE, path / CONFIG_FILE]
    roles = DEFAULT_ROLES_DIR
    try:
        raw = yaml.safe_load((path / CONFIG_FILE).read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict) and isinstance(raw.get("roles_dir"), str):
            roles = raw["roles_dir"]
    except (OSError, yaml.YAMLError):
        pass  # the load itself reports a broken config
    roles_dir = path / roles
    if roles_dir.is_dir():
        files += sorted(roles_dir.glob("*.md"))
    out: list[tuple[str, int, int]] = []
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        out.append((str(f), st.st_mtime_ns, st.st_size))
    return tuple(out)


def load_skill(path: str | os.PathLike[str] | Skill | None = None) -> Skill:
    """Load (and validate) a skill directory; see :func:`skill_dir` for ``None``. A loaded
    skill is cached until one of its files changes."""
    if isinstance(path, Skill):
        return path
    directory = skill_dir(path)
    return _load_cached(directory, _signature(directory))


@lru_cache(maxsize=16)
def _load_cached(directory: FsPath, signature: tuple[tuple[str, int, int], ...]) -> Skill:
    del signature  # part of the cache key only
    return _load(directory)


def _load(directory: FsPath) -> Skill:
    if not directory.is_dir():
        raise SkillError(f"skill directory not found: {directory}")
    skill_md = directory / SKILL_FILE
    if not skill_md.is_file():
        raise SkillError(f"{directory}: no {SKILL_FILE}")
    try:
        meta, body, _ = _split_frontmatter(skill_md.read_text(encoding="utf-8"), str(skill_md))
    except OSError as exc:
        raise SkillError(f"{skill_md}: cannot read: {exc}") from None
    name = meta.get("name")
    description = meta.get("description")
    if not isinstance(name, str) or not name.strip():
        raise SkillError(f"{skill_md}: frontmatter needs a name")
    if not isinstance(description, str) or not description.strip():
        raise SkillError(f"{skill_md}: frontmatter needs a description")
    config_path = directory / CONFIG_FILE
    config = load_config(config_path) if config_path.is_file() else SkillConfig()
    roles_dir = directory / config.roles_dir
    if not roles_dir.is_dir():
        raise SkillError(f"{config_path}: roles_dir {roles_dir} is not a directory")
    known = {p.stem: p for p in sorted(roles_dir.glob("*.md"))}
    unknown = sorted(set(known) - set(ROLE_SPECS))
    if unknown:
        raise SkillError(
            f"{roles_dir}: unknown role file(s) {', '.join(f'{u}.md' for u in unknown)}; "
            f"roles: {', '.join(ROLE_NAMES)}"
        )
    missing = [r for r in ROLE_NAMES if r not in known]
    if missing:
        raise SkillError(
            f"{roles_dir}: missing role file(s) {', '.join(f'{m}.md' for m in missing)}"
        )
    roles = {name_: load_role(known[name_], name_) for name_ in ROLE_NAMES}
    if roles["planner-local"].provider != OLLAMA:
        raise SkillError(
            f"{roles['planner-local'].path}: provider must be ollama (this role's prompts are "
            "used only when the planner is an ollama: model)"
        )
    return Skill(
        path=directory,
        name=name.strip(),
        description=" ".join(description.split()),
        body=body.strip(),
        config=config,
        roles=roles,
    )


SkillRef = Skill | str | os.PathLike[str] | None
"""What every prompt-rendering function accepts: a loaded skill, a skill directory, or ``None``
for :func:`skill_dir`'s default."""


def role_of(skill: SkillRef, name: str) -> Role:
    """The named role of ``skill`` (loaded through :func:`load_skill`)."""
    return load_skill(skill).role(name)


def default_skill() -> Skill:
    """The skill library calls use when none is passed (``MCPSIM_SKILL`` or the packaged one)."""
    return load_skill(None)
