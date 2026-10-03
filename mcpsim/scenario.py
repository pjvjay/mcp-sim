"""Scenario file model and loader (DESIGN §1, §3).

A scenario is one test case: who the agent acts for (``role``), what they want (``goal``),
the policies the agent must follow (``instructions``), what success looks like
(``expected_outcome``), which server to drive (``server``) and the run budgets.

Scenario v2 (DESIGN §3 "Scenario v2") adds, all optional so v1 files keep loading: a runner
``category`` and display ``title``; ``user_instructions``, the second-person brief the simulated
user plays (default built from ``role`` + ``goal``); ``context`` (device, location, language,
details), which the simulated user always gets and the agent only when ``agent_visible``;
``expected_behavior``, the observable behaviours the judge grades one by one (default: the
``instructions``); and ``agent``, the agent under test's standard operating procedure (a
SKILL.md, frontmatter stripped) and extra system ``notes``.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Sequence
from pathlib import Path as FsPath
from typing import Any, Literal, Protocol

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

DEFAULT_AGENT_MODEL = "claude-sonnet-5-5"
DEFAULT_PLANNER_MODEL = "claude-opus-5-5"
DEFAULT_JUDGE_MODEL = "claude-opus-5-5"
DEFAULT_USER_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OBSERVER_MODEL = "claude-sonnet-5-5"
MODEL_ROLES: tuple[str, ...] = ("planner", "agent", "judge", "user", "observer")
# Every role's built-in default; all of them are Anthropic API models. The local (Ollama)
# planner profile is reached only by naming an ``ollama:`` planner explicitly.
DEFAULT_MODELS: dict[str, str] = {
    "planner": DEFAULT_PLANNER_MODEL,
    "agent": DEFAULT_AGENT_MODEL,
    "user": DEFAULT_USER_MODEL,
    "observer": DEFAULT_OBSERVER_MODEL,
    "judge": DEFAULT_JUDGE_MODEL,
}

DEFAULT_CATEGORY = "Uncategorized"
SKILL_FILE = "SKILL.md"
SKILL_ENV_PREFIX = "env:"
# The validation-context key :func:`load_scenario` sets so a relative ``agent.skill`` resolves
# against the scenario file's directory.
BASE_DIR_CONTEXT = "base_dir"
# Set to False in the validation context to validate a scenario without reading any file: an
# ``agent.skill`` without ``skill_text`` then stays unresolved (a run's ``scenario.json`` read
# for display, where the path is data, not something to open).
READ_SKILL_CONTEXT = "read_skill"


class ScenarioError(ValueError):
    """A scenario file could not be read or failed validation."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


with warnings.catch_warnings():
    # The field is called ``json`` by contract (DESIGN §3); pydantic warns that it shadows
    # the deprecated ``BaseModel.json`` method. The attribute access still returns the field.
    warnings.simplefilter("ignore", UserWarning)

    class ExpectedOutcome(_Strict):
        """What success looks like: prose for the LLM judge and/or a JSON spec (DESIGN §4)."""

        text: str | None = None
        json: dict[str, Any] | None = None  # type: ignore[assignment]

        @model_validator(mode="after")
        def _at_least_one(self) -> ExpectedOutcome:
            if self.text is None and self.json is None:
                raise ValueError("expected_outcome needs at least one of 'text' or 'json'")
            if self.text is not None and not self.text.strip():
                raise ValueError("expected_outcome.text must not be blank")
            return self


class StdioSpec(_Strict):
    """Launch the server as a subprocess speaking MCP over stdio."""

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    setup: str | None = None


class HttpSpec(_Strict):
    """Connect to a Streamable HTTP MCP endpoint; the bearer token is read from ``bearer_env``."""

    url: str
    bearer_env: str | None = None


class ServerSpec(_Strict):
    """Exactly one of ``stdio`` or ``http``."""

    stdio: StdioSpec | None = None
    http: HttpSpec | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ServerSpec:
        if (self.stdio is None) == (self.http is None):
            raise ValueError("server needs exactly one of 'stdio' or 'http'")
        return self

    @property
    def kind(self) -> str:
        return "stdio" if self.stdio is not None else "http"


class Budgets(_Strict):
    max_turns: int = Field(default=12, ge=1)
    max_tool_calls: int = Field(default=20, ge=1)
    max_cost_usd: float = Field(default=1.0, gt=0)


class Models(_Strict):
    """Which model serves each role, as ``provider:model`` (a bare name means ``anthropic``).

    Every role defaults to an Anthropic model (:data:`DEFAULT_MODELS`): planner and judge
    ``claude-opus-5-5``, agent and observers ``claude-sonnet-5-5``, the simulated user
    ``claude-haiku-4-5-20251001``. A ``null`` ``user`` / ``observer`` (as v1 ``scenario.json``
    files hold) means that default. ``allow_same_judge`` lets the judge and the agent share a
    model (DESIGN §6 warns against it; docs/LOCAL_MODELS.md explains when a local-only setup
    needs it). :meth:`explicit` says which roles the scenario file itself named.
    """

    agent: str = Field(default=DEFAULT_AGENT_MODEL, min_length=1)
    planner: str = Field(default=DEFAULT_PLANNER_MODEL, min_length=1)
    judge: str = Field(default=DEFAULT_JUDGE_MODEL, min_length=1)
    user: str = Field(default=DEFAULT_USER_MODEL, min_length=1)
    observer: str = Field(default=DEFAULT_OBSERVER_MODEL, min_length=1)
    allow_same_judge: bool = False

    @field_validator("user", "observer", mode="before")
    @classmethod
    def _null_is_default(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            return DEFAULT_MODELS[str(info.field_name)]
        return value

    def explicit(self) -> dict[str, str]:
        """``{role: spec}`` for the roles set explicitly when this object was validated (a
        scenario file's own ``models``), so configuration layers can sit underneath it. A copy
        made through ``model_validate(model_dump())`` marks every role explicit, so call this
        on the freshly loaded scenario."""
        return {
            role: str(getattr(self, role)) for role in MODEL_ROLES if role in self.model_fields_set
        }

    @property
    def user_model(self) -> str:
        """The simulated user's model (``user``)."""
        return self.user

    @property
    def observer_model(self) -> str:
        """The default model for LLM observers (``observer``).

        An observer with its own ``model`` field overrides this (:meth:`model_for_observer`).
        """
        return self.observer

    def model_for_observer(self, observer: Observer) -> str:
        return observer.model if observer.model is not None else self.observer_model

    def for_role(self, role: str) -> str:
        """The model spec for ``planner`` / ``agent`` / ``judge`` / ``user`` / ``observer``."""
        if role not in MODEL_ROLES:
            raise KeyError(f"unknown model role {role!r}; expected one of {', '.join(MODEL_ROLES)}")
        if role == "user":
            return self.user_model
        if role == "observer":
            return self.observer_model
        return str(getattr(self, role))


Disclosure = Literal["all", "plan", "progressive"]
DISCLOSURES: tuple[Disclosure, ...] = ("all", "plan", "progressive")


class ToolPolicy(_Strict):
    """Which of the server's tools the agent under test may see, and when (DESIGN §2).

    ``allow`` then ``deny`` are ``fnmatch`` globs over tool names; what survives is the
    *allowed catalog* that the planner, the agent and the dry run work from. ``disclosure``
    says how much of it the agent sees at once: ``all`` from turn one; ``plan`` only the path's
    tools in guided mode (everything in free mode); ``progressive`` a relevance-scored starting
    set (or the explicit ``initial`` globs) that grows through the framework's
    ``discover_tools`` meta-tool (unless ``discover_tool`` is false) and through observers.
    A glob that matches nothing is not an error here (the catalog is unknown until the server
    answers); the runner warns about it.
    """

    allow: list[str] = Field(default_factory=lambda: ["*"])
    deny: list[str] = Field(default_factory=list)
    disclosure: Disclosure = "all"
    initial: list[str] | None = None
    discover_tool: bool = True

    @model_validator(mode="after")
    def _progressive_only_fields(self) -> ToolPolicy:
        if self.initial is not None and self.disclosure != "progressive":
            raise ValueError(
                f"tools.initial only applies to disclosure 'progressive' "
                f"(this scenario says {self.disclosure!r})"
            )
        for field_name, patterns in (("allow", self.allow), ("deny", self.deny)):
            for i, pattern in enumerate(patterns):
                if not pattern.strip():
                    raise ValueError(f"tools.{field_name}[{i}] is blank")
        if not self.allow:
            raise ValueError("tools.allow must list at least one glob ('*' allows every tool)")
        return self


# --- observers: the Informant-Report Method (DESIGN §2b) --------------------------------------

Watch = Literal["conversation", "tool_traffic", "final_answer", "scout", "all"]
WATCHES: tuple[Watch, ...] = ("conversation", "tool_traffic", "final_answer", "scout", "all")
Trigger = Literal["scout", "turn", "tool_result", "end"]
TRIGGERS: tuple[Trigger, ...] = ("scout", "turn", "tool_result", "end")
ObserverKind = Literal["llm", "code", "group"]
DEFAULT_WATCHES: tuple[Watch, ...] = ("all",)
# The cheap pair: observers report at plan time and after the final answer unless asked for more.
DEFAULT_TRIGGERS: tuple[Trigger, ...] = ("scout", "end")

IDENTIFIER_PATTERN = r"^[a-z_][a-z0-9_]*$"
_IDENTIFIER_RE = re.compile(IDENTIFIER_PATTERN)
# A group condition term: ``<observer>.<condition>`` or ``!<observer>.<condition>``.
CONDITION_REF_PATTERN = r"^!?[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$"
_CONDITION_REF_RE = re.compile(CONDITION_REF_PATTERN)

# ``check`` vocabularies (``kind: code``).
CHECK_SLICES: tuple[str, ...] = ("final_answer", "last_assistant", "conversation", "errors")
_TOOL_RESULT_SLICE_RE = re.compile(r"^tool_result\[([A-Za-z_][A-Za-z0-9_]*)\]$")
COUNT_OPERATORS: tuple[str, ...] = ("gt", "gte", "lt", "lte", "eq")
USE_KEY = "use"


class Effect(_Strict):
    """What a reported condition does to the run (``then`` on true, ``otherwise`` on false).

    ``enable_tools`` / ``disable_tools`` are names or globs over the allowed catalog (disclosure
    grows with reason ``observer:<observer>.<condition>``); ``enable_goal`` is appended to the
    subject's instructions for its next turns; ``flag`` is recorded for the judge; ``fail`` is a
    deterministic failure of the run, like a matcher failure; ``note`` is free text for people.
    """

    enable_tools: list[str] = Field(default_factory=list)
    disable_tools: list[str] = Field(default_factory=list)
    enable_goal: str | None = None
    flag: str | None = None
    fail: bool = False
    note: str | None = None

    @model_validator(mode="after")
    def _non_blank(self) -> Effect:
        for field_name in ("enable_tools", "disable_tools"):
            for i, pattern in enumerate(getattr(self, field_name)):
                if not pattern.strip():
                    raise ValueError(f"{field_name}[{i}] is blank")
        for field_name in ("enable_goal", "flag", "note"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        if self.flag is not None and not _IDENTIFIER_RE.match(self.flag):
            raise ValueError(f"flag {self.flag!r} must match {IDENTIFIER_PATTERN}")
        return self

    def is_empty(self) -> bool:
        return not (
            self.enable_tools
            or self.disable_tools
            or self.enable_goal
            or self.flag
            or self.fail
            or self.note
        )


def is_check_slice(name: str) -> bool:
    """One of :data:`CHECK_SLICES` or ``tool_result[<tool>]``."""
    return name in CHECK_SLICES or _TOOL_RESULT_SLICE_RE.match(name) is not None


def tool_result_slice(name: str) -> str | None:
    """The tool a ``tool_result[<tool>]`` slice names, or ``None``."""
    m = _TOOL_RESULT_SLICE_RE.match(name)
    return m.group(1) if m else None


class Check(_Strict):
    """A deterministic validator for a ``kind: code`` condition; exactly one key is set.

    * ``word_count``: ``{of: <slice>, gt|gte|lt|lte|eq: n}``
    * ``regex``: ``{of: <slice>|tool_result[<tool>], pattern: str, flags?: "i"}``
    * ``tool_result``: ``{tool: str, where: <matcher spec over the LAST structured result>}``
    * ``tool_called``: a tool name; true iff a ``tool_call`` event for it exists

    where ``<slice>`` is ``final_answer``, ``last_assistant``, ``conversation`` or ``errors``.
    """

    word_count: dict[str, Any] | None = None
    regex: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    tool_called: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Check:
        set_keys = [
            k
            for k in ("word_count", "regex", "tool_result", "tool_called")
            if getattr(self, k) is not None
        ]
        if len(set_keys) != 1:
            raise ValueError(
                "check needs exactly one of word_count, regex, tool_result, tool_called "
                f"(got {', '.join(set_keys) or 'none'})"
            )
        if self.word_count is not None:
            spec = self.word_count
            of = spec.get("of")
            if of not in CHECK_SLICES:
                raise ValueError(
                    f"word_count.of must be one of {', '.join(CHECK_SLICES)}, got {of!r}"
                )
            ops = [k for k in spec if k in COUNT_OPERATORS]
            extra = [k for k in spec if k != "of" and k not in COUNT_OPERATORS]
            if extra:
                raise ValueError(f"word_count has unknown key(s): {', '.join(extra)}")
            if len(ops) != 1:
                raise ValueError(f"word_count needs exactly one of {', '.join(COUNT_OPERATORS)}")
            if isinstance(spec[ops[0]], bool) or not isinstance(spec[ops[0]], int):
                raise ValueError(f"word_count.{ops[0]} must be an integer")
        if self.regex is not None:
            spec = self.regex
            of = spec.get("of")
            if not isinstance(of, str) or not is_check_slice(of):
                raise ValueError(
                    f"regex.of must be one of {', '.join(CHECK_SLICES)} or tool_result[<tool>], "
                    f"got {of!r}"
                )
            pattern = spec.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise ValueError("regex.pattern must be a non-empty string")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"regex.pattern is not a valid regular expression: {exc}"
                ) from None
            flags = spec.get("flags", "")
            if not isinstance(flags, str) or any(ch not in "imsx" for ch in flags):
                raise ValueError(f"regex.flags must be letters from i, m, s, x; got {flags!r}")
            extra = [k for k in spec if k not in ("of", "pattern", "flags")]
            if extra:
                raise ValueError(f"regex has unknown key(s): {', '.join(extra)}")
        if self.tool_result is not None:
            spec = self.tool_result
            tool = spec.get("tool")
            if not isinstance(tool, str) or not tool.strip():
                raise ValueError("tool_result.tool must name a tool")
            where = spec.get("where")
            if not isinstance(where, dict) or not where:
                raise ValueError("tool_result.where must be a non-empty matcher spec (a mapping)")
            extra = [k for k in spec if k not in ("tool", "where")]
            if extra:
                raise ValueError(f"tool_result has unknown key(s): {', '.join(extra)}")
        if self.tool_called is not None and not self.tool_called.strip():
            raise ValueError("tool_called must name a tool")
        return self

    @property
    def kind(self) -> str:
        for k in ("word_count", "regex", "tool_result", "tool_called"):
            if getattr(self, k) is not None:
                return k
        raise AssertionError("unreachable: Check validation guarantees one key")


class Condition(_Strict):
    """One ``when(...)`` clause: the conditional an observer validates, and its effects.

    ``when`` is the natural-language conditional and is always present (the record of intent).
    ``check`` is the code validator (``kind: code``); ``all_of`` / ``any_of`` combine other
    observers' reports (``kind: group``). ``then`` applies when the condition is reported true,
    ``otherwise`` when it is reported false; neither applies to an unknown report.
    """

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    when: str = Field(min_length=1)
    check: Check | None = None
    all_of: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)
    then: Effect = Field(default_factory=Effect)
    otherwise: Effect = Field(default_factory=Effect)

    @model_validator(mode="after")
    def _shape(self) -> Condition:
        if not self.when.strip():
            raise ValueError("when must not be blank")
        for field_name in ("all_of", "any_of"):
            for i, ref in enumerate(getattr(self, field_name)):
                if not _CONDITION_REF_RE.match(ref):
                    raise ValueError(
                        f"{field_name}[{i}] {ref!r} must be '<observer>.<condition>' or "
                        "'!<observer>.<condition>'"
                    )
        return self

    def references(self) -> list[tuple[str, str, bool]]:
        """``(observer, condition, negated)`` for every ``all_of`` / ``any_of`` term."""
        found: list[tuple[str, str, bool]] = []
        for ref in [*self.all_of, *self.any_of]:
            negated = ref.startswith("!")
            observer, _, condition = ref.lstrip("!").partition(".")
            found.append((observer, condition, negated))
        return found


class Observer(_Strict):
    """An informant with a social identity, reporting on the subject from what it watches.

    ``kind`` is ``llm`` (one model call per trigger covering every condition), ``code``
    (deterministic ``check`` validators, no model) or ``group`` (boolean algebra over other
    observers' reports). ``watches`` are the transcript slices it may see; ``on`` the triggers at
    which it reports (default ``scout`` and ``end``, the cheap pair). ``model`` overrides
    ``models.observer`` (which defaults to the agent's model).
    """

    name: str = Field(pattern=IDENTIFIER_PATTERN)
    identity: str = Field(min_length=1)
    kind: ObserverKind = "llm"
    watches: list[Watch] = Field(default_factory=lambda: list(DEFAULT_WATCHES))
    on: list[Trigger] = Field(default_factory=lambda: list(DEFAULT_TRIGGERS))
    model: str | None = Field(default=None, min_length=1)
    conditions: list[Condition] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _yaml_on_key(cls, data: Any) -> Any:
        """YAML 1.1 reads a bare ``on:`` key as the boolean ``true``; accept it as ``on``."""
        if isinstance(data, dict) and True in data and "on" not in data:
            data = {("on" if k is True else k): v for k, v in data.items()}
        return data

    @model_validator(mode="after")
    def _kind_shape(self) -> Observer:
        if not self.identity.strip():
            raise ValueError("identity must not be blank")
        if not self.watches:
            raise ValueError("watches must list at least one slice")
        if not self.on:
            raise ValueError("on must list at least one trigger")
        seen: set[str] = set()
        for i, cond in enumerate(self.conditions):
            where = f"conditions[{i}] ({cond.id!r})"
            if cond.id in seen:
                raise ValueError(f"{where}: duplicate condition id within observer {self.name!r}")
            seen.add(cond.id)
            grouped = bool(cond.all_of or cond.any_of)
            if self.kind == "code":
                if cond.check is None:
                    raise ValueError(f"{where}: kind 'code' needs a check")
                if grouped:
                    raise ValueError(f"{where}: kind 'code' takes a check, not all_of/any_of")
            elif self.kind == "group":
                if not grouped:
                    raise ValueError(f"{where}: kind 'group' needs all_of or any_of")
                if cond.check is not None:
                    raise ValueError(f"{where}: kind 'group' combines reports; it takes no check")
            else:
                if cond.check is not None:
                    raise ValueError(
                        f"{where}: kind 'llm' answers from what it watches; a check belongs to "
                        "kind 'code'"
                    )
                if grouped:
                    raise ValueError(
                        f"{where}: kind 'llm' takes no all_of/any_of; use kind 'group' to combine"
                    )
        return self

    def condition(self, condition_id: str) -> Condition:
        for cond in self.conditions:
            if cond.id == condition_id:
                return cond
        raise KeyError(f"observer {self.name!r} has no condition {condition_id!r}")

    def has_condition(self, condition_id: str) -> bool:
        return any(c.id == condition_id for c in self.conditions)


def resolve_observer_entry(entry: Any, library: dict[str, dict[str, Any]]) -> Any:
    """``{use: <name>}`` → the library observer's declaration; anything else passes through."""
    if not isinstance(entry, dict) or USE_KEY not in entry:
        return entry
    name = entry[USE_KEY]
    extra = sorted(k for k in entry if k != USE_KEY)
    if extra:
        raise ValueError(
            f"an observer entry with '{USE_KEY}' takes no other keys (got {', '.join(extra)})"
        )
    if name not in library:
        known = ", ".join(sorted(library)) or "(none)"
        raise ValueError(f"unknown built-in observer {name!r}; the library has: {known}")
    return dict(library[name])


def validate_observer_graph(observers: list[Observer]) -> None:
    """Unique names; group terms reference conditions of observers declared EARLIER in the list
    (so there are no cycles by construction). Raises ``ValueError`` naming the field."""
    by_name: dict[str, Observer] = {}
    for i, obs in enumerate(observers):
        if obs.name in by_name:
            raise ValueError(f"observers[{i}]: duplicate observer name {obs.name!r}")
        for c_index, cond in enumerate(obs.conditions):
            for observer_name, condition_id, _ in cond.references():
                where = f"observers[{i}].conditions[{c_index}] ({obs.name}.{cond.id})"
                if observer_name == obs.name:
                    raise ValueError(
                        f"{where}: references its own observer; a group may only combine "
                        "observers declared earlier in the list"
                    )
                earlier = by_name.get(observer_name)
                if earlier is None:
                    raise ValueError(
                        f"{where}: references {observer_name}.{condition_id}, but no observer "
                        f"named {observer_name!r} is declared earlier in the list"
                    )
                if not earlier.has_condition(condition_id):
                    known = ", ".join(c.id for c in earlier.conditions)
                    raise ValueError(
                        f"{where}: observer {observer_name!r} has no condition "
                        f"{condition_id!r} (it has: {known})"
                    )
        by_name[obs.name] = obs


class ObserverLike(Protocol):
    """What :meth:`Scenario.with_observers` accepts besides an :class:`Observer`: a builder."""

    def build(self) -> Observer: ...


# --- scenario v2: context, the agent's SOP skill, defaults ------------------------------------


class Context(_Strict):
    """The situation the simulated user is in (DESIGN §3 "Scenario v2").

    The simulated user always gets it; the agent under test only when ``agent_visible`` is
    true (a deployed agent may know the channel and location, or may not). ``details`` holds
    anything else worth saying (``{account: "guest", time: "Friday 6 pm"}``).
    """

    device: str | None = None
    location: str | None = None
    language: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    agent_visible: bool = False

    @model_validator(mode="after")
    def _non_blank(self) -> Context:
        for field_name in ("device", "location", "language"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"context.{field_name} must not be blank")
        for key in self.details:
            if not str(key).strip():
                raise ValueError("context.details has a blank key")
        return self

    def is_empty(self) -> bool:
        return not (self.device or self.location or self.language or self.details)

    def items(self) -> list[tuple[str, str]]:
        """``(label, value)`` pairs in display order: device, location, language, then
        ``details`` in file order (a non-string value as compact JSON)."""
        pairs: list[tuple[str, str]] = []
        for field_name in ("device", "location", "language"):
            value = getattr(self, field_name)
            if value is not None:
                pairs.append((field_name, value.strip()))
        for key, value in self.details.items():
            shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            pairs.append((str(key).strip(), str(shown).strip()))
        return pairs


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """``(frontmatter, body)`` of a Markdown file whose first line may open a ``---`` YAML block.

    A file without frontmatter is all body. Raises ``ValueError`` when the block is not closed
    or is not a YAML mapping.
    """
    stripped = text.lstrip("﻿")
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, stripped.strip()
    end = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end is None:
        raise ValueError("the YAML frontmatter opened by '---' on line 1 is never closed")
    raw = "\n".join(lines[1:end])
    try:
        meta = yaml.safe_load(raw) if raw.strip() else {}
    except yaml.YAMLError as exc:
        raise ValueError(f"the YAML frontmatter is not valid YAML: {exc}") from None
    if not isinstance(meta, dict):
        raise ValueError(f"the YAML frontmatter must be a mapping, got {type(meta).__name__}")
    return meta, "\n".join(lines[end + 1 :]).strip()


def resolve_skill_path(spec: str, base_dir: str | os.PathLike[str] | None = None) -> FsPath:
    """The SKILL.md an ``agent.skill`` value names (DESIGN §3 "Scenario v2").

    * ``env:VAR`` — the path held in environment variable ``VAR`` (a relative value resolves
      against the current working directory, where the variable was set);
    * an absolute path (``~`` expands);
    * a relative path, resolved against ``base_dir`` (the scenario file's directory) or, without
      one, the current working directory.

    A directory means its ``SKILL.md``. Raises ``ValueError`` with the spec and the path tried
    when the variable is unset or the file does not exist.
    """
    text = spec.strip()
    if text.startswith(SKILL_ENV_PREFIX):
        var = text[len(SKILL_ENV_PREFIX) :].strip()
        if not var:
            raise ValueError(f"skill {spec!r}: name the environment variable after 'env:'")
        value = os.environ.get(var, "").strip()
        if not value:
            raise ValueError(
                f"skill {spec!r}: environment variable {var} is not set (it must hold the path "
                f"to a {SKILL_FILE})"
            )
        path = FsPath(value).expanduser()
        if not path.is_absolute():
            path = FsPath.cwd() / path
    else:
        path = FsPath(text).expanduser()
        if not path.is_absolute():
            path = (FsPath(base_dir) if base_dir is not None else FsPath.cwd()) / path
    if path.is_dir():
        path = path / SKILL_FILE
    if not path.is_file():
        raise ValueError(f"skill {spec!r}: no {SKILL_FILE} at {path}")
    return path.resolve()


def read_skill(path: str | os.PathLike[str]) -> tuple[str, str]:
    """``(name, body)`` of a SKILL.md: the frontmatter ``name`` (else the folder's name) and the
    Markdown body with the frontmatter stripped. Raises ``ValueError`` for an unreadable file,
    a broken frontmatter block or an empty body."""
    fs_path = FsPath(path)
    try:
        text = fs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read skill {fs_path}: {exc}") from None
    try:
        meta, body = split_frontmatter(text)
    except ValueError as exc:
        raise ValueError(f"skill {fs_path}: {exc}") from None
    if not body:
        raise ValueError(f"skill {fs_path}: no procedure after the frontmatter")
    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        name = fs_path.parent.name or fs_path.stem
    return name.strip(), body


class AgentSpec(_Strict):
    """The agent under test's standard operating procedure and extra system text.

    ``skill`` names a SKILL.md (see :func:`resolve_skill_path`); when the scenario is validated
    it is read and ``skill_path`` / ``skill_name`` / ``skill_text`` are filled, so the
    ``scenario.json`` of a run records the exact procedure the agent ran on (and a re-judge
    never re-reads the file). ``skill_text`` may also be given inline instead of ``skill``.
    With ``read_skill: False`` in the validation context (:data:`READ_SKILL_CONTEXT`) nothing
    is read and a ``skill`` without ``skill_text`` stays unresolved.
    ``notes`` is extra system text for the agent, such as the limits of this environment.
    """

    skill: str | None = None
    notes: str | None = None
    skill_path: str | None = None
    skill_name: str | None = None
    skill_text: str | None = None

    @model_validator(mode="after")
    def _resolve_skill(self, info: ValidationInfo) -> AgentSpec:
        if self.skill is not None and not self.skill.strip():
            raise ValueError("agent.skill must not be blank")
        if self.notes is not None and not self.notes.strip():
            raise ValueError("agent.notes must not be blank")
        if self.skill_text is not None and not self.skill_text.strip():
            raise ValueError("agent.skill_text must not be blank")
        context = info.context if isinstance(info.context, dict) else {}
        if (
            self.skill is not None
            and self.skill_text is None
            and context.get(READ_SKILL_CONTEXT, True) is not False
        ):
            path = resolve_skill_path(self.skill, context.get(BASE_DIR_CONTEXT))
            name, body = read_skill(path)
            self.skill_path = str(path)
            self.skill_name = self.skill_name or name
            self.skill_text = body
        if self.skill_text is not None and not self.skill_name:
            self.skill_name = "inline"
        return self

    @property
    def has_sop(self) -> bool:
        """True when the agent runs on a standard operating procedure."""
        return self.skill_text is not None


def default_title(name: str) -> str:
    """``cheapest-penne`` -> ``Cheapest penne``."""
    words = re.sub(r"[-_.]+", " ", name).strip()
    return words[:1].upper() + words[1:] if words else name


def default_user_instructions(role: str, goal: str) -> str:
    """The v1 fallback for ``user_instructions``: the persona and the goal, in the second
    person."""
    return (
        f"You are this person: {role.strip()}\n\n"
        f"What you want from the assistant: {goal.strip()}"
    )


class Scenario(_Strict):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    # v2 (all optional; filled with their defaults after validation, see _v2_defaults)
    category: str = DEFAULT_CATEGORY
    title: str | None = None
    user_instructions: str | None = None
    context: Context = Field(default_factory=Context)
    expected_behavior: list[str] | None = None
    agent: AgentSpec = Field(default_factory=AgentSpec)
    # v1
    role: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    instructions: list[str] = Field(default_factory=list)
    expected_outcome: ExpectedOutcome
    server: ServerSpec
    repeat: int = Field(default=3, ge=1)
    judge_votes: int = Field(default=3, ge=1)
    budgets: Budgets = Field(default_factory=Budgets)
    models: Models = Field(default_factory=Models)
    concurrency: int = Field(default=4, ge=1)
    tools: ToolPolicy = Field(default_factory=ToolPolicy)
    observers: list[Observer] = Field(default_factory=list)

    @model_validator(mode="after")
    def _v2_defaults(self) -> Scenario:
        """``title`` from ``name``, ``user_instructions`` from ``role`` + ``goal``,
        ``expected_behavior`` from ``instructions``; blanks are errors, not defaults."""
        if not self.category.strip():
            raise ValueError("category must not be blank")
        self.category = self.category.strip()
        if self.title is not None and not self.title.strip():
            raise ValueError("title must not be blank")
        if self.title is None:
            self.title = default_title(self.name)
        if self.user_instructions is not None and not self.user_instructions.strip():
            raise ValueError("user_instructions must not be blank")
        if self.user_instructions is None:
            self.user_instructions = default_user_instructions(self.role, self.goal)
        if self.expected_behavior is None:
            # A blank instruction is reported by _instructions_non_blank below.
            self.expected_behavior = [item.strip() for item in self.instructions]
            return self
        for i, item in enumerate(self.expected_behavior):
            if not item.strip():
                raise ValueError(f"expected_behavior[{i}] is blank")
        self.expected_behavior = [item.strip() for item in self.expected_behavior]
        return self

    @property
    def display_title(self) -> str:
        return self.title or default_title(self.name)

    @property
    def simulated_user_instructions(self) -> str:
        """What the simulated user is told (always set after validation)."""
        return self.user_instructions or default_user_instructions(self.role, self.goal)

    @property
    def behaviors(self) -> list[str]:
        """The expected-behaviour items the judge grades, in order (always set after
        validation; the ``instructions`` when the file gives none)."""
        if self.expected_behavior is None:
            return [item.strip() for item in self.instructions]
        return [item.strip() for item in self.expected_behavior]

    @model_validator(mode="before")
    @classmethod
    def _resolve_builtin_observers(cls, data: Any) -> Any:
        """``observers: [{use: fabrication_auditor}, ...]`` pulls built-ins from the library."""
        if not isinstance(data, dict) or not isinstance(data.get("observers"), list):
            return data
        from mcpsim.observer_library import BUILTIN_OBSERVERS

        resolved: list[Any] = []
        for i, entry in enumerate(data["observers"]):
            try:
                resolved.append(resolve_observer_entry(entry, BUILTIN_OBSERVERS))
            except ValueError as exc:
                raise ValueError(f"observers[{i}]: {exc}") from None
        return {**data, "observers": resolved}

    @model_validator(mode="after")
    def _instructions_non_blank(self) -> Scenario:
        for i, item in enumerate(self.instructions):
            if not item.strip():
                raise ValueError(f"instructions[{i}] is blank")
        validate_observer_graph(self.observers)
        return self

    def observer(self, name: str) -> Observer:
        for obs in self.observers:
            if obs.name == name:
                return obs
        raise KeyError(f"scenario {self.name!r} has no observer {name!r}")

    def with_observers(
        self,
        observers: Sequence[Observer | ObserverLike | dict[str, Any]],
        *,
        replace: bool = False,
    ) -> Scenario:
        """A copy with ``observers`` appended (or, with ``replace``, substituted); validated.

        Accepts :class:`Observer` models, DSL builders (anything with a ``build()`` that returns
        one, see :mod:`mcpsim.observers`) and plain declarations as a scenario file holds them
        (``{use: <name>}`` included).
        """
        built: list[Observer | dict[str, Any]] = [
            o if isinstance(o, Observer | dict) else o.build() for o in observers
        ]
        merged = built if replace else [*self.observers, *built]
        return Scenario.model_validate({**self.model_dump(mode="python"), "observers": merged})


def _format_validation_error(path: FsPath, exc: ValidationError) -> str:
    lines = [f"{path}: invalid scenario"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)


def parse_scenario(
    data: Any,
    *,
    source: str = "<data>",
    base_dir: str | os.PathLike[str] | None = None,
) -> Scenario:
    """Validate an already-parsed mapping into a :class:`Scenario`.

    ``base_dir`` is where a relative ``agent.skill`` resolves (the scenario file's directory;
    the current working directory when not given).
    """
    if not isinstance(data, dict):
        raise ScenarioError(f"{source}: top level must be a mapping, got {type(data).__name__}")
    context = {BASE_DIR_CONTEXT: str(base_dir)} if base_dir is not None else None
    try:
        return Scenario.model_validate(data, context=context)
    except ValidationError as exc:
        raise ScenarioError(_format_validation_error(FsPath(source), exc)) from exc


def read_scenario_mapping(path: str | FsPath) -> dict[str, Any] | None:
    """A scenario file's top-level mapping as plain YAML / JSON, without validating it, or
    ``None`` when it cannot be read or parsed. For showing and selecting a file that does not
    load: its ``name``, ``title`` and ``category`` say where it belongs."""
    fs_path = FsPath(path)
    try:
        text = fs_path.read_text(encoding="utf-8")
        data = json.loads(text) if fs_path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_scenario(path: str | FsPath) -> Scenario:
    """Load a YAML or JSON scenario file (a relative ``agent.skill`` resolves against the
    file's directory)."""
    fs_path = FsPath(path)
    try:
        raw = fs_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(f"{fs_path}: cannot read scenario: {exc}") from exc
    try:
        if fs_path.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            data = yaml.safe_load(raw)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ScenarioError(f"{fs_path}: cannot parse scenario: {exc}") from exc
    return parse_scenario(data, source=str(fs_path), base_dir=fs_path.resolve().parent)
