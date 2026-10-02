"""Scenario file model and loader (DESIGN §1, §3).

A scenario is one test case: who the agent acts for (``role``), what they want (``goal``),
the policies the agent must follow (``instructions``), what success looks like
(``expected_outcome``), which server to drive (``server``) and the run budgets.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Sequence
from pathlib import Path as FsPath
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_AGENT_MODEL = "claude-sonnet-5-5"
DEFAULT_PLANNER_MODEL = "claude-opus-5-5"
DEFAULT_JUDGE_MODEL = "claude-opus-5-5"
MODEL_ROLES: tuple[str, ...] = ("planner", "agent", "judge", "user", "observer")


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

    ``user`` is the simulated user's model and defaults to the agent's. ``allow_same_judge``
    lets the judge and the agent share a model (DESIGN §6 warns against it; docs/LOCAL_MODELS.md
    explains when a local-only setup needs it).
    """

    agent: str = Field(default=DEFAULT_AGENT_MODEL, min_length=1)
    planner: str = Field(default=DEFAULT_PLANNER_MODEL, min_length=1)
    judge: str = Field(default=DEFAULT_JUDGE_MODEL, min_length=1)
    user: str | None = Field(default=None, min_length=1)
    observer: str | None = Field(default=None, min_length=1)
    allow_same_judge: bool = False

    @property
    def user_model(self) -> str:
        """The simulated user's model: ``user`` when set, else the agent's."""
        return self.user if self.user is not None else self.agent

    @property
    def observer_model(self) -> str:
        """The default model for LLM observers: ``observer`` when set, else the agent's.

        An observer with its own ``model`` field overrides this (:meth:`model_for_observer`).
        """
        return self.observer if self.observer is not None else self.agent

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


class Scenario(_Strict):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
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
        self, observers: Sequence[Observer | ObserverLike], *, replace: bool = False
    ) -> Scenario:
        """A copy with ``observers`` appended (or, with ``replace``, substituted); validated.

        Accepts :class:`Observer` models and DSL builders (anything with a ``build()`` that
        returns one, see :mod:`mcpsim.observers`).
        """
        built = [o if isinstance(o, Observer) else o.build() for o in observers]
        merged = built if replace else [*self.observers, *built]
        return Scenario.model_validate({**self.model_dump(mode="python"), "observers": merged})


def _format_validation_error(path: FsPath, exc: ValidationError) -> str:
    lines = [f"{path}: invalid scenario"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)


def parse_scenario(data: Any, *, source: str = "<data>") -> Scenario:
    """Validate an already-parsed mapping into a :class:`Scenario`."""
    if not isinstance(data, dict):
        raise ScenarioError(f"{source}: top level must be a mapping, got {type(data).__name__}")
    try:
        return Scenario.model_validate(data)
    except ValidationError as exc:
        raise ScenarioError(_format_validation_error(FsPath(source), exc)) from exc


def load_scenario(path: str | FsPath) -> Scenario:
    """Load a YAML or JSON scenario file."""
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
    return parse_scenario(data, source=str(fs_path))
