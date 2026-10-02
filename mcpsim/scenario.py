"""Scenario file model and loader (DESIGN §1, §3).

A scenario is one test case: who the agent acts for (``role``), what they want (``goal``),
the policies the agent must follow (``instructions``), what success looks like
(``expected_outcome``), which server to drive (``server``) and the run budgets.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path as FsPath
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_AGENT_MODEL = "claude-sonnet-5-5"
DEFAULT_PLANNER_MODEL = "claude-opus-5-5"
DEFAULT_JUDGE_MODEL = "claude-opus-5-5"
MODEL_ROLES: tuple[str, ...] = ("planner", "agent", "judge", "user")


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
    allow_same_judge: bool = False

    @property
    def user_model(self) -> str:
        """The simulated user's model: ``user`` when set, else the agent's."""
        return self.user if self.user is not None else self.agent

    def for_role(self, role: str) -> str:
        """The model spec for ``planner`` / ``agent`` / ``judge`` / ``user``."""
        if role not in MODEL_ROLES:
            raise KeyError(f"unknown model role {role!r}; expected one of {', '.join(MODEL_ROLES)}")
        return self.user_model if role == "user" else str(getattr(self, role))


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

    @model_validator(mode="after")
    def _instructions_non_blank(self) -> Scenario:
        for i, item in enumerate(self.instructions):
            if not item.strip():
                raise ValueError(f"instructions[{i}] is blank")
        return self


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
