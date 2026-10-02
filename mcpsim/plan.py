"""Execution plan model (DESIGN §1 "Execution plan", "Path").

The planner emits an :class:`ExecutionPlan`; the executor runs each :class:`Path`; the judge
reads each path's ``checkpoints``. Plans are saved as ``plan.json`` and are meant to be
reviewed, edited and re-run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path as FsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PathKind = Literal["happy", "recovery", "alternative", "boundary", "policy"]
PATH_KINDS: tuple[PathKind, ...] = ("happy", "recovery", "alternative", "boundary", "policy")

Mode = Literal["guided", "free"]
MODES: tuple[Mode, ...] = ("guided", "free")

# An ``arguments_sketch`` value of this shape is a reference to an earlier step's result rather
# than a literal: ``{"$from_step": 1, "path": "summary.lines[*].product_id"}``.
REFERENCE_KEY = "$from_step"

# A checkpoint is ``<where>: <observable condition>`` with ``<where>`` one of ``final_result``,
# ``tool_result[<tool>]`` or ``transcript`` (LOCAL_MODELS.md, "Checkpoints need a shape"), or
# ``report: <observer>.<condition> is true|false`` over an informant report (DESIGN §2b).
CHECKPOINT_PATTERN = re.compile(
    r"^(?:(final_result|tool_result\[[a-z_]+\]|transcript):\s*\S"
    r"|report:\s*[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\s+is\s+(true|false)\s*$)"
)
CHECKPOINT_SHAPE = (
    "'<where>: <observable condition>' where <where> is final_result, "
    "tool_result[<tool_name>] or transcript, or 'report: <observer>.<condition> is true|false'"
)


class StepReference(BaseModel):
    """``{"$from_step": n, "path": "..."}``: take a value from step ``n``'s structured result.

    ``from_step`` is the 1-based index of an earlier step in the same path; ``path`` is a dotted
    path into that step's structured result, with one ``[*]`` allowed (every element) exactly as
    in :mod:`mcpsim.matcher`. The executor resolves it at run time; the planner never has to
    invent identifiers it cannot know.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_step: int = Field(alias="$from_step", ge=1)  # == REFERENCE_KEY (mypy wants a literal)
    path: str = Field(min_length=1)

    def describe(self) -> str:
        return f"<from step {self.from_step}: {self.path}>"


def parse_reference(value: Any) -> StepReference | None:
    """The :class:`StepReference` a sketch value encodes, or ``None`` for a literal.

    Raises :class:`ValueError` when the value carries ``$from_step`` but is not a well-formed
    reference, so a half-written reference is never mistaken for a literal object.
    """
    if not isinstance(value, dict) or REFERENCE_KEY not in value:
        return None
    try:
        return StepReference.model_validate(value)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'reference'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ValueError(
            f"malformed step reference {json.dumps(value, sort_keys=True)}: {details}; "
            f'the shape is {{"{REFERENCE_KEY}": <1-based step index>, "path": "<dotted path>"}}'
        ) from None


class Step(BaseModel):
    """One intended move: what the agent is trying to do, with which tool, roughly how.

    ``expect_error`` marks a step that deliberately sends input the server rejects (the failure
    a recovery path must contain); an error result on such a step is the intended outcome.
    """

    model_config = ConfigDict(extra="forbid")

    intent: str
    tool: str | None = None
    arguments_sketch: dict[str, Any] = Field(default_factory=dict)
    success_looks_like: str = ""
    expect_error: bool = False

    def references(self) -> dict[str, StepReference]:
        """Argument name → reference, for every sketch value that is a well-formed reference."""
        found: dict[str, StepReference] = {}
        for key, value in self.arguments_sketch.items():
            try:
                ref = parse_reference(value)
            except ValueError:
                continue
            if ref is not None:
                found[key] = ref
        return found


class Path(BaseModel):
    """One way through the server to the goal."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    kind: PathKind
    title: str
    rationale: str = ""
    steps: list[Step] = Field(default_factory=list)
    checkpoints: list[str] = Field(default_factory=list)

    def tools_used(self) -> list[str]:
        """Distinct tool names referenced by this path's steps, in first-use order."""
        seen: list[str] = []
        for step in self.steps:
            if step.tool is not None and step.tool not in seen:
                seen.append(step.tool)
        return seen

    def expects_error(self) -> bool:
        """Does any step deliberately provoke a server error?"""
        return any(step.expect_error for step in self.steps)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str
    catalog_digest: str
    paths: list[Path] = Field(default_factory=list)

    def path(self, path_id: str) -> Path:
        for p in self.paths:
            if p.id == path_id:
                return p
        raise KeyError(f"no path with id {path_id!r} in plan for {self.scenario!r}")

    def tools_used(self) -> list[str]:
        seen: list[str] = []
        for p in self.paths:
            for name in p.tools_used():
                if name not in seen:
                    seen.append(name)
        return seen

    def save(self, path: str | FsPath) -> FsPath:
        fs_path = FsPath(path)
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        fs_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return fs_path

    @classmethod
    def load(cls, path: str | FsPath) -> ExecutionPlan:
        data = json.loads(FsPath(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
