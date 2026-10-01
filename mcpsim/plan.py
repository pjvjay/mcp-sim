"""Execution plan model (DESIGN §1 "Execution plan", "Path").

The planner emits an :class:`ExecutionPlan`; the executor runs each :class:`Path`; the judge
reads each path's ``checkpoints``. Plans are saved as ``plan.json`` and are meant to be
reviewed, edited and re-run.
"""

from __future__ import annotations

import json
from pathlib import Path as FsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PathKind = Literal["happy", "recovery", "alternative", "boundary", "policy"]
PATH_KINDS: tuple[PathKind, ...] = ("happy", "recovery", "alternative", "boundary", "policy")

Mode = Literal["guided", "free"]
MODES: tuple[Mode, ...] = ("guided", "free")


class Step(BaseModel):
    """One intended move: what the agent is trying to do, with which tool, roughly how."""

    model_config = ConfigDict(extra="forbid")

    intent: str
    tool: str | None = None
    arguments_sketch: dict[str, Any] = Field(default_factory=dict)
    success_looks_like: str = ""


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
