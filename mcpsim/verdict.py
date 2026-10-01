"""Judge output models (DESIGN §1 "Verdict", §2 "Judge").

Two layers are kept apart: deterministic :class:`Match` results from the matcher and the
LLM judge's :class:`ChecklistItem` list. Any failed match forces ``passed = False``.
"""

from __future__ import annotations

import json
from pathlib import Path as FsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Match(BaseModel):
    """One deterministic comparison from the matcher."""

    model_config = ConfigDict(extra="forbid")

    path: str
    op: str
    expected: Any = None
    actual: Any = None
    passed: bool
    detail: str = ""


class ChecklistItem(BaseModel):
    """One LLM-judged item with a verbatim quote (or "no evidence") as evidence."""

    model_config = ConfigDict(extra="forbid")

    item: str
    passed: bool
    evidence: str = ""


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path_id: str
    mode: str
    index: int
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    matches: list[Match] = Field(default_factory=list)
    checklist: list[ChecklistItem] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    votes: int = Field(ge=0)
    judge_model: str

    @property
    def matcher_passed(self) -> bool:
        return all(m.passed for m in self.matches)

    def save(self, path: str | FsPath) -> FsPath:
        fs_path = FsPath(path)
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        fs_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return fs_path

    @classmethod
    def load(cls, path: str | FsPath) -> Verdict:
        return cls.model_validate(json.loads(FsPath(path).read_text(encoding="utf-8")))
