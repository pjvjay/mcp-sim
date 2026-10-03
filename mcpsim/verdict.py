"""Judge output models (DESIGN §1 "Verdict", §2 "Judge").

Two layers are kept apart: deterministic :class:`Match` results from the matcher and the
LLM judge's :class:`ChecklistItem` list (one item per expected behaviour, then honesty). Any
failed match forces ``passed = False``. ``goal_achieved`` and ``sop_followed`` are the judge's
majority on the goal and on the agent's standard operating procedure; ``None`` when nobody
graded them (dry run) or, for ``sop_followed``, when the scenario gives the agent no SOP.
"""

from __future__ import annotations

import json
from pathlib import Path as FsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mcpsim.llm import Usage


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
    # Observer ``flag`` effects on the run (DESIGN §2b); a flag that did not ``fail`` is kept
    # here and joins ``failure_reasons`` only when the LLM votes fail.
    flags: list[str] = Field(default_factory=list)
    votes: int = Field(ge=0)
    judge_model: str
    goal_achieved: bool | None = None
    sop_followed: bool | None = None
    # What the judge's votes cost (estimate from the rate table; empty in a dry run).
    judge_usage: dict[str, Usage] = Field(default_factory=dict)
    judge_cost_usd: float = Field(default=0.0, ge=0.0)

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
