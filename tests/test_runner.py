"""Runner tests (DESIGN §8): the whole pipeline on disk, in dry run, against the fake server.

Every test here is synchronous on purpose: the runner's public functions wrap ``asyncio.run``
themselves, so they must be called from outside an event loop, exactly as the CLI does.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path as FsPath
from typing import Any

import pytest
import yaml

from mcpsim import runner
from mcpsim.judge import DRY_RUN_JUDGE_MODEL, VERDICT_TOOL
from mcpsim.llm import AnthropicLLM
from mcpsim.mcpclient import MCPClientError, Session
from mcpsim.plan import ExecutionPlan, Path, Step
from mcpsim.planner import DRY_RUN_PATH_ID
from mcpsim.report import Report, SuiteReport
from mcpsim.scenario import Scenario, ServerSpec, parse_scenario
from mcpsim.scout import ScoutResult
from mcpsim.transcript import (
    AssistantEvent,
    EndEvent,
    FinalResultEvent,
    SystemEvent,
    ToolCallEvent,
    ToolResultEvent,
    Transcript,
    UsageEvent,
    UserEvent,
)
from mcpsim.verdict import Verdict
from tests.fake_llm import ScriptedLLM, structured_response

PENNE: dict[str, Any] = {
    "slug": "penne",
    "price": 2.49,
    "store": "Fake Mart",
    "origin_status": "verified",
}


@pytest.fixture
def quick_data(scenario_data: dict[str, Any]) -> dict[str, Any]:
    """The conftest scenario with ``repeat: 1`` so a dry run spawns two subprocesses, not six."""
    return {**scenario_data, "repeat": 1}


@pytest.fixture
def quick_path(tmp_path: FsPath, quick_data: dict[str, Any]) -> FsPath:
    path = tmp_path / "fake-lookup.yaml"
    path.write_text(yaml.safe_dump(quick_data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def out_dir(tmp_path: FsPath) -> FsPath:
    return tmp_path / "runs"


def _stems(run_dir: FsPath, folder: str, suffix: str) -> list[str]:
    return sorted(p.stem for p in (run_dir / folder).glob(f"*{suffix}"))


# --- run_scenario in dry run ------------------------------------------------------------------


def test_dry_run_writes_every_artefact(quick_path: FsPath, out_dir: FsPath) -> None:
    run_dir = runner.run_scenario(quick_path, out_dir, dry_run=True)

    assert run_dir.parent == out_dir / "fake-lookup"
    assert (run_dir / "scenario.json").is_file()
    assert (run_dir / "plan.json").is_file()
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "report.md").is_file()
    stems = [f"{DRY_RUN_PATH_ID}-free-0", f"{DRY_RUN_PATH_ID}-guided-0"]
    assert _stems(run_dir, "transcripts", ".jsonl") == stems
    assert _stems(run_dir, "verdicts", ".json") == stems

    # scenario.json is the validated scenario, round-trippable.
    saved = Scenario.model_validate_json((run_dir / "scenario.json").read_text(encoding="utf-8"))
    assert saved.name == "fake-lookup" and saved.repeat == 1

    plan = ExecutionPlan.load(run_dir / "plan.json")
    assert plan.scenario == "fake-lookup"
    assert [p.id for p in plan.paths] == [DRY_RUN_PATH_ID]
    # The dry run plans the one goal-relevant tool with the expected outcome's slug.
    assert [(s.tool, s.arguments_sketch) for s in plan.paths[0].steps] == [
        ("lookup", {"slug": "penne"})
    ]

    # The transcript's tool_result events came from the real (subprocess) server.
    transcript = Transcript.read_jsonl(run_dir / "transcripts" / f"{stems[1]}.jsonl")
    assert transcript.outcome == "completed"
    assert transcript.kinds()[:3] == ["system", "tools_offered", "user"]
    assert transcript.tools_offered() == ["lookup"]
    results = transcript.tool_results()
    assert [(r.name, r.is_error) for r in results] == [("lookup", False)]
    assert results[0].structured == PENNE
    assert transcript.final_result == PENNE

    # Verdicts come from the matcher alone, and a right answer passes it.
    verdict = Verdict.load(run_dir / "verdicts" / f"{stems[1]}.json")
    assert verdict.judge_model == DRY_RUN_JUDGE_MODEL
    assert verdict.votes == 0
    assert verdict.passed is True
    assert verdict.checklist == []
    assert verdict.failure_reasons == []
    assert {m.path for m in verdict.matches} == {"slug", "price", "origin_status"}
    assert all(m.passed for m in verdict.matches)

    report = Report.load(run_dir / "report.json")
    assert (report.scenario, report.runs, report.passed) == ("fake-lookup", 2, 2)
    assert report.judge_models == [DRY_RUN_JUDGE_MODEL]
    assert report.run_dir == str(run_dir)
    assert sorted((c.path_id, c.mode) for c in report.cells) == [
        (DRY_RUN_PATH_ID, "free"),
        (DRY_RUN_PATH_ID, "guided"),
    ]
    md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# mcp-sim report: fake-lookup\n")
    assert f"| {DRY_RUN_PATH_ID} | guided | 1 | 1 |" in md
    assert "deterministic:" not in md


def test_two_runs_in_the_same_second_get_distinct_directories(
    quick_path: FsPath, out_dir: FsPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "timestamp", lambda: "20260101T000000Z")
    first = runner.run_scenario(quick_path, out_dir, dry_run=True, repeat=1, mode="guided")
    second = runner.run_scenario(quick_path, out_dir, dry_run=True, repeat=1, mode="guided")
    assert first.name == "20260101T000000Z"
    assert second.name == "20260101T000000Z-1"
    assert first != second and (second / "report.json").is_file()


def test_dry_run_builds_no_llm(
    quick_path: FsPath, out_dir: FsPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(scenario: Scenario, role: str) -> Any:
        raise AssertionError(f"make_llm_for({role}) must not be called in dry run")

    monkeypatch.setattr(runner, "make_llm_for", boom)
    monkeypatch.delenv(runner.API_KEY_ENV, raising=False)
    run_dir = runner.run_scenario(quick_path, out_dir, dry_run=True, repeat=1, mode="guided")
    assert (run_dir / "report.json").is_file()


# --- judge_run_dir / report_run_dir rebuild from disk --------------------------------------------


def test_judge_run_dir_rebuilds_verdicts_and_report(quick_path: FsPath, out_dir: FsPath) -> None:
    run_dir = runner.run_scenario(quick_path, out_dir, dry_run=True)
    originals = {p.name: p.read_text(encoding="utf-8") for p in (run_dir / "verdicts").iterdir()}
    for p in (run_dir / "verdicts").iterdir():
        p.unlink()
    (run_dir / "verdicts").rmdir()
    (run_dir / "report.json").unlink()
    (run_dir / "report.md").unlink()

    written = runner.judge_run_dir(run_dir)

    assert sorted(p.name for p in written) == sorted(originals)
    for p in written:
        assert p.is_file()
        assert Verdict.load(p).judge_model == DRY_RUN_JUDGE_MODEL
        assert p.read_text(encoding="utf-8") == originals[p.name]
    report = Report.load(run_dir / "report.json")
    assert report.runs == 2 and (run_dir / "report.md").is_file()


def test_report_run_dir_rebuilds_from_verdicts_and_transcripts(
    quick_path: FsPath, out_dir: FsPath
) -> None:
    run_dir = runner.run_scenario(quick_path, out_dir, dry_run=True)
    before = Report.load(run_dir / "report.json")
    (run_dir / "report.json").unlink()
    (run_dir / "report.md").unlink()

    json_path, md_path = runner.report_run_dir(run_dir)

    assert (json_path, md_path) == (run_dir / "report.json", run_dir / "report.md")
    after = Report.load(json_path)
    assert after.model_dump(exclude={"generated_at"}) == before.model_dump(exclude={"generated_at"})
    assert md_path.read_text(encoding="utf-8").startswith("# mcp-sim report: fake-lookup\n")


def test_report_run_dir_lists_unjudged_transcripts(quick_path: FsPath, out_dir: FsPath) -> None:
    run_dir = runner.run_scenario(quick_path, out_dir, dry_run=True, repeat=1, mode="guided")
    (run_dir / "verdicts" / f"{DRY_RUN_PATH_ID}-guided-0.json").unlink()
    json_path, _ = runner.report_run_dir(run_dir)
    report = Report.load(json_path)
    assert report.runs == 0
    assert report.unjudged == [f"{DRY_RUN_PATH_ID}-guided-0"]


def test_judge_and_report_refuse_a_directory_that_is_not_a_run(tmp_path: FsPath) -> None:
    with pytest.raises(FileNotFoundError, match="scenario.json"):
        runner.judge_run_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="scenario.json"):
        runner.report_run_dir(tmp_path)


def _vote(passed: bool, score: float, instruction_count: int) -> Any:
    ok = {"passed": True, "evidence": '[5] "price": 2.49'}
    bad = {"passed": False, "evidence": "no evidence"}
    item = ok if passed else bad
    return structured_response(
        VERDICT_TOOL,
        {
            "goal": item,
            "instructions": [item] * instruction_count,
            "honesty": item,
            "recovery": ok,
            "efficiency": ok,
            "scope": ok,
            "passed": passed,
            "score": score,
            "failure_reasons": [] if passed else ["goal missed"],
        },
    )


def _real_transcript(scenario: Scenario) -> Transcript:
    """A transcript as a real (non dry-run) agent would have left it."""
    events: list[Any] = [
        SystemEvent(
            scenario=scenario.name,
            path_id="happy",
            index=0,
            mode="guided",
            models={"agent": scenario.models.agent, "user": scenario.models.agent},
        ),
        UserEvent(text="How much is penne and where?"),
        AssistantEvent(
            text="Looking it up.",
            tool_uses=[
                {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"slug": "penne"}}
            ],
            stop_reason="tool_use",
        ),
        ToolCallEvent(name="lookup", arguments={"slug": "penne"}, tool_use_id="t1"),
        ToolResultEvent(name="lookup", is_error=False, structured=dict(PENNE), tool_use_id="t1"),
        AssistantEvent(text="Penne is 2.49 at Fake Mart.\n```json final_result\n{}\n```"),
        FinalResultEvent(parsed=dict(PENNE), raw="{}"),
        UsageEvent(per_model={}, cost_usd=0.0),
        EndEvent(outcome="completed", reason="final answer delivered"),
    ]
    return Transcript.from_events(events)


def test_judge_run_dir_uses_the_llm_judge_for_real_transcripts(
    quick_data: dict[str, Any], tmp_path: FsPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-built run directory with a real transcript is judged through make_llm_for."""
    scenario = parse_scenario(quick_data)
    run_dir = tmp_path / "runs" / scenario.name / "20260101T000000Z"
    run_dir.mkdir(parents=True)
    (run_dir / "scenario.json").write_text(scenario.model_dump_json(indent=2), encoding="utf-8")
    plan = ExecutionPlan(
        scenario=scenario.name,
        catalog_digest="x",
        paths=[
            Path(
                id="happy",
                kind="happy",
                title="Look up penne",
                steps=[
                    Step(intent="look up penne", tool="lookup", arguments_sketch={"slug": "penne"})
                ],
            )
        ],
    )
    plan.save(run_dir / "plan.json")
    transcript = _real_transcript(scenario)
    transcript.write_jsonl(run_dir / "transcripts" / f"{transcript.stem}.jsonl")

    n = len(scenario.instructions)
    llm = ScriptedLLM([_vote(True, 0.9, n), _vote(True, 0.8, n), _vote(False, 0.4, n)])
    roles: list[str] = []

    def fake_make_llm_for(s: Scenario, role: str) -> ScriptedLLM:
        roles.append(role)
        return llm

    monkeypatch.setattr(runner, "make_llm_for", fake_make_llm_for)
    written = runner.judge_run_dir(run_dir, votes=3)

    assert roles == ["judge"]
    assert [p.name for p in written] == ["happy-guided-0.json"]
    verdict = Verdict.load(written[0])
    assert verdict.judge_model == scenario.models.judge
    assert verdict.votes == 3
    assert verdict.passed is True and verdict.score == pytest.approx(0.7)
    assert [m.passed for m in verdict.matches] == [True, True, True]
    # goal, n instructions, honesty, recovery, efficiency, scope
    assert len(verdict.checklist) == 5 + n
    assert len(llm.calls) == 3 and all(c["model"] == scenario.models.judge for c in llm.calls)
    report = Report.load(run_dir / "report.json")
    assert (report.runs, report.passed, report.judge_models) == (1, 1, [scenario.models.judge])


def test_judge_run_dir_rejects_zero_votes(quick_path: FsPath, out_dir: FsPath) -> None:
    run_dir = runner.run_scenario(quick_path, out_dir, dry_run=True, repeat=1, mode="guided")
    with pytest.raises(ValueError, match="votes"):
        runner.judge_run_dir(run_dir, votes=0)


# --- tool scoping: the allowed catalog is what the plan, the runs and the digest see ----------


def test_deny_globs_scope_the_plan_the_runs_and_the_digest(
    tmp_path: FsPath,
    quick_data: dict[str, Any],
    out_dir: FsPath,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quick_data["tools"] = {"deny": ["fail", "expensive_*", "nope_*"]}
    path = tmp_path / "scoped.yaml"
    path.write_text(yaml.safe_dump(quick_data, sort_keys=False), encoding="utf-8")

    run_dir = runner.run_scenario(path, out_dir, dry_run=True, repeat=1, mode="guided")

    plan = ExecutionPlan.load(run_dir / "plan.json")
    assert plan.paths[0].tools_used() == ["lookup"]
    full = asyncio.run(_fake_catalog())
    assert plan.catalog_digest == full.filtered(["*"], ["fail", "expensive_*"]).digest()
    assert plan.catalog_digest != full.digest()
    stem = f"{DRY_RUN_PATH_ID}-guided-0"
    transcript = Transcript.read_jsonl(run_dir / "transcripts" / f"{stem}.jsonl")
    assert [c.name for c in transcript.tool_calls()] == ["lookup"]
    err = capsys.readouterr().err
    assert "warning: tools.deny glob 'nope_*' matches no tool of this server" in err
    assert "glob 'fail'" not in err and "glob 'expensive_*'" not in err


async def _fake_catalog() -> Any:
    from tests.conftest import open_session

    async with open_session() as session:
        return await session.catalog()


# --- scout: read-only observations before planning, saved beside the plan --------------------


def test_progressive_scenario_is_scouted_and_scout_json_feeds_the_plan(
    tmp_path: FsPath,
    quick_data: dict[str, Any],
    out_dir: FsPath,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quick_data["tools"] = {"disclosure": "progressive", "initial": ["lookup", "fail"]}
    quick_data["observers"] = [
        {
            "name": "shelf_auditor",
            "identity": "An independent auditor.",
            "kind": "code",
            "watches": ["scout"],
            "on": ["scout"],
            "conditions": [
                {
                    "id": "direct_match",
                    "when": "lookup returned penne",
                    "check": {"tool_result": {"tool": "lookup", "where": {"slug": "penne"}}},
                    "then": {"enable_tools": ["echo"]},
                }
            ],
        }
    ]
    path = tmp_path / "scouted.yaml"
    path.write_text(yaml.safe_dump(quick_data, sort_keys=False), encoding="utf-8")

    plan_path = runner.plan_scenario(path, out_dir, dry_run=True)
    scout = ScoutResult.load(plan_path.parent / "scout.json")
    assert [o.call_label() for o in scout.observations] == [
        "fake://about",
        'lookup(slug="penne")',
        "fail()",
    ]
    assert scout.disclosed == ["lookup", "fail", "echo"]
    assert scout.on_request == ["list_items", "expensive_report"]
    assert [(r.key, r.value) for r in scout.reports] == [("shelf_auditor.direct_match", True)]
    assert scout.planner_prompt_chars > 1000
    plan = ExecutionPlan.load(plan_path)
    assert plan.paths[0].steps[0].arguments_sketch == {"slug": "penne"}
    assert plan.paths[0].checkpoints[-1] == "report: shelf_auditor.direct_match is true"
    err = capsys.readouterr().err
    assert (
        "fake-lookup: scouted 3 observation(s) (2 tool call(s) of 10); 3 tool(s) disclosed, "
        "2 on request; 1 informant report(s)"
    ) in err

    run_dir = runner.run_scenario(path, out_dir, dry_run=True, repeat=1, mode="guided")
    assert (run_dir / "scout.json").is_file()
    assert ScoutResult.load(run_dir / "scout.json").disclosed == scout.disclosed

    reused = runner.run_scenario(
        path, out_dir, plan_path=plan_path, dry_run=True, repeat=1, mode="guided"
    )
    assert not (reused / "scout.json").exists(), "a reused plan skips the scout"


def test_all_disclosure_is_not_scouted(quick_path: FsPath, out_dir: FsPath) -> None:
    plan_path = runner.plan_scenario(quick_path, out_dir, dry_run=True)
    assert not (plan_path.parent / "scout.json").exists()
    run_dir = runner.run_scenario(quick_path, out_dir, dry_run=True, repeat=1, mode="guided")
    assert not (run_dir / "scout.json").exists()


# --- observers: the runner wires them into every run and warns about dead globs --------------


def test_dry_run_records_code_observer_reports_and_warns_about_unmatched_effect_globs(
    tmp_path: FsPath,
    quick_data: dict[str, Any],
    out_dir: FsPath,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quick_data["observers"] = [
        {
            "name": "clerk",
            "identity": "reads the lookup result",
            "kind": "code",
            "on": ["tool_result"],
            "conditions": [
                {
                    "id": "found",
                    "when": "lookup returned penne",
                    "check": {"tool_result": {"tool": "lookup", "where": {"slug": "penne"}}},
                    "then": {"enable_tools": ["echo", "nope_*"], "flag": "found"},
                    "otherwise": {"disable_tools": ["zzz_*"]},
                }
            ],
        },
        {
            "name": "auditor",
            "identity": "needs a model, so the dry run skips it",
            "on": ["end"],
            "conditions": [{"id": "x", "when": "x", "then": {"fail": True}}],
        },
    ]
    path = tmp_path / "observed.yaml"
    path.write_text(yaml.safe_dump(quick_data, sort_keys=False), encoding="utf-8")

    run_dir = runner.run_scenario(path, out_dir, dry_run=True, repeat=1, mode="guided")

    stem = f"{DRY_RUN_PATH_ID}-guided-0"
    transcript = Transcript.read_jsonl(run_dir / "transcripts" / f"{stem}.jsonl")
    assert transcript.kinds()[:7] == [
        "system",
        "tools_offered",
        "user",
        "tool_call",
        "tool_result",
        "informant_report",
        "tools_offered",
    ]
    reports = transcript.informant_reports()
    assert [(r.observer, r.condition, r.value, r.trigger) for r in reports] == [
        ("clerk", "found", True, "tool_result")
    ], "the LLM auditor is not consulted in dry run"
    assert transcript.flags == ["found"] and transcript.hard_failures == []
    offered = [e for e in transcript.events if e.kind == "tools_offered"]
    assert (offered[1].added, offered[1].reason) == (["echo"], "observer:clerk.found")  # type: ignore[union-attr]
    err = capsys.readouterr().err
    assert (
        "warning: observer clerk.found then.enable_tools glob 'nope_*' matches no allowed tool"
    ) in err
    assert (
        "warning: observer clerk.found otherwise.disable_tools glob 'zzz_*' matches no allowed"
    ) in err
    assert "glob 'echo'" not in err
    verdict = Verdict.load(run_dir / "verdicts" / f"{stem}.json")
    assert verdict.passed is True, "lookup's result covers the expected keys; the matcher passes"


def test_reused_plan_is_compared_against_the_allowed_catalog(
    tmp_path: FsPath,
    quick_data: dict[str, Any],
    out_dir: FsPath,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unscoped = tmp_path / "unscoped.yaml"
    unscoped.write_text(yaml.safe_dump(quick_data, sort_keys=False), encoding="utf-8")
    plan_path = runner.plan_scenario(unscoped, out_dir, dry_run=True)
    scoped_data = {**quick_data, "tools": {"deny": ["fail"]}}
    scoped = tmp_path / "scoped.yaml"
    scoped.write_text(yaml.safe_dump(scoped_data, sort_keys=False), encoding="utf-8")

    runner.run_scenario(scoped, out_dir, plan_path=plan_path, dry_run=True, repeat=1, mode="free")
    err = capsys.readouterr().err
    assert "warning: the allowed catalog changed since plan" in err
    assert "the scenario's tools policy" in err

    capsys.readouterr()
    runner.run_scenario(unscoped, out_dir, plan_path=plan_path, dry_run=True, repeat=1, mode="free")
    assert "allowed catalog changed" not in capsys.readouterr().err


# --- narrowing: plan reuse, only_path, repeat, mode -------------------------------------------


def test_plan_scenario_writes_plan_and_scenario_json(quick_path: FsPath, out_dir: FsPath) -> None:
    plan_path = runner.plan_scenario(quick_path, out_dir, dry_run=True)
    assert plan_path.name == "plan.json"
    assert plan_path.parent.parent == out_dir / "fake-lookup"
    assert (plan_path.parent / "scenario.json").is_file()
    assert not (plan_path.parent / "transcripts").exists()
    plan = ExecutionPlan.load(plan_path)
    assert plan.paths[0].id == DRY_RUN_PATH_ID
    assert plan.paths[0].tools_used() == ["lookup"]
    assert len(plan.catalog_digest) == 64


def test_only_path_repeat_and_mode_narrow_the_runs(quick_path: FsPath, out_dir: FsPath) -> None:
    plan_path = runner.plan_scenario(quick_path, out_dir, dry_run=True)

    run_dir = runner.run_scenario(
        quick_path,
        out_dir,
        plan_path=plan_path,
        only_path=DRY_RUN_PATH_ID,
        repeat=1,
        mode="guided",
        dry_run=True,
    )
    assert _stems(run_dir, "transcripts", ".jsonl") == [f"{DRY_RUN_PATH_ID}-guided-0"]
    assert _stems(run_dir, "verdicts", ".json") == [f"{DRY_RUN_PATH_ID}-guided-0"]
    # The reused plan is copied into the new run directory unchanged.
    assert json.loads((run_dir / "plan.json").read_text()) == json.loads(plan_path.read_text())

    run_dir = runner.run_scenario(
        quick_path, out_dir, plan_path=plan_path, repeat=2, mode="free", dry_run=True
    )
    assert _stems(run_dir, "transcripts", ".jsonl") == [
        f"{DRY_RUN_PATH_ID}-free-0",
        f"{DRY_RUN_PATH_ID}-free-1",
    ]
    assert Report.load(run_dir / "report.json").runs == 2


def test_unknown_only_path_raises_value_error(quick_path: FsPath, out_dir: FsPath) -> None:
    plan_path = runner.plan_scenario(quick_path, out_dir, dry_run=True)
    with pytest.raises(ValueError, match=r"no path with id 'nope'.*happy-dry-run"):
        runner.run_scenario(
            quick_path, out_dir, plan_path=plan_path, only_path="nope", dry_run=True
        )


def test_missing_plan_file_raises_before_anything_runs(quick_path: FsPath, out_dir: FsPath) -> None:
    with pytest.raises(FileNotFoundError, match="plan file not found"):
        runner.run_scenario(quick_path, out_dir, plan_path=out_dir / "nope.json", dry_run=True)
    assert not out_dir.exists()


@pytest.mark.parametrize("repeat", [0, -1])
def test_repeat_below_one_is_rejected(quick_path: FsPath, out_dir: FsPath, repeat: int) -> None:
    with pytest.raises(ValueError, match="repeat"):
        runner.run_scenario(quick_path, out_dir, repeat=repeat, dry_run=True)


def test_run_matrix_runs_free_mode_only_for_the_happy_path() -> None:
    plan = ExecutionPlan(
        scenario="s",
        catalog_digest="d",
        paths=[
            Path(id="recovery-1", kind="recovery", title="r"),
            Path(id="happy", kind="happy", title="h"),
        ],
    )
    cells = [
        (p.id, m, i) for p, m, i in runner.run_matrix(plan, only_path=None, repeat=2, mode=None)
    ]
    assert cells == [
        ("recovery-1", "guided", 0),
        ("recovery-1", "guided", 1),
        ("happy", "guided", 0),
        ("happy", "guided", 1),
        ("happy", "free", 0),
        ("happy", "free", 1),
    ]
    free = runner.run_matrix(plan, only_path=None, repeat=1, mode="free")
    assert [(p.id, m, i) for p, m, i in free] == [("happy", "free", 0)]
    only = runner.run_matrix(plan, only_path="recovery-1", repeat=1, mode=None)
    assert [(p.id, m, i) for p, m, i in only] == [("recovery-1", "guided", 0)]
    with pytest.raises(ValueError, match="mode"):
        runner.run_matrix(plan, only_path=None, repeat=1, mode="sideways")  # type: ignore[arg-type]


# --- run_suite --------------------------------------------------------------------------------


def _suite_dir(tmp_path: FsPath, quick_data: dict[str, Any]) -> FsPath:
    folder = tmp_path / "scenarios"
    folder.mkdir()
    for name in ("fake-b", "fake-a"):
        data = {**quick_data, "name": name}
        if name == "fake-b":
            # fake-b expects an origin_status the server never returns, so its dry run fails
            # the matcher while fake-a's (lookup(slug="penne")) passes it.
            data["expected_outcome"] = {"json": {"slug": "penne", "origin_status": "unverified"}}
        (folder / f"{name}.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
    (folder / "notes.txt").write_text("not a scenario", encoding="utf-8")
    return folder


def test_run_suite_returns_the_threshold_exit_code(
    tmp_path: FsPath,
    quick_data: dict[str, Any],
    out_dir: FsPath,
    capsys: pytest.CaptureFixture[str],
) -> None:
    folder = _suite_dir(tmp_path, quick_data)

    assert runner.run_suite(folder, out_dir, threshold=1.0, dry_run=True) == 1
    out = capsys.readouterr().out
    assert out.index("fake-a: 2/2 runs passed") < out.index("fake-b: 0/2 runs passed")
    assert "suite: 2/4 runs passed (50.0%)" in out

    suite_dirs = sorted(out_dir.glob("suite-*"))
    assert len(suite_dirs) == 1
    suite = SuiteReport.load(suite_dirs[0] / "suite.json")
    assert [s.scenario for s in suite.scenarios] == ["fake-a", "fake-b"]
    assert (suite.runs, suite.passed, suite.pass_rate) == (4, 2, 0.5)
    assert all(FsPath(s.run_dir).is_dir() for s in suite.scenarios)
    md = (suite_dirs[0] / "suite.md").read_text(encoding="utf-8")
    assert md.startswith("# mcp-sim suite report\n") and "| fake-a | 2 | 2 |" in md
    assert "| fake-b | 2 | 0 |" in md
    assert sorted(p.name for p in out_dir.iterdir() if not p.name.startswith("suite-")) == [
        "fake-a",
        "fake-b",
    ]

    # Half the runs pass, so a threshold at or below the pass rate yields exit 0.
    assert runner.run_suite(folder, out_dir, threshold=0.5, dry_run=True) == 0
    assert len(sorted(out_dir.glob("suite-*"))) == 2


def test_run_suite_rejects_bad_inputs(tmp_path: FsPath, out_dir: FsPath) -> None:
    with pytest.raises(FileNotFoundError, match="scenario directory"):
        runner.run_suite(tmp_path / "missing", out_dir, dry_run=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no scenario files"):
        runner.run_suite(empty, out_dir, dry_run=True)
    with pytest.raises(ValueError, match="threshold"):
        runner.run_suite(empty, out_dir, threshold=1.5, dry_run=True)


def test_scenario_files_are_filtered_and_sorted(tmp_path: FsPath) -> None:
    for name in ("b.yaml", "a.json", "c.yml", "notes.txt", "d.YAML"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "sub.yaml").mkdir()
    assert [p.name for p in runner.scenario_files(tmp_path)] == [
        "a.json",
        "b.yaml",
        "c.yml",
        "d.YAML",
    ]


# --- setup command and LLM construction -------------------------------------------------------


def test_setup_runs_once_per_run_scenario_with_the_stdio_env(
    quick_data: dict[str, Any], tmp_path: FsPath, out_dir: FsPath
) -> None:
    marker = tmp_path / "setup.log"
    script = tmp_path / "setup.py"
    script.write_text(
        "import os, pathlib\n"
        "p = pathlib.Path(os.environ['MCPSIM_TEST_SETUP_OUT'])\n"
        "p.open('a').write(os.environ['MCPSIM_TEST_MARK'] + '\\n')\n",
        encoding="utf-8",
    )
    stdio = quick_data["server"]["stdio"]
    stdio["setup"] = f"{sys.executable} {script}"
    stdio["env"] = {
        **stdio["env"],
        "MCPSIM_TEST_SETUP_OUT": str(marker),
        "MCPSIM_TEST_MARK": "seeded",
    }
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(quick_data, sort_keys=False), encoding="utf-8")

    runner.run_scenario(path, out_dir, dry_run=True)  # two runs, one setup
    assert marker.read_text(encoding="utf-8") == "seeded\n"
    runner.plan_scenario(path, out_dir, dry_run=True)  # planning connects too, so it seeds too
    assert marker.read_text(encoding="utf-8") == "seeded\nseeded\n"


def test_failing_setup_raises_a_runtime_error(
    quick_data: dict[str, Any], tmp_path: FsPath, out_dir: FsPath
) -> None:
    quick_data["server"]["stdio"]["setup"] = f"{sys.executable} -c 'raise SystemExit(3)'"
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(quick_data, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="setup command failed with exit 3"):
        runner.run_scenario(path, out_dir, dry_run=True)
    assert not out_dir.exists()

    quick_data["server"]["stdio"]["setup"] = "/no/such/seed-command --now"
    path.write_text(yaml.safe_dump(quick_data, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="setup command not found"):
        runner.run_scenario(path, out_dir, dry_run=True)


def test_http_server_has_no_setup(quick_data: dict[str, Any]) -> None:
    quick_data["server"] = {"http": {"url": "http://127.0.0.1:9/mcp"}}
    runner.run_setup(parse_scenario(quick_data))  # no-op, no error


def test_make_llm_for_needs_an_api_key(
    quick_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = parse_scenario(quick_data)
    monkeypatch.delenv(runner.API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=f"{runner.API_KEY_ENV} is not set.*dry-run"):
        runner.make_llm_for(scenario, "planner")
    monkeypatch.setenv(runner.API_KEY_ENV, "not-a-real-key")
    assert isinstance(runner.make_llm_for(scenario, "agent"), AnthropicLLM)


def test_real_run_without_a_key_fails_before_any_run(
    quick_path: FsPath, out_dir: FsPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(runner.API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=runner.API_KEY_ENV):
        runner.run_scenario(quick_path, out_dir, dry_run=False)
    run_dirs = list((out_dir / "fake-lookup").iterdir())
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "transcripts").exists()


def test_same_judge_and_agent_model_is_refused_up_front(
    quick_data: dict[str, Any], tmp_path: FsPath, out_dir: FsPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    quick_data["models"] = {"agent": "claude-opus-5-5", "judge": "claude-opus-5-5"}
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(quick_data, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv(runner.API_KEY_ENV, "not-a-real-key")
    with pytest.raises(ValueError, match="same as the agent model"):
        runner.run_scenario(path, out_dir, dry_run=False)


# --- failures that must not crash a run -------------------------------------------------------


def test_unreachable_server_is_a_user_facing_error(
    quick_data: dict[str, Any], tmp_path: FsPath, out_dir: FsPath
) -> None:
    quick_data["server"] = {"stdio": {"command": str(tmp_path / "no-such-server")}}
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(quick_data, sort_keys=False), encoding="utf-8")
    with pytest.raises(OSError, match="no-such-server"):
        runner.run_scenario(path, out_dir, dry_run=True)


def test_a_session_that_fails_to_open_becomes_an_error_run(
    quick_path: FsPath, out_dir: FsPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = runner.connect
    opened = {"n": 0}

    @asynccontextmanager
    async def flaky(spec: ServerSpec) -> AsyncIterator[Session]:
        opened["n"] += 1
        if opened["n"] > 1:  # the first connection (catalog discovery) works, the runs do not
            raise MCPClientError("server went away")
        async with real_connect(spec) as session:
            yield session

    monkeypatch.setattr(runner, "connect", flaky)
    run_dir = runner.run_scenario(quick_path, out_dir, dry_run=True)

    stems = _stems(run_dir, "transcripts", ".jsonl")
    assert len(stems) == 2
    for stem in stems:
        transcript = Transcript.read_jsonl(run_dir / "transcripts" / f"{stem}.jsonl")
        assert transcript.outcome == "error"
        assert "server went away" in transcript.reason
        assert transcript.kinds() == ["system", "end"]
        verdict = Verdict.load(run_dir / "verdicts" / f"{stem}.json")
        assert verdict.passed is False
        assert any(r.startswith("run: error") for r in verdict.failure_reasons)
    report = Report.load(run_dir / "report.json")
    assert report.outcomes == {"error": 2}
    assert report.worst_failures[0].outcome == "error"
