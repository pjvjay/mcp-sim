from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mcpsim.llm import Usage
from mcpsim.report import (
    EXIT_FAIL,
    EXIT_PASS,
    Report,
    SuiteReport,
    aggregate,
    aggregate_suite,
    exit_code,
    failure_reasons,
    pass_k,
    render_markdown,
    render_suite_markdown,
    summary_line,
)
from mcpsim.transcript import EndEvent, SystemEvent, Transcript
from mcpsim.verdict import ChecklistItem, Match, Verdict

JUDGE = "claude-opus-5-5"
T0 = datetime.fromisoformat("2026-10-03T10:00:00.000+00:00")


def _verdict(
    path_id: str,
    mode: str,
    index: int,
    *,
    passed: bool,
    score: float,
    reasons: list[str] | None = None,
    matches: list[Match] | None = None,
    checklist: list[ChecklistItem] | None = None,
    judge_model: str = JUDGE,
    goal_achieved: bool | None = None,
    sop_followed: bool | None = None,
    judge_usage: dict[str, Usage] | None = None,
    judge_cost: float = 0.0,
) -> Verdict:
    return Verdict(
        path_id=path_id,
        mode=mode,
        index=index,
        passed=passed,
        score=score,
        matches=matches or [],
        checklist=checklist or [],
        failure_reasons=reasons or [],
        votes=3,
        judge_model=judge_model,
        goal_achieved=goal_achieved,
        sop_followed=sop_followed,
        judge_usage=judge_usage or {},
        judge_cost_usd=judge_cost,
    )


def _transcript(
    path_id: str,
    mode: str,
    index: int,
    *,
    cost: float = 0.01,
    outcome: str = "completed",
    reason: str = "",
    usage: dict[str, Usage] | None = None,
    seconds: float = 0.0,
    start: datetime = T0,
) -> Transcript:
    t = Transcript(
        scenario="fake-lookup",
        path_id=path_id,
        mode=mode,
        index=index,
        outcome=outcome,  # type: ignore[arg-type]
        reason=reason,
        usage=usage or {"claude-sonnet-5-5": Usage(input_tokens=100, output_tokens=20)},
        cost_usd=cost,
    )
    end = start + timedelta(seconds=seconds)
    t.add(
        SystemEvent(
            t=start.isoformat(timespec="milliseconds"),
            scenario="fake-lookup",
            path_id=path_id,
            mode=mode,
            index=index,
        )
    )
    t.add(
        EndEvent(t=end.isoformat(timespec="milliseconds"), outcome=outcome, reason=reason)  # type: ignore[arg-type]
    )
    return t


def _five_runs() -> tuple[list[Verdict], list[Transcript]]:
    """3 happy/guided (2 pass), 1 happy/free (pass), 1 recovery/guided (fail): 3/5 overall."""
    verdicts = [
        _verdict("happy", "guided", 0, passed=True, score=1.0),
        _verdict("happy", "guided", 1, passed=True, score=0.9),
        _verdict("happy", "guided", 2, passed=False, score=0.4, reasons=["honesty: claimed clean"]),
        _verdict("happy", "free", 0, passed=True, score=0.8),
        _verdict(
            "recovery",
            "guided",
            0,
            passed=False,
            score=0.2,
            reasons=["deterministic: origin_status $eq"],
        ),
    ]
    transcripts = [
        _transcript("happy", "guided", 0, cost=0.10, seconds=10),
        _transcript("happy", "guided", 1, cost=0.20, seconds=20),
        _transcript("happy", "guided", 2, cost=0.30, seconds=30),
        _transcript(
            "happy",
            "free",
            0,
            cost=0.40,
            usage={"claude-haiku-4-5": Usage(input_tokens=5, output_tokens=5)},
            seconds=40,
        ),
        _transcript(
            "recovery",
            "guided",
            0,
            cost=0.50,
            outcome="budget_exceeded",
            reason="max_turns",
            seconds=50,
        ),
    ]
    return verdicts, transcripts


def test_aggregate_arithmetic() -> None:
    verdicts, transcripts = _five_runs()
    report = aggregate(verdicts, transcripts)

    assert report.scenario == "fake-lookup"
    assert (report.runs, report.passed, report.pass_rate) == (5, 3, 0.6)
    assert report.mean_score == pytest.approx((1.0 + 0.9 + 0.4 + 0.8 + 0.2) / 5)
    assert report.cost_usd == pytest.approx(1.5)
    assert (report.run_cost_usd, report.judge_cost_usd) == (pytest.approx(1.5), 0.0)
    assert report.judge_models == [JUDGE]
    assert report.outcomes == {"budget_exceeded": 1, "completed": 4}
    assert report.unjudged == []
    # Run time: the five runs' durations summed; the wall clock spans the concurrent runs.
    assert report.duration_s == pytest.approx(10 + 20 + 30 + 40 + 50)
    assert report.wall_clock_s == pytest.approx(50.0)
    # pass^k: k is the repeat count (3, the happy/guided cell); two cells are short of it and
    # two runs failed, so it does not hold.
    assert report.pass_k.model_dump() == {"k": 3, "all_passed": False}

    cells = {(c.path_id, c.mode): c for c in report.cells}
    assert set(cells) == {("happy", "guided"), ("happy", "free"), ("recovery", "guided")}
    hg = cells[("happy", "guided")]
    assert (hg.runs, hg.passed, hg.pass_rate) == (3, 2, 0.6667)
    assert hg.mean_score == pytest.approx(round((1.0 + 0.9 + 0.4) / 3, 4))
    assert hg.cost_usd == pytest.approx(0.6)
    assert hg.duration_s == pytest.approx(60.0)
    assert hg.outcomes == {"completed": 3}
    rg = cells[("recovery", "guided")]
    assert (rg.runs, rg.passed, rg.pass_rate, rg.cost_usd) == (1, 0, 0.0, 0.5)
    assert rg.outcomes == {"budget_exceeded": 1}

    # Token usage is merged per model across transcripts.
    assert report.usage["claude-sonnet-5-5"] == Usage(input_tokens=400, output_tokens=80)
    assert report.usage["claude-haiku-4-5"] == Usage(input_tokens=5, output_tokens=5)


def test_worst_failures_are_ordered_by_score_with_transcript_pointers() -> None:
    verdicts, transcripts = _five_runs()
    report = aggregate(verdicts, transcripts)

    stems = [f.stem for f in report.worst_failures]
    assert stems == ["recovery-guided-0", "happy-guided-2"]
    worst = report.worst_failures[0]
    assert worst.transcript == "transcripts/recovery-guided-0.jsonl"
    assert worst.verdict == "verdicts/recovery-guided-0.json"
    assert worst.outcome == "budget_exceeded"
    assert "deterministic: origin_status $eq" in worst.reasons
    assert "outcome budget_exceeded: max_turns" in worst.reasons

    absolute = aggregate(verdicts, transcripts, run_dir="/tmp/runs/x")
    assert absolute.run_dir == "/tmp/runs/x"
    assert (
        absolute.worst_failures[0].transcript == "/tmp/runs/x/transcripts/recovery-guided-0.jsonl"
    )


def test_max_failures_caps_the_list() -> None:
    verdicts = [
        _verdict("happy", "guided", i, passed=False, score=i / 10, reasons=["x"]) for i in range(6)
    ]
    report = aggregate(verdicts, max_failures=2)
    assert [f.index for f in report.worst_failures] == [0, 1]
    assert report.runs == 6 and report.passed == 0


def test_failure_reasons_are_synthesised_from_matches_and_checklist() -> None:
    v = _verdict(
        "happy",
        "guided",
        0,
        passed=False,
        score=0.3,
        matches=[
            Match(path="price", op="$gt", expected=0, actual="5", passed=False, detail="type"),
            Match(path="slug", op="$eq", expected="penne", actual="penne", passed=True),
        ],
        checklist=[
            ChecklistItem(item="goal achieved", passed=True, evidence="turn 4"),
            ChecklistItem(item="honesty", passed=False, evidence="turn 5: 'basket is clean'"),
        ],
    )
    reasons = failure_reasons(v)
    assert reasons == [
        "deterministic: price $gt expected 0, actual '5' (type)",
        "checklist: honesty: turn 5: 'basket is clean'",
    ]
    # Judge-provided reasons win over synthesis.
    assert failure_reasons(_verdict("h", "guided", 0, passed=False, score=0, reasons=["r"])) == [
        "r"
    ]
    bare = _verdict("h", "guided", 0, passed=False, score=0)
    assert failure_reasons(bare) == ["judge failed the run without a recorded reason"]


def test_unjudged_transcripts_are_listed_and_still_cost() -> None:
    verdicts = [_verdict("happy", "guided", 0, passed=True, score=1.0)]
    transcripts = [
        _transcript("happy", "guided", 0, cost=0.1),
        _transcript("happy", "guided", 1, cost=0.2),
        _transcript("boundary", "guided", 0, cost=0.3),
    ]
    report = aggregate(verdicts, transcripts)
    assert report.runs == 1 and report.passed == 1 and report.pass_rate == 1.0
    assert report.unjudged == ["boundary-guided-0", "happy-guided-1"]
    assert report.cost_usd == pytest.approx(0.6)
    assert "happy-guided-1" not in [f.stem for f in report.worst_failures]


def test_aggregate_empty() -> None:
    report = aggregate([], [])
    assert report.runs == 0 and report.pass_rate == 0.0 and report.cells == []
    assert report.worst_failures == [] and report.scenario == ""
    assert summary_line(report) == "no judged runs"
    assert exit_code(report, 0.0) == EXIT_FAIL


def test_scenario_name_override() -> None:
    report = aggregate([_verdict("happy", "guided", 0, passed=True, score=1.0)], scenario="x")
    assert report.scenario == "x"


def test_render_markdown_contains_table_and_failures() -> None:
    verdicts, transcripts = _five_runs()
    report = aggregate(verdicts, transcripts, run_dir="/runs/fake-lookup/t")
    md = render_markdown(report)

    assert md.startswith("# mcp-sim report: fake-lookup\n")
    assert (
        "**3/5 runs passed (60.0%), pass^3 no, mean score 0.66, est. cost $1.5000, "
        "run time 150.0s**"
    ) in md
    assert "- pass^3: **no**: 0 of 3 path × mode cell(s) passed all 3 repeat(s)." in md
    assert "- Run time: 150.0s across 5 run(s) (wall clock 50.0s)." in md
    assert "- Cost: $1.5000 (runs $1.5000, judge $0.0000)." in md
    assert "Goal achieved" not in md, "no verdict graded the goal"
    assert "## Expected behavior" not in md, "no verdict has a checklist"
    assert (
        "| path | mode | runs | passed | pass rate | mean score | outcomes | time (s) "
        "| cost (USD) |"
    ) in md
    assert "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |" in md
    assert "| happy | guided | 3 | 2 | 66.7% | 0.77 | completed 3 | 60.0 | 0.6000 |" in md
    assert "| happy | free | 1 | 1 | 100.0% | 0.80 | completed 1 | 40.0 | 0.4000 |" in md
    assert "| recovery | guided | 1 | 0 | 0.0% | 0.20 | budget_exceeded 1 | 50.0 | 0.5000 |" in md
    assert (
        "| **all** | | 5 | 3 | 60.0% | 0.66 | budget_exceeded 1, completed 4 | 150.0 | 1.5000 |"
    ) in md

    assert "## Worst failures" in md
    assert "1. `recovery-guided-0`: score 0.20, outcome `budget_exceeded`" in md
    assert "   - deterministic: origin_status $eq" in md
    assert "   - transcript: `/runs/fake-lookup/t/transcripts/recovery-guided-0.jsonl`" in md
    assert "2. `happy-guided-2`: score 0.40, outcome `completed`" in md

    assert "## Cost and usage" in md and "estimate" in md
    assert "| claude-sonnet-5-5 | 400 | 80 |" in md
    assert f"Judge model(s): `{JUDGE}`." in md
    assert md.endswith("\n")


def test_render_markdown_all_pass_and_escapes_pipes() -> None:
    verdicts = [
        _verdict("happy", "guided", 0, passed=True, score=1.0),
        _verdict("a|b", "guided", 0, passed=False, score=0.1, reasons=["x | y"]),
    ]
    md = render_markdown(aggregate(verdicts, scenario="s"))
    assert "| a\\|b | guided |" in md
    assert "x | y" in md  # list items are not table cells; no escaping needed there

    clean = render_markdown(aggregate(verdicts[:1], scenario="s"))
    assert "None: every judged run passed." in clean
    assert "## Unjudged" not in clean


def test_render_markdown_lists_unjudged() -> None:
    md = render_markdown(aggregate([], [_transcript("happy", "guided", 0)], scenario="s"))
    assert "No judged runs." in md
    assert "## Unjudged transcripts" in md and "- `transcripts/happy-guided-0.jsonl`" in md


@pytest.mark.parametrize(
    ("passed", "runs", "threshold", "expected"),
    [
        (4, 5, 0.8, EXIT_PASS),  # exactly on the boundary passes
        (3, 5, 0.8, EXIT_FAIL),
        (5, 5, 1.0, EXIT_PASS),
        (4, 5, 1.0, EXIT_FAIL),
        (0, 5, 0.0, EXIT_PASS),
        (7, 9, 7 / 9, EXIT_PASS),
        (1, 3, 1 / 3, EXIT_PASS),
        (1, 3, 0.34, EXIT_FAIL),
    ],
)
def test_exit_code_threshold_boundary(
    passed: int, runs: int, threshold: float, expected: int
) -> None:
    verdicts = [
        _verdict("happy", "guided", i, passed=i < passed, score=1.0 if i < passed else 0.0)
        for i in range(runs)
    ]
    report = aggregate(verdicts)
    assert exit_code(report, threshold) == expected


def test_exit_code_rejects_bad_threshold() -> None:
    report = aggregate([_verdict("happy", "guided", 0, passed=True, score=1.0)])
    with pytest.raises(ValueError):
        exit_code(report, 1.5)
    with pytest.raises(ValueError):
        exit_code(report, -0.1)


def test_report_round_trip(tmp_path: Path) -> None:
    verdicts, transcripts = _five_runs()
    report = aggregate(verdicts, transcripts, run_dir=str(tmp_path))
    path = report.save(tmp_path / "report.json")
    loaded = Report.load(path)
    assert loaded == report
    assert render_markdown(loaded) == render_markdown(report)


def test_suite_aggregation_and_markdown(tmp_path: Path) -> None:
    verdicts, transcripts = _five_runs()
    a = aggregate(verdicts, transcripts, run_dir="/runs/fake-lookup/t1")
    b_verdicts = [_verdict("happy", "guided", i, passed=True, score=0.9) for i in range(3)]
    b = aggregate(b_verdicts, scenario="all-good", run_dir="/runs/all-good/t1")

    suite = aggregate_suite([a, b])
    assert [s.scenario for s in suite.scenarios] == ["all-good", "fake-lookup"]
    assert (suite.runs, suite.passed, suite.pass_rate) == (8, 6, 0.75)
    assert suite.mean_score == pytest.approx(round((0.66 * 5 + 0.9 * 3) / 8, 4), abs=1e-4)
    assert suite.cost_usd == pytest.approx(1.5)
    assert suite.duration_s == pytest.approx(150.0)
    assert suite.pass_k_scenarios == 1, "all-good held pass^3, fake-lookup did not"
    rows = {s.scenario: s for s in suite.scenarios}
    assert rows["fake-lookup"].worst == "recovery-guided-0"
    assert rows["all-good"].worst == ""
    assert rows["all-good"].pass_k.model_dump() == {"k": 3, "all_passed": True}
    assert rows["fake-lookup"].pass_k.model_dump() == {"k": 3, "all_passed": False}

    assert exit_code(suite, 0.75) == EXIT_PASS
    assert exit_code(suite, 0.76) == EXIT_FAIL
    assert exit_code(aggregate_suite([]), 0.0) == EXIT_FAIL

    md = render_suite_markdown(suite)
    assert md.startswith("# mcp-sim suite report\n")
    assert "**6/8 runs passed (75.0%), pass^k held in 1/2 scenario(s), mean score" in md
    assert (
        "| scenario | runs | passed | pass rate | pass^k | mean score | worst failure "
        "| time (s) | cost (USD) |"
    ) in md
    assert (
        "| fake-lookup | 5 | 3 | 60.0% | pass^3 no | 0.66 | `recovery-guided-0` | 150.0 | 1.5000 |"
    ) in md
    assert "| all-good | 3 | 3 | 100.0% | pass^3 yes | 0.90 | - | 0.0 | 0.0000 |" in md
    assert "| **all** | 8 | 6 | 75.0% | 1/2 |" in md
    assert "- fake-lookup: `/runs/fake-lookup/t1`" in md

    path = suite.save(tmp_path / "suite.json")
    assert SuiteReport.load(path) == suite


# --- pass^k, judge cost, expected-behaviour tallies (Sierra-style reports) ----------------------


def _all(passed: list[bool], path_id: str = "happy", mode: str = "guided") -> list[Verdict]:
    return [
        _verdict(path_id, mode, i, passed=ok, score=1.0 if ok else 0.0)
        for i, ok in enumerate(passed)
    ]


def test_pass_k_holds_only_when_every_repeat_of_every_cell_passed() -> None:
    both = [*_all([True, True, True]), *_all([True, True, True], mode="free")]
    assert pass_k(both).model_dump() == {"k": 3, "all_passed": True}
    one_miss = [*_all([True, True, True]), *_all([True, False, True], mode="free")]
    assert pass_k(one_miss).model_dump() == {"k": 3, "all_passed": False}
    # A cell short of a repeat cannot hold pass^3, even if every run it has passed.
    short = [*_all([True, True, True]), *_all([True, True], path_id="recovery")]
    assert pass_k(short).model_dump() == {"k": 3, "all_passed": False}
    assert pass_k(_all([True])).model_dump() == {"k": 1, "all_passed": True}
    assert pass_k([]).model_dump() == {"k": 0, "all_passed": False}
    # The pass rate and pass^k answer different questions: 5/6 is not pass^3.
    report = aggregate(one_miss)
    assert (report.pass_rate, report.pass_k.all_passed) == (0.8333, False)
    assert "pass^3 no" in summary_line(report)
    held = aggregate(both)
    assert "pass^3 yes" in summary_line(held)
    assert "- pass^3: **yes**: all 3 repeat(s) of every path and mode passed." in render_markdown(
        held
    )


def test_pass_k_uses_the_requested_repeat_and_counts_unjudged_runs_as_failures() -> None:
    two_each = [*_all([True, True]), *_all([True, True], mode="free")]
    # Inferred from the runs, two clean repeats hold pass^2 ...
    assert pass_k(two_each).model_dump() == {"k": 2, "all_passed": True}
    # ... but when three were asked for, a missing repeat is not a pass.
    assert pass_k(two_each, repeat=3).model_dump() == {"k": 3, "all_passed": False}
    assert pass_k(two_each, repeat=2).model_dump() == {"k": 2, "all_passed": True}
    # A run that was recorded but never judged (its transcript has no verdict) is not a pass.
    transcripts = [
        *(_transcript("happy", "guided", i) for i in range(3)),
        *(_transcript("happy", "free", i) for i in range(2)),
    ]
    assert pass_k(two_each, transcripts).model_dump() == {"k": 3, "all_passed": False}
    assert pass_k(two_each, transcripts[:2] + transcripts[3:]).model_dump() == {
        "k": 2,
        "all_passed": True,
    }
    # Transcripts alone (nothing judged yet): k from the runs, never passed.
    assert pass_k([], transcripts).model_dump() == {"k": 3, "all_passed": False}
    with pytest.raises(ValueError, match="repeat must be >= 1"):
        pass_k(two_each, repeat=0)
    # aggregate() threads the runner's repeat through; the markdown counts the cells that held.
    report = aggregate(two_each, repeat=3)
    assert report.pass_k.model_dump() == {"k": 3, "all_passed": False}
    assert "- pass^3: **no**: 0 of 2 path × mode cell(s) passed all 3 repeat(s)." in (
        render_markdown(report)
    )
    unjudged = aggregate(two_each, transcripts)
    assert unjudged.unjudged == ["happy-guided-2"]
    assert unjudged.pass_k.model_dump() == {"k": 3, "all_passed": False}


def test_judge_cost_and_usage_join_the_run_totals() -> None:
    opus = Usage(input_tokens=1000, output_tokens=100)
    verdicts = [
        _verdict(
            "happy", "guided", i, passed=True, score=1.0, judge_usage={JUDGE: opus},
            judge_cost=0.0225,
        )
        for i in range(2)
    ]
    transcripts = [_transcript("happy", "guided", i, cost=0.01) for i in range(2)]
    report = aggregate(verdicts, transcripts)
    assert report.run_cost_usd == pytest.approx(0.02)
    assert report.judge_cost_usd == pytest.approx(0.045)
    assert report.cost_usd == pytest.approx(0.065)
    assert report.cells[0].cost_usd == pytest.approx(0.065)
    assert report.usage[JUDGE] == Usage(input_tokens=2000, output_tokens=200)
    assert report.usage["claude-sonnet-5-5"] == Usage(input_tokens=200, output_tokens=40)
    assert "judge calls recorded in verdicts" in report.cost_note
    assert "- Cost: $0.0650 (runs $0.0200, judge $0.0450)." in render_markdown(report)


def test_expected_behavior_goal_and_sop_are_tallied_across_runs() -> None:
    def items(quote: bool, honest: bool) -> list[ChecklistItem]:
        return [
            ChecklistItem(item="Quotes the price exactly", passed=quote, evidence="[5] 2.49"),
            ChecklistItem(item="honesty: claims supported", passed=honest, evidence="[6]"),
        ]

    verdicts = [
        _verdict("happy", "guided", 0, passed=True, score=1.0, checklist=items(True, True),
                 goal_achieved=True, sop_followed=True),
        _verdict("happy", "guided", 1, passed=False, score=0.2, checklist=items(False, True),
                 goal_achieved=False, sop_followed=False),
        _verdict("happy", "guided", 2, passed=True, score=0.9, checklist=items(True, True),
                 goal_achieved=True, sop_followed=True),
        # A dry-run verdict: nothing graded, so it adds to no tally.
        _verdict("happy", "free", 0, passed=True, score=1.0),
    ]
    report = aggregate(verdicts)
    assert [(b.item, b.passed, b.graded) for b in report.behavior] == [
        ("Quotes the price exactly", 2, 3),
        ("honesty: claims supported", 3, 3),
    ]
    assert (report.goal_achieved.passed, report.goal_achieved.graded) == (2, 3)
    assert (report.sop_followed.passed, report.sop_followed.graded) == (2, 3)
    md = render_markdown(report)
    assert "## Expected behavior" in md
    assert "| Quotes the price exactly | 2 | 3 | 66.7% |" in md
    assert "| honesty: claims supported | 3 | 3 | 100.0% |" in md
    assert "- Goal achieved: 2/3 judged run(s)." in md
    assert "- Standard operating procedure followed: 2/3 judged run(s)." in md
    # Without an SOP nobody grades it and the line is left out.
    no_sop = aggregate([_verdict("h", "guided", 0, passed=True, score=1.0, goal_achieved=True)])
    assert no_sop.sop_followed.graded == 0
    assert "Standard operating procedure" not in render_markdown(no_sop)


def test_reports_and_verdicts_written_before_v2_still_load(tmp_path: Path) -> None:
    old_report = {
        "scenario": "s",
        "runs": 1,
        "passed": 1,
        "pass_rate": 1.0,
        "mean_score": 1.0,
        "cost_usd": 0.1,
        "cells": [{"path_id": "happy", "mode": "guided", "runs": 1, "passed": 1}],
    }
    loaded = Report.model_validate(old_report)
    assert loaded.pass_k.model_dump() == {"k": 0, "all_passed": False}
    assert loaded.duration_s == 0.0 and loaded.behavior == []
    old_verdict = {
        "path_id": "happy",
        "mode": "guided",
        "index": 0,
        "passed": True,
        "score": 1.0,
        "votes": 3,
        "judge_model": JUDGE,
        "checklist": [{"item": "goal achieved", "passed": True, "evidence": "[3]"}],
    }
    path = tmp_path / "v.json"
    path.write_text(json.dumps(old_verdict), encoding="utf-8")
    v = Verdict.load(path)
    assert (v.goal_achieved, v.sop_followed, v.judge_cost_usd, v.judge_usage) == (
        None,
        None,
        0.0,
        {},
    )
