from __future__ import annotations

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
    render_markdown,
    render_suite_markdown,
    summary_line,
)
from mcpsim.transcript import EndEvent, SystemEvent, Transcript
from mcpsim.verdict import ChecklistItem, Match, Verdict

JUDGE = "claude-opus-5-5"


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
    t.add(SystemEvent(scenario="fake-lookup", path_id=path_id, mode=mode, index=index))
    t.add(EndEvent(outcome=outcome, reason=reason))  # type: ignore[arg-type]
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
        _transcript("happy", "guided", 0, cost=0.10),
        _transcript("happy", "guided", 1, cost=0.20),
        _transcript("happy", "guided", 2, cost=0.30),
        _transcript(
            "happy",
            "free",
            0,
            cost=0.40,
            usage={"claude-haiku-4-5": Usage(input_tokens=5, output_tokens=5)},
        ),
        _transcript(
            "recovery", "guided", 0, cost=0.50, outcome="budget_exceeded", reason="max_turns"
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
    assert report.judge_models == [JUDGE]
    assert report.outcomes == {"budget_exceeded": 1, "completed": 4}
    assert report.unjudged == []

    cells = {(c.path_id, c.mode): c for c in report.cells}
    assert set(cells) == {("happy", "guided"), ("happy", "free"), ("recovery", "guided")}
    hg = cells[("happy", "guided")]
    assert (hg.runs, hg.passed, hg.pass_rate) == (3, 2, 0.6667)
    assert hg.mean_score == pytest.approx(round((1.0 + 0.9 + 0.4) / 3, 4))
    assert hg.cost_usd == pytest.approx(0.6)
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
    assert "**3/5 runs passed (60.0%), mean score 0.66, est. cost $1.5000**" in md
    assert "| path | mode | runs | passed | pass rate | mean score | outcomes | cost (USD) |" in md
    assert "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |" in md
    assert "| happy | guided | 3 | 2 | 66.7% | 0.77 | completed 3 | 0.6000 |" in md
    assert "| happy | free | 1 | 1 | 100.0% | 0.80 | completed 1 | 0.4000 |" in md
    assert "| recovery | guided | 1 | 0 | 0.0% | 0.20 | budget_exceeded 1 | 0.5000 |" in md
    assert "| **all** | | 5 | 3 | 60.0% | 0.66 | budget_exceeded 1, completed 4 | 1.5000 |" in md

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
    rows = {s.scenario: s for s in suite.scenarios}
    assert rows["fake-lookup"].worst == "recovery-guided-0"
    assert rows["all-good"].worst == ""

    assert exit_code(suite, 0.75) == EXIT_PASS
    assert exit_code(suite, 0.76) == EXIT_FAIL
    assert exit_code(aggregate_suite([]), 0.0) == EXIT_FAIL

    md = render_suite_markdown(suite)
    assert md.startswith("# mcp-sim suite report\n")
    assert (
        "| scenario | runs | passed | pass rate | mean score | worst failure | cost (USD) |" in md
    )
    assert "| fake-lookup | 5 | 3 | 60.0% | 0.66 | `recovery-guided-0` | 1.5000 |" in md
    assert "| all-good | 3 | 3 | 100.0% | 0.90 | - | 0.0000 |" in md
    assert "| **all** | 8 | 6 | 75.0% |" in md
    assert "- fake-lookup: `/runs/fake-lookup/t1`" in md

    path = suite.save(tmp_path / "suite.json")
    assert SuiteReport.load(path) == suite
