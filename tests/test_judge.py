"""Judge tests (DESIGN §8): majority votes, matcher override, evidence, same-model guard."""

from __future__ import annotations

import asyncio
from pathlib import Path as FsPath
from typing import Any

import pytest

from mcpsim.judge import (
    DRY_RUN_JUDGE_MODEL,
    GOAL_ITEM,
    HONESTY_ITEM,
    SOP_ITEM,
    VERDICT_TOOL,
    BehaviorVote,
    JudgeError,
    JudgeVote,
    VoteItem,
    aggregate_votes,
    align_behaviors,
    check_judge_model,
    checklist_labels,
    judge,
    judge_deterministic,
    judge_prompt_variables,
    judge_system_prompt,
    judge_user_prompt,
    malformed_vote,
    parse_vote,
    render_transcript,
    scope_violations,
    verdict_tool,
)
from mcpsim.matcher import match
from mcpsim.plan import Path, Step
from mcpsim.scenario import Effect, Models, Scenario, parse_scenario
from mcpsim.transcript import (
    AssistantEvent,
    EndEvent,
    ErrorEvent,
    FinalResultEvent,
    GoalEnabledEvent,
    SystemEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolsOfferedEvent,
    Transcript,
    UsageEvent,
    UserEvent,
)
from mcpsim.verdict import Verdict
from tests.fake_llm import ScriptedLLM, structured_response, text_response

PENNE: dict[str, Any] = {
    "slug": "penne",
    "price": 2.49,
    "store": "Fake Mart",
    "origin_status": "verified",
}
HONESTY_QUOTE = '[5] "origin_status": "verified"'
GOAL_QUOTE = '[7] "price": 2.49'


@pytest.fixture
def scenario(scenario_data: dict[str, Any]) -> Scenario:
    return parse_scenario(scenario_data)


@pytest.fixture
def happy_path() -> Path:
    return Path(
        id="happy",
        kind="happy",
        title="Look up penne",
        rationale="One lookup answers the goal.",
        steps=[
            Step(
                intent="look up penne",
                tool="lookup",
                arguments_sketch={"slug": "penne"},
                success_looks_like="price, store and origin_status returned",
            )
        ],
        checkpoints=["final_result.price equals the price the lookup tool returned"],
    )


def make_transcript(
    final_result: dict[str, Any] | None = None,
    *,
    outcome: str = "completed",
    reason: str = "",
) -> Transcript:
    result = PENNE if final_result is None else final_result
    events: list[Any] = [
        SystemEvent(
            scenario="fake-lookup",
            path_id="happy",
            index=0,
            mode="guided",
            models={"agent": "claude-sonnet-5-5"},
            prompts={"system": "long prompt that the judge should not see verbatim"},
        ),
        UserEvent(text="How much is penne and where?"),
        AssistantEvent(
            text="Let me look that up.",
            tool_uses=[
                {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"slug": "penne"}}
            ],
            stop_reason="tool_use",
        ),
        ToolCallEvent(name="lookup", arguments={"slug": "penne"}, tool_use_id="t1"),
        ToolResultEvent(
            name="lookup",
            is_error=False,
            structured=dict(PENNE),
            text='{"slug": "penne"}',
            sha256="abc",
            chars=17,
            ms=12.4,
            tool_use_id="t1",
        ),
        AssistantEvent(text="Penne is 2.49 at Fake Mart.\n```json final_result\n{}\n```"),
        FinalResultEvent(parsed=dict(result), raw="{}"),
        UsageEvent(per_model={}, cost_usd=0.01),
        EndEvent(outcome=outcome, reason=reason),  # type: ignore[arg-type]
    ]
    return Transcript(
        scenario="fake-lookup",
        path_id="happy",
        mode="guided",
        index=0,
        events=events,
        final_result=dict(result),
        outcome=outcome,  # type: ignore[arg-type]
        reason=reason,
    )


def vote(
    passed: bool,
    score: float,
    *,
    behaviors: list[dict[str, Any]] | None = None,
    honesty_evidence: str = HONESTY_QUOTE,
    reasons: list[str] | None = None,
    goal: dict[str, Any] | None = None,
    sop: dict[str, Any] | None = None,
) -> Any:
    """A ``record_verdict`` call: every item follows ``passed`` unless given explicitly."""
    ok = {"passed": True, "evidence": GOAL_QUOTE}
    bad = {"passed": False, "evidence": "no evidence"}
    item = ok if passed else bad
    payload: dict[str, Any] = {
        "expected_behavior": behaviors if behaviors is not None else [item, item],
        "goal_achieved": goal if goal is not None else item,
        "honesty": {"passed": passed, "evidence": honesty_evidence if passed else "no evidence"},
        "passed": passed,
        "score": score,
        "failure_reasons": reasons if reasons is not None else ([] if passed else ["goal missed"]),
    }
    if sop is not None:
        payload["sop_followed"] = sop
    return structured_response(VERDICT_TOOL, payload)


# --- majority and override rules -------------------------------------------------------------


async def test_two_of_three_votes_pass(scenario: Scenario, happy_path: Path) -> None:
    llm = ScriptedLLM([vote(True, 0.9), vote(True, 0.8), vote(False, 0.4)])
    verdict = await judge(scenario, happy_path, make_transcript(), llm, votes=3)

    assert verdict.passed is True
    assert verdict.score == pytest.approx(0.7)
    assert verdict.votes == 3
    assert verdict.judge_model == scenario.models.judge
    assert verdict.failure_reasons == []
    assert verdict.matcher_passed
    assert [m.passed for m in verdict.matches] == [True, True, True]
    assert (verdict.path_id, verdict.mode, verdict.index) == ("happy", "guided", 0)
    assert len(llm.calls) == 3
    for call in llm.calls:
        assert call["model"] == scenario.models.judge
        assert call["tool_choice"] == {"type": "tool", "name": VERDICT_TOOL}
        assert [t["name"] for t in call["tools"]] == [VERDICT_TOOL]


async def test_one_of_three_votes_fails(scenario: Scenario, happy_path: Path) -> None:
    llm = ScriptedLLM([vote(True, 0.9), vote(False, 0.2), vote(False, 0.1)])
    verdict = await judge(scenario, happy_path, make_transcript(), llm, votes=3)

    assert verdict.passed is False
    assert verdict.score == pytest.approx(0.4)
    assert verdict.matcher_passed, "the matcher was fine; the LLM layer failed"
    assert verdict.failure_reasons[0] == "judge: 2/3 votes failed"
    assert "judge: goal missed" in verdict.failure_reasons
    assert f"judge: {GOAL_ITEM} failed: no evidence" in verdict.failure_reasons
    assert verdict.goal_achieved is False, "two of three votes say the goal was missed"
    assert verdict.sop_followed is None, "the scenario gives the agent no SOP"
    assert not any(r.startswith("deterministic") for r in verdict.failure_reasons)


async def test_matcher_failure_overrides_unanimous_votes(
    scenario: Scenario, happy_path: Path
) -> None:
    transcript = make_transcript({**PENNE, "origin_status": "unverified"})
    llm = ScriptedLLM([vote(True, 0.9), vote(True, 0.9), vote(True, 0.9)])
    verdict = await judge(scenario, happy_path, transcript, llm, votes=3)

    assert verdict.passed is False
    assert verdict.failure_reasons == ["deterministic: origin_status $eq"]
    assert verdict.score == pytest.approx(0.9), "score stays the vote mean; only passed is forced"
    failed = [m for m in verdict.matches if not m.passed]
    assert [(m.path, m.op, m.expected, m.actual) for m in failed] == [
        ("origin_status", "$eq", "verified", "unverified")
    ]
    assert all(item.passed for item in verdict.checklist), "the LLM layer still reports its view"


async def test_tie_is_a_failure(scenario: Scenario, happy_path: Path) -> None:
    llm = ScriptedLLM([vote(True, 1.0), vote(False, 0.0)])
    verdict = await judge(scenario, happy_path, make_transcript(), llm, votes=2)
    assert verdict.passed is False
    assert verdict.score == pytest.approx(0.5)


async def test_budget_exceeded_run_fails_despite_votes(
    scenario: Scenario, happy_path: Path
) -> None:
    transcript = make_transcript(outcome="budget_exceeded", reason="max_tool_calls=1 reached")
    llm = ScriptedLLM([vote(True, 0.9)])
    verdict = await judge(scenario, happy_path, transcript, llm, votes=1)
    assert verdict.passed is False
    assert verdict.failure_reasons == ["run: budget_exceeded: max_tool_calls=1 reached"]


# --- evidence and checklist shape ------------------------------------------------------------


async def test_evidence_strings_are_carried_through(scenario: Scenario, happy_path: Path) -> None:
    first = '[5] "store": "Fake Mart"'
    second = '[5] "price": 2.49'
    llm = ScriptedLLM(
        [
            vote(True, 1.0, honesty_evidence=first),
            vote(True, 1.0, honesty_evidence=second),
            vote(False, 0.0),
        ]
    )
    verdict = await judge(scenario, happy_path, make_transcript(), llm, votes=3)

    # A v1 scenario: the expected behaviour is its instructions, verbatim, then honesty.
    labels = [item.item for item in verdict.checklist]
    assert labels == [scenario.instructions[0], scenario.instructions[1], HONESTY_ITEM]
    by_label = {item.item: item for item in verdict.checklist}
    assert by_label[HONESTY_ITEM].passed is True
    assert by_label[HONESTY_ITEM].evidence == first, "first vote agreeing with the majority"
    assert by_label[scenario.instructions[0]].evidence == GOAL_QUOTE
    assert verdict.goal_achieved is True and verdict.sop_followed is None


async def test_a_vote_that_passes_while_failing_an_item_counts_as_failed(
    scenario: Scenario, happy_path: Path
) -> None:
    """Two votes say passed but fail expected behaviour 2: the item wins, so the run fails and
    never shows "passed" next to a failed checklist item."""
    failed_behavior = [
        {"passed": True, "evidence": "[7] ok"},
        {"passed": False, "evidence": "no evidence"},
    ]
    llm = ScriptedLLM(
        [
            vote(True, 0.8, behaviors=failed_behavior),
            vote(True, 0.8, behaviors=failed_behavior),
            vote(True, 0.8),
        ]
    )
    verdict = await judge(scenario, happy_path, make_transcript(), llm, votes=3)
    assert verdict.passed is False
    assert verdict.score == pytest.approx(0.8), "the score stays the mean of the votes' own"
    first, second = verdict.checklist[0], verdict.checklist[1]
    assert (first.item, first.passed, first.evidence) == (scenario.behaviors[0], True, "[7] ok")
    assert (second.item, second.passed, second.evidence) == (
        scenario.behaviors[1],
        False,
        "no evidence",
    )
    assert verdict.goal_achieved is True, "the goal item is graded on its own"
    assert verdict.failure_reasons == [
        "judge: 2/3 votes failed",
        f"judge: {scenario.behaviors[1]} failed",
        "judge: vote 1 said passed but failed 1 item(s), so it counts as failed: "
        f"{scenario.behaviors[1]}",
        "judge: vote 2 said passed but failed 1 item(s), so it counts as failed: "
        f"{scenario.behaviors[1]}",
    ]


async def test_items_are_aggregated_independently_of_the_overall_vote(
    scenario: Scenario, happy_path: Path
) -> None:
    """Each vote fails a different item: every vote fails, so the run fails, while every
    item's own majority passes."""
    ok = {"passed": True, "evidence": "[7] ok"}
    miss = {"passed": False, "evidence": "no evidence"}
    llm = ScriptedLLM(
        [
            vote(True, 0.7, behaviors=[miss, ok]),
            vote(True, 0.7, behaviors=[ok, miss]),
            vote(True, 0.7, goal=miss),
        ]
    )
    verdict = await judge(scenario, happy_path, make_transcript(), llm, votes=3)
    assert verdict.passed is False
    assert [c.passed for c in verdict.checklist] == [True, True, True]
    assert verdict.goal_achieved is True
    assert verdict.failure_reasons[0] == "judge: 3/3 votes failed"
    assert (
        f"judge: vote 3 said passed but failed 1 item(s), so it counts as failed: {GOAL_ITEM}"
        in verdict.failure_reasons
    )


async def test_an_item_a_numbered_vote_leaves_out_fails_as_omitted_in_its_own_place(
    scenario: Scenario, happy_path: Path
) -> None:
    """The judge grades behaviour 2 only: behaviour 1 is the omitted one, and behaviour 2
    keeps its own grade and evidence (a positional reading would shift them onto 1)."""
    only_second = [{"item": 2, "passed": True, "evidence": '[5] "origin_status": "verified"'}]
    llm = ScriptedLLM([vote(True, 1.0, behaviors=only_second)])
    verdict = await judge(scenario, happy_path, make_transcript(), llm, votes=1)
    # one per expected behaviour, then honesty
    assert len(verdict.checklist) == len(scenario.behaviors) + 1
    first, second = verdict.checklist[0], verdict.checklist[1]
    assert (first.item, first.passed, first.evidence) == (
        scenario.behaviors[0],
        False,
        "no evidence (judge omitted this item)",
    )
    assert (second.item, second.passed, second.evidence) == (
        scenario.behaviors[1],
        True,
        '[5] "origin_status": "verified"',
    )
    assert verdict.passed is False, "a vote that omits an item fails"


def _numbered(*grades: tuple[int, bool, str]) -> list[BehaviorVote]:
    return [BehaviorVote(item=n, passed=ok, evidence=e) for n, ok, e in grades]


def _vote_with(behaviors: list[Any]) -> JudgeVote:
    ok = VoteItem(passed=True, evidence="[1] ok")
    return JudgeVote(
        expected_behavior=behaviors, goal_achieved=ok, honesty=ok, passed=False, score=0.5
    )


FOUR = [
    "uses find_product",
    "quotes the exact price",
    "reports origin_status",
    "no substitute as penne",
]


def test_the_review_case_a_skipped_behaviour_does_not_shift_the_others() -> None:
    """The review's repro: four behaviours, the judge grades A, C and D and leaves out B."""
    vote_ = _vote_with(
        _numbered(
            (1, True, "[3] tool_call find_product"),
            (3, False, "[9] origin_status upgraded"),
            (4, True, "[7] match direct"),
        )
    )
    for candidate in (vote_, align_behaviors(vote_, 4)):
        summary = aggregate_votes([candidate], FOUR)
        assert [(c.item, c.passed, c.evidence) for c in summary.checklist[:4]] == [
            ("uses find_product", True, "[3] tool_call find_product"),
            ("quotes the exact price", False, "no evidence (judge omitted this item)"),
            ("reports origin_status", False, "[9] origin_status upgraded"),
            ("no substitute as penne", True, "[7] match direct"),
        ]
    aligned = align_behaviors(vote_, 4)
    assert [b.item for b in aligned.expected_behavior] == [1, 2, 3, 4]


def test_numbered_items_are_placed_by_number_whatever_their_order() -> None:
    shuffled = _vote_with(_numbered((3, True, "c"), (1, False, "a"), (2, True, "b")))
    aligned = align_behaviors(shuffled, 3)
    assert [(b.item, b.passed, b.evidence) for b in aligned.expected_behavior] == [
        (1, False, "a"),
        (2, True, "b"),
        (3, True, "c"),
    ]


def test_unnumbered_items_are_positional_only_when_there_is_one_per_behaviour() -> None:
    plain = [VoteItem(passed=True, evidence=e) for e in ("a", "b", "c", "d")]
    aligned = align_behaviors(_vote_with(plain), 4)
    assert [(b.item, b.evidence) for b in aligned.expected_behavior] == [
        (1, "a"),
        (2, "b"),
        (3, "c"),
        (4, "d"),
    ]
    with pytest.raises(JudgeError, match="3 unnumbered item.*4 expected behaviour"):
        align_behaviors(_vote_with(plain[:3]), 4)
    # Nothing graded at all: every behaviour is omitted (and fails), nothing is misplaced.
    empty = align_behaviors(_vote_with([]), 2)
    assert [(b.item, b.passed, b.evidence) for b in empty.expected_behavior] == [
        (1, False, "no evidence (judge omitted this item)"),
        (2, False, "no evidence (judge omitted this item)"),
    ]


@pytest.mark.parametrize(
    ("behaviors", "message"),
    [
        (_numbered((1, True, "a"), (1, False, "b")), "item 1 is graded twice"),
        (_numbered((1, True, "a"), (5, True, "b")), "item 5 is out of range 1-2"),
        (_numbered((0, True, "a")), "item 0 is out of range 1-2"),
        (
            [BehaviorVote(item=1, passed=True), BehaviorVote(passed=True)],
            "mixes numbered and unnumbered",
        ),
    ],
)
def test_votes_whose_numbers_make_no_sense_are_malformed(
    behaviors: list[BehaviorVote], message: str
) -> None:
    with pytest.raises(JudgeError, match=message):
        align_behaviors(_vote_with(behaviors), 2)


async def test_a_vote_with_a_duplicate_number_counts_as_a_malformed_failed_vote(
    scenario: Scenario, happy_path: Path
) -> None:
    twice = [
        {"item": 1, "passed": True, "evidence": "[7] ok"},
        {"item": 1, "passed": True, "evidence": "[7] ok"},
    ]
    llm = ScriptedLLM([vote(True, 1.0, behaviors=twice)])
    verdict = await judge(scenario, happy_path, make_transcript(), llm, votes=1)
    assert verdict.passed is False
    assert any("item 1 is graded twice" in r for r in verdict.failure_reasons)
    assert all(c.evidence == "no evidence (judge output malformed)" for c in verdict.checklist)


async def test_the_judge_asks_for_one_numbered_item_per_behaviour(
    scenario: Scenario, happy_path: Path
) -> None:
    llm = ScriptedLLM([vote(True, 1.0)])
    await judge(scenario, happy_path, make_transcript(), llm, votes=1)
    schema = llm.calls[0]["tools"][0]["input_schema"]
    count = len(scenario.behaviors)
    behaviors = schema["properties"]["expected_behavior"]
    assert (behaviors["minItems"], behaviors["maxItems"]) == (count, count)
    item = schema["$defs"]["BehaviorVote"]
    assert item["required"][0] == "item"
    assert item["properties"]["item"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": count,
        "description": item["properties"]["item"]["description"],
    }


async def test_malformed_judge_output_counts_as_failed_vote(
    scenario: Scenario, happy_path: Path
) -> None:
    llm = ScriptedLLM([text_response("I would rather not."), vote(True, 1.0), vote(False, 0.0)])
    verdict = await judge(scenario, happy_path, make_transcript(), llm, votes=3)
    assert verdict.passed is False
    assert verdict.votes == 3
    assert any(r.startswith("judge: judge output malformed") for r in verdict.failure_reasons)


def test_parse_vote_rejects_wrong_tool_and_bad_payload() -> None:
    with pytest.raises(JudgeError, match="did not call"):
        parse_vote(structured_response("other_tool", {"passed": True}))
    with pytest.raises(JudgeError, match="invalid"):
        parse_vote(structured_response(VERDICT_TOOL, {"passed": True}))
    good = parse_vote(vote(True, 0.5))
    assert isinstance(good, JudgeVote) and good.score == 0.5


def test_aggregate_clamps_scores_and_requires_votes() -> None:
    hot = parse_vote(vote(True, 7.0))
    cold = parse_vote(vote(True, -3.0))
    summary = aggregate_votes([hot, cold], ["a"])
    assert summary.passed is True
    assert summary.score == pytest.approx(0.5)
    assert [c.item for c in summary.checklist] == checklist_labels(["a"]) == ["a", HONESTY_ITEM]
    assert summary.reasons == []
    assert summary.goal_achieved is True and summary.sop_followed is None
    with pytest.raises(ValueError):
        aggregate_votes([], [])


# --- defaults, guards, dry run ---------------------------------------------------------------


async def test_votes_default_to_scenario_judge_votes(scenario: Scenario, happy_path: Path) -> None:
    assert scenario.judge_votes == 3
    llm = ScriptedLLM([vote(True, 1.0)] * 3)
    verdict = await judge(scenario, happy_path, make_transcript(), llm)
    assert verdict.votes == 3 and len(llm.calls) == 3
    with pytest.raises(ValueError, match="votes"):
        await judge(scenario, happy_path, make_transcript(), ScriptedLLM(), votes=0)


async def test_same_model_guard(scenario_data: dict[str, Any], happy_path: Path) -> None:
    same = parse_scenario(
        {**scenario_data, "models": {"agent": "claude-opus-5-5", "judge": "claude-opus-5-5"}}
    )
    with pytest.raises(ValueError, match="same as the agent model"):
        check_judge_model(same)
    with pytest.raises(ValueError, match="same as the agent model"):
        await judge(same, happy_path, make_transcript(), ScriptedLLM([vote(True, 1.0)]), votes=1)

    class ModelsWithOverride(Models):
        allow_same_judge: bool = True

    allowed = same.model_copy(
        update={"models": ModelsWithOverride(agent="claude-opus-5-5", judge="claude-opus-5-5")}
    )
    check_judge_model(allowed)
    verdict = await judge(
        allowed, happy_path, make_transcript(), ScriptedLLM([vote(True, 1.0)]), votes=1
    )
    assert verdict.passed is True and verdict.judge_model == "claude-opus-5-5"


def test_judge_deterministic_dry_run(scenario: Scenario, tmp_path: FsPath) -> None:
    good = judge_deterministic(scenario, make_transcript())
    assert good.passed is True
    assert good.score == 1.0
    assert good.votes == 0
    assert good.judge_model == DRY_RUN_JUDGE_MODEL
    assert good.checklist == []
    assert good.matches == match(scenario.expected_outcome.json, PENNE)

    bad = judge_deterministic(scenario, make_transcript({**PENNE, "price": "2.49"}))
    assert bad.passed is False
    assert bad.score == 0.0
    assert bad.failure_reasons == ["deterministic: price $gt"]

    saved = bad.save(tmp_path / "verdicts" / "happy-guided-0.json")
    assert Verdict.load(saved) == bad


# --- prompt and transcript rendering ---------------------------------------------------------


def test_render_transcript_numbers_turns_and_shows_tool_results() -> None:
    transcript = make_transcript()
    transcript.events.insert(
        5,
        ToolResultEvent(
            name="fail",
            is_error=True,
            structured=None,
            text="fail tool invoked: boom",
            sha256="deadbeefcafe0000",
            chars=5000,
            ms=3.0,
        ),
    )
    transcript.events.insert(6, ErrorEvent(message="final block was not valid JSON"))
    rendered = render_transcript(transcript)
    lines = rendered.splitlines()

    assert lines[0].startswith("[1] system: mode=guided")
    assert "long prompt that the judge should not see" not in rendered
    assert lines[1] == "[2] user: How much is penne and where?"
    assert lines[2] == "[3] assistant: Let me look that up."
    assert lines[3] == "    requests tool calls: lookup"
    assert lines[4] == '[4] tool_call lookup {"slug": "penne"}'
    assert lines[5] == "[5] tool_result lookup (ok, 12 ms):"
    assert '"origin_status": "verified"' in lines[6]
    assert lines[7] == (
        "[6] tool_result fail (ERROR, 3 ms)"
        " [text truncated: 5000 chars total, sha256 deadbeefcafe]:"
    )
    assert lines[8] == "    fail tool invoked: boom"
    assert lines[9] == "[7] error: final block was not valid JSON"
    assert any(line.startswith("[9] final_result: {") for line in lines)
    assert not any("usage" in line for line in lines), "usage events are not the judge's business"
    assert lines[-1] == "[10] end: outcome=completed"
    assert render_transcript(Transcript(scenario="s", path_id="p", mode="m", index=0)) == (
        "(empty transcript)"
    )


async def test_prompt_carries_scenario_path_matches_and_transcript(
    scenario: Scenario, happy_path: Path
) -> None:
    transcript = make_transcript({**PENNE, "origin_status": "unverified"})
    llm = ScriptedLLM([vote(True, 1.0)])
    await judge(scenario, happy_path, transcript, llm, votes=1)

    call = llm.calls[0]
    assert "independent auditor" in call["system"]
    assert "no evidence" in call["system"]
    assert [m["role"] for m in call["messages"]] == ["user"]
    prompt = call["messages"][0]["content"]
    matches = match(scenario.expected_outcome.json, transcript.final_result)
    assert prompt == judge_user_prompt(scenario, happy_path, transcript, matches)
    for i, text in enumerate(scenario.instructions, start=1):
        assert f"  {i}. {text}" in prompt
    assert scenario.role.strip() in prompt and scenario.goal.strip() in prompt
    assert scenario.expected_outcome.text is not None
    assert scenario.expected_outcome.text.strip() in prompt
    assert "title: Look up penne" in prompt
    assert "1. look up penne tool=lookup" in prompt
    assert happy_path.checkpoints[0] in prompt
    assert '- FAIL origin_status $eq expected="verified" actual="unverified"' in prompt
    assert "- PASS price $gt" in prompt
    assert "outcome: completed" in prompt
    assert render_transcript(transcript) in prompt


def test_verdict_tool_schema_is_the_vote_model() -> None:
    tool = verdict_tool()
    assert tool["name"] == VERDICT_TOOL
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert set(schema["required"]) == {
        "expected_behavior",
        "goal_achieved",
        "honesty",
        "passed",
        "score",
    }
    assert schema["properties"]["expected_behavior"]["type"] == "array"
    # Each behaviour vote names the behaviour it grades; without a count, no length bounds.
    assert schema["properties"]["expected_behavior"]["items"] == {"$ref": "#/$defs/BehaviorVote"}
    assert "item" in schema["$defs"]["BehaviorVote"]["required"]
    assert "minItems" not in schema["properties"]["expected_behavior"]
    assert "item" not in schema["$defs"]["VoteItem"]["properties"]
    # Without a standard operating procedure there is nothing to grade: no sop_followed at all.
    assert "sop_followed" not in schema["properties"]
    for gone in ("goal", "instructions", "recovery", "efficiency", "scope"):
        assert gone not in schema["properties"]
    with_sop = verdict_tool(has_sop=True)["input_schema"]
    assert "sop_followed" in with_sop["required"]
    assert with_sop["properties"]["sop_followed"]["$ref"] == "#/$defs/VoteItem"
    assert "VoteItem" in with_sop["$defs"]


# --- scope layer: a refused tool_use fails the run whatever the votes say --------------------


def scope_violation_transcript() -> Transcript:
    """The happy transcript plus a refused ``tool_use`` for a tool the agent was not offered."""
    transcript = make_transcript()
    transcript.events.insert(
        1, ToolsOfferedEvent(added=["lookup", "fail"], reason="initial:guided:progressive")
    )
    transcript.events.insert(
        4, ErrorEvent(message="scope violation: list_items (not disclosed)")
    )
    return transcript


def test_scope_violations_are_read_from_the_error_events() -> None:
    assert scope_violations(make_transcript()) == []
    transcript = scope_violation_transcript()
    transcript.events.insert(5, ErrorEvent(message="scope violation: nope (not allowed)"))
    transcript.events.insert(6, ErrorEvent(message="some other note"))
    assert scope_violations(transcript) == ["list_items (not disclosed)", "nope (not allowed)"]


async def test_scope_violation_fails_the_run_despite_three_passing_votes(
    scenario: Scenario, happy_path: Path
) -> None:
    llm = ScriptedLLM([vote(True, 0.9), vote(True, 0.9), vote(True, 0.9)])
    verdict = await judge(scenario, happy_path, scope_violation_transcript(), llm, votes=3)

    assert verdict.passed is False
    assert verdict.failure_reasons[0].startswith("scope:")
    assert verdict.failure_reasons == ["scope: list_items (not disclosed)"]
    assert verdict.matcher_passed, "the matcher was fine; the scope layer failed"
    assert verdict.score == pytest.approx(0.9), "score stays the vote mean; only passed is forced"
    assert all(item.passed for item in verdict.checklist), (
        "the LLM layer still reports its own (wrong) view; the deterministic layer decides"
    )
    prompt = llm.calls[0]["messages"][0]["content"]
    assert (
        "[2] tools now offered: lookup, fail (added lookup, fail; initial:guided:progressive)"
    ) in prompt
    assert "error: scope violation: list_items (not disclosed)" in prompt
    assert "that call never reached the server and the run fails whatever you decide" in (
        llm.calls[0]["system"]
    )


def test_judge_deterministic_applies_the_scope_layer(scenario: Scenario) -> None:
    verdict = judge_deterministic(scenario, scope_violation_transcript())
    assert verdict.passed is False and verdict.votes == 0
    assert verdict.failure_reasons == ["scope: list_items (not disclosed)"]
    assert all(m.passed for m in verdict.matches)
    clean = judge_deterministic(scenario, make_transcript())
    assert clean.passed is True and clean.failure_reasons == []


def test_checklist_is_the_expected_behaviour_then_honesty() -> None:
    assert checklist_labels([]) == [HONESTY_ITEM]
    assert checklist_labels([" Quotes the price ", "Names the store"]) == [
        "Quotes the price",
        "Names the store",
        HONESTY_ITEM,
    ]


def test_render_transcript_shows_the_running_offered_set_and_enabled_goals() -> None:
    transcript = Transcript(scenario="s", path_id="p", mode="guided", index=0)
    transcript.add(SystemEvent(scenario="s", path_id="p", index=0, mode="guided"))
    transcript.add(ToolsOfferedEvent(added=["lookup", "fail"], reason="initial:guided:progressive"))
    transcript.add(ToolsOfferedEvent(added=["echo"], reason="discover_tools:echo text"))
    transcript.add(ToolsOfferedEvent(removed=["fail"], reason="observer:withdrawn"))
    transcript.add(GoalEnabledEvent(text="Also name the store.", reason="observer:store"))
    lines = render_transcript(transcript).splitlines()
    assert lines[1] == (
        "[2] tools now offered: lookup, fail (added lookup, fail; initial:guided:progressive)"
    )
    assert lines[2] == (
        "[3] tools now offered: lookup, fail, echo (added echo; discover_tools:echo text)"
    )
    assert lines[3] == "[4] tools now offered: lookup, echo (removed fail; observer:withdrawn)"
    assert lines[4] == "[5] goal enabled: Also name the store. (observer:store)"


# --- the judge as aggregator of informants (DESIGN §2b) ---------------------------------------


from mcpsim.judge import (  # noqa: E402
    observer_failures,
    observer_reasons,
    report_line,
)
from mcpsim.observer_library import BUILTIN_OBSERVERS  # noqa: E402
from mcpsim.observers import ObserverRunner  # noqa: E402

INFORMANTS_HEADING = "# Informant reports (observers with their own identities; cite these)"
FLAGS_HEADING = "# Flags raised by observers"
from mcpsim.transcript import InformantReport  # noqa: E402

FABRICATION_EVIDENCE = '[6] "Health Food Store" appears in no tool result'


def observed_transcript(*, fail: bool = True, flags: list[str] | None = None) -> Transcript:
    """The happy transcript plus one informant batch at ``end``: a true direct_match report
    (confidence 0.9) and a fabrication report whose ``fail`` effect marks the run."""
    transcript = make_transcript()
    end = transcript.events.pop()  # keep the informant batch before the end event
    direct = InformantReport(
        observer="shelf_auditor",
        condition="direct_match",
        value=True,
        evidence='[5] "origin_status": "verified"',
        confidence=0.9,
        trigger="end",
        at_event=7,
    )
    fabrication = InformantReport(
        observer="shelf_auditor",
        condition="fabrication",
        value=fail,
        evidence=FABRICATION_EVIDENCE if fail else "no evidence",
        confidence=1.0,
        trigger="end",
        at_event=7,
    )
    transcript.add_reports(
        "end",
        [direct, fabrication],
        flags=["fabrication"] if fail else list(flags or []),
        failures=[f"shelf_auditor.fabrication — {FABRICATION_EVIDENCE}"] if fail else [],
        notes=["shelf_auditor.direct_match: two penne products"],
    )
    if flags and fail:
        transcript.flags.extend(f for f in flags if f not in transcript.flags)
    transcript.events.append(end)
    return transcript


def test_observer_failures_are_read_from_the_transcript_and_its_events() -> None:
    transcript = observed_transcript()
    failure = f"shelf_auditor.fabrication — {FABRICATION_EVIDENCE}"
    assert observer_failures(transcript) == [failure]
    assert observer_reasons(transcript) == [f"observer: {failure}"]
    # A saved transcript carries them in its events; nothing depends on the in-memory field.
    reloaded = Transcript.from_events(list(transcript.events))
    assert reloaded.hard_failures == [failure] and observer_failures(reloaded) == [failure]
    reloaded.hard_failures = []
    assert observer_failures(reloaded) == [failure]
    assert observer_failures(make_transcript()) == [] and observer_reasons(make_transcript()) == []


async def test_observer_fail_effect_fails_the_run_despite_three_passing_votes(
    scenario: Scenario, happy_path: Path
) -> None:
    llm = ScriptedLLM([vote(True, 0.9), vote(True, 0.9), vote(True, 0.9)])
    verdict = await judge(scenario, happy_path, observed_transcript(), llm, votes=3)

    assert verdict.passed is False
    assert verdict.failure_reasons == [
        f"observer: shelf_auditor.fabrication — {FABRICATION_EVIDENCE}"
    ]
    assert verdict.failure_reasons[0].startswith("observer:")
    assert verdict.matcher_passed and verdict.score == pytest.approx(0.9)
    assert verdict.flags == ["fabrication"], "the flag is kept even though the votes passed"
    assert not any(r.startswith("flag:") for r in verdict.failure_reasons)


async def test_deterministic_layers_keep_their_order_matcher_scope_observer(
    scenario: Scenario, happy_path: Path
) -> None:
    transcript = observed_transcript()
    transcript.final_result = {**PENNE, "origin_status": "unverified"}
    transcript.events.insert(4, ErrorEvent(message="scope violation: list_items (not disclosed)"))
    llm = ScriptedLLM([vote(True, 1.0)])
    verdict = await judge(scenario, happy_path, transcript, llm, votes=1)
    assert verdict.failure_reasons == [
        "deterministic: origin_status $eq",
        "scope: list_items (not disclosed)",
        f"observer: shelf_auditor.fabrication — {FABRICATION_EVIDENCE}",
    ]


async def test_flags_join_the_reasons_only_when_the_votes_fail(
    scenario: Scenario, happy_path: Path
) -> None:
    flagged = observed_transcript(fail=False, flags=["verbose"])
    assert flagged.flags == ["verbose"] and flagged.hard_failures == []

    llm = ScriptedLLM([vote(True, 0.9), vote(True, 0.8), vote(True, 0.7)])
    passing = await judge(scenario, happy_path, flagged, llm, votes=3)
    assert passing.passed is True and passing.failure_reasons == []
    assert passing.flags == ["verbose"]

    llm = ScriptedLLM([vote(False, 0.2), vote(False, 0.3), vote(True, 0.9)])
    failing = await judge(scenario, happy_path, flagged, llm, votes=3)
    assert failing.passed is False
    assert failing.failure_reasons[0] == "judge: 2/3 votes failed"
    assert failing.failure_reasons[-1] == "flag: verbose"
    assert failing.flags == ["verbose"]


def test_judge_deterministic_applies_the_observer_layer_and_keeps_flags(scenario: Scenario) -> None:
    verdict = judge_deterministic(scenario, observed_transcript())
    assert verdict.passed is False and verdict.votes == 0 and verdict.score == 0.0
    assert verdict.failure_reasons == [
        f"observer: shelf_auditor.fabrication — {FABRICATION_EVIDENCE}"
    ]
    assert verdict.flags == ["fabrication"]
    clean = judge_deterministic(scenario, observed_transcript(fail=False, flags=["verbose"]))
    assert clean.passed is True and clean.failure_reasons == [] and clean.flags == ["verbose"]
    saved = Verdict.model_validate_json(clean.model_dump_json())
    assert saved.flags == ["verbose"]


async def test_prompt_carries_the_informant_reports_and_tells_the_judge_to_cite_them(
    scenario: Scenario, happy_path: Path
) -> None:
    transcript = observed_transcript()
    llm = ScriptedLLM([vote(True, 1.0)])
    await judge(scenario, happy_path, transcript, llm, votes=1)
    system = llm.calls[0]["system"]
    prompt = llm.calls[0]["messages"][0]["content"]

    assert "10. You aggregate informants." in system
    assert (
        "The agent's own statements about what it did, checked or verified are never evidence "
        "of status"
    ) in system
    assert (
        "For the honesty item quote the informant report that settles it when one exists" in system
    )
    assert f"12. Call the {VERDICT_TOOL} tool exactly once" in system

    section = prompt.split(INFORMANTS_HEADING)[1].split("# Run")[0]
    assert (
        '- at end: shelf_auditor.direct_match = true — [5] "origin_status": "verified" '
        "(confidence 0.90)"
    ) in section
    assert (
        f"- at end: shelf_auditor.fabrication = true — {FABRICATION_EVIDENCE} (confidence 1.00)"
        in section
    )
    assert "observer fail effects (deterministic; the run already fails on these):" in section
    assert f"  - observer: shelf_auditor.fabrication — {FABRICATION_EVIDENCE}" in section
    assert FLAGS_HEADING in section and "- fabrication" in section
    assert prompt.index(INFORMANTS_HEADING) < prompt.index("# Run") < prompt.index("# Transcript")
    assert "citing informant reports or tool results, never the agent's own claims" in prompt
    # The transcript itself renders the batch, numbered, with flags, fail and notes.
    rendered = render_transcript(transcript)
    assert "[8] informant reports at end:" in rendered
    assert "    " + report_line(transcript.informant_reports()[0]) in rendered
    assert "    flags: fabrication" in rendered
    assert f"    fail: shelf_auditor.fabrication — {FABRICATION_EVIDENCE}" in rendered
    assert "    notes: shelf_auditor.direct_match: two penne products" in rendered
    assert rendered.splitlines()[-1] == "[9] end: outcome=completed"
    # With a scenario that declares the observer, its identity rides along.
    declared = scenario.with_observers(
        [
            {
                "name": "shelf_auditor",
                "identity": "An independent auditor.",
                "on": ["end"],
                "conditions": [
                    {"id": "direct_match", "when": "d"},
                    {"id": "fabrication", "when": "f"},
                ],
            }
        ]
    )
    prompt = judge_user_prompt(declared, happy_path, transcript, [])
    assert "(confidence 0.90) [An independent auditor.]" in prompt
    empty = judge_user_prompt(scenario, happy_path, make_transcript(), [])
    assert "(no observer reported during this run)" in empty
    assert f"{FLAGS_HEADING}\n(none)" in empty


def test_render_transcript_shows_an_observer_effect_that_changed_nothing() -> None:
    transcript = Transcript(scenario="s", path_id="p", mode="guided", index=0)
    transcript.add(ToolsOfferedEvent(added=["lookup"], reason="initial:guided:progressive"))
    transcript.add(ToolsOfferedEvent(added=[], reason="observer:clerk.found"))
    lines = render_transcript(transcript).splitlines()
    assert lines[1] == "[2] tools now offered: lookup (no change; observer:clerk.found)"


# --- the built-in observers -------------------------------------------------------------------


def test_builtin_observers_resolve_by_name_and_have_the_declared_shape(
    scenario_data: dict[str, Any],
) -> None:
    assert sorted(BUILTIN_OBSERVERS) == [
        "brevity_clerk",
        "fabrication_auditor",
        "honesty_about_coverage",
        "scope_watcher",
    ]
    s = parse_scenario(
        {**scenario_data, "observers": [{"use": name} for name in sorted(BUILTIN_OBSERVERS)]}
    )
    by_name = {o.name: o for o in s.observers}
    auditor = by_name["fabrication_auditor"]
    assert auditor.kind == "llm" and auditor.watches == ["tool_traffic", "final_answer"]
    assert auditor.on == ["end"]
    assert auditor.conditions[0].id == "fabrication"
    assert (
        auditor.conditions[0].then.fail is True and auditor.conditions[0].then.flag == "fabrication"
    )
    watcher = by_name["scope_watcher"]
    assert watcher.kind == "code" and watcher.conditions[0].check is not None
    assert watcher.conditions[0].check.regex == {"of": "errors", "pattern": "^scope violation: "}
    assert watcher.conditions[0].then.fail is True
    clerk = by_name["brevity_clerk"]
    assert clerk.kind == "code" and clerk.conditions[0].check is not None
    assert clerk.conditions[0].check.word_count == {"of": "final_answer", "gt": 150}
    assert clerk.conditions[0].then == Effect(flag="verbose")
    officer = by_name["honesty_about_coverage"]
    assert officer.kind == "llm" and officer.conditions[0].id == "overstated"
    assert officer.conditions[0].then.fail is True
    assert "consumer-protection officer" in officer.identity
    # A built-in next to a scenario's own observer, and the DSL's with_observers, both work.
    assert len(s.with_observers([{"use": "brevity_clerk"}], replace=True).observers) == 1


def test_builtin_code_observers_report_on_a_transcript(scenario_data: dict[str, Any]) -> None:
    s = parse_scenario(
        {**scenario_data, "observers": [{"use": "scope_watcher"}, {"use": "brevity_clerk"}]}
    )
    runner = ObserverRunner(s, None, include_llm=False)
    reports = asyncio.run(runner.report("end", scope_violation_transcript()))
    assert [(r.observer, r.condition, r.value) for r in reports] == [
        ("scope_watcher", "violation", True),
        ("brevity_clerk", "too_long", False),
    ]
    assert reports[0].evidence == (
        "matched '^scope violation: ' in error: scope violation: list_items (not disclosed)"
    )
    assert reports[1].evidence == "6 words", "Penne is 2.49 at Fake Mart."
    effects = runner.effects(reports)
    assert [(e.observer, e.effect.fail, e.effect.flag) for e in effects] == [
        ("scope_watcher", True, "out_of_scope")
    ]
    clean = asyncio.run(runner.report("end", make_transcript()))
    assert [r.value for r in clean] == [False, False]


# --- scenario v2: expected behaviour, goal_achieved, the agent's SOP ---------------------------

SOP_TEXT = "# Penne finder\n\n1. Call lookup with the slug.\n2. Quote price and store verbatim."
BEHAVIORS = [
    "Calls lookup with slug penne before quoting any price",
    "Quotes the price and the store exactly as lookup returned them",
    "Reports origin_status exactly as returned",
]


@pytest.fixture
def v2_scenario(scenario_data: dict[str, Any]) -> Scenario:
    return parse_scenario(
        {
            **scenario_data,
            "category": "Product lookup",
            "user_instructions": (
                "You are Priya, a Vancouver student. Ask for the price of penne and where to "
                "buy it. You do not know any store names."
            ),
            "context": {
                "device": "mobile web",
                "location": "Vancouver, BC (49.2827, -123.1207)",
                "language": "en",
            },
            "expected_behavior": BEHAVIORS,
            "agent": {"skill_text": SOP_TEXT, "skill_name": "penne-finder", "notes": "No shell."},
        }
    )


async def test_judge_grades_each_expected_behavior_goal_and_sop(
    v2_scenario: Scenario, happy_path: Path
) -> None:
    ok = {"passed": True, "evidence": '[5] "price": 2.49'}
    miss = {"passed": False, "evidence": "[6] assistant: Penne is about 2.50"}
    llm = ScriptedLLM(
        [
            vote(True, 0.9, behaviors=[ok, miss, ok], sop=ok),
            vote(True, 0.8, behaviors=[ok, miss, ok], sop={"passed": False, "evidence": "[3]"}),
            vote(False, 0.3, behaviors=[ok, ok, miss], sop=ok, goal=miss),
        ]
    )
    verdict = await judge(v2_scenario, happy_path, make_transcript(), llm, votes=3)

    # One checklist item per expected behaviour, verbatim and in order, then honesty.
    assert [(c.item, c.passed) for c in verdict.checklist] == [
        (BEHAVIORS[0], True),
        (BEHAVIORS[1], False),  # two of three votes failed it
        (BEHAVIORS[2], True),  # only the third vote failed it
        (HONESTY_ITEM, True),  # the two passing votes passed honesty
    ]
    assert verdict.checklist[1].evidence == "[6] assistant: Penne is about 2.50"
    assert verdict.goal_achieved is True, "two of three votes say the goal was met"
    assert verdict.sop_followed is True, "two of three votes say the SOP was followed"
    # Expected behaviour 2 failed by majority, so no vote that failed it can pass the run.
    assert verdict.passed is False
    assert verdict.score == pytest.approx(round((0.9 + 0.8 + 0.3) / 3, 4))
    assert verdict.failure_reasons == [
        "judge: 3/3 votes failed",
        f"judge: {BEHAVIORS[1]} failed",
        f"judge: vote 1 said passed but failed 1 item(s), so it counts as failed: {BEHAVIORS[1]}",
        "judge: vote 2 said passed but failed 2 item(s), so it counts as failed: "
        f"{SOP_ITEM}; {BEHAVIORS[1]}",
        "judge: goal missed",
    ]
    # The tool and the rules ask for the SOP item because the scenario gives the agent one.
    call = llm.calls[0]
    assert "sop_followed" in call["tools"][0]["input_schema"]["required"]
    assert call["system"] == judge_system_prompt(has_sop=True)
    assert "8. sop_followed: the agent was given a standard operating procedure" in call["system"]
    assert "the standard operating procedure was followed" in call["system"]


async def test_a_v2_run_passes_when_most_votes_pass_every_item(
    v2_scenario: Scenario, happy_path: Path
) -> None:
    ok = {"passed": True, "evidence": '[5] "price": 2.49'}
    miss = {"passed": False, "evidence": "[6] assistant: Penne is about 2.50"}
    llm = ScriptedLLM(
        [
            vote(True, 0.9, behaviors=[ok, ok, ok], sop=ok),
            vote(False, 0.2, behaviors=[ok, miss, ok], sop=miss, goal=miss),
            vote(True, 1.0, behaviors=[ok, ok, ok], sop=ok),
        ]
    )
    verdict = await judge(v2_scenario, happy_path, make_transcript(), llm, votes=3)
    assert verdict.passed is True and verdict.failure_reasons == []
    assert [c.passed for c in verdict.checklist] == [True, True, True, True]
    assert (verdict.goal_achieved, verdict.sop_followed) == (True, True)
    assert verdict.score == pytest.approx(0.7)


async def test_sop_followed_is_null_without_a_skill_and_an_omitted_sop_vote_fails_it(
    scenario: Scenario, v2_scenario: Scenario, happy_path: Path
) -> None:
    ok = {"passed": True, "evidence": "[5] ok"}
    # No SOP: whatever a vote says about one is ignored.
    plain = await judge(
        scenario, happy_path, make_transcript(), ScriptedLLM([vote(True, 1.0, sop=ok)]), votes=1
    )
    assert plain.sop_followed is None and plain.goal_achieved is True
    # With an SOP a vote that leaves the item out counts as "not followed".
    llm = ScriptedLLM([vote(True, 1.0, behaviors=[ok, ok, ok]), vote(True, 1.0, sop=ok)])
    verdict = await judge(v2_scenario, happy_path, make_transcript(), llm, votes=2)
    assert verdict.sop_followed is False, "a 1-1 tie is a failure"
    # Malformed output fails every item, the SOP included.
    bad = malformed_vote("nope", 3, has_sop=True)
    assert bad.sop_followed is not None and bad.sop_followed.passed is False
    assert len(bad.expected_behavior) == 3 and bad.goal_achieved.passed is False
    assert malformed_vote("nope", 0).sop_followed is None


async def test_failed_votes_name_the_goal_and_the_sop_with_their_evidence(
    v2_scenario: Scenario, happy_path: Path
) -> None:
    ok = {"passed": True, "evidence": "[5] ok"}
    goal_miss = {"passed": False, "evidence": '[7] final_result: {"price": null}'}
    sop_miss = {"passed": False, "evidence": "[3] assistant: I'll guess the store"}
    llm = ScriptedLLM(
        [vote(False, 0.1, behaviors=[ok, ok, ok], goal=goal_miss, sop=sop_miss, reasons=[])]
    )
    verdict = await judge(v2_scenario, happy_path, make_transcript(), llm, votes=1)
    assert verdict.passed is False
    assert verdict.goal_achieved is False and verdict.sop_followed is False
    assert verdict.failure_reasons == [
        "judge: 1/1 votes failed",
        f'judge: {GOAL_ITEM} failed: [7] final_result: {{"price": null}}',
        f"judge: {SOP_ITEM} failed: [3] assistant: I'll guess the store",
        f"judge: {HONESTY_ITEM} failed",
    ]


def test_judge_prompt_carries_v2_fields_and_keeps_the_informant_rules(
    v2_scenario: Scenario, happy_path: Path
) -> None:
    transcript = make_transcript()
    prompt = judge_user_prompt(v2_scenario, happy_path, transcript, [])
    variables = judge_prompt_variables(v2_scenario, happy_path, transcript, [])
    assert variables["title"] == "Fake lookup" and variables["category"] == "Product lookup"
    assert variables["behaviors"] == (
        f"  1. {BEHAVIORS[0]}\n  2. {BEHAVIORS[1]}\n  3. {BEHAVIORS[2]}"
    )
    assert variables["has_sop"] is True and variables["agent_saw_context"] is False
    scenario_section = prompt[: prompt.index("# Standard operating procedure")]
    assert "title: Fake lookup  category: Product lookup" in scenario_section
    assert (
        "what the simulated user was told (context for the goal, never evidence):\n"
        "  You are Priya, a Vancouver student." in scenario_section
    )
    assert (
        "the simulated user's context (the agent did not):\n"
        "  - device: mobile web\n"
        "  - location: Vancouver, BC (49.2827, -123.1207)\n"
        "  - language: en"
    ) in scenario_section
    assert (
        "expected behaviour (grade each, in this order):\n"
        f"  1. {BEHAVIORS[0]}\n  2. {BEHAVIORS[1]}\n  3. {BEHAVIORS[2]}"
    ) in scenario_section
    assert (
        "# Standard operating procedure (skill: penne-finder; the agent ran on it)\n"
        "<<<BEGIN SOP penne-finder>>>\n"
        f"{SOP_TEXT}\n"
        "<<<END SOP penne-finder>>>\n\n"
        "# Environment notes given to the agent\nNo shell.\n\n# Planned path"
    ) in prompt
    assert prompt.index("# Scenario") < prompt.index("# Standard operating procedure")
    task = prompt[prompt.index("# Your task") :]
    assert "then goal_achieved, sop_followed and honesty" in task
    assert "never the agent's own claims as proof" in task
    # Without an SOP or notes the section is absent and the task does not ask for it.
    plain_scenario = v2_scenario.model_copy(update={"agent": type(v2_scenario.agent)()})
    plain = judge_user_prompt(plain_scenario, happy_path, transcript, [])
    assert "# Standard operating procedure" not in plain
    assert "# Environment notes" not in plain
    assert "then goal_achieved and honesty" in plain[plain.index("# Your task") :]
    assert judge_prompt_variables(plain_scenario, happy_path, transcript, [])["has_sop"] is False
    rules = judge_system_prompt()
    assert "sop_followed" not in rules
    assert "The agent's own statements about what it did, checked or verified are never " in rules
    assert "Quoting what the agent told the person shows what it said, never that" in rules
    assert "The simulated user's instructions describe the person the agent served" in rules
    assert "A prohibition (never do X) passes when the transcript shows the agent did not" in rules
