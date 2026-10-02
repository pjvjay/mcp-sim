from __future__ import annotations

import json
from typing import Any, cast

import pytest

from mcpsim.agent import (
    DISCOVER_TOOL_NAME,
    LiveRun,
    SimulatedUser,
    StepReferenceError,
    agent_prompt_sections,
    build_agent_system_prompt,
    extract_final_result,
    readable_sketch,
    resolve_arguments,
    resolve_reference,
    run_path,
    steps_section,
    tool_result_block,
)
from mcpsim.judge import scope_violations
from mcpsim.llm import Usage, estimate_cost_usd
from mcpsim.mcpclient import ToolResult
from mcpsim.observers import REPORT_TOOL
from mcpsim.plan import Mode, Path, Step, StepReference
from mcpsim.scenario import Scenario, parse_scenario
from mcpsim.transcript import (
    AssistantEvent,
    ErrorEvent,
    FinalResultEvent,
    GoalEnabledEvent,
    InformantReportEvent,
    ToolsOfferedEvent,
    Transcript,
    UserEvent,
)
from tests.conftest import open_session
from tests.fake_llm import ScriptedLLM, structured_response, text_response, tool_use_response

HAPPY_SEQUENCE = [
    "system",
    "tools_offered",
    "user",
    "assistant",
    "tool_call",
    "tool_result",
    "assistant",
    "tool_call",
    "tool_result",
    "assistant",
    "final_result",
    "usage",
    "end",
]

PENNE = {"slug": "penne", "price": 2.49, "store": "Fake Mart", "origin_status": "verified"}
FINAL_TEXT = (
    "Penne is $2.49 at Fake Mart and its origin is verified.\n\n"
    "```json final_result\n" + json.dumps(PENNE) + "\n```"
)


def two_tool_path() -> Path:
    return Path(
        id="happy",
        kind="happy",
        title="Look up penne, then list the catalogue",
        rationale="The direct route.",
        steps=[
            Step(
                intent="Look up penne",
                tool="lookup",
                arguments_sketch={"slug": "penne"},
                success_looks_like="a price, a store and origin_status",
            ),
            Step(
                intent="List the first page of products",
                tool="list_items",
                arguments_sketch={"cursor": 0, "limit": 2},
            ),
        ],
        checkpoints=["price in the final answer equals what lookup returned"],
    )


def opening() -> Any:
    return text_response("Hi, I need the price and store of penne, no guessing please.")


@pytest.fixture
def scenario(scenario_data: dict[str, Any]) -> Scenario:
    return parse_scenario(scenario_data, source="fixture")


def with_budgets(scenario: Scenario, **budgets: Any) -> Scenario:
    return scenario.model_copy(update={"budgets": scenario.budgets.model_copy(update=budgets)})


async def run(
    scenario: Scenario,
    llm: ScriptedLLM | None,
    *,
    path: Path | None = None,
    mode: Mode = "guided",
    index: int = 0,
    dry_run: bool = False,
) -> Transcript:
    async with open_session() as session:
        return await run_path(
            scenario, path or two_tool_path(), mode, index, session, llm, dry_run=dry_run
        )


# ------------------------------------------------------------------------------------------
# the loop


async def test_two_tool_happy_path_produces_the_exact_event_sequence(scenario: Scenario) -> None:
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response("lookup", {"slug": "penne"}, tool_use_id="toolu_1", text="Looking."),
            tool_use_response("list_items", {"cursor": 0, "limit": 2}, tool_use_id="toolu_2"),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(scenario, llm)

    assert t.kinds() == HAPPY_SEQUENCE
    assert t.outcome == "completed" and t.reason == "final answer delivered"
    assert t.final_result == PENNE
    assert (t.scenario, t.path_id, t.mode, t.index) == ("fake-lookup", "happy", "guided", 0)
    assert llm.remaining == 0 and len(llm.calls) == 4

    # The simulated user spoke first, in its own call, without tools.
    user_call, first_agent_call, second_agent_call, final_call = llm.calls
    assert user_call["tools"] is None
    assert "Who you are" in user_call["system"]
    assert t.events[2].text.startswith("Hi, I need the price")

    # The agent saw the catalog's tool schemas and the opening message.
    assert {tool["name"] for tool in first_agent_call["tools"]} == {
        "lookup",
        "fail",
        "list_items",
        "echo",
        "expensive_report",
    }
    assert first_agent_call["tool_choice"] is None
    assert first_agent_call["model"] == scenario.models.agent
    assert first_agent_call["messages"] == [{"role": "user", "content": t.events[2].text}]

    # The first tool result went back as a tool_result block keyed by the tool_use id.
    assert second_agent_call["messages"][1]["role"] == "assistant"
    returned = second_agent_call["messages"][2]
    assert returned["role"] == "user"
    block = returned["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "toolu_1"
    assert json.loads(block["content"]) == PENNE
    assert "is_error" not in block

    # Transcript events carry the call and the normalised result.
    calls = t.tool_calls()
    assert [(c.name, c.arguments, c.tool_use_id) for c in calls] == [
        ("lookup", {"slug": "penne"}, "toolu_1"),
        ("list_items", {"cursor": 0, "limit": 2}, "toolu_2"),
    ]
    results = t.tool_results()
    assert results[0].structured == PENNE and results[0].is_error is False
    assert isinstance(results[1].structured, dict) and results[1].structured["total"] == 5
    assert results[1].tool_use_id == "toolu_2"
    assert t.events[3].text == "Looking." and t.events[3].stop_reason == "tool_use"
    assert t.events[3].tool_uses[0]["name"] == "lookup"
    assert t.events[10].raw == json.dumps(PENNE)

    # Usage is summed per model and costed from the rate table.
    assert t.usage == {scenario.models.agent: Usage(input_tokens=40, output_tokens=20)}
    assert t.cost_usd == pytest.approx(
        estimate_cost_usd(scenario.models.agent, Usage(input_tokens=40, output_tokens=20))
    )
    assert t.cost_usd > 0
    assert t.events[11].per_model == t.usage and t.events[11].estimate is True


async def test_denied_tools_are_never_offered_even_when_the_agent_discovers_the_catalog(
    scenario_data: dict[str, Any],
) -> None:
    scoped = parse_scenario(
        {**scenario_data, "tools": {"deny": ["fail", "expensive_*"]}}, source="fixture"
    )
    llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(scoped, llm)  # no catalog passed: run_path discovers and filters it itself
    assert t.outcome == "completed"
    assert [tool["name"] for tool in llm.calls[1]["tools"]] == ["lookup", "list_items", "echo"]


async def test_error_tool_result_is_passed_back_and_seen_next_turn(scenario: Scenario) -> None:
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response("fail", {"reason": "bad slug"}, tool_use_id="toolu_err"),
            tool_use_response("lookup", {"slug": "penne"}, tool_use_id="toolu_ok"),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(scenario, llm)

    assert t.outcome == "completed"
    results = t.tool_results()
    assert results[0].name == "fail" and results[0].is_error is True
    assert "fail tool invoked: bad slug" in results[0].text
    assert results[1].is_error is False

    seen = llm.calls[2]["messages"][-1]["content"][0]
    assert seen["type"] == "tool_result" and seen["tool_use_id"] == "toolu_err"
    assert seen["is_error"] is True
    assert "fail tool invoked: bad slug" in seen["content"]


async def test_text_only_tool_result_is_passed_through(scenario: Scenario) -> None:
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response("echo", {"text": "HELLO"}, tool_use_id="toolu_echo"),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(scenario, llm)
    assert t.outcome == "completed"
    assert llm.calls[2]["messages"][-1]["content"][0]["content"] == "HELLO"
    assert t.tool_results()[0].structured is None and t.tool_results()[0].text == "HELLO"


async def test_max_tool_calls_budget_stops_the_run(scenario: Scenario) -> None:
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response("lookup", {"slug": "penne"}, tool_use_id="toolu_1"),
            tool_use_response("lookup", {"slug": "basil"}, tool_use_id="toolu_2"),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(with_budgets(scenario, max_tool_calls=1), llm)

    assert t.outcome == "budget_exceeded"
    assert "max_tool_calls=1" in t.reason and "lookup" in t.reason
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "assistant",
        "tool_call",
        "tool_result",
        "assistant",
        "usage",
        "end",
    ]
    assert len(t.tool_calls()) == 1
    assert t.final_result is None
    assert llm.remaining == 1  # the final answer was never requested


async def test_max_turns_budget_stops_the_run(scenario: Scenario) -> None:
    llm = ScriptedLLM(
        [
            opening(),
            text_response("Do you want the cheapest penne or any penne?"),
            text_response("Any penne is fine."),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(with_budgets(scenario, max_turns=1), llm)
    assert t.outcome == "budget_exceeded" and "max_turns=1" in t.reason
    assert t.kinds() == ["system", "tools_offered", "user", "assistant", "user", "usage", "end"]
    assert llm.remaining == 1


async def test_max_cost_budget_stops_the_run(scenario: Scenario) -> None:
    llm = ScriptedLLM([opening(), tool_use_response("lookup", {"slug": "penne"})])
    t = await run(with_budgets(scenario, max_cost_usd=1e-9), llm)
    assert t.outcome == "budget_exceeded" and "max_cost_usd=1e-09" in t.reason
    # The opening user turn alone blows a one-nanodollar budget: no agent turn happens.
    assert t.kinds() == ["system", "tools_offered", "user", "usage", "end"]
    assert t.cost_usd > 1e-9


async def test_malformed_final_block_gives_none_and_an_error_event(scenario: Scenario) -> None:
    llm = ScriptedLLM(
        [
            opening(),
            text_response('Here you go.\n```json final_result\n{"slug": penne,}\n```'),
        ]
    )
    t = await run(scenario, llm)
    assert t.outcome == "completed"
    assert t.final_result is None
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "assistant",
        "final_result",
        "error",
        "usage",
        "end",
    ]
    final_event, error_event = t.events[4], t.events[5]
    assert final_event.parsed is None and final_event.raw == '{"slug": penne,}'
    assert "not valid JSON" in error_event.message
    assert "not valid JSON" in t.reason


async def test_clarifying_question_is_answered_by_the_simulated_user(scenario: Scenario) -> None:
    llm = ScriptedLLM(
        [
            opening(),
            text_response("Which store do you prefer?"),
            text_response("I don't mind, just use what the server has."),
            tool_use_response("lookup", {"slug": "penne"}, tool_use_id="toolu_1"),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(scenario, llm)
    assert t.outcome == "completed"
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool_call",
        "tool_result",
        "assistant",
        "final_result",
        "usage",
        "end",
    ]
    user_reply_call = llm.calls[2]
    assert user_reply_call["tools"] is None
    assert user_reply_call["messages"][-1] == {
        "role": "user",
        "content": "Which store do you prefer?",
    }
    assert t.events[4].text == "I don't mind, just use what the server has."
    agent_next_call = llm.calls[3]
    assert agent_next_call["messages"][-1] == {
        "role": "user",
        "content": "I don't mind, just use what the server has.",
    }


async def test_session_exception_becomes_an_error_outcome(scenario: Scenario) -> None:
    llm = ScriptedLLM(
        [opening(), tool_use_response("lookup", {"slug": "penne"}, tool_use_id="toolu_1")]
    )

    async def boom(name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        raise ConnectionError("pipe closed")

    async with open_session() as session:
        session.call_tool = boom  # type: ignore[method-assign]
        t = await run_path(scenario, two_tool_path(), "guided", 0, session, llm)
    assert t.outcome == "error"
    assert t.reason == "ConnectionError: pipe closed"
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "assistant",
        "tool_call",
        "usage",
        "end",
    ]


async def test_llm_exception_becomes_an_error_outcome(scenario: Scenario) -> None:
    llm = ScriptedLLM([opening()])  # nothing scripted for the agent's turn
    t = await run(scenario, llm)
    assert t.outcome == "error" and t.reason.startswith("RuntimeError: ScriptedLLM")


async def test_llm_is_required_unless_dry_run(scenario: Scenario) -> None:
    # The guard fires before the session is touched, so no server is needed here (and raising
    # inside an open in-memory session would surface as an anyio ExceptionGroup instead).
    with pytest.raises(ValueError, match="dry_run"):
        await run_path(scenario, two_tool_path(), "guided", 0, cast(Any, None), None)


async def test_system_event_records_prompts_models_and_mode(scenario: Scenario) -> None:
    llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(scenario, llm, mode="free", index=2)
    system = t.events[0]
    assert system.kind == "system"
    assert (system.scenario, system.path_id, system.mode, system.index) == (
        "fake-lookup",
        "happy",
        "free",
        2,
    )
    assert system.models == {"agent": scenario.models.agent, "user": scenario.models.agent}
    assert system.prompts["agent"] == build_agent_system_prompt(scenario, two_tool_path(), "free")
    assert system.prompts["agent"] == llm.calls[1]["system"]
    assert system.prompts["user"] == llm.calls[0]["system"]
    assert t.stem == "happy-free-2"


async def test_simulated_user_falls_back_to_the_goal_when_the_llm_says_nothing(
    scenario: Scenario,
) -> None:
    llm = ScriptedLLM([text_response(""), text_response("")])
    user = SimulatedUser(scenario, llm, "claude-sonnet-5-5")
    text, usage = await user.open()
    assert text == scenario.goal.strip() and usage.input_tokens == 10
    reply, _ = await user.reply("")
    assert reply  # a non-empty fallback so the agent is never sent an empty message
    assert llm.calls[1]["messages"][-1]["content"] == "(the assistant sent an empty message)"
    assert llm.calls[1]["max_tokens"] == 512


# ------------------------------------------------------------------------------------------
# dry run


async def test_dry_run_calls_every_step_and_synthesises_final_result(scenario: Scenario) -> None:
    path = Path(
        id="dry",
        kind="happy",
        title="dry",
        steps=[
            Step(intent="look", tool="lookup", arguments_sketch={"slug": "penne"}),
            Step(intent="break", tool="fail", arguments_sketch={"reason": "x"}),
            Step(intent="echo", tool="echo", arguments_sketch={"text": "hi"}),
            Step(intent="think", tool=None),
            Step(intent="page", tool="list_items", arguments_sketch={"cursor": 0, "limit": 2}),
        ],
    )
    t = await run(scenario, None, path=path, dry_run=True)
    assert t.outcome == "completed" and "dry run" in t.reason
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "final_result",
        "usage",
        "end",
    ]
    assert [c.name for c in t.tool_calls()] == ["lookup", "fail", "echo", "list_items"]
    assert [r.is_error for r in t.tool_results()] == [False, True, False, False]
    # lookup's result covers slug, price and origin_status of the expected outcome; the
    # list_items page that came last covers none of them.
    assert t.final_result == t.tool_results()[0].structured == PENNE
    assert t.events[2].text == scenario.goal.strip()
    assert t.events[0].models == {"agent": "dry-run", "user": "dry-run"}
    assert t.usage == {} and t.cost_usd == 0.0


async def test_dry_run_without_structured_results_has_null_final_result(
    scenario: Scenario,
) -> None:
    path = Path(
        id="dry",
        kind="happy",
        title="dry",
        steps=[Step(intent="echo", tool="echo", arguments_sketch={"text": "hi"})],
    )
    t = await run(scenario, None, path=path, dry_run=True)
    assert t.outcome == "completed" and t.final_result is None
    assert t.kinds()[-4:] == ["final_result", "error", "usage", "end"]


async def test_dry_run_respects_the_tool_call_budget(scenario: Scenario) -> None:
    t = await run(with_budgets(scenario, max_tool_calls=1), None, dry_run=True)
    assert t.outcome == "budget_exceeded" and "max_tool_calls=1" in t.reason
    assert len(t.tool_calls()) == 1


def ref(from_step: int, path: str) -> dict[str, Any]:
    return {"$from_step": from_step, "path": path}


async def test_dry_run_resolves_a_from_step_reference_end_to_end(scenario: Scenario) -> None:
    path = Path(
        id="ref",
        kind="happy",
        title="look up, then echo the store the lookup returned",
        steps=[
            Step(intent="look up penne", tool="lookup", arguments_sketch={"slug": "penne"}),
            Step(intent="echo its store", tool="echo", arguments_sketch={"text": ref(1, "store")}),
        ],
    )
    t = await run(scenario, None, path=path, dry_run=True)

    assert t.outcome == "completed"
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "final_result",
        "usage",
        "end",
    ]
    lookup_call, echo_call = t.tool_calls()
    assert lookup_call.arguments == {"slug": "penne"}
    assert echo_call.arguments == {"text": "Fake Mart"}  # the reference, resolved
    echoed = t.tool_results()[1]
    assert echoed.name == "echo" and echoed.is_error is False
    assert echoed.text == "Fake Mart"  # the server echoed the looked-up value
    assert t.final_result == PENNE  # echo has no structured content; lookup's stands


async def test_dry_run_skips_a_step_whose_reference_cannot_be_resolved(
    scenario: Scenario,
) -> None:
    path = Path(
        id="ref",
        kind="happy",
        title="bad references",
        steps=[
            Step(intent="look", tool="lookup", arguments_sketch={"slug": "penne"}),
            Step(intent="missing key", tool="echo", arguments_sketch={"text": ref(1, "nope")}),
            Step(intent="text-only step", tool="echo", arguments_sketch={"text": "hi"}),
            Step(intent="from text-only", tool="echo", arguments_sketch={"text": ref(3, "x")}),
            Step(intent="forward", tool="echo", arguments_sketch={"text": ref(9, "x")}),
            Step(intent="think", tool=None),
            Step(intent="from no-tool", tool="echo", arguments_sketch={"text": ref(6, "x")}),
            Step(intent="still runs", tool="list_items", arguments_sketch={"cursor": 0}),
        ],
    )
    t = await run(scenario, None, path=path, dry_run=True)

    assert t.outcome == "completed"
    assert [c.name for c in t.tool_calls()] == ["lookup", "echo", "list_items"]
    errors = [e.message for e in t.events if e.kind == "error"]
    assert len(errors) == 4
    assert errors[0].startswith("dry run: step 2 (echo) skipped")
    assert "argument 'text': path 'nope' is missing from step 1's result" in errors[0]
    assert "step 3 (echo) returned text only, no structured content" in errors[1]
    assert "step 9 has not run" in errors[2]
    assert "step 6 called no tool" in errors[3]
    assert "4 step(s) skipped over unresolved references" in t.reason
    assert t.final_result == PENNE, "lookup's result covers the expected keys; the page does not"


async def test_dry_run_treats_an_expected_error_as_the_planned_outcome(
    scenario: Scenario,
) -> None:
    path = Path(
        id="recovery",
        kind="recovery",
        title="provoke the rejection, then correct",
        steps=[
            Step(
                intent="misspelt slug",
                tool="lookup",
                arguments_sketch={"slug": "pene"},
                expect_error=True,
                success_looks_like="unknown slug with suggestions",
            ),
            Step(intent="corrected", tool="lookup", arguments_sketch={"slug": "penne"}),
            Step(
                intent="should fail but will not",
                tool="echo",
                arguments_sketch={"text": "ok"},
                expect_error=True,
            ),
            Step(intent="unplanned failure", tool="fail", arguments_sketch={"reason": "x"}),
        ],
    )
    t = await run(scenario, None, path=path, dry_run=True)

    assert t.outcome == "completed"
    assert [r.is_error for r in t.tool_results()] == [True, False, False, True]
    assert "1 expected error(s) returned as planned" in t.reason
    assert "1 unexpected error result(s)" in t.reason
    errors = [e.message for e in t.events if e.kind == "error"]
    assert errors == [
        "dry run: step 3 (echo) was expected to be rejected by the server but succeeded"
    ]
    assert t.final_result == PENNE  # the corrected call's result


def test_resolve_reference_handles_scalars_wildcards_and_failures() -> None:
    lookup = ToolResult(name="lookup", is_error=False, structured=PENNE)
    page = ToolResult(
        name="list_items",
        is_error=False,
        structured={"items": [{"slug": "a", "n": 1}, {"slug": "b"}], "next_cursor": None},
    )
    results: dict[str, Any] = {1: lookup, 2: page, 3: None}
    assert resolve_reference(StepReference.model_validate(ref(1, "store")), results) == "Fake Mart"
    assert resolve_reference(StepReference.model_validate(ref(2, "items[*].slug")), results) == [
        "a",
        "b",
    ]
    assert resolve_reference(StepReference.model_validate(ref(2, "items[1].slug")), results) == "b"
    assert resolve_reference(StepReference.model_validate(ref(2, "next_cursor")), results) is None

    with pytest.raises(StepReferenceError, match="1 of 2 element"):
        resolve_reference(StepReference.model_validate(ref(2, "items[*].n")), results)
    with pytest.raises(StepReferenceError, match="is not an array"):
        resolve_reference(StepReference.model_validate(ref(1, "store[*]")), results)
    with pytest.raises(StepReferenceError, match="called no tool"):
        resolve_reference(StepReference.model_validate(ref(3, "x")), results)
    with pytest.raises(StepReferenceError, match="has not run"):
        resolve_reference(StepReference.model_validate(ref(4, "x")), results)
    with pytest.raises(StepReferenceError, match="at most one"):
        resolve_reference(StepReference.model_validate(ref(2, "items[*].x[*]")), results)

    step = Step(
        intent="x",
        tool="echo",
        arguments_sketch={"text": ref(1, "slug"), "literal": {"not": "a reference"}, "n": 2},
    )
    assert resolve_arguments(step, results) == {
        "text": "penne",
        "literal": {"not": "a reference"},
        "n": 2,
    }
    with pytest.raises(StepReferenceError, match="argument 'text'"):
        resolve_arguments(
            Step(intent="x", tool="echo", arguments_sketch={"text": ref(9, "a")}), results
        )
    with pytest.raises(ValueError, match="malformed step reference"):
        resolve_arguments(
            Step(intent="x", tool="echo", arguments_sketch={"t": {"$from_step": 1}}), results
        )


# ------------------------------------------------------------------------------------------
# prompts


def test_guided_and_free_prompts_differ_exactly_by_the_steps_section(scenario: Scenario) -> None:
    path = two_tool_path()
    guided = agent_prompt_sections(scenario, path, "guided")
    free = agent_prompt_sections(scenario, path, "free")
    assert len(guided) == len(free) + 1
    assert [s for s in guided if s not in free] == [steps_section(path)]
    assert [s for s in free if s not in guided] == []
    # Order is preserved: the steps sit before the answer contract.
    assert guided.index(steps_section(path)) == len(guided) - 2
    assert build_agent_system_prompt(scenario, path, "guided") == "\n\n".join(guided)
    assert steps_section(path) not in build_agent_system_prompt(scenario, path, "free")


def test_prompt_carries_role_goal_instructions_and_contract_fields(scenario: Scenario) -> None:
    prompt = build_agent_system_prompt(scenario, two_tool_path(), "free")
    assert scenario.role.strip() in prompt and scenario.goal.strip() in prompt
    for item in scenario.instructions:
        assert f"- {item}" in prompt
    assert "```json final_result" in prompt
    assert "`slug`, `price`, `origin_status`" in prompt


def test_steps_section_lists_tools_and_sketches() -> None:
    section = steps_section(two_tool_path())
    assert section.startswith("## Suggested approach")
    assert '1. Look up penne (tool: lookup, arguments roughly {"slug": "penne"})' in section
    assert "success looks like: a price, a store and origin_status" in section
    assert "2. List the first page of products (tool: list_items" in section
    assert "EXPECT AN ERROR" not in section


def test_steps_section_renders_expect_error_and_references_readably() -> None:
    path = Path(
        id="recovery",
        kind="recovery",
        title="t",
        steps=[
            Step(
                intent="Send a misspelt slug",
                tool="lookup",
                arguments_sketch={"slug": "pene"},
                expect_error=True,
                success_looks_like="an error naming the valid slugs",
            ),
            Step(
                intent="Echo the store of the corrected lookup",
                tool="echo",
                arguments_sketch={"text": {"$from_step": 1, "path": "items[*].store"}},
            ),
        ],
    )
    section = steps_section(path)
    assert "An argument written <from step n: path> means" in section
    assert (
        '1. Send a misspelt slug (tool: lookup, arguments roughly {"slug": "pene"}) — EXPECT AN '
        "ERROR: the server should reject this call; read its message and correct the next call "
        "from it — success looks like: an error naming the valid slugs"
    ) in section
    assert (
        "2. Echo the store of the corrected lookup (tool: echo, arguments roughly "
        '{"text": "<from step 1: items[*].store>"})'
    ) in section
    assert readable_sketch(path.steps[1]) == '{"text": "<from step 1: items[*].store>"}'
    # A half-written reference is shown as the literal it is, never crashes the prompt.
    broken = Step(intent="x", tool="echo", arguments_sketch={"text": {"$from_step": "one"}})
    assert readable_sketch(broken) == '{"text": {"$from_step": "one"}}'


# ------------------------------------------------------------------------------------------
# final_result extraction


def test_extract_named_fence() -> None:
    r = extract_final_result('Done.\n```json final_result\n{"a": 1}\n```\nBye.')
    assert r.found and r.parsed == {"a": 1} and r.error is None and r.raw == '{"a": 1}'


def test_extract_prefers_the_last_named_fence_over_other_json_fences() -> None:
    text = (
        'Example:\n```json\n{"a": 0}\n```\nAnswer:\n```json final_result\n{"a": 1}\n```\n'
        '```json\n{"a": 2}\n```'
    )
    assert extract_final_result(text).parsed == {"a": 1}


def test_extract_plain_json_fence() -> None:
    r = extract_final_result('```json\n{"a": 1, "b": [1, 2]}\n```')
    assert r.found and r.parsed == {"a": 1, "b": [1, 2]}


def test_extract_bare_fence_named_final_result() -> None:
    r = extract_final_result('```final_result\n{"a": 1}\n```')
    assert r.found and r.parsed == {"a": 1}


def test_extract_trailing_bare_object_takes_the_outermost_object() -> None:
    r = extract_final_result('Here it is:\n{"a": {"b": 1}, "c": "x"}')
    assert r.found and r.parsed == {"a": {"b": 1}, "c": "x"}
    assert r.raw == '{"a": {"b": 1}, "c": "x"}'


def test_extract_nothing_in_prose() -> None:
    r = extract_final_result("Which store do you prefer?")
    assert not r.found and r.parsed is None and r.error is None


def test_extract_malformed_fence_is_found_but_unparsed() -> None:
    r = extract_final_result("```json final_result\n{nope}\n```")
    assert r.found and r.parsed is None and r.error is not None
    assert "not valid JSON" in r.error and r.raw == "{nope}"


def test_extract_non_object_is_an_error() -> None:
    r = extract_final_result("```json final_result\n[1, 2]\n```")
    assert r.found and r.parsed is None and r.error is not None
    assert "must be a JSON object" in r.error and "list" in r.error


def test_extract_malformed_trailing_object_line_is_found_but_unparsed() -> None:
    r = extract_final_result('Final:\n{"a": }')
    assert r.found and r.parsed is None and r.error is not None


def test_extract_prose_ending_with_brace_but_no_object_is_not_found() -> None:
    assert not extract_final_result("Set notation: a ∈ {1, 2}").found


# ------------------------------------------------------------------------------------------
# tool_result blocks


def test_tool_result_block_shapes() -> None:
    structured = ToolResult(name="t", is_error=False, structured={"k": "v"}, text="ignored")
    assert tool_result_block("id1", structured) == {
        "type": "tool_result",
        "tool_use_id": "id1",
        "content": '{"k": "v"}',
    }
    text = ToolResult(name="t", is_error=False, text="plain")
    assert tool_result_block("id2", text)["content"] == "plain"
    error = ToolResult(name="t", is_error=True, text="boom")
    assert tool_result_block("id3", error) == {
        "type": "tool_result",
        "tool_use_id": "id3",
        "content": "boom",
        "is_error": True,
    }
    silent = ToolResult(name="t", is_error=True)
    assert tool_result_block("id4", silent)["content"] == "tool returned an error with no message"


# ------------------------------------------------------------------------------------------
# tool disclosure (DESIGN §2 "Tool scoping and disclosure"): which tools, which turn, why

ALL_FAKE_TOOLS = ["lookup", "fail", "list_items", "echo", "expensive_report"]


def scoped(scenario_data: dict[str, Any], **tools: Any) -> Scenario:
    return parse_scenario({**scenario_data, "tools": tools}, source="fixture")


def offered_to(llm: ScriptedLLM, call: int) -> list[str]:
    """The tool names the LLM was offered on its ``call``-th call (call 0 is the user sim)."""
    return [tool["name"] for tool in llm.calls[call]["tools"] or []]


def offered_events(t: Transcript) -> list[tuple[list[str], str]]:
    return [(e.added, e.reason) for e in t.events if isinstance(e, ToolsOfferedEvent)]


def error_messages(t: Transcript) -> list[str]:
    return [e.message for e in t.events if isinstance(e, ErrorEvent)]


async def test_all_disclosure_offers_every_allowed_tool_from_turn_one(scenario: Scenario) -> None:
    llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(scenario, llm)
    assert offered_to(llm, 1) == ALL_FAKE_TOOLS
    assert offered_events(t) == [(ALL_FAKE_TOOLS, "initial:guided:all")]
    assert t.kinds()[:3] == ["system", "tools_offered", "user"]
    assert t.tools_offered() == ALL_FAKE_TOOLS


async def test_plan_disclosure_offers_the_paths_tools_when_guided_and_everything_when_free(
    scenario_data: dict[str, Any],
) -> None:
    planned = scoped(scenario_data, disclosure="plan")
    guided = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(planned, guided, mode="guided")
    assert offered_to(guided, 1) == ["lookup", "list_items"], "two_tool_path's tools, step order"
    assert offered_events(t) == [(["lookup", "list_items"], "initial:guided:plan")]

    free = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(planned, free, mode="free")
    assert offered_to(free, 1) == ALL_FAKE_TOOLS
    assert offered_events(t) == [(ALL_FAKE_TOOLS, "initial:free:plan")]


async def test_progressive_disclosure_offers_the_scored_initial_set_plus_discover_tools(
    scenario_data: dict[str, Any],
) -> None:
    progressive = scoped(scenario_data, disclosure="progressive")
    llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(progressive, llm)
    # lookup is the only tool that shares vocabulary with the goal (name, and price / store /
    # slug / origin_status in its output); the floor of three pads with fail and list_items in
    # catalog order. discover_tools comes last and is the framework's, not the server's.
    assert offered_to(llm, 1) == ["lookup", "fail", "list_items", DISCOVER_TOOL_NAME]
    assert offered_events(t) == [(["lookup", "fail", "list_items"], "initial:guided:progressive")]
    assert t.tools_offered() == ["lookup", "fail", "list_items"]
    assert llm.calls[1]["tools"][-1]["input_schema"]["required"] == ["query"]

    free = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(progressive, free, mode="free")
    assert offered_to(free, 1) == offered_to(llm, 1), "the same starting set in free mode"
    assert offered_events(t) == [(["lookup", "fail", "list_items"], "initial:free:progressive")]


async def test_progressive_initial_globs_replace_the_scoring_and_discover_tool_can_be_off(
    scenario_data: dict[str, Any],
) -> None:
    explicit = scoped(
        scenario_data, disclosure="progressive", initial=["echo", "list_*"], discover_tool=False
    )
    llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(explicit, llm, mode="free")
    assert offered_to(llm, 1) == ["list_items", "echo"], "glob matches in catalog order"
    assert offered_events(t) == [(["list_items", "echo"], "initial:free:progressive")]
    llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(explicit, llm, mode="guided")
    assert offered_to(llm, 1) == ["list_items", "echo", "lookup"], "plus the path's lookup"
    assert offered_events(t) == [
        (["list_items", "echo"], "initial:guided:progressive"),
        (["lookup"], "initial:guided:path"),
    ]


async def test_discover_tools_adds_the_matching_tool_for_the_next_turn_without_a_server_call(
    scenario_data: dict[str, Any],
) -> None:
    progressive = scoped(scenario_data, disclosure="progressive", initial=["lookup"])
    progressive = with_budgets(progressive, max_tool_calls=1)
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response(DISCOVER_TOOL_NAME, {"query": "items"}, tool_use_id="toolu_d"),
            tool_use_response("list_items", {"cursor": 0, "limit": 2}, tool_use_id="toolu_2"),
            text_response(FINAL_TEXT),
        ]
    )
    async with open_session() as session:
        # free mode: in guided mode the path's own tools would already be offered (see
        # test_guided_progressive_offers_the_paths_tools_too below)
        t = await run_path(progressive, two_tool_path(), "free", 0, session, llm)
        assert session.tool_calls == 1, "discover_tools never reached the server; list_items did"

    assert t.outcome == "completed", t.reason  # max_tool_calls=1 was not spent on discovery
    assert offered_to(llm, 1) == ["lookup", DISCOVER_TOOL_NAME]
    assert offered_to(llm, 2) == ["lookup", "list_items", DISCOVER_TOOL_NAME]
    assert offered_events(t) == [
        (["lookup"], "initial:free:progressive"),
        (["list_items"], "discover_tools:items"),
    ]
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "assistant",
        "tool_call",
        "tools_offered",
        "tool_result",
        "assistant",
        "tool_call",
        "tool_result",
        "assistant",
        "final_result",
        "usage",
        "end",
    ]
    assert [(c.name, c.arguments) for c in t.tool_calls()] == [
        (DISCOVER_TOOL_NAME, {"query": "items"}),
        ("list_items", {"cursor": 0, "limit": 2}),
    ]
    answer = t.tool_results()[0]
    assert answer.name == DISCOVER_TOOL_NAME and answer.is_error is False
    assert "list_items: List products a page at a time. (now available)" in answer.text
    assert "lookup" not in answer.text and "echo" not in answer.text
    block = llm.calls[2]["messages"][-1]["content"][0]
    assert block["tool_use_id"] == "toolu_d" and block["content"] == answer.text
    assert "is_error" not in block
    assert t.tools_offered() == ["lookup", "list_items"]


async def test_discover_tools_with_no_match_offers_nothing_and_says_so(
    scenario_data: dict[str, Any],
) -> None:
    progressive = scoped(scenario_data, disclosure="progressive", initial=["lookup"])
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response(DISCOVER_TOOL_NAME, {"query": "zzz"}, tool_use_id="toolu_d"),
            tool_use_response(DISCOVER_TOOL_NAME, {}, tool_use_id="toolu_e"),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(progressive, llm, mode="free")
    assert offered_events(t) == [(["lookup"], "initial:free:progressive")]
    no_match, no_query = t.tool_results()
    assert no_match.is_error is False
    assert no_match.text.startswith("No further tool matches 'zzz'; 4 undisclosed tool(s)")
    assert no_query.is_error is True and "query" in no_query.text
    assert offered_to(llm, 3) == ["lookup", DISCOVER_TOOL_NAME]


async def test_tool_use_for_an_undisclosed_tool_is_refused_without_reaching_the_server(
    scenario_data: dict[str, Any],
) -> None:
    progressive = scoped(scenario_data, disclosure="progressive", initial=["lookup"])
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response("list_items", {"cursor": 0}, tool_use_id="toolu_x"),
            tool_use_response("lookup", {"slug": "penne"}, tool_use_id="toolu_1"),
            text_response(FINAL_TEXT),
        ]
    )
    async with open_session() as session:
        t = await run_path(progressive, two_tool_path(), "free", 0, session, llm)
        assert session.tool_calls == 1, "only lookup reached the server"

    assert t.outcome == "completed" and t.final_result == PENNE
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "assistant",
        "error",
        "assistant",
        "tool_call",
        "tool_result",
        "assistant",
        "final_result",
        "usage",
        "end",
    ]
    assert error_messages(t) == ["scope violation: list_items (not disclosed)"]
    assert scope_violations(t) == ["list_items (not disclosed)"]
    assert [c.name for c in t.tool_calls()] == ["lookup"]
    assert llm.calls[2]["messages"][-1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_x",
            "content": "tool list_items is not available in this conversation",
            "is_error": True,
        }
    ]


async def test_tool_use_for_a_denied_tool_is_refused_as_not_allowed(
    scenario_data: dict[str, Any],
) -> None:
    denied = scoped(scenario_data, deny=["fail"])
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response("fail", {"reason": "x"}, tool_use_id="toolu_f"),
            text_response(FINAL_TEXT),
        ]
    )
    async with open_session() as session:
        t = await run_path(denied, two_tool_path(), "guided", 0, session, llm)
        assert session.tool_calls == 0
    assert offered_to(llm, 1) == ["lookup", "list_items", "echo", "expensive_report"]
    assert error_messages(t) == ["scope violation: fail (not allowed)"]
    assert t.tool_calls() == [] and t.outcome == "completed"

    # discover_tools is a server-unknown name outside progressive disclosure: not allowed.
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response(DISCOVER_TOOL_NAME, {"query": "items"}, tool_use_id="toolu_d"),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(denied, llm)
    assert error_messages(t) == ["scope violation: discover_tools (not allowed)"]
    assert t.tool_calls() == []


async def test_observer_hooks_offer_tools_and_enable_goal(scenario_data: dict[str, Any]) -> None:
    progressive = scoped(scenario_data, disclosure="progressive", initial=["lookup"])
    handles: list[LiveRun] = []

    def on_start(live: LiveRun) -> None:
        assert live.offered == ["lookup"]
        added = live.offer_tools(["echo", "nope_*", "lookup"], "observer:needs echo")
        assert added == ["echo"], "globs matching nothing and tools already offered add nothing"
        live.enable_goal("Also tell me the store's opening hours.", "observer:hours")
        handles.append(live)

    class HookingLLM(ScriptedLLM):
        async def complete(self, **kwargs: Any) -> Any:
            response = await super().complete(**kwargs)
            if len(self.calls) == 2:  # the agent's first turn just came back
                assert handles[0].offer_tools(["list_items"], "observer:page") == ["list_items"]
                handles[0].enable_goal("And the total number of products.", "observer:total")
            return response

    llm = HookingLLM(
        [
            opening(),
            tool_use_response("lookup", {"slug": "penne"}, tool_use_id="toolu_1"),
            text_response(FINAL_TEXT),
        ]
    )
    async with open_session() as session:
        t = await run_path(progressive, two_tool_path(), "free", 0, session, llm, on_start=on_start)

    assert t.outcome == "completed"
    assert offered_to(llm, 1) == ["lookup", "echo", DISCOVER_TOOL_NAME]
    assert offered_to(llm, 2) == ["lookup", "echo", "list_items", DISCOVER_TOOL_NAME]
    assert offered_events(t) == [
        (["lookup"], "initial:free:progressive"),
        (["echo"], "observer:needs echo"),
        (["list_items"], "observer:page"),
    ]
    assert [(e.text, e.reason) for e in t.events if isinstance(e, GoalEnabledEvent)] == [
        ("Also tell me the store's opening hours.", "observer:hours"),
        ("And the total number of products.", "observer:total"),
    ]
    assert t.kinds()[:5] == ["system", "tools_offered", "tools_offered", "goal_enabled", "user"]

    # A goal enabled before the first turn rides with the opening; one enabled mid-run rides
    # with the next tool results. Each is delivered once.
    opening_text = next(e for e in t.events if isinstance(e, UserEvent)).text
    first = llm.calls[1]["messages"][0]["content"]
    assert first[0] == {"type": "text", "text": opening_text}
    assert first[1]["type"] == "text" and first[1]["text"].startswith("## Additional goal")
    assert (
        "- Goal enabled by observation (observer:hours): Also tell me the store's opening hours."
    ) in first[1]["text"]
    second = llm.calls[2]["messages"][-1]["content"]
    assert second[0]["type"] == "tool_result" and second[0]["tool_use_id"] == "toolu_1"
    assert second[1]["type"] == "text" and "total number of products" in second[1]["text"]
    assert "opening hours" not in second[1]["text"]
    # Both goals also persist in the system prompt of every later turn.
    assert "## Goals enabled by observation" not in llm.calls[0]["system"]
    assert "Goal enabled by observation (observer:hours)" in llm.calls[1]["system"]
    assert "(observer:total): And the total number of products." in llm.calls[2]["system"]
    assert "(observer:total)" not in llm.calls[1]["system"]


async def test_dry_run_refuses_a_step_whose_tool_the_scenario_denies(
    scenario_data: dict[str, Any],
) -> None:
    denied = scoped(scenario_data, deny=["fail"])
    path = Path(
        id="dry",
        kind="happy",
        title="dry",
        steps=[
            Step(intent="look", tool="lookup", arguments_sketch={"slug": "penne"}),
            Step(intent="break", tool="fail", arguments_sketch={"reason": "x"}),
            Step(intent="page", tool="list_items", arguments_sketch={"cursor": 0, "limit": 2}),
        ],
    )
    async with open_session() as session:
        t = await run_path(denied, path, "guided", 0, session, None, dry_run=True)
        assert session.tool_calls == 2

    assert t.outcome == "completed"
    assert t.reason == "dry run: called 2 planned tool(s); 1 step(s) refused as out of scope"
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "tool_call",
        "tool_result",
        "error",
        "tool_call",
        "tool_result",
        "final_result",
        "usage",
        "end",
    ]
    assert offered_events(t) == [(["lookup", "list_items"], "initial:guided:dry-run")]
    assert [c.name for c in t.tool_calls()] == ["lookup", "list_items"]
    assert error_messages(t) == ["scope violation: fail (not allowed)"]
    assert scope_violations(t) == ["fail (not allowed)"]


# ------------------------------------------------------------------------------------------
# guided + progressive: the path's tools are offered too (no discover_tools detour)


async def test_guided_progressive_offers_the_paths_tools_too(scenario_data: dict[str, Any]) -> None:
    progressive = scoped(scenario_data, disclosure="progressive", initial=["lookup"])
    llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(progressive, llm, mode="guided")
    assert offered_to(llm, 1) == ["lookup", "list_items", DISCOVER_TOOL_NAME]
    assert offered_events(t) == [
        (["lookup"], "initial:guided:progressive"),
        (["list_items"], "initial:guided:path"),
    ]
    assert t.kinds()[:4] == ["system", "tools_offered", "tools_offered", "user"]
    # Already-offered path tools add no second event; free mode never unions the path.
    llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(scoped(scenario_data, disclosure="progressive"), llm, mode="guided")
    assert offered_events(t) == [(["lookup", "fail", "list_items"], "initial:guided:progressive")]
    llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(progressive, llm, mode="free")
    assert offered_events(t) == [(["lookup"], "initial:free:progressive")]
    # Not under plan disclosure either (guided already offers exactly the path's tools).
    llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    t = await run(scoped(scenario_data, disclosure="plan"), llm, mode="guided")
    assert offered_events(t) == [(["lookup", "list_items"], "initial:guided:plan")]


# ------------------------------------------------------------------------------------------
# observers in the loop: reports, then effects, before the next LLM call


def observed(scenario_data: dict[str, Any], *observers: dict[str, Any], **tools: Any) -> Scenario:
    data: dict[str, Any] = {**scenario_data, "observers": list(observers)}
    if tools:
        data["tools"] = tools
    return parse_scenario(data, source="fixture")


CLERK: dict[str, Any] = {
    "name": "clerk",
    "identity": "A stock clerk who reads the lookup result and nothing else.",
    "kind": "code",
    "watches": ["tool_traffic"],
    "on": ["tool_result"],
    "conditions": [
        {
            "id": "found",
            "when": "lookup returned the penne record",
            "check": {"tool_result": {"tool": "lookup", "where": {"slug": "penne"}}},
            "then": {
                "enable_tools": ["echo"],
                "enable_goal": "Echo the store name back to the person.",
                "note": "the clerk saw penne",
            },
        }
    ],
}


def report_events(t: Transcript) -> list[InformantReportEvent]:
    return [e for e in t.events if isinstance(e, InformantReportEvent)]


async def test_tool_result_observer_enables_a_tool_and_a_goal_for_the_next_turn(
    scenario_data: dict[str, Any],
) -> None:
    scenario = observed(scenario_data, CLERK, disclosure="progressive", initial=["lookup"])
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response("lookup", {"slug": "penne"}, tool_use_id="toolu_1"),
            tool_use_response("echo", {"text": "Fake Mart"}, tool_use_id="toolu_2"),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(scenario, llm, mode="free")

    assert t.outcome == "completed", t.reason
    assert offered_to(llm, 1) == ["lookup", DISCOVER_TOOL_NAME], "echo is not offered yet"
    assert offered_to(llm, 2) == ["lookup", "echo", DISCOVER_TOOL_NAME], "enabled by the report"
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "assistant",
        "tool_call",
        "tool_result",
        "informant_report",  # the report is recorded BEFORE anything changes
        "tools_offered",
        "goal_enabled",
        "assistant",
        "tool_call",
        "tool_result",
        "informant_report",  # the clerk reports again: still true, so no effect fires again
        "assistant",
        "final_result",
        "usage",
        "end",
    ]
    assert offered_events(t) == [
        (["lookup"], "initial:free:progressive"),
        (["echo"], "observer:clerk.found"),
    ]
    [first, second] = report_events(t)
    assert first.trigger == "tool_result" and second.trigger == "tool_result"
    [report] = first.reports
    assert (report.observer, report.condition, report.value) == ("clerk", "found", True)
    assert report.evidence == "lookup.slug == 'penne'"
    assert report.confidence == 1.0 and report.at_event == 5, "the tool_result event's index"
    assert first.notes == ["clerk.found: the clerk saw penne"]
    assert first.flags == [] and first.failures == []
    goals = [e for e in t.events if isinstance(e, GoalEnabledEvent)]
    assert len(goals) == 1, "a condition that stays true enables its goal once"
    assert (goals[0].text, goals[0].reason, goals[0].observer, goals[0].condition) == (
        "Echo the store name back to the person.",
        "observer:clerk.found",
        "clerk",
        "found",
    )
    goal_line = "Goal enabled by observation (clerk.found): Echo the store name back to the person."
    assert goal_line not in llm.calls[1]["system"]
    assert "## Goals enabled by observation\n- " + goal_line in llm.calls[2]["system"]
    note = llm.calls[2]["messages"][-1]["content"]
    assert note[0]["type"] == "tool_result" and note[1]["text"].startswith("## Additional goal")
    assert goal_line in note[1]["text"]
    assert t.flags == [] and t.hard_failures == []
    assert t.usage == {scenario.models.agent: Usage(input_tokens=40, output_tokens=20)}, (
        "code observers cost nothing"
    )


async def test_turn_observer_reports_after_every_assistant_turn(
    scenario_data: dict[str, Any],
) -> None:
    counter = {
        "name": "counter",
        "identity": "counts turns",
        "kind": "code",
        "on": ["turn"],
        "conditions": [
            {
                "id": "spoke",
                "when": "the assistant said something",
                "check": {"regex": {"of": "last_assistant", "pattern": "."}},
            }
        ],
    }
    scenario = observed(scenario_data, counter)
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response("lookup", {"slug": "penne"}, tool_use_id="toolu_1", text="Looking."),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(scenario, llm)
    events = report_events(t)
    assert [e.trigger for e in events] == ["turn", "turn"]
    assistant_indexes = [i for i, e in enumerate(t.events) if isinstance(e, AssistantEvent)]
    assert [e.reports[0].at_event for e in events] == assistant_indexes
    assert [e.reports[0].value for e in events] == [True, True]
    assert t.kinds()[3:5] == ["assistant", "informant_report"]


async def test_fail_effect_marks_the_transcript_and_flags_are_kept(
    scenario_data: dict[str, Any],
) -> None:
    watcher = {
        "name": "scope_watcher",
        "identity": "A gatekeeper who only reads the error log.",
        "kind": "code",
        "watches": ["all"],
        "on": ["end"],
        "conditions": [
            {
                "id": "violation",
                "when": "a tool outside the offered set was called",
                "check": {"regex": {"of": "errors", "pattern": "^scope violation: "}},
                "then": {"flag": "out_of_scope", "fail": True},
            },
            {
                "id": "brief",
                "when": "the final answer is under 150 words",
                "check": {"word_count": {"of": "final_answer", "lt": 150}},
                "otherwise": {"flag": "verbose"},
            },
        ],
    }
    scenario = observed(scenario_data, watcher, disclosure="progressive", initial=["lookup"])
    llm = ScriptedLLM(
        [
            opening(),
            tool_use_response("list_items", {"cursor": 0}, tool_use_id="toolu_x"),
            tool_use_response("lookup", {"slug": "penne"}, tool_use_id="toolu_1"),
            text_response(FINAL_TEXT),
        ]
    )
    t = await run(scenario, llm, mode="free")
    assert t.outcome == "completed"
    [event] = report_events(t)
    assert event.trigger == "end"
    assert [(r.condition, r.value) for r in event.reports] == [("violation", True), ("brief", True)]
    assert event.flags == ["out_of_scope"]
    assert event.failures == [
        "scope_watcher.violation — matched '^scope violation: ' in error: scope violation: "
        "list_items (not disclosed)"
    ]
    assert t.flags == ["out_of_scope"] and t.hard_failures == event.failures
    assert t.kinds()[-4:] == ["final_result", "informant_report", "usage", "end"]
    saved = Transcript.from_events(list(t.events))
    assert saved.hard_failures == t.hard_failures and saved.flags == t.flags


async def test_fail_effect_in_the_loop_fails_the_judge_despite_three_passing_votes(
    scenario_data: dict[str, Any],
) -> None:
    """End to end: a code observer's ``fail`` effect at ``end`` lands on the agent-loop
    transcript and the judge cannot overrule it, whatever the votes say."""
    from mcpsim.judge import judge
    from tests.test_judge import vote

    clerk = {
        "name": "strict_clerk",
        "identity": "A clerk who allows five words and not one more.",
        "kind": "code",
        "watches": ["final_answer"],
        "on": ["end"],
        "conditions": [
            {
                "id": "too_long",
                "when": "the final answer is longer than five words",
                "check": {"word_count": {"of": "final_answer", "gt": 5}},
                "then": {"flag": "verbose", "fail": True},
            }
        ],
    }
    scenario = observed(scenario_data, clerk)
    t = await run(scenario, ScriptedLLM([opening(), text_response(FINAL_TEXT)]))
    assert t.outcome == "completed" and t.final_result == PENNE
    assert t.hard_failures == ["strict_clerk.too_long — 11 words"], (
        "Penne is $2.49 at Fake Mart and its origin is verified. (the fenced block not counted)"
    )
    assert t.flags == ["verbose"]
    assert t.kinds()[-4:] == ["final_result", "informant_report", "usage", "end"]

    votes = ScriptedLLM([vote(True, 1.0), vote(True, 1.0), vote(True, 1.0)])
    verdict = await judge(scenario, two_tool_path(), t, votes, votes=3)
    assert verdict.matcher_passed and verdict.votes == 3 and verdict.score == 1.0
    assert verdict.passed is False, "3/3 passing votes cannot overrule an observer fail effect"
    assert verdict.failure_reasons == ["observer: strict_clerk.too_long — 11 words"]
    assert verdict.failure_reasons[0].startswith("observer:")
    assert verdict.flags == ["verbose"]
    # The judge saw the report that failed the run, with its trigger and evidence.
    prompt = votes.calls[0]["messages"][0]["content"]
    assert "- at end: strict_clerk.too_long = true — 11 words (confidence 1.00)" in prompt
    assert "  - observer: strict_clerk.too_long — 11 words" in prompt


async def test_llm_observer_usage_is_charged_to_the_run(scenario_data: dict[str, Any]) -> None:
    auditor = {
        "name": "auditor",
        "identity": "An independent auditor.",
        "watches": ["final_answer"],
        "on": ["end"],
        "model": "claude-haiku-4-5-20251001",
        "conditions": [
            {"id": "ok", "when": "the answer quotes a price", "then": {"flag": "priced"}}
        ],
    }
    scenario = observed(scenario_data, auditor)
    agent_llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    observer_llm = ScriptedLLM(
        [
            structured_response(
                REPORT_TOOL,
                {"reports": [{"condition_id": "ok", "value": True, "evidence": "[4] $2.49"}]},
            )
        ]
    )
    async with open_session() as session:
        t = await run_path(
            scenario, two_tool_path(), "guided", 0, session, agent_llm, observer_llm=observer_llm
        )
    assert t.outcome == "completed"
    assert len(observer_llm.calls) == 1 and observer_llm.calls[0]["model"] == auditor["model"]
    assert "## Final answer" in observer_llm.calls[0]["messages"][0]["content"]
    assert "An independent auditor." in observer_llm.calls[0]["system"]
    assert t.usage == {
        scenario.models.agent: Usage(input_tokens=20, output_tokens=10),
        "claude-haiku-4-5-20251001": Usage(input_tokens=10, output_tokens=5),
    }
    assert t.flags == ["priced"] and t.hard_failures == []
    [event] = report_events(t)
    assert event.reports[0].evidence == "[4] $2.49" and event.reports[0].trigger == "end"
    # Without observer_llm the agent's LLM serves the observers (and gets the call).
    both = ScriptedLLM(
        [
            opening(),
            text_response(FINAL_TEXT),
            structured_response(REPORT_TOOL, {"reports": [{"condition_id": "ok", "value": False}]}),
        ]
    )
    t = await run(scenario, both)
    assert both.calls[2]["model"] == auditor["model"]
    assert report_events(t)[0].reports[0].value is False and t.flags == []


async def test_observer_cost_counts_against_the_cost_budget(scenario_data: dict[str, Any]) -> None:
    auditor = {
        "name": "auditor",
        "identity": "An independent auditor.",
        "on": ["turn"],
        "conditions": [{"id": "ok", "when": "fine"}],
    }
    scenario = with_budgets(observed(scenario_data, auditor), max_cost_usd=0.001)
    heavy = Usage(input_tokens=100_000, output_tokens=1)
    agent_llm = ScriptedLLM([opening(), text_response(FINAL_TEXT)])
    observer_llm = ScriptedLLM(
        [
            structured_response(REPORT_TOOL, {"reports": []}, model="x").model_copy(
                update={"usage": heavy}
            )
        ]
    )
    async with open_session() as session:
        t = await run_path(
            scenario, two_tool_path(), "guided", 0, session, agent_llm, observer_llm=observer_llm
        )
    assert t.outcome == "budget_exceeded" and t.reason.startswith("max_cost_usd=0.001 exceeded")
    assert t.usage[scenario.models.agent].input_tokens == 100_020, "user + agent + observer"
    assert t.kinds()[-3:] == ["informant_report", "usage", "end"]


# ------------------------------------------------------------------------------------------
# dry run: code observers, their effects, and the covering final_result


async def test_dry_run_runs_code_observers_and_applies_their_effects(
    scenario_data: dict[str, Any],
) -> None:
    llm_observer = {
        "name": "auditor",
        "identity": "needs a model",
        "on": ["tool_result", "end"],
        "conditions": [{"id": "x", "when": "x", "then": {"fail": True}}],
    }
    scenario = observed(scenario_data, CLERK, llm_observer)
    path = Path(
        id="dry",
        kind="happy",
        title="dry",
        steps=[
            Step(intent="look", tool="lookup", arguments_sketch={"slug": "penne"}),
            Step(intent="page", tool="list_items", arguments_sketch={"cursor": 0, "limit": 2}),
        ],
    )
    t = await run(scenario, None, path=path, dry_run=True)
    assert t.outcome == "completed", t.reason
    assert t.kinds() == [
        "system",
        "tools_offered",
        "user",
        "tool_call",
        "tool_result",
        "informant_report",
        "tools_offered",
        "goal_enabled",
        "tool_call",
        "tool_result",
        "informant_report",  # still true after list_items: reported, no effect re-applied
        "final_result",
        "usage",
        "end",
    ], "no LLM observer report anywhere: the dry run runs code and group observers only"
    assert offered_events(t) == [
        (["lookup", "list_items"], "initial:guided:dry-run"),
        (["echo"], "observer:clerk.found"),
    ]
    assert t.tools_offered() == ["lookup", "list_items", "echo"]
    assert report_events(t)[0].reports[0].evidence == "lookup.slug == 'penne'"
    assert t.hard_failures == [] and t.flags == []
    assert t.final_result == PENNE, "lookup's result covers slug, price and origin_status"


def test_best_final_result_prefers_the_result_covering_the_most_expected_keys() -> None:
    from mcpsim.agent import best_final_result

    page = {"items": [PENNE], "next_cursor": None, "total": 5}
    assert best_final_result([PENNE, page], ["slug", "price", "origin_status"]) == PENNE
    assert best_final_result([page, PENNE], ["slug", "price", "origin_status"]) == PENNE
    assert best_final_result([PENNE, page], ["items", "total"]) == page
    assert best_final_result([PENNE, page], ["nothing"]) == page, "no coverage: the last one"
    assert best_final_result([PENNE, {"slug": "x"}], ["slug"]) == {"slug": "x"}, "tie: the last"
    assert best_final_result([[1, 2], PENNE], ["result"]) == {"result": [1, 2]}
    assert best_final_result([], ["slug"]) is None


async def test_dry_run_final_result_is_the_covering_result_not_the_last(
    scenario: Scenario,
) -> None:
    path = Path(
        id="dry",
        kind="happy",
        title="dry",
        steps=[
            Step(intent="look", tool="lookup", arguments_sketch={"slug": "penne"}),
            Step(intent="page", tool="list_items", arguments_sketch={"cursor": 0, "limit": 2}),
        ],
    )
    t = await run(scenario, None, path=path, dry_run=True)
    assert t.final_result == PENNE
    final = next(e for e in t.events if isinstance(e, FinalResultEvent))
    assert final.raw == json.dumps(PENNE, ensure_ascii=False)
