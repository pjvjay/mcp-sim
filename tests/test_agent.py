from __future__ import annotations

import json
from typing import Any, cast

import pytest

from mcpsim.agent import (
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
from mcpsim.llm import Usage, estimate_cost_usd
from mcpsim.mcpclient import ToolResult
from mcpsim.plan import Mode, Path, Step, StepReference
from mcpsim.scenario import Scenario, parse_scenario
from mcpsim.transcript import Transcript
from tests.conftest import open_session
from tests.fake_llm import ScriptedLLM, text_response, tool_use_response

HAPPY_SEQUENCE = [
    "system",
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
    assert t.events[1].text.startswith("Hi, I need the price")

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
    assert first_agent_call["messages"] == [{"role": "user", "content": t.events[1].text}]

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
    assert t.events[2].text == "Looking." and t.events[2].stop_reason == "tool_use"
    assert t.events[2].tool_uses[0]["name"] == "lookup"
    assert t.events[9].raw == json.dumps(PENNE)

    # Usage is summed per model and costed from the rate table.
    assert t.usage == {scenario.models.agent: Usage(input_tokens=40, output_tokens=20)}
    assert t.cost_usd == pytest.approx(
        estimate_cost_usd(scenario.models.agent, Usage(input_tokens=40, output_tokens=20))
    )
    assert t.cost_usd > 0
    assert t.events[10].per_model == t.usage and t.events[10].estimate is True


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
    assert t.kinds() == ["system", "user", "assistant", "user", "usage", "end"]
    assert llm.remaining == 1


async def test_max_cost_budget_stops_the_run(scenario: Scenario) -> None:
    llm = ScriptedLLM([opening(), tool_use_response("lookup", {"slug": "penne"})])
    t = await run(with_budgets(scenario, max_cost_usd=1e-9), llm)
    assert t.outcome == "budget_exceeded" and "max_cost_usd=1e-09" in t.reason
    # The opening user turn alone blows a one-nanodollar budget: no agent turn happens.
    assert t.kinds() == ["system", "user", "usage", "end"]
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
    assert t.kinds() == ["system", "user", "assistant", "final_result", "error", "usage", "end"]
    final_event, error_event = t.events[3], t.events[4]
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
    assert t.events[3].text == "I don't mind, just use what the server has."
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
    assert t.kinds() == ["system", "user", "assistant", "tool_call", "usage", "end"]


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
    assert t.final_result == t.tool_results()[-1].structured
    assert isinstance(t.final_result, dict) and t.final_result["total"] == 5
    assert t.events[1].text == scenario.goal.strip()
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
    assert isinstance(t.final_result, dict) and t.final_result["total"] == 5


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
