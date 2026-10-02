from __future__ import annotations

from typing import Any

import pytest

from mcpsim.llm import LLMResponse
from mcpsim.mcpclient import Catalog, ToolInfo
from mcpsim.plan import ExecutionPlan
from mcpsim.planner import (
    DRY_RUN_PATH_ID,
    PLAN_TOOL_NAME,
    PlanDraft,
    PlanError,
    build_system_prompt,
    default_arguments,
    default_for,
    dry_run_plan,
    extract_draft,
    is_expensive,
    plan,
    plan_tool_definition,
    validate_draft,
)
from mcpsim.scenario import Scenario, parse_scenario
from tests.conftest import open_session
from tests.fake_llm import ScriptedLLM, structured_response, text_response
from tests.fake_server import EXPENSIVE_TOOL_NAMES, FREE_TOOL_NAMES, TOOL_NAMES, build_server


@pytest.fixture
def scenario(scenario_data: dict[str, Any]) -> Scenario:
    return parse_scenario(scenario_data, source="test")


async def fake_catalog() -> Catalog:
    async with open_session() as session:
        return await session.catalog()


def step(tool: str | None, **arguments: Any) -> dict[str, Any]:
    return {
        "intent": f"call {tool}" if tool else "answer",
        "tool": tool,
        "arguments_sketch": arguments,
        "success_looks_like": "a result",
    }


def path(path_id: str, kind: str, *steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": path_id,
        "kind": kind,
        "title": path_id,
        "rationale": "because",
        "steps": list(steps),
        "checkpoints": [f"{path_id}: final_result.slug equals the slug lookup returned"],
    }


GOOD_PLAN: dict[str, Any] = {
    "paths": [
        path("happy", "happy", step("lookup", slug="penne"), step(None)),
        path("recovery", "recovery", step("lookup", slug="pene"), step("lookup", slug="penne")),
        path("boundary", "boundary", step("list_items", cursor=0, limit=2)),
    ]
}

BAD_PLAN: dict[str, Any] = {
    "paths": [path("happy", "happy", step("price_lookup", slug="penne"))],
}


# --- structured output over the LLM --------------------------------------------------------


async def test_good_plan_first_try_uses_one_forced_tool_call(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    llm = ScriptedLLM([structured_response(PLAN_TOOL_NAME, GOOD_PLAN)])

    result = await plan(scenario, catalog, llm)

    assert isinstance(result, ExecutionPlan)
    assert result.scenario == scenario.name
    assert result.catalog_digest == catalog.digest()
    assert [p.id for p in result.paths] == ["happy", "recovery", "boundary"]
    assert [p.kind for p in result.paths] == ["happy", "recovery", "boundary"]
    assert result.tools_used() == ["lookup", "list_items"]
    assert result.path("happy").steps[1].tool is None

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["model"] == scenario.models.planner
    assert call["tool_choice"] == {"type": "tool", "name": PLAN_TOOL_NAME}
    assert [t["name"] for t in call["tools"]] == [PLAN_TOOL_NAME]
    assert call["tools"][0]["input_schema"] == PlanDraft.model_json_schema()
    assert call["messages"][0]["role"] == "user"
    user_text = call["messages"][0]["content"]
    assert scenario.goal in user_text and scenario.role in user_text
    for instruction in scenario.instructions:
        assert instruction in user_text
    assert '"slug":"penne"' in user_text  # the expected_outcome.json spec is shown
    for name in TOOL_NAMES:
        assert f"- {name}" in call["system"]
    assert "Look up a product by slug" in call["system"]
    assert '"required":["slug"]' in call["system"]
    for kind in ("happy", "recovery", "alternative", "boundary", "policy"):
        assert kind in call["system"]


async def test_fabricated_tool_is_rejected_and_reasked_once(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, BAD_PLAN),
            structured_response(PLAN_TOOL_NAME, GOOD_PLAN),
        ]
    )

    result = await plan(scenario, catalog, llm)

    assert len(llm.calls) == 2
    assert [p.id for p in result.paths] == ["happy", "recovery", "boundary"]
    assert llm.remaining == 0

    first, second = llm.calls
    assert len(first["messages"]) == 1
    # The re-ask keeps the rejected assistant turn and appends the validation error.
    assert len(second["messages"]) == 3
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][1]["content"][0]["type"] == "tool_use"
    assert second["messages"][1]["content"][0]["input"] == BAD_PLAN
    reask = second["messages"][2]
    assert reask["role"] == "user"
    # The Messages API requires a tool_result answering the rejected tool_use's id.
    assert reask["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": second["messages"][1]["content"][0]["id"],
            "content": reask["content"][0]["content"],
            "is_error": True,
        }
    ]
    error_text = reask["content"][0]["content"]
    assert "price_lookup" in error_text
    assert "unknown tool" in error_text
    assert "lookup" in error_text  # the catalog's real tools are listed
    assert second["tool_choice"] == first["tool_choice"]
    assert second["system"] == first["system"]


async def test_two_bad_plans_raise_plan_error_after_exactly_two_calls(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, BAD_PLAN),
            structured_response(PLAN_TOOL_NAME, BAD_PLAN),
            structured_response(PLAN_TOOL_NAME, GOOD_PLAN),  # must never be consumed
        ]
    )

    with pytest.raises(PlanError) as excinfo:
        await plan(scenario, catalog, llm)

    assert len(llm.calls) == 2
    assert llm.remaining == 1
    assert "price_lookup" in str(excinfo.value)
    assert scenario.name in str(excinfo.value)


async def test_response_without_the_plan_tool_is_reasked(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    llm = ScriptedLLM(
        [
            text_response("Here is my plan in prose instead."),
            structured_response(PLAN_TOOL_NAME, GOOD_PLAN),
        ]
    )

    result = await plan(scenario, catalog, llm)

    assert len(llm.calls) == 2
    reask = llm.calls[1]["messages"][2]["content"]
    assert isinstance(reask, str)  # no tool_use id to answer, so plain text
    assert PLAN_TOOL_NAME in reask
    assert len(result.paths) == 3


async def test_schema_invalid_plan_is_reasked_with_the_field_error(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    broken = {"paths": [{"id": "happy", "kind": "joyful", "title": "x", "steps": []}]}
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, broken),
            structured_response(PLAN_TOOL_NAME, GOOD_PLAN),
        ]
    )

    await plan(scenario, catalog, llm)

    assert len(llm.calls) == 2
    assert "paths.0.kind" in llm.calls[1]["messages"][2]["content"][0]["content"]


async def test_plan_without_happy_path_is_rejected(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    no_happy = {"paths": [path("alt", "alternative", step("lookup", slug="penne"))]}
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, no_happy),
            structured_response(PLAN_TOOL_NAME, no_happy),
        ]
    )

    with pytest.raises(PlanError, match="no path of kind 'happy'"):
        await plan(scenario, catalog, llm)
    assert len(llm.calls) == 2


async def test_llm_required_unless_dry_run(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    with pytest.raises(PlanError, match="LLM is required"):
        await plan(scenario, catalog, None)


def test_validate_draft_reports_every_problem() -> None:
    catalog = Catalog(tools=[ToolInfo(name="lookup")])
    draft = PlanDraft.model_validate(
        {
            "paths": [
                path("a", "alternative", step("lookup"), step("nope")),
                path("a", "boundary"),
            ]
        }
    )
    problems = validate_draft(draft, catalog)
    assert any("unknown tool 'nope'" in p for p in problems)
    assert any("duplicate path id 'a'" in p for p in problems)
    assert any("has no steps" in p for p in problems)
    assert any("no path of kind 'happy'" in p for p in problems)
    assert validate_draft(PlanDraft(), catalog) == [
        "plan has no paths; at least a happy path is required"
    ]


def test_extract_draft_rejects_non_object_input() -> None:
    catalog = Catalog(tools=[ToolInfo(name="lookup")])
    response = LLMResponse(
        content=[{"type": "tool_use", "id": "t", "name": PLAN_TOOL_NAME, "input": "nope"}],
        stop_reason="tool_use",
    )
    draft, problems = extract_draft(response, catalog)
    assert draft is None
    assert problems and "JSON object" in problems[0]


def test_plan_tool_definition_shape() -> None:
    tool = plan_tool_definition()
    assert set(tool) == {"name", "description", "input_schema"}
    assert tool["name"] == PLAN_TOOL_NAME
    assert "paths" in tool["input_schema"]["properties"]


# --- dry run ----------------------------------------------------------------------------------


async def test_dry_run_plan_covers_exactly_the_free_tools(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    llm = ScriptedLLM()  # nothing queued: any LLM call would raise

    result = await plan(scenario, catalog, llm, dry_run=True)

    assert llm.calls == []
    assert result.scenario == scenario.name
    assert result.catalog_digest == catalog.digest()
    assert len(result.paths) == 1
    only = result.paths[0]
    assert only.id == DRY_RUN_PATH_ID and only.kind == "happy"
    assert [s.tool for s in only.steps] == list(FREE_TOOL_NAMES)
    assert sorted(result.tools_used()) == sorted(FREE_TOOL_NAMES)
    for name in EXPENSIVE_TOOL_NAMES:
        assert name not in result.tools_used()
        assert name in only.rationale
    assert "cost" in only.rationale
    assert only.checkpoints


async def test_dry_run_sketch_arguments_default_required_fields_only(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    result = await plan(scenario, catalog, None, dry_run=True)
    sketches = {s.tool: s.arguments_sketch for s in result.paths[0].steps}
    assert sketches == {
        "lookup": {"slug": ""},
        "fail": {},
        "list_items": {},
        "echo": {"text": ""},
    }


async def test_dry_run_skips_tools_with_undefaultable_required_args(scenario: Scenario) -> None:
    server = build_server()

    @server.tool()
    def by_filter(filter: dict[str, int]) -> dict[str, Any]:  # noqa: A002 - deliberate
        """Filter by an object nobody can default."""
        return {"filter": filter}

    @server.tool()
    def flagged(on: bool, names: list[str], n: int) -> dict[str, Any]:
        """Every required type the spec defaults."""
        return {"on": on, "names": names, "n": n}

    async with open_session(server) as session:
        catalog = await session.catalog()

    result = dry_run_plan(scenario, catalog)
    only = result.paths[0]
    tools = [s.tool for s in only.steps]
    assert "by_filter" not in tools
    assert "flagged" in tools
    assert "by_filter" in only.rationale and "filter" in only.rationale
    flagged_step = next(s for s in only.steps if s.tool == "flagged")
    assert flagged_step.arguments_sketch == {"on": False, "names": [], "n": 1}


async def test_catalog_digest_changes_when_the_catalog_changes(scenario: Scenario) -> None:
    base = await fake_catalog()
    bigger = build_server()

    @bigger.tool()
    def extra(x: int) -> dict[str, int]:
        """An extra tool."""
        return {"x": x}

    async with open_session(bigger) as session:
        changed = await session.catalog()

    first = dry_run_plan(scenario, base)
    second = dry_run_plan(scenario, changed)
    again = dry_run_plan(scenario, base)
    assert first.catalog_digest == again.catalog_digest
    assert first.catalog_digest != second.catalog_digest
    assert "extra" in second.tools_used() and "extra" not in first.tools_used()


def test_default_for_table() -> None:
    assert default_for({"type": "string"}) == (True, "")
    assert default_for({"type": "integer"}) == (True, 1)
    assert default_for({"type": "number"}) == (True, 1)
    assert default_for({"type": "boolean"}) == (True, False)
    assert default_for({"type": "array", "items": {"type": "string"}}) == (True, [])
    assert default_for({"type": "string", "default": "x"}) == (True, "x")
    assert default_for({"enum": ["a", "b"]}) == (True, "a")
    assert default_for({"anyOf": [{"type": "null"}, {"type": "integer"}]}) == (True, 1)
    assert default_for({"type": ["null", "string"]}) == (True, "")
    assert default_for({"type": "object"}) == (False, None)
    assert default_for({}) == (False, None)


def test_default_arguments_and_expensive_detection() -> None:
    tool = ToolInfo(
        name="t",
        description="Costs credits per call.",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "object"}, "c": {"type": "x"}},
            "required": ["a", "b", "c", "missing"],
        },
    )
    assert default_arguments(tool) == (None, ["b", "c", "missing"])
    assert default_arguments(ToolInfo(name="bare")) == ({}, [])
    assert is_expensive(ToolInfo(name="x", description="SLOW: talks to an LLM"))
    assert is_expensive(ToolInfo(name="x", description="costs money"))
    assert is_expensive(ToolInfo(name="x", description="consumes one credit"))
    assert not is_expensive(ToolInfo(name="x", description="Look up a product by slug."))


def test_system_prompt_lists_resources_templates_and_prompts() -> None:
    catalog = Catalog(
        tools=[ToolInfo(name="lookup", description="Look up.")],
        resources=[],
        resource_templates=[],
        prompts=[],
    )
    text = build_system_prompt(catalog)
    assert "TOOLS (1)" in text and "RESOURCES (0)" in text
    assert "RESOURCE TEMPLATES (0)" in text and "PROMPTS (0)" in text
    assert PLAN_TOOL_NAME in text
