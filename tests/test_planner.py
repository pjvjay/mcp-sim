from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from mcpsim.llm import LLMResponse
from mcpsim.mcpclient import Catalog, ToolInfo
from mcpsim.plan import CHECKPOINT_PATTERN, ExecutionPlan
from mcpsim.planner import (
    DRY_RUN_PATH_ID,
    PLAN_TOOL_NAME,
    PlanDraft,
    PlanError,
    build_system_prompt,
    default_arguments,
    default_for,
    dry_run_arguments,
    dry_run_plan,
    example_step,
    extract_draft,
    is_expensive,
    output_keys,
    plan,
    plan_input_schema,
    plan_tool_definition,
    render_arguments,
    render_catalog_for_prompt,
    render_tool_line,
    tool_name_enum,
    validate_draft,
)
from mcpsim.scenario import Scenario, parse_scenario
from tests.conftest import open_session
from tests.fake_llm import ScriptedLLM, structured_response, text_response
from tests.fake_server import EXPENSIVE_TOOL_NAMES, TOOL_NAMES, build_server
from tests.test_scoping import cheapest_penne_like, pantry_catalog, pantry_scenario


@pytest.fixture
def scenario(scenario_data: dict[str, Any]) -> Scenario:
    return parse_scenario(scenario_data, source="test")


async def fake_catalog() -> Catalog:
    async with open_session() as session:
        return await session.catalog()


def step(tool: str | None, *, expect_error: bool = False, **arguments: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "intent": f"call {tool}" if tool else "answer",
        "tool": tool,
        "arguments_sketch": arguments,
        "success_looks_like": "the server rejects it" if expect_error else "a result",
    }
    if expect_error:
        out["expect_error"] = True
    return out


def path(
    path_id: str, kind: str, *steps: dict[str, Any], checkpoints: list[str] | None = None
) -> dict[str, Any]:
    return {
        "id": path_id,
        "kind": kind,
        "title": path_id,
        "rationale": "because",
        "steps": list(steps),
        "checkpoints": checkpoints
        if checkpoints is not None
        else [f"final_result: slug equals the slug lookup returned ({path_id})"],
    }


GOOD_PLAN: dict[str, Any] = {
    "paths": [
        path("happy", "happy", step("lookup", slug="penne"), step(None)),
        path(
            "recovery",
            "recovery",
            step("lookup", slug="pene", expect_error=True),
            step("lookup", slug="penne"),
        ),
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
    assert call["tools"][0]["input_schema"] == plan_tool_definition(catalog)["input_schema"]
    assert call["tools"][0]["input_schema"] != PlanDraft.model_json_schema()  # constrained
    assert call["messages"][0]["role"] == "user"
    user_text = call["messages"][0]["content"]
    assert scenario.goal in user_text and scenario.role in user_text
    for instruction in scenario.instructions:
        assert instruction in user_text
    assert '"slug":"penne"' in user_text  # the expected_outcome.json spec is shown
    for name in TOOL_NAMES:
        assert f"- {name}" in call["system"]
    assert "Look up a product by slug" in call["system"]
    assert "- lookup(slug: string)" in call["system"]
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
    catalog = Catalog(tools=[ToolInfo(name="b"), ToolInfo(name="a")])
    tool = plan_tool_definition(catalog)
    assert set(tool) == {"name", "description", "input_schema"}
    assert tool["name"] == PLAN_TOOL_NAME
    assert "paths" in tool["input_schema"]["properties"]
    step_tool = tool["input_schema"]["$defs"]["Step"]["properties"]["tool"]
    assert step_tool["anyOf"] == [{"type": "string", "enum": ["a", "b"]}, {"type": "null"}]
    assert tool_name_enum(catalog) == ["a", "b"]
    # Path.kind is an enum too, and expect_error is a boolean that defaults to false.
    kind = tool["input_schema"]["$defs"]["Path"]["properties"]["kind"]
    assert set(kind["enum"]) == {"happy", "recovery", "alternative", "boundary", "policy"}
    expect = tool["input_schema"]["$defs"]["Step"]["properties"]["expect_error"]
    assert expect == {"default": False, "title": "Expect Error", "type": "boolean"}
    # An empty catalog leaves only the no-tool step (an empty enum is not a valid schema).
    empty = plan_input_schema(Catalog())["$defs"]["Step"]["properties"]["tool"]
    assert empty["anyOf"] == [{"type": "null"}]


# --- constrain, don't just validate (LOCAL_MODELS.md) ----------------------------------------


async def test_emitted_tool_schema_carries_the_enum_of_the_fake_server_tools(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    llm = ScriptedLLM([structured_response(PLAN_TOOL_NAME, GOOD_PLAN)])
    await plan(scenario, catalog, llm)
    schema = llm.calls[0]["tools"][0]["input_schema"]
    step_tool = schema["$defs"]["Step"]["properties"]["tool"]
    assert step_tool["anyOf"] == [
        {"type": "string", "enum": sorted(TOOL_NAMES)},
        {"type": "null"},
    ]
    # Validation uses the same enum the schema carries.
    assert tool_name_enum(catalog) == sorted(TOOL_NAMES)


async def test_unknown_argument_key_is_reasked_once_with_the_key_named(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    bad = {"paths": [path("happy", "happy", step("list_items", slug="penne"))]}
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, bad),
            structured_response(PLAN_TOOL_NAME, GOOD_PLAN),
        ]
    )

    result = await plan(scenario, catalog, llm)

    assert len(llm.calls) == 2 and len(result.paths) == 3
    error_text = llm.calls[1]["messages"][2]["content"][0]["content"]
    assert "argument 'slug' is not accepted by tool list_items" in error_text
    assert "its arguments are: cursor, limit" in error_text
    assert "paths[0].steps[0]" in error_text


async def test_wrong_scalar_type_is_reasked(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    bad = {"paths": [path("happy", "happy", step("list_items", cursor="first page"))]}
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, bad),
            structured_response(PLAN_TOOL_NAME, GOOD_PLAN),
        ]
    )

    await plan(scenario, catalog, llm)

    assert len(llm.calls) == 2
    error_text = llm.calls[1]["messages"][2]["content"][0]["content"]
    assert "argument 'cursor' of tool list_items must be integer, got string" in error_text
    assert '"first page"' in error_text
    assert "$from_step" in error_text  # the fix for a value only the server knows


async def test_recovery_path_without_expect_error_is_rejected(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    no_failure = {
        "paths": [
            path("happy", "happy", step("lookup", slug="penne")),
            path("recovery", "recovery", step("lookup", slug="pene"), step("lookup", slug="penne")),
        ]
    }
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, no_failure),
            structured_response(PLAN_TOOL_NAME, no_failure),
        ]
    )
    with pytest.raises(PlanError) as excinfo:
        await plan(scenario, catalog, llm)
    message = str(excinfo.value)
    assert "paths[1] ('recovery')" in message
    assert "recovery path must contain at least one step with expect_error true" in message
    assert len(llm.calls) == 2


async def test_checkpoint_without_the_prefix_is_rejected(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    vague = {
        "paths": [
            path(
                "happy",
                "happy",
                step("lookup", slug="penne"),
                checkpoints=["Priced shopping list for penne", "transcript: lookup was called"],
            )
        ]
    }
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, vague),
            structured_response(PLAN_TOOL_NAME, GOOD_PLAN),
        ]
    )
    await plan(scenario, catalog, llm)
    error_text = llm.calls[1]["messages"][2]["content"][0]["content"]
    assert "paths[0].checkpoints[0] ('happy')" in error_text
    assert "'Priced shopping list for penne'" in error_text
    assert "final_result, tool_result[<tool_name>] or transcript" in error_text
    assert "checkpoints[1]" not in error_text  # the shaped one passed


@pytest.mark.parametrize(
    "text",
    [
        "final_result: slug equals penne",
        "tool_result[lookup]: origin_status is verified",
        "transcript:no call to expensive_report",
        "  transcript: leading space is fine",
        "report: shelf_auditor.direct_match is true",
        "report:librarian.exists is false",
    ],
)
def test_checkpoint_shapes_accepted(text: str) -> None:
    catalog = Catalog(tools=[ToolInfo(name="lookup")])
    draft = PlanDraft.model_validate(
        {"paths": [path("happy", "happy", step("lookup"), checkpoints=[text])]}
    )
    assert validate_draft(draft, catalog) == []


@pytest.mark.parametrize(
    "text",
    [
        "the agent did well",
        "final_result:",
        "final_result:   ",
        "tool_result[Lookup]: uppercase tool names are not allowed by the shape",
        "tool_result: needs the tool name in brackets",
        "Final_result: wrong case",
        "report: direct_match is true",
        "report: shelf_auditor.direct_match is maybe",
        "report: shelf_auditor.direct_match",
    ],
)
def test_checkpoint_shapes_rejected(text: str) -> None:
    assert CHECKPOINT_PATTERN.match(text.strip()) is None
    catalog = Catalog(tools=[ToolInfo(name="lookup")])
    draft = PlanDraft.model_validate(
        {"paths": [path("happy", "happy", step("lookup"), checkpoints=[text])]}
    )
    problems = validate_draft(draft, catalog)
    assert len(problems) == 1 and "checkpoint" in problems[0] and "shape" in problems[0]


def _typed_catalog() -> Catalog:
    return Catalog(
        tools=[
            ToolInfo(
                name="search",
                input_schema={
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "limit": {"type": "integer"},
                        "lat": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "verbose": {"type": "boolean"},
                        "decision": {"enum": ["approve", "reject"]},
                        "opts": {"$ref": "#/$defs/Opts"},
                    },
                    "required": ["q"],
                    "$defs": {"Opts": {"type": "object", "properties": {"a": {"type": "integer"}}}},
                },
            ),
            ToolInfo(
                name="loose",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
            ),
        ]
    )


def _problems(*steps: dict[str, Any]) -> list[str]:
    draft = PlanDraft.model_validate({"paths": [path("happy", "happy", *steps)]})
    return validate_draft(draft, _typed_catalog())


def test_argument_types_are_checked_per_json_type() -> None:
    assert _problems(step("search", q="x", limit=3, lat=1.5, tags=["a"], verbose=True)) == []
    assert _problems(step("search", q="x", lat=2)) == []  # integer satisfies number
    assert _problems(step("search", q="x", lat=None)) == []  # nullable
    assert _problems(step("search", q="x", opts={"a": 1})) == []  # $ref resolves to object
    assert _problems(step("search", q="x", decision="approve")) == []

    assert "must be string, got integer 5" in _problems(step("search", q=5))[0]
    assert "must be integer, got boolean true" in _problems(step("search", q="x", limit=True))[0]
    assert "must be integer, got number 1.5" in _problems(step("search", q="x", limit=1.5))[0]
    assert "must be array, got string" in _problems(step("search", q="x", tags="a"))[0]
    assert "must be boolean, got string" in _problems(step("search", q="x", verbose="yes"))[0]
    assert "must be object, got string" in _problems(step("search", q="x", opts="none"))[0]
    enum_problem = _problems(step("search", q="x", decision="maybe"))[0]
    assert 'must be one of ["approve","reject"], got "maybe"' in enum_problem


def test_unknown_keys_are_allowed_when_additional_properties_is_true() -> None:
    assert _problems(step("loose", anything="goes", n=1)) == []
    assert (
        "argument 'nope' is not accepted by tool search"
        in _problems(step("search", q="x", nope=1))[0]
    )


def test_references_must_point_at_an_earlier_step_with_a_tool() -> None:
    ref = {"$from_step": 1, "path": "items[*].id"}
    assert _problems(step("search", q="first"), step("search", q=ref)) == []
    # Any declared type accepts a reference: it is resolved at run time.
    assert _problems(step("search", q="first"), step("search", q="x", limit=ref)) == []

    forward = _problems(step("search", q={"$from_step": 2, "path": "x"}), step("search", q="y"))
    assert "references step 2, but a reference must point at an EARLIER step" in forward[0]
    assert "this is step 1" in forward[0]
    self_ref = _problems(step("search", q={"$from_step": 1, "path": "x"}))
    assert "must point at an EARLIER step" in self_ref[0]
    no_tool = _problems(step(None), step("search", q={"$from_step": 1, "path": "x"}))
    assert "references step 1, which calls no tool" in no_tool[0]
    malformed = _problems(step("search", q="a"), step("search", q={"$from_step": "one"}))
    assert "malformed step reference" in malformed[0] and "path" in malformed[0]


def test_validate_draft_passes_the_good_plan_and_dry_run_plan(scenario: Scenario) -> None:
    catalog = Catalog(
        tools=[
            ToolInfo(
                name="lookup",
                input_schema={
                    "type": "object",
                    "properties": {"slug": {"type": "string"}},
                    "required": ["slug"],
                },
            ),
            ToolInfo(
                name="list_items",
                input_schema={
                    "type": "object",
                    "properties": {"cursor": {"type": "integer"}, "limit": {"type": "integer"}},
                },
            ),
        ]
    )
    assert validate_draft(PlanDraft.model_validate(GOOD_PLAN), catalog) == []
    dry = dry_run_plan(scenario, catalog)
    assert validate_draft(PlanDraft(paths=dry.paths), catalog) == []
    for checkpoint in dry.paths[0].checkpoints:
        assert CHECKPOINT_PATTERN.match(checkpoint)
    assert all(step.expect_error is False for step in dry.paths[0].steps)


# --- the catalog digest ---------------------------------------------------------------------


class Hit(BaseModel):
    """A typed return for the digest test (module level so the SDK can resolve the annotation)."""

    slug: str
    price: float
    store: str


async def test_digest_shows_returns_keys_for_structured_tools() -> None:
    server = build_server()

    @server.tool()
    def typed_lookup(slug: str) -> Hit:
        """Look up a product as a typed record. Second sentence is dropped."""
        return Hit(slug=slug, price=1.0, store="x")

    @server.tool()
    def typed_list(limit: int = 2, tag: str | None = None) -> list[Hit]:
        """List typed records."""
        return []

    async with open_session(server) as session:
        catalog = await session.catalog()
    digest = render_catalog_for_prompt(catalog)
    lines = {line.split("(")[0][2:]: line for line in digest.splitlines() if line.startswith("- ")}

    # The fake server's own structured tools publish an object schema without named keys.
    assert lines["lookup"] == "- lookup(slug: string) → returns object — Look up a product by slug."
    assert lines["list_items"].startswith(
        "- list_items(cursor?: integer, limit?: integer) → returns object — List products"
    )
    # Text-only tools have no output schema and no arrow.
    assert "→" not in lines["echo"]
    # A typed return shows its top-level keys; a list return is unwrapped from "result".
    assert lines["typed_lookup"] == (
        "- typed_lookup(slug: string) → returns slug, price, store"
        " — Look up a product as a typed record."
    )
    assert lines["typed_list"] == (
        "- typed_list(limit?: integer, tag?: string) → returns list of {slug, price, store}"
        " — List typed records."
    )
    assert render_tool_line(catalog.tool("typed_lookup")) == lines["typed_lookup"]


def test_output_keys_resolves_refs_and_result_wrapping() -> None:
    defs = {"Item": {"type": "object", "properties": {"id": {"type": "integer"}, "n": {}}}}
    assert output_keys({"type": "object", "properties": {"a": {}, "b": {}}}) == "a, b"
    assert output_keys({"$ref": "#/$defs/Item", "$defs": defs}) == "id, n"
    assert (
        output_keys(
            {"type": "object", "properties": {"result": {"$ref": "#/$defs/Item"}}, "$defs": defs}
        )
        == "id, n"
    )
    wrapped_list = {
        "type": "object",
        "properties": {"result": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
        "$defs": defs,
    }
    assert output_keys(wrapped_list) == "list of {id, n}"
    assert (
        output_keys({"properties": {"result": {"type": "array", "items": {"type": "string"}}}})
        == "list"
    )
    assert output_keys({"properties": {"result": {"type": "integer"}}}) == "result (integer)"
    assert output_keys({"type": "object", "additionalProperties": True}) == "object"
    assert output_keys({"type": "string"}) == "string"
    many = {"type": "object", "properties": {f"k{i}": {} for i in range(30)}}
    assert output_keys(many).endswith("k23, … (6 more)")


def test_render_arguments_marks_optional_and_types() -> None:
    tool = Catalog(tools=_typed_catalog().tools).tool("search")
    assert render_arguments(tool) == (
        "q: string, limit?: integer, lat?: number, tags?: string[], verbose?: boolean, "
        "decision?: approve|reject, opts?: object"
    )
    assert render_arguments(ToolInfo(name="bare")) == ""
    nullable_required = ToolInfo(
        name="t",
        input_schema={
            "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            "required": ["x"],
        },
    )
    assert render_arguments(nullable_required) == "x?: string"


async def test_system_prompt_states_the_six_rules_and_a_real_example_step() -> None:
    catalog = await fake_catalog()
    text = build_system_prompt(catalog)
    for n in range(1, 7):
        assert f"\n{n}. " in text
    for phrase in (
        "expect_error: true",
        "$from_step",
        "final_result, tool_result[<tool_name>] or transcript",
        "→ returns",
    ):
        assert phrase in text
    example = example_step(catalog)
    assert example.tool in TOOL_NAMES
    assert example.tool == "lookup"  # every fake tool has at most one required argument
    assert example.arguments_sketch == {"slug": "example"}
    rendered = text.split("Example of ONE well-formed step")[1].split("\n")[2]
    assert rendered == (
        '{"arguments_sketch":{"slug":"example"},"expect_error":false,'
        '"intent":"Call lookup to make progress on the goal",'
        '"success_looks_like":"lookup returns object with values the goal can use",'
        '"tool":"lookup"}'
    )
    # The example step itself passes validation against the catalog it was built from.
    draft = PlanDraft.model_validate(
        {"paths": [path("happy", "happy", example.model_dump(mode="json"))]}
    )
    assert validate_draft(draft, catalog) == []
    assert example_step(Catalog()).tool is None


# --- dry run ----------------------------------------------------------------------------------


async def test_dry_run_plans_the_relevant_tool_with_arguments_from_the_expected_outcome(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    llm = ScriptedLLM()  # nothing queued: any LLM call would raise

    result = await plan(scenario, catalog, llm, dry_run=True)

    assert llm.calls == []
    assert result.scenario == scenario.name
    assert result.catalog_digest == catalog.digest()
    assert len(result.paths) == 1
    only = result.paths[0]
    assert only.id == DRY_RUN_PATH_ID and only.kind == "happy"
    # lookup is the one tool that shares vocabulary with "find the price and store of penne"
    # (its name is in the instructions; price, store, slug and origin_status are its output
    # keys); the expected outcome pins slug to "penne", so that is the argument, not "".
    assert [(s.tool, s.arguments_sketch) for s in only.steps] == [("lookup", {"slug": "penne"})]
    assert only.title == "Dry run over 1 goal-relevant tool(s)"
    assert "Call lookup with arguments from the expected outcome" in only.steps[0].intent
    assert only.steps[0].success_looks_like.startswith('lookup returns a result for slug="penne"')
    for name in EXPENSIVE_TOOL_NAMES:
        assert name not in result.tools_used()
        assert name in only.rationale
    assert "cost" in only.rationale
    assert "irrelevant (no shared vocabulary with the scenario): fail, list_items, echo." in (
        only.rationale
    )
    assert 'take that value: lookup(slug="penne")' in only.rationale
    assert only.checkpoints == [
        "transcript: contains a tool_call for lookup",
        "final_result: equals the structured content of the successful tool result that covers "
        "the most expected_outcome.json keys (the last one otherwise), or is null when no step "
        "returned structured content",
    ]


async def test_dry_run_arguments_prefer_expected_values_then_schema_defaults() -> None:
    server = build_server()

    @server.tool()
    def search(query: str, limit: int = 2, verbose: bool = False, slug: str = "") -> dict[str, Any]:
        """Search by query."""
        return {"query": query, "limit": limit, "verbose": verbose, "slug": slug}

    async with open_session(server) as session:
        catalog = await session.catalog()
    tool = catalog.tool("search")

    spec = {"query": "penne", "limit": 5, "verbose": True, "slug": 7, "price": {"$gt": 0}}
    # A plain string/number of the right type fills required and optional arguments alike; a
    # boolean is not plain; a number for a string property is ignored.
    assert dry_run_arguments(tool, spec) == (
        {"query": "penne", "limit": 5},
        [],
        ["query", "limit"],
    )
    assert dry_run_arguments(tool, {"price": {"$gt": 0}}) == ({"query": ""}, [], []), (
        "no expected value: required arguments take placeholders, optional ones are left out"
    )
    assert dry_run_arguments(tool, None) == ({"query": ""}, [], [])
    assert dry_run_arguments(catalog.tool("lookup"), {"slug": "penne"}) == (
        {"slug": "penne"},
        [],
        ["slug"],
    )
    assert dry_run_arguments(catalog.tool("fail"), {"reason": "x"}) == (
        {"reason": "x"},
        [],
        ["reason"],
    )


async def test_dry_run_sketch_defaults_required_fields_when_the_outcome_says_nothing(
    scenario_data: dict[str, Any],
) -> None:
    scenario = parse_scenario(
        {**scenario_data, "expected_outcome": {"json": {"price": {"$gt": 0}}}}, source="test"
    )
    catalog = await fake_catalog()
    result = await plan(scenario, catalog, None, dry_run=True)
    assert [s.arguments_sketch for s in result.paths[0].steps] == [{"slug": ""}]
    assert "Required arguments take schema placeholders." in result.paths[0].rationale
    assert "Call lookup with placeholder arguments" in result.paths[0].steps[0].intent


async def test_dry_run_skips_tools_with_undefaultable_required_args(
    scenario_data: dict[str, Any],
) -> None:
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
    scenario = parse_scenario(
        {
            **scenario_data,
            "goal": "Use by_filter and flagged to check the names.",
            "instructions": [],
            "expected_outcome": {"text": "The flagged names."},
        },
        source="test",
    )

    result = dry_run_plan(scenario, catalog)
    only = result.paths[0]
    tools = [s.tool for s in only.steps]
    assert "by_filter" not in tools
    assert tools == ["flagged"], "the only relevant tool whose arguments can be defaulted"
    assert "by_filter (filter)" in only.rationale
    flagged_step = next(s for s in only.steps if s.tool == "flagged")
    assert flagged_step.arguments_sketch == {"on": False, "names": [], "n": 1}


async def test_dry_run_falls_back_to_catalog_order_when_nothing_is_relevant(
    scenario_data: dict[str, Any],
) -> None:
    scenario = parse_scenario(
        {**scenario_data, "goal": "Zzz.", "instructions": [], "expected_outcome": {"text": "zzz"}},
        source="test",
    )
    catalog = await fake_catalog()
    result = dry_run_plan(scenario, catalog)
    only = result.paths[0]
    assert [s.tool for s in only.steps] == ["lookup", "fail", "list_items"], (
        "the first three non-expensive candidates in catalog order"
    )
    assert "No tool shares vocabulary with the scenario" in only.rationale
    assert "irrelevant (no shared vocabulary with the scenario): echo." in only.rationale
    assert "expensive_report" in only.rationale


async def test_dry_run_never_plans_more_steps_than_the_tool_budget(
    scenario_data: dict[str, Any],
) -> None:
    scenario = parse_scenario(
        {
            **scenario_data,
            "goal": "Zzz.",
            "instructions": [],
            "expected_outcome": {"text": "zzz"},
            "budgets": {"max_tool_calls": 2},
        },
        source="test",
    )
    catalog = await fake_catalog()
    only = dry_run_plan(scenario, catalog).paths[0]
    assert [s.tool for s in only.steps] == ["lookup", "fail"]


# --- dry run against the pantry catalog fixture (what the real suite will plan) ---------------


def test_pantry_dry_run_plans_find_product_penne_first_and_no_write_tool(
    scenario_data: dict[str, Any],
) -> None:
    catalog = pantry_catalog()
    scenario = cheapest_penne_like(scenario_data["server"])
    only = dry_run_plan(scenario, catalog).paths[0]

    assert only.steps[0].tool == "find_product"
    assert only.steps[0].arguments_sketch == {"query": "penne"}
    used = only.tools_used()
    assert "submit_origin_evidence" not in used and "review_origin_submission" not in used
    assert not any(name.startswith("plan_") for name in used), "the costly planners"
    assert len(only.steps) <= scenario.budgets.max_tool_calls
    assert used[:3] == ["find_product", "get_product", "get_product_origins"]
    assert only.steps[1].arguments_sketch == {"product_id": 1}, "schema placeholder"
    assert (
        "Skipped as write tools (neither the goal nor an instruction asks for a write): "
        "submit_origin_evidence, review_origin_submission."
    ) in only.rationale
    assert "plan_recipe, plan_from_text, plan_week" in only.rationale


def test_pantry_dry_run_respects_the_scenario_budget(scenario_data: dict[str, Any]) -> None:
    catalog = pantry_catalog()
    scenario = cheapest_penne_like(scenario_data["server"])
    scenario = scenario.model_copy(
        update={"budgets": scenario.budgets.model_copy(update={"max_tool_calls": 3})}
    )
    only = dry_run_plan(scenario, catalog).paths[0]
    assert only.tools_used() == ["find_product", "get_product", "get_product_origins"]
    assert "Skipped as least relevant beyond max_tool_calls=3: " in only.rationale

    committed = pantry_scenario("cheapest-penne")
    only = dry_run_plan(committed, catalog.filtered(["*"], ["submit_*", "review_*"])).paths[0]
    assert only.steps[0].tool == "find_product"
    assert only.steps[0].arguments_sketch == {"query": "penne"}
    assert len(only.steps) <= committed.budgets.max_tool_calls


def test_pantry_dry_run_with_write_intent_fills_the_submission_from_the_expected_outcome() -> None:
    catalog = pantry_catalog().filtered(
        [
            "find_product",
            "get_product",
            "list_products",
            "submit_origin_evidence",
            "list_origin_submissions",
            "pipeline_status",
        ],
        [],
    )
    label = pantry_scenario("label-submission")
    only = dry_run_plan(label, catalog).paths[0]
    assert only.steps[0].tool == "submit_origin_evidence"
    assert only.steps[0].arguments_sketch == {
        "claim_type": "product-of",
        "confidence": "high",
        "country": "Italy",
        "product_id": 1,
        "verbatim": "Product of Italy",
    }, "plain expected values fill the optional confidence too; product_id is an operator"
    assert "review_origin_submission" not in only.tools_used(), "not in the allowed catalog"
    assert "Skipped as write tools" not in only.rationale


async def test_catalog_digest_changes_when_the_catalog_changes(scenario: Scenario) -> None:
    base = await fake_catalog()
    bigger = build_server()

    @bigger.tool()
    def extra(x: int) -> dict[str, int]:
        """An extra tool that knows the price of penne."""
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
    # The pantry server's free tools all say this; the first heuristic matched
    # the bare word "LLM" and skipped them, leaving the dry run one tool.
    assert not is_expensive(ToolInfo(
        name="x", description="List all seeded recipes. Free — no LLM calls."))
    assert not is_expensive(ToolInfo(
        name="x", description="Resolved origin evidence per product. Free, no LLM calls."))
    assert is_expensive(ToolInfo(
        name="x", description="Run the pipeline. SLOW (10-60s) and costs real Claude API credits."))
    assert not is_expensive(ToolInfo(name="x", description="Talks to an LLM"))  # no cost claimed


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
