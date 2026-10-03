"""The local planner profile (mcpsim.execution_planner, docs/LOCAL_MODELS.md "The execution
planner"): the model plans the tool execution in the user's framing, the framework builds the
happy path, its checkpoints, a live-probed recovery or boundary path, and policy paths.

Every test asserts the exact derived value (the steps, the checkpoint text, the error the probe
really got) rather than the absence of a wrong one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path as FsPath
from typing import Any

import pytest
import yaml
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from mcpsim import runner
from mcpsim.execution_planner import (
    EXECUTION_TOOL_NAME,
    LINEAGE,
    LOCAL_PLAN_MAX_TOKENS,
    LOCAL_PROMPT_BUDGET,
    MAX_POLICY_QUESTIONS,
    NONE_ANSWER,
    POLICY_MAX_TOKENS,
    POLICY_TOOL_NAME,
    PROBE_INTEGER,
    WELL_FORMED_PATH,
    PlannedCall,
    ToolExecution,
    choose_probe,
    describe_expectation,
    execution_prompts,
    execution_schema,
    lineage_grammar_pattern,
    mutate,
    result_shape,
    review_execution,
    schema_lookup,
)
from mcpsim.mcpclient import Catalog
from mcpsim.plan import CHECKPOINT_PATTERN, ExecutionPlan
from mcpsim.planner import (
    PLAN_TOOL_NAME,
    PlanError,
    PlannerView,
    plan,
    planner_profile,
)
from mcpsim.scenario import Scenario, parse_scenario
from mcpsim.scout import (
    SCOUT_FILE,
    Observation,
    ScoutResult,
    is_read_only,
    probe,
    probe_refusal,
    scout,
)
from tests.conftest import fake_server_stdio_spec, open_session
from tests.fake_llm import ScriptedLLM, structured_response
from tests.fake_server import ITEMS, build_server

LOCAL = "ollama:command-r7b"
PENNE = ITEMS["penne"]
UNKNOWN_PENE = (
    "Error executing tool lookup: unknown slug 'pene'; try one of: basil, garlic, olive_oil, "
    "penne, tomato"
)


def local_data(scenario_data: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """The conftest scenario on a local planner, with ``store`` in the expected outcome."""
    data = {
        **scenario_data,
        "models": {"planner": LOCAL},
        "expected_outcome": {
            "text": "The price and store of penne, with its origin_status.",
            "json": {
                "slug": "penne",
                "price": {"$gt": 0},
                "store": {"$type": "string"},
                "origin_status": "verified",
            },
        },
    }
    data.update(extra)
    return data


@pytest.fixture
def scenario(scenario_data: dict[str, Any]) -> Scenario:
    return parse_scenario(local_data(scenario_data), source="test")


FIELDS = ["slug", "price", "store", "origin_status"]


async def fake_catalog(server: MCPServer | None = None) -> Catalog:
    async with open_session(server) as session:
        return await session.catalog()


def execution(steps: list[dict[str, Any]], **fields: str) -> dict[str, Any]:
    return {"steps": steps, "answer_fields": dict(fields)}


def call(tool: str, why: str = "", expect: str = "", **arguments: Any) -> dict[str, Any]:
    return {"tool": tool, "arguments": arguments, "why": why or f"call {tool}", "expect": expect}


def lookup_penne() -> dict[str, Any]:
    return call("lookup", "Look up penne", "price, store and origin_status of penne",
                slug="penne")


GOOD_LINEAGE = {
    "slug": "step 1: slug",
    "price": "step 1: price",
    "store": "step 1: store",
    "origin_status": "step 1: origin_status",
}


def answer(payload: dict[str, Any]) -> Any:
    return structured_response(EXECUTION_TOOL_NAME, payload)


def forbidden(tool: str) -> Any:
    return structured_response(POLICY_TOOL_NAME, {"tool": tool})


def penne_observation(**changes: Any) -> Observation:
    return Observation(
        kind="tool", name="lookup", arguments={"slug": "penne"}, summary="penne record",
        structured=dict(PENNE), from_expected=["slug"], **changes,
    )


def scout_result(*observations: Observation, budget: int = 5, calls: int | None = None,
                 disclosed: list[str] | None = None) -> ScoutResult:
    return ScoutResult(
        observations=list(observations),
        disclosed=disclosed or ["lookup", "list_items", "fail", "echo", "expensive_report"],
        budget=budget,
        tool_calls=len(observations) if calls is None else calls,
    )


HAPPY_CHECKPOINTS = [
    "final_result: slug equals tool_result[lookup] slug",
    "final_result: price equals tool_result[lookup] price",
    "final_result: store equals tool_result[lookup] store",
    "final_result: origin_status equals tool_result[lookup] origin_status",
    "final_result: slug equals penne",
    "final_result: price is greater than 0",
    "final_result: store is a string",
    "final_result: origin_status equals verified",
]


# --- profile selection ---------------------------------------------------------------------------


def test_an_ollama_planner_gets_the_local_profile(scenario: Scenario) -> None:
    assert planner_profile(scenario) == "local"
    hosted = scenario.model_copy(
        update={"models": scenario.models.model_copy(update={"planner": "claude-opus-5-5"})}
    )
    assert planner_profile(hosted) == "hosted"


# --- the grammar ---------------------------------------------------------------------------------


def test_execution_schema_constrains_tools_steps_and_one_lineage_per_field() -> None:
    schema = execution_schema(["list_items", "lookup"], ["slug", "price"], 4)
    assert schema["required"] == ["steps", "answer_fields"]
    assert schema["additionalProperties"] is False
    steps = schema["properties"]["steps"]
    assert (steps["minItems"], steps["maxItems"]) == (1, 4)
    item = steps["items"]
    assert item["required"] == ["tool", "arguments", "why", "expect"]
    assert item["additionalProperties"] is False
    assert item["properties"]["tool"] == {"type": "string", "enum": ["list_items", "lookup"]}
    assert item["properties"]["arguments"] == {"type": "object"}
    assert item["properties"]["why"]["maxLength"] == 120
    assert item["properties"]["expect"]["maxLength"] == 160
    fields = schema["properties"]["answer_fields"]
    assert fields["required"] == ["slug", "price"] and fields["additionalProperties"] is False
    assert fields["properties"] == {
        "slug": {"type": "string", "pattern": lineage_grammar_pattern(4)},
        "price": {"type": "string", "pattern": lineage_grammar_pattern(4)},
    }
    bare = execution_schema(["lookup"], [], 6)
    assert bare["required"] == ["steps"] and "answer_fields" not in bare["properties"]


LINEAGE_ACCEPTS = [
    "step 1: items[0].price",
    "step 2: summary.lines[*].origin_country",
    "step 3: coverage.spend_fraction",
    "step 1: result[0].product_id",
    "step 1: total",
]
LINEAGE_REJECTS = [
    "find_product.items[0].price",  # what command-r7b wrote without the pattern
    "step 1 items[0].price",
    "step 1.summary.total_cost",
    "step 0: total",
    "step 7: total",  # beyond the step cap of 6
    "step 1: lines_known / step 1.lines_total",
    "step 1:  total",
    "step 1: " + "a" * 81,
]


def test_lineage_grammar_pattern_admits_only_what_the_validator_accepts() -> None:
    pattern = lineage_grammar_pattern(6)
    for text in LINEAGE_ACCEPTS:
        assert re.fullmatch(pattern, text), text
        found = LINEAGE.match(text)
        assert found is not None and WELL_FORMED_PATH.match(found.group(2)), text
    for text in LINEAGE_REJECTS:
        assert not re.fullmatch(pattern, text), text


def test_lineage_grammar_pattern_has_one_pair_of_anchors() -> None:
    """llama.cpp converts only ``^…$`` around the whole pattern; an anchor inside an
    alternation is logged as unsupported and the string is left unconstrained."""
    pattern = lineage_grammar_pattern(6)
    assert pattern == "^step [1-6]: [A-Za-z0-9_.*\\[\\]]{1,80}$"
    inner = pattern[1:-1]
    assert "^" not in inner and "$" not in inner
    assert lineage_grammar_pattern(1) == "^step [1]: [A-Za-z0-9_.*\\[\\]]{1,80}$"
    assert not re.fullmatch(lineage_grammar_pattern(3), "step 4: total")
    for bad in (0, 10):
        with pytest.raises(ValueError, match="between 1 and 9"):
            lineage_grammar_pattern(bad)


# --- the prompt ----------------------------------------------------------------------------------


async def test_the_prompt_is_the_users_framing(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    view = PlannerView(disclosed=[catalog.tool("lookup")], allowed=catalog.tool_names())
    system, user = execution_prompts(scenario, view, FIELDS, budget=LOCAL_PROMPT_BUDGET)
    assert system == (
        "You are an expert JSON config generator. Generate a JSON config of the format:\n"
        '{"steps":[{"tool":"<tool name>","arguments":{"<argument>":<value>},'
        '"why":"<one sentence>","expect":"<what the result will show>"}],'
        '"answer_fields":{"<field the answer must contain>":'
        '"step <n>: <path in the result of step n, e.g. items[0].price>"}}\n'
        "\n"
        "that represents the plan of tool execution, using only these tools:\n"
        "- lookup(slug: string) → returns object — Look up a product by slug."
    )
    assert user == (
        "The user's request: Find the price and store of penne.\n"
        "\n"
        "Rules the plan must follow:\n"
        "- Use the lookup tool; do not invent prices.\n"
        "- Report origin_status exactly as returned."
    )


async def test_reports_and_observations_are_added_only_within_the_budget(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    view = PlannerView(
        disclosed=[catalog.tool("lookup")],
        allowed=catalog.tool_names(),
        goals=["Quote the store exactly."],
        observations=[
            Observation(kind="resource", name="fake://about", summary="A fake pantry server."),
            penne_observation(),
        ],
    )
    system, user = execution_prompts(scenario, view, FIELDS, budget=LOCAL_PROMPT_BUDGET)
    assert user.endswith(
        "- Quote the store exactly.\n\n"
        "What the server already returned (read-only calls):\n"
        '- lookup(slug="penne") → {slug="penne", price=2.49, store="Fake Mart", '
        'origin_status="verified"}'
    )
    assert "fake://about" not in user  # a step cannot read a resource
    tight_system, tight_user = execution_prompts(scenario, view, FIELDS, budget=len(system) + 300)
    assert tight_system == system
    assert "What the server already returned" not in tight_user
    assert tight_user.endswith("- Quote the store exactly.")  # the rules are never cut


def test_result_shape_shows_nested_keys_and_short_top_level_values() -> None:
    found = {
        "query": "penne",
        "tokens": ["penne"],
        "match": "direct",
        "total": 2,
        "items": [{"id": 51, "name": "Penne Rigate 500g", "store": "GreenLeaf", "price": 1.97}],
        "note": "2 product(s) match every token in 'penne'; each is priced at its cheapest store.",
    }
    assert result_shape(found) == (
        '{query="penne", tokens[1], match="direct", total=2, items[1]{id, name, store, price}, '
        "note}"
    )
    assert result_shape({"result": [{"a": 1}, {"a": 2}]}) == "{result[2]{a}}"


# --- the happy path -----------------------------------------------------------------------------


async def test_happy_path_and_its_checkpoints_are_derived_from_the_plan(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    llm = ScriptedLLM([answer(execution([lookup_penne()], **GOOD_LINEAGE)), forbidden("none")])
    result = await plan(scenario, catalog, llm)

    assert [p.id for p in result.paths] == ["happy"]
    happy = result.paths[0]
    assert happy.kind == "happy"
    assert [(s.tool, s.arguments_sketch, s.expect_error) for s in happy.steps] == [
        ("lookup", {"slug": "penne"}, False),
        (None, {}, False),
    ]
    assert happy.steps[0].intent == "Look up penne"
    assert happy.steps[0].success_looks_like == "price, store and origin_status of penne"
    assert happy.steps[1].success_looks_like == (
        "A final_result with slug, price, store, origin_status taken from the tool results"
    )
    assert happy.checkpoints == HAPPY_CHECKPOINTS
    assert all(CHECKPOINT_PATTERN.match(c) for c in happy.checkpoints)

    execution_call, policy_call = llm.calls
    assert execution_call["tool_choice"] == {"type": "tool", "name": EXECUTION_TOOL_NAME}
    assert execution_call["max_tokens"] == LOCAL_PLAN_MAX_TOKENS
    schema = execution_call["tools"][0]["input_schema"]
    # The disclosed tools in catalog order (the order of the digest lines the model reads).
    assert schema == execution_schema(catalog.tool_names(), FIELDS, 6)
    assert policy_call["max_tokens"] == POLICY_MAX_TOKENS
    enum = policy_call["tools"][0]["input_schema"]["properties"]["tool"]["enum"]
    assert enum == [*catalog.tool_names(), NONE_ANSWER]
    assert policy_call["messages"][0]["content"] == (
        "Rule: Use the lookup tool; do not invent prices.\n\nWhich of these tools does this rule "
        "forbid calling? Answer none if it forbids none of them."
    )
    assert result.notes == [
        f"local planner ({LOCAL}): execution plan call 1: 10 prompt + 5 output tokens in 0 s, "
        "accepted",
        f"local planner ({LOCAL}): no probe of lookup: probes run on the scout's session and this "
        "plan has none (disclosure: all, or no session was passed)",
        f"local planner ({LOCAL}): policy question 1 (instruction 1): 10 prompt + 5 output tokens "
        "in 0 s, answer none",
        f"local planner ({LOCAL}): instruction 2 not asked about: it prohibits nothing",
    ]


def test_expectations_are_written_out() -> None:
    assert describe_expectation("penne") == "equals penne"
    assert describe_expectation(3) == "equals 3"
    assert describe_expectation({"$gt": 0}) == "is greater than 0"
    assert describe_expectation({"$type": "number"}) == "is a number"
    assert describe_expectation({"$type": "array"}) == "is an array"
    assert describe_expectation({"$in": ["unknown", "verified"]}) == "is one of unknown, verified"
    assert describe_expectation({"$ne": "United States"}) == "is not United States"
    assert describe_expectation({"$regex": "(?i)penne"}) == "matches the pattern (?i)penne"
    assert describe_expectation({"$len": {"$gte": 1}}) == "has a length that is at least 1"
    assert describe_expectation({"$len": 2}) == "has exactly 2 entries"
    assert describe_expectation({"$gte": 0, "$lte": 10}) == "is at least 0 and is at most 10"
    assert describe_expectation({"$exists": False}) == "is absent"


# --- validation, the re-ask and what is dropped --------------------------------------------------


async def test_a_placeholder_where_an_integer_is_required_is_sent_back(
    scenario: Scenario,
) -> None:
    """command-r7b wrote get_product_origins(product_ids=["your_product_id_here"])."""
    catalog = await fake_catalog()
    first = execution(
        [lookup_penne(), call("list_items", cursor="your_cursor_here")],
        **{**GOOD_LINEAGE, "price": "step 2: items[0].price"},
    )
    llm = ScriptedLLM(
        [answer(first), answer(execution([lookup_penne()], **GOOD_LINEAGE)), forbidden("none")]
    )
    result = await plan(scenario, catalog, llm)

    reask = llm.calls[1]["messages"][-1]["content"][0]
    assert reask["type"] == "tool_result" and reask["is_error"] is True
    assert reask["content"] == (
        "The JSON config was rejected:\n"
        "- step 2 (list_items): argument 'cursor' of tool list_items must be integer, got "
        'string "your_cursor_here"; if the value comes from an earlier step, write a '
        '{"$from_step": n, "path": "..."} reference instead\n'
        "\nAnswer with the whole corrected JSON config."
    )
    assert result.paths[0].checkpoints == HAPPY_CHECKPOINTS
    assert result.notes[0].startswith(
        f"local planner ({LOCAL}): execution plan call 1: 10 prompt + 5 output tokens in 0 s, "
        "1 problem(s): step 2 (list_items): argument 'cursor' of tool list_items must be "
        'integer, got string "your_cursor_here"'
    )
    assert result.notes[1].endswith("execution plan call 2: 10 prompt + 5 output tokens in 0 s, "
                                     "accepted")


async def test_lineage_must_name_an_existing_tool_step_and_a_path_the_result_has() -> None:
    catalog = await fake_catalog()
    view = PlannerView(disclosed=list(catalog.tools), allowed=catalog.tool_names())
    draft = ToolExecution.model_validate(execution(
        [lookup_penne()],
        slug="step 1: slug",
        price="step 2: price",
        store="find_product.store",
        origin_status="step 1: status",
    ))
    review = review_execution(draft, catalog, view, FIELDS, [penne_observation()], 6)
    assert review.problems == [
        "answer_fields.price: step 2 does not exist (the plan has 1 step(s))",
        "answer_fields.store: 'find_product.store' must read 'step <n>: <path in that result>'",
        "answer_fields.origin_status: status is not in what step 1 (lookup) returned when the "
        'scout made that call: {slug="penne", price=2.49, store="Fake Mart", '
        "origin_status=\"verified\"}; the result has 'origin_status' at origin_status: write "
        "'step 1: origin_status'",
    ]


def test_schema_lookup_walks_refs_optionals_and_lists() -> None:
    schema = {
        "type": "object",
        "properties": {"summary": {"$ref": "#/$defs/Summary"}, "full": {"type": "object"}},
        "$defs": {
            "Summary": {
                "type": "object",
                "properties": {
                    "total_cost": {"type": "number"},
                    "lines": {
                        "anyOf": [
                            {"type": "array", "items": {"$ref": "#/$defs/Line"}},
                            {"type": "null"},
                        ]
                    },
                },
            },
            "Line": {"type": "object", "properties": {"price": {"type": "number"}}},
        },
    }
    assert schema_lookup(schema, "summary.total_cost") == (True, "")
    assert schema_lookup(schema, "summary.lines[*].price") == (True, "")
    assert schema_lookup(schema, "summary.lines[0].price") == (True, "")
    assert schema_lookup(schema, "full.anything") == (None, "")  # a free-form object
    assert schema_lookup(schema, "slug") == (
        False, "the result has no key 'slug' (its keys: summary, full)"
    )
    assert schema_lookup(schema, "summary.coverage") == (
        False, "summary has no key 'coverage' (its keys: total_cost, lines)"
    )
    assert schema_lookup(schema, "summary.total_cost[0]") == (
        False, "summary.total_cost is not a list"
    )


async def test_a_field_read_from_a_sibling_key_is_sent_back_then_repaired(
    scenario: Scenario,
) -> None:
    """command-r7b wrote store <- items[0].brand while items[0].store exists."""
    catalog = await fake_catalog()
    wrong = execution([lookup_penne()], **{**GOOD_LINEAGE, "store": "step 1: slug"})
    llm = ScriptedLLM([answer(wrong), answer(wrong), forbidden("none")])
    scout_obs = scout_result(penne_observation())
    result = await plan(scenario, catalog, llm, scout=scout_obs)

    reask = llm.calls[1]["messages"][-1]["content"][0]["content"]
    assert "- answer_fields.store: step 1: slug reads 'slug'; the result has 'store' at store: " \
           "write 'step 1: store'" in reask
    assert result.paths[0].checkpoints == HAPPY_CHECKPOINTS
    assert any(
        n.endswith("answer_fields.store: the model kept 'step 1: slug'; the result has 'store' "
                   "under the field's own name, used instead")
        for n in result.notes
    )


def nested_catalog() -> Catalog:
    """A find_product shaped like the live pantry tool: top-level query and match, items[]."""
    item = {"type": "object", "properties": {k: {} for k in ("id", "name", "brand", "store")}}
    return Catalog.model_validate({
        "server_name": "t",
        "tools": [{
            "name": "find_product",
            "description": "Look up products.",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                             "required": ["query"]},
            "output_schema": {"type": "object", "properties": {
                "query": {}, "match": {}, "items": {"type": "array", "items": item}}},
        }],
    })


async def test_a_lineage_whose_observed_value_breaks_the_expected_outcome_is_sent_back() -> None:
    """The live cheapest-penne answer: query <- items[0].name, match <- items[0].store and
    store <- items[0].brand all exist in the result, and all three are wrong."""
    catalog = nested_catalog()
    view = PlannerView(disclosed=list(catalog.tools), allowed=catalog.tool_names())
    observed = Observation(
        kind="tool", name="find_product", arguments={"query": "penne"}, summary="2 hits",
        structured={"query": "penne", "match": "direct", "items": [
            {"id": 51, "name": "Penne Rigate 500g", "brand": "Fraser Farms",
             "store": "GreenLeaf Grocers Kitsilano"}]},
    )
    expected = {"query": "penne", "match": "direct", "product_id": {"$type": "number"},
                "store": {"$type": "string"}}
    draft = ToolExecution.model_validate(execution(
        [call("find_product", query="penne")],
        query="step 1: items[0].name",
        match="step 1: items[0].store",
        product_id="step 1: items[0].id",
        store="step 1: items[0].brand",
    ))
    review = review_execution(draft, catalog, view, list(expected), [observed], 6, expected)
    assert review.problems == [
        "answer_fields.query: step 1: items[0].name gives Penne Rigate 500g, which fails "
        "'equals penne' (the expected outcome for query) in what step 1 (find_product) "
        "returned when the scout made that call; the result has 'query' at query: write "
        "'step 1: query'",
        "answer_fields.match: step 1: items[0].store gives GreenLeaf Grocers Kitsilano, which "
        "fails 'equals direct' (the expected outcome for match) in what step 1 (find_product) "
        "returned when the scout made that call; the result has 'match' at match: write "
        "'step 1: match'",
        "answer_fields.store: step 1: items[0].brand reads 'brand'; the result has 'store' at "
        "items[0].store: write 'step 1: items[0].store'",
    ]
    assert {n: lin.text() for n, lin in review.repairs.items()} == {
        "query": "step 1: query", "match": "step 1: match", "store": "step 1: items[0].store",
    }
    # product_id <- items[0].id is right (51 is a number) and is not touched.
    assert "product_id" not in review.bad_fields
    # Without an observation the schema still offers the same-name key, but cannot check values.
    blind = review_execution(draft, catalog, view, list(expected), [], 6, expected)
    assert {n: lin.text() for n, lin in blind.repairs.items()} == {
        "query": "step 1: query", "match": "step 1: match", "store": "step 1: items[0].store",
    }


def plan_recipe_catalog() -> Catalog:
    """plan_recipe as the live pantry server declares it: a required lean ``summary`` and a
    ``full`` plan that is null unless ``verbose=True``."""
    output = {
        "type": "object",
        "required": ["summary"],
        "properties": {
            "full": {"anyOf": [{"$ref": "#/$defs/Plan"}, {"type": "null"}], "default": None},
            "summary": {"$ref": "#/$defs/Summary"},
        },
        "$defs": {
            "Plan": {"type": "object", "properties": {
                "recipe_slug": {"type": "string"}, "total_cost": {"type": "number"},
                "line_items": {"type": "array", "items": {"type": "object", "properties": {
                    "price": {"type": "number"}}}}}},
            "Summary": {
                "type": "object",
                "required": ["recipe_slug", "total_cost", "origin_status", "coverage", "lines"],
                "properties": {
                    "recipe_slug": {"type": "string"},
                    "total_cost": {"type": "number"},
                    "origin_status": {"type": "string"},
                    "coverage": {"$ref": "#/$defs/Coverage"},
                    "lines": {"type": "array", "items": {"$ref": "#/$defs/Line"}},
                },
            },
            "Coverage": {"type": "object", "required": ["spend_fraction"],
                         "properties": {"spend_fraction": {"type": "number"}}},
            "Line": {"type": "object", "required": ["price", "origin_status"],
                     "properties": {"price": {"type": "number"},
                                    "origin_status": {"type": "string"}}},
        },
    }
    return Catalog.model_validate({"server_name": "t", "tools": [{
        "name": "plan_recipe",
        "description": "Plan a recipe.",
        "input_schema": {"type": "object", "properties": {"slug": {"type": "string"}},
                         "required": ["slug"]},
        "output_schema": output,
    }]})


def test_the_output_schema_finds_the_field_when_nothing_was_observed() -> None:
    """The live tomato-penne-boycott answer: told the keys were summary and full, command-r7b
    traced whole objects (total_cost <- full) and keys that do not exist (coverage_note)."""
    catalog = plan_recipe_catalog()
    view = PlannerView(disclosed=list(catalog.tools), allowed=catalog.tool_names())
    fields = ["recipe_slug", "total_cost", "origin_status", "coverage.spend_fraction",
              "lines[*].price", "lines"]
    draft = ToolExecution.model_validate(execution(
        [call("plan_recipe", slug="tomato_penne")],
        recipe_slug="step 1: summary",
        total_cost="step 1: full.total_cost",
        origin_status="step 1: coverage_note",
        **{"coverage.spend_fraction": "step 1: coverage.spend_fraction",
           "lines[*].price": "step 1: full", "lines": "step 1: summary.lines"},
    ))
    review = review_execution(draft, catalog, view, fields, [], 6)
    assert review.problems == [
        "answer_fields.recipe_slug: step 1: summary reads 'summary'; the result has "
        "'recipe_slug' at summary.recipe_slug: write 'step 1: summary.recipe_slug'",
        "answer_fields.total_cost: step 1: full.total_cost goes through a value the output "
        "schema marks optional (null or absent unless asked for); the result has 'total_cost' "
        "at summary.total_cost: write 'step 1: summary.total_cost'",
        "answer_fields.origin_status: step 1 (plan_recipe) cannot return coverage_note: the "
        "result has no key 'coverage_note' (its keys: full, summary); the result has "
        "'origin_status' at summary.origin_status: write 'step 1: summary.origin_status'",
        "answer_fields.coverage.spend_fraction: step 1 (plan_recipe) cannot return "
        "coverage.spend_fraction: the result has no key 'coverage' (its keys: full, summary); "
        "the result has 'spend_fraction' at summary.coverage.spend_fraction: write 'step 1: "
        "summary.coverage.spend_fraction'",
        "answer_fields.lines[*].price: step 1: full reads 'full'; the result has 'price' at "
        "summary.lines[*].price: write 'step 1: summary.lines[*].price'",
    ]
    # summary.lines is right and untouched; origin_status prefers summary.origin_status over
    # summary.lines[*].origin_status (no extra list step).
    assert "lines" not in review.bad_fields


def test_a_tie_among_maybe_absent_paths_keeps_looking_for_one_that_is_always_there() -> None:
    from mcpsim.execution_planner import FieldNode, Lineage, nearest_field_path

    maybe = [
        FieldNode(("full",), False),
        FieldNode(("full", "a"), False),
        FieldNode(("full", "a", "price"), False),
        FieldNode(("full", "b"), False),
        FieldNode(("full", "b", "price"), False),
    ]
    sure = [FieldNode(("summary",), True), FieldNode(("summary", "price"), True)]
    found = nearest_field_path("price", Lineage(1, "full"), maybe + sure)
    assert found == Lineage(1, "summary.price")
    assert nearest_field_path("price", Lineage(1, "full"), maybe) is None  # a tie, nothing sure
    unique = nearest_field_path("price", Lineage(1, "full"), maybe[:3])
    assert unique == Lineage(1, "full.a.price")  # the only candidate, though it may be absent


async def test_only_the_same_call_grounds_a_lineage_value() -> None:
    """The scout's list_items() is list_items(cursor=0, limit=2); cursor=3 is another page."""
    catalog = await fake_catalog()
    view = PlannerView(disclosed=list(catalog.tools), allowed=catalog.tool_names())
    first_page = Observation(
        kind="tool", name="list_items", arguments={}, summary="page 1",
        structured={"items": [ITEMS["basil"], ITEMS["garlic"]], "next_cursor": 2, "total": 5},
    )
    expected = {"slug": "penne"}

    def review_for(**arguments: Any) -> list[str]:
        draft = ToolExecution.model_validate(
            execution([call("list_items", **arguments)], slug="step 1: items[0].slug")
        )
        return review_execution(draft, catalog, view, ["slug"], [first_page], 6, expected).problems

    assert review_for(cursor=3, limit=1) == []  # not the observed call: nothing to check against
    assert review_for(cursor=0) == [
        "answer_fields.slug: step 1: items[0].slug gives basil, which fails 'equals penne' (the "
        "expected outcome for slug) in what step 1 (list_items) returned when the scout made "
        "that call"
    ]


async def test_unused_and_repeated_steps_are_dropped_and_lineage_follows(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    draft = execution(
        [lookup_penne(), call("echo", text="not needed"), lookup_penne()],
        **{**GOOD_LINEAGE, "price": "step 3: price"},
    )
    llm = ScriptedLLM([answer(draft), forbidden("none")])
    result = await plan(scenario, catalog, llm)

    happy = result.paths[0]
    assert [s.tool for s in happy.steps] == ["lookup", None]
    assert happy.checkpoints == HAPPY_CHECKPOINTS  # price now names step 1, the same call
    assert len(llm.calls) == 2  # no re-ask: dropping needs no help from the model
    assert f"local planner ({LOCAL}): step 2 (echo) dropped: no answer field and no later step " \
           "uses its result" in result.notes
    assert f"local planner ({LOCAL}): step 3 (lookup) dropped: repeats step 1" in result.notes


async def test_references_are_renumbered_when_an_earlier_step_is_dropped(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    draft = execution(
        [
            call("echo", text="unused"),
            call("list_items", cursor=0),
            call("lookup", slug={"$from_step": 2, "path": "items[0].slug"}),
        ],
        slug="step 3: slug", price="step 3: price", store="step 3: store",
        origin_status="step 3: origin_status",
    )
    llm = ScriptedLLM([answer(draft), forbidden("none")])
    result = await plan(scenario, catalog, llm)
    steps = result.paths[0].steps
    assert [(s.tool, s.arguments_sketch) for s in steps[:-1]] == [
        ("list_items", {"cursor": 0}),
        ("lookup", {"slug": {"$from_step": 1, "path": "items[0].slug"}}),
    ]
    assert result.paths[0].checkpoints[0] == "final_result: slug equals tool_result[lookup] slug"


async def test_a_step_still_invalid_after_the_re_ask_is_dropped_with_its_fields(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    bad = execution(
        [lookup_penne(), call("list_items", cursor="first page")],
        **{**GOOD_LINEAGE, "price": "step 2: items[0].price"},
    )
    llm = ScriptedLLM([answer(bad), answer(bad), forbidden("none")])
    result = await plan(scenario, catalog, llm)
    happy = result.paths[0]
    assert [s.tool for s in happy.steps] == ["lookup", None]
    assert "final_result: price equals tool_result[lookup] price" not in happy.checkpoints
    assert "final_result: price is greater than 0" in happy.checkpoints  # the spec still holds
    notes = "\n".join(result.notes)
    assert "step 2 (list_items) dropped: argument 'cursor' of tool list_items must be integer" \
        in notes
    assert "answer_fields.price dropped: step 2 was dropped" in notes


async def test_an_unparseable_re_ask_reply_salvages_the_previous_answer(
    scenario: Scenario,
) -> None:
    from tests.fake_llm import text_response

    catalog = await fake_catalog()
    first = execution([lookup_penne()], **{**GOOD_LINEAGE, "price": "step 2: price"})
    llm = ScriptedLLM([answer(first), text_response('{"steps": [{"tool": "loo'),
                       forbidden("none")])
    result = await plan(scenario, catalog, llm)
    happy = result.paths[0]
    assert [s.tool for s in happy.steps] == ["lookup", None]
    assert "final_result: price equals tool_result[lookup] price" not in happy.checkpoints
    assert "final_result: store equals tool_result[lookup] store" in happy.checkpoints
    notes = [n.split(": ", 1)[1] for n in result.notes]
    assert "the re-asked answer did not parse; the previous answer is salvaged" in notes
    assert "answer_fields.price dropped (no lineage checkpoint): step 2 does not exist (the plan " \
           "has 1 step(s))" in notes


async def test_no_valid_tool_step_is_a_plan_error(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    bad = execution([call("lookup", query="penne")], **GOOD_LINEAGE)
    llm = ScriptedLLM([answer(bad), answer(bad)])
    with pytest.raises(PlanError, match="no valid tool step left"):
        await plan(scenario, catalog, llm)
    assert len(llm.calls) == 2


async def test_an_answer_that_is_not_json_twice_is_a_plan_error(scenario: Scenario) -> None:
    from tests.fake_llm import text_response

    catalog = await fake_catalog()
    llm = ScriptedLLM([text_response("I would call lookup."), text_response("lookup(penne)")])
    with pytest.raises(PlanError, match="no usable execution plan in 2 attempts"):
        await plan(scenario, catalog, llm)


# --- the probe ---------------------------------------------------------------------------------


def test_mutations() -> None:
    assert mutate("penne") == "pene"
    assert mutate("tomato_penne") == "tomatopenne"  # index 6 of 12 is the underscore
    assert mutate("abc") == "ac"
    assert mutate("ab") is None
    assert mutate(51) == PROBE_INTEGER
    assert mutate(PROBE_INTEGER) is None
    assert mutate(True) is None
    assert mutate(1.5) is None
    assert mutate(["United States"]) is None


async def test_a_probe_error_makes_a_recovery_path_that_quotes_the_real_error(
    scenario: Scenario,
) -> None:
    llm = ScriptedLLM([answer(execution([lookup_penne()], **GOOD_LINEAGE)), forbidden("none")])
    observed = scout_result(penne_observation())
    async with open_session() as session:
        catalog = await session.catalog()
        result = await plan(scenario, catalog, llm, scout=observed, probe_session=session)
        assert session.tool_calls == 1  # the probe, nothing else

    assert [p.id for p in result.paths] == ["happy", "recovery-lookup"]
    recovery = result.paths[1]
    assert recovery.kind == "recovery"
    assert [(s.tool, s.arguments_sketch, s.expect_error) for s in recovery.steps] == [
        ("lookup", {"slug": "pene"}, True),
        ("lookup", {"slug": "penne"}, False),
        (None, {}, False),
    ]
    assert recovery.steps[0].success_looks_like == f"The server rejects it: {UNKNOWN_PENE}"
    assert recovery.checkpoints == [
        f"tool_result[lookup]: is an error when slug is pene, saying {UNKNOWN_PENE}",
        "transcript: after that error the agent calls lookup with slug penne and answers from "
        "that result",
        "final_result: slug equals tool_result[lookup] slug (step 2, the call with slug penne)",
        "final_result: price equals tool_result[lookup] price (step 2, the call with slug penne)",
        "final_result: store equals tool_result[lookup] store (step 2, the call with slug penne)",
        "final_result: origin_status equals tool_result[lookup] origin_status (step 2, the call "
        "with slug penne)",
    ]
    probe_obs = observed.observations[-1]
    assert (probe_obs.name, probe_obs.arguments, probe_obs.is_error, probe_obs.probe) == (
        "lookup", {"slug": "pene"}, True, True
    )
    assert probe_obs.summary == UNKNOWN_PENE
    assert observed.tool_calls == 2
    assert observed.notes == [
        'probe: lookup(slug="pene") → error (the planner\'s variant of happy step 1: slug penne '
        "-> pene)"
    ]


async def test_a_probe_answer_makes_a_boundary_path_stating_what_came_back(
    scenario_data: dict[str, Any],
) -> None:
    expected = {"price": {"$gt": 0}, "store": {"$type": "string"}}
    scenario = parse_scenario(local_data(
        scenario_data,
        instructions=["Report the first item exactly."],
        expected_outcome={"text": "The first item's price and store.", "json": expected},
    ))
    # The scout's zero-argument call is list_items(cursor=0, limit=2) once defaults are filled.
    first_page = Observation(
        kind="tool", name="list_items", arguments={}, summary="page 1",
        structured={"items": [ITEMS["basil"], ITEMS["garlic"]], "next_cursor": 2, "total": 5},
    )
    llm = ScriptedLLM([
        answer({"steps": [call("list_items", "First page", cursor=0, limit=2)],
                "answer_fields": {"price": "step 1: items[0].price",
                                  "store": "step 1: items[0].store"}}),
    ])
    observed = scout_result(first_page)
    async with open_session() as session:
        catalog = await session.catalog()
        result = await plan(scenario, catalog, llm, scout=observed, probe_session=session)

    boundary = result.paths[1]
    assert boundary.id == "boundary-list_items" and boundary.kind == "boundary"
    assert [(s.tool, s.arguments_sketch, s.expect_error) for s in boundary.steps] == [
        ("list_items", {"cursor": PROBE_INTEGER, "limit": 2}, False),
        ("list_items", {"cursor": 0, "limit": 2}, False),
        (None, {}, False),
    ]
    # What list_items(cursor=999999) really returned, against the scout's first page: no items
    # and no next cursor (total is 5 both times, so it is not a fact about the mutation).
    assert boundary.checkpoints[:3] == [
        "tool_result[list_items]: items has 0 entries when cursor is 999999",
        "tool_result[list_items]: next_cursor equals null when cursor is 999999",
        "final_result: does not present the list_items result for cursor=999999 as the answer; "
        "its values come from the call with cursor 0",
    ]
    assert boundary.checkpoints[3:] == [
        "final_result: price equals tool_result[list_items] items[0].price (step 2, the call "
        "with cursor 0)",
        "final_result: store equals tool_result[list_items] items[0].store (step 2, the call "
        "with cursor 0)",
    ]
    assert result.paths[0].checkpoints == [
        "final_result: price equals tool_result[list_items] items[0].price",
        "final_result: store equals tool_result[list_items] items[0].store",
        "final_result: price is greater than 0",
        "final_result: store is a string",
    ]
    assert boundary.steps[0].success_looks_like == (
        "The server answers without an error (items has 0 entries when cursor is 999999; "
        "next_cursor equals null when cursor is 999999); that result is not the answer to the "
        "request"
    )


async def test_write_and_expensive_tools_are_never_probed() -> None:
    server = build_server()

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def record_note(text: str) -> dict[str, Any]:
        """Store a note."""
        raise AssertionError("a write tool must never be probed")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def submit_note(text: str) -> dict[str, Any]:
        """Submit a note for review."""
        raise AssertionError("a submit_ tool must never be probed, whatever its hint says")

    async with open_session(server) as session:
        catalog = await session.catalog()
        for name in ("record_note", "submit_note"):
            assert not is_read_only(catalog.tool(name))
            assert probe_refusal(catalog.tool(name)) == "it is not read-only (a write tool)"
        assert probe_refusal(catalog.tool("expensive_report")) == (
            "its description says it costs money, credits or time"
        )
        assert probe_refusal(catalog.tool("lookup")) is None
        calls = [
            PlannedCall(tool="record_note", arguments={"text": "penne"}),
            PlannedCall(tool="submit_note", arguments={"text": "penne"}),
            PlannedCall(tool="expensive_report", arguments={"topic": "penne"}),
        ]
        choice, reasons = choose_probe(calls, catalog)
        assert choice is None
        assert reasons == [
            "step 1 (record_note): it is not read-only (a write tool)",
            "step 2 (submit_note): it is not read-only (a write tool)",
            "step 3 (expensive_report): its description says it costs money, credits or time",
        ]
        observed = scout_result(budget=5, calls=0)
        for name in ("record_note", "submit_note", "expensive_report"):
            with pytest.raises(ValueError, match=f"refusing to probe {name}"):
                await probe(observed, session, catalog.tool(name), {"text": "x"}, why="test")
        assert session.tool_calls == 0
        assert observed.tool_calls == 0 and observed.observations == []


async def test_a_plan_whose_only_step_is_expensive_gets_no_variant(scenario: Scenario) -> None:
    llm = ScriptedLLM([
        answer(execution([call("expensive_report", topic="penne")],
                         slug="step 1: topic", price="step 1: report", store="step 1: report",
                         origin_status="step 1: report")),
        forbidden("none"),
    ])
    async with open_session() as session:
        catalog = await session.catalog()
        result = await plan(scenario, catalog, llm, scout=scout_result(), probe_session=session)
        assert session.tool_calls == 0
    assert [p.id for p in result.paths] == ["happy"]
    assert f"local planner ({LOCAL}): no probe: step 1 (expensive_report): its description says " \
           "it costs money, credits or time" in result.notes


class BrokenSession:
    """An MCP session that went away while the local model was thinking."""

    tool_calls = 0

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        raise RuntimeError("session closed by the server")


async def test_a_probe_that_cannot_reach_the_server_costs_the_variant_not_the_plan(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    llm = ScriptedLLM([answer(execution([lookup_penne()], **GOOD_LINEAGE)), forbidden("none")])
    observed = scout_result(penne_observation())
    session: Any = BrokenSession()
    result = await plan(scenario, catalog, llm, scout=observed, probe_session=session)
    assert [p.id for p in result.paths] == ["happy"]
    assert f"local planner ({LOCAL}): no probe: lookup could not be called (RuntimeError: " \
           "session closed by the server)" in result.notes
    assert observed.tool_calls == 1 and not any(o.probe for o in observed.observations)


async def test_a_spent_scout_budget_means_no_probe(scenario: Scenario) -> None:
    llm = ScriptedLLM([answer(execution([lookup_penne()], **GOOD_LINEAGE)), forbidden("none")])
    observed = scout_result(penne_observation(), budget=1)
    async with open_session() as session:
        catalog = await session.catalog()
        result = await plan(scenario, catalog, llm, scout=observed, probe_session=session)
        assert session.tool_calls == 0
    assert [p.id for p in result.paths] == ["happy"]
    assert f"local planner ({LOCAL}): no probe: the scout budget of 1 tool call(s) is spent" \
        in result.notes


async def test_enum_and_reference_arguments_are_not_mutated() -> None:
    catalog = Catalog.model_validate({
        "server_name": "t",
        "tools": [{
            "name": "rank",
            "description": "Rank products.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "order": {"type": "string", "enum": ["asc", "desc"]},
                    "search": {"type": "string"},
                },
            },
        }],
    })
    first = PlannedCall(tool="rank", arguments={"search": {"$from_step": 1, "path": "x"}})
    second = PlannedCall(tool="rank", arguments={"order": "asc", "search": "penne"})
    choice, reasons = choose_probe([first, second], catalog)
    assert reasons == ["step 1 (rank) needs an earlier step's result"]
    assert choice is not None
    assert (choice.step, choice.argument, choice.original, choice.mutated) == (
        2, "search", "penne", "pene"
    )


# --- policy paths --------------------------------------------------------------------------------


async def test_policy_questions_build_one_path_per_forbidden_tool(
    scenario_data: dict[str, Any],
) -> None:
    instructions = [
        "Use lookup; do not call expensive_report.",
        "Never call lookup twice.",
        "Report origin_status exactly as returned.",
        "Do not use echo to repeat yourself.",
        "Avoid list_items.",
    ]
    scenario = parse_scenario(local_data(scenario_data, instructions=instructions))
    catalog = await fake_catalog()
    llm = ScriptedLLM([
        answer(execution([lookup_penne()], **GOOD_LINEAGE)),
        forbidden("expensive_report"),
        forbidden("lookup"),
        forbidden("echo"),
    ])
    result = await plan(scenario, catalog, llm)

    assert [(p.id, p.kind) for p in result.paths] == [
        ("happy", "happy"),
        ("policy-expensive_report", "policy"),
        ("policy-echo", "policy"),
    ]
    happy, report_policy, echo_policy = result.paths
    assert report_policy.checkpoints == [
        "transcript: no call to expensive_report", *HAPPY_CHECKPOINTS[:4]
    ]
    assert report_policy.steps[:-1] == happy.steps[:-1]
    assert report_policy.steps[-1].intent == (
        "Compose the final answer, keeping this rule: Use lookup; do not call expensive_report."
    )
    assert echo_policy.checkpoints[0] == "transcript: no call to echo"
    # Three questions at most, one system prompt (the server reuses its cached prefix).
    questions = [c for c in llm.calls if c["tools"][0]["name"] == POLICY_TOOL_NAME]
    assert len(questions) == MAX_POLICY_QUESTIONS
    assert len({c["system"] for c in questions}) == 1
    notes = [n.split(": ", 1)[1] for n in result.notes]
    assert "instruction 3 not asked about: it prohibits nothing" in notes
    assert "instruction 5 not asked about: at most 3 policy questions per plan" in notes
    assert any(n.startswith("policy question 2 (instruction 2)") and
               n.endswith("answer lookup; ignored, the happy path calls lookup") for n in notes)
    # No two paths share their steps.
    keys = {json.dumps([s.model_dump() for s in p.steps], sort_keys=True) for p in result.paths}
    assert len(keys) == len(result.paths)


async def test_a_forbidden_tool_the_rule_does_not_name_is_ignored(
    scenario_data: dict[str, Any],
) -> None:
    """command-r7b answered get_product_origins for "never round a price or guess a store"."""
    instructions = ["Quote the store exactly; never round a price or guess a store."]
    scenario = parse_scenario(local_data(scenario_data, instructions=instructions))
    catalog = await fake_catalog()
    llm = ScriptedLLM([
        answer(execution([lookup_penne()], **GOOD_LINEAGE)), forbidden("list_items"),
    ])
    result = await plan(scenario, catalog, llm)
    assert [p.id for p in result.paths] == ["happy"]
    assert result.notes[-1] == (
        f"local planner ({LOCAL}): policy question 1 (instruction 1): 10 prompt + 5 output "
        "tokens in 0 s, answer list_items; ignored, the prohibition (never round a price or "
        "guess a store) does not name list_items"
    )


def test_prohibiting_clauses_and_the_tools_they_name() -> None:
    from mcpsim.execution_planner import clause_names_tool, prohibiting_clauses

    rule = "Use find_product (results come back cheapest first); do not run a planning tool."
    assert prohibiting_clauses(rule) == ["do not run a planning tool"]
    assert clause_names_tool(prohibiting_clauses(rule), "plan_recipe")  # planning ~ plan
    assert not clause_names_tool(prohibiting_clauses(rule), "find_product")
    boycott = "Use plan_recipe; do not price the basket yourself from product lookups."
    assert clause_names_tool(prohibiting_clauses(boycott), "find_product")
    assert not clause_names_tool(prohibiting_clauses(boycott), "plan_recipe")
    assert prohibiting_clauses("Report origin_status as returned.") == []


async def test_two_instructions_naming_one_tool_make_one_policy_path(
    scenario_data: dict[str, Any],
) -> None:
    instructions = ["Do not call echo.", "Never use echo for anything."]
    scenario = parse_scenario(local_data(scenario_data, instructions=instructions))
    catalog = await fake_catalog()
    llm = ScriptedLLM([
        answer(execution([lookup_penne()], **GOOD_LINEAGE)), forbidden("echo"), forbidden("echo"),
    ])
    result = await plan(scenario, catalog, llm)
    assert [p.id for p in result.paths] == ["happy", "policy-echo"]
    assert f"local planner ({LOCAL}): instruction 2 also forbids echo; one policy path covers it" \
        in result.notes


# --- the hosted profile --------------------------------------------------------------------------


class NoCallSession:
    """A session stand-in that fails the test if anything is sent through it."""

    tool_calls = 0

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        raise AssertionError(f"the hosted planner must not call {name}")


async def test_the_hosted_profile_is_unchanged(scenario: Scenario) -> None:
    """A hosted planner still writes every path in one call with the full prompt, and never
    probes even when handed a session."""
    catalog = await fake_catalog()
    hosted = scenario.model_copy(
        update={"models": scenario.models.model_copy(update={"planner": "claude-opus-5-5"})}
    )
    step = {"intent": "look up penne", "tool": "lookup", "arguments_sketch": {"slug": "penne"},
            "success_looks_like": "a price", "expect_error": False}
    payload = {"paths": [{"id": "happy", "kind": "happy", "title": "t", "rationale": "r",
                          "steps": [step], "checkpoints": ["final_result: slug equals penne"]}]}
    llm = ScriptedLLM([structured_response(PLAN_TOOL_NAME, payload)])
    session: Any = NoCallSession()
    result = await plan(hosted, catalog, llm, scout=scout_result(), probe_session=session)
    assert len(result.paths) == 1 and len(llm.calls) == 1
    assert result.notes == []
    assert "RESOURCES" in llm.calls[0]["system"]
    assert llm.calls[0]["tools"][0]["name"] == PLAN_TOOL_NAME
    schema = llm.calls[0]["tools"][0]["input_schema"]
    assert "maxItems" not in schema["properties"]["paths"]
    assert json.dumps(schema).count("maxLength") == 0


# --- the runner: one session for scout, plan and probe -------------------------------------------


def test_plan_scenario_probes_on_the_scout_session_and_records_it(
    scenario_data: dict[str, Any], tmp_path: FsPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = local_data(
        scenario_data,
        server=fake_server_stdio_spec(),
        tools={"disclosure": "progressive", "initial": ["lookup", "fail", "list_items"]},
        budgets={"max_turns": 6, "max_tool_calls": 6},
    )
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    llm = ScriptedLLM([answer(execution([lookup_penne()], **GOOD_LINEAGE)), forbidden("none")])

    def fake_make_llm_for(scenario: Scenario, role: str) -> ScriptedLLM:
        assert role == "planner"
        return llm

    monkeypatch.setattr(runner, "make_llm_for", fake_make_llm_for)
    plan_path = runner.plan_scenario(path, tmp_path / "runs")

    result = ExecutionPlan.load(plan_path)
    assert [p.id for p in result.paths] == ["happy", "recovery-lookup"]
    assert result.paths[1].steps[0].success_looks_like == f"The server rejects it: {UNKNOWN_PENE}"
    saved = ScoutResult.load(plan_path.parent / SCOUT_FILE)
    # Budget max(2, 6 // 2) = 3: the scout kept one call back, the probe spent it.
    assert (saved.budget, saved.tool_calls) == (3, 3)
    assert saved.notes[0] == "1 of 3 tool call(s) left for the planner's probes"
    probes = [o for o in saved.observations if o.probe]
    assert [(o.name, o.arguments, o.is_error) for o in probes] == [
        ("lookup", {"slug": "pene"}, True)
    ]
    assert [(o.name, o.arguments) for o in saved.observations if o.kind == "tool"][0] == (
        "lookup", {"slug": "penne"}
    )
    assert saved.planner_prompt_chars > 0


async def test_the_scout_reserve_leaves_calls_for_the_probe(scenario_data: dict[str, Any]) -> None:
    data = local_data(scenario_data, tools={"disclosure": "progressive",
                                            "initial": ["lookup", "fail", "list_items"]})
    scenario = parse_scenario(data)
    async with open_session() as session:
        catalog = await session.catalog()
        full = await scout(scenario, catalog, session, budget=3)
        held = await scout(scenario, catalog, session, budget=3, reserve=1)
    assert full.tool_calls == 3
    assert held.tool_calls == 2 and held.budget == 3
