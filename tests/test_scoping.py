"""Scoping tests: which tools, in which order, for which scenario (DESIGN §2 "Tool scoping").

The pantry catalog fixture is trimmed from ``mcpsim catalog --json`` against the real server
(names, descriptions, annotations, input schemas, top-level output keys), so these tests say
what the real suite will see.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcpsim.mcpclient import Catalog, ToolInfo
from mcpsim.scenario import Scenario, load_scenario, parse_scenario
from mcpsim.scoping import (
    covers_expected_key,
    expected_top_level_keys,
    first_sentence,
    initial_tools,
    is_write_tool,
    output_key_names,
    plain_expected_value,
    rank_tools,
    relevance,
    relevance_to_terms,
    scenario_terms,
    stem,
    tokens,
    tool_terms,
    write_intent,
)

FIXTURES = Path(__file__).parent / "fixtures"
PANTRY = Path(__file__).resolve().parent.parent / "scenarios" / "pantry"
PANTRY_TOOL_NAMES = [
    "list_recipes",
    "get_recipe",
    "list_products",
    "find_product",
    "get_product",
    "plan_recipe",
    "plan_from_text",
    "plan_week",
    "get_product_origins",
    "rank_products_by_origin",
    "origin_triage",
    "submit_origin_evidence",
    "list_origin_submissions",
    "review_origin_submission",
    "pipeline_status",
]


def pantry_catalog() -> Catalog:
    raw = json.loads((FIXTURES / "pantry_catalog.json").read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    return Catalog.model_validate(raw)


def pantry_scenario(name: str) -> Scenario:
    return load_scenario(PANTRY / f"{name}.yaml")


@pytest.fixture
def catalog() -> Catalog:
    return pantry_catalog()


def cheapest_penne_like(server: dict[str, Any]) -> Scenario:
    """A cheapest-penne-shaped scenario that does not depend on the committed YAML."""
    return parse_scenario(
        {
            "name": "cheapest-penne-like",
            "role": "A bargain hunter who wants one answer.",
            "goal": (
                "Tell me the cheapest penne product in the catalog, its price, and which store "
                "sells it at that price."
            ),
            "instructions": [
                "Use find_product (results come back cheapest first); do not run a planning "
                "tool for a lookup.",
                "Quote the product name, price and store exactly as the product record has "
                "them; never round a price or guess a store.",
            ],
            "expected_outcome": {
                "text": "The cheapest penne product by name, with its exact price and store.",
                "json": {
                    "query": "penne",
                    "match": "direct",
                    "product_id": {"$type": "number"},
                    "product_name": {"$regex": "(?i)penne"},
                    "store": {"$type": "string"},
                    "price": {"$gt": 0},
                    "origin_status": {"$in": ["unknown", "resolved", "verified"]},
                },
            },
            "server": server,
        },
        source="test",
    )


# --- tokens -----------------------------------------------------------------------------------


def test_tokens_lowercase_split_stem_and_drop_stopwords() -> None:
    assert tokens("Find the cheapest Penne; quote its price and store") == {
        "find",
        "cheapest",
        "penne",
        "quote",
        "price",
        "store",
    }
    assert tokens("origin_status lines[*].origin_country") == {
        "origin",
        "statu",
        "line",
        "country",
    }
    assert tokens("Free — no LLM calls.") == set()
    assert stem("prices") == "price" and stem("status") == "statu" and stem("has") == "has"
    assert stem("class") == "class" and stem("penne") == "penne"


def test_first_sentence() -> None:
    assert first_sentence("Look up a product.\n  Returns price. Free.") == "Look up a product."
    assert first_sentence("no terminal punctuation") == "no terminal punctuation"


def test_expected_top_level_keys_and_plain_values() -> None:
    spec = {
        "query": "penne",
        "match": "direct",
        "price": {"$gt": 0},
        "lines[*].origin_country": {"$ne": "United States"},
        "coverage.spend_fraction": {"$type": "number"},
        "budget": 120,
        "visible": True,
    }
    assert expected_top_level_keys(spec) == [
        "query",
        "match",
        "price",
        "lines",
        "coverage",
        "budget",
        "visible",
    ]
    assert expected_top_level_keys(None) == []
    assert plain_expected_value(spec, "query") == (True, "penne")
    assert plain_expected_value(spec, "budget") == (True, 120)
    assert plain_expected_value(spec, "price") == (False, None), "an operator object"
    assert plain_expected_value(spec, "visible") == (False, None), "booleans are not plain"
    assert plain_expected_value(spec, "nope") == (False, None)
    assert plain_expected_value(None, "query") == (False, None)


# --- scenario and tool terms, relevance ------------------------------------------------------


def test_scenario_terms_cover_goal_instructions_outcome_keys_and_plain_values(
    scenario_data: dict[str, Any],
) -> None:
    scenario = cheapest_penne_like(scenario_data["server"])
    terms = scenario_terms(scenario)
    assert {"cheapest", "penne", "price", "store"} <= terms, "from the goal"
    assert {"find", "product", "quote"} <= terms, "from the instructions"
    assert {"query", "match", "id", "name", "origin", "statu"} <= terms, "from the json keys"
    assert "direct" in terms, "a plain string value of the json spec"
    assert "regex" not in terms and "number" not in terms, "operator objects are not terms"
    assert "bargain" not in terms, "the role's persona is not a term"


def test_tool_terms_and_output_keys(catalog: Catalog) -> None:
    find_product = catalog.tool("find_product")
    assert output_key_names(find_product.output_schema) == [
        "items",
        "match",
        "note",
        "query",
        "tokens",
        "total",
    ]
    assert output_key_names(catalog.tool("origin_triage").output_schema) == [
        "product_id",
        "product_name",
        "reason",
        "status",
    ], "through the SDK's single `result` wrapper, the list element keys"
    assert output_key_names(None) == [] and output_key_names({"type": "string"}) == []
    terms = tool_terms(find_product)
    assert {"find", "product", "query", "match", "lat", "lon", "limit", "cheapest"} <= terms


def test_relevance_weights_name_output_input_and_description() -> None:
    tool = ToolInfo(
        name="find_product",
        description="Look up catalog products. Items come back cheapest first.",
        input_schema={"type": "object", "properties": {"query": {}, "limit": {}}},
        output_schema={"type": "object", "properties": {"items": {}, "match": {}, "query": {}}},
    )
    assert relevance_to_terms({"product"}, tool) == 3 + 1, "name token 3, description word 1"
    assert relevance_to_terms({"match"}, tool) == 2, "output key"
    assert relevance_to_terms({"limit"}, tool) == 2, "input property"
    assert relevance_to_terms({"query"}, tool) == 2 + 2, "both an input and an output key"
    assert relevance_to_terms({"cheapest"}, tool) == 1
    assert relevance_to_terms({"nothing"}, tool) == 0
    assert relevance_to_terms(set(), tool) == 0


def test_rank_tools_keeps_catalog_order_on_ties(catalog: Catalog) -> None:
    ranked = rank_tools(set(), catalog.tools)
    assert [t.name for t, _ in ranked] == PANTRY_TOOL_NAMES
    assert all(score == 0 for _, score in ranked)
    ranked = rank_tools({"recipe"}, catalog.tools)
    names = [t.name for t, score in ranked if score > 0]
    assert names[:2] == ["list_recipes", "get_recipe"], "name hits first, catalog order on ties"


# --- write tools and write intent ------------------------------------------------------------


def test_is_write_tool_from_annotations_then_name(catalog: Catalog) -> None:
    writes = [t.name for t in catalog.tools if is_write_tool(t)]
    assert writes == ["submit_origin_evidence", "review_origin_submission"]
    assert is_write_tool(ToolInfo(name="create_order"))
    assert is_write_tool(ToolInfo(name="set_location"))
    assert is_write_tool(ToolInfo(name="note", annotations={"read_only_hint": False}))
    assert is_write_tool(ToolInfo(name="note", annotations={"readOnlyHint": False}))
    assert is_write_tool(ToolInfo(name="wipe", annotations={"destructive_hint": True}))
    assert not is_write_tool(ToolInfo(name="lookup"))
    assert not is_write_tool(ToolInfo(name="lookup", annotations={"read_only_hint": True}))
    assert not is_write_tool(ToolInfo(name="submitted_count")), "submit_ prefix, not 'submitted'"


def test_write_intent_reads_imperatives_not_nouns(scenario_data: dict[str, Any]) -> None:
    assert write_intent(cheapest_penne_like(scenario_data["server"])) is False, (
        "'as the product record has them' is a noun, not an instruction to record"
    )

    def with_text(goal: str, *instructions: str) -> Scenario:
        return parse_scenario(
            {**scenario_data, "goal": goal, "instructions": list(instructions)}, source="t"
        )

    assert write_intent(with_text("Record the label I read on the olive oil."))
    assert write_intent(with_text("Find the product, then submit the reading."))
    assert write_intent(with_text("Find the product.", "After submitting, list the queue."))
    assert write_intent(with_text("Find the product.", "Use submit_origin_evidence for it."))
    assert write_intent(with_text("Report a label reading for the olive oil."))
    assert write_intent(with_text("Find the product.", "Do not call review_origin_submission."))
    assert not write_intent(with_text("Find the price.", "When the server rejects a name, retry."))
    assert not write_intent(with_text("Find the price.", "If a country is rejected, say so."))
    assert not write_intent(with_text("Show me the review queue and the latest record."))
    assert not write_intent(with_text("Tell me the address of the store."))


def test_write_intent_on_the_pantry_suite() -> None:
    assert write_intent(pantry_scenario("label-submission")) is True
    for name in (
        "cheapest-penne",
        "misspelled-country",
        "tomato-penne-boycott",
        "unknown-recipe",
        "week-under-budget",
    ):
        assert write_intent(pantry_scenario(name)) is False, name


# --- initial tools ----------------------------------------------------------------------------


def test_initial_tools_for_cheapest_penne_put_find_product_first_and_no_write_tool(
    catalog: Catalog, scenario_data: dict[str, Any]
) -> None:
    scenario = cheapest_penne_like(scenario_data["server"])
    chosen = [t.name for t in initial_tools(scenario, catalog)]
    assert chosen[0] == "find_product"
    assert "submit_origin_evidence" not in chosen and "review_origin_submission" not in chosen
    assert len(chosen) == 5
    assert chosen == [
        "find_product",
        "get_product",
        "get_product_origins",
        "origin_triage",
        "list_products",
    ]
    scores = {t.name: relevance(scenario, t) for t in catalog.tools}
    assert scores["find_product"] == max(scores.values())
    # The same answer from the allowed catalog the real scenario will use.
    allowed = catalog.filtered(["*"], ["submit_*", "review_*"])
    assert [t.name for t in initial_tools(scenario, allowed)] == chosen


def test_initial_tools_forces_in_tools_whose_output_covers_an_expected_key(
    catalog: Catalog, scenario_data: dict[str, Any]
) -> None:
    scenario = parse_scenario(
        {
            **scenario_data,
            "goal": "What is the active routing strategy?",
            "instructions": [],
            "expected_outcome": {"json": {"routing_strategy": {"$type": "string"}}},
        },
        source="t",
    )
    chosen = [t.name for t in initial_tools(scenario, catalog, k=1)]
    assert covers_expected_key(catalog.tool("pipeline_status"), ["routing_strategy"])
    assert "pipeline_status" in chosen
    assert chosen[0] == "pipeline_status", "the only tool with any relevance ranks first"


def test_initial_tools_includes_write_tools_only_with_write_intent(catalog: Catalog) -> None:
    label = pantry_scenario("label-submission")
    allowed = catalog.filtered(
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
    chosen = [t.name for t in initial_tools(label, allowed)]
    assert chosen[0] == "submit_origin_evidence"
    assert "find_product" in chosen and "list_origin_submissions" in chosen
    assert "review_origin_submission" not in chosen, "not in the allowed catalog"


def test_initial_tools_never_fewer_than_three_when_the_catalog_has_them(
    scenario_data: dict[str, Any],
) -> None:
    scenario = parse_scenario(
        {**scenario_data, "goal": "Zzz.", "instructions": [], "expected_outcome": {"text": "zzz"}},
        source="t",
    )
    small = Catalog(
        tools=[
            ToolInfo(name="alpha"),
            ToolInfo(name="submit_beta"),
            ToolInfo(name="gamma"),
            ToolInfo(name="delta"),
        ]
    )
    assert [t.name for t in initial_tools(scenario, small)] == ["alpha", "gamma", "delta"], (
        "no relevance anywhere: padded in catalog order, write tool left out"
    )
    assert [t.name for t in initial_tools(scenario, Catalog(tools=[ToolInfo(name="only")]))] == [
        "only"
    ]
    assert initial_tools(scenario, Catalog()) == []


def test_initial_tools_for_the_rest_of_the_pantry_suite(catalog: Catalog) -> None:
    allowed = catalog.filtered(["*"], ["submit_*", "review_*"])
    first = {
        name: initial_tools(pantry_scenario(name), allowed)[0].name
        for name in (
            "cheapest-penne",
            "misspelled-country",
            "tomato-penne-boycott",
            "unknown-recipe",
            "week-under-budget",
        )
    }
    assert first == {
        "cheapest-penne": "find_product",
        "misspelled-country": "plan_recipe",
        "tomato-penne-boycott": "plan_recipe",
        "unknown-recipe": "list_recipes",
        "week-under-budget": "plan_week",
    }
