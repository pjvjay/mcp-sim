"""The local planner profile (docs/LOCAL_MODELS.md "Speed"): compact prompt, one path per call.

A CPU-bound 7-8B model reads about 20 prompt tokens/s and writes about 3-4, so the profile keeps
the prompt near 4,000 characters, asks for one path per call, caps the output with the grammar
and continues one conversation so the server only evaluates each new turn.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from mcpsim.mcpclient import Catalog
from mcpsim.plan import CHECKPOINT_PATTERN, PATH_KINDS
from mcpsim.planner import (
    LOCAL_DESCRIPTION_LIMIT,
    LOCAL_MAX_CHECKPOINTS,
    LOCAL_MAX_STEPS,
    LOCAL_PATHS_ENV,
    LOCAL_PLAN_MAX_TOKENS,
    LOCAL_PROMPT_BUDGET,
    LOCAL_STRING_CAPS,
    PLAN_TOOL_NAME,
    PlanError,
    build_prompts,
    checkpoint_grammar_pattern,
    local_plan_input_schema,
    local_plan_paths,
    plan,
    planner_profile,
)
from mcpsim.scenario import Scenario, parse_scenario
from tests.conftest import open_session
from tests.fake_llm import ScriptedLLM, structured_response
from tests.test_scoping import pantry_catalog, pantry_scenario

LOCAL = "ollama:command-r7b"
NON_HAPPY = [k for k in PATH_KINDS if k != "happy"]


def local(scenario: Scenario) -> Scenario:
    return scenario.model_copy(
        update={"models": scenario.models.model_copy(update={"planner": LOCAL})}
    )


@pytest.fixture
def scenario(scenario_data: dict[str, Any]) -> Scenario:
    return local(parse_scenario(scenario_data, source="test"))


async def fake_catalog() -> Catalog:
    async with open_session() as session:
        return await session.catalog()


def one_path(path_id: str, kind: str, *steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "paths": [
            {
                "id": path_id,
                "kind": kind,
                "title": f"{kind} path",
                "rationale": "because",
                "steps": list(steps),
                "checkpoints": ["final_result: slug equals the slug lookup returned"],
            }
        ]
    }


def lookup(slug: str, *, expect_error: bool = False) -> dict[str, Any]:
    return {
        "intent": f"look up {slug}",
        "tool": "lookup",
        "arguments_sketch": {"slug": slug},
        "success_looks_like": "the server rejects it" if expect_error else "a price",
        "expect_error": expect_error,
    }


def kinds_offered(call: dict[str, Any]) -> list[str]:
    schema = call["tools"][0]["input_schema"]
    return list(schema["$defs"]["Path"]["properties"]["kind"]["enum"])


# --- profile selection --------------------------------------------------------------------------


def test_an_ollama_planner_gets_the_local_profile(scenario: Scenario) -> None:
    assert planner_profile(scenario) == "local"
    hosted = scenario.model_copy(
        update={"models": scenario.models.model_copy(update={"planner": "claude-opus-5-5"})}
    )
    assert planner_profile(hosted) == "hosted"


def test_local_plan_paths_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOCAL_PATHS_ENV, raising=False)
    assert local_plan_paths() == 3
    monkeypatch.setenv(LOCAL_PATHS_ENV, "1")
    assert local_plan_paths() == 1
    for bad in ("0", "6", "two"):
        monkeypatch.setenv(LOCAL_PATHS_ENV, bad)
        with pytest.raises(ValueError, match=LOCAL_PATHS_ENV):
            local_plan_paths()


# --- the per-call schema ------------------------------------------------------------------------


async def test_local_schema_asks_for_one_path_of_the_offered_kinds_with_caps() -> None:
    catalog = await fake_catalog()
    schema = local_plan_input_schema(catalog, None, ["recovery", "policy"])
    assert schema["properties"]["paths"]["minItems"] == 1
    assert schema["properties"]["paths"]["maxItems"] == 1
    path_props = schema["$defs"]["Path"]["properties"]
    assert path_props["kind"] == {"type": "string", "enum": ["recovery", "policy"]}
    assert path_props["steps"]["maxItems"] == LOCAL_MAX_STEPS
    assert path_props["checkpoints"]["maxItems"] == LOCAL_MAX_CHECKPOINTS
    assert path_props["checkpoints"]["items"]["pattern"] == checkpoint_grammar_pattern(
        sorted(catalog.tool_names())
    )
    for model, caps in LOCAL_STRING_CAPS.items():
        for name, cap in caps.items():
            assert schema["$defs"][model]["properties"][name]["maxLength"] == cap
    # Still strict: every field required, Path.title included.
    assert "title" in schema["$defs"]["Path"]["required"]
    assert schema["$defs"]["Step"]["properties"]["tool"]["anyOf"][0]["enum"] == sorted(
        catalog.tool_names()
    )


# --- the compact prompt -------------------------------------------------------------------------


def disclosed_view(catalog: Catalog, names: set[str]) -> Any:
    from mcpsim.planner import PlannerView

    return PlannerView(
        disclosed=[t for t in catalog.tools if t.name in names],
        on_request=[n for n in catalog.tool_names() if n not in names],
        discoverable=True,
        allowed=catalog.tool_names(),
    )


# What the cheapest-penne scout discloses on the live pantry server.
PENNE_DISCLOSED = {"find_product", "get_product", "list_products", "get_product_origins",
                   "origin_triage"}


def test_compact_prompt_drops_what_a_step_cannot_name_and_is_much_shorter() -> None:
    catalog = pantry_catalog()
    scenario = local(pantry_scenario("cheapest-penne"))
    view = disclosed_view(catalog, PENNE_DISCLOSED)
    hosted_system, hosted_user = build_prompts(scenario, catalog, view, budget=100_000)
    system, user = build_prompts(scenario, catalog, view, budget=100_000, compact=True)
    assert "RESOURCES" not in system and "PROMPTS" not in system
    assert "RESOURCES" in hosted_system
    for line in system.splitlines():
        if line.startswith("- ") and " — " in line:
            assert len(line.split(" — ", 1)[1]) <= LOCAL_DESCRIPTION_LIMIT, line
    # Every disclosed tool still gets its digest line, and the example names one of them
    # (the hosted example may pick any catalog tool, here a write tool the planner cannot use).
    for name in PENNE_DISCLOSED:
        assert f"- {name}(" in system
    example = system.split("Example step: ", 1)[1].split("\n", 1)[0]
    assert json.loads(example)["tool"] in PENNE_DISCLOSED
    assert "returns avg_rating, brand, category, description, dietary_tags, evidence, … (11 more)" \
        in system
    # The fixture catalog has no resources or prompts; on the live server (four resources, three
    # prompts, eight observations) the cut was 11,702 characters to 4,752.
    assert len(system) + len(user) < (len(hosted_system) + len(hosted_user)) * 0.6
    assert user.endswith("Emit the first path now: the happy path.")


def test_compact_prompt_defaults_to_the_local_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Observations are trimmed against LOCAL_PROMPT_BUDGET, not the hosted 12,000."""
    from mcpsim.planner import PlannerView, planner_prompt_budget

    monkeypatch.delenv("MCPSIM_PLANNER_PROMPT_BUDGET", raising=False)
    assert planner_prompt_budget(LOCAL_PROMPT_BUDGET) == LOCAL_PROMPT_BUDGET
    catalog = pantry_catalog()
    scenario = local(pantry_scenario("cheapest-penne"))
    disclosed = [t for t in catalog.tools if t.name in {"find_product", "get_product"}]
    view = PlannerView(disclosed=disclosed, on_request=[], allowed=catalog.tool_names())
    system, user = build_prompts(scenario, catalog, view, compact=True)
    assert len(system) + len(user) <= LOCAL_PROMPT_BUDGET


# --- one path per call --------------------------------------------------------------------------


async def test_local_plan_asks_one_path_per_call_and_continues_the_conversation(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, one_path("1", "happy", lookup("penne"))),
            # Same id as the happy path: each call sees only its own path, so it is renamed.
            structured_response(
                PLAN_TOOL_NAME,
                one_path("1", "recovery", lookup("pene", expect_error=True), lookup("penne")),
            ),
            structured_response(PLAN_TOOL_NAME, one_path("policy", "policy", lookup("penne"))),
        ]
    )
    result = await plan(scenario, catalog, llm)

    assert [p.kind for p in result.paths] == ["happy", "recovery", "policy"]
    assert [p.id for p in result.paths] == ["1", "recovery", "policy"]
    assert len(llm.calls) == 3
    assert [kinds_offered(c) for c in llm.calls] == [
        ["happy"],
        NON_HAPPY,
        [k for k in NON_HAPPY if k != "recovery"],
    ]
    assert all(c["max_tokens"] == LOCAL_PLAN_MAX_TOKENS for c in llm.calls)
    # The same system prompt every call, and each call's messages extend the previous call's.
    assert len({c["system"] for c in llm.calls}) == 1
    first, second, third = (c["messages"] for c in llm.calls)
    assert second[: len(first)] == first
    assert third[: len(second)] == second
    assert second[1] == {"role": "assistant", "content": llm_content(PLAN_TOOL_NAME, "happy")}
    follow_up = second[2]["content"]
    assert follow_up[0] == {
        "type": "tool_result", "tool_use_id": "toolu_structured", "content": "accepted",
    }
    assert "kind not used yet: " + ", ".join(NON_HAPPY) in follow_up[1]["text"]
    # Notes record every call for the reviewer, accepted or not.
    assert len(result.notes) == 3
    assert all(n.startswith(f"local planner ({LOCAL}): call ") for n in result.notes)
    assert all(n.endswith("accepted") for n in result.notes)


def llm_content(name: str, kind: str) -> list[dict[str, Any]]:
    payload = one_path("1", kind, lookup("penne"))
    return [{"type": "tool_use", "id": "toolu_structured", "name": name, "input": payload}]


async def test_a_path_that_fails_twice_is_dropped_and_its_kind_not_asked_again(
    scenario: Scenario,
) -> None:
    catalog = await fake_catalog()
    no_failure = one_path("rec", "recovery", lookup("penne"))  # recovery without expect_error
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, one_path("happy", "happy", lookup("penne"))),
            structured_response(PLAN_TOOL_NAME, no_failure),
            structured_response(PLAN_TOOL_NAME, no_failure),
            structured_response(PLAN_TOOL_NAME, one_path("edge", "boundary", lookup("zzz"))),
            structured_response(PLAN_TOOL_NAME, one_path("rules", "policy", lookup("penne"))),
        ]
    )
    result = await plan(scenario, catalog, llm)

    assert [p.kind for p in result.paths] == ["happy", "boundary", "policy"]
    assert len(llm.calls) == 5
    assert kinds_offered(llm.calls[4]) == ["alternative", "policy"]
    # The retry after the drop starts again from the accepted happy path, without the
    # rejected turns, and no longer offers recovery.
    assert kinds_offered(llm.calls[3]) == [k for k in NON_HAPPY if k != "recovery"]
    assert llm.calls[3]["messages"][:2] == llm.calls[1]["messages"][:2]
    assert len(llm.calls[3]["messages"]) == 3
    assert any("dropped a recovery path after 2 invalid draft(s)" in n for n in result.notes)
    assert any("expect_error" in n for n in result.notes)


async def test_no_valid_happy_path_is_a_plan_error(scenario: Scenario) -> None:
    catalog = await fake_catalog()
    unknown_tool = one_path("happy", "happy", {**lookup("penne"), "tool": "price_lookup"})
    llm = ScriptedLLM([structured_response(PLAN_TOOL_NAME, unknown_tool)] * 2)
    with pytest.raises(PlanError, match="no valid happy path"):
        await plan(scenario, catalog, llm)
    assert len(llm.calls) == 2


async def test_a_draft_with_no_usable_path_stops_the_plan(scenario: Scenario) -> None:
    """Nothing to exclude means the next request would be identical: stop, keep what passed."""
    catalog = await fake_catalog()
    llm = ScriptedLLM(
        [
            structured_response(PLAN_TOOL_NAME, one_path("happy", "happy", lookup("penne"))),
            structured_response(PLAN_TOOL_NAME, {"paths": []}),
            structured_response(PLAN_TOOL_NAME, {"paths": []}),
        ]
    )
    result = await plan(scenario, catalog, llm)
    assert [p.kind for p in result.paths] == ["happy"]
    assert len(llm.calls) == 3
    assert any("dropped" in n for n in result.notes)


async def test_one_requested_path_means_one_call(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LOCAL_PATHS_ENV, "1")
    catalog = await fake_catalog()
    llm = ScriptedLLM(
        [structured_response(PLAN_TOOL_NAME, one_path("happy", "happy", lookup("penne")))]
    )
    result = await plan(scenario, catalog, llm)
    assert [p.kind for p in result.paths] == ["happy"]
    assert len(llm.calls) == 1


async def test_the_hosted_profile_is_unchanged(scenario: Scenario) -> None:
    """A hosted planner still gets one call for every path, with the full prompt."""
    catalog = await fake_catalog()
    hosted = scenario.model_copy(
        update={"models": scenario.models.model_copy(update={"planner": "claude-opus-5-5"})}
    )
    payload = {
        "paths": [
            one_path("happy", "happy", lookup("penne"))["paths"][0],
            one_path("rec", "recovery", lookup("pene", expect_error=True),
                     lookup("penne"))["paths"][0],
        ]
    }
    llm = ScriptedLLM([structured_response(PLAN_TOOL_NAME, payload)])
    result = await plan(hosted, catalog, llm)
    assert len(result.paths) == 2 and len(llm.calls) == 1
    assert result.notes == []
    assert "RESOURCES" in llm.calls[0]["system"]
    schema = llm.calls[0]["tools"][0]["input_schema"]
    assert "maxItems" not in schema["properties"]["paths"]
    assert json.dumps(schema).count("maxLength") == 0


async def test_a_plan_never_makes_more_than_two_calls_per_requested_path(
    scenario: Scenario,
) -> None:
    """Happy, then three kinds that each fail twice: the cap (3 paths x 2) stops it at 6."""
    catalog = await fake_catalog()
    bad = {"paths": [{**one_path("x", "recovery", lookup("penne"))["paths"][0]}]}
    responses = [structured_response(PLAN_TOOL_NAME, one_path("happy", "happy", lookup("penne")))]
    for kind in ("recovery", "alternative", "boundary"):
        payload = json.loads(json.dumps(bad))
        payload["paths"][0]["kind"] = kind
        payload["paths"][0]["steps"][0]["tool"] = "price_lookup"  # unknown tool: invalid
        responses += [structured_response(PLAN_TOOL_NAME, payload)] * 2
    llm = ScriptedLLM(responses)
    result = await plan(scenario, catalog, llm)
    assert [p.kind for p in result.paths] == ["happy"]
    assert len(llm.calls) == 6
    assert any("stopped after 6 calls" in n for n in result.notes)


# --- the checkpoint grammar ---------------------------------------------------------------------


GRAMMAR_ACCEPTS = [
    "tool_result[find_product]: match equals direct",
    "tool_result[pantry-find-product]: total is 2",
    "final_result: price equals 1.97, the cheapest offer",
    "transcript: no call to plan_recipe",
    "report: shelf_clerk.direct_match is true",
]
GRAMMAR_REJECTS = [
    "find_product: match == 'direct'",  # what command-r7b wrote when told in prose
    "tool_result[get_product]: a tool this catalog does not have",
    "final_result:  starts with a space",
    'final_result: holds a "quote" the decoder would write raw',
    "final_result: x\nsecond line",
    "report: shelf_clerk.direct_match is maybe",
]


def test_checkpoint_grammar_admits_only_what_the_validator_accepts() -> None:
    pattern = checkpoint_grammar_pattern(["find_product", "pantry-find-product"])
    for text in GRAMMAR_ACCEPTS:
        assert re.fullmatch(pattern, text), text
        assert CHECKPOINT_PATTERN.match(text), text
    for text in GRAMMAR_REJECTS:
        assert not re.fullmatch(pattern, text), text


def test_checkpoint_grammar_has_one_pair_of_anchors() -> None:
    """llama.cpp converts only ``^…$`` around the whole pattern; an anchor inside an
    alternation is logged as unsupported and the string is left unconstrained."""
    pattern = checkpoint_grammar_pattern(["find_product", "a.b", "x-y"])
    assert pattern.startswith("^(") and pattern.endswith(")$")
    inner = pattern[1:-1]
    assert "^" not in inner.replace("[^", "") and "$" not in inner
    assert "a\\.b" in pattern and "x-y" in pattern  # regex specials escaped, hyphen literal
    assert re.fullmatch(pattern, "tool_result[a.b]: ok")
    assert not re.fullmatch(pattern, "tool_result[aXb]: ok")


@pytest.mark.parametrize(
    ("steps", "problem"),
    [
        ([lookup("penne")], "at least one step with expect_error true"),
        ([lookup("pene", expect_error=True), {**lookup("penne"), "tool": None}],
         "no later step makes the corrected call"),
        ([lookup("penne", expect_error=True), lookup("penne")], "same arguments"),
    ],
)
async def test_a_recovery_path_must_show_the_failure_and_the_fix(
    steps: list[dict[str, Any]], problem: str
) -> None:
    """What command-r7b wrote: a "recovery" whose failing step sends good input, with no fix."""
    from mcpsim.planner import PlanDraft, validate_draft

    catalog = await fake_catalog()
    draft = PlanDraft.model_validate(one_path("rec", "recovery", *steps))
    problems = validate_draft(draft, catalog, require_happy=False)
    assert len(problems) == 1 and problem in problems[0], problems
    good = PlanDraft.model_validate(
        one_path("rec", "recovery", lookup("pene", expect_error=True), lookup("penne"))
    )
    assert validate_draft(good, catalog, require_happy=False) == []
