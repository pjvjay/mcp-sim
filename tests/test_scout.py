"""Scout and orchestrating planner (DESIGN §2b "Scout → orchestrator"): read-only observations,
informant reports at plan time, the disclosed toolset in the prompt, and the rules the plan
must obey about on-request tools."""

from __future__ import annotations

from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer

from mcpsim.mcpclient import Catalog, Session
from mcpsim.observers import ObserverRunner
from mcpsim.plan import CHECKPOINT_PATTERN
from mcpsim.planner import (
    DEFAULT_PROMPT_BUDGET,
    PLAN_TOOL_NAME,
    PlanDraft,
    build_prompts,
    build_system_prompt,
    build_user_prompt,
    dry_run_plan,
    plan,
    plan_tool_definition,
    planner_prompt_budget,
    planner_view,
    tool_name_enum,
    validate_draft,
)
from mcpsim.scenario import Scenario, parse_scenario
from mcpsim.scoping import DISCOVER_TOOL_NAME
from mcpsim.scout import (
    Observation,
    ScoutResult,
    initial_disclosed,
    lookup_arguments,
    scout,
    scout_budget,
    should_scout,
    summarise_json,
)
from mcpsim.transcript import GoalEnabledEvent, InformantReportEvent, ToolsOfferedEvent
from tests.conftest import open_session
from tests.fake_llm import ScriptedLLM, structured_response
from tests.fake_server import build_server

PENNE = {"slug": "penne", "price": 2.49, "store": "Fake Mart", "origin_status": "verified"}

AUDITOR: dict[str, Any] = {
    "name": "shelf_auditor",
    "identity": "An independent auditor who trusts only the store's own records.",
    "kind": "code",
    "watches": ["scout", "tool_traffic"],
    "on": ["scout", "tool_result"],
    "conditions": [
        {
            "id": "direct_match",
            "when": "lookup has returned the penne record",
            "check": {"tool_result": {"tool": "lookup", "where": {"slug": "penne"}}},
            "then": {"enable_tools": ["echo"], "enable_goal": "Quote the store exactly."},
        }
    ],
}


def progressive(
    scenario_data: dict[str, Any], *observers: dict[str, Any], **tools: Any
) -> Scenario:
    data: dict[str, Any] = {
        **scenario_data,
        "tools": {"disclosure": "progressive", "initial": ["lookup", "fail"], **tools},
    }
    if observers:
        data["observers"] = list(observers)
    return parse_scenario(data, source="fixture")


async def run_scout(
    scenario: Scenario,
    *,
    server: MCPServer | None = None,
    observers: bool = True,
    budget: int | None = None,
) -> tuple[ScoutResult, Catalog]:
    async with open_session(server) as session:
        catalog = (await session.catalog()).filtered(scenario.tools.allow, scenario.tools.deny)
        runner = ObserverRunner(scenario, None, include_llm=False) if observers else None
        result = await scout(scenario, catalog, session, observers=runner, budget=budget)
    return result, catalog


# --- the scout --------------------------------------------------------------------------------


def test_scout_budget_and_when_it_runs(scenario_data: dict[str, Any]) -> None:
    base = parse_scenario(scenario_data)
    assert scout_budget(base) == 10 and not should_scout(base), "disclosure all: nothing to scout"
    small = parse_scenario({**scenario_data, "budgets": {"max_tool_calls": 3}})
    assert scout_budget(small) == 2, "never fewer than two calls"
    assert should_scout(progressive(scenario_data))
    assert should_scout(parse_scenario({**scenario_data, "tools": {"disclosure": "plan"}}))


async def test_scout_reads_resources_then_expected_lookups_then_zero_argument_reads(
    scenario_data: dict[str, Any],
) -> None:
    scenario = progressive(scenario_data)
    result, catalog = await run_scout(scenario, observers=False)
    assert initial_disclosed(scenario, catalog) == ["lookup", "fail"]
    assert [(o.kind, o.name, o.arguments) for o in result.observations] == [
        ("resource", "fake://about", {}),
        ("tool", "lookup", {"slug": "penne"}),  # slug: penne from the expected outcome
        ("tool", "fail", {}),  # a zero-argument read among the disclosed set
    ]
    about, lookup, fail = result.observations
    assert about.summary == "text/plain, 39 chars: A fake pantry server for mcp-sim tests."
    assert about.chars == 39 and about.mime_type == "text/plain"
    assert lookup.from_expected == ["slug"] and lookup.is_error is False
    assert lookup.structured == PENNE and lookup.structured_keys == list(PENNE)
    assert lookup.summary == (
        'object with keys slug, price, store, origin_status; slug="penne"; price=2.49; '
        'store="Fake Mart"; origin_status="verified"'
    )
    assert lookup.call_label() == 'lookup(slug="penne")'
    assert fail.is_error is True and "fail tool invoked" in fail.summary
    assert result.tool_calls == 2 and result.budget == 10
    assert result.disclosed == ["lookup", "fail"]
    assert result.on_request == ["list_items", "echo", "expensive_report"]
    assert result.reports == [] and result.events == [] and result.goals == []
    assert result.lookups() == [lookup]


async def test_scout_respects_the_budget_and_never_calls_write_or_expensive_tools(
    scenario_data: dict[str, Any],
) -> None:
    server = build_server()

    @server.tool()
    def submit_note(text: str = "") -> dict[str, Any]:
        """Record a note (a write, by name)."""
        return {"ok": True}

    everything = progressive(scenario_data, initial=["*"])
    result, _ = await run_scout(everything, server=server, observers=False, budget=2)
    assert [o.name for o in result.tool_observations()] == ["lookup", "fail"], (
        "budget 2: the expected lookup, then one zero-argument read in relevance/catalog order"
    )
    result, _ = await run_scout(everything, server=server, observers=False)
    names = [o.name for o in result.tool_observations()]
    assert "submit_note" not in names and "expensive_report" not in names
    assert "echo" not in names, "echo needs a text argument the outcome does not pin"
    assert names == ["lookup", "fail", "list_items"]
    assert result.tool_calls == 3

    tight = parse_scenario(
        {**scenario_data, "budgets": {"max_tool_calls": 4}, "tools": {"disclosure": "progressive"}}
    )
    assert scout_budget(tight) == 2
    result, _ = await run_scout(tight, observers=False)
    assert result.tool_calls == 2 and result.budget == 2


async def test_scout_observers_report_from_observations_and_their_effects_disclose_tools(
    scenario_data: dict[str, Any],
) -> None:
    scenario = progressive(scenario_data, AUDITOR)
    result, _ = await run_scout(scenario)
    [report] = result.reports
    assert (report.observer, report.condition, report.value, report.trigger) == (
        "shelf_auditor",
        "direct_match",
        True,
        "scout",
    )
    assert report.evidence == "lookup.slug == 'penne'" and report.at_event == -1
    assert result.disclosed == ["lookup", "fail", "echo"]
    assert result.on_request == ["list_items", "expensive_report"], "shrank by exactly echo"
    assert result.goals == ["Quote the store exactly."]
    assert [type(e).__name__ for e in result.events] == [
        "InformantReportEvent",
        "ToolsOfferedEvent",
        "GoalEnabledEvent",
    ]
    offered = result.events[1]
    assert isinstance(offered, ToolsOfferedEvent)
    assert (offered.added, offered.reason) == (["echo"], "observer:shelf_auditor.direct_match")
    goal = result.events[2]
    assert isinstance(goal, GoalEnabledEvent) and goal.observer == "shelf_auditor"
    assert isinstance(result.events[0], InformantReportEvent)
    # echo needs an argument the outcome does not pin, so the one extra pass made no call.
    assert [o.name for o in result.tool_observations()] == ["lookup", "fail"]


async def test_scout_false_report_applies_otherwise_and_a_missing_source_is_unknown(
    scenario_data: dict[str, Any],
) -> None:
    librarian = {
        "name": "librarian",
        "identity": "Knows the catalogue.",
        "kind": "code",
        "on": ["scout"],
        "conditions": [
            {
                "id": "exists",
                "when": "the slug the user named exists",
                "check": {"tool_result": {"tool": "lookup", "where": {"slug": "penne"}}},
                "otherwise": {"enable_goal": "Say the product does not exist.", "flag": "missing"},
            },
            {
                "id": "listed",
                "when": "list_items was consulted",
                "check": {"tool_called": "list_items"},
            },
        ],
    }
    data = {**scenario_data, "expected_outcome": {"json": {"slug": "nope", "price": {"$gt": 0}}}}
    scenario = progressive(data, librarian)
    result, _ = await run_scout(scenario)
    lookup = result.tool_observations()[0]
    assert lookup.arguments == {"slug": "nope"} and lookup.is_error is True
    assert result.lookups() == [], "an error result is not a proven lookup"
    exists, listed = result.reports
    assert exists.value is False
    assert exists.evidence.startswith("lookup returned an error: Error executing tool lookup")
    assert listed.value is False and listed.evidence == "list_items not called"
    assert result.goals == ["Say the product does not exist."] and result.flags == ["missing"]


async def test_scout_round_trips_through_scout_json(
    scenario_data: dict[str, Any], tmp_path: Any
) -> None:
    result, _ = await run_scout(progressive(scenario_data, AUDITOR))
    path = result.save(tmp_path / "scout.json")
    back = ScoutResult.load(path)
    assert back == result
    assert back.events[1].kind == "tools_offered"


def test_summarise_json_shapes() -> None:
    assert summarise_json({"a": [1, 2, 3, 4], "b": {"x": 1}, "c": "s"}) == (
        'object with keys a, b, c; a: list of 4; a[0]=1; a[1]=2; a[2]=3; b={"x": 1}; c="s"'
    )
    assert summarise_json([{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]) == (
        'list of 4; [0]={"id": 1}; [1]={"id": 2}; [2]={"id": 3}'
    )
    assert summarise_json({}) == "empty object"
    many = {f"k{i}": i for i in range(14)}
    assert "k11, … (2 more)" in summarise_json(many)
    assert len(summarise_json({"text": "x" * 5000})) <= 600


async def test_lookup_arguments_need_a_string_pinned_by_the_outcome() -> None:
    async with open_session() as session:
        catalog = await session.catalog()
    lookup = catalog.tool("lookup")
    assert lookup_arguments(lookup, {"slug": "penne"}) == ({"slug": "penne"}, ["slug"])
    assert lookup_arguments(lookup, {"slug": 5}) is None, "the schema says string"
    assert lookup_arguments(lookup, {"price": {"$gt": 0}}) is None
    assert lookup_arguments(catalog.tool("list_items"), {"cursor": 3}) is None, "not a string"


# --- the orchestrating planner ----------------------------------------------------------------


async def scouted(
    scenario_data: dict[str, Any], *observers: dict[str, Any]
) -> tuple[Scenario, Catalog, ScoutResult]:
    scenario = progressive(scenario_data, *(observers or (AUDITOR,)))
    result, catalog = await run_scout(scenario)
    return scenario, catalog, result


async def test_planner_prompt_shows_disclosed_lines_on_request_names_reports_and_observations(
    scenario_data: dict[str, Any],
) -> None:
    scenario, catalog, result = await scouted(scenario_data)
    view = planner_view(scenario, catalog, result)
    assert view.disclosed_names == ["lookup", "fail", "echo"]
    assert view.on_request == ["list_items", "expensive_report"] and view.discoverable
    system, user = build_prompts(scenario, catalog, view)

    assert "TOOLS (3 disclosed of 5 allowed)" in system
    assert "- lookup(slug: string) → returns object — Look up a product by slug." in system
    assert "- echo(text: string)" in system and "- fail(reason?: string)" in system
    assert f"- {DISCOVER_TOOL_NAME}(query: string) — Ask for more tools." in system
    assert "- list_items(" not in system and "- expensive_report(" not in system
    assert (
        f"AVAILABLE ON REQUEST through {DISCOVER_TOOL_NAME} (name only; a step may use one of "
        "these only after a discover_tools step whose query names what it needs, or after an "
        "observer effect enables it): list_items, expensive_report"
    ) in system
    assert "\n7. You are the orchestrator" in system and "\n8. A report that is FALSE" in system
    assert "\n9. A tool listed as available on request" in system

    assert "INFORMANT REPORTS (observers watched the scout's calls; plan from these):\n" in user
    assert "- shelf_auditor.direct_match = true — lookup.slug == 'penne'" in user
    assert (
        "GOALS ENABLED BY OBSERVATION (the agent will be told these too):\n"
        "- Quote the store exactly."
    ) in user
    assert (
        "OBSERVATIONS (read-only calls already made against the live server; use these values):"
        in user
    )
    assert (
        '- lookup(slug="penne") → ok: object with keys slug, price, store, origin_status; '
        'slug="penne"'
    ) in user
    assert "- fake://about → ok: text/plain, 39 chars:" in user
    assert "- fail() → ERROR: Error executing tool fail" in user
    assert user.endswith("Produce the execution plan for this scenario now.")
    assert result.planner_prompt_chars == 0, "set by plan(), not by build_prompts"
    # Without a scout nothing changes: every allowed tool, no informant sections.
    assert "TOOLS (5) — the ONLY tools that exist:" in build_system_prompt(catalog)
    assert "INFORMANT REPORTS" not in build_user_prompt(scenario)


async def test_plan_tool_enum_is_disclosed_plus_discover_tools(
    scenario_data: dict[str, Any],
) -> None:
    scenario, catalog, result = await scouted(scenario_data)
    view = planner_view(scenario, catalog, result)
    assert tool_name_enum(catalog, view) == ["echo", "fail", "lookup", DISCOVER_TOOL_NAME]
    schema = plan_tool_definition(catalog, view)["input_schema"]
    assert schema["$defs"]["Step"]["properties"]["tool"]["anyOf"][0]["enum"] == [
        "echo",
        "fail",
        "lookup",
        DISCOVER_TOOL_NAME,
    ]
    no_discover = scenario.model_copy(
        update={"tools": scenario.tools.model_copy(update={"discover_tool": False})}
    )
    assert tool_name_enum(catalog, planner_view(no_discover, catalog, result)) == [
        "echo",
        "fail",
        "lookup",
    ]
    assert "NOT DISCLOSED (declared by the server" in build_system_prompt(
        catalog, planner_view(no_discover, catalog, result)
    )
    # plan disclosure: the planner may name every allowed tool (guided runs offer the path's).
    planned = parse_scenario({**scenario_data, "tools": {"disclosure": "plan"}})
    assert tool_name_enum(catalog, planner_view(planned, catalog, result)) == sorted(
        catalog.tool_names()
    )


def step(tool: str | None, **arguments: Any) -> dict[str, Any]:
    return {"intent": f"call {tool}", "tool": tool, "arguments_sketch": arguments}


def draft(*steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "paths": [
            {
                "id": "happy",
                "kind": "happy",
                "title": "t",
                "steps": list(steps),
                "checkpoints": ["final_result: slug equals penne"],
            }
        ]
    }


async def test_on_request_tool_needs_a_discover_step_or_an_enabling_observer(
    scenario_data: dict[str, Any],
) -> None:
    scenario, catalog, result = await scouted(scenario_data)
    view = planner_view(scenario, catalog, result)
    bare = PlanDraft.model_validate(
        draft(step("lookup", slug="penne"), step("list_items", cursor=0))
    )
    problems = validate_draft(bare, catalog, view, scenario)
    assert problems == [
        "paths[0].steps[1] ('happy'): tool 'list_items' is not disclosed at plan time; put a "
        f"{DISCOVER_TOOL_NAME} step (query naming what you need) before it, or use a disclosed tool"
    ]
    after_discover = PlanDraft.model_validate(
        draft(step(DISCOVER_TOOL_NAME, query="list items"), step("list_items", cursor=0))
    )
    assert validate_draft(after_discover, catalog, view, scenario) == []
    bad_discover = PlanDraft.model_validate(draft(step(DISCOVER_TOOL_NAME, q="x")))
    assert validate_draft(bad_discover, catalog, view, scenario) == [
        "paths[0].steps[0] ('happy'): argument 'q' is not accepted by tool discover_tools; "
        "its arguments are: query"
    ]
    # The scout's auditor watches tool_traffic at tool_result and enables echo; but echo is
    # already disclosed here. An observer that enables list_items after a tool step makes a
    # direct list_items step legal when a tool step precedes it.
    enabler = {
        **AUDITOR,
        "name": "pager",
        "conditions": [
            {**AUDITOR["conditions"][0], "id": "paged", "then": {"enable_tools": ["list_*"]}}
        ],
    }
    enabled = scenario.with_observers([enabler])
    view2 = planner_view(enabled, catalog, result)
    assert view2.enabling_observer(enabled, "list_items") == "pager.paged"
    assert view2.enabling_observer(enabled, "expensive_report") is None
    assert validate_draft(bare, catalog, view2, enabled) == []
    first = PlanDraft.model_validate(draft(step("list_items", cursor=0)))
    assert len(validate_draft(first, catalog, view2, enabled)) == 1, "nothing precedes it"
    # An observer that reports only at end, or watches only the conversation, cannot enable
    # a tool mid-run.
    late = {**enabler, "on": ["end"]}
    assert (
        planner_view(scenario.with_observers([late]), catalog, result).enabling_observer(
            scenario.with_observers([late]), "list_items"
        )
        is None
    )
    blind = {**enabler, "watches": ["conversation"]}
    assert (
        planner_view(scenario.with_observers([blind]), catalog, result).enabling_observer(
            scenario.with_observers([blind]), "list_items"
        )
        is None
    )
    # Without discover_tools the message says so.
    no_discover = scenario.model_copy(
        update={"tools": scenario.tools.model_copy(update={"discover_tool": False})}
    )
    [problem] = validate_draft(
        bare, catalog, planner_view(no_discover, catalog, result), no_discover
    )
    assert "only an observer effect can enable it after a tool step" in problem
    # A tool outside the allowed catalog is still simply unknown.
    unknown = PlanDraft.model_validate(draft(step("price_lookup")))
    assert validate_draft(unknown, catalog, view, scenario)[0].startswith(
        "paths[0].steps[0] ('happy'): unknown tool 'price_lookup'"
    )


async def test_plan_reasks_once_naming_the_on_request_tool(scenario_data: dict[str, Any]) -> None:
    scenario, catalog, result = await scouted(scenario_data)
    bad = draft(step("lookup", slug="penne"), step("list_items", cursor=0))
    good = draft(
        step("lookup", slug="penne"),
        step(DISCOVER_TOOL_NAME, query="list items"),
        step("list_items", cursor=0),
    )
    llm = ScriptedLLM(
        [structured_response(PLAN_TOOL_NAME, bad), structured_response(PLAN_TOOL_NAME, good)]
    )
    execution_plan = await plan(scenario, catalog, llm, scout=result)
    assert len(llm.calls) == 2
    reask = llm.calls[1]["messages"][-1]["content"][0]["content"]
    assert "tool 'list_items' is not disclosed at plan time" in reask
    assert execution_plan.paths[0].tools_used() == ["lookup", DISCOVER_TOOL_NAME, "list_items"]
    assert result.planner_prompt_chars == len(llm.calls[0]["system"]) + len(
        llm.calls[0]["messages"][0]["content"]
    )
    assert result.planner_prompt_chars > 0
    enum = llm.calls[0]["tools"][0]["input_schema"]["$defs"]["Step"]["properties"]["tool"]["anyOf"][
        0
    ]["enum"]
    assert enum == ["echo", "fail", "lookup", DISCOVER_TOOL_NAME]


async def test_prompt_budget_trims_observations_first_then_the_on_request_list(
    scenario_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario, catalog, result = await scouted(scenario_data)
    # A summary is at most 600 characters (the scout clips it); 25 of them is 15k, over budget.
    noise = [
        Observation(kind="tool", name="fail", arguments={"reason": str(i)}, summary="z" * 600)
        for i in range(25)
    ]
    result.observations = [*noise, *result.observations]
    view = planner_view(scenario, catalog, result)
    system, user = build_prompts(scenario, catalog, view, budget=100_000)
    assert user.count("z" * 600) == 25, "under the budget nothing is trimmed"
    assert len(system) + len(user) > DEFAULT_PROMPT_BUDGET

    system, user = build_prompts(scenario, catalog, view)
    assert len(system) + len(user) <= DEFAULT_PROMPT_BUDGET
    left_out = 28 - user.count("→ ok") - user.count("→ ERROR")
    assert left_out >= 1 and f"({left_out} earlier observation(s) left out" in user
    assert 'fail(reason="0")' not in user, "the oldest observation went first"
    assert '- lookup(slug="penne") → ok:' in user, "the later ones stay while they fit"
    assert "- shelf_auditor.direct_match = true — lookup.slug == 'penne'" in user
    assert "- lookup(slug: string) → returns object" in system
    assert "list_items, expensive_report" in system

    # Far too small for the observations and the names: the on-request list becomes a count,
    # but the digest, the reports and the scenario never go.
    system, user = build_prompts(scenario, catalog, view, budget=10)
    assert "OBSERVATIONS" in user and "(28 earlier observation(s) left out" in user
    assert "- shelf_auditor.direct_match = true" in user and scenario.goal in user
    assert "- lookup(slug: string) → returns object" in system
    assert "list_items, expensive_report" not in system and "2 tools" in system

    monkeypatch.setenv("MCPSIM_PLANNER_PROMPT_BUDGET", "5000")
    assert planner_prompt_budget() == 5000
    system, user = build_prompts(scenario, catalog, view)
    assert len(system) + len(user) <= 5000 or "2 tools" in system
    monkeypatch.setenv("MCPSIM_PLANNER_PROMPT_BUDGET", "big")
    with pytest.raises(ValueError, match="MCPSIM_PLANNER_PROMPT_BUDGET must be an integer"):
        planner_prompt_budget()


@pytest.mark.parametrize(
    "text",
    [
        "report: shelf_auditor.direct_match is true",
        "report: librarian.exists is false",
        "report:policy_desk.ready_to_quote is true",
    ],
)
def test_report_checkpoints_are_accepted(text: str) -> None:
    assert CHECKPOINT_PATTERN.match(text)


@pytest.mark.parametrize(
    "text",
    [
        "report: direct_match is true",
        "report: shelf_auditor.direct_match is maybe",
        "report: shelf_auditor.direct_match",
        "reports: a.b is true",
    ],
)
def test_report_checkpoints_are_rejected(text: str) -> None:
    assert not CHECKPOINT_PATTERN.match(text)


async def test_dry_run_plan_puts_the_scouts_proven_lookups_first_and_records_reports(
    scenario_data: dict[str, Any],
) -> None:
    # The outcome pins only echo's text; lookup (named in the instructions, with price and
    # store among its outputs) still outranks echo by relevance.
    data = {**scenario_data, "expected_outcome": {"json": {"text": "hi"}}}
    scenario = progressive(data, AUDITOR, initial=["echo", "lookup"])
    result, catalog = await run_scout(scenario)
    assert [o.call_label() for o in result.lookups()] == ['echo(text="hi")']
    without = dry_run_plan(scenario, catalog)
    assert without.paths[0].tools_used() == ["lookup", "echo"], "relevance alone: lookup leads"
    with_scout = dry_run_plan(scenario, catalog, result)
    only = with_scout.paths[0]
    assert only.tools_used() == ["echo", "lookup"], "the proven lookup comes first"
    assert only.steps[0].arguments_sketch == {"text": "hi"}
    assert (
        "The scout already proved these expected-outcome lookups answer, so they come first: "
        "echo."
    ) in only.rationale
    # lookup was never called at scout time, so the auditor's report is unknown and no
    # report checkpoint is written for it.
    assert [(r.key, r.value) for r in result.reports] == [("shelf_auditor.direct_match", None)]
    assert not any(c.startswith("report:") for c in only.checkpoints)
    assert result.planner_prompt_chars > 0, "the prompt the LLM planner would get is measured"
    assert validate_draft(PlanDraft(paths=list(with_scout.paths)), catalog) == []
    assert with_scout.catalog_digest == catalog.digest()


async def test_plan_without_a_scout_is_unchanged(scenario_data: dict[str, Any]) -> None:
    scenario = parse_scenario(scenario_data)
    async with open_session() as session:
        catalog = await session.catalog()
    llm = ScriptedLLM([structured_response(PLAN_TOOL_NAME, draft(step("lookup", slug="penne")))])
    await plan(scenario, catalog, llm)
    assert "INFORMANT REPORTS" not in llm.calls[0]["messages"][0]["content"]
    assert "TOOLS (5) — the ONLY tools that exist:" in llm.calls[0]["system"]


async def test_read_resource_normalises_text(scenario_data: dict[str, Any]) -> None:
    async with open_session() as session:
        assert isinstance(session, Session)
        content = await session.read_resource("fake://about")
    assert content.uri == "fake://about" and content.mime_type == "text/plain"
    assert content.text == "A fake pantry server for mcp-sim tests." and content.chars == 39
    assert content.ms >= 0
