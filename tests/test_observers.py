"""Observer declarations (DESIGN §2b): YAML and the Python DSL build identical models, and
validation errors name the field."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path as FsPath
from typing import Any

import pytest
import yaml

from mcpsim.observers import Check, Effect, Observer, observer
from mcpsim.scenario import (
    MODEL_ROLES,
    Models,
    Scenario,
    ScenarioError,
    load_scenario,
    parse_scenario,
)
from mcpsim.transcript import (
    GoalEnabledEvent,
    InformantReport,
    InformantReportEvent,
    SystemEvent,
    Transcript,
    parse_event,
)

# The §1 declaration of the spec, verbatim in shape.
OBSERVERS_YAML = """
observers:
  - name: shelf_auditor
    identity: >
      An independent auditor who trusts only what the store's own records say and treats every
      unsupported claim as false until proven.
    watches: [tool_traffic, final_answer]
    on: [scout, turn, end]
    model: claude-sonnet-5-5
    conditions:
      - id: direct_match
        when: "find_product has returned at least one DIRECT match for penne"
        then:
          enable_tools: [get_product]
          enable_goal: "Quote the cheapest direct hit by exact name, price and store."
      - id: fabrication
        when: "the final answer names a product, price or store that appears in no tool result"
        then: { flag: fabrication, fail: true }
  - name: word_clerk
    identity: A clerk who counts words and nothing else.
    kind: code
    watches: [final_answer]
    on: [end]
    conditions:
      - id: too_long
        when: "the final answer is longer than 100 words"
        check: { word_count: { of: final_answer, gt: 100 } }
        then: { flag: verbose }
  - name: policy_desk
    identity: The policy desk; it only combines what others reported.
    kind: group
    conditions:
      - id: ready_to_quote
        when: "a direct match was reported and no fabrication"
        all_of: [shelf_auditor.direct_match, "!shelf_auditor.fabrication"]
        then: { enable_goal: "Deliver the final answer now." }
"""


def observers_data() -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = yaml.safe_load(OBSERVERS_YAML)["observers"]
    return loaded


@pytest.fixture
def scenario(scenario_data: dict[str, Any]) -> Scenario:
    return parse_scenario({**scenario_data, "observers": observers_data()}, source="fixture")


def dsl_observers() -> list[Any]:
    """The same three observers through the Python DSL."""
    auditor = observer(
        "shelf_auditor",
        identity=(
            "An independent auditor who trusts only what the store's own records say and treats "
            "every unsupported claim as false until proven.\n"
        ),
        watches=["tool_traffic", "final_answer"],
        on=["scout", "turn", "end"],
        model="claude-sonnet-5-5",
    )
    auditor.when(
        "find_product has returned at least one DIRECT match for penne", id="direct_match"
    ).then(
        enable_tools=["get_product"],
        enable_goal="Quote the cheapest direct hit by exact name, price and store.",
    ).when(
        "the final answer names a product, price or store that appears in no tool result",
        id="fabrication",
    ).then(flag="fabrication", fail=True)
    clerk = observer(
        "word_clerk",
        identity="A clerk who counts words and nothing else.",
        kind="code",
        watches=["final_answer"],
        on=["end"],
    )
    clerk.when("the final answer is longer than 100 words", id="too_long").check(
        word_count={"of": "final_answer", "gt": 100}
    ).then(flag="verbose")
    desk = observer(
        "policy_desk",
        identity="The policy desk; it only combines what others reported.",
        kind="group",
    )
    desk.when("a direct match was reported and no fabrication", id="ready_to_quote").all_of(
        "shelf_auditor.direct_match", "!shelf_auditor.fabrication"
    ).then(enable_goal="Deliver the final answer now.")
    return [auditor, clerk, desk]


# --- declarations -----------------------------------------------------------------------------


def test_yaml_declares_identities_conditions_and_effects(scenario: Scenario) -> None:
    auditor, clerk, desk = scenario.observers
    assert (auditor.name, auditor.kind, auditor.model) == (
        "shelf_auditor",
        "llm",
        "claude-sonnet-5-5",
    )
    assert auditor.identity.startswith("An independent auditor")
    assert auditor.watches == ["tool_traffic", "final_answer"]
    assert auditor.on == ["scout", "turn", "end"], "YAML's bare on: key is read as 'on'"
    direct, fabrication = auditor.conditions
    assert direct.id == "direct_match"
    assert direct.when == "find_product has returned at least one DIRECT match for penne"
    assert direct.then == Effect(
        enable_tools=["get_product"],
        enable_goal="Quote the cheapest direct hit by exact name, price and store.",
    )
    assert direct.otherwise == Effect() and direct.otherwise.is_empty()
    assert fabrication.then == Effect(flag="fabrication", fail=True)
    assert fabrication.check is None and fabrication.all_of == []

    assert (clerk.kind, clerk.watches, clerk.on) == ("code", ["final_answer"], ["end"])
    assert clerk.conditions[0].check == Check(word_count={"of": "final_answer", "gt": 100})
    assert clerk.conditions[0].check is not None and clerk.conditions[0].check.kind == "word_count"
    assert clerk.conditions[0].then.flag == "verbose" and clerk.conditions[0].then.fail is False

    assert desk.kind == "group"
    assert desk.watches == ["all"] and desk.on == ["scout", "end"], "the defaults"
    ready = desk.conditions[0]
    assert ready.all_of == ["shelf_auditor.direct_match", "!shelf_auditor.fabrication"]
    assert ready.references() == [
        ("shelf_auditor", "direct_match", False),
        ("shelf_auditor", "fabrication", True),
    ]
    assert scenario.observer("word_clerk") is clerk
    assert auditor.condition("fabrication") is fabrication
    with pytest.raises(KeyError):
        scenario.observer("nobody")


def test_python_dsl_builds_the_same_models_as_yaml(scenario: Scenario) -> None:
    built = [b.build() for b in dsl_observers()]
    assert built == scenario.observers
    assert [o.model_dump() for o in built] == [
        Observer.model_validate(d).model_dump() for d in observers_data()
    ]


def test_with_observers_appends_builders_and_validates(scenario_data: dict[str, Any]) -> None:
    base = parse_scenario(scenario_data)
    assert base.observers == []
    auditor, clerk, desk = dsl_observers()
    extended = base.with_observers([auditor, clerk.build(), desk])
    assert [o.name for o in extended.observers] == ["shelf_auditor", "word_clerk", "policy_desk"]
    assert base.observers == [], "the original is untouched"
    assert extended.with_observers([clerk], replace=True).observers == [clerk.build()]
    # The group's references are checked against what is already declared.
    lonely = observer("desk", identity="d", kind="group")
    lonely.when("x", id="x").all_of("shelf_auditor.direct_match")
    with pytest.raises(ValueError, match="no observer named 'shelf_auditor'"):
        base.with_observers([lonely])


def test_dsl_rejects_bad_effects_and_checks_at_the_call_site() -> None:
    builder = observer("o", identity="i", kind="code")
    with pytest.raises(ValueError, match="extra_forbidden|flag"):
        builder.when("x", id="x").then(enable_tool=["lookup"])
    with pytest.raises(ValueError, match="exactly one of"):
        builder.when("y", id="y").check(
            tool_called="lookup", regex={"of": "errors", "pattern": "a"}
        )
    with pytest.raises(ValueError, match="kind 'code' needs a check"):
        observer("o", identity="i", kind="code").when("x", id="x").build()


def test_models_observer_defaults_to_sonnet_then_per_observer() -> None:
    assert MODEL_ROLES == ("planner", "agent", "judge", "user", "observer")
    m = Models()
    assert m.observer == m.observer_model == "claude-sonnet-5-5"
    assert m.for_role("observer") == "claude-sonnet-5-5"
    # The observers' default is their own, not a copy of the agent's.
    assert Models(agent="claude-fable-5-1").observer_model == "claude-sonnet-5-5"
    with_observer = Models(observer="claude-haiku-4-5-20251001")
    assert with_observer.for_role("observer") == "claude-haiku-4-5-20251001"
    own = Observer.model_validate(
        {
            "name": "o",
            "identity": "i",
            "model": "ollama:qwen2.5:7b",
            "conditions": [{"id": "c", "when": "w"}],
        }
    )
    default = Observer.model_validate(
        {"name": "p", "identity": "i", "conditions": [{"id": "c", "when": "w"}]}
    )
    assert with_observer.model_for_observer(own) == "ollama:qwen2.5:7b"
    assert with_observer.model_for_observer(default) == "claude-haiku-4-5-20251001"
    assert Models().model_for_observer(default) == "claude-sonnet-5-5"


# --- validation errors name the field --------------------------------------------------------


def _write(tmp_path: FsPath, data: dict[str, Any]) -> FsPath:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _obs(**fields: Any) -> dict[str, Any]:
    return {
        "name": "o",
        "identity": "an identity",
        "conditions": [{"id": "c", "when": "w"}],
        **fields,
    }


@pytest.mark.parametrize(
    ("observers", "needle"),
    [
        (
            [_obs(kind="code")],
            "observers.0: Value error, conditions[0] ('c'): kind 'code' needs a check",
        ),
        (
            [
                _obs(
                    kind="code",
                    conditions=[
                        {"id": "c", "when": "w", "check": {"tool_called": "x"}, "all_of": ["a.b"]}
                    ],
                )
            ],
            "kind 'code' takes a check, not all_of/any_of",
        ),
        ([_obs(kind="group")], "kind 'group' needs all_of or any_of"),
        (
            [
                _obs(
                    kind="group",
                    conditions=[
                        {"id": "c", "when": "w", "any_of": ["a.b"], "check": {"tool_called": "x"}}
                    ],
                )
            ],
            "kind 'group' combines reports; it takes no check",
        ),
        (
            [_obs(conditions=[{"id": "c", "when": "w", "check": {"tool_called": "x"}}])],
            "kind 'llm' answers from what it watches; a check belongs to kind 'code'",
        ),
        (
            [_obs(conditions=[{"id": "c", "when": "w", "all_of": ["a.b"]}])],
            "kind 'llm' takes no all_of/any_of",
        ),
        (
            [_obs(kind="group", conditions=[{"id": "c", "when": "w", "all_of": ["nobody.x"]}])],
            "observers[0].conditions[0] (o.c): references nobody.x, but no observer named "
            "'nobody' is declared earlier",
        ),
        (
            [
                _obs(name="a"),
                _obs(
                    name="b",
                    kind="group",
                    conditions=[{"id": "c", "when": "w", "all_of": ["a.zzz"]}],
                ),
            ],
            "observer 'a' has no condition 'zzz' (it has: c)",
        ),
        (
            [
                _obs(
                    name="b", kind="group", conditions=[{"id": "c", "when": "w", "all_of": ["b.c"]}]
                )
            ],
            "references its own observer",
        ),
        (
            # Declared later: a group may only combine what is already declared (no cycles).
            [
                _obs(
                    name="g", kind="group", conditions=[{"id": "c", "when": "w", "all_of": ["a.c"]}]
                ),
                _obs(name="a"),
            ],
            "no observer named 'a' is declared earlier in the list",
        ),
        ([_obs(name="a"), _obs(name="a")], "observers[1]: duplicate observer name 'a'"),
        (
            [_obs(conditions=[{"id": "c", "when": "w"}, {"id": "c", "when": "v"}])],
            "duplicate condition id",
        ),
        ([_obs(conditions=[{"id": "Bad-Id", "when": "w"}])], "observers.0.conditions.0.id"),
        (
            [_obs(conditions=[{"id": "c", "when": "w", "all_of": ["bare"]}])],
            "observers.0.conditions.0: Value error, all_of[0] 'bare' must be",
        ),
        ([_obs(conditions=[])], "observers.0.conditions"),
        ([_obs(watches=["everything"])], "observers.0.watches.0"),
        ([_obs(on=["sometimes"])], "observers.0.on.0"),
        ([_obs(kind="judge")], "observers.0.kind"),
        ([_obs(extra=1)], "observers.0.extra"),
        (
            [
                _obs(
                    kind="code",
                    conditions=[
                        {
                            "id": "c",
                            "when": "w",
                            "check": {
                                "word_count": {"of": "final_answer", "gt": 1},
                                "tool_called": "x",
                            },
                        }
                    ],
                )
            ],
            "observers.0.conditions.0.check: Value error, check needs exactly one of "
            "word_count, regex, tool_result, tool_called (got word_count, tool_called)",
        ),
        (
            [
                _obs(
                    kind="code",
                    conditions=[
                        {"id": "c", "when": "w", "check": {"word_count": {"of": "scout", "gt": 1}}}
                    ],
                )
            ],
            "word_count.of must be one of final_answer, last_assistant, conversation, errors, "
            "got 'scout'",
        ),
        (
            [
                _obs(
                    kind="code",
                    conditions=[
                        {
                            "id": "c",
                            "when": "w",
                            "check": {"word_count": {"of": "final_answer", "gt": 1, "lt": 5}},
                        }
                    ],
                )
            ],
            "word_count needs exactly one of gt, gte, lt, lte, eq",
        ),
        (
            [
                _obs(
                    kind="code",
                    conditions=[
                        {
                            "id": "c",
                            "when": "w",
                            "check": {"regex": {"of": "conversation", "pattern": "("}},
                        }
                    ],
                )
            ],
            "regex.pattern is not a valid regular expression",
        ),
        (
            [
                _obs(
                    kind="code",
                    conditions=[
                        {
                            "id": "c",
                            "when": "w",
                            "check": {"regex": {"of": "tool_result", "pattern": "a"}},
                        }
                    ],
                )
            ],
            "regex.of must be one of final_answer, last_assistant, conversation, errors or "
            "tool_result[<tool>], got 'tool_result'",
        ),
        (
            [
                _obs(
                    kind="code",
                    conditions=[
                        {"id": "c", "when": "w", "check": {"tool_result": {"tool": "lookup"}}}
                    ],
                )
            ],
            "tool_result.where must be a non-empty matcher spec",
        ),
        (
            [_obs(conditions=[{"id": "c", "when": "w", "then": {"flag": "Not Valid"}}])],
            "observers.0.conditions.0.then: Value error, flag 'Not Valid' must match",
        ),
        (
            [_obs(conditions=[{"id": "c", "when": "w", "then": {"enable_tool": ["x"]}}])],
            "observers.0.conditions.0.then.enable_tool",
        ),
        ([{"use": "nobody"}], "observers[0]: unknown built-in observer 'nobody'; the library has:"),
        (
            [{"use": "x", "name": "y"}],
            "an observer entry with 'use' takes no other keys (got name)",
        ),
    ],
)
def test_validation_errors_name_the_field(
    tmp_path: FsPath, scenario_data: dict[str, Any], observers: list[dict[str, Any]], needle: str
) -> None:
    path = _write(tmp_path, {**scenario_data, "observers": observers})
    with pytest.raises(ScenarioError) as exc_info:
        load_scenario(path)
    assert needle in str(exc_info.value), str(exc_info.value)


def test_scenario_without_observers_is_unchanged(scenario_data: dict[str, Any]) -> None:
    s = parse_scenario(scenario_data)
    assert s.observers == [] and s.models.observer == "claude-sonnet-5-5"
    assert "observers" in Scenario.model_fields


# --- reports and transcript events ------------------------------------------------------------


def test_informant_report_shape_and_line() -> None:
    report = InformantReport(
        observer="shelf_auditor",
        condition="direct_match",
        value=True,
        evidence='"match": "direct", "total": 2',
        confidence=0.9,
        trigger="scout",
        at_event=3,
    )
    assert report.key == "shelf_auditor.direct_match" and report.shown == "true"
    assert report.line() == 'shelf_auditor.direct_match = true — "match": "direct", "total": 2'
    unknown = InformantReport(observer="a", condition="b", value=None, trigger="end")
    assert unknown.shown == "unknown" and unknown.evidence == "no evidence"
    assert unknown.confidence == 1.0 and unknown.at_event == -1
    assert (
        InformantReport(observer="a", condition="b", value=False, trigger="turn").shown == "false"
    )
    with pytest.raises(ValueError):
        InformantReport(observer="a", condition="b", value=True, trigger="later")
    with pytest.raises(ValueError):
        InformantReport(observer="a", condition="b", value=True, trigger="end", confidence=1.5)


def test_report_events_round_trip_and_rebuild_flags_and_hard_failures(tmp_path: FsPath) -> None:
    t = Transcript(scenario="s", path_id="p", mode="guided", index=0)
    t.add(SystemEvent(scenario="s", path_id="p", index=0, mode="guided"))
    first = InformantReport(
        observer="shelf_auditor",
        condition="direct_match",
        value=True,
        evidence='"match": "direct"',
        trigger="tool_result",
        at_event=1,
    )
    second = InformantReport(
        observer="shelf_auditor",
        condition="fabrication",
        value=True,
        evidence="[4] a Health Food Store",
        trigger="end",
        at_event=5,
    )
    event = t.add_reports("tool_result", [first])
    assert isinstance(event, InformantReportEvent) and event.kind == "informant_report"
    assert event.flags == [] and event.failures == []
    t.add(
        GoalEnabledEvent(
            text="Quote the cheapest direct hit.",
            reason="observer:shelf_auditor.direct_match",
            observer="shelf_auditor",
            condition="direct_match",
        )
    )
    t.add_reports(
        "end",
        [second],
        flags=["fabrication", "verbose"],
        failures=["shelf_auditor.fabrication — [4] a Health Food Store"],
    )
    t.add_reports("end", [], flags=["verbose"])  # a repeated flag is kept once
    assert t.flags == ["fabrication", "verbose"]
    assert t.hard_failures == ["shelf_auditor.fabrication — [4] a Health Food Store"]
    assert t.informant_reports() == [first, second]
    assert t.kinds() == [
        "system",
        "informant_report",
        "goal_enabled",
        "informant_report",
        "informant_report",
    ]

    path = t.write_jsonl(tmp_path / "t.jsonl")
    back = Transcript.read_jsonl(path)
    assert back.events == t.events
    assert back.flags == ["fabrication", "verbose"]
    assert back.hard_failures == ["shelf_auditor.fabrication — [4] a Health Food Store"]
    assert back.informant_reports() == [first, second]
    goal = back.events[2]
    assert isinstance(goal, GoalEnabledEvent)
    assert (goal.observer, goal.condition) == ("shelf_auditor", "direct_match")

    parsed = parse_event(
        {
            "t": "2026-10-02T00:00:00.000+00:00",
            "kind": "informant_report",
            "trigger": "scout",
            "reports": [first.model_dump()],
        }
    )
    assert isinstance(parsed, InformantReportEvent) and parsed.reports == [first]
    with pytest.raises(ValueError):
        parse_event({"t": "x", "kind": "informant_report", "trigger": "scout", "bogus": 1})


# --- the runner: code observers on a hand-built transcript -------------------------------------


from mcpsim.observers import (  # noqa: E402
    BUDGET_EXHAUSTED,
    METHOD,
    OMITTED,
    REPORT_TOOL,
    ObserverRunner,
    TriggeredEffect,
    evaluate_check,
    parse_reports,
    render_slices,
)
from mcpsim.transcript import (  # noqa: E402
    AssistantEvent,
    ErrorEvent,
    FinalResultEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolsOfferedEvent,
    UserEvent,
)
from tests.fake_llm import ScriptedLLM, structured_response, text_response  # noqa: E402

FIND_PRODUCT = {
    "query": "penne",
    "match": "direct",
    "total": 2,
    "items": [{"id": 51, "name": "Penne Rigate 500g", "price": 1.97, "store": "GreenLeaf"}],
}
FINAL = {"product_name": "Penne Rigate 500g", "price": 1.97}


def hand_built_transcript(*, words: int = 112, final: bool = True) -> Transcript:
    """[1] system, [2] tools_offered, [3] user (with "bear"), [4] assistant, [5] tool_call,
    [6] tool_result find_product (direct), [7] error (scope violation), [8] assistant final
    answer of ``words`` words plus the fenced block, [9] final_result."""
    t = Transcript(scenario="s", path_id="p", mode="guided", index=0)
    t.add(SystemEvent(scenario="s", path_id="p", index=0, mode="guided"))
    t.add(ToolsOfferedEvent(added=["find_product"], reason="initial:guided:progressive"))
    t.add(UserEvent(text="Hi, I saw a bear near the penne aisle. Cheapest penne please."))
    t.add(
        AssistantEvent(
            text="Looking.",
            tool_uses=[
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "find_product",
                    "input": {"query": "penne"},
                }
            ],
        )
    )
    t.add(ToolCallEvent(name="find_product", arguments={"query": "penne"}, tool_use_id="t1"))
    t.add(
        ToolResultEvent(
            name="find_product", is_error=False, structured=dict(FIND_PRODUCT), tool_use_id="t1"
        )
    )
    t.add(ErrorEvent(message="scope violation: list_items (not disclosed)"))
    if final:
        prose = " ".join(f"w{i}" for i in range(words))
        t.add(AssistantEvent(text=prose + "\n```json final_result\n" + json.dumps(FINAL) + "\n```"))
        t.add(FinalResultEvent(parsed=dict(FINAL), raw=json.dumps(FINAL)))
        t.final_result = dict(FINAL)
    return t


def check(**spec: Any) -> Check:
    return Check.model_validate(spec)


def test_word_count_check_measures_the_final_answer_prose() -> None:
    t = hand_built_transcript(words=112)
    assert evaluate_check(check(word_count={"of": "final_answer", "gt": 100}), t) == (
        True,
        "112 words",
    )
    assert evaluate_check(check(word_count={"of": "final_answer", "lte": 100}), t) == (
        False,
        "112 words",
    )
    assert evaluate_check(check(word_count={"of": "final_answer", "eq": 112}), t) == (
        True,
        "112 words",
    )
    # The fenced final_result block is not prose and is not counted.
    assert evaluate_check(check(word_count={"of": "last_assistant", "gt": 112}), t)[0] is True
    assert evaluate_check(check(word_count={"of": "conversation", "gte": 1}), t)[1] == (
        "134 words"
    ), "conversation counts every text turn verbatim: 12 (user) + 1 (Looking.) + 112 + 9 (block)"
    # No final answer yet: the clerk cannot tell.
    assert evaluate_check(
        check(word_count={"of": "final_answer", "gt": 100}), hand_built_transcript(final=False)
    ) == (
        None,
        "no final answer yet",
    )
    assert evaluate_check(
        check(word_count={"of": "conversation", "gt": 0}),
        Transcript(scenario="s", path_id="p", mode="m", index=0),
    ) == (
        None,
        "no conversation yet",
    )


def test_regex_check_cites_the_turn_number_or_the_slice() -> None:
    t = hand_built_transcript()
    assert evaluate_check(check(regex={"of": "conversation", "pattern": "bear"}), t) == (
        True,
        "matched 'bear' at turn 3",
    )
    assert (
        evaluate_check(check(regex={"of": "conversation", "pattern": "BEAR", "flags": "i"}), t)[0]
        is True
    )
    assert evaluate_check(check(regex={"of": "conversation", "pattern": "wolf"}), t) == (
        False,
        "no match for 'wolf' in conversation",
    )
    assert evaluate_check(check(regex={"of": "errors", "pattern": "^scope violation: "}), t) == (
        True,
        "matched '^scope violation: ' in error: scope violation: list_items (not disclosed)",
    )
    clean = hand_built_transcript()
    clean.events = [e for e in clean.events if not isinstance(e, ErrorEvent)]
    assert evaluate_check(
        check(regex={"of": "errors", "pattern": "^scope violation: "}), clean
    ) == (
        False,
        "no match for '^scope violation: ' in 0 error(s)",
    ), "no error is an answer (false), not unknown"
    assert evaluate_check(
        check(regex={"of": "tool_result[find_product]", "pattern": "Rigate"}), t
    ) == (
        True,
        "matched 'Rigate' in tool_result[find_product]",
    )
    assert evaluate_check(check(regex={"of": "tool_result[get_product]", "pattern": "x"}), t) == (
        None,
        "no result from get_product yet",
    )
    assert evaluate_check(check(regex={"of": "final_answer", "pattern": "w5 w6"}), t) == (
        True,
        "matched 'w5 w6' in the final answer",
    )
    assert evaluate_check(check(regex={"of": "final_answer", "pattern": "Penne Rigate"}), t) == (
        True,
        "matched 'Penne Rigate' in final_result",
    ), "the final answer slice is the prose, then the parsed final_result"


def test_tool_result_check_matches_the_last_structured_result() -> None:
    t = hand_built_transcript()
    assert evaluate_check(
        check(tool_result={"tool": "find_product", "where": {"match": "direct"}}), t
    ) == (
        True,
        "find_product.match == 'direct'",
    )
    assert evaluate_check(
        check(
            tool_result={"tool": "find_product", "where": {"match": "direct", "total": {"$gte": 1}}}
        ),
        t,
    ) == (True, "find_product.match == 'direct', find_product.total == 2")
    assert evaluate_check(
        check(tool_result={"tool": "find_product", "where": {"match": "relaxed"}}), t
    ) == (
        False,
        "find_product.match == 'direct' (expected $eq 'relaxed')",
    )
    assert evaluate_check(check(tool_result={"tool": "get_product", "where": {"id": 1}}), t) == (
        None,
        "no result from get_product yet",
    )
    # The LAST result counts; an error result is false with the server's message as evidence.
    t.add(ToolCallEvent(name="find_product", arguments={"query": ""}, tool_use_id="t2"))
    t.add(
        ToolResultEvent(
            name="find_product", is_error=True, text="query must not be empty", tool_use_id="t2"
        )
    )
    assert evaluate_check(
        check(tool_result={"tool": "find_product", "where": {"match": "direct"}}), t
    ) == (
        False,
        "find_product returned an error: query must not be empty",
    )
    t.add(ToolCallEvent(name="find_product", arguments={"query": "x"}, tool_use_id="t3"))
    t.add(ToolResultEvent(name="find_product", is_error=False, text="plain text", tool_use_id="t3"))
    assert evaluate_check(
        check(tool_result={"tool": "find_product", "where": {"match": "direct"}}), t
    ) == (
        None,
        "find_product returned text only, nothing structured to check",
    )


def test_tool_called_check_counts_calls() -> None:
    t = hand_built_transcript()
    assert evaluate_check(check(tool_called="find_product"), t) == (
        True,
        "find_product called 1 time(s)",
    )
    assert evaluate_check(check(tool_called="get_product"), t) == (False, "get_product not called")


# --- slices: an observer sees ONLY what it watches ----------------------------------------------


def test_render_slices_shows_only_the_watched_slices_with_turn_numbers() -> None:
    t = hand_built_transcript(words=3)
    traffic = render_slices(["tool_traffic"], t)
    assert traffic.startswith("## Tool traffic\n")
    assert '[5] tool_call find_product {"query": "penne"}' in traffic
    assert '[6] tool_result find_product (ok): {"items": [' in traffic
    assert "bear" not in traffic and "user:" not in traffic, "no user text in a tool_traffic slice"
    assert "final_result" not in traffic

    conversation = render_slices(["conversation"], t)
    assert "[3] user: Hi, I saw a bear" in conversation
    assert "[4] assistant: Looking." in conversation and "[8] assistant: w0 w1 w2" in conversation
    assert "tool_call" not in conversation and "tool_result" not in conversation

    final = render_slices(["final_answer"], t)
    assert "[8] assistant: w0 w1 w2" in final
    assert '[9] final_result: {"price": 1.97, "product_name": "Penne Rigate 500g"}' in final
    assert "[4] assistant" not in final and "bear" not in final

    everything = render_slices(["all"], t)
    for needle in (
        "[1] system: mode=guided",
        "[2] tools offered: added find_product (initial:guided:progressive)",
        "[3] user: Hi, I saw a bear",
        "[7] error: scope violation: list_items (not disclosed)",
        "[9] final_result:",
        "## Scout observations",
        "(nothing to watch yet)",
    ):
        assert needle in everything, needle

    empty = Transcript(scenario="s", path_id="p", mode="m", index=0)
    assert render_slices(["conversation", "tool_traffic"], empty) == (
        "## Conversation\n(nothing to watch yet)\n\n## Tool traffic\n(nothing to watch yet)"
    )
    assert render_slices(["scout"], None) == (
        "## Scout observations (read-only calls made before planning)\n(nothing to watch yet)"
    )


def test_tool_result_text_is_truncated_in_slices() -> None:
    t = Transcript(scenario="s", path_id="p", mode="m", index=0)
    t.add(ToolResultEvent(name="echo", is_error=False, text="x" * 5000, chars=5000))
    rendered = render_slices(["tool_traffic"], t)
    assert len(rendered) < 1400 and rendered.endswith("…")


# --- group observers: three-valued logic --------------------------------------------------------


def group_scenario(
    scenario_data: dict[str, Any],
    *,
    all_of: list[str] | None = None,
    any_of: list[str] | None = None,
) -> Scenario:
    desk: dict[str, Any] = {
        "name": "desk",
        "identity": "combines",
        "kind": "group",
        "on": ["end"],
        "conditions": [
            {
                "id": "ready",
                "when": "x and not y",
                "then": {"flag": "ready"},
                "otherwise": {"flag": "not_ready"},
            }
        ],
    }
    if all_of is not None:
        desk["conditions"][0]["all_of"] = all_of
    if any_of is not None:
        desk["conditions"][0]["any_of"] = any_of
    return parse_scenario(
        {
            **scenario_data,
            "observers": [
                {
                    "name": "a",
                    "identity": "i",
                    "on": ["end"],
                    "conditions": [{"id": "x", "when": "x"}, {"id": "y", "when": "y"}],
                },
                desk,
            ],
        }
    )


def _seed(runner: ObserverRunner, **values: bool | None) -> None:
    for cond, value in values.items():
        runner.latest[("a", cond)] = InformantReport(
            observer="a", condition=cond, value=value, trigger="end", confidence=0.8
        )


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, None, False),  # one false settles all_of whatever the rest says
        (True, None, None),  # not decidable: unknown propagates
        (None, False, None),
        (None, None, None),
    ],
)
def test_group_all_of_truth_table_with_none_propagation(
    scenario_data: dict[str, Any], x: bool | None, y: bool | None, expected: bool | None
) -> None:
    runner = ObserverRunner(
        group_scenario(scenario_data, all_of=["a.x", "!a.y"]), llm=None, include_llm=False
    )
    _seed(runner, x=x, y=y)
    [report] = asyncio.run(runner.report("end"))
    assert (report.observer, report.condition, report.trigger) == ("desk", "ready", "end")
    assert report.value is expected
    shown_x = "unknown" if x is None else str(x).lower()
    shown_not_y = "unknown" if y is None else str(not y).lower()
    assert report.evidence == f"a.x is {shown_x}, !a.y is {shown_not_y}"
    assert report.confidence == 0.8, "the weakest referenced report's confidence"


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (True, None, True),
        (None, True, True),
        (False, False, False),
        (False, None, None),
        (None, None, None),
    ],
)
def test_group_any_of_is_decidable_with_one_true(
    scenario_data: dict[str, Any], x: bool | None, y: bool | None, expected: bool | None
) -> None:
    runner = ObserverRunner(
        group_scenario(scenario_data, any_of=["a.x", "a.y"]), llm=None, include_llm=False
    )
    _seed(runner, x=x, y=y)
    [report] = asyncio.run(runner.report("end"))
    assert report.value is expected


def test_group_without_any_earlier_report_is_unknown(scenario_data: dict[str, Any]) -> None:
    runner = ObserverRunner(
        group_scenario(scenario_data, all_of=["a.x"]), llm=None, include_llm=False
    )
    [report] = asyncio.run(runner.report("end"))
    assert report.value is None and report.evidence == "a.x is unknown" and report.confidence == 1.0


def test_effects_fire_on_transitions_not_on_repeated_values(scenario_data: dict[str, Any]) -> None:
    clerk = {
        "name": "clerk",
        "identity": "counts",
        "kind": "code",
        "on": ["tool_result"],
        "conditions": [
            {
                "id": "called",
                "when": "lookup was called",
                "check": {"tool_called": "lookup"},
                "then": {"flag": "seen"},
                "otherwise": {"flag": "unseen"},
            }
        ],
    }
    runner = ObserverRunner(
        parse_scenario({**scenario_data, "observers": [clerk]}), None, include_llm=False
    )
    empty = Transcript(scenario="s", path_id="p", mode="m", index=0)
    first = asyncio.run(runner.report("tool_result", empty))
    assert [e.effect.flag for e in runner.effects(first)] == ["unseen"], "false at first: otherwise"
    again = asyncio.run(runner.report("tool_result", empty))
    assert again[0].value is False and runner.effects(again) == [], "still false: nothing new"
    called = Transcript(scenario="s", path_id="p", mode="m", index=0)
    called.add(ToolCallEvent(name="lookup", arguments={"slug": "penne"}))
    now_true = asyncio.run(runner.report("tool_result", called))
    assert [e.effect.flag for e in runner.effects(now_true)] == ["seen"], "became true: then"
    repeated = asyncio.run(runner.report("tool_result", called))
    assert repeated[0].value is True and runner.effects(repeated) == []
    assert runner.is_transition(now_true[0]) is False, "judged against the report before it"


def test_effects_follow_then_on_true_otherwise_on_false_nothing_on_unknown(
    scenario_data: dict[str, Any],
) -> None:
    scenario = group_scenario(scenario_data, all_of=["a.x"])
    runner = ObserverRunner(scenario, llm=None, include_llm=False)
    true = InformantReport(
        observer="desk", condition="ready", value=True, evidence="e", trigger="end"
    )
    false = true.model_copy(update={"value": False})
    unknown = true.model_copy(update={"value": None})
    bare = InformantReport(observer="a", condition="x", value=True, trigger="end")
    effects = runner.effects([true, false, unknown, bare])
    assert [(e.observer, e.condition, e.effect.flag) for e in effects] == [
        ("desk", "ready", "ready"),
        ("desk", "ready", "not_ready"),
    ], "a.x has empty then/otherwise, so it triggers nothing"
    assert effects[0].reason == "observer:desk.ready"
    assert effects[0].failure == "desk.ready — e"
    assert isinstance(effects[0], TriggeredEffect)


# --- llm observers through the scripted LLM -------------------------------------------------------


def llm_scenario(scenario_data: dict[str, Any], *observers: dict[str, Any]) -> Scenario:
    return parse_scenario({**scenario_data, "observers": list(observers)})


AUDITOR: dict[str, Any] = {
    "name": "shelf_auditor",
    "identity": "An independent auditor who trusts only the store's own records.",
    "watches": ["tool_traffic"],
    "on": ["tool_result", "end"],
    "conditions": [
        {"id": "direct_match", "when": "find_product has returned a DIRECT match for penne"},
        {
            "id": "fabrication",
            "when": "the final answer names a product no tool result returned",
            "then": {"flag": "fabrication", "fail": True},
        },
    ],
}


def reply(**by_id: Any) -> Any:
    return structured_response(
        REPORT_TOOL,
        {"reports": [{"condition_id": cid, **fields} for cid, fields in by_id.items()]},
    )


def test_llm_observer_prompt_carries_identity_method_conditions_and_only_the_watched_slices(
    scenario_data: dict[str, Any],
) -> None:
    scenario = llm_scenario(scenario_data, AUDITOR)
    llm = ScriptedLLM(
        [
            reply(
                direct_match={
                    "value": True,
                    "evidence": '[6] "match": "direct"',
                    "confidence": 0.95,
                },
                fabrication={"value": False, "evidence": "no evidence", "confidence": 0.7},
            )
        ]
    )
    runner = ObserverRunner(scenario, llm=llm, max_calls=5)
    t = hand_built_transcript()
    reports = asyncio.run(runner.report("tool_result", t))

    assert len(llm.calls) == 1, "ONE call per observer per trigger, covering every condition"
    call = llm.calls[0]
    assert call["model"] == scenario.models.agent, "observers default to the agent model"
    assert call["tool_choice"] == {"type": "tool", "name": REPORT_TOOL}
    assert [tool["name"] for tool in call["tools"]] == [REPORT_TOOL]
    system = call["system"]
    assert AUDITOR["identity"] in system
    assert METHOD in system
    assert "- direct_match: find_product has returned a DIRECT match for penne" in system
    assert "- fabrication: the final answer names a product no tool result returned" in system
    user = call["messages"][0]["content"]
    assert user == render_slices(["tool_traffic"], t)
    assert user.startswith("## Tool traffic\n")
    assert "bear" not in user and "user:" not in user, (
        "a tool_traffic-only observer sees no user text"
    )
    assert "## Conversation" not in user and "## Final answer" not in user

    assert [(r.observer, r.condition, r.value, r.evidence, r.confidence) for r in reports] == [
        ("shelf_auditor", "direct_match", True, '[6] "match": "direct"', 0.95),
        ("shelf_auditor", "fabrication", False, "no evidence", 0.7),
    ]
    assert all(r.trigger == "tool_result" and r.at_event == len(t.events) - 1 for r in reports)
    assert runner.latest[("shelf_auditor", "direct_match")] == reports[0]
    assert runner.usage[scenario.models.agent].input_tokens == 10
    taken = runner.take_usage()
    assert taken[scenario.models.agent].input_tokens == 10 and runner.take_usage() == {}


def test_llm_observer_uses_its_own_model_then_models_observer(
    scenario_data: dict[str, Any],
) -> None:
    own = {**AUDITOR, "model": "ollama:qwen2.5:7b"}
    scenario = llm_scenario(scenario_data, own)
    llm = ScriptedLLM([reply(), reply()])
    runner = ObserverRunner(scenario, llm=llm)
    asyncio.run(runner.report("end", hand_built_transcript()))
    assert llm.calls[0]["model"] == "ollama:qwen2.5:7b"
    scenario = scenario.model_copy(update={"models": Models(observer="claude-haiku-4-5-20251001")})
    scenario = scenario.with_observers([Observer.model_validate(AUDITOR)], replace=True)
    runner = ObserverRunner(scenario, llm=llm)
    asyncio.run(runner.report("end", hand_built_transcript()))
    assert llm.calls[1]["model"] == "claude-haiku-4-5-20251001"


def test_omitted_condition_is_unknown_and_malformed_reply_is_unknown_with_the_error(
    scenario_data: dict[str, Any],
) -> None:
    observer = Observer.model_validate(AUDITOR)
    omitted = parse_reports(
        observer,
        reply(direct_match={"value": True, "evidence": "[6] direct"}),
        trigger="end",
        at_event=3,
    )
    assert [(r.condition, r.value, r.evidence, r.confidence) for r in omitted] == [
        ("direct_match", True, "[6] direct", 1.0),
        ("fabrication", None, OMITTED, 0.0),
    ]
    prose = parse_reports(observer, text_response("I think it is fine."), trigger="end", at_event=3)
    assert [r.value for r in prose] == [None, None]
    assert all(
        r.evidence.startswith(f"malformed observer reply: did not call {REPORT_TOOL}")
        for r in prose
    )
    bad_value = parse_reports(
        observer,
        structured_response(
            REPORT_TOOL, {"reports": [{"condition_id": "direct_match", "value": "maybe"}]}
        ),
        trigger="end",
        at_event=3,
    )
    assert [r.value for r in bad_value] == [None, None]
    assert all("malformed observer reply: reports.0.value" in r.evidence for r in bad_value)
    not_object = parse_reports(
        observer, structured_response(REPORT_TOOL, {"reports": "none"}), trigger="end", at_event=0
    )
    assert all(r.value is None and "malformed observer reply" in r.evidence for r in not_object)
    # Evidence is clipped to 300 characters; confidence is clamped; empty evidence is "no evidence".
    long = parse_reports(
        observer,
        reply(
            direct_match={"value": False, "evidence": "x" * 400, "confidence": 7},
            fabrication={"value": None, "evidence": ""},
        ),
        trigger="end",
        at_event=0,
    )
    assert (
        len(long[0].evidence) == 300
        and long[0].evidence.endswith("…")
        and long[0].confidence == 1.0
    )
    assert long[1].value is None and long[1].evidence == "no evidence"


def test_observer_call_failure_is_an_unknown_report_not_a_crash(
    scenario_data: dict[str, Any],
) -> None:
    class Broken(ScriptedLLM):
        async def complete(self, **kwargs: Any) -> Any:
            raise RuntimeError("api down")

    runner = ObserverRunner(llm_scenario(scenario_data, AUDITOR), llm=Broken())
    reports = asyncio.run(runner.report("end", hand_built_transcript()))
    assert [r.value for r in reports] == [None, None]
    assert reports[0].evidence == "observer call failed: RuntimeError: api down"
    no_model = ObserverRunner(llm_scenario(scenario_data, AUDITOR), llm=None)
    reports = asyncio.run(no_model.report("end", hand_built_transcript()))
    assert [r.evidence for r in reports] == ["no observer model available"] * 2


def test_observer_budget_caps_llm_calls_and_says_so(
    scenario_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    second = {**AUDITOR, "name": "second_auditor"}
    scenario = llm_scenario(scenario_data, AUDITOR, second)
    monkeypatch.setenv("MCPSIM_OBSERVER_MAX_CALLS", "1")
    llm = ScriptedLLM([reply(direct_match={"value": True, "evidence": "[6]"})])
    runner = ObserverRunner(scenario, llm=llm)
    assert runner.max_calls == 1
    reports = asyncio.run(runner.report("end", hand_built_transcript()))
    assert len(llm.calls) == 1 and runner.calls == 1
    by_observer = {(r.observer, r.condition): r for r in reports}
    assert by_observer[("shelf_auditor", "direct_match")].value is True
    starved = by_observer[("second_auditor", "direct_match")]
    assert starved.value is None and starved.evidence == BUDGET_EXHAUSTED
    assert by_observer[("second_auditor", "fabrication")].evidence == BUDGET_EXHAUSTED
    monkeypatch.setenv("MCPSIM_OBSERVER_MAX_CALLS", "many")
    with pytest.raises(ValueError, match="MCPSIM_OBSERVER_MAX_CALLS must be an integer"):
        ObserverRunner(scenario, llm=llm)
    monkeypatch.delenv("MCPSIM_OBSERVER_MAX_CALLS")
    assert ObserverRunner(scenario, llm=llm).max_calls == 12


def test_runner_selects_observers_by_trigger_and_skips_llm_observers_when_asked(
    scenario_data: dict[str, Any],
) -> None:
    clerk = {
        "name": "clerk",
        "identity": "counts",
        "kind": "code",
        "on": ["turn"],
        "conditions": [
            {
                "id": "called",
                "when": "find_product was called",
                "check": {"tool_called": "find_product"},
            }
        ],
    }
    scenario = llm_scenario(scenario_data, AUDITOR, clerk)
    runner = ObserverRunner(scenario, llm=ScriptedLLM(), include_llm=False)
    assert [o.name for o in runner.observers] == ["clerk"]
    assert runner.fires_at("turn") and not runner.fires_at("end") and not runner.fires_at("scout")
    assert asyncio.run(runner.report("end", hand_built_transcript())) == []
    [report] = asyncio.run(runner.report("turn", hand_built_transcript()))
    assert (report.observer, report.value, report.evidence) == (
        "clerk",
        True,
        "find_product called 1 time(s)",
    )
    full = ObserverRunner(scenario, llm=ScriptedLLM())
    assert [o.name for o in full.observers_for("end")] == ["shelf_auditor"]
    assert [o.name for o in full.observers_for("turn")] == ["clerk"]
