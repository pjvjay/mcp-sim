"""Observer declarations (DESIGN §2b): YAML and the Python DSL build identical models, and
validation errors name the field."""

from __future__ import annotations

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


def test_models_observer_defaults_to_the_agent_model_then_per_observer() -> None:
    assert MODEL_ROLES == ("planner", "agent", "judge", "user", "observer")
    m = Models()
    assert m.observer is None and m.observer_model == m.agent
    assert m.for_role("observer") == m.agent
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
    assert Models().model_for_observer(default) == Models().agent


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
    assert s.observers == [] and s.models.observer is None
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
