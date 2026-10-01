from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from mcpsim.scenario import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PLANNER_MODEL,
    Scenario,
    ScenarioError,
    load_scenario,
    parse_scenario,
)


def _write(tmp_path: Path, data: dict[str, Any], name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_load_valid_yaml_applies_defaults(scenario_path: Path) -> None:
    s = load_scenario(scenario_path)
    assert isinstance(s, Scenario)
    assert s.name == "fake-lookup"
    assert s.instructions[0].startswith("Use the lookup tool")
    assert s.expected_outcome.json == {
        "slug": "penne",
        "price": {"$gt": 0},
        "origin_status": "verified",
    }
    assert s.server.kind == "stdio"
    assert s.server.stdio is not None and s.server.stdio.args == ["-m", "tests.fake_server"]
    assert s.server.http is None
    assert (s.repeat, s.judge_votes, s.concurrency) == (3, 3, 4)
    assert (s.budgets.max_turns, s.budgets.max_tool_calls, s.budgets.max_cost_usd) == (
        12,
        20,
        1.0,
    )
    assert (s.models.agent, s.models.planner, s.models.judge) == (
        DEFAULT_AGENT_MODEL,
        DEFAULT_PLANNER_MODEL,
        DEFAULT_JUDGE_MODEL,
    )


def test_load_json_file(tmp_path: Path, scenario_data: dict[str, Any]) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(scenario_data), encoding="utf-8")
    assert load_scenario(path).name == "fake-lookup"


def test_http_server_and_overrides(tmp_path: Path, scenario_data: dict[str, Any]) -> None:
    scenario_data["server"] = {
        "http": {"url": "https://host/pantry/api/mcp", "bearer_env": "PANTRY_MCP_TOKEN"}
    }
    scenario_data["repeat"] = 1
    scenario_data["judge_votes"] = 5
    scenario_data["budgets"] = {"max_turns": 3}
    scenario_data["models"] = {"agent": "claude-haiku-4-5-20251001"}
    scenario_data["concurrency"] = 2
    s = load_scenario(_write(tmp_path, scenario_data))
    assert s.server.kind == "http"
    assert s.server.http is not None and s.server.http.bearer_env == "PANTRY_MCP_TOKEN"
    assert s.repeat == 1 and s.judge_votes == 5 and s.concurrency == 2
    assert s.budgets.max_turns == 3 and s.budgets.max_tool_calls == 20
    assert s.models.agent == "claude-haiku-4-5-20251001"
    assert s.models.judge == DEFAULT_JUDGE_MODEL


def test_expected_outcome_text_only_is_fine(scenario_data: dict[str, Any]) -> None:
    scenario_data["expected_outcome"] = {"text": "A price."}
    s = parse_scenario(scenario_data)
    assert s.expected_outcome.json is None


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda d: d.update(expected_outcome={}), "expected_outcome"),
        (lambda d: d.update(expected_outcome={"text": "   "}), "must not be blank"),
        (lambda d: d.update(server={}), "exactly one of 'stdio' or 'http'"),
        (
            lambda d: d.update(
                server={"stdio": {"command": "x"}, "http": {"url": "https://h/mcp"}}
            ),
            "exactly one of 'stdio' or 'http'",
        ),
        (lambda d: d.update(server={"stdio": {}}), "server.stdio.command"),
        (lambda d: d.pop("goal"), "goal"),
        (lambda d: d.pop("role"), "role"),
        (lambda d: d.update(name="has space"), "name"),
        (lambda d: d.update(name=""), "name"),
        (lambda d: d.update(instructions=["ok", "  "]), "instructions[1] is blank"),
        (lambda d: d.update(repeat=0), "repeat"),
        (lambda d: d.update(judge_votes=0), "judge_votes"),
        (lambda d: d.update(budgets={"max_cost_usd": 0}), "budgets.max_cost_usd"),
        (lambda d: d.update(unknown_key=1), "unknown_key"),
        (lambda d: d.update(models={"agent": 5}), "models.agent"),
    ],
)
def test_validation_errors_name_the_field(
    tmp_path: Path, scenario_data: dict[str, Any], mutate: Any, needle: str
) -> None:
    mutate(scenario_data)
    path = _write(tmp_path, scenario_data)
    with pytest.raises(ScenarioError) as exc_info:
        load_scenario(path)
    message = str(exc_info.value)
    assert str(path) in message
    assert needle in message


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match="top level must be a mapping"):
        load_scenario(path)


def test_unparseable_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match="cannot parse scenario"):
        load_scenario(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="cannot read scenario"):
        load_scenario(tmp_path / "nope.yaml")
