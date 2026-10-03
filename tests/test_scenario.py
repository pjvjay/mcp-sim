from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from mcpsim.llm import is_local_model
from mcpsim.planner import planner_profile
from mcpsim.scenario import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_CATEGORY,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PLANNER_MODEL,
    MODEL_ROLES,
    AgentSpec,
    Context,
    Scenario,
    ScenarioError,
    default_title,
    default_user_instructions,
    load_scenario,
    parse_scenario,
    read_skill,
    resolve_skill_path,
    split_frontmatter,
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


# --- tools policy (DESIGN §2 "Tool scoping and disclosure") ----------------------------------


def test_tools_policy_defaults_allow_everything_all_at_once(scenario_data: dict[str, Any]) -> None:
    s = parse_scenario(scenario_data)
    assert s.tools.allow == ["*"]
    assert s.tools.deny == []
    assert s.tools.disclosure == "all"
    assert s.tools.initial is None
    assert s.tools.discover_tool is True


def test_tools_policy_is_parsed(scenario_data: dict[str, Any]) -> None:
    scenario_data["tools"] = {
        "deny": ["submit_*", "review_*"],
        "disclosure": "progressive",
        "initial": ["lookup", "list_*"],
        "discover_tool": False,
    }
    s = parse_scenario(scenario_data)
    assert s.tools.allow == ["*"]
    assert s.tools.deny == ["submit_*", "review_*"]
    assert s.tools.disclosure == "progressive"
    assert s.tools.initial == ["lookup", "list_*"]
    assert s.tools.discover_tool is False
    scenario_data["tools"] = {"allow": ["find_product", "get_product"], "disclosure": "plan"}
    s = parse_scenario(scenario_data)
    assert s.tools.allow == ["find_product", "get_product"] and s.tools.disclosure == "plan"


@pytest.mark.parametrize(
    ("tools", "needle"),
    [
        ({"initial": ["lookup"]}, "tools.initial only applies to disclosure 'progressive'"),
        (
            {"disclosure": "plan", "initial": ["lookup"]},
            "tools.initial only applies to disclosure 'progressive'",
        ),
        # Reveal rules were removed from the design (v2): the key is simply unknown.
        ({"reveal": [{"after_tool": "lookup", "tools": ["echo"]}]}, "tools.reveal"),
        ({"disclosure": "sometimes"}, "tools.disclosure"),
        ({"allow": []}, "tools.allow must list at least one glob"),
        ({"deny": ["ok", " "]}, "tools.deny[1] is blank"),
        ({"allow": "lookup"}, "tools.allow"),
    ],
)
def test_tools_policy_validation_errors_name_the_field(
    tmp_path: Path, scenario_data: dict[str, Any], tools: dict[str, Any], needle: str
) -> None:
    scenario_data["tools"] = tools
    path = _write(tmp_path, scenario_data)
    with pytest.raises(ScenarioError) as exc_info:
        load_scenario(path)
    assert needle in str(exc_info.value)


# --- scenario v2 (DESIGN §3 "Scenario v2") ----------------------------------------------------

SKILL_MD = """---
name: penne-finder
description: Finds the cheapest penne. Use when someone asks for penne prices.
---

# Penne finder

1. Call lookup with the slug.
2. Quote the price and store verbatim.
"""
SKILL_BODY = (
    "# Penne finder\n\n1. Call lookup with the slug.\n2. Quote the price and store verbatim."
)


def _skill(folder: Path, text: str = SKILL_MD) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_v1_file_gets_the_v2_defaults(scenario_path: Path, scenario_data: dict[str, Any]) -> None:
    s = load_scenario(scenario_path)
    assert s.category == DEFAULT_CATEGORY == "Uncategorized"
    assert s.title == s.display_title == "Fake lookup"
    assert s.user_instructions == s.simulated_user_instructions == (
        "You are this person: A shopper who wants the price of penne and will not accept "
        "guesses.\n\nWhat you want from the assistant: Find the price and store of penne."
    )
    assert s.user_instructions == default_user_instructions(s.role, s.goal)
    assert s.expected_behavior == s.behaviors == scenario_data["instructions"]
    assert s.context == Context() and s.context.is_empty() and s.context.items() == []
    assert s.context.agent_visible is False
    assert s.agent == AgentSpec() and s.agent.has_sop is False
    assert s.models.explicit() == {}


def test_v2_fields_are_parsed(scenario_data: dict[str, Any]) -> None:
    scenario_data.update(
        category="Product lookup",
        title="Penne, cheapest first",
        user_instructions="You are Dev, a bargain hunter. Ask for the cheapest penne.",
        context={
            "device": "desktop web",
            "location": "Vancouver, BC (49.2827, -123.1207)",
            "language": "en",
            "details": {"time": "Friday 6 pm", "basket": ["penne"]},
            "agent_visible": True,
        },
        expected_behavior=["  Calls lookup before quoting a price  ", "Quotes the store verbatim"],
        agent={"notes": "There is no shell here."},
        models={"agent": "claude-fable-5-1"},
    )
    s = parse_scenario(scenario_data)
    assert (s.category, s.title) == ("Product lookup", "Penne, cheapest first")
    assert s.user_instructions == "You are Dev, a bargain hunter. Ask for the cheapest penne."
    assert s.context.items() == [
        ("device", "desktop web"),
        ("location", "Vancouver, BC (49.2827, -123.1207)"),
        ("language", "en"),
        ("time", "Friday 6 pm"),
        ("basket", '["penne"]'),
    ]
    assert s.context.agent_visible is True
    assert s.expected_behavior == [
        "Calls lookup before quoting a price",
        "Quotes the store verbatim",
    ]
    assert s.behaviors != s.instructions, "explicit expected behaviour replaces the default"
    assert s.agent.notes == "There is no shell here." and s.agent.has_sop is False
    assert s.models.explicit() == {"agent": "claude-fable-5-1"}
    assert default_title("week_under-budget.v2") == "Week under budget v2"


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda d: d.update(category="  "), "category must not be blank"),
        (lambda d: d.update(title=""), "title must not be blank"),
        (lambda d: d.update(user_instructions=" "), "user_instructions must not be blank"),
        (lambda d: d.update(expected_behavior=["ok", " "]), "expected_behavior[1] is blank"),
        (lambda d: d.update(context={"device": " "}), "context.device must not be blank"),
        (lambda d: d.update(context={"timezone": "PST"}), "context.timezone"),
        (lambda d: d.update(agent={"skill": " "}), "agent.skill must not be blank"),
        (lambda d: d.update(agent={"notes": ""}), "agent.notes must not be blank"),
        (lambda d: d.update(agent={"sop": "x"}), "agent.sop"),
    ],
)
def test_v2_validation_errors_name_the_field(
    tmp_path: Path, scenario_data: dict[str, Any], mutate: Any, needle: str
) -> None:
    mutate(scenario_data)
    with pytest.raises(ScenarioError) as exc_info:
        load_scenario(_write(tmp_path, scenario_data))
    assert needle in str(exc_info.value)


def test_a_relative_skill_resolves_against_the_scenario_file_not_the_cwd(
    tmp_path: Path, scenario_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = tmp_path / "suite"
    skill = _skill(suite / "skills" / "penne-finder")
    scenario_data["agent"] = {"skill": "skills/penne-finder/SKILL.md", "notes": "No shell."}
    path = _write(suite, scenario_data)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    s = load_scenario(path)
    assert s.agent.skill == "skills/penne-finder/SKILL.md"
    assert s.agent.skill_path == str(skill.resolve())
    assert s.agent.skill_name == "penne-finder"
    assert s.agent.skill_text == SKILL_BODY, "frontmatter stripped, body verbatim"
    assert "description:" not in s.agent.skill_text and "---" not in s.agent.skill_text
    assert s.agent.has_sop and s.agent.notes == "No shell."
    # A directory means its SKILL.md.
    scenario_data["agent"] = {"skill": "skills/penne-finder"}
    assert load_scenario(_write(suite, scenario_data)).agent.skill_text == SKILL_BODY
    # An absolute path works from anywhere.
    scenario_data["agent"] = {"skill": str(skill)}
    assert load_scenario(_write(tmp_path, scenario_data, "abs.yaml")).agent.skill_text == SKILL_BODY


def test_an_env_skill_reads_the_path_from_the_variable(
    tmp_path: Path, scenario_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _skill(tmp_path / "anywhere" / "penne-finder")
    scenario_data["agent"] = {"skill": "env:PENNE_SKILL"}
    path = _write(tmp_path, scenario_data)

    monkeypatch.delenv("PENNE_SKILL", raising=False)
    with pytest.raises(ScenarioError) as exc_info:
        load_scenario(path)
    assert "skill 'env:PENNE_SKILL': environment variable PENNE_SKILL is not set" in str(
        exc_info.value
    )

    monkeypatch.setenv("PENNE_SKILL", str(skill))
    s = load_scenario(path)
    assert (s.agent.skill_name, s.agent.skill_text) == ("penne-finder", SKILL_BODY)
    assert s.agent.skill_path == str(skill.resolve())
    # A relative value resolves against the working directory the variable was set in.
    monkeypatch.chdir(tmp_path / "anywhere")
    monkeypatch.setenv("PENNE_SKILL", "penne-finder/SKILL.md")
    assert load_scenario(path).agent.skill_path == str(skill.resolve())
    assert resolve_skill_path("env:PENNE_SKILL") == skill.resolve()
    with pytest.raises(ValueError, match="name the environment variable"):
        resolve_skill_path("env:")


def test_a_missing_skill_file_is_a_clear_error(
    tmp_path: Path, scenario_data: dict[str, Any]
) -> None:
    scenario_data["agent"] = {"skill": "skills/nope/SKILL.md"}
    path = _write(tmp_path, scenario_data)
    with pytest.raises(ScenarioError) as exc_info:
        load_scenario(path)
    message = str(exc_info.value)
    assert str(path) in message and "agent" in message
    assert (
        f"skill 'skills/nope/SKILL.md': no SKILL.md at {tmp_path / 'skills/nope/SKILL.md'}"
        in message
    )
    (tmp_path / "empty").mkdir()
    scenario_data["agent"] = {"skill": "empty"}
    with pytest.raises(ScenarioError, match="no SKILL.md at .*empty/SKILL.md"):
        load_scenario(_write(tmp_path, scenario_data))


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("---\nname: x\n# never closed\n", "never closed"),
        ("---\n- a list\n---\nbody\n", "must be a mapping"),
        ("---\nname: [unclosed\n---\nbody\n", "not valid YAML"),
        ("---\nname: x\n---\n\n   \n", "no procedure after the frontmatter"),
    ],
)
def test_a_broken_skill_file_is_a_clear_error(
    tmp_path: Path, scenario_data: dict[str, Any], text: str, needle: str
) -> None:
    _skill(tmp_path / "bad", text)
    scenario_data["agent"] = {"skill": "bad/SKILL.md"}
    with pytest.raises(ScenarioError, match=needle):
        load_scenario(_write(tmp_path, scenario_data))


def test_frontmatter_is_optional_and_the_name_falls_back_to_the_folder(tmp_path: Path) -> None:
    assert split_frontmatter("# Just a body\n\nstep 1") == ({}, "# Just a body\n\nstep 1")
    assert split_frontmatter("﻿---\nname: n\n---\nbody") == ({"name": "n"}, "body")
    plain = _skill(tmp_path / "recipe-shopper", "# Recipe shopper\n\n1. Fetch the page.\n")
    assert read_skill(plain) == ("recipe-shopper", "# Recipe shopper\n\n1. Fetch the page.")
    nameless = _skill(tmp_path / "folder-name", "---\ndescription: d\n---\nbody\n")
    assert read_skill(nameless) == ("folder-name", "body")


def test_the_resolved_sop_travels_with_the_scenario_and_is_never_reread(
    tmp_path: Path, scenario_data: dict[str, Any]
) -> None:
    """scenario.json records the procedure the agent ran on; a re-judge needs no SKILL.md."""
    skill = _skill(tmp_path / "skills" / "penne-finder")
    scenario_data["agent"] = {"skill": "skills/penne-finder"}
    s = load_scenario(_write(tmp_path, scenario_data))
    saved = s.model_dump_json()
    skill.unlink()
    again = Scenario.model_validate_json(saved)
    assert again.agent == s.agent and again.agent.skill_text == SKILL_BODY
    assert again.with_observers([]).agent.skill_text == SKILL_BODY
    # An inline procedure needs no file at all.
    inline = parse_scenario({**scenario_data, "agent": {"skill_text": "1. Look it up."}})
    assert (inline.agent.skill_name, inline.agent.has_sop) == ("inline", True)


# pantry-api's skills/recipe-shopper/SKILL.md, when this machine has it (the variable the
# recipe scenarios use); the test is skipped elsewhere.
RECIPE_SHOPPER = os.environ.get("RECIPE_SHOPPER_SKILL", "")


@pytest.mark.skipif(
    not RECIPE_SHOPPER or not Path(RECIPE_SHOPPER).is_file(),
    reason="RECIPE_SHOPPER_SKILL does not point at pantry-api's recipe-shopper SKILL.md",
)
def test_the_real_recipe_shopper_skill_loads_without_its_frontmatter(
    tmp_path: Path, scenario_data: dict[str, Any]
) -> None:
    scenario_data["agent"] = {"skill": "env:RECIPE_SHOPPER_SKILL"}
    s = load_scenario(_write(tmp_path, scenario_data))
    assert s.agent.skill_name == "recipe-shopper"
    assert s.agent.skill_text is not None
    assert s.agent.skill_text.startswith("# Recipe shopper")
    assert not s.agent.skill_text.startswith("---") and "\ndescription:" not in s.agent.skill_text


# --- the repository's own scenarios ------------------------------------------------------------

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"
PANTRY_CATEGORIES = {"Product lookup", "Recipe planning", "Provenance", "Origin submissions"}
VANCOUVER = "Vancouver, BC (49.2827, -123.1207)"


def _repo_scenarios() -> list[Path]:
    return sorted(p for p in SCENARIOS_DIR.rglob("*") if p.suffix in (".yaml", ".yml", ".json"))


def test_no_scenario_in_the_repo_names_a_local_model() -> None:
    files = _repo_scenarios()
    assert len(files) >= 6
    for path in files:
        assert "ollama" not in path.read_text(encoding="utf-8").lower(), path
        s = load_scenario(path)
        for role in MODEL_ROLES:
            assert not is_local_model(s.models.for_role(role)), (path, role)
        for obs in s.observers:
            assert not is_local_model(s.models.model_for_observer(obs)), (path, obs.name)
        assert planner_profile(s) == "hosted", path


def test_the_pantry_scenarios_are_v2() -> None:
    for path in sorted((SCENARIOS_DIR / "pantry").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in ("category", "title", "user_instructions", "context", "expected_behavior"):
            assert key in data, (path.name, key)
        s = load_scenario(path)
        assert s.category in PANTRY_CATEGORIES, path.name
        assert s.simulated_user_instructions.startswith("You are "), path.name
        assert "Vancouver" in s.simulated_user_instructions, path.name
        assert (s.context.location, s.context.language) == (VANCOUVER, "en"), path.name
        assert s.context.device in ("desktop web", "mobile web"), path.name
        assert len(s.behaviors) >= 4 and s.behaviors != s.instructions, path.name
        assert s.expected_outcome.json, f"{path.name} keeps its deterministic checks"
        assert not s.agent.has_sop, "the pantry suite tests the server, not a skill"
