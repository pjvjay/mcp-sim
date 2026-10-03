"""The runner UI's two isolated adapters: scenario v2 fields and the skill config reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from mcpsim.ui.scenario_view import (
    DEFAULT_CATEGORY,
    derive_title,
    derive_user_instructions,
    load_scenario_file,
    scenario_view,
)
from mcpsim.ui.skill_config import (
    SkillConfig,
    default_skill_dir,
    expand_sources,
    load_skill_config,
    parse_frontmatter,
    resolve_roles,
    split_spec,
)
from mcpsim.ui.store import run_sort_key, split_stem, status_of
from tests.conftest import fake_server_stdio_spec


def _scenario(**extra: Any) -> dict[str, Any]:
    return {
        "name": "penne",
        "role": "An impatient shopper.",
        "goal": "Find penne.",
        "instructions": ["Use lookup."],
        "expected_outcome": {"text": "A price."},
        "server": fake_server_stdio_spec(),
        **extra,
    }


def test_v2_file_loads_through_the_pre_v2_model(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(
        yaml.safe_dump(
            _scenario(
                category="Pasta",
                context={"device": "desktop", "language": "en-CA", "details": {"n": 2}},
                expected_behavior=["Calls lookup", "  "],
            )
        )
    )
    model, raw, error = load_scenario_file(path)
    assert error is None and model is not None and model.name == "penne"
    assert raw["category"] == "Pasta"
    view = scenario_view(path)
    assert view.category == "Pasta" and view.title == "Penne"
    assert view.context.device == "desktop" and view.context.location == ""
    assert view.context.details == {"n": "2"} and view.context.agent_visible is False
    assert view.expected_behavior == ["Calls lookup"] and not view.expected_behavior_derived
    assert view.user_instructions_derived
    assert view.user_instructions == (
        "You are this person: An impatient shopper.\n\nWhat you want from the assistant: "
        "Find penne."
    )


def test_invalid_v1_part_of_a_v2_file_still_reports_the_error(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(_scenario(title="T", repeat=0)))
    view = scenario_view(path)
    assert view.error is not None and "repeat" in view.error
    assert view.title == "T" and view.name == "penne"


def test_unknown_key_without_v2_fields_is_the_loader_error(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(_scenario(colour="red")))
    view = scenario_view(path)
    assert view.error is not None and "colour" in view.error


def test_unparseable_file_has_an_error_and_a_fallback_name(tmp_path: Path) -> None:
    path = tmp_path / "half-written.yaml"
    path.write_text("name: [unclosed\n")
    view = scenario_view(path)
    assert view.error is not None
    assert view.name == "half-written" and view.category == DEFAULT_CATEGORY


def test_derived_defaults() -> None:
    assert derive_title("week_under-budget") == "Week under budget"
    assert derive_user_instructions(" The head chef ", "Plan dinner.") == (
        "You are this person: The head chef\n\nWhat you want from the assistant: Plan dinner."
    )


def test_written_v2_fields_are_not_marked_derived(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(
        yaml.safe_dump(
            _scenario(user_instructions="You are Dev.", expected_behavior=["Calls lookup"])
        )
    )
    view = scenario_view(path)
    assert (view.user_instructions, view.user_instructions_derived) == ("You are Dev.", False)
    assert (view.expected_behavior, view.expected_behavior_derived) == (["Calls lookup"], False)
    # No expected_behavior: the instructions are graded instead, and the view says so.
    plain = tmp_path / "plain.yaml"
    plain.write_text(yaml.safe_dump(_scenario()))
    v1 = scenario_view(plain)
    assert (v1.expected_behavior, v1.expected_behavior_derived) == (["Use lookup."], True)


def test_snapshot_agent_sop_fields(tmp_path: Path) -> None:
    from mcpsim.ui.scenario_view import build_view

    snapshot = _scenario(
        agent={
            "skill": "env:RECIPE_SHOPPER_SKILL",
            "skill_name": "recipe-shopper",
            "skill_path": "/abs/skills/recipe-shopper/SKILL.md",
            "skill_text": "1. Search first.",
            "notes": "No write tools.",
        }
    )
    view = build_view(snapshot, file="scenario.json")
    assert view.agent_skill == "env:RECIPE_SHOPPER_SKILL"
    assert view.agent_skill_name == "recipe-shopper"
    assert view.agent_skill_path == "/abs/skills/recipe-shopper/SKILL.md"
    assert view.agent_skill_text == "1. Search first."
    assert view.agent_notes == "No write tools."


def test_split_spec() -> None:
    assert split_spec("claude-opus-5-5") == ("anthropic", "claude-opus-5-5")
    assert split_spec("anthropic:claude-haiku-4-5-20251001") == (
        "anthropic",
        "claude-haiku-4-5-20251001",
    )
    assert split_spec("ollama:llama3.2:3b") == ("ollama", "llama3.2:3b")


def test_frontmatter() -> None:
    meta, body = parse_frontmatter("---\nmodel: x\n---\nHello {{ name }}\n")
    assert meta == {"model": "x"} and body == "Hello {{ name }}\n"
    assert parse_frontmatter("no frontmatter") == ({}, "no frontmatter")


def test_role_precedence_lowest_to_highest(tmp_path: Path) -> None:
    roles = tmp_path / "roles"
    roles.mkdir()
    (roles / "planner.md").write_text("---\nprovider: anthropic\nmodel: from-frontmatter\n---\n")
    (roles / "agent.md").write_text("---\nprovider: anthropic\nmodel: from-frontmatter\n---\n")
    (roles / "judge.md").write_text("---\nprovider: anthropic\nmodel: from-frontmatter\n---\n")
    (roles / "user.md").write_text("---\nprovider: anthropic\nmodel: from-frontmatter\n---\n")
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "defaults": {"agent": "from-defaults", "judge": "from-defaults", "user": "u-d"},
                "overrides": [
                    {"match": {"name": "pen*"}, "models": {"judge": "from-override", "user": "o"}},
                    {"match": {"name": "other"}, "models": {"user": "never"}},
                    {"match": {"category": "Pasta", "name": "nope"}, "models": {"user": "never"}},
                ],
            }
        )
    )
    config = load_skill_config(tmp_path)
    base = resolve_roles(config)
    assert {r: (base[r].model, base[r].source) for r in base} == {
        "planner": ("from-frontmatter", "frontmatter"),
        "agent": ("from-defaults", "config defaults"),
        "user": ("u-d", "config defaults"),
        "observer": ("claude-sonnet-5-5", "built-in"),
        "judge": ("from-defaults", "config defaults"),
    }
    scoped = resolve_roles(
        config,
        scenario_name="penne",
        category="Pasta",
        scenario_models={"user": "from-scenario", "agent": "ollama:qwen2.5:7b"},
        run_models={"user": "from-run"},
    )
    assert (scoped["judge"].model, scoped["judge"].source) == ("from-override", "config override")
    assert (scoped["user"].model, scoped["user"].source) == ("from-run", "run override")
    assert (scoped["agent"].provider, scoped["agent"].model) == ("ollama", "qwen2.5:7b")
    # Every role has its own built-in default; the observer does not follow the agent.
    assert (scoped["observer"].spec, scoped["observer"].source) == (
        "anthropic:claude-sonnet-5-5",
        "built-in",
    )


def test_no_skill_means_builtin_defaults() -> None:
    roles = resolve_roles(SkillConfig())
    assert {r: (roles[r].spec, roles[r].source) for r in roles} == {
        "planner": ("anthropic:claude-opus-5-5", "built-in"),
        "agent": ("anthropic:claude-sonnet-5-5", "built-in"),
        "user": ("anthropic:claude-haiku-4-5-20251001", "built-in"),
        "observer": ("anthropic:claude-sonnet-5-5", "built-in"),
        "judge": ("anthropic:claude-opus-5-5", "built-in"),
    }
    assert load_skill_config(None).warnings


def test_ollama_planner_uses_the_local_planner_role_file(tmp_path: Path) -> None:
    roles = tmp_path / "roles"
    roles.mkdir()
    (roles / "planner.md").write_text(
        "---\nrole: planner\nprovider: anthropic\nmodel: claude-opus-5-5\nmax_tokens: 8000\n"
        "---\nPlan {{ scenario }}.\n"
    )
    (roles / "planner-local.md").write_text(
        "---\nrole: planner\nprovider: ollama\nmodel: command-r7b\ntemperature: 0.1\n"
        "max_tokens: 2048\n---\nPlan locally.\n"
    )
    config = load_skill_config(tmp_path)
    api = resolve_roles(config)["planner"]
    assert (api.spec, api.file, api.max_tokens) == (
        "anthropic:claude-opus-5-5",
        str(roles / "planner.md"),
        8000,
    )
    # planner-local.md is not a role of its own and never sets the API planner's model.
    assert "planner-local" not in resolve_roles(config)
    local = resolve_roles(config, run_models={"planner": "ollama:command-r7b"})["planner"]
    assert (local.spec, local.source) == ("ollama:command-r7b", "run override")
    assert (local.file, local.temperature, local.max_tokens) == (
        str(roles / "planner-local.md"),
        0.1,
        2048,
    )


def test_default_skill_dir_order(tmp_path: Path) -> None:
    assert default_skill_dir("cli", {"MCPSIM_SKILL": "env"}) == Path("cli")
    assert default_skill_dir(None, {"MCPSIM_SKILL": str(tmp_path)}) == tmp_path


def test_expand_sources_dirs_globs_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a" / "deep").mkdir(parents=True)
    (tmp_path / "a" / "one.yaml").write_text("x")
    (tmp_path / "a" / "deep" / "two.yml").write_text("x")
    (tmp_path / "a" / "skip.md").write_text("x")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "three.json").write_text("{}")
    monkeypatch.setenv("MCPSIM_TEST_SCEN_DIR", str(tmp_path / "b"))
    monkeypatch.delenv("MCPSIM_UNSET_VAR_X", raising=False)
    files, warnings = expand_sources(
        ["a", "$MCPSIM_TEST_SCEN_DIR", "a/*.yaml", "missing", "$MCPSIM_UNSET_VAR_X/x"],
        [tmp_path],
    )
    root = tmp_path.resolve()
    assert files == [
        root / "a" / "deep" / "two.yml",
        root / "a" / "one.yaml",
        root / "b" / "three.json",
    ]
    assert warnings == [
        "scenarios entry 'missing' matched nothing",
        "scenarios entry '$MCPSIM_UNSET_VAR_X/x': environment variable not set",
    ]


def test_run_ids_sort_by_time_including_same_second_suffixes() -> None:
    ids = ["20261002T120836Z-10", "20261002T120836Z", "20261002T120836Z-2", "20261003T000000Z"]
    assert sorted(ids, key=run_sort_key, reverse=True) == [
        "20261003T000000Z",
        "20261002T120836Z-10",
        "20261002T120836Z-2",
        "20261002T120836Z",
    ]


def test_stem_and_status_helpers() -> None:
    assert split_stem("happy-dry-run-guided-0") == ("happy-dry-run", "guided", 0)
    assert split_stem("1-free-12") == ("1", "free", 12)
    assert split_stem("odd") == ("odd", "", None)
    assert status_of(0, 0) == "incomplete"
    assert status_of(3, 3) == "passed" and status_of(3, 0) == "failed"
    assert status_of(3, 1) == "partial"


def test_computed_pass_k_matches_the_report_definition() -> None:
    from mcpsim.ui.store import Store

    def v(path: str, mode: str, passed: bool) -> dict[str, Any]:
        return {"path_id": path, "mode": mode, "passed": passed}

    full = {"a-guided-0": v("a", "guided", True), "a-guided-1": v("a", "guided", True)}
    assert Store._computed_pass_k(full) == {"k": 2, "all_passed": True, "computed": True}
    # Every run passed, but one cell is short of k runs: pass^k does not hold.
    short = {**full, "a-free-0": v("a", "free", True)}
    assert Store._computed_pass_k(short) == {"k": 2, "all_passed": False, "computed": True}
    failed = {**full, "a-guided-1": v("a", "guided", False)}
    assert Store._computed_pass_k(failed) == {"k": 2, "all_passed": False, "computed": True}
    assert Store._computed_pass_k({}) is None
