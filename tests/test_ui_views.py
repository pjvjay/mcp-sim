"""What the runner UI reads through the real loaders: scenario views from the scenario model,
the skill's roles and per-scenario resolution from :mod:`mcpsim.skill`, and the store's
helpers (scenario sources, run ids, pass^k)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from mcpsim.scenario import DEFAULT_CATEGORY, load_scenario
from mcpsim.skill import CHECKOUT_DIR, load_skill
from mcpsim.ui.scenario_view import load_view, snapshot_view, view_from_scenario
from mcpsim.ui.skill_view import SkillSource, overrides_json, role_rows, scenario_settings
from mcpsim.ui.store import Store, run_sort_key, split_stem, status_of
from tests.conftest import fake_server_stdio_spec


def _scenario(**extra: Any) -> dict[str, Any]:
    return {
        "name": "penne",
        "role": "An impatient shopper.",
        "goal": "Find penne.",
        "instructions": ["Use lookup."],
        "expected_outcome": {"text": "A price.", "json": {"price": {"$gt": 0}}},
        "server": fake_server_stdio_spec(),
        **extra,
    }


def _write(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# --- scenario views ---------------------------------------------------------------------------


def test_a_v2_file_is_read_through_the_scenario_model(tmp_path: Path) -> None:
    sop = tmp_path / "skills" / "shop" / "SKILL.md"
    sop.parent.mkdir(parents=True)
    sop.write_text("---\nname: shopper-sop\n---\nSearch first.\n")
    path = _write(
        tmp_path / "s.yaml",
        _scenario(
            category="Pasta",
            context={
                "device": "desktop",
                "language": "en-CA",
                "details": {"n": 2, "who": "guest", "budget": {"max": 20}},
            },
            expected_behavior=["  Calls lookup  "],
            agent={"skill": "skills/shop", "notes": "No writes."},
            models={"judge": "anthropic:claude-fable-5-1"},
        ),
    )
    view, scenario = load_view(path, display_path="s.yaml")
    assert scenario is not None and view.error is None and view.file == "s.yaml"
    assert (view.category, view.title) == ("Pasta", "Penne")
    assert view.context.device == "desktop" and view.context.location == ""
    # details as Context.items() renders them: a non-string value as compact JSON, file order.
    assert view.context.details == {"n": "2", "who": "guest", "budget": '{"max": 20}'}
    assert view.context.agent_visible is False
    assert view.expected_behavior == ["Calls lookup"] and not view.expected_behavior_derived
    assert view.user_instructions_derived
    assert view.user_instructions == (
        "You are this person: An impatient shopper.\n\nWhat you want from the assistant: "
        "Find penne."
    )
    # The relative SOP path resolved against the scenario file; its frontmatter is stripped.
    assert (view.agent_skill, view.agent_skill_name) == ("skills/shop", "shopper-sop")
    assert view.agent_skill_path == str(sop.resolve())
    assert (view.agent_skill_text, view.agent_notes) == ("Search first.", "No writes.")
    assert view.models == {"judge": "anthropic:claude-fable-5-1"}, "the models the file names"
    assert view.expected_outcome_json == {"price": {"$gt": 0}}
    assert (view.server, view.repeat, view.judge_votes) == ("stdio", 3, 3)


def test_written_v2_fields_are_not_marked_derived(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "s.yaml",
        _scenario(user_instructions="You are Dev.", expected_behavior=["Calls lookup"]),
    )
    view, _ = load_view(path)
    assert (view.user_instructions, view.user_instructions_derived) == ("You are Dev.", False)
    assert (view.expected_behavior, view.expected_behavior_derived) == (["Calls lookup"], False)
    # No expected_behavior: the instructions are graded instead, and the view says so.
    v1, _ = load_view(_write(tmp_path / "plain.yaml", _scenario()))
    assert (v1.expected_behavior, v1.expected_behavior_derived) == (["Use lookup."], True)
    assert v1.models == {} and v1.category == DEFAULT_CATEGORY


def test_a_file_the_runner_would_refuse_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UI_VIEW_SOP", raising=False)
    path = _write(
        tmp_path / "s.yaml",
        _scenario(title="T", category="Pasta", agent={"skill": "env:UI_VIEW_SOP"}),
    )
    view, scenario = load_view(path)
    assert scenario is None
    assert view.error is not None and "UI_VIEW_SOP is not set" in view.error
    # The raw file still places it in the list.
    assert (view.name, view.title, view.category) == ("penne", "T", "Pasta")
    assert view.user_instructions.startswith("You are this person: An impatient shopper.")


def test_invalid_and_unparseable_files(tmp_path: Path) -> None:
    view, _ = load_view(_write(tmp_path / "s.yaml", _scenario(repeat=0)))
    assert view.error is not None and "repeat" in view.error
    view, _ = load_view(_write(tmp_path / "c.yaml", _scenario(colour="red")))
    assert view.error is not None and "colour" in view.error
    half = tmp_path / "half-written.yaml"
    half.write_text("name: [unclosed\n")
    view, scenario = load_view(half)
    assert scenario is None and view.error is not None
    assert view.name == "half-written" and view.category == DEFAULT_CATEGORY


def test_a_runs_snapshot_is_validated_like_a_rejudge_reads_it(tmp_path: Path) -> None:
    # What the runner writes: every field present, models resolved, the SOP text inlined (so
    # the snapshot never needs the SKILL.md again).
    sop = tmp_path / "SKILL.md"
    sop.write_text("---\nname: recipe-shopper\n---\n1. Search first.\n")
    path = _write(
        tmp_path / "s.yaml",
        _scenario(agent={"skill": str(sop), "notes": "No write tools."}),
    )
    snapshot = json.loads(load_scenario(path).model_dump_json())
    sop.unlink()
    view = snapshot_view(snapshot, fallback_name="penne")
    assert view.error is None and view.file == "scenario.json"
    assert view.agent_skill_name == "recipe-shopper"
    assert view.agent_skill_text == "1. Search first."
    assert view.agent_notes == "No write tools."
    # Every field is written, so "derived" means "equal to the default".
    assert view.user_instructions_derived and view.expected_behavior_derived
    assert view.models["user"] == "claude-haiku-4-5-20251001", "every model the run used"
    # A pre-v2 snapshot (user: null) still validates; a broken one falls back to the raw file.
    old = {**snapshot, "models": {**snapshot["models"], "user": None, "observer": None}}
    assert snapshot_view(old, fallback_name="penne").error is None
    broken = snapshot_view({"name": "penne", "title": "Old"}, fallback_name="x")
    assert broken.error is not None and (broken.name, broken.title) == ("penne", "Old")


def test_view_of_a_scenario_written_explicitly_with_its_defaults(tmp_path: Path) -> None:
    scenario = load_scenario(_write(tmp_path / "s.yaml", _scenario()))
    view = view_from_scenario(scenario, file="x", raw=_scenario(expected_behavior=["Use lookup."]))
    assert not view.expected_behavior_derived, "the file wrote it, even if equal"


# --- the skill --------------------------------------------------------------------------------


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    target = tmp_path / "skill"
    shutil.copytree(CHECKOUT_DIR, target)
    (target / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "defaults": {"agent": "claude-fable-5-1"},
                "run": {"repeat": 2},
                "scenarios": ["scenarios"],
                "overrides": [
                    {"match": {"name": "pen*"}, "models": {"judge": "anthropic:claude-sonnet-5-5"}},
                    {"match": {"category": "Pasta", "name": "nope"}, "models": {"user": "x"}},
                    {"match": {"category": "Pas*"}, "run": {"judge_votes": 1}},
                ],
            }
        )
    )
    return target


def test_role_rows_come_from_describe(skill_dir: Path) -> None:
    skill = load_skill(skill_dir)
    rows = {r["role"]: r for r in role_rows(skill, skill.describe())}
    assert list(rows) == ["planner", "planner-local", "agent", "user", "observer", "judge"]
    assert (rows["agent"]["spec"], rows["agent"]["source"]) == (
        "anthropic:claude-fable-5-1",
        "config.yaml defaults",
    )
    assert (rows["planner"]["spec"], rows["planner"]["source"]) == (
        "anthropic:claude-opus-5-5",
        "roles/planner.md",
    )
    assert rows["user"]["max_tokens"] == 512 and rows["judge"]["extra"] == {"votes": 3}
    assert rows["planner-local"]["extra"] == {"policy_max_tokens": 40}
    assert rows["planner-local"]["spec"] == "ollama:command-r7b"
    assert rows["agent"]["summary"] == "agent under test"
    assert overrides_json(skill) == [
        {"match": {"name": "pen*"}, "models": {"judge": "claude-sonnet-5-5"}, "run": {}},
        {"match": {"name": "nope", "category": "Pasta"}, "models": {"user": "x"}, "run": {}},
        {"match": {"category": "Pas*"}, "models": {}, "run": {"judge_votes": 1}},
    ]


def test_scenario_settings_follow_the_precedence_ladder(skill_dir: Path, tmp_path: Path) -> None:
    skill = load_skill(skill_dir)
    scenario = load_scenario(
        _write(
            tmp_path / "s.yaml",
            _scenario(category="Pasta", models={"user": "claude-sonnet-5-5"}, judge_votes=5),
        )
    )
    models, run = scenario_settings(skill, scenario, {"observer": "claude-haiku-4-5-20251001"})
    assert {r: (m["model"], m["source"]) for r, m in models.items()} == {
        "planner": ("claude-opus-5-5", "roles/planner.md"),
        "agent": ("claude-fable-5-1", "config.yaml defaults"),
        "judge": ("claude-sonnet-5-5", "config.yaml overrides[1] (name=pen*)"),
        "user": ("claude-sonnet-5-5", "scenario file"),
        "observer": ("claude-haiku-4-5-20251001", "command line"),
    }
    assert run["repeat"] == {"value": 2, "source": "config.yaml run"}
    assert run["judge_votes"] == {"value": 5, "source": "scenario file"}
    assert models["planner"]["file"] == str(skill.roles["planner"].path)
    # An ollama: planner is served by planner-local.md and its settings.
    local, _ = scenario_settings(skill, scenario, {"planner": "ollama:command-r7b"})
    assert local["planner"]["spec"] == "ollama:command-r7b"
    assert local["planner"]["file"] == str(skill.roles["planner-local"].path)
    assert local["planner"]["max_tokens"] == 1200


def test_skill_source_keeps_the_last_good_skill(skill_dir: Path) -> None:
    source = SkillSource(load_skill(skill_dir))
    skill, error = source.load()
    assert error is None and skill.config.run.repeat == 2
    config = skill_dir / "config.yaml"
    good = config.read_text()
    config.write_text(good.replace("repeat: 2", "repeat: 0"))
    skill, error = source.load()
    assert error is not None and "repeat" in error
    assert skill.config.run.repeat == 2, "the last skill that loaded"
    config.write_text(good.replace("repeat: 2", "repeat: 4"))
    skill, error = source.load()
    assert error is None and skill.config.run.repeat == 4


# --- the store --------------------------------------------------------------------------------


def test_store_sources_expand_like_the_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "a" / "one.yaml", _scenario(name="one"))
    _write(tmp_path / "a" / "deep" / "two.yml", _scenario(name="two"))
    (tmp_path / "a" / "skip.md").write_text("x")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "three.json").write_text(json.dumps(_scenario(name="three")))
    monkeypatch.setenv("UI_SCEN_DIR", str(tmp_path / "b"))
    monkeypatch.delenv("UI_UNSET_DIR", raising=False)
    entries = ["a", "${UI_SCEN_DIR}", "a/*.yaml", "missing", "${UI_UNSET_DIR}/x", "a/deep/*"]
    store = Store(scenario_sources=lambda: entries, runs_dir=tmp_path / "runs", cwd=tmp_path)
    files, notes = store.files()
    root = tmp_path.resolve()
    # A directory is not searched recursively (as `mcpsim suite`); a file named twice is one.
    assert files == [root / "a" / "one.yaml", root / "b" / "three.json", root / "a/deep/two.yml"]
    assert notes == [
        f"scenarios entry 'missing' ({tmp_path / 'missing'}): not found",
        "scenarios entry '${UI_UNSET_DIR}/x': skipped: UI_UNSET_DIR is not set",
    ]
    index = store.scan()
    assert sorted(index.views) == ["one", "three", "two"]
    assert sorted(index.scenarios) == ["one", "three", "two"]
    assert index.views["one"].file == "a/one.yaml"


def test_store_duplicate_names_keep_the_first(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "x.yaml", _scenario(name="same"))
    _write(tmp_path / "a" / "y.yaml", _scenario(name="same", title="Second"))
    store = Store(scenario_sources=lambda: ["a"], runs_dir=tmp_path / "runs", cwd=tmp_path)
    index = store.scan()
    assert list(index.views) == ["same"] and index.views["same"].title == "Same"
    assert any("also used by" in w for w in index.warnings)


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
