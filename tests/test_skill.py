"""The simulate skill (mcpsim.skill): loading and validating SKILL.md, config.yaml and the role
files, the model and run-setting precedence, scenario sources and selection, and that the
prompts and settings in the role files are what the calls actually use."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from mcpsim.agent import build_agent_system_prompt
from mcpsim.judge import VERDICT_TOOL, judge
from mcpsim.plan import Path as PlanPath
from mcpsim.plan import Step
from mcpsim.scenario import DEFAULT_CATEGORY, DEFAULT_MODELS, MODEL_ROLES, parse_scenario
from mcpsim.skill import (
    BUILTIN,
    CHECKOUT_DIR,
    CLI_LAYER,
    ROLE_NAMES,
    ROLE_SPECS,
    SCENARIO_LAYER,
    SKILL_ENV,
    Skill,
    SkillError,
    canonical_spec,
    expand_env,
    expand_source,
    load_entries,
    load_role,
    load_skill,
    select_entries,
    skill_dir,
)
from mcpsim.transcript import EndEvent, FinalResultEvent, SystemEvent, Transcript
from tests.fake_llm import ScriptedLLM, structured_response


@pytest.fixture(autouse=True)
def _no_skill_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SKILL_ENV, raising=False)
    monkeypatch.delenv("PANTRY_GATEWAY_SCENARIOS", raising=False)


@pytest.fixture
def skill_copy(tmp_path: Path) -> Path:
    """A writable copy of the bundled skill."""
    target = tmp_path / "simulate"
    shutil.copytree(CHECKOUT_DIR, target)
    return target


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def write_config(skill: Path, config: dict[str, Any]) -> None:
    (skill / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def scenario(name: str = "s", category: str = "Cat", **fields: Any) -> Any:
    data: dict[str, Any] = {
        "name": name,
        "category": category,
        "role": "A shopper.",
        "goal": "Find penne.",
        "expected_outcome": {"text": "Penne."},
        "server": {"stdio": {"command": "true"}},
        **fields,
    }
    return parse_scenario(data)


# --- the bundled skill ------------------------------------------------------------------------


def test_the_bundled_skill_loads_with_every_role_on_the_anthropic_api() -> None:
    skill = load_skill()
    assert skill.path == CHECKOUT_DIR and skill.name == "simulate"
    assert "simulated users" in skill.description
    assert tuple(skill.roles) == ROLE_NAMES
    for name, role in skill.roles.items():
        assert set(role.prompts) == set(ROLE_SPECS[name].prompts)
        if name != "planner-local":
            assert role.provider == "anthropic", name
            assert role.temperature is None, "Opus 5.5 / Sonnet 5.5 reject temperature"
    assert skill.roles["planner-local"].provider == "ollama"
    # Every model role resolves to the Anthropic defaults part 1 set.
    resolved = skill.resolve_models("any", "any")
    assert {r: s.value for r, s in resolved.items()} == DEFAULT_MODELS
    # The bundled max_tokens are the values the code used before the move.
    tokens = {name: role.max_tokens for name, role in skill.roles.items()}
    assert tokens == {
        "planner": 8192,
        "planner-local": 1200,
        "agent": 4096,
        "user": 512,
        "observer": 1024,
        "judge": 4096,
    }
    assert skill.roles["judge"].extra == {"votes": 3}
    assert skill.roles["planner-local"].extra == {"policy_max_tokens": 40}


def test_the_bundled_config_runs_defaults_and_sources(tmp_path: Path) -> None:
    skill = load_skill()
    run = skill.resolve_run("x", "y")
    assert {k: s.value for k, s in run.items()} == {
        "repeat": 1,
        "modes": ["guided", "free"],
        "judge_votes": 3,
        "concurrency": 2,
    }
    assert skill.config.scenarios[0] == "scenarios/pantry"
    assert "PANTRY_GATEWAY_SCENARIOS" in skill.config.scenarios[1]
    assert skill.runs_dir(tmp_path) == tmp_path / "runs" / "simulate"
    repo = CHECKOUT_DIR.parent.parent
    sources = skill.sources(repo)
    assert sources[0].path == str(repo / "scenarios" / "pantry")
    first_two = [f.name for f in sources[0].files][:2]
    assert first_two == ["cheapest-penne.yaml", "label-submission.yaml"]


# --- skill_dir resolution ---------------------------------------------------------------------


def test_skill_dir_prefers_the_flag_then_the_environment_then_the_package(
    skill_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert skill_dir() == CHECKOUT_DIR
    other = tmp_path / "other"
    monkeypatch.setenv(SKILL_ENV, str(skill_copy))
    assert skill_dir() == skill_copy.resolve()
    assert skill_dir(other) == other.resolve()
    assert load_skill().path == skill_copy.resolve()


def test_a_missing_skill_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SkillError, match="skill directory not found"):
        load_skill(tmp_path / "nope")


def test_an_edited_role_file_is_reloaded(skill_copy: Path) -> None:
    first = load_skill(skill_copy)
    edit(skill_copy / "roles" / "user.md", "max_tokens: 512", "max_tokens: 600")
    second = load_skill(skill_copy)
    assert first.roles["user"].max_tokens == 512 and second.roles["user"].max_tokens == 600


# --- validation: role files -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "old", "new", "message"),
    [
        ("judge", "votes: 3", "votes: 3\nsamples: 2", "unknown frontmatter key(s) samples"),
        ("agent", "provider: anthropic\n", "", "frontmatter needs provider"),
        ("agent", "role: agent", "role: user", "frontmatter says role 'user', the file is 'agent'"),
        ("agent", "provider: anthropic", "provider: openai", "provider must be one of"),
        ("agent", "max_tokens: 4096", "max_tokens: 0", "max_tokens must be a positive integer"),
        ("judge", "votes: 3", "votes: three", "votes must be a positive integer"),
        ("user", "max_tokens: 512", "max_tokens: 512\ntemperature: 3", "between 0 and 2"),
        ("observer", "{% prompt user %}", "{% prompt users %}", "unknown prompt(s) users"),
        (
            "observer",
            "{% prompt user %}\n{{ watched }}",
            "",
            "missing prompt(s) user",
        ),
        # A comment line inside a prompt is fine: the opening keeps its text below it.
        ("user", "{% prompt opening %}\n", "{% prompt opening %}\n{# nothing #}\n", None),
        (
            "agent",
            "## Goal\n{{ goal }}",
            "## Goal\n{{ goal }} {{ budget }}",
            "unknown placeholder 'budget' in prompt system; it accepts: role, goal",
        ),
        (
            "observer",
            "{{ conditions }}",
            "(conditions left out)",
            "the prompt must insert {{ conditions }}",
        ),
        (
            "judge",
            "{{ transcript }}",
            "{% if transcript %}see the transcript{% endif %}",
            "the prompt must insert {{ transcript }}",
        ),
        ("planner", "CATALOG:\n{{ catalog }}", "CATALOG:\n{{ catalog", "'{{' is never closed"),
        ("planner-local", "provider: ollama", "provider: anthropic", "provider must be ollama"),
    ],
)
def test_invalid_role_files_name_the_file_and_the_problem(
    skill_copy: Path, role: str, old: str, new: str, message: str | None
) -> None:
    path = skill_copy / "roles" / f"{role}.md"
    edit(path, old, new)
    if message is None:
        load_skill(skill_copy)
        return
    with pytest.raises(SkillError) as info:
        load_skill(skill_copy)
    assert str(path) in str(info.value)
    assert message in str(info.value)


def test_an_unknown_placeholder_error_names_the_line(skill_copy: Path) -> None:
    path = skill_copy / "roles" / "agent.md"
    edit(path, "## Goal\n{{ goal }}", "## Goal\n{{ goall }}")
    lines = path.read_text(encoding="utf-8").splitlines()
    line = next(i for i, text in enumerate(lines, start=1) if "{{ goall }}" in text)
    with pytest.raises(SkillError, match=rf"agent.md, line {line}: unknown placeholder 'goall'"):
        load_skill(skill_copy)


def test_an_empty_prompt_is_refused(skill_copy: Path) -> None:
    edit(
        skill_copy / "roles" / "user.md",
        "{% prompt silent_agent %}\n(the assistant sent an empty message)",
        "{% prompt silent_agent %}\n{# left empty #}",
    )
    with pytest.raises(SkillError, match="prompt silent_agent"):
        load_skill(skill_copy)


def test_missing_and_unknown_role_files(skill_copy: Path) -> None:
    (skill_copy / "roles" / "judge.md").rename(skill_copy / "roles" / "critic.md")
    with pytest.raises(SkillError, match=r"unknown role file\(s\) critic.md"):
        load_skill(skill_copy)
    (skill_copy / "roles" / "critic.md").unlink()
    with pytest.raises(SkillError, match=r"missing role file\(s\) judge.md"):
        load_skill(skill_copy)


def test_a_role_file_without_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "agent.md"
    path.write_text("{% prompt system %}\nhi", encoding="utf-8")
    with pytest.raises(SkillError, match="starts with a '---' YAML frontmatter block"):
        load_role(path)


def test_skill_md_needs_a_name_and_a_description(skill_copy: Path) -> None:
    edit(skill_copy / "SKILL.md", "name: simulate\n", "")
    with pytest.raises(SkillError, match="frontmatter needs a name"):
        load_skill(skill_copy)


# --- validation: config.yaml ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"defaults": {"critic": "claude-opus-5-5"}}, "unknown role(s) critic"),
        ({"defaults": {"judge": ""}}, "a model is 'provider:model'"),
        ({"run": {"modes": []}}, "modes must list at least one"),
        ({"run": {"modes": ["guided", "guided"]}}, "modes lists a mode twice"),
        ({"run": {"modes": ["replay"]}}, "run.modes.0"),
        ({"run": {"repeat": 0}}, "run.repeat"),
        ({"overrides": [{"match": {}, "run": {"repeat": 2}}]}, "match needs name and/or category"),
        ({"overrides": [{"match": {"name": "x"}, "models": {"critic": "x"}}]}, "critic"),
        ({"scenarios": ["  "]}, "scenarios[0] is blank"),
        ({"retries": 3}, "retries"),
        ({"roles_dir": "nowhere"}, "roles_dir"),
    ],
)
def test_invalid_config_is_refused(skill_copy: Path, config: dict[str, Any], message: str) -> None:
    write_config(skill_copy, config)
    with pytest.raises(SkillError) as info:
        load_skill(skill_copy)
    assert message in str(info.value)


# --- precedence -------------------------------------------------------------------------------


LAYERS = ("builtin", "frontmatter", "defaults", "override1", "override2", "scenario", "cli")


def layered_skill(skill_copy: Path, layers: set[str]) -> Skill:
    """A skill whose judge model is set at the given layers (each to its own model)."""
    if "frontmatter" in layers:
        edit(skill_copy / "roles" / "judge.md", "model: claude-opus-5-5", "model: m-frontmatter")
    config: dict[str, Any] = {"overrides": []}
    if "defaults" in layers:
        config["defaults"] = {"judge": "anthropic:m-defaults"}
    if "override1" in layers:
        config["overrides"].append({"match": {"category": "Cat*"}, "models": {"judge": "m-o1"}})
    if "override2" in layers:
        config["overrides"].append(
            {"match": {"name": "s", "category": "Cat"}, "models": {"judge": "m-o2"}}
        )
    write_config(skill_copy, config)
    return load_skill(skill_copy)


@pytest.mark.parametrize(
    ("layers", "expected", "source"),
    [
        (set(), "claude-opus-5-5", "roles/judge.md"),
        ({"frontmatter"}, "m-frontmatter", "roles/judge.md"),
        ({"frontmatter", "defaults"}, "m-defaults", "config.yaml defaults"),
        ({"defaults", "override1"}, "m-o1", "config.yaml overrides[1] (category=Cat*)"),
        ({"override1", "override2"}, "m-o2", "config.yaml overrides[2] (name=s, category=Cat)"),
        ({"frontmatter", "defaults", "override1", "scenario"}, "m-scenario", SCENARIO_LAYER),
        ({"defaults", "override2", "scenario", "cli"}, "m-cli", CLI_LAYER),
        ({"frontmatter", "cli"}, "m-cli", CLI_LAYER),
    ],
)
def test_model_precedence_matrix(
    skill_copy: Path, layers: set[str], expected: str, source: str
) -> None:
    skill = layered_skill(skill_copy, layers)
    models = {"judge": "m-scenario"} if "scenario" in layers else {}
    s = scenario(models=models) if models else scenario()
    cli = {"judge": "anthropic:m-cli"} if "cli" in layers else None
    resolved = skill.resolve(s, model_overrides=cli).models["judge"]
    assert (resolved.value, resolved.source) == (expected, source)


def test_the_builtin_default_shows_when_no_layer_names_a_model(skill_copy: Path) -> None:
    write_config(skill_copy, {})
    loaded = load_skill(skill_copy)
    # A skill without the observer's role file: the ladder falls back to the built-in default.
    roles = {k: v for k, v in loaded.roles.items() if k != "observer"}
    skill = dataclasses.replace(loaded, roles=roles)
    resolved = skill.resolve_models("s", "Cat")
    assert resolved["observer"].source == BUILTIN
    assert resolved["observer"].value == DEFAULT_MODELS["observer"]


def test_an_override_that_does_not_match_is_ignored(skill_copy: Path) -> None:
    write_config(
        skill_copy,
        {"overrides": [{"match": {"name": "other", "category": "Cat"}, "models": {"user": "u"}}]},
    )
    skill = load_skill(skill_copy)
    assert skill.resolve(scenario()).models["user"].source == "roles/user.md"


def test_run_settings_precedence(skill_copy: Path) -> None:
    edit(skill_copy / "roles" / "judge.md", "votes: 3", "votes: 4")
    write_config(skill_copy, {})
    skill = load_skill(skill_copy)
    plain = scenario()
    run = skill.resolve(plain).run
    assert (run["judge_votes"].value, run["judge_votes"].source) == (4, "roles/judge.md")
    assert run["repeat"].value == 3 and run["repeat"].source == BUILTIN
    write_config(
        skill_copy,
        {
            "run": {"repeat": 2, "judge_votes": 5, "modes": ["guided"]},
            "overrides": [
                {"match": {"category": "Cat"}, "run": {"repeat": 4, "concurrency": 1}},
                {"match": {"name": "nomatch"}, "run": {"repeat": 9}},
            ],
        },
    )
    skill = load_skill(skill_copy)
    run = skill.resolve(plain).run
    assert {k: s.value for k, s in run.items()} == {
        "repeat": 4,
        "modes": ["guided"],
        "judge_votes": 5,
        "concurrency": 1,
    }
    assert run["repeat"].source == "config.yaml overrides[1] (category=Cat)"
    assert run["judge_votes"].source == "config.yaml run"
    # The scenario file's own values win over the config; the command line over everything.
    explicit = scenario(repeat=7, judge_votes=1)
    run = skill.resolve(explicit, run_overrides={"repeat": 2, "modes": ["free"]}).run
    assert (run["repeat"].value, run["repeat"].source) == (2, CLI_LAYER)
    assert (run["judge_votes"].value, run["judge_votes"].source) == (1, SCENARIO_LAYER)
    assert run["modes"].value == ["free"] and run["concurrency"].value == 1


def test_apply_returns_the_scenario_as_it_will_run(skill_copy: Path) -> None:
    write_config(skill_copy, {"run": {"repeat": 2, "concurrency": 1}})
    skill = load_skill(skill_copy)
    s = scenario(models={"user": "anthropic:claude-haiku-4-5-20251001"}, judge_votes=1)
    applied, resolved = skill.apply(s, model_overrides={"agent": "anthropic:claude-opus-5"})
    assert applied.models.user == "claude-haiku-4-5-20251001", "canonical spelling"
    assert applied.models.agent == "claude-opus-5"
    assert (applied.repeat, applied.judge_votes, applied.concurrency) == (2, 1, 1)
    assert resolved.models["agent"].source == CLI_LAYER
    # A resolved scenario round-trips through scenario.json unchanged.
    again = type(applied).model_validate_json(applied.model_dump_json())
    assert again.models == applied.models and again.repeat == 2
    with pytest.raises(ValueError, match="unknown model role"):
        skill.apply(s, model_overrides={"critic": "x"})


def test_canonical_spec_spells_one_model_one_way() -> None:
    assert canonical_spec("anthropic:claude-opus-5-5") == "claude-opus-5-5"
    assert canonical_spec("claude-opus-5-5") == "claude-opus-5-5"
    assert canonical_spec("ollama:llama3.2:3b") == "ollama:llama3.2:3b"
    assert set(MODEL_ROLES) == set(DEFAULT_MODELS)


def test_a_temperature_the_resolved_model_rejects_is_refused(skill_copy: Path) -> None:
    edit(skill_copy / "roles" / "user.md", "max_tokens: 512", "max_tokens: 512\ntemperature: 0.3")
    skill = load_skill(skill_copy)  # haiku accepts temperature
    applied, _ = skill.apply(scenario())
    assert applied.models.user == "claude-haiku-4-5-20251001"
    with pytest.raises(SkillError, match="claude-sonnet-5-5, which rejects it"):
        skill.apply(scenario(), model_overrides={"user": "claude-sonnet-5-5"})
    # A local model takes any temperature.
    skill.apply(scenario(), model_overrides={"user": "ollama:llama3.2:3b"})


# --- the roles drive the calls ----------------------------------------------------------------


def two_step_path() -> PlanPath:
    return PlanPath(
        id="p", kind="happy", title="t", steps=[Step(intent="Look it up", tool="lookup")]
    )


def test_an_edited_prompt_changes_what_the_agent_is_told(skill_copy: Path) -> None:
    edit(skill_copy / "roles" / "agent.md", "## Goal\n{{ goal }}", "## What they want\n{{ goal }}")
    s = scenario()
    bundled = build_agent_system_prompt(s, two_step_path(), "free")
    edited = build_agent_system_prompt(s, two_step_path(), "free", skill=skill_copy)
    assert "## Goal\nFind penne." in bundled
    assert "## What they want\nFind penne." in edited and "## Goal" not in edited
    assert edited.replace("## What they want", "## Goal") == bundled


async def test_the_judge_call_uses_the_role_files_settings_and_prompts(skill_copy: Path) -> None:
    path = skill_copy / "roles" / "judge.md"
    edit(path, "model: claude-opus-5-5", "model: claude-haiku-4-5-20251001")
    edit(path, "max_tokens: 4096", "max_tokens: 1234\ntemperature: 0.1")
    edit(path, "Rules:\n", "Rules (edited):\n")
    write_config(skill_copy, {})
    skill = load_skill(skill_copy)
    s, resolved = skill.apply(scenario())
    assert resolved.models["judge"].source == "roles/judge.md"
    transcript = Transcript(scenario="s", path_id="p", mode="free", index=0)
    transcript.add(SystemEvent(scenario="s", path_id="p", index=0, mode="free"))
    transcript.add(FinalResultEvent(parsed={"x": 1}, raw='{"x": 1}'))
    transcript.add(EndEvent(outcome="completed", reason="final answer delivered"))
    transcript.outcome, transcript.reason = "completed", "final answer delivered"
    vote = {
        "expected_behavior": [],
        "goal_achieved": {"passed": True, "evidence": "[2] final"},
        "honesty": {"passed": True, "evidence": "[2] final"},
        "passed": True,
        "score": 1.0,
    }
    llm = ScriptedLLM([structured_response(VERDICT_TOOL, vote)])
    verdict = await judge(s, two_step_path(), transcript, llm, votes=1, skill=skill)
    assert verdict.passed, verdict.failure_reasons
    call = llm.calls[0]
    assert call["max_tokens"] == 1234 and call["temperature"] == 0.1
    assert call["model"] == "claude-haiku-4-5-20251001"
    assert "Rules (edited):\n1. Judge what happened" in call["system"]


# --- scenario sources and selection -----------------------------------------------------------


def test_env_references_expand_with_defaults_or_skip() -> None:
    env = {"A": "/x"}
    assert expand_env("$A/s", env) == ("/x/s", None)
    assert expand_env("${A}/s", env) == ("/x/s", None)
    assert expand_env("${B:-fallback}/s", env) == ("fallback/s", None)
    assert expand_env("${A:-fallback}", env) == ("/x", None)
    assert expand_env("$B/s", env) == (None, "B is not set")


def write_scenario(folder: Path, name: str, category: str, **extra: Any) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "category": category,
        "role": "r",
        "goal": "g",
        "expected_outcome": {"text": "t"},
        "server": {"stdio": {"command": "true"}},
        **extra,
    }
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_sources_are_directories_globs_or_files(tmp_path: Path) -> None:
    write_scenario(tmp_path / "a", "one", "X")
    write_scenario(tmp_path / "a", "two", "Y")
    (tmp_path / "a" / "notes.txt").write_text("x", encoding="utf-8")
    write_scenario(tmp_path / "b" / "deep", "three", "X")
    folder = expand_source("a", base=tmp_path)
    assert [f.name for f in folder.files] == ["one.yaml", "two.yaml"] and folder.note is None
    globbed = expand_source("b/**/*.yaml", base=tmp_path)
    assert [f.name for f in globbed.files] == ["three.yaml"]
    single = expand_source(str(tmp_path / "a" / "two.yaml"))
    assert [f.name for f in single.files] == ["two.yaml"]
    assert expand_source("missing", base=tmp_path).note == "not found"
    unset = expand_source("$NOPE_NOT_SET/x", base=tmp_path)
    assert unset.note == "skipped: NOPE_NOT_SET is not set"
    assert expand_source("b/*.json", base=tmp_path).note == "no scenario file matches"


def test_entries_load_every_configured_scenario_and_keep_load_errors(
    skill_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_scenario(tmp_path / "s1", "alpha", "Lookup")
    write_scenario(tmp_path / "s1", "beta", "Planning")
    write_scenario(tmp_path / "s2", "gamma", "Lookup", agent={"skill": "env:NOT_SET_SKILL"})
    write_scenario(tmp_path / "s2", "alpha-dup", "Lookup")
    dup = tmp_path / "s2" / "alpha-dup.yaml"
    dup.write_text(dup.read_text(encoding="utf-8").replace("alpha-dup", "alpha"), "utf-8")
    monkeypatch.setenv("S2_DIR", str(tmp_path / "s2"))
    write_config(skill_copy, {"scenarios": ["s1", "$S2_DIR", "${UNSET_DIR}"]})
    skill = load_skill(skill_copy)
    entries = skill.entries(tmp_path)
    # A file that does not load keeps the name and category it states, so filters still
    # select it (and the suite reports it) instead of dropping it.
    assert [(e.name, e.category, e.error is None) for e in entries] == [
        ("alpha", "Lookup", True),
        ("beta", "Planning", True),
        ("alpha", "Lookup", False),
        ("gamma", "Lookup", False),
    ]
    assert "also defined by" in (entries[2].error or "")
    assert "NOT_SET_SKILL is not set" in (entries[3].error or "")
    assert [s.note for s in skill.sources(tmp_path)][2] == "skipped: UNSET_DIR is not set"

    def names(**kwargs: Any) -> list[str]:
        return [e.name for e in select_entries(entries, **kwargs)]

    assert names(names=["al*"]) == ["alpha", "alpha"]
    assert names(categories=["Look*"]) == ["alpha", "alpha", "gamma"]
    assert names(categories=["Plan*"]) == ["beta"]
    assert names(names=["beta", "gam*"]) == ["beta", "gamma"]
    assert names(names=["*a"], categories=["Planning"]) == ["beta"]
    assert names() == ["alpha", "beta", "alpha", "gamma"]


def test_broken_scenarios_are_selected_by_the_name_and_category_their_file_states(
    tmp_path: Path,
) -> None:
    """A gateway copy whose file stem differs from its name, a broken file without a category
    and one that does not even parse: name and category filters keep each one they could
    match, so a filtered suite reports it and exits 1 instead of going green."""
    folder = tmp_path / "s"
    write_scenario(folder, "ok-gateway", "Recipe planning")
    broken = write_scenario(folder, "cheapest-penne-gateway", "Product lookup", repeat=0)
    broken.rename(folder / "cheapest-penne.yaml")
    no_category = folder / "uncategorized.yaml"
    no_category.write_text(yaml.safe_dump({"name": "plain-broken", "role": "r"}), "utf-8")
    (folder / "garbled.yaml").write_text("name: [unclosed\n", "utf-8")
    entries = load_entries([(f, "s") for f in sorted(folder.iterdir())])
    by_file = {e.file.name: e for e in entries}
    assert (by_file["cheapest-penne.yaml"].name, by_file["cheapest-penne.yaml"].category) == (
        "cheapest-penne-gateway",
        "Product lookup",
    )
    assert by_file["cheapest-penne.yaml"].title == "Cheapest penne gateway"
    assert (by_file["uncategorized.yaml"].name, by_file["uncategorized.yaml"].category) == (
        "plain-broken",
        DEFAULT_CATEGORY,
    )
    assert (by_file["garbled.yaml"].name, by_file["garbled.yaml"].category) == ("garbled", "")
    assert all(e.error for name, e in by_file.items() if name != "ok-gateway.yaml")

    def names(**kwargs: Any) -> list[str]:
        return sorted(e.name for e in select_entries(entries, **kwargs))

    assert names(names=["*-gateway"]) == ["cheapest-penne-gateway", "ok-gateway"]
    # The category a broken file states is matched; one that cannot be parsed has an
    # unknown category and is kept by every category filter.
    assert names(categories=["Product*"]) == ["cheapest-penne-gateway", "garbled"]
    assert names(categories=["Recipe*"]) == ["garbled", "ok-gateway"]
    assert names(categories=[DEFAULT_CATEGORY]) == ["garbled", "plain-broken"]
    assert names(names=["*-gateway"], categories=["Recipe*"]) == ["ok-gateway"]


def test_describe_reports_roles_sources_and_the_scenarios_resolution(skill_copy: Path) -> None:
    write_config(
        skill_copy,
        {
            "defaults": {"agent": "claude-sonnet-5-5"},
            "run": {"repeat": 1},
            "runs_dir": "out",
            "overrides": [{"match": {"category": "Cat"}, "models": {"judge": "claude-opus-5"}}],
        },
    )
    skill = load_skill(skill_copy)
    info = skill.describe()
    assert info["skill"]["name"] == "simulate"
    assert info["roles"]["agent"]["model"] == "claude-sonnet-5-5"
    assert info["roles"]["agent"]["model_source"] == "config.yaml defaults"
    assert info["roles"]["judge"]["model"] == "claude-opus-5-5"
    assert info["roles"]["judge"]["votes"] == 3
    assert "model" not in info["roles"]["planner-local"]
    assert info["roles"]["planner"]["prompts"]["system"]["requires"] == ["catalog"]
    assert info["run"]["repeat"] == {"value": 1, "source": "config.yaml run"}
    assert info["runs_dir"].endswith("/out")
    scoped = skill.describe(scenario=scenario())
    assert scoped["scenario"] == "s"
    assert scoped["roles"]["judge"]["model"] == "claude-opus-5"
    assert scoped["roles"]["judge"]["model_source"].startswith("config.yaml overrides[1]")


def test_a_custom_roles_dir_is_used_and_watched(skill_copy: Path) -> None:
    (skill_copy / "roles").rename(skill_copy / "prompts")
    write_config(skill_copy, {"roles_dir": "prompts"})
    first = load_skill(skill_copy)
    assert first.roles["judge"].path == skill_copy / "prompts" / "judge.md"
    edit(skill_copy / "prompts" / "judge.md", "votes: 3", "votes: 5")
    assert load_skill(skill_copy).roles["judge"].extra == {"votes": 5}
