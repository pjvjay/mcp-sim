"""Model addressing: ``provider:model`` specs, the ``Models`` fields, the provider factory, the
runner's ``make_llm_for`` / ``apply_model_overrides`` and the CLI's ``--models`` flag."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path as FsPath
from typing import Any

import pytest

from mcpsim import runner
from mcpsim.agent import run_path
from mcpsim.cli import EXIT_OK, RUNNER_MODULE, main, model_kwargs, parse_model_overrides
from mcpsim.judge import check_judge_model
from mcpsim.llm import (
    API_KEY_ENV,
    LOCAL_COST_NOTE,
    AnthropicLLM,
    LLMResponse,
    OllamaLLM,
    RoutingLLM,
    clear_llm_cache,
    is_local_model,
    make_llm,
    parse_model_spec,
    rate_for,
)
from mcpsim.plan import Path, Step
from mcpsim.report import Report
from mcpsim.scenario import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PLANNER_MODEL,
    MODEL_ROLES,
    Models,
    Scenario,
    ScenarioError,
    parse_scenario,
)
from tests.conftest import open_session
from tests.fake_llm import ScriptedLLM, text_response


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_llm_cache()


@pytest.fixture
def scenario(scenario_data: dict[str, Any]) -> Scenario:
    return parse_scenario(scenario_data, source="fixture")


# --- parse_model_spec ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("ollama:command-r7b", ("ollama", "command-r7b")),
        ("ollama:llama3.2:3b", ("ollama", "llama3.2:3b")),
        ("anthropic:claude-opus-5-5", ("anthropic", "claude-opus-5-5")),
        ("claude-sonnet-5-5", ("anthropic", "claude-sonnet-5-5")),
        ("claude-haiku-4-5-20251001", ("anthropic", "claude-haiku-4-5-20251001")),
        ("  Ollama:qwen2.5:7b ", ("ollama", "qwen2.5:7b")),
        ("llama3.2:3b", ("anthropic", "llama3.2:3b")),  # unknown prefix is not a provider
    ],
)
def test_parse_model_spec(spec: str, expected: tuple[str, str]) -> None:
    assert parse_model_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "   ", "ollama:", "anthropic:  "])
def test_parse_model_spec_rejects_empty_name(spec: str) -> None:
    with pytest.raises(ValueError, match="no model name"):
        parse_model_spec(spec)


def test_is_local_model_and_rates() -> None:
    assert is_local_model("ollama:command-r7b") is True
    assert is_local_model("claude-sonnet-5-5") is False
    assert rate_for("anthropic:claude-sonnet-5-5") == (2.0, 10.0)
    assert rate_for("ollama:claude-sonnet-5-5") is None  # local, whatever it is called


# --- Models fields ---------------------------------------------------------------------------


def test_models_defaults_and_new_fields() -> None:
    m = Models()
    assert (m.agent, m.planner, m.judge) == (
        DEFAULT_AGENT_MODEL,
        DEFAULT_PLANNER_MODEL,
        DEFAULT_JUDGE_MODEL,
    )
    # Every role defaults to an Anthropic API model; the simulated user to Haiku.
    assert (m.user, m.observer) == ("claude-haiku-4-5-20251001", "claude-sonnet-5-5")
    assert m.allow_same_judge is False
    assert m.user_model == m.for_role("user") == "claude-haiku-4-5-20251001"
    assert m.observer_model == m.for_role("observer") == "claude-sonnet-5-5"
    assert {r: m.for_role(r) for r in MODEL_ROLES} == {
        "planner": "claude-opus-5-5",
        "agent": "claude-sonnet-5-5",
        "user": "claude-haiku-4-5-20251001",
        "observer": "claude-sonnet-5-5",
        "judge": "claude-opus-5-5",
    }
    assert not any(is_local_model(m.for_role(r)) for r in MODEL_ROLES)
    assert MODEL_ROLES == ("planner", "agent", "judge", "user", "observer")
    with_user = Models(user="ollama:llama3.2:3b", allow_same_judge=True)
    assert with_user.user_model == "ollama:llama3.2:3b"
    assert [with_user.for_role(r) for r in MODEL_ROLES] == [
        DEFAULT_PLANNER_MODEL,
        DEFAULT_AGENT_MODEL,
        DEFAULT_JUDGE_MODEL,
        "ollama:llama3.2:3b",
        "claude-sonnet-5-5",  # observers keep their own default, not the user's
    ]
    # A v1 scenario.json holds null for user / observer: that is the default, not an error.
    legacy = Models.model_validate({"user": None, "observer": None})
    assert (legacy.user, legacy.observer) == ("claude-haiku-4-5-20251001", "claude-sonnet-5-5")
    with pytest.raises(KeyError, match="unknown model role"):
        with_user.for_role("critic")


def test_models_from_scenario_file(scenario_data: dict[str, Any]) -> None:
    scenario_data["models"] = {
        "planner": "ollama:command-r7b",
        "agent": "ollama:command-r7b",
        "user": "ollama:llama3.2:3b",
        "judge": "ollama:command-r7b",
        "allow_same_judge": True,
    }
    s = parse_scenario(scenario_data)
    assert s.models.user == "ollama:llama3.2:3b" and s.models.allow_same_judge is True
    check_judge_model(s)  # same judge and agent, allowed by the real field

    scenario_data["models"]["allow_same_judge"] = False
    with pytest.raises(ValueError, match="same as the agent model"):
        check_judge_model(parse_scenario(scenario_data))

    scenario_data["models"] = {"user": ""}
    with pytest.raises(ScenarioError, match="models.user"):
        parse_scenario(scenario_data)
    scenario_data["models"] = {"critic": "x"}
    with pytest.raises(ScenarioError, match="models.critic"):
        parse_scenario(scenario_data)


# --- the simulated user's model (agent.py reads the real field) -------------------------------


async def test_user_sim_uses_models_user(scenario: Scenario) -> None:
    scenario = scenario.model_copy(
        update={"models": Models(agent="ollama:command-r7b", user="ollama:llama3.2:3b")}
    )
    final = "Done.\n\n```json final_result\n" + json.dumps({"slug": "penne"}) + "\n```"
    llm = ScriptedLLM([text_response("Hi, price of penne please."), text_response(final)])
    path = Path(
        id="happy",
        kind="happy",
        title="t",
        rationale="r",
        steps=[Step(intent="look", tool="lookup", arguments_sketch={"slug": "penne"})],
    )
    async with open_session() as session:
        t = await run_path(scenario, path, "free", 0, session, llm)
    assert t.events[0].kind == "system"
    assert t.events[0].models == {"agent": "ollama:command-r7b", "user": "ollama:llama3.2:3b"}  # type: ignore[union-attr]
    assert llm.calls[0]["model"] == "ollama:llama3.2:3b"  # the opening message
    assert llm.calls[1]["model"] == "ollama:command-r7b"  # the agent turn
    assert t.outcome == "completed"


# --- make_llm / RoutingLLM -------------------------------------------------------------------


def test_make_llm_caches_one_client_per_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    local = make_llm("ollama")
    assert isinstance(local, OllamaLLM)
    assert make_llm("OLLAMA") is local
    with pytest.raises(RuntimeError, match=f"{API_KEY_ENV} is not set.*dry-run.*ollama:<model>"):
        make_llm("anthropic", purpose="planner")
    monkeypatch.setenv(API_KEY_ENV, "not-a-real-key")
    hosted = make_llm("anthropic")
    assert isinstance(hosted, AnthropicLLM) and make_llm("anthropic") is hosted
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match="needed for the judge"):
        make_llm("anthropic", purpose="judge")  # the cache never hides a missing key
    with pytest.raises(ValueError, match="unknown model provider 'openai'"):
        make_llm("openai")
    clear_llm_cache()
    assert make_llm("ollama") is not local


async def test_routing_llm_dispatches_on_the_model_spec() -> None:
    seen: list[str] = []

    class Stub:
        def __init__(self, provider: str) -> None:
            self.provider = provider

        async def complete(self, *, model: str, **kwargs: Any) -> LLMResponse:
            seen.append(f"{self.provider}<-{model}")
            return LLMResponse(content=[], stop_reason="end_turn")

    clients = {"ollama": Stub("ollama"), "anthropic": Stub("anthropic")}
    router = RoutingLLM(lambda provider: clients[provider])
    await router.complete(model="ollama:llama3.2:3b", system="", messages=[])
    await router.complete(model="claude-sonnet-5-5", system="", messages=[])
    assert seen == ["ollama<-ollama:llama3.2:3b", "anthropic<-claude-sonnet-5-5"]


# --- runner.make_llm_for / apply_model_overrides ------------------------------------------------


def test_make_llm_for_picks_the_provider_per_role(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    local = scenario.model_copy(
        update={"models": Models(planner="ollama:command-r7b", agent="ollama:command-r7b")}
    )
    assert isinstance(runner.make_llm_for(local, "planner"), OllamaLLM)
    # The simulated user keeps its Anthropic default, so the agent's client needs the key too.
    with pytest.raises(RuntimeError, match="needed for the simulated user"):
        runner.make_llm_for(local, "agent")
    with pytest.raises(RuntimeError, match="needed for the user"):
        runner.make_llm_for(local, "user")
    with pytest.raises(RuntimeError, match="needed for the judge"):
        runner.make_llm_for(local, "judge")  # still the default Anthropic judge, no key
    all_local = scenario.model_copy(
        update={"models": Models(agent="ollama:command-r7b", user="ollama:llama3.2:3b")}
    )
    assert isinstance(runner.make_llm_for(all_local, "agent"), OllamaLLM)
    assert isinstance(runner.make_llm_for(all_local, "user"), OllamaLLM)

    mixed = scenario.model_copy(
        update={"models": Models(agent="ollama:command-r7b", user="claude-haiku-4-5-20251001")}
    )
    with pytest.raises(RuntimeError, match="needed for the simulated user"):
        runner.make_llm_for(mixed, "agent")
    monkeypatch.setenv(API_KEY_ENV, "not-a-real-key")
    assert isinstance(runner.make_llm_for(mixed, "agent"), RoutingLLM)
    assert isinstance(runner.make_llm_for(scenario, "agent"), AnthropicLLM)
    assert runner.API_KEY_ENV == API_KEY_ENV


def test_make_llm_for_observers_follows_each_observers_model(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    def obs(name: str, **fields: Any) -> dict[str, Any]:
        return {"name": name, "identity": "i", "conditions": [{"id": "c", "when": "w"}], **fields}

    monkeypatch.delenv(API_KEY_ENV, raising=False)
    # No LLM observer: the default observer model decides the provider (Anthropic by
    # default, whatever the agent runs on; a local one when models.observer says so).
    local = scenario.model_copy(update={"models": Models(agent="ollama:command-r7b")})
    assert runner.llm_observers(local) == [] and runner.observer_providers(local) == []
    with pytest.raises(RuntimeError, match="needed for the observer"):
        runner.make_llm_for(local, "observer")
    local_observer = scenario.model_copy(
        update={"models": Models(agent="ollama:command-r7b", observer="ollama:command-r7b")}
    )
    assert isinstance(runner.make_llm_for(local_observer, "observer"), OllamaLLM)
    # models.observer overrides the agent's; a per-observer model overrides that.
    hosted = local.model_copy(
        update={"models": Models(agent="ollama:command-r7b", observer="claude-sonnet-5-5")}
    ).with_observers([obs("a")])
    assert runner.observer_providers(hosted) == ["anthropic"]
    with pytest.raises(RuntimeError, match="needed for the observer"):
        runner.make_llm_for(hosted, "observer")
    own = hosted.with_observers([obs("a", model="ollama:qwen2.5:7b")], replace=True)
    assert runner.observer_providers(own) == ["ollama"]
    assert isinstance(runner.make_llm_for(own, "observer"), OllamaLLM)
    # A code observer needs no model; two providers among the LLM observers route per call.
    mixed = hosted.with_observers(
        [
            obs("a", model="ollama:qwen2.5:7b"),
            obs("b"),
            {
                "name": "c",
                "identity": "i",
                "kind": "code",
                "conditions": [{"id": "c", "when": "w", "check": {"tool_called": "x"}}],
            },
        ],
        replace=True,
    )
    assert [o.name for o in runner.llm_observers(mixed)] == ["a", "b"]
    assert runner.observer_providers(mixed) == ["ollama", "anthropic"]
    with pytest.raises(RuntimeError, match="needed for the observer"):
        runner.make_llm_for(mixed, "observer")
    monkeypatch.setenv(API_KEY_ENV, "not-a-real-key")
    assert isinstance(runner.make_llm_for(mixed, "observer"), RoutingLLM)
    assert isinstance(runner.make_llm_for(hosted, "observer"), AnthropicLLM)


def test_apply_model_overrides(scenario: Scenario) -> None:
    assert runner.apply_model_overrides(scenario, None) is scenario
    assert runner.apply_model_overrides(scenario, {}) is scenario
    updated = runner.apply_model_overrides(
        scenario, {"planner": "ollama:command-r7b", "user": "ollama:llama3.2:3b"}
    )
    assert updated.models.planner == "ollama:command-r7b"
    assert updated.models.user == "ollama:llama3.2:3b"
    assert updated.models.agent == scenario.models.agent  # untouched roles keep their value
    assert updated.models.allow_same_judge is False
    assert scenario.models.planner == DEFAULT_PLANNER_MODEL  # the original is not mutated
    assert updated.name == scenario.name and updated.server == scenario.server

    same = runner.apply_model_overrides(
        scenario,
        {"agent": "ollama:qwen2.5:7b", "judge": "ollama:qwen2.5:7b"},
        allow_same_judge=True,
    )
    assert same.models.allow_same_judge is True
    check_judge_model(same)
    with pytest.raises(ValueError, match="same as the agent model"):
        check_judge_model(runner.apply_model_overrides(scenario, {"judge": scenario.models.agent}))

    already = scenario.model_copy(update={"models": Models(allow_same_judge=True)})
    assert runner.apply_model_overrides(already, {"user": "x"}).models.allow_same_judge is True
    with pytest.raises(ValueError, match="unknown model role\\(s\\) critic"):
        runner.apply_model_overrides(scenario, {"critic": "x"})
    with pytest.raises(ValueError, match="no model name"):
        runner.apply_model_overrides(scenario, {"agent": "ollama:"})
    assert runner.uses_local_models(updated) is True
    assert runner.uses_local_models(scenario) is False


def test_report_cost_note_says_local(scenario: Scenario, tmp_path: FsPath) -> None:
    local = runner.apply_model_overrides(scenario, {"agent": "ollama:command-r7b"})
    json_path, md_path = runner._write_report(tmp_path, local, [], [])
    report = Report.load(json_path)
    assert report.cost_note.startswith(LOCAL_COST_NOTE)
    assert "local model(s) via Ollama cost 0" in md_path.read_text(encoding="utf-8")
    hosted_json, _ = runner._write_report(tmp_path, scenario, [], [])
    assert "local" not in Report.load(hosted_json).cost_note


def test_overrides_reach_the_run_directory(
    scenario_path: FsPath, tmp_path: FsPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``scenario.json`` in the run dir records the effective models (dry run, no LLM)."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    run_dir = runner.run_scenario(
        scenario_path,
        tmp_path / "runs",
        repeat=1,
        mode="guided",
        dry_run=True,
        model_overrides={"planner": "ollama:command-r7b", "judge": "ollama:qwen2.5:7b"},
        allow_same_judge=True,
    )
    saved = Scenario.model_validate_json((run_dir / "scenario.json").read_text(encoding="utf-8"))
    assert saved.models.planner == "ollama:command-r7b"
    assert saved.models.judge == "ollama:qwen2.5:7b"
    assert saved.models.allow_same_judge is True
    assert "local model(s) via Ollama" in (run_dir / "report.md").read_text(encoding="utf-8")


# --- CLI --models / --allow-same-judge ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("planner=ollama:command-r7b", {"planner": "ollama:command-r7b"}),
        (
            "agent=ollama:command-r7b,user=ollama:llama3.2:3b,judge=ollama:qwen2.5:7b",
            {
                "agent": "ollama:command-r7b",
                "user": "ollama:llama3.2:3b",
                "judge": "ollama:qwen2.5:7b",
            },
        ),
        (" judge = claude-opus-5-5 , ", {"judge": "claude-opus-5-5"}),
        ("agent=a=b", {"agent": "a=b"}),  # split on the first '=' only
    ],
)
def test_parse_model_overrides(text: str, expected: dict[str, str]) -> None:
    assert parse_model_overrides(text) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("critic=x", "unknown key 'critic'"),
        ("planner", "expects key=value"),
        ("=x", "expects key=value"),
        ("planner=", "expects key=value"),
        ("", "at least one"),
        (",", "at least one"),
    ],
)
def test_parse_model_overrides_errors(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_model_overrides(text)


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def plan_scenario(self, *args: Any, **kwargs: Any) -> FsPath:
        self.calls.append(("plan_scenario", args, kwargs))
        return FsPath("plan.json")

    def run_scenario(self, *args: Any, **kwargs: Any) -> FsPath:
        self.calls.append(("run_scenario", args, kwargs))
        return FsPath("nowhere")  # no report.json there, so exit 0

    def run_suite(self, *args: Any, **kwargs: Any) -> int:
        self.calls.append(("run_suite", args, kwargs))
        return 0


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    fake = FakeRunner()
    module = types.ModuleType(RUNNER_MODULE)
    for name in ("plan_scenario", "run_scenario", "run_suite"):
        setattr(module, name, getattr(fake, name))
    monkeypatch.setitem(sys.modules, RUNNER_MODULE, module)
    monkeypatch.delenv("MCPSIM_DRY_RUN", raising=False)
    return fake


def test_cli_passes_model_overrides_only_when_given(fake_runner: FakeRunner) -> None:
    assert main(["plan", "s.yaml"]) == EXIT_OK
    assert fake_runner.calls[-1][2] == {"dry_run": False}

    assert main(["plan", "s.yaml", "--models", "planner=ollama:command-r7b"]) == EXIT_OK
    assert fake_runner.calls[-1] == (
        "plan_scenario",
        ("s.yaml", "runs"),
        {"dry_run": False, "model_overrides": {"planner": "ollama:command-r7b"}},
    )

    code = main(
        [
            "run",
            "s.yaml",
            "--repeat",
            "1",
            "--models",
            "agent=ollama:command-r7b,user=ollama:llama3.2:3b",
            "--models",
            "judge=ollama:qwen2.5:7b",
            "--allow-same-judge",
        ]
    )
    assert code == EXIT_OK
    name, args, kwargs = fake_runner.calls[-1]
    assert (name, args) == ("run_scenario", ("s.yaml", "runs"))
    assert kwargs == {
        "plan_path": None,
        "only_path": None,
        "repeat": 1,
        "mode": None,
        "dry_run": False,
        "model_overrides": {
            "agent": "ollama:command-r7b",
            "user": "ollama:llama3.2:3b",
            "judge": "ollama:qwen2.5:7b",
        },
        "allow_same_judge": True,
    }

    assert main(["suite", "scenarios", "--allow-same-judge", "--models", "judge=x"]) == EXIT_OK
    assert fake_runner.calls[-1][2] == {
        "threshold": 1.0,
        "dry_run": False,
        "model_overrides": {"judge": "x"},
        "allow_same_judge": True,
    }
    assert main(["run", "s.yaml"]) == EXIT_OK
    assert "model_overrides" not in fake_runner.calls[-1][2]
    assert "allow_same_judge" not in fake_runner.calls[-1][2]


def test_cli_rejects_bad_models_key(
    fake_runner: FakeRunner, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "s.yaml", "--models", "critic=x"])
    assert exc_info.value.code == 2
    assert "unknown key 'critic'" in capsys.readouterr().err
    assert fake_runner.calls == []


def test_model_kwargs_merges_repeated_flags() -> None:
    ns = types.SimpleNamespace(models=[{"agent": "a"}, {"agent": "b", "user": "u"}])
    assert model_kwargs(ns) == {"model_overrides": {"agent": "b", "user": "u"}}  # type: ignore[arg-type]
    assert model_kwargs(types.SimpleNamespace()) == {}  # type: ignore[arg-type]
