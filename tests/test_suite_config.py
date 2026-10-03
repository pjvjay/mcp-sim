"""``mcpsim config`` and ``mcpsim suite``: the CLI over the simulate skill, and the configured
suite run end to end in dry run (the fake stdio server, no LLM)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from mcpsim import runner
from mcpsim.cli import EXIT_FAILURE, EXIT_OK, RUNNER_MODULE, main
from mcpsim.report import SuiteReport
from mcpsim.skill import CHECKOUT_DIR, SKILL_ENV

FAKE_OUTCOME = {"slug": "penne", "price": {"$gt": 0}, "origin_status": "verified"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (SKILL_ENV, "MCPSIM_DRY_RUN", "MCPSIM_DEBUG", "SUITE_EXTRA"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def skill(tmp_path: Path) -> Path:
    target = tmp_path / "skill"
    shutil.copytree(CHECKOUT_DIR, target)
    return target


def write_config(skill: Path, **config: Any) -> None:
    (skill / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def write_scenario(
    folder: Path, name: str, category: str, scenario_data: dict[str, Any], **extra: Any
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    data = {**scenario_data, "name": name, "category": category, **extra}
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def suite_setup(
    skill: Path, tmp_path: Path, scenario_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Three scenarios in two sources (one through an environment reference) plus one that does
    not load; the config's overrides give 'lookup-*' one judge vote."""
    write_scenario(tmp_path / "direct", "lookup-a", "Lookup", scenario_data)
    write_scenario(
        tmp_path / "direct",
        "lookup-b",
        "Lookup",
        scenario_data,
        expected_outcome={"json": {"slug": "penne", "origin_status": "unverified"}},
    )
    write_scenario(tmp_path / "extra", "plan-c", "Planning", scenario_data)
    write_scenario(
        tmp_path / "extra", "broken-d", "Planning", scenario_data, agent={"skill": "env:NOPE_SKILL"}
    )
    monkeypatch.setenv("SUITE_EXTRA", str(tmp_path / "extra"))
    write_config(
        skill,
        defaults={"judge": "anthropic:claude-opus-5-5"},
        run={"repeat": 1, "modes": ["guided"], "judge_votes": 3, "concurrency": 2},
        scenarios=["direct", "${SUITE_EXTRA}", "${UNSET_SUITE_DIR}/x"],
        runs_dir="runs/sim",
        overrides=[{"match": {"name": "lookup-*"}, "run": {"judge_votes": 1}}],
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- mcpsim config ----------------------------------------------------------------------------


def test_config_prints_roles_models_sources_and_scenarios(
    suite_setup: Path, skill: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "--skill", str(skill)]) == EXIT_OK
    out = capsys.readouterr().out
    assert f"skill: simulate  ({skill})" in out
    assert "  judge: claude-opus-5-5  (from config.yaml defaults)" in out
    assert "  agent: claude-sonnet-5-5  (from roles/agent.md)" in out
    assert "  planner-local: ollama:command-r7b  (local execution planner" in out
    assert "      max_tokens 4096; votes 3; prompts: system, user" in out
    assert "  repeat: 1  (from config.yaml run)" in out
    assert "  modes: guided  (from config.yaml run)" in out
    assert f"  direct -> {suite_setup / 'direct'}: 2 file(s)" in out
    assert "  ${UNSET_SUITE_DIR}/x -> -: 0 file(s); skipped: UNSET_SUITE_DIR is not set" in out
    assert f"runs_dir: {suite_setup / 'runs' / 'sim'}" in out
    assert "  1. match name=lookup-*: run judge_votes=1" in out


def test_config_json_and_per_scenario_resolution(
    suite_setup: Path, skill: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "--skill", str(skill), "--json", "--scenario", "lookup-a"]) == EXIT_OK
    info = json.loads(capsys.readouterr().out)
    assert info["scenario"] == "lookup-a"
    assert info["run"]["judge_votes"] == {
        "value": 1,
        "source": "config.yaml overrides[1] (name=lookup-*)",
    }
    assert info["roles"]["user"]["model"] == "claude-haiku-4-5-20251001"
    assert info["roles"]["observer"]["prompts"]["user"] == {
        "accepts": ["watched"],
        "requires": ["watched"],
    }
    # A scenario file path works too, and a name that is not configured is a clear error.
    path = suite_setup / "extra" / "plan-c.yaml"
    assert main(["config", "--skill", str(skill), "--json", "--scenario", str(path)]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["run"]["judge_votes"]["value"] == 3
    assert main(["config", "--skill", str(skill), "--scenario", "nope"]) == EXIT_FAILURE
    assert "no configured scenario named 'nope'" in capsys.readouterr().err


def test_config_reports_an_invalid_skill(skill: Path, capsys: pytest.CaptureFixture[str]) -> None:
    role = skill / "roles" / "observer.md"
    text = role.read_text(encoding="utf-8")
    role.write_text(text.replace("{{ identity }}", "{{ who }}"), encoding="utf-8")
    assert main(["config", "--skill", str(skill)]) == EXIT_FAILURE
    err = capsys.readouterr().err
    assert err.startswith("mcpsim config: ") and "unknown placeholder 'who'" in err


def test_the_environment_variable_selects_the_skill(
    skill: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(SKILL_ENV, str(skill))
    assert main(["config", "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["skill"]["path"] == str(skill)


# --- mcpsim suite --list ----------------------------------------------------------------------


def test_suite_list_selects_by_name_and_category(
    suite_setup: Path, skill: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["suite", "--skill", str(skill), "--list"]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    # Sources in config order, files in name order within each.
    assert [line.split()[0] for line in lines] == ["lookup-a", "lookup-b", "broken-d", "plan-c"]
    assert "[Lookup] repeat 1, modes guided, judge_votes 1, judge claude-opus-5-5" in lines[0]
    assert "error: " in lines[2] and "NOPE_SKILL is not set" in lines[2]
    assert "[Planning] repeat 1, modes guided, judge_votes 3" in lines[3]

    assert main(["suite", "--skill", str(skill), "--list", "--category", "Plan*"]) == EXIT_OK
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()] == ["plan-c"]
    args = ["suite", "--skill", str(skill), "--list", "--name", "lookup-b", "--name", "plan-*"]
    assert main(args) == EXIT_OK
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()] == [
        "lookup-b",
        "plan-c",
    ]
    assert main(["suite", "--skill", str(skill), "--list", "--name", "zzz*"]) == EXIT_FAILURE
    assert "matches --name zzz*" in capsys.readouterr().err


def test_suite_list_json_carries_the_resolution(
    suite_setup: Path, skill: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["suite", "--skill", str(skill), "--list", "--json", "--repeat", "4"]
    assert main([*args, "--models", "agent=claude-opus-5", "--name", "lookup-a"]) == EXIT_OK
    [row] = json.loads(capsys.readouterr().out)
    assert row["name"] == "lookup-a" and row["category"] == "Lookup" and row["error"] is None
    assert row["resolved"]["run"]["repeat"] == {"value": 4, "source": "command line"}
    assert row["resolved"]["models"]["agent"] == {
        "model": "claude-opus-5",
        "source": "command line",
    }


# --- mcpsim suite: wiring ---------------------------------------------------------------------


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def run_suite(self, *args: Any, **kwargs: Any) -> int:
        self.calls.append(("run_suite", args, kwargs))
        return 0

    def run_scenario(self, *args: Any, **kwargs: Any) -> Path:
        self.calls.append(("run_scenario", args, kwargs))
        raise RuntimeError("stop here")

    def judge_run_dir(self, *args: Any, **kwargs: Any) -> list[Path]:
        self.calls.append(("judge_run_dir", args, kwargs))
        return []


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    fake = FakeRunner()
    module = types.ModuleType(RUNNER_MODULE)
    for name in ("run_suite", "run_scenario", "judge_run_dir"):
        setattr(module, name, getattr(fake, name))
    monkeypatch.setitem(sys.modules, RUNNER_MODULE, module)
    return fake


def test_suite_passes_the_new_options_only_when_given(fake_runner: FakeRunner) -> None:
    assert main(["suite"]) == EXIT_OK
    plain = {"threshold": 1.0, "dry_run": False}
    assert fake_runner.calls[-1] == ("run_suite", (None, None), plain)
    args = [
        "suite", "--skill", "sk", "--name", "a*", "--name", "b", "--category", "C*",
        "--repeat", "2", "--modes", "guided", "--out", "o",
    ]
    assert main(args) == EXIT_OK
    assert fake_runner.calls[-1] == (
        "run_suite",
        (None, "o"),
        {
            "threshold": 1.0,
            "dry_run": False,
            "skill": "sk",
            "names": ["a*", "b"],
            "categories": ["C*"],
            "repeat": 2,
            "modes": ["guided"],
        },
    )


def test_suite_refuses_unknown_modes(fake_runner: FakeRunner) -> None:
    with pytest.raises(SystemExit) as info:
        main(["suite", "--modes", "replay"])
    assert info.value.code == 2


def test_run_and_judge_take_the_skill(fake_runner: FakeRunner) -> None:
    assert main(["run", "s.yaml", "--skill", "sk"]) == EXIT_FAILURE  # the fake stops the run
    assert fake_runner.calls[-1][2]["skill"] == "sk"
    main(["run", "s.yaml"])
    assert "skill" not in fake_runner.calls[-1][2]
    main(["judge", "rd", "--skill", "sk"])
    assert fake_runner.calls[-1] == ("judge_run_dir", ("rd",), {"votes": None, "skill": "sk"})


# --- mcpsim suite: a real dry run -------------------------------------------------------------


def test_the_configured_suite_runs_into_runs_dir_with_a_pass_k_table(
    suite_setup: Path, skill: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = runner.run_suite(skill=skill, dry_run=True, names=["lookup-*", "broken-*"])
    assert code == 1, "lookup-b fails its matcher and broken-d does not load"
    out = capsys.readouterr().out
    runs = suite_setup / "runs" / "sim"
    assert sorted(p.name for p in runs.iterdir() if not p.name.startswith("suite-")) == [
        "lookup-a",
        "lookup-b",
    ]
    table = out[out.index("scenario  ") :]
    rows = table.splitlines()
    assert rows[0].split() == [
        "scenario", "category", "passed", "rate", "pass^k", "cost", "time", "result"
    ]
    assert rows[1].split()[:6] == ["lookup-a", "Lookup", "1/1", "100%", "pass^1", "yes"]
    assert rows[2].split()[:6] == ["lookup-b", "Lookup", "0/1", "0%", "pass^1", "no"]
    assert rows[3].split()[:3] == ["broken-d", "-", "-"], "it never loaded: no category"
    assert "error: " in rows[3] and "NOPE_SKILL is not set" in rows[3]
    [suite_dir] = sorted(runs.glob("suite-*"))
    suite = SuiteReport.load(suite_dir / "suite.json")
    assert [s.scenario for s in suite.scenarios] == ["lookup-a", "lookup-b"]
    assert suite.pass_k_scenarios == 1
    assert [e.scenario for e in suite.errors] == ["broken-d"]
    assert "## Scenarios that could not run" in (suite_dir / "suite.md").read_text("utf-8")
    # The run directory records what ran: the config's settings, the override's single vote.
    [run_dir] = sorted((runs / "lookup-a").iterdir())
    recorded = json.loads((run_dir / "scenario.json").read_text(encoding="utf-8"))
    assert (recorded["repeat"], recorded["judge_votes"], recorded["concurrency"]) == (1, 1, 2)
    assert recorded["models"]["judge"] == "claude-opus-5-5"
    assert sorted(p.stem for p in (run_dir / "transcripts").glob("*.jsonl")) == [
        "happy-dry-run-guided-0"
    ], "modes: [guided] in config.yaml drops the free run"


def test_a_suite_over_a_directory_still_works_and_the_command_line_wins(
    tmp_path: Path, skill: Path, scenario_data: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_scenario(tmp_path / "dir", "only", "X", scenario_data)
    write_config(skill, run={"repeat": 1, "modes": ["guided"]})
    monkeypatch.chdir(tmp_path)
    code = runner.run_suite(
        tmp_path / "dir", tmp_path / "out", skill=skill, dry_run=True, repeat=2, modes=["free"]
    )
    assert code == 0
    [run_dir] = sorted((tmp_path / "out" / "only").iterdir())
    assert sorted(p.stem for p in (run_dir / "transcripts").glob("*.jsonl")) == [
        "happy-dry-run-free-0",
        "happy-dry-run-free-1",
    ]
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["pass_k"] == {"k": 2, "all_passed": True}


def test_a_scenario_whose_run_raises_is_a_row_and_the_suite_goes_on(
    suite_setup: Path,
    skill: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real = runner.run_scenario

    class CreditError(Exception):
        """Stands in for anthropic.BadRequestError ('credit balance is too low')."""

    def flaky(path: Any, *args: Any, **kwargs: Any) -> Path:
        if Path(path).stem == "lookup-a":
            raise CreditError("Error code: 400 - credit balance is too low")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(runner, "run_scenario", flaky)
    code = runner.run_suite(skill=skill, dry_run=True, names=["lookup-*", "plan-*"])
    assert code == 1
    out = capsys.readouterr().out
    assert "lookup-a: error: CreditError: Error code: 400 - credit balance is too low" in out
    assert "lookup-b: 0/1 runs passed" in out and "plan-c: 1/1 runs passed" in out
    [suite_dir] = sorted((suite_setup / "runs" / "sim").glob("suite-*"))
    suite = SuiteReport.load(suite_dir / "suite.json")
    assert [s.scenario for s in suite.scenarios] == ["lookup-b", "plan-c"]
    assert [(e.scenario, e.error) for e in suite.errors] == [
        ("lookup-a", "CreditError: Error code: 400 - credit balance is too low")
    ]
    monkeypatch.setenv("MCPSIM_DEBUG", "1")
    with pytest.raises(CreditError):
        runner.run_suite(skill=skill, dry_run=True, names=["lookup-a"])


def test_run_sh_is_valid_bash_and_documents_its_commands() -> None:
    script = CHECKOUT_DIR / "scripts" / "run.sh"
    assert os.access(script, os.X_OK)
    subprocess.run(["bash", "-n", str(script)], check=True)
    shown = subprocess.run(
        [str(script), "help"], capture_output=True, text=True, check=True
    ).stdout
    for command in ("preflight", "scenarios", "suite", "ui", "config", "report", "all"):
        assert f"run.sh {command}" in shown
