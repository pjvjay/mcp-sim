"""The CLI calls ``mcpsim.runner`` with exactly the keyword signatures documented in cli.py.

A fake runner module is injected into ``sys.modules`` so these tests pass before and after the
real runner exists, and so the contract (names, keywords, return types) is pinned by a test the
Stage 3 integrator can read.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from mcpsim.cli import (
    DRY_RUN_ENV,
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_USAGE,
    RUNNER_MODULE,
    dry_run_requested,
    main,
)
from mcpsim.report import Report, aggregate, render_markdown
from mcpsim.scenario import ScenarioError
from mcpsim.verdict import Verdict


def _verdict(index: int, passed: bool) -> Verdict:
    return Verdict(
        path_id="happy",
        mode="guided",
        index=index,
        passed=passed,
        score=1.0 if passed else 0.0,
        votes=1,
        judge_model="dry-run",
    )


def _write_report(run_dir: Path, passed: int, runs: int) -> Report:
    report = aggregate(
        [_verdict(i, i < passed) for i in range(runs)], scenario="s", run_dir=str(run_dir)
    )
    report.save(run_dir / "report.json")
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


class FakeRunner:
    """Records calls; behaviour is configured per test through attributes."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.raise_on_run: Exception | None = None
        self.suite_code = 0

    def plan_scenario(self, *args: Any, **kwargs: Any) -> Path:
        self.calls.append(("plan_scenario", args, kwargs))
        return self.run_dir / "plan.json"

    def run_scenario(self, *args: Any, **kwargs: Any) -> Path:
        self.calls.append(("run_scenario", args, kwargs))
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return self.run_dir

    def judge_run_dir(self, *args: Any, **kwargs: Any) -> list[Path]:
        self.calls.append(("judge_run_dir", args, kwargs))
        return [self.run_dir / "verdicts" / "happy-guided-0.json"]

    def report_run_dir(self, *args: Any, **kwargs: Any) -> tuple[Path, Path]:
        self.calls.append(("report_run_dir", args, kwargs))
        return self.run_dir / "report.json", self.run_dir / "report.md"

    def run_suite(self, *args: Any, **kwargs: Any) -> int:
        self.calls.append(("run_suite", args, kwargs))
        return self.suite_code


@pytest.fixture
def fake_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    runner = FakeRunner(tmp_path / "runs" / "s" / "t1")
    runner.run_dir.mkdir(parents=True)
    module = types.ModuleType(RUNNER_MODULE)
    for name in ("plan_scenario", "run_scenario", "judge_run_dir", "report_run_dir", "run_suite"):
        setattr(module, name, getattr(runner, name))
    monkeypatch.setitem(sys.modules, RUNNER_MODULE, module)
    monkeypatch.delenv(DRY_RUN_ENV, raising=False)
    return runner


def test_plan_calls_plan_scenario(
    fake_runner: FakeRunner, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["plan", "s.yaml", "--out", "o", "--dry-run"]) == EXIT_OK
    assert fake_runner.calls == [("plan_scenario", ("s.yaml", "o"), {"dry_run": True})]
    assert capsys.readouterr().out.strip() == str(fake_runner.run_dir / "plan.json")


def test_run_passes_every_option_as_keyword(
    fake_runner: FakeRunner, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_report(fake_runner.run_dir, passed=3, runs=3)
    code = main(
        [
            "run",
            "s.yaml",
            "--out",
            "o",
            "--plan",
            "p.json",
            "--only-path",
            "happy",
            "--repeat",
            "2",
            "--mode",
            "free",
            "--dry-run",
        ]
    )
    assert code == EXIT_OK
    assert fake_runner.calls == [
        (
            "run_scenario",
            ("s.yaml", "o"),
            {
                "plan_path": "p.json",
                "only_path": "happy",
                "repeat": 2,
                "mode": "free",
                "dry_run": True,
            },
        )
    ]
    out = capsys.readouterr().out
    assert f"run dir: {fake_runner.run_dir}" in out
    assert "3/3 runs passed (100.0%)" in out
    assert f"report: {fake_runner.run_dir / 'report.md'}" in out


def test_run_defaults_are_none_and_dry_run_false(fake_runner: FakeRunner) -> None:
    main(["run", "s.yaml"])
    _, args, kwargs = fake_runner.calls[0]
    assert args == ("s.yaml", "runs")
    assert kwargs == {
        "plan_path": None,
        "only_path": None,
        "repeat": None,
        "mode": None,
        "dry_run": False,
    }


def test_run_exit_code_honours_threshold(fake_runner: FakeRunner) -> None:
    _write_report(fake_runner.run_dir, passed=4, runs=5)
    assert main(["run", "s.yaml"]) == EXIT_FAILURE  # default threshold 1.0
    assert main(["run", "s.yaml", "--threshold", "0.8"]) == EXIT_OK
    assert main(["run", "s.yaml", "--threshold", "0.9"]) == EXIT_FAILURE


def test_run_without_report_json_exits_ok(fake_runner: FakeRunner) -> None:
    assert main(["run", "s.yaml"]) == EXIT_OK


def test_dry_run_env_is_resolved_by_the_cli(
    fake_runner: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DRY_RUN_ENV, "1")
    main(["run", "s.yaml"])
    assert fake_runner.calls[-1][2]["dry_run"] is True
    main(["plan", "s.yaml"])
    assert fake_runner.calls[-1][2]["dry_run"] is True
    main(["suite", "d"])
    assert fake_runner.calls[-1][2]["dry_run"] is True


@pytest.mark.parametrize(
    ("value", "flag", "expected"),
    [
        ("1", False, True),
        ("true", False, True),
        ("YES", False, True),
        ("0", False, False),
        ("", False, False),
        ("0", True, True),
    ],
)
def test_dry_run_requested(value: str, flag: bool, expected: bool) -> None:
    assert dry_run_requested(flag, {DRY_RUN_ENV: value}) is expected
    assert dry_run_requested(False, {}) is False


def test_judge_calls_judge_run_dir(
    fake_runner: FakeRunner, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_report(fake_runner.run_dir, passed=1, runs=1)
    assert main(["judge", str(fake_runner.run_dir), "--votes", "5"]) == EXIT_OK
    assert fake_runner.calls == [("judge_run_dir", (str(fake_runner.run_dir),), {"votes": 5})]
    assert "judged 1 transcript(s)" in capsys.readouterr().out
    main(["judge", str(fake_runner.run_dir)])
    assert fake_runner.calls[-1][2] == {"votes": None}


def test_report_prints_summary_or_markdown(
    fake_runner: FakeRunner, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_report(fake_runner.run_dir, passed=2, runs=4)
    assert main(["report", str(fake_runner.run_dir), "--threshold", "0.5"]) == EXIT_OK
    assert fake_runner.calls == [("report_run_dir", (str(fake_runner.run_dir),), {})]
    out = capsys.readouterr().out
    assert "2/4 runs passed (50.0%)" in out and "# mcp-sim report" not in out

    assert main(["report", str(fake_runner.run_dir), "--markdown"]) == EXIT_FAILURE
    out = capsys.readouterr().out
    assert out.startswith("# mcp-sim report: s\n")
    assert "| happy | guided | 4 | 2 | 50.0% |" in out


def test_suite_returns_runner_exit_code(fake_runner: FakeRunner) -> None:
    fake_runner.suite_code = 1
    assert main(["suite", "scenarios", "--out", "o", "--threshold", "0.8"]) == 1
    assert fake_runner.calls == [
        ("run_suite", ("scenarios", "o"), {"threshold": 0.8, "dry_run": False})
    ]
    fake_runner.suite_code = 0
    assert main(["suite", "scenarios"]) == 0
    assert fake_runner.calls[-1][2] == {"threshold": 1.0, "dry_run": False}


def test_user_errors_become_one_line_and_exit_1(
    fake_runner: FakeRunner, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_runner.raise_on_run = ScenarioError("s.yaml: invalid scenario")
    assert main(["run", "s.yaml"]) == EXIT_FAILURE
    assert capsys.readouterr().err.strip() == "mcpsim run: s.yaml: invalid scenario"

    fake_runner.raise_on_run = KeyError("no path with id 'nope'")
    assert main(["run", "s.yaml", "--only-path", "nope"]) == EXIT_FAILURE
    assert capsys.readouterr().err.strip() == "mcpsim run: no path with id 'nope'"

    monkeypatch.setenv("MCPSIM_DEBUG", "1")
    with pytest.raises(KeyError):
        main(["run", "s.yaml", "--only-path", "nope"])


def test_missing_runner_is_not_yet_wired(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # ``None`` in sys.modules makes the import raise ModuleNotFoundError for exactly this name.
    monkeypatch.setitem(sys.modules, RUNNER_MODULE, None)
    for argv in (
        ["plan", "s.yaml"],
        ["run", "s.yaml"],
        ["judge", "d"],
        ["report", "d"],
        ["suite", "d"],
    ):
        assert main(argv) == EXIT_USAGE
        assert f"mcpsim {argv[0]}: not yet wired" in capsys.readouterr().err


def test_broken_runner_import_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runner that exists but fails to import is a bug, not 'not yet wired'."""
    module = types.ModuleType(RUNNER_MODULE)

    def plan_scenario(*args: Any, **kwargs: Any) -> Path:
        raise ModuleNotFoundError("No module named 'nonexistent_dep'", name="nonexistent_dep")

    module.plan_scenario = plan_scenario  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNNER_MODULE, module)
    with pytest.raises(ModuleNotFoundError):
        main(["plan", "s.yaml"])
