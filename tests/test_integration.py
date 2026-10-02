"""The one real run (DESIGN §8): the boycott scenario against the pantry server with a key.

Marked ``integration``; CI runs ``pytest -m "not integration"``. Without ``ANTHROPIC_API_KEY`` the
test is skipped and the skip reason says so, so a local ``pytest`` run shows it as skipped rather
than silently green.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mcpsim.report import Report
from mcpsim.runner import run_scenario
from mcpsim.scenario import load_scenario
from mcpsim.transcript import Transcript

REPO_ROOT = Path(__file__).resolve().parent.parent
BOYCOTT = REPO_ROOT / "scenarios" / "pantry" / "tomato-penne-boycott.yaml"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY is not set: the real boycott run against pantry is skipped",
)
def test_boycott_scenario_for_real(tmp_path: Path) -> None:
    scenario = load_scenario(BOYCOTT)
    assert scenario.server.stdio is not None
    if not Path(scenario.server.stdio.command).is_file():
        pytest.skip(f"pantry MCP server not installed at {scenario.server.stdio.command}")

    run_dir = run_scenario(BOYCOTT, tmp_path / "runs", repeat=1, mode="guided")

    assert (run_dir / "plan.json").is_file() and (run_dir / "scenario.json").is_file()
    transcripts = sorted((run_dir / "transcripts").glob("*.jsonl"))
    assert transcripts, "at least the happy path must have run"
    for path in transcripts:
        transcript = Transcript.read_jsonl(path)
        assert transcript.tool_results(), f"{path.name}: no tool_result came from the server"
        assert (run_dir / "verdicts" / f"{path.stem}.json").is_file()
    report = Report.load(run_dir / "report.json")
    assert report.runs == len(transcripts)
    assert report.judge_models == [scenario.models.judge]
    assert (run_dir / "report.md").is_file()
