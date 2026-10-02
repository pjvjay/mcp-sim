"""The committed cheapest-penne scenario in dry run against the real pantry server.

Skipped when the pantry server's ``pantry-mcp`` script is not on this machine. The scenario is
copied with ``DB_URL`` pointing at a separate seeded SQLite file so a live run on the suite's
database is never disturbed.
"""

from __future__ import annotations

from pathlib import Path as FsPath

import pytest
import yaml

from mcpsim import runner
from mcpsim.plan import ExecutionPlan
from mcpsim.scout import ScoutResult
from mcpsim.transcript import Transcript
from mcpsim.verdict import Verdict

SCENARIO = FsPath(__file__).resolve().parent.parent / "scenarios" / "pantry" / "cheapest-penne.yaml"
TEST_DB = FsPath(__file__).resolve().parent.parent / "runs" / "pantry-sim-fable.db"


def _server_command() -> str:
    data = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    return str(data["server"]["stdio"]["command"])


pytestmark = pytest.mark.skipif(
    not FsPath(_server_command()).is_file(), reason="the pantry MCP server is not installed here"
)


@pytest.fixture
def scenario_copy(tmp_path: FsPath) -> FsPath:
    data = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    data["server"]["stdio"]["env"]["DB_URL"] = f"sqlite:///{TEST_DB}"
    path = tmp_path / "cheapest-penne.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_cheapest_penne_dry_run_yields_find_products_result_and_passes_the_matcher(
    scenario_copy: FsPath, tmp_path: FsPath
) -> None:
    run_dir = runner.run_scenario(
        scenario_copy, tmp_path / "runs", dry_run=True, repeat=1, mode="guided"
    )

    scout = ScoutResult.load(run_dir / "scout.json")
    assert scout.observations[0].kind == "resource", "the four pantry resources come first"
    lookups = scout.lookups()
    assert [o.call_label() for o in lookups] == ['find_product(query="penne")']
    assert lookups[0].structured is not None and lookups[0].structured["match"] == "direct"  # type: ignore[index]
    reports = {(r.key, r.trigger): r for r in scout.reports}
    assert reports[("shelf_clerk.direct_match", "scout")].value is True
    assert reports[("shelf_clerk.direct_match", "scout")].evidence == (
        "find_product.match == 'direct', find_product.total == 2"
    )
    assert "get_product" in scout.disclosed and scout.goals == [
        "Quote the cheapest direct hit by exact name, price and store."
    ]
    assert scout.planner_prompt_chars > 0

    plan = ExecutionPlan.load(run_dir / "plan.json")
    [only] = plan.paths
    assert only.steps[0].tool == "find_product" and only.steps[0].arguments_sketch == {
        "query": "penne"
    }
    assert "report: shelf_clerk.direct_match is true" in only.checkpoints
    assert not any(t.startswith(("submit_", "review_", "plan_")) for t in only.tools_used())

    transcript = Transcript.read_jsonl(run_dir / "transcripts" / f"{only.id}-guided-0.jsonl")
    assert transcript.outcome == "completed"
    find_product = transcript.tool_results()[0]
    assert (
        find_product.name == "find_product" and transcript.final_result == find_product.structured
    )
    assert transcript.final_result is not None
    assert (
        transcript.final_result["query"] == "penne" and transcript.final_result["match"] == "direct"
    )
    kinds = transcript.kinds()
    assert kinds[:7] == [
        "system",
        "tools_offered",
        "user",
        "tool_call",
        "tool_result",
        "informant_report",
        "tools_offered",
    ]
    offered = [e for e in transcript.events if e.kind == "tools_offered"]
    assert offered[1].reason == "observer:shelf_clerk.direct_match"  # type: ignore[union-attr]
    goals = [e for e in transcript.events if e.kind == "goal_enabled"]
    assert goals and goals[0].observer == "shelf_clerk"  # type: ignore[union-attr]

    verdict = Verdict.load(run_dir / "verdicts" / f"{only.id}-guided-0.json")
    assert verdict.judge_model == "dry-run" and verdict.votes == 0
    failed = [m for m in verdict.matches if not m.passed]
    # find_product's result has query, match and items; the per-product fields (product_id,
    # product_name, store, price, origin_status) live inside items and stay missing at the top
    # level, so the matcher is honest about what a dry run can and cannot prove.
    assert [m.path for m in verdict.matches if m.passed] == ["query", "match"]
    assert {m.path for m in failed} == {
        "product_id",
        "product_name",
        "store",
        "price",
        "origin_status",
    }
    assert verdict.flags == [] and not any(
        r.startswith("observer:") for r in verdict.failure_reasons
    )
