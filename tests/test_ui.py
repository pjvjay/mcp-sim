"""``mcpsim ui``: every endpoint, the refusals, the job lifecycle and an HTML smoke test.

The fixture builds a scenario directory (a v1 file, a v2 file whose agent runs on an SOP, a
broken file), a runs directory shaped per contract B (judged runs, a plan-only run, a run whose
report predates ``pass_k``) and a copy of the simulate skill with its own ``config.yaml`` and
role frontmatter, loaded by :func:`mcpsim.skill.load_skill`. Jobs run ``tests/fake_mcpsim.py``
instead of the real CLI, except one test that runs the real ``mcpsim run --skill`` in dry run.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import socket
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from starlette.testclient import TestClient

from mcpsim.cli import EXIT_FAILURE, EXIT_USAGE, main
from mcpsim.judge import HONESTY_ITEM
from mcpsim.scenario import load_scenario
from mcpsim.skill import CHECKOUT_DIR, SKILL_ENV, SkillError, load_skill
from mcpsim.ui.app import (
    TOKEN_HEADER,
    UISettings,
    create_app,
    host_allowed,
    host_name,
    is_loopback,
    remote_host_names,
    resolve_settings,
    serve,
)
from mcpsim.ui.jobs import JobManager, RunOptions, build_command, parse_run_options
from tests.conftest import REPO_ROOT, fake_server_stdio_spec

TOKEN = "test-token-0123456789"
BASE = "http://127.0.0.1:8765"
FAKE_MCPSIM = Path(__file__).resolve().parent / "fake_mcpsim.py"


# --- fixture data -----------------------------------------------------------------------------


def _scenario(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "role": "A shopper who wants the price of penne and will not accept guesses.",
        "goal": "Find the price and store of penne.",
        "instructions": ["Use the lookup tool; do not invent prices.", "Report origin_status."],
        "expected_outcome": {"text": "The price and store of penne."},
        "server": fake_server_stdio_spec(),
        "repeat": 2,
        **extra,
    }


V2_EXTRA: dict[str, Any] = {
    "category": "Shopping",
    "title": "Weekly shop on a phone",
    "user_instructions": (
        "You are a parent in Lyon shopping on your phone. Ask for the cheapest penne; "
        "refuse anything that is not penne."
    ),
    "context": {
        "device": "mobile",
        "location": "Lyon, FR",
        "language": "fr-FR",
        "details": {"budget": "20 EUR"},
        "agent_visible": True,
    },
    "expected_behavior": [
        "Calls the lookup tool before quoting a price",
        "Quotes the store exactly as the record has it",
    ],
    "agent": {"skill": "env:RECIPE_SHOPPER_SKILL", "notes": "No write tools in this setup."},
    "models": {"judge": "claude-fable-5-1"},
}


def _verdict(path_id: str, mode: str, index: int, passed: bool, **extra: Any) -> dict[str, Any]:
    return {
        "path_id": path_id,
        "mode": mode,
        "index": index,
        "passed": passed,
        "score": 1.0 if passed else 0.25,
        "matches": [
            {
                "path": "price",
                "op": "$gt",
                "expected": 0,
                "actual": 1.97,
                "passed": True,
                "detail": "",
            }
        ],
        "checklist": [],
        "failure_reasons": [] if passed else ["checklist: quotes the store"],
        "flags": [],
        "votes": 3,
        "judge_model": "claude-opus-5-5",
        **extra,
    }


HOSTILE = '<img src=x onerror="alert(1)"><script>alert(2)</script>'


def _events(cost: float = 0.0123) -> list[dict[str, Any]]:
    return [
        {
            "t": "2025-09-01T10:00:00.000+00:00",
            "kind": "system",
            "scenario": "fake-lookup",
            "path_id": "happy",
            "index": 0,
            "mode": "guided",
            "models": {"agent": "claude-sonnet-5-5", "user": "claude-haiku-4-5-20251001"},
            "prompts": {"agent": "You are an assistant."},
        },
        {
            "t": "2025-09-01T10:00:00.100+00:00",
            "kind": "tools_offered",
            "added": ["lookup"],
            "removed": [],
            "reason": "initial:guided:all",
        },
        {"t": "2025-09-01T10:00:01.000+00:00", "kind": "user", "text": "Cheapest penne?"},
        {
            "t": "2025-09-01T10:00:02.000+00:00",
            "kind": "assistant",
            "text": "Looking it up.",
            "tool_uses": [{"id": "tu1", "name": "lookup", "input": {"slug": "penne"}}],
            "stop_reason": "tool_use",
        },
        {
            "t": "2025-09-01T10:00:02.100+00:00",
            "kind": "tool_call",
            "name": "lookup",
            "arguments": {"slug": "penne"},
            "tool_use_id": "tu1",
        },
        {
            "t": "2025-09-01T10:00:02.300+00:00",
            "kind": "tool_result",
            "name": "lookup",
            "is_error": False,
            "structured": {"name": "Penne", "price": 1.97, "store": HOSTILE},
            "text": "{...}",
            "sha256": "ab",
            "chars": 60,
            "ms": 200.0,
            "tool_use_id": "tu1",
        },
        {
            "t": "2025-09-01T10:00:02.400+00:00",
            "kind": "informant_report",
            "trigger": "end",
            "reports": [
                {
                    "observer": "shelf",
                    "condition": "fabrication",
                    "value": True,
                    "evidence": "quoted a store",
                    "confidence": 0.9,
                    "trigger": "end",
                    "at_event": 5,
                }
            ],
            "flags": ["fabrication"],
            "failures": ["shelf.fabrication — quoted a store"],
            "notes": [],
        },
        {
            "t": "2025-09-01T10:00:03.000+00:00",
            "kind": "assistant",
            "text": f"Penne at {HOSTILE}",
            "tool_uses": [],
            "stop_reason": "end_turn",
        },
        {
            "t": "2025-09-01T10:00:03.000+00:00",
            "kind": "final_result",
            "parsed": {"price": 1.97},
            "raw": "{}",
        },
        {
            "t": "2025-09-01T10:00:03.000+00:00",
            "kind": "usage",
            "per_model": {"claude-sonnet-5-5": {"input_tokens": 10, "output_tokens": 5}},
            "cost_usd": cost,
            "estimate": True,
        },
        {
            "t": "2025-09-01T10:00:04.500+00:00",
            "kind": "end",
            "outcome": "completed",
            "reason": "final answer delivered",
        },
    ]


def _write_run(
    folder: Path,
    *,
    verdicts: list[dict[str, Any]],
    report: dict[str, Any] | None,
    scenario: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    garbage_line: bool = False,
) -> None:
    folder.mkdir(parents=True)
    if scenario is not None:
        (folder / "scenario.json").write_text(json.dumps(scenario))
    if plan is not None:
        (folder / "plan.json").write_text(json.dumps(plan))
    if report is not None:
        (folder / "report.json").write_text(json.dumps(report))
    (folder / "verdicts").mkdir()
    (folder / "transcripts").mkdir()
    for v in verdicts:
        stem = f"{v['path_id']}-{v['mode']}-{v['index']}"
        (folder / "verdicts" / f"{stem}.json").write_text(json.dumps(v))
        lines = [json.dumps(e) for e in _events()]
        if garbage_line:
            lines.insert(2, "this is not json")
        (folder / "transcripts" / f"{stem}.jsonl").write_text("\n".join(lines) + "\n")


PLAN = {
    "scenario": "fake-lookup",
    "catalog_digest": "abc",
    "paths": [
        {
            "id": "happy",
            "kind": "happy",
            "title": "Look it up",
            "rationale": "The direct route.",
            "steps": [
                {
                    "intent": "Find penne",
                    "tool": "lookup",
                    "arguments_sketch": {"slug": "penne"},
                    "success_looks_like": "a price",
                    "expect_error": False,
                }
            ],
            "checkpoints": ["final_result: price is greater than 0"],
        }
    ],
    "notes": ["planner: 1 call"],
}


@dataclass
class Env:
    root: Path
    scenarios: Path
    runs: Path
    skill: Path
    outside: Path

    def settings(self, **extra: Any) -> UISettings:
        """The scenario sources follow the skill's config.yaml unless ``extra`` names them."""
        return UISettings(skill=load_skill(self.skill), runs_dir=self.runs, cwd=self.root, **extra)


def set_frontmatter(path: Path, **values: Any) -> None:
    """Set (or add) top-level frontmatter keys of a role file."""
    text = path.read_text(encoding="utf-8")
    head, body = text.split("\n---\n", 1)
    for key, value in values.items():
        line = f"{key}: {value}"
        pattern = re.compile(rf"^{key}: .*$", re.MULTILINE)
        head = pattern.sub(line, head) if pattern.search(head) else f"{head}\n{line}"
    path.write_text(f"{head}\n---\n{body}", encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (SKILL_ENV, "MCPSIM_DRY_RUN", "MCPSIM_DEBUG"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    # The v2 scenario's agent runs on an SOP named through env:RECIPE_SHOPPER_SKILL.
    sop = tmp_path / "sop" / "SKILL.md"
    sop.parent.mkdir()
    sop.write_text("---\nname: shop-sop\n---\n1. Look the product up before quoting it.\n")
    monkeypatch.setenv("RECIPE_SHOPPER_SKILL", str(sop))
    scenarios = tmp_path / "scenarios"
    (scenarios / "nested").mkdir(parents=True)
    (scenarios / "fake-lookup.yaml").write_text(yaml.safe_dump(_scenario("fake-lookup")))
    (scenarios / "nested" / "v2-shopper.yaml").write_text(
        yaml.safe_dump(_scenario("v2-shopper", **V2_EXTRA))
    )
    broken = _scenario("broken-one")
    del broken["goal"]
    (scenarios / "broken.yaml").write_text(yaml.safe_dump(broken))
    (scenarios / "notes.txt").write_text("not a scenario")

    runs = tmp_path / "runs"
    snap = _scenario("fake-lookup")
    # Oldest: one of two passed, report written before pass_k / duration_s existed.
    _write_run(
        runs / "fake-lookup" / "20250901T100000Z",
        verdicts=[_verdict("happy", "guided", 0, True), _verdict("happy", "guided", 1, False)],
        report={
            "scenario": "fake-lookup",
            "generated_at": "2025-09-01T10:00:08+00:00",
            "runs": 2,
            "passed": 1,
            "pass_rate": 0.5,
            "mean_score": 0.625,
            "cost_usd": 0.0246,
            "judge_models": ["claude-opus-5-5"],
        },
        scenario=snap,
        plan=PLAN,
        garbage_line=True,
    )
    # Newest judged: all passed, contract-B report fields present.
    _write_run(
        runs / "fake-lookup" / "20250902T100000Z",
        verdicts=[_verdict("happy", "guided", 0, True), _verdict("happy", "guided", 1, True)],
        report={
            "scenario": "fake-lookup",
            "runs": 2,
            "passed": 2,
            "pass_rate": 1.0,
            "mean_score": 1.0,
            "cost_usd": 0.03,
            "duration_s": 42.5,
            "pass_k": {"k": 2, "all_passed": True},
            "judge_models": ["claude-opus-5-5"],
        },
        scenario=snap,
        plan=PLAN,
    )
    # Newest directory overall: plan only (mcpsim plan), no verdicts.
    plan_only = runs / "fake-lookup" / "20250903T100000Z"
    plan_only.mkdir(parents=True)
    (plan_only / "plan.json").write_text(json.dumps(PLAN))
    # v2 scenario: no report.json, verdicts carry the expected-behavior checklist.
    checklist = [
        {
            "item": "Calls the lookup tool before quoting a price",
            "passed": True,
            "evidence": 'lookup(slug="penne")',
        },
        {
            "item": "Quotes the store exactly as the record has it",
            "passed": False,
            "evidence": "no evidence",
        },
    ]
    _write_run(
        runs / "v2-shopper" / "20250905T120000Z",
        verdicts=[
            _verdict(
                "happy",
                "guided",
                0,
                False,
                checklist=checklist,
                goal_achieved=True,
                sop_followed=False,
            ),
            _verdict(
                "happy", "free", 0, True, checklist=[{**c, "passed": True} for c in checklist]
            ),
        ],
        report=None,
        # As the runner writes it: the resolved scenario, the SOP's text included.
        scenario=json.loads(
            load_scenario(scenarios / "nested" / "v2-shopper.yaml").model_dump_json()
        ),
    )

    # A copy of the real simulate skill with its own config.yaml and two frontmatter edits.
    skill = tmp_path / "skill"
    shutil.copytree(CHECKOUT_DIR, skill)
    (skill / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "roles_dir": "roles",
                "defaults": {"planner": "anthropic:claude-sonnet-5-5"},
                "run": {"repeat": 3, "modes": ["guided", "free"], "judge_votes": 3},
                "scenarios": ["scenarios", "scenarios/nested"],
                "runs_dir": "runs",
                "overrides": [
                    {
                        "match": {"category": "Shop*"},
                        "models": {"agent": "claude-haiku-4-5-20251001"},
                        "run": {"modes": ["guided"]},
                    }
                ],
            }
        )
    )
    set_frontmatter(skill / "roles" / "agent.md", model="claude-opus-5-5", max_tokens=2048)
    set_frontmatter(skill / "roles" / "user.md", temperature=0.2)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jsonl").write_text('{"kind": "user", "text": "SECRET"}\n')
    (outside / "report.json").write_text('{"secret": true}')
    return Env(root=tmp_path, scenarios=scenarios, runs=runs, skill=skill, outside=outside)


@pytest.fixture
def client(env: Env) -> Iterator[TestClient]:
    app = create_app(env.settings(), token=TOKEN)
    with TestClient(app, base_url=BASE) as c:
        yield c


def _post(client: TestClient, path: str, body: Any, **headers: str) -> Any:
    return client.post(path, json=body, headers={TOKEN_HEADER: TOKEN, **headers})


# --- page -------------------------------------------------------------------------------------


def test_index_embeds_token_and_ships_security_headers(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert f'<meta name="mcpsim-token" content="{TOKEN}">' in res.text
    assert "__MCPSIM_TOKEN__" not in res.text
    csp = res.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp
    assert res.headers["x-content-type-options"] == "nosniff"
    for landmark in ('id="scenario-list"', 'id="detail"', 'id="settings"', 'id="jobs"'):
        assert landmark in res.text
    assert '<script src="/app.js" defer></script>' in res.text


def test_static_assets_are_served_and_render_text_safely(client: TestClient) -> None:
    js = client.get("/app.js")
    css = client.get("/app.css")
    assert js.status_code == 200 and js.headers["content-type"].startswith("text/javascript")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert "prefers-color-scheme: dark" in css.text
    # All API text reaches the DOM through textContent / text nodes.
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in js.text
    assert "X-MCPSim-Token" in js.text
    assert client.get("/index.html").status_code == 404
    assert client.get("/static/app.js").status_code == 404


def test_token_is_random_per_process(env: Env) -> None:
    pages = []
    for _ in range(2):
        with TestClient(create_app(env.settings()), base_url=BASE) as c:
            pages.append(c.get("/").text)
    assert pages[0] != pages[1]


# --- config -----------------------------------------------------------------------------------


def test_config_resolves_roles_with_their_sources(client: TestClient, env: Env) -> None:
    data = client.get("/api/config").json()
    roles = {r["role"]: r for r in data["roles"]}
    assert list(roles) == ["planner", "planner-local", "agent", "user", "observer", "judge"]
    # config.yaml defaults beat the planner's frontmatter (claude-opus-5-5).
    assert (roles["planner"]["spec"], roles["planner"]["source"]) == (
        "anthropic:claude-sonnet-5-5",
        "config.yaml defaults",
    )
    assert roles["planner"]["max_tokens"] == 8192
    # The frontmatter edits show, with the file that set them.
    assert (roles["agent"]["model"], roles["agent"]["source"]) == (
        "claude-opus-5-5",
        "roles/agent.md",
    )
    assert roles["agent"]["max_tokens"] == 2048 and roles["agent"]["temperature"] is None
    assert (roles["user"]["spec"], roles["user"]["source"]) == (
        "anthropic:claude-haiku-4-5-20251001",
        "roles/user.md",
    )
    assert roles["user"]["temperature"] == 0.2
    assert (roles["observer"]["spec"], roles["observer"]["source"]) == (
        "anthropic:claude-sonnet-5-5",
        "roles/observer.md",
    )
    assert roles["judge"]["extra"] == {"votes": 3}
    assert roles["judge"]["file"] == str(env.skill.resolve() / "roles" / "judge.md")
    assert roles["judge"]["prompts"] == ["system", "user"]
    # The local planner only serves an ollama: planner; it shows its own frontmatter.
    local = roles["planner-local"]
    assert (local["provider"], local["model"]) == ("ollama", "command-r7b")
    assert "ollama" in local["source"]
    assert data["run"] == {
        "repeat": 3,
        "modes": ["guided", "free"],
        "judge_votes": 3,
        "concurrency": 4,
    }
    assert data["run_sources"]["repeat"] == "config.yaml run"
    assert data["run_sources"]["concurrency"] == "built-in default"
    skill = data["skill"]
    assert (skill["name"], skill["path"]) == ("simulate", str(env.skill.resolve()))
    assert skill["config_file"] == str(env.skill.resolve() / "config.yaml")
    assert skill["error"] is None
    assert skill["overrides"] == [
        {
            "match": {"category": "Shop*"},
            "models": {"agent": "claude-haiku-4-5-20251001"},
            "run": {"modes": ["guided"]},
        }
    ]
    assert data["scenario_sources"] == ["scenarios", "scenarios/nested"]
    assert data["scenario_sources_from"] == "config.yaml"
    assert data["runs_dir"] == str(env.runs)
    assert data["honesty_item"] == HONESTY_ITEM
    assert "claude-fable-5-1" in data["known_models"]
    assert data["model_roles"] == ["planner", "agent", "judge", "user", "observer"]
    assert data["warnings"] == []


def test_config_follows_skill_edits_and_survives_a_broken_one(
    client: TestClient, env: Env
) -> None:
    config = env.skill / "config.yaml"
    good = config.read_text()
    config.write_text(good.replace("claude-sonnet-5-5", "claude-fable-5-1"))
    roles = {r["role"]: r for r in client.get("/api/config").json()["roles"]}
    assert roles["planner"]["model"] == "claude-fable-5-1"
    # A broken edit: the last skill that loaded is shown, with the loader's error.
    config.write_text(good + "\nnot_a_key: 1\n")
    data = client.get("/api/config").json()
    assert data["skill"]["error"] and "not_a_key" in data["skill"]["error"]
    assert any("no longer loads" in w for w in data["warnings"])
    assert {r["role"]: r for r in data["roles"]}["planner"]["model"] == "claude-fable-5-1"
    config.write_text(good)
    data = client.get("/api/config").json()
    assert data["skill"]["error"] is None and data["warnings"] == []
    # A config.yaml source that matches nothing is a warning, and the list follows the edit.
    config.write_text(good.replace("- scenarios/nested", "- scenarios/nowhere"))
    data = client.get("/api/config").json()
    assert any("scenarios/nowhere" in w and "not found" in w for w in data["warnings"])
    names = {s["name"] for s in client.get("/api/scenarios").json()["scenarios"]}
    assert names == {"fake-lookup", "broken-one"}


# --- scenarios --------------------------------------------------------------------------------


def test_scenarios_grouped_with_status_and_last_run(client: TestClient) -> None:
    data = client.get("/api/scenarios").json()
    by_name = {s["name"]: s for s in data["scenarios"]}
    assert set(by_name) == {"fake-lookup", "v2-shopper", "broken-one"}

    lookup = by_name["fake-lookup"]
    assert lookup["title"] == "Fake lookup" and lookup["category"] == "Uncategorized"
    # The plan-only directory is newer but has no verdicts; the newest JUDGED run counts.
    assert lookup["last_run"]["run_id"] == "20250902T100000Z"
    assert lookup["status"] == "passed"
    assert lookup["pass_rate"] == 1.0
    assert lookup["pass_k"] == {"k": 2, "all_passed": True}
    assert lookup["run_count"] == 3
    assert lookup["user_instructions"].startswith(
        "You are this person: A shopper who wants the price"
    )

    shopper = by_name["v2-shopper"]
    assert (shopper["title"], shopper["category"]) == ("Weekly shop on a phone", "Shopping")
    # Its only run directory has verdicts but no report.json: it never finished, so it is not
    # the scenario's result (no status, no pass^k), only a pointer to an incomplete run.
    assert shopper["status"] == "never" and shopper["last_run"] is None
    assert shopper["pass_k"] is None
    assert shopper["latest_incomplete"] == "20250905T120000Z" and shopper["run_count"] == 1
    assert lookup["latest_incomplete"] == "20250903T100000Z", "the newer plan-only directory"
    assert shopper["user_instructions"].startswith("You are a parent in Lyon")

    broken = by_name["broken-one"]
    assert broken["error"] and "goal" in broken["error"]
    assert broken["status"] == "never" and broken["last_run"] is None

    groups = {g["name"]: g for g in data["categories"]}
    assert groups["Shopping"]["total"] == 1 and groups["Shopping"]["counts"]["never"] == 1
    assert groups["Uncategorized"]["total"] == 2
    assert groups["Uncategorized"]["counts"] == {
        "passed": 1,
        "failed": 0,
        "partial": 0,
        "running": 0,
        "never": 1,
    }


def test_scenario_detail_reads_v2_fields_and_effective_models(client: TestClient) -> None:
    data = client.get("/api/scenarios/v2-shopper").json()
    s = data["scenario"]
    assert s["context"] == {
        "device": "mobile",
        "location": "Lyon, FR",
        "language": "fr-FR",
        "details": {"budget": "20 EUR"},
        "agent_visible": True,
    }
    assert s["expected_behavior"] == V2_EXTRA["expected_behavior"]
    assert s["expected_behavior_derived"] is False and s["user_instructions_derived"] is False
    assert s["agent_skill"] == "env:RECIPE_SHOPPER_SKILL"
    assert s["agent_notes"] == "No write tools in this setup."
    # The model read the SOP through the environment variable and stripped its frontmatter.
    assert s["agent_skill_name"] == "shop-sop"
    assert s["agent_skill_text"] == "1. Look the product up before quoting it."
    assert s["file"] == os.path.join("scenarios", "nested", "v2-shopper.yaml")
    assert s["models"] == {"judge": "claude-fable-5-1"}, "what the file names itself"
    models = data["models"]
    # Category override (Shop*) beats the agent's frontmatter; the scenario's judge beats both.
    assert (models["agent"]["model"], models["agent"]["source"]) == (
        "claude-haiku-4-5-20251001",
        "config.yaml overrides[1] (category=Shop*)",
    )
    assert (models["judge"]["model"], models["judge"]["source"]) == (
        "claude-fable-5-1",
        "scenario file",
    )
    assert (models["user"]["spec"], models["user"]["source"]) == (
        "anthropic:claude-haiku-4-5-20251001",
        "roles/user.md",
    )
    assert models["user"]["temperature"] == 0.2
    assert (models["planner"]["model"], models["planner"]["source"]) == (
        "claude-sonnet-5-5",
        "config.yaml defaults",
    )
    assert models["planner"]["file"].endswith("planner.md")
    run = data["run_settings"]
    override = "config.yaml overrides[1] (category=Shop*)"
    assert run["modes"] == {"value": ["guided"], "source": override}
    assert run["repeat"] == {"value": 2, "source": "scenario file"}
    assert run["judge_votes"] == {"value": 3, "source": "config.yaml run"}
    assert data["warnings"] == []

    v1_data = client.get("/api/scenarios/fake-lookup").json()
    v1 = v1_data["scenario"]
    assert v1["expected_behavior"] == v1["instructions"] and v1["expected_behavior_derived"]
    assert v1["context"]["device"] == "" and v1["context"]["agent_visible"] is False
    assert v1["models"] == {}
    assert v1_data["models"]["agent"]["source"] == "roles/agent.md"
    assert v1_data["run_settings"]["modes"]["value"] == ["guided", "free"]


def test_a_scenario_the_skill_cannot_run_says_why(client: TestClient, env: Env) -> None:
    # A temperature on the agent role: fine for Haiku (the Shop* override), refused for the
    # Opus 5.5 agent every other scenario resolves to, as `mcpsim run` would refuse it.
    set_frontmatter(env.skill / "roles" / "agent.md", temperature=0.5)
    lookup = client.get("/api/scenarios/fake-lookup").json()
    assert lookup["models"]["agent"]["temperature"] == 0.5
    [warning] = lookup["warnings"]
    assert "will not run" in warning and "rejects it" in warning
    assert client.get("/api/scenarios/v2-shopper").json()["warnings"] == []


def test_an_unset_sop_variable_makes_the_scenario_invalid(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RECIPE_SHOPPER_SKILL")
    listing = {s["name"]: s for s in client.get("/api/scenarios").json()["scenarios"]}
    shopper = listing["v2-shopper"]
    assert "RECIPE_SHOPPER_SKILL is not set" in shopper["error"]
    # The list still places it (title and category from the raw file), and it cannot run.
    assert (shopper["title"], shopper["category"]) == ("Weekly shop on a phone", "Shopping")
    detail = client.get("/api/scenarios/v2-shopper").json()
    assert detail["models"] is None and detail["run_settings"] is None
    refused = _post(client, "/api/run", {"scenarios": ["v2-shopper"]})
    assert refused.status_code == 400 and "do not validate" in refused.json()["error"]


def test_scenario_edits_show_without_a_restart(client: TestClient, env: Env) -> None:
    assert client.get("/api/scenarios/v2-shopper").json()["scenario"]["title"] == (
        "Weekly shop on a phone"
    )
    path = env.scenarios / "nested" / "v2-shopper.yaml"
    path.write_text(yaml.safe_dump(_scenario("v2-shopper", **{**V2_EXTRA, "title": "Renamed"})))
    assert client.get("/api/scenarios/v2-shopper").json()["scenario"]["title"] == "Renamed"
    listing = {s["name"]: s for s in client.get("/api/scenarios").json()["scenarios"]}
    assert listing["v2-shopper"]["title"] == "Renamed"
    # A file that stops validating shows its error; one that disappears is unknown.
    broken = _scenario("v2-shopper", **V2_EXTRA)
    del broken["role"]
    path.write_text(yaml.safe_dump(broken))
    assert "role" in client.get("/api/scenarios/v2-shopper").json()["scenario"]["error"]
    path.unlink()
    assert client.get("/api/scenarios/v2-shopper").status_code == 404


def test_scenario_runs_history_newest_first(client: TestClient) -> None:
    runs = client.get("/api/scenarios/fake-lookup/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == [
        "20250903T100000Z",
        "20250902T100000Z",
        "20250901T100000Z",
    ]
    plan_only, newest, oldest = runs
    assert plan_only["status"] == "incomplete" and plan_only["runs"] == 0
    assert plan_only["has_plan"] and not plan_only["has_report"]
    assert newest["status"] == "passed" and newest["duration_s"] == 42.5
    assert oldest["status"] == "partial" and oldest["passed"] == 1 and oldest["runs"] == 2
    # Report without duration_s: generated_at minus the run id's timestamp.
    assert oldest["duration_s"] == 8.0
    assert oldest["pass_k"] == {"k": 2, "all_passed": False, "computed": True}
    assert oldest["started_at"] == "2025-09-01T10:00:00+00:00"
    assert oldest["cost_usd"] == 0.0246
    assert oldest["models"] == {}


def test_run_detail_has_plan_verdicts_transcripts(client: TestClient) -> None:
    data = client.get("/api/runs/fake-lookup/20250901T100000Z").json()
    assert data["summary"]["status"] == "partial"
    assert data["plan"]["paths"][0]["id"] == "happy"
    assert set(data["verdicts"]) == {"happy-guided-0", "happy-guided-1"}
    rows = {t["stem"]: t for t in data["transcripts"]}
    assert rows["happy-guided-1"]["passed"] is False
    assert rows["happy-guided-0"]["file"] == "happy-guided-0.jsonl"
    assert rows["happy-guided-0"]["outcome"] == "completed"
    assert rows["happy-guided-0"]["cost_usd"] == 0.0123
    assert rows["happy-guided-0"]["duration_s"] == 4.5
    assert data["scenario"]["name"] == "fake-lookup"
    assert data["scenario"]["error"] is None
    assert data["has_scout"] is False

    shopper = client.get("/api/runs/v2-shopper/20250905T120000Z").json()
    summary = shopper["summary"]
    # No report.json: what is on disk is shown, but the run is incomplete, never partial or
    # passed, and pass^k's k is the repeat scenario.json records (2), not the cells' 1 run.
    assert (summary["status"], summary["runs"], summary["passed"]) == ("incomplete", 2, 1)
    assert summary["finished"] is False and summary["repeat"] == 2
    assert summary["pass_k"] == {"k": 2, "all_passed": False, "computed": True}
    # No report.json: cost and duration come from the transcripts themselves.
    assert summary["cost_usd"] == 0.0246 and summary["duration_s"] == 4.5
    assert shopper["scenario"]["context"]["location"] == "Lyon, FR"
    assert shopper["scenario"]["agent_skill_text"] == "1. Look the product up before quoting it."
    row = next(t for t in shopper["transcripts"] if t["stem"] == "happy-guided-0")
    assert (row["goal_achieved"], row["sop_followed"]) == (True, False)


def test_transcript_events_verdict_and_facts(client: TestClient) -> None:
    res = client.get("/api/runs/fake-lookup/20250901T100000Z/transcripts/happy-guided-0.jsonl")
    assert res.status_code == 200
    data = res.json()
    kinds = [e["kind"] for e in data["events"]]
    assert kinds[:3] == ["system", "tools_offered", "unparsed"]
    assert data["events"][2]["raw"] == "this is not json"
    assert "tool_call" in kinds and "informant_report" in kinds and kinds[-1] == "end"
    # Content is returned as data, untouched; escaping is the page's job (textContent).
    assert data["events"][-4]["text"] == f"Penne at {HOSTILE}"
    assert data["verdict"]["passed"] is True
    assert data["facts"]["flags"] == ["fabrication"]
    assert data["facts"]["hard_failures"] == ["shelf.fabrication — quoted a store"]


# --- refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/scenarios/nope",
        "/api/scenarios/..",
        "/api/scenarios/nope/runs",
        "/api/runs/nope/20250901T100000Z",
        "/api/runs/fake-lookup/29990101T000000Z",
        "/api/runs/fake-lookup/..",
        "/api/runs/fake-lookup/%2e%2e",
        "/api/runs/..%2f..%2foutside/x",
        "/api/runs/fake-lookup/..%2F..%2Foutside",
        "/api/runs/fake-lookup/20250901T100000Z/transcripts/nope.jsonl",
        "/api/runs/fake-lookup/20250901T100000Z/transcripts/..%2Freport.json",
        "/api/runs/fake-lookup/20250901T100000Z/transcripts/..%2F..%2F..%2F..%2Foutside%2Fsecret.jsonl",
        "/api/runs/fake-lookup/20250901T100000Z/transcripts/happy-guided-0.json",
        "/api/runs/fake-lookup/20250901T100000Z/transcripts/.hidden.jsonl",
        "/api/jobs/does-not-exist",
    ],
)
def test_unknown_ids_and_traversal_are_404(client: TestClient, path: str) -> None:
    res = client.get(path)
    assert res.status_code == 404, path
    assert "SECRET" not in res.text and '"secret"' not in res.text


def test_symlinks_out_of_the_runs_dir_are_not_followed(client: TestClient, env: Env) -> None:
    run = env.runs / "fake-lookup" / "20250901T100000Z"
    (run / "transcripts" / "leak.jsonl").symlink_to(env.outside / "secret.jsonl")
    (env.runs / "fake-lookup" / "20250904T000000Z").symlink_to(env.outside)
    (env.runs / "v2-shopper" / "20250906T000000Z").mkdir()
    (env.runs / "v2-shopper" / "20250906T000000Z" / "report.json").symlink_to(
        env.outside / "report.json"
    )

    res = client.get("/api/runs/fake-lookup/20250901T100000Z/transcripts/leak.jsonl")
    assert res.status_code == 404
    detail = client.get("/api/runs/fake-lookup/20250901T100000Z").json()
    assert "leak" not in {t["stem"] for t in detail["transcripts"]}
    ids = [r["run_id"] for r in client.get("/api/scenarios/fake-lookup/runs").json()["runs"]]
    assert "20250904T000000Z" not in ids
    assert client.get("/api/runs/fake-lookup/20250904T000000Z").status_code == 404
    linked = client.get("/api/runs/v2-shopper/20250906T000000Z").json()
    assert linked["report"] is None and linked["summary"]["has_report"] is False


def test_non_loopback_host_header_is_refused(env: Env) -> None:
    app = create_app(env.settings(), token=TOKEN)
    with TestClient(app, base_url="http://evil.example:8765") as c:
        assert c.get("/api/scenarios").status_code == 421
        assert c.get("/").status_code == 421
    for base in ("http://localhost:8765", "http://[::1]:8765", "http://127.0.0.1"):
        with TestClient(app, base_url=base) as c:
            assert c.get("/api/config").status_code == 200, base
    remote = create_app(env.settings(allow_remote=True), token=TOKEN)
    with TestClient(remote, base_url="http://10.0.0.5:8765") as c:
        assert c.get("/api/config").status_code == 200


def test_config_warns_about_a_temperature_the_resolved_model_rejects(env: Env) -> None:
    set_frontmatter(env.skill / "roles" / "judge.md", temperature=0.3)
    app = create_app(env.settings(), token=TOKEN)
    with TestClient(app, base_url=BASE) as c:
        warnings = c.get("/api/config").json()["warnings"]
    assert any(
        "judge.md: temperature 0.3 is set" in w and "claude-opus-5-5, which rejects it" in w
        for w in warnings
    ), warnings


def test_allow_remote_still_refuses_a_dns_rebinding_host(env: Env) -> None:
    """Under --allow-remote a page that rebinds its own domain to this machine sends that
    domain as Host: it can neither load the page (and its token) nor read or post."""
    settings = env.settings(allow_remote=True, allow_hosts=["runner.team.example"])
    app = create_app(settings, token=TOKEN)
    rebound = "rebind.attacker.test:8765"
    with TestClient(app, base_url=f"http://{rebound}") as c:
        page = c.get("/")
        assert page.status_code == 421 and TOKEN not in page.text
        assert c.get("/api/scenarios").status_code == 421
        res = _post(c, "/api/run", {"scenarios": ["fake-lookup"]}, origin=f"http://{rebound}")
        assert res.status_code == 421
    # IP addresses, this machine's name and --allow-host names are still answered.
    hostname = socket.gethostname().lower()
    for base in (
        "http://10.0.0.5:8765",
        "http://[fd00::5]:8765",
        f"http://{hostname}:8765",
        "http://runner.team.example:8765",
        "http://RUNNER.team.example.:8765",
        "http://localhost:8765",
    ):
        with TestClient(app, base_url=base) as c:
            assert c.get("/api/config").status_code == 200, base


def test_host_allowed_rules() -> None:
    names = remote_host_names("runner.lan", ["extra.example"])
    assert {"runner.lan", "runner", "extra.example", "extra"} <= names
    assert socket.gethostname().lower() in names
    assert "0.0.0.0" not in remote_host_names("0.0.0.0")
    assert host_allowed("127.0.0.1", allow_remote=False)
    assert not host_allowed("10.0.0.5", allow_remote=False)
    assert host_allowed("10.0.0.5", allow_remote=True)
    assert host_allowed("Extra.Example", allow_remote=True, names=names)
    for bad in ("rebind.attacker.test", "runner.lan.attacker.test", "", "extra.example.evil"):
        assert not host_allowed(bad, allow_remote=True, names=names), bad


def test_cli_ui_allow_host_needs_allow_remote_and_reaches_the_settings(
    env: Env, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(env.root)
    assert main(["ui", "--skill", str(env.skill), "--allow-host", "x.example"]) == EXIT_USAGE
    assert "--allow-host only applies with --allow-remote" in capsys.readouterr().err
    seen: dict[str, Any] = {}

    def fake_serve(settings: UISettings, *, host: str, port: int) -> None:
        seen.update(settings=settings, host=host)

    monkeypatch.setattr("mcpsim.ui.app.serve", fake_serve)
    args = ["ui", "--skill", str(env.skill), "--host", "0.0.0.0", "--allow-remote"]
    assert main([*args, "--allow-host", "a.example", "--allow-host", "b.example"]) == 0
    assert seen["settings"].allow_hosts == ["a.example", "b.example"]
    assert seen["settings"].bind_host == "0.0.0.0"


def test_serve_warns_about_allow_remote(env: Env, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    serve(env.settings(allow_remote=True, allow_hosts=["team.example"]), host="0.0.0.0", port=1)
    err = capsys.readouterr().err
    assert "warning: --allow-remote" in err and "team.example" in err and "421" in err


def test_a_run_snapshot_never_makes_the_server_read_another_file(
    client: TestClient, env: Env
) -> None:
    """scenario.json is data: an agent.skill without skill_text (edited, or copied from a
    teammate) is shown as a name, and the file it points at is never read or served."""
    secret = env.outside / "secret.txt"
    secret.write_text("TOP-SECRET-OUTSIDE-RUNS-DIR", encoding="utf-8")
    snapshot = {**_scenario("fake-lookup"), "agent": {"skill": str(secret)}}
    run = env.runs / "fake-lookup" / "20250907T000000Z"
    _write_run(run, verdicts=[_verdict("happy", "guided", 0, True)], report=None,
               scenario=snapshot)
    for skill in (str(secret), "../../../outside/secret.txt", "env:RECIPE_SHOPPER_SKILL"):
        snapshot["agent"] = {"skill": skill}
        (run / "scenario.json").write_text(json.dumps(snapshot), encoding="utf-8")
        res = client.get("/api/runs/fake-lookup/20250907T000000Z")
        assert res.status_code == 200
        assert "TOP-SECRET" not in res.text and "Look the product up" not in res.text, skill
        view = res.json()["scenario"]
        assert view["agent_skill"] == skill and view["error"] is None
        assert view["agent_skill_text"] is None and view["agent_skill_path"] is None


def test_post_requires_the_token(client: TestClient) -> None:
    body = {"scenarios": ["fake-lookup"]}
    assert client.post("/api/run", json=body).status_code == 403
    assert client.post("/api/run", json=body, headers={TOKEN_HEADER: "wrong"}).status_code == 403
    assert client.post("/api/run", json=body, headers={TOKEN_HEADER: ""}).status_code == 403
    assert client.post("/api/jobs/abc/cancel").status_code == 403
    foreign = _post(client, "/api/run", body, origin="http://evil.example")
    assert foreign.status_code == 403
    assert client.get("/api/run").status_code == 405


@pytest.mark.parametrize(
    ("body", "status", "needle"),
    [
        ([1, 2], 400, "JSON object"),
        ({"scenarios": ["fake-lookup"], "extra": 1}, 400, "unknown field"),
        ({"scenarios": []}, 400, "scenarios must be"),
        ({"scenarios": "some"}, 400, "scenarios must be"),
        ({"scenarios": [1]}, 400, "scenarios must be"),
        ({"scenarios": ["nope"]}, 404, "unknown scenario"),
        ({"scenarios": ["../fake-lookup"]}, 404, "unknown scenario"),
        ({"scenarios": ["broken-one"]}, 400, "do not validate"),
        ({"scenarios": ["fake-lookup"], "models": {"pilot": "x"}}, 400, "unknown role"),
        ({"scenarios": ["fake-lookup"], "models": {"agent": "--evil"}}, 400, "not a model spec"),
        ({"scenarios": ["fake-lookup"], "models": {"agent": "a b"}}, 400, "not a model spec"),
        ({"scenarios": ["fake-lookup"], "models": "agent=x"}, 400, "models must be"),
        ({"scenarios": ["fake-lookup"], "repeat": 0}, 400, "repeat"),
        ({"scenarios": ["fake-lookup"], "repeat": True}, 400, "repeat"),
        ({"scenarios": ["fake-lookup"], "repeat": "2"}, 400, "repeat"),
        ({"scenarios": ["fake-lookup"], "modes": ["sideways"]}, 400, "unknown mode"),
        ({"scenarios": ["fake-lookup"], "modes": []}, 400, "modes must be"),
        ({"scenarios": ["fake-lookup"], "dry_run": "yes"}, 400, "dry_run"),
    ],
)
def test_run_request_validation(client: TestClient, body: Any, status: int, needle: str) -> None:
    res = _post(client, "/api/run", body)
    assert res.status_code == status, res.text
    assert needle in res.json()["error"]


def test_run_request_body_size_and_json(client: TestClient) -> None:
    big = {"scenarios": ["fake-lookup"], "pad": "x" * (70 * 1024)}
    assert _post(client, "/api/run", big).status_code == 413
    bad = client.post(
        "/api/run",
        content=b"{not json",
        headers={TOKEN_HEADER: TOKEN, "content-type": "application/json"},
    )
    assert bad.status_code == 400


# --- jobs -------------------------------------------------------------------------------------


@pytest.fixture
def job_client(env: Env, request: pytest.FixtureRequest) -> Iterator[tuple[TestClient, Env]]:
    plan = getattr(request, "param", {})
    jobs = JobManager(
        runs_dir=env.runs.resolve(),
        skill_dir=env.skill,
        command=[sys.executable, str(FAKE_MCPSIM)],
        env={"FAKE_MCPSIM_PLAN": json.dumps(plan), "MCPSIM_SKILL": str(env.skill)},
        cwd=env.root,
        output_drain_s=0.5,
    )
    app = create_app(env.settings(), token=TOKEN, jobs=jobs)
    with TestClient(app, base_url=BASE) as c:
        yield c, env
    jobs.shutdown()


def _wait(client: TestClient, job_id: str, timeout: float = 30.0) -> dict[str, Any]:
    ui = client.app.state.ui  # type: ignore[attr-defined]
    state = ui.jobs.wait(job_id, timeout=timeout)
    assert state is not None
    return client.get(f"/api/jobs/{job_id}").json()


@pytest.mark.parametrize(
    "job_client", [{"fake-lookup": "pass", "v2-shopper": "partial"}], indirect=True
)
def test_job_lifecycle_runs_each_scenario_and_records_status(
    job_client: tuple[TestClient, Env],
) -> None:
    client, env = job_client
    res = _post(
        client,
        "/api/run",
        {
            "scenarios": ["v2-shopper", "fake-lookup"],
            "models": {"agent": "claude-haiku-4-5-20251001", "judge": "anthropic:claude-opus-5-5"},
            "repeat": 2,
            "modes": ["guided"],
            "dry_run": True,
        },
    )
    assert res.status_code == 202, res.text
    job_id = res.json()["job_id"]
    assert res.json()["scenarios"] == ["v2-shopper", "fake-lookup"]
    first = client.get(f"/api/jobs/{job_id}").json()
    assert first["status"] in ("queued", "running", "done")

    done = _wait(client, job_id)
    assert done["status"] == "done"
    tasks = {t["name"]: t for t in done["scenarios"]}
    assert tasks["fake-lookup"]["status"] == "passed" and tasks["fake-lookup"]["exit_code"] == 0
    assert tasks["v2-shopper"]["status"] == "partial" and tasks["v2-shopper"]["exit_code"] == 1
    assert all(t["run_id"] and t["finished_at"] for t in tasks.values())
    log = "\n".join(done["log_tail"])
    assert "[v2-shopper] mcpsim: planning v2-shopper" in log
    assert f"[fake-lookup] mcpsim: skill={env.skill} env={env.skill}" in log
    assert "[fake-lookup] exit 0: passed" in log
    assert done["options"]["repeat"] == 2 and done["options"]["dry_run"] is True

    run_id = tasks["fake-lookup"]["run_id"]
    argv = json.loads((env.runs / "fake-lookup" / run_id / "argv.json").read_text())
    assert argv[0] == "run" and argv[1].endswith("fake-lookup.yaml")
    assert argv[2:4] == ["--skill", str(env.skill)]
    assert argv[argv.index("--out") + 1] == str(env.runs.resolve())
    assert argv[argv.index("--models") + 1] == (
        "agent=claude-haiku-4-5-20251001,judge=anthropic:claude-opus-5-5"
    )
    assert argv[argv.index("--repeat") + 1] == "2"
    assert argv[argv.index("--modes") + 1] == "guided"
    assert "--dry-run" in argv and "--allow-same-judge" not in argv

    listing = {s["name"]: s for s in client.get("/api/scenarios").json()["scenarios"]}
    assert listing["fake-lookup"]["last_run"]["run_id"] == run_id
    assert listing["fake-lookup"]["status"] == "passed"
    assert listing["v2-shopper"]["status"] == "partial"
    assert listing["v2-shopper"]["pass_k"] == {"k": 2, "all_passed": False}
    history = client.get("/api/scenarios/fake-lookup/runs").json()["runs"]
    assert history[0]["run_id"] == run_id and history[0]["duration_s"] == 1.5
    assert client.get("/api/jobs").json()["jobs"][0]["job_id"] == job_id


def _finished_failing_run(env: Env) -> str:
    """A complete, failing repeat-2 run of fake-lookup, newer than the fixture's passing one."""
    run_id = "20250904T000000Z"
    _write_run(
        env.runs / "fake-lookup" / run_id,
        verdicts=[_verdict("happy", "guided", 0, False), _verdict("happy", "guided", 1, False)],
        report={
            "scenario": "fake-lookup",
            "runs": 2,
            "passed": 0,
            "pass_rate": 0.0,
            "pass_k": {"k": 2, "all_passed": False},
            "judge_models": ["claude-opus-5-5"],
        },
        scenario=_scenario("fake-lookup"),
        plan=PLAN,
    )
    return run_id


@pytest.mark.parametrize("job_client", [{"fake-lookup": "stall"}], indirect=True)
def test_a_cancelled_rerun_with_a_passing_verdict_does_not_replace_the_failing_result(
    job_client: tuple[TestClient, Env],
) -> None:
    """The review's case: the last complete run failed; a re-run is stopped after its first
    (passing) verdict. The list keeps the failure, and the stopped run is incomplete with
    pass^3 (the repeat it asked for) not held, never "passed" with pass^1."""
    client, env = job_client
    finished = _finished_failing_run(env)
    listing = {s["name"]: s for s in client.get("/api/scenarios").json()["scenarios"]}
    assert listing["fake-lookup"]["status"] == "failed"
    job_id = _post(client, "/api/run", {"scenarios": ["fake-lookup"]}).json()["job_id"]

    def stopped() -> list[Path]:
        return [
            d
            for d in (env.runs / "fake-lookup").iterdir()
            if (d / "verdicts" / "happy-guided-0.json").is_file()
            and not (d / "report.json").is_file()
        ]

    for _ in range(500):
        if stopped():
            break
        time.sleep(0.02)
    [stopped_dir] = stopped()
    assert _post(client, f"/api/jobs/{job_id}/cancel", {}).status_code == 200
    done = _wait(client, job_id)
    assert done["scenarios"][0]["status"] == "cancelled"

    listing = {s["name"]: s for s in client.get("/api/scenarios").json()["scenarios"]}
    row = listing["fake-lookup"]
    assert row["status"] == "failed"
    assert row["last_run"]["run_id"] == finished
    assert row["pass_k"] == {"k": 2, "all_passed": False}
    assert row["latest_incomplete"] == stopped_dir.name
    detail = client.get("/api/scenarios/fake-lookup").json()
    assert detail["status"] == "failed" and detail["latest_incomplete"] == stopped_dir.name
    history = client.get("/api/scenarios/fake-lookup/runs").json()["runs"]
    assert history[0]["run_id"] == stopped_dir.name
    assert (history[0]["status"], history[0]["runs"], history[0]["passed"]) == (
        "incomplete",
        1,
        1,
    )
    assert history[0]["pass_k"] == {"k": 3, "all_passed": False, "computed": True}


def test_a_crashed_run_or_a_report_listing_gaps_is_incomplete_never_passed(env: Env) -> None:
    """Without the job manager: a crashed run (passing verdicts, no report.json) and a report
    rebuilt over a partial directory (it lists unjudged / missing runs) are both incomplete;
    the scenario's result stays the newest finished run."""
    finished = _finished_failing_run(env)
    crashed = env.runs / "fake-lookup" / "20250906T000000Z"
    _write_run(
        crashed,
        verdicts=[_verdict("happy", "guided", 0, True)],
        report=None,
        scenario={**_scenario("fake-lookup"), "repeat": 3},
    )
    rebuilt = env.runs / "fake-lookup" / "20250905T000000Z"
    _write_run(
        rebuilt,
        verdicts=[_verdict("happy", "guided", 0, True)],
        report={
            "scenario": "fake-lookup",
            "runs": 1,
            "passed": 1,
            "pass_rate": 1.0,
            "pass_k": {"k": 3, "all_passed": False},
            "unjudged": [],
            "missing": ["happy-guided-1", "happy-guided-2"],
        },
        scenario={**_scenario("fake-lookup"), "repeat": 3},
    )
    app = create_app(env.settings(), token=TOKEN)
    with TestClient(app, base_url=BASE) as client:
        row = {s["name"]: s for s in client.get("/api/scenarios").json()["scenarios"]}[
            "fake-lookup"
        ]
        assert (row["status"], row["last_run"]["run_id"]) == ("failed", finished)
        assert row["latest_incomplete"] == crashed.name
        runs = client.get("/api/scenarios/fake-lookup/runs").json()["runs"]
        history = {r["run_id"]: r for r in runs}
        for run in (crashed.name, rebuilt.name):
            assert history[run]["status"] == "incomplete", run
            assert history[run]["pass_k"]["k"] == 3, run
            assert history[run]["pass_k"]["all_passed"] is False, run
        assert history[finished]["status"] == "failed" and history[finished]["finished"] is True


@pytest.mark.parametrize("job_client", [{"fake-lookup": "error"}], indirect=True)
def test_job_marks_a_cli_error_and_fails(job_client: tuple[TestClient, Env]) -> None:
    client, _ = job_client
    job_id = _post(client, "/api/run", {"scenarios": ["fake-lookup"]}).json()["job_id"]
    done = _wait(client, job_id)
    assert done["status"] == "failed"
    task = done["scenarios"][0]
    assert (task["status"], task["exit_code"], task["run_id"]) == ("error", 2, None)
    assert any("cannot connect to server" in line for line in done["log_tail"])


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.parametrize("job_client", [{"fake-lookup": "sleep"}], indirect=True)
def test_running_scenario_conflicts_and_can_be_cancelled(
    job_client: tuple[TestClient, Env],
) -> None:
    client, env = job_client
    job_id = _post(client, "/api/run", {"scenarios": ["fake-lookup"]}).json()["job_id"]
    ui = client.app.state.ui  # type: ignore[attr-defined]
    child_file = env.runs / "fake-lookup.child"
    for _ in range(400):
        state = ui.jobs.get(job_id)
        if (
            state["scenarios"][0]["status"] == "running"
            and state["log_lines"] >= 3
            and child_file.is_file()
        ):
            break
        time.sleep(0.02)
    listing = {s["name"]: s for s in client.get("/api/scenarios").json()["scenarios"]}
    assert listing["fake-lookup"]["status"] == "running"
    # The directory being written is not the last run yet: the newest JUDGED run still is.
    assert listing["fake-lookup"]["last_run"]["run_id"] == "20250902T100000Z"
    assert listing["fake-lookup"]["run_count"] == 4
    detail = client.get("/api/scenarios/fake-lookup").json()
    assert detail["status"] == "running" and detail["last_run"]["run_id"] == "20250902T100000Z"
    history = client.get("/api/scenarios/fake-lookup/runs").json()["runs"]
    assert history[0]["status"] == "running" and history[0]["run_id"] > "20250903T100000Z"
    # Older plan-only directories are not mistaken for the running one.
    assert [r["status"] for r in history[1:]] == ["incomplete", "passed", "partial"]
    running = client.get(f"/api/jobs/{job_id}").json()
    assert running["status"] == "running"
    assert any("still running" in line for line in running["log_tail"])

    clash = _post(client, "/api/run", {"scenarios": ["v2-shopper", "fake-lookup"]})
    assert clash.status_code == 409
    assert clash.json()["scenarios"] == ["fake-lookup"]

    grandchild = int(child_file.read_text())
    assert _alive(grandchild)
    assert _post(client, f"/api/jobs/{job_id}/cancel", {}).status_code == 200
    done = _wait(client, job_id)
    assert done["scenarios"][0]["status"] == "cancelled"
    assert done["status"] == "done" and done["cancel_requested"] is True
    # Stopping the job stops the whole process group, not just the CLI process.
    for _ in range(200):
        if not _alive(grandchild):
            break
        time.sleep(0.02)
    assert not _alive(grandchild)
    # Once nothing runs, the cancelled run's directory is an ordinary (incomplete) run.
    after = client.get("/api/scenarios/fake-lookup/runs").json()["runs"]
    assert after[0]["status"] == "incomplete"
    assert client.get("/api/scenarios/fake-lookup").json()["status"] == "passed"
    assert _post(client, "/api/jobs/nope/cancel", {}).status_code == 404


@pytest.mark.parametrize(
    "job_client", [{"fake-lookup": "silent", "v2-shopper": "unjudged"}], indirect=True
)
def test_job_status_needs_judged_results(job_client: tuple[TestClient, Env]) -> None:
    client, env = job_client
    res = _post(client, "/api/run", {"scenarios": ["fake-lookup", "v2-shopper"]})
    done = _wait(client, res.json()["job_id"])
    tasks = {t["name"]: t for t in done["scenarios"]}
    # Exit 0 without naming a run directory is not a pass: there is nothing to show for it.
    assert (tasks["fake-lookup"]["status"], tasks["fake-lookup"]["exit_code"]) == ("error", 0)
    assert tasks["fake-lookup"]["run_id"] is None
    # A run directory with no verdicts did not pass either.
    shopper = tasks["v2-shopper"]
    assert (shopper["status"], shopper["exit_code"]) == ("failed", 1)
    assert shopper["run_id"] and (env.runs / "v2-shopper" / shopper["run_id"]).is_dir()
    assert done["status"] == "failed"


@pytest.mark.parametrize("job_client", [{"fake-lookup": "orphan"}], indirect=True)
def test_job_does_not_wait_for_an_orphan_holding_the_output(
    job_client: tuple[TestClient, Env],
) -> None:
    client, env = job_client
    started = time.monotonic()
    job_id = _post(client, "/api/run", {"scenarios": ["fake-lookup"]}).json()["job_id"]
    try:
        done = _wait(client, job_id, timeout=20)
        elapsed = time.monotonic() - started
    finally:
        orphan = int((env.runs / "fake-lookup.orphan").read_text())
        with contextlib.suppress(ProcessLookupError):
            os.kill(orphan, signal.SIGKILL)
    assert done["status"] == "done", done
    task = done["scenarios"][0]
    assert (task["status"], task["exit_code"]) == ("passed", 0) and task["run_id"]
    assert any("output still open after exit" in line for line in done["log_tail"])
    assert elapsed < 15  # the orphan sleeps 60 s


@pytest.mark.parametrize("job_client", [{}], indirect=True)
def test_run_all_runs_every_valid_scenario(job_client: tuple[TestClient, Env]) -> None:
    client, _ = job_client
    res = _post(client, "/api/run", {"scenarios": "all"})
    assert res.status_code == 202
    assert res.json()["scenarios"] == ["fake-lookup", "v2-shopper"]  # "broken" is skipped
    done = _wait(client, res.json()["job_id"])
    assert [t["status"] for t in done["scenarios"]] == ["passed", "passed"]


# --- options, commands, binding ---------------------------------------------------------------


def test_build_command_maps_options_to_cli_flags(tmp_path: Path) -> None:
    prefix = ["python", "-m", "mcpsim.cli"]
    file, out = tmp_path / "s.yaml", tmp_path / "runs"
    assert build_command(prefix, file, out, RunOptions()) == [
        *prefix,
        "run",
        str(file),
        "--out",
        str(out),
    ]
    skill = tmp_path / "skill"
    assert build_command(prefix, file, out, RunOptions(), skill) == [
        *prefix,
        "run",
        str(file),
        "--skill",
        str(skill),
        "--out",
        str(out),
    ]
    # Both modes asked for are passed on (the command line's layer), so a config.yaml or an
    # override that narrows modes cannot silently drop the free runs the request asked for.
    both = parse_run_options({"modes": ["guided", "free"], "models": {"user": ""}})
    assert both.modes == ["guided", "free"] and both.models == {}
    cmd = build_command(prefix, file, out, both)
    assert cmd[cmd.index("--modes") + 1] == "guided,free" and "--models" not in cmd
    # No modes: the skill and the scenario decide.
    assert "--modes" not in build_command(prefix, file, out, parse_run_options({}))
    opts = parse_run_options(
        {"models": {"planner": "ollama:llama3.2:3b"}, "allow_same_judge": True, "modes": ["free"]}
    )
    cmd = build_command(prefix, file, out, opts)
    assert cmd[-5:] == [
        "--models",
        "planner=ollama:llama3.2:3b",
        "--modes",
        "free",
        "--allow-same-judge",
    ]


def test_requested_modes_reach_the_run_even_when_the_skill_narrows_them(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's repro: config.yaml's override narrows Shopping scenarios to [guided]; an
    API request for both modes must run both (the command line's layer), as the job says."""
    from mcpsim.cli import build_parser
    from mcpsim.runner import prepare_scenario

    monkeypatch.chdir(env.root)
    file = env.scenarios / "nested" / "v2-shopper.yaml"
    opts = parse_run_options({"modes": ["guided", "free"]})
    cmd = build_command(["mcpsim"], file, env.runs, opts, env.skill)
    args = build_parser().parse_args(cmd[1:])
    _, resolved, _ = prepare_scenario(
        args.scenario, skill=args.skill, modes=[args.mode] if args.mode else args.modes
    )
    assert opts.to_json()["modes"] == resolved.run["modes"].value == ["guided", "free"]
    assert resolved.run["modes"].source == "command line"
    # Without a request the override still decides.
    plain = build_parser().parse_args(build_command(["m"], file, env.runs, RunOptions())[1:])
    _, resolved, _ = prepare_scenario(plain.scenario, skill=env.skill, modes=plain.modes)
    assert resolved.run["modes"].value == ["guided"]


def test_loopback_detection() -> None:
    for host in ("127.0.0.1", "127.0.0.2", "::1", "[::1]", "localhost", "LOCALHOST"):
        assert is_loopback(host), host
    for host in ("0.0.0.0", "10.0.0.5", "::", "example.com", "", "localhost.evil.com"):
        assert not is_loopback(host), host
    assert host_name("127.0.0.1:8765") == "127.0.0.1"
    assert host_name("[::1]:8765") == "::1"
    assert host_name("localhost") == "localhost"


def test_serve_refuses_a_non_loopback_bind(env: Env) -> None:
    with pytest.raises(ValueError, match="--allow-remote"):
        serve(env.settings(), host="0.0.0.0", port=0)


def test_cli_ui_refuses_non_loopback_without_allow_remote(
    env: Env, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(env.root)
    code = main(["ui", "--host", "0.0.0.0", "--skill", str(env.skill)])
    assert code == EXIT_USAGE
    assert "--allow-remote" in capsys.readouterr().err


def test_cli_ui_passes_flags_to_serve(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_serve(settings: UISettings, *, host: str, port: int) -> None:
        seen.update(settings=settings, host=host, port=port)

    monkeypatch.setattr("mcpsim.ui.app.serve", fake_serve)
    monkeypatch.chdir(env.root)
    code = main(
        ["ui", "--skill", str(env.skill), "--port", "9999", "--scenarios", "scenarios/nested"]
    )
    assert code == 0
    settings = seen["settings"]
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 9999)
    assert settings.scenario_sources == ["scenarios/nested"]
    assert settings.runs_dir == env.root.resolve() / "runs"  # from config.yaml runs_dir
    assert settings.skill_dir == env.skill.resolve()
    assert settings.allow_remote is False


def test_cli_ui_refuses_a_skill_that_does_not_load(
    env: Env, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(env.root)
    assert main(["ui", "--skill", str(env.root / "missing")]) == EXIT_FAILURE
    assert "skill directory not found" in capsys.readouterr().err
    (env.skill / "roles" / "judge.md").unlink()
    assert main(["ui", "--skill", str(env.skill)]) == EXIT_FAILURE
    assert "missing role file(s) judge.md" in capsys.readouterr().err


def test_resolve_settings_precedence(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from_config = resolve_settings(skill=str(env.skill), cwd=env.root)
    assert from_config.scenario_sources is None, "follows config.yaml's scenarios"
    assert from_config.runs_dir == env.root.resolve() / "runs"
    assert from_config.skill.config.scenarios == ["scenarios", "scenarios/nested"]
    flags = resolve_settings(
        skill=str(env.skill), runs="elsewhere", scenarios=["a", "b"], cwd=env.root
    )
    assert flags.scenario_sources == ["a", "b"]
    assert flags.runs_dir == env.root.resolve() / "elsewhere"
    monkeypatch.setenv(SKILL_ENV, str(env.skill))
    assert resolve_settings(cwd=env.root).skill_dir == env.skill.resolve()
    monkeypatch.delenv(SKILL_ENV)
    # Neither --skill nor MCPSIM_SKILL: the packaged skill (the checkout's, in a source tree).
    packaged = resolve_settings(cwd=env.root)
    assert packaged.skill.name == "simulate"
    assert packaged.skill_dir != env.skill.resolve()
    assert packaged.runs_dir == env.root.resolve() / "runs" / "simulate"
    with pytest.raises(SkillError, match="skill directory not found"):
        resolve_settings(skill=str(env.root / "missing"), cwd=env.root)


def test_a_job_runs_the_real_cli_with_the_skill(env: Env) -> None:
    """The page's Run button end to end: the real `mcpsim run --skill` in dry run (no model is
    called) against the fake stdio server; the run directory records the skill's resolution."""
    jobs = JobManager(
        runs_dir=env.runs.resolve(),
        skill_dir=env.skill.resolve(),
        env={"PYTHONPATH": str(REPO_ROOT)},
        cwd=env.root,
        output_drain_s=0.5,
    )
    app = create_app(env.settings(), token=TOKEN, jobs=jobs)
    try:
        with TestClient(app, base_url=BASE) as c:
            res = _post(c, "/api/run", {"scenarios": ["v2-shopper"], "repeat": 1, "dry_run": True})
            assert res.status_code == 202, res.text
            done = _wait(c, res.json()["job_id"], timeout=180)
            task = done["scenarios"][0]
            log = "\n".join(done["log_tail"])
            assert task["status"] in ("passed", "failed", "partial"), log
            assert f"--skill {env.skill.resolve()}" in log
            run_dir = env.runs / "v2-shopper" / task["run_id"]
            recorded = json.loads((run_dir / "scenario.json").read_text())
            assert recorded["models"]["planner"] == "claude-sonnet-5-5", "config.yaml defaults"
            assert recorded["models"]["agent"] == "claude-haiku-4-5-20251001", "the override"
            assert recorded["models"]["judge"] == "claude-fable-5-1", "the scenario file"
            assert (recorded["repeat"], recorded["judge_votes"]) == (1, 3)
            assert recorded["agent"]["skill_name"] == "shop-sop"
            stems = sorted(p.stem for p in (run_dir / "transcripts").glob("*.jsonl"))
            assert stems == ["happy-dry-run-guided-0"], "the override's modes: [guided]"
            listing = {s["name"]: s for s in c.get("/api/scenarios").json()["scenarios"]}
            assert listing["v2-shopper"]["last_run"]["run_id"] == task["run_id"]
            assert listing["v2-shopper"]["pass_k"]["k"] == 1
    finally:
        jobs.shutdown()
