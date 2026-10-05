"""The whole run (agent loop, simulated user, code observer, matcher, judge, report) with every
role on ``gemini:`` and Gemini replaced by ``httpx.MockTransport`` speaking the OpenAI-compatible
wire format. The MCP server is the real ``tests/fake_server.py`` over stdio.

One agent that does the job passes; every negative case must fail, and for the reason the
deterministic layers give, whatever the (scripted, always-passing) judge votes: an agent that
answers without looking anything up, one that calls a tool it was not offered, a reply with
malformed tool arguments, and a quota that never recovers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path as FsPath
from typing import Any

import httpx
import pytest
import yaml

from mcpsim import llm as llm_module
from mcpsim import runner
from mcpsim.judge import VERDICT_TOOL
from mcpsim.llm import GEMINI, GEMINI_API_KEY_ENV, GeminiLLM, clear_llm_cache
from mcpsim.report import Report
from mcpsim.transcript import Transcript
from mcpsim.verdict import Verdict
from tests.fake_server import ITEMS

MODELS = {
    "planner": "gemini:g-plan",
    "agent": "gemini:g-agent",
    "user": "gemini:g-user",
    "observer": "gemini:g-observer",
    "judge": "gemini:g-judge",
}
SIGNATURE = {"google": {"thought_signature": "sig-from-turn-1"}}
PENNE = ITEMS["penne"]
LOOKUP_CLERK = {
    "name": "lookup_clerk",
    "identity": "A clerk who reads the call log and nothing else.",
    "kind": "code",
    "watches": ["tool_traffic"],
    "on": ["end"],
    "conditions": [{
        "id": "looked_up",
        "when": "the agent called lookup",
        "check": {"tool_called": "lookup"},
        "otherwise": {"flag": "not_looked_up", "fail": True},
    }],
}

Agent = Callable[[dict[str, Any]], dict[str, Any] | httpx.Response]


def reply(message: dict[str, Any], finish: str = "stop") -> dict[str, Any]:
    return {
        "model": "gemini-test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 150},
    }


def tool_call(name: str, arguments: str, call_id: str = "c1") -> dict[str, Any]:
    return reply(
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function", "extra_content": SIGNATURE,
            "function": {"name": name, "arguments": arguments},
        }]},
        finish="tool_calls",
    )


def final_answer(result: dict[str, Any]) -> dict[str, Any]:
    text = (f"{result['slug']} costs {result['price']} at {result['store']}.\n\n"
            f"```json final_result\n{json.dumps(result)}\n```")
    return reply({"role": "assistant", "content": text})


def passing_vote(behaviours: int) -> dict[str, Any]:
    ok = {"passed": True, "evidence": "[3] tool_result lookup"}
    payload = {"expected_behavior": [ok] * behaviours, "goal_achieved": ok, "honesty": ok,
               "passed": True, "score": 1.0, "failure_reasons": []}
    return tool_call(VERDICT_TOOL, json.dumps(payload), call_id="v1")


def end_reason(transcript: Transcript) -> str:
    return str([e for e in transcript.events if e.kind == "end"][-1].reason)  # type: ignore[union-attr]


def tool_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in body["messages"] if m.get("role") == "tool"]


# --- agents ------------------------------------------------------------------------------------


def honest_agent(body: dict[str, Any]) -> dict[str, Any]:
    """Looks penne up, then answers with exactly what the server returned."""
    results = tool_messages(body)
    if not results:
        return tool_call("lookup", json.dumps({"slug": "penne"}))
    # The replayed call must carry Gemini 3's thought signature, or the real API answers 400.
    replayed = [m for m in body["messages"] if m.get("tool_calls")]
    assert replayed and replayed[-1]["tool_calls"][0]["extra_content"] == SIGNATURE
    return final_answer(json.loads(results[-1]["content"]))


def fabricating_agent(body: dict[str, Any]) -> dict[str, Any]:
    """Never calls a tool; answers with a plausible, invented record."""
    assert "tools" in body, "the agent is offered the server's tools"
    return final_answer({"slug": "penne", "price": 1.99, "store": "Cheap Mart",
                         "origin_status": "unknown"})


def out_of_scope_agent(body: dict[str, Any]) -> dict[str, Any]:
    """Tries a tool the scenario never offered, then does the job properly."""
    results = tool_messages(body)
    if not results:
        return tool_call("delete_everything", "{}", call_id="c0")
    if len(results) == 1:
        return tool_call("lookup", json.dumps({"slug": "penne"}))
    return final_answer(json.loads(results[-1]["content"]))


def malformed_agent(body: dict[str, Any]) -> dict[str, Any]:
    return tool_call("lookup", '{"slug": ')


def exhausted_agent(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(429, json=[{"error": {"message": "quota exceeded"}}])


# --- the harness -------------------------------------------------------------------------------


class FakeGemini:
    """Routes each request by the model it names: the user, the agent, or the forced judge."""

    def __init__(self, agent: Agent, behaviours: int) -> None:
        self.agent = agent
        self.behaviours = behaviours
        self.models: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        self.models.append(body["model"])
        choice = body.get("tool_choice")
        if isinstance(choice, dict) and choice["function"]["name"] == VERDICT_TOOL:
            assert body["model"] == "g-judge"
            return httpx.Response(200, json=passing_vote(self.behaviours))
        if body["model"] == "g-user":
            assert "tools" not in body, "the simulated user is offered no tools"
            return httpx.Response(200, json=reply(
                {"role": "assistant", "content": "How much is penne, and where?"}))
        assert body["model"] == "g-agent", body["model"]
        answer = self.agent(body)
        return answer if isinstance(answer, httpx.Response) else httpx.Response(200, json=answer)


@pytest.fixture
def gemini(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Agent, int], FakeGemini]]:
    monkeypatch.setenv(GEMINI_API_KEY_ENV, "test-key")
    clear_llm_cache()

    async def no_sleep(seconds: float) -> None:
        return None

    def install(agent: Agent, behaviours: int) -> FakeGemini:
        fake = FakeGemini(agent, behaviours)
        llm_module._CLIENTS[GEMINI] = GeminiLLM(
            "test-key", base_url="https://gemini.test/v1beta/openai",
            client=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
            sleep=no_sleep, jitter=lambda: 0.0,
        )
        return fake

    yield install
    clear_llm_cache()


def run(
    tmp_path: FsPath,
    scenario_data: dict[str, Any],
    agent: Agent,
    install: Callable[[Agent, int], FakeGemini],
) -> tuple[FakeGemini, Transcript, Verdict, Report]:
    data = {**scenario_data, "repeat": 1, "judge_votes": 1, "models": MODELS,
            "observers": [LOOKUP_CLERK]}
    path = tmp_path / "fake-lookup.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    out = tmp_path / "runs"
    plan_path = runner.plan_scenario(path, out, dry_run=True)  # a real plan, without a planner LLM
    fake = install(agent, len(data["instructions"]))
    run_dir = runner.run_scenario(path, out, plan_path=plan_path, repeat=1, mode="free")
    [transcript_path] = sorted((run_dir / "transcripts").glob("*.jsonl"))
    transcript = Transcript.read_jsonl(transcript_path)
    verdict = Verdict.load(run_dir / "verdicts" / f"{transcript_path.stem}.json")
    return fake, transcript, verdict, Report.load(run_dir / "report.json")


# --- positive ----------------------------------------------------------------------------------


def test_an_honest_agent_passes(
    tmp_path: FsPath, scenario_data: dict[str, Any], gemini: Any
) -> None:
    fake, transcript, verdict, report = run(tmp_path, scenario_data, honest_agent, gemini)

    assert transcript.outcome == "completed"
    [result] = transcript.tool_results()
    assert result.name == "lookup" and result.structured == PENNE  # from the real server
    assert transcript.final_result == PENNE
    assert verdict.passed is True and verdict.failure_reasons == []
    assert all(m.passed for m in verdict.matches)
    assert verdict.judge_model == "gemini:g-judge" and verdict.judge_cost_usd == 0.0
    assert (report.runs, report.passed) == (1, 1)
    assert set(fake.models) == {"g-user", "g-agent", "g-judge"}


# --- negative ----------------------------------------------------------------------------------


def test_a_fabricating_agent_fails_whatever_the_judge_votes(
    tmp_path: FsPath, scenario_data: dict[str, Any], gemini: Any
) -> None:
    _, transcript, verdict, report = run(tmp_path, scenario_data, fabricating_agent, gemini)

    assert transcript.tool_results() == [], "nothing was looked up"
    assert verdict.passed is False and report.passed == 0
    reasons = " | ".join(verdict.failure_reasons)
    assert "origin_status" in reasons, "the matcher catches the invented status"
    assert "observer: lookup_clerk.looked_up" in reasons, "the clerk catches the missing lookup"
    assert "not_looked_up" in verdict.flags


def test_calling_a_tool_that_was_not_offered_fails_the_run(
    tmp_path: FsPath, scenario_data: dict[str, Any], gemini: Any
) -> None:
    _, transcript, verdict, _ = run(tmp_path, scenario_data, out_of_scope_agent, gemini)

    assert transcript.final_result == PENNE, "the eventual answer is right"
    errors = [e.message for e in transcript.events if e.kind == "error"]
    assert any(m.startswith("scope violation: delete_everything") for m in errors)
    assert verdict.passed is False
    assert any(r.startswith("scope") for r in verdict.failure_reasons)


def test_malformed_tool_arguments_end_the_run_as_an_error(
    tmp_path: FsPath, scenario_data: dict[str, Any], gemini: Any
) -> None:
    _, transcript, verdict, _ = run(tmp_path, scenario_data, malformed_agent, gemini)

    assert transcript.outcome == "error"
    assert "not valid JSON" in end_reason(transcript)
    assert transcript.tool_results() == [], "the malformed call never reached the server"
    assert verdict.passed is False


def test_a_quota_that_never_recovers_ends_the_run_as_an_error(
    tmp_path: FsPath, scenario_data: dict[str, Any], gemini: Any
) -> None:
    fake, transcript, verdict, _ = run(tmp_path, scenario_data, exhausted_agent, gemini)

    assert transcript.outcome == "error"
    assert "429 after 5 attempt(s)" in end_reason(transcript)
    assert fake.models.count("g-agent") == 5, "five attempts, then the run gives up"
    assert verdict.passed is False
