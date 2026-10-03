"""Every prompt the simulation sends, rendered for a fixed set of inputs.

:func:`render_all` drives the real call sites (planner, local execution planner, agent,
simulated user, observers, judge) with a recording LLM and returns ``{case id: text}`` for every
system prompt, user message and re-ask they produce, plus the ``max_tokens`` each call asked
for. ``tests/fixtures/prompts/golden.json.gz`` holds the output captured from the code *before* the
prompt text moved into ``skills/simulate/roles/*.md``; ``tests/test_prompt_golden.py`` checks
that the bundled templates still render exactly those bytes.

Inputs: the six pantry scenarios, a copy of ``cheapest-penne`` with an inline standard operating
procedure and environment notes, and a minimal v1 scenario (no context, no instructions, a text
outcome only); the pantry catalog fixture; the scout, plans and transcripts in
``tests/fixtures/prompts`` (a live Anthropic run, the dry-run example and the local-run
example).

Regenerate the golden file only on purpose (a deliberate prompt change)::

    python -m tests.prompt_cases --write
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path as FsPath
from typing import Any

from mcpsim.llm import LLMResponse, Usage
from mcpsim.mcpclient import Catalog
from mcpsim.plan import ExecutionPlan, Path
from mcpsim.scenario import Scenario, load_scenario, parse_scenario
from mcpsim.scout import ScoutResult
from mcpsim.transcript import Transcript

ROOT = FsPath(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
PROMPTS = FIXTURES / "prompts"
GOLDEN = PROMPTS / "golden.json.gz"
PANTRY = ROOT / "scenarios" / "pantry"
PANTRY_NAMES = (
    "cheapest-penne",
    "label-submission",
    "misspelled-country",
    "tomato-penne-boycott",
    "unknown-recipe",
    "week-under-budget",
)
GOALS = [
    "Goal enabled by observation (shelf_clerk.direct_match): Quote the cheapest direct hit by "
    "exact name, price and store.",
    "Goal enabled by observation (shelf_auditor.direct_match): Check the origin before answering.",
]
SOP_TEXT = (
    "# Price check\n\n1. Look the item up with `find_product`, using the shopper's own words as "
    "the query (results\n   come back cheapest first).\n2. If the match is \"direct\", take the "
    "first item.\n3. Finish with the `final_result` block."
)
SOP_NOTES = "You cannot run scripts or open a shell here; use the tools you are offered."


# --- inputs -----------------------------------------------------------------------------------


def pantry_catalog() -> Catalog:
    raw = json.loads((FIXTURES / "pantry_catalog.json").read_text(encoding="utf-8"))
    raw.pop("_comment", None)
    return Catalog.model_validate(raw)


def scenarios() -> dict[str, Scenario]:
    found = {name: load_scenario(PANTRY / f"{name}.yaml") for name in PANTRY_NAMES}
    base = found["cheapest-penne"].model_dump(mode="python")
    base["name"] = "cheapest-penne-sop"
    base["agent"] = {"skill_text": SOP_TEXT, "skill_name": "price-check", "notes": SOP_NOTES}
    found["cheapest-penne-sop"] = Scenario.model_validate(base)
    found["v1-minimal"] = parse_scenario(
        {
            "name": "v1-minimal",
            "role": "A shopper.",
            "goal": "Find penne.",
            "expected_outcome": {"text": "Some penne."},
            "server": {"stdio": {"command": "true"}},
            "observers": [{"use": "fabrication_auditor"}],
        }
    )
    return found


def scout() -> ScoutResult:
    return ScoutResult.load(PROMPTS / "live-scout.json")


def plans() -> dict[str, ExecutionPlan]:
    return {
        name: ExecutionPlan.load(PROMPTS / f"{name}.json")
        for name in ("live-plan", "local-plan", "dry-plan", "local-run-plan")
    }


def transcripts() -> list[tuple[str, Transcript, str, str]]:
    """``(label, transcript, plan name, path id)``."""
    return [
        ("live", Transcript.read_jsonl(PROMPTS / "live-free-0.jsonl"), "live-plan",
         "happy-find-penne"),
        ("dry", Transcript.read_jsonl(PROMPTS / "dry-guided-0.jsonl"), "dry-plan",
         "happy-dry-run"),
        ("local", Transcript.read_jsonl(PROMPTS / "local-guided-0.jsonl"), "local-run-plan",
         "1"),
    ]


# --- a recording LLM --------------------------------------------------------------------------


class Recorder:
    """Answers every call with the next scripted response (the last one repeats) and records
    what was sent."""

    def __init__(self, *responses: LLMResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def text(value: str) -> LLMResponse:
    return LLMResponse(
        content=[{"type": "text", "text": value}], stop_reason="end_turn", usage=Usage()
    )


def tool_call(name: str, payload: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=[{"type": "tool_use", "id": "toolu_rec", "name": name, "input": payload}],
        stop_reason="tool_use",
        usage=Usage(),
    )


def content_text(content: Any) -> str:
    """A message's content as text: a string as is, blocks joined (tool results by content)."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if block.get("type") == "text":
            parts.append(str(block["text"]))
        elif block.get("type") == "tool_result":
            parts.append(f"<tool_result error={bool(block.get('is_error'))}>{block['content']}")
        else:
            parts.append(json.dumps(block, sort_keys=True))
    return "\n<block>\n".join(parts)


def record_calls(out: dict[str, str], prefix: str, calls: list[dict[str, Any]]) -> None:
    """``<prefix>.call<n>.system`` / ``.message<i>`` / ``.max_tokens`` / ``.temperature``."""
    for n, call in enumerate(calls, start=1):
        key = f"{prefix}.call{n}"
        out[f"{key}.system"] = call["system"]
        for i, message in enumerate(call["messages"]):
            if message["role"] == "user":
                out[f"{key}.message{i}"] = content_text(message["content"])
        out[f"{key}.max_tokens"] = str(call.get("max_tokens"))
        if call.get("temperature") is not None:
            out[f"{key}.temperature"] = str(call["temperature"])


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# --- the cases --------------------------------------------------------------------------------


def planner_cases(out: dict[str, str], name: str, scenario: Scenario, catalog: Catalog) -> None:
    from mcpsim.planner import PLAN_TOOL_NAME, PlanError, plan_with_llm

    variants: list[tuple[str, ScoutResult | None, int | None, LLMResponse]] = [
        ("noscout", None, None, tool_call(PLAN_TOOL_NAME, {"paths": []})),
        ("scout", scout(), None, tool_call(PLAN_TOOL_NAME, {"paths": []})),
        ("scout-tight", scout(), 9000, text("I would rather describe the plan in prose.")),
        ("scout-tiny", scout(), 3000, tool_call(PLAN_TOOL_NAME, {"paths": []})),
    ]
    for label, scout_result, budget, response in variants:
        llm = Recorder(response)
        try:
            run(plan_with_llm(scenario, catalog, llm, scout_result, prompt_budget=budget))
        except PlanError:
            pass
        record_calls(out, f"{name}.planner.{label}", llm.calls)


def local_planner_cases(
    out: dict[str, str], name: str, scenario: Scenario, catalog: Catalog
) -> None:
    from mcpsim.execution_planner import (
        EXECUTION_TOOL_NAME,
        POLICY_TOOL_NAME,
        ask_forbidden_tools,
        plan_execution,
    )
    from mcpsim.planner import PlanError

    local = scenario.model_copy(
        update={"models": scenario.models.model_copy(update={"planner": "ollama:command-r7b"})}
    )
    variants: list[tuple[str, ScoutResult | None, int | None, LLMResponse]] = [
        ("noscout", None, None, tool_call(EXECUTION_TOOL_NAME, {"steps": "not a list"})),
        ("scout", scout(), None, tool_call(EXECUTION_TOOL_NAME, {"steps": "not a list"})),
        ("scout-tight", scout(), 2600, text("no json today")),
    ]
    for label, scout_result, budget, response in variants:
        llm = Recorder(response)
        try:
            run(plan_execution(local, catalog, llm, scout_result, prompt_budget=budget))
        except PlanError:
            pass
        record_calls(out, f"{name}.local.{label}", llm.calls)
    llm = Recorder(tool_call(POLICY_TOOL_NAME, {"tool": "none"}))
    notes: list[str] = []
    run(ask_forbidden_tools(local, llm, catalog.tool_names(), {"find_product"}, notes))
    record_calls(out, f"{name}.local.policy", llm.calls)


def agent_cases(out: dict[str, str], name: str, scenario: Scenario) -> None:
    from mcpsim.agent import build_agent_system_prompt, goal_note

    for plan_name, plan in plans().items():
        for path in plan.paths:
            key = f"{name}.agent.{plan_name}.{path.id}"
            out[f"{key}.guided"] = build_agent_system_prompt(scenario, path, "guided")
        first = plan.paths[0]
        out[f"{name}.agent.{plan_name}.free"] = build_agent_system_prompt(scenario, first, "free")
        out[f"{name}.agent.{plan_name}.guided-goals"] = build_agent_system_prompt(
            scenario, first, "guided", GOALS
        )
        out[f"{name}.agent.{plan_name}.free-goals"] = build_agent_system_prompt(
            scenario, first, "free", GOALS[:1]
        )
    out[f"{name}.agent.goal_note"] = goal_note(GOALS)


def user_cases(out: dict[str, str], name: str, scenario: Scenario) -> None:
    from mcpsim.agent import SimulatedUser, build_user_system_prompt

    out[f"{name}.user.system"] = build_user_system_prompt(scenario)
    llm = Recorder(text("Hi, I need the cheapest penne."), text(""))

    async def converse() -> None:
        user = SimulatedUser(scenario, llm, scenario.models.user)  # type: ignore[arg-type]
        await user.open()
        await user.reply("   ")

    run(converse())
    record_calls(out, f"{name}.user", llm.calls)


def observer_cases(out: dict[str, str], name: str, scenario: Scenario) -> None:
    from mcpsim.observers import REPORT_TOOL, ObserverRunner

    observations = scout().observations
    for label, transcript, _plan, _path in transcripts():
        llm = Recorder(tool_call(REPORT_TOOL, {"reports": []}))
        runner = ObserverRunner(scenario, llm, max_calls=1000)  # type: ignore[arg-type]
        for trigger in ("turn", "tool_result", "end"):
            run(runner.report(trigger, transcript, observations))
        record_calls(out, f"{name}.observer.{label}", llm.calls)
    llm = Recorder(tool_call(REPORT_TOOL, {"reports": []}))
    runner = ObserverRunner(scenario, llm, max_calls=1000)  # type: ignore[arg-type]
    run(runner.report("scout", None, observations))
    record_calls(out, f"{name}.observer.scout", llm.calls)


# The dry-run transcript is long (every pantry tool's result); two scenarios cover it.
DRY_JUDGE_SCENARIOS = ("cheapest-penne", "v1-minimal")


def judge_cases(out: dict[str, str], name: str, scenario: Scenario) -> None:
    from mcpsim.judge import VERDICT_TOOL, judge

    all_plans = plans()
    for label, transcript, plan_name, path_id in transcripts():
        if label == "dry" and name not in DRY_JUDGE_SCENARIOS:
            continue
        path: Path = all_plans[plan_name].path(path_id)
        llm = Recorder(tool_call(VERDICT_TOOL, {"passed": True}))
        run(judge(scenario, path, transcript, llm, votes=1))  # type: ignore[arg-type]
        record_calls(out, f"{name}.judge.{label}", llm.calls)


CASES: tuple[Callable[..., None], ...] = (agent_cases, user_cases, observer_cases, judge_cases)


def render_all() -> dict[str, str]:
    """Every case id -> rendered text, in a stable order."""
    catalog = pantry_catalog()
    out: dict[str, str] = {}
    for name, scenario in scenarios().items():
        planner_cases(out, name, scenario, catalog)
        local_planner_cases(out, name, scenario, catalog)
        for case in CASES:
            case(out, name, scenario)
    return out


def pack(cases: dict[str, str]) -> dict[str, Any]:
    """Content-addressed: ``{"cases": {id: sha}, "texts": {sha: text}}`` (system prompts repeat
    across scenarios, so each text is stored once)."""
    ids: dict[str, str] = {}
    texts: dict[str, str] = {}
    for case_id, value in cases.items():
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        ids[case_id] = digest
        texts[digest] = value
    return {"cases": ids, "texts": dict(sorted(texts.items()))}


def unpack(data: dict[str, Any]) -> dict[str, str]:
    return {case_id: data["texts"][digest] for case_id, digest in data["cases"].items()}


def load_golden() -> dict[str, str]:
    """The captured cases (gzip: the texts are long and repeat most of their words)."""
    return unpack(json.loads(gzip.decompress(GOLDEN.read_bytes()).decode("utf-8")))


def main(argv: list[str]) -> int:
    cases = render_all()
    if "--write" in argv:
        raw = json.dumps(pack(cases), indent=1, ensure_ascii=False) + "\n"
        GOLDEN.write_bytes(gzip.compress(raw.encode("utf-8"), mtime=0))
        print(f"wrote {len(cases)} cases to {GOLDEN}")
        return 0
    golden = load_golden()
    differ = sorted(k for k in set(golden) | set(cases) if golden.get(k) != cases.get(k))
    for case_id in differ[:20]:
        print(f"differs: {case_id}")
    print(f"{len(cases)} cases, {len(differ)} differ")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
