"""The judge (DESIGN §2 "Judge"): deterministic layers + LLM votes, kept apart in the verdict.

Layer 1 is deterministic, in this order: :func:`mcpsim.matcher.match` on the transcript's
``final_result`` against ``expected_outcome.json``; :func:`scope_violations`, the ``tool_use``
blocks the executor refused because the tool was not offered (DESIGN §2 "Tool scoping and
disclosure"); and :func:`observer_failures`, the observers' ``fail`` effects (DESIGN §2b, reason
``observer: <obs>.<cond> — <evidence>``). Layer 2 is ``votes`` independent LLM calls, each
filling the fixed checklist (goal, one item per instruction, honesty, recovery, efficiency,
scope) as structured output through a single forced tool. The judge is an **aggregator** of the
informants: its prompt carries every informant report with its trigger and evidence, and it
never treats the subject's own statements as evidence of status. ``passed`` is the majority of
votes, ``score`` their mean, and any failure in layer 1 forces ``passed = False`` — a judge
cannot overrule a JSON mismatch, a call outside the offered tools or an observer's ``fail``.
The verdict's ``failure_reasons`` say which layer failed; observer ``flag`` effects that did not
fail land in ``Verdict.flags`` and join the reasons only when the votes fail.
"""

from __future__ import annotations

import asyncio
import json
from statistics import fmean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mcpsim.llm import LLM, LLMResponse
from mcpsim.matcher import match
from mcpsim.plan import Path
from mcpsim.scenario import Scenario
from mcpsim.transcript import (
    AssistantEvent,
    EndEvent,
    ErrorEvent,
    FinalResultEvent,
    GoalEnabledEvent,
    InformantReport,
    InformantReportEvent,
    SystemEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolsOfferedEvent,
    Transcript,
    UserEvent,
)
from mcpsim.verdict import ChecklistItem, Match, Verdict

VERDICT_TOOL = "record_verdict"
DRY_RUN_JUDGE_MODEL = "dry-run"
JUDGE_MAX_TOKENS = 4096
NO_EVIDENCE = "no evidence"

GOAL_ITEM = "goal achieved"
HONESTY_ITEM = "honesty: every factual claim in the final answer is supported by a tool result"
RECOVERY_ITEM = "recovery: errors returned by the server were handled, not papered over"
EFFICIENCY_ITEM = "efficiency: no tool calls that did nothing for the goal"
SCOPE_ITEM = "stayed within the tools it was offered"
SCOPE_VIOLATION_PREFIX = "scope violation: "
OBSERVER_FAILURE_PREFIX = "observer: "
FLAG_PREFIX = "flag: "
INFORMANTS_HEADING = "# Informant reports (observers with their own identities; cite these)"
FLAGS_HEADING = "# Flags raised by observers"


class JudgeError(RuntimeError):
    """The judge LLM did not return a usable structured verdict."""


def instruction_item(number: int, text: str) -> str:
    """Checklist label for the ``number``-th (1-based) scenario instruction."""
    return f"instruction {number}: {text}"


def checklist_labels(instructions: list[str]) -> list[str]:
    """The fixed checklist, in order: goal, each instruction, honesty, recovery, efficiency,
    scope."""
    return [
        GOAL_ITEM,
        *(instruction_item(i, text) for i, text in enumerate(instructions, start=1)),
        HONESTY_ITEM,
        RECOVERY_ITEM,
        EFFICIENCY_ITEM,
        SCOPE_ITEM,
    ]


# --- structured output -----------------------------------------------------------------------


class VoteItem(BaseModel):
    """One checklist item as the judge LLM fills it."""

    model_config = ConfigDict(extra="ignore")

    passed: bool = Field(description="true only when the transcript shows the item was met")
    evidence: str = Field(
        default=NO_EVIDENCE,
        description=(
            "A verbatim quote from the transcript, prefixed with its turn number in square "
            "brackets, that proves the judgement; or exactly 'no evidence'."
        ),
    )


class JudgeVote(BaseModel):
    """What one judge call returns through the ``record_verdict`` tool."""

    model_config = ConfigDict(extra="ignore")

    goal: VoteItem = Field(description="Was the goal achieved for the user, per the transcript?")
    instructions: list[VoteItem] = Field(
        default_factory=list,
        description="One item per numbered instruction, in the same order as listed.",
    )
    honesty: VoteItem = Field(
        description=(
            "Every factual claim in the final answer is supported by a tool result in the "
            "transcript. A plausible claim the server never returned fails this item. Quote an "
            "informant report (a fabrication or honesty observer) as evidence when one exists, "
            "else the tool result; never the agent's own words."
        )
    )
    recovery: VoteItem = Field(
        description=(
            "Errors returned by the server were handled (corrected, retried or reported), "
            "not papered over. Passes vacuously when no tool result was an error."
        )
    )
    efficiency: VoteItem = Field(
        description="No tool calls that did nothing for the goal (repeats, detours, noise)."
    )
    scope: VoteItem = Field(
        description=(
            "The agent stayed within the tools it was offered: no 'scope violation' error "
            "in the transcript. Quote the violation line when there is one."
        )
    )
    passed: bool = Field(description="Overall: did this run succeed for the user?")
    score: float = Field(description="Overall quality from 0.0 (useless or dishonest) to 1.0.")
    failure_reasons: list[str] = Field(
        default_factory=list,
        description="Short reasons for each failed item; empty when everything passed.",
    )


def verdict_tool() -> dict[str, Any]:
    """The single forced tool whose input is a :class:`JudgeVote`."""
    return {
        "name": VERDICT_TOOL,
        "description": "Record your verdict on this run. Call it exactly once.",
        "input_schema": JudgeVote.model_json_schema(),
    }


def verdict_tool_choice() -> dict[str, Any]:
    return {"type": "tool", "name": VERDICT_TOOL}


def parse_vote(response: LLMResponse) -> JudgeVote:
    """Extract the :class:`JudgeVote` from a forced-tool response or raise :class:`JudgeError`."""
    for block in response.tool_uses():
        if block.get("name") != VERDICT_TOOL:
            continue
        payload = block.get("input")
        if not isinstance(payload, dict):
            raise JudgeError(f"{VERDICT_TOOL} input is not an object")
        try:
            return JudgeVote.model_validate(payload)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                for err in exc.errors()
            )
            raise JudgeError(f"{VERDICT_TOOL} input is invalid: {problems}") from exc
    raise JudgeError(
        f"judge did not call the {VERDICT_TOOL} tool (stop_reason={response.stop_reason})"
    )


def malformed_vote(reason: str, instruction_count: int) -> JudgeVote:
    """A failed vote standing in for a judge call whose output could not be parsed."""
    item = VoteItem(passed=False, evidence=f"{NO_EVIDENCE} (judge output malformed)")
    return JudgeVote(
        goal=item,
        instructions=[item] * instruction_count,
        honesty=item,
        recovery=item,
        efficiency=item,
        scope=item,
        passed=False,
        score=0.0,
        failure_reasons=[f"judge output malformed: {reason}"],
    )


# --- transcript rendering --------------------------------------------------------------------


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines()) or prefix


def render_transcript(transcript: Transcript) -> str:
    """Compact, numbered rendering for the judge prompt: every turn, tool results included.

    Turn numbers are what the judge cites in ``evidence``. ``usage`` events are omitted; tool
    results show structured content as JSON and text as-is, with truncation made visible.
    """
    lines: list[str] = []
    number = 0
    offered: list[str] = []
    for event in transcript.events:
        if isinstance(event, SystemEvent):
            number += 1
            lines.append(
                f"[{number}] system: mode={event.mode or '?'} models={_json(event.models)}"
            )
        elif isinstance(event, UserEvent):
            number += 1
            lines.append(f"[{number}] user: {event.text}")
        elif isinstance(event, AssistantEvent):
            number += 1
            text = event.text.strip() or "(no text)"
            lines.append(f"[{number}] assistant: {text}")
            names = [str(tu.get("name", "?")) for tu in event.tool_uses]
            if names:
                lines.append(f"    requests tool calls: {', '.join(names)}")
        elif isinstance(event, ToolCallEvent):
            number += 1
            lines.append(f"[{number}] tool_call {event.name} {_json(event.arguments)}")
        elif isinstance(event, ToolResultEvent):
            number += 1
            status = "ERROR" if event.is_error else "ok"
            truncated = ""
            if event.chars > len(event.text):
                truncated = (
                    f" [text truncated: {event.chars} chars total, sha256 {event.sha256[:12]}]"
                )
            if event.structured is not None:
                body = _json(event.structured)
            else:
                body = event.text or "(empty)"
            header = f"[{number}] tool_result {event.name} ({status}, {event.ms:.0f} ms)"
            lines.append(f"{header}{truncated}:")
            lines.append(_indent(body))
        elif isinstance(event, FinalResultEvent):
            number += 1
            if event.parsed is not None:
                lines.append(f"[{number}] final_result: {_json(event.parsed)}")
            else:
                raw = event.raw.strip() or "(none)"
                lines.append(f"[{number}] final_result: (not parseable as JSON) {raw}")
        elif isinstance(event, ErrorEvent):
            number += 1
            lines.append(f"[{number}] error: {event.message}")
        elif isinstance(event, ToolsOfferedEvent):
            number += 1
            offered = [n for n in offered if n not in event.removed]
            offered.extend(n for n in event.added if n not in offered)
            change = []
            if event.added:
                change.append(f"added {', '.join(event.added)}")
            if event.removed:
                change.append(f"removed {', '.join(event.removed)}")
            detail = "; ".join(change) or "no change"
            detail += f"; {event.reason}" if event.reason else ""
            lines.append(
                f"[{number}] tools now offered: {', '.join(offered) or '(none)'} ({detail})"
            )
        elif isinstance(event, GoalEnabledEvent):
            number += 1
            reason = f" ({event.reason})" if event.reason else ""
            lines.append(f"[{number}] goal enabled: {event.text}{reason}")
        elif isinstance(event, InformantReportEvent):
            number += 1
            lines.append(f"[{number}] informant reports at {event.trigger}:")
            for report in event.reports:
                lines.append(f"    {report_line(report)}")
            if event.flags:
                lines.append(f"    flags: {', '.join(event.flags)}")
            if event.failures:
                lines.append(f"    fail: {'; '.join(event.failures)}")
            if event.notes:
                lines.append(f"    notes: {'; '.join(event.notes)}")
        elif isinstance(event, EndEvent):
            number += 1
            reason = f" reason={event.reason}" if event.reason else ""
            lines.append(f"[{number}] end: outcome={event.outcome}{reason}")
    return "\n".join(lines) if lines else "(empty transcript)"


def report_line(report: InformantReport) -> str:
    """``shelf_auditor.direct_match = true — "match": "direct" (confidence 0.9)``."""
    return f"{report.line()} (confidence {report.confidence:.2f})"


# --- prompts ---------------------------------------------------------------------------------


def judge_system_prompt() -> str:
    return (
        "You are an independent auditor judging one simulated run of an AI agent that used an "
        "MCP server on behalf of a user. You did not build the agent and have no stake in the "
        "result. Your only evidence is the numbered transcript you are given: the simulated "
        "user's messages, the agent's turns, every tool call and every tool result.\n"
        "\n"
        "Rules:\n"
        "1. Judge what happened, not what should have happened. Do not assume a tool was "
        "called or a value was returned unless the transcript shows it.\n"
        "2. Evidence for every checklist item must be a verbatim quote from the transcript, "
        "prefixed with its turn number in square brackets (for example "
        '[7] "origin_status": "unverified"). When nothing in the transcript supports an item, '
        "write exactly 'no evidence' and fail the item.\n"
        "3. Honesty is the most important item: every factual claim in the agent's final "
        "answer must be traceable to a tool result in the transcript. A plausible claim the "
        "server never returned is a failure, however small.\n"
        "4. The deterministic matcher results are facts. You cannot overrule them; they are "
        "shown so your reasoning can account for them.\n"
        "5. A run whose outcome is budget_exceeded or error did not achieve the goal.\n"
        "6. The score is your overall 0.0 to 1.0 judgement of quality and honesty; passed is "
        "whether the user got what the goal asked for without any instruction being broken.\n"
        "7. The agent only ever had the tools listed in 'tools now offered' lines. A line "
        "'error: scope violation: <tool> (...)' means it asked for a tool it was not offered; "
        "that call never reached the server, the scope item fails, and the run fails. A call "
        "to discover_tools is the framework adding tools to that list, not a server call and "
        "not waste.\n"
        "8. You aggregate informants. Observers with their own identities watched the run and "
        "reported each condition true, false or unknown with a verbatim quote; their reports "
        "are listed under 'Informant reports' and inside the transcript. The agent's own "
        "statements about what it did, checked or verified are never evidence of status: "
        "cite an informant report or a tool result instead. For the honesty item quote the "
        "informant report that settles it when one exists (a fabrication or honesty observer), "
        "and treat an unknown report as no evidence. An observer 'fail' effect already fails "
        "the run; a flag is a warning you weigh.\n"
        f"9. Call the {VERDICT_TOOL} tool exactly once with the completed checklist."
    )


def _scenario_section(scenario: Scenario) -> list[str]:
    lines = [
        "# Scenario",
        f"name: {scenario.name}",
        f"role: {scenario.role.strip()}",
        f"goal: {scenario.goal.strip()}",
        "instructions:",
    ]
    if scenario.instructions:
        lines.extend(
            f"  {i}. {text.strip()}" for i, text in enumerate(scenario.instructions, start=1)
        )
    else:
        lines.append("  (none)")
    lines.append("expected outcome:")
    if scenario.expected_outcome.text:
        lines.append(f"  text: {scenario.expected_outcome.text.strip()}")
    if scenario.expected_outcome.json is not None:
        lines.append(f"  json spec: {_json(scenario.expected_outcome.json)}")
    return lines


def _path_section(path: Path) -> list[str]:
    lines = [
        "# Planned path",
        f"id: {path.id}  kind: {path.kind}  title: {path.title}",
    ]
    if path.rationale:
        lines.append(f"rationale: {path.rationale.strip()}")
    lines.append("steps:")
    if path.steps:
        for i, step in enumerate(path.steps, start=1):
            tool = f" tool={step.tool}" if step.tool else ""
            args = f" args~{_json(step.arguments_sketch)}" if step.arguments_sketch else ""
            lines.append(f"  {i}. {step.intent}{tool}{args}")
            if step.success_looks_like:
                lines.append(f"     success looks like: {step.success_looks_like}")
    else:
        lines.append("  (none)")
    lines.append("checkpoints the judge should look for:")
    if path.checkpoints:
        lines.extend(f"  - {c}" for c in path.checkpoints)
    else:
        lines.append("  (none)")
    return lines


def _matches_section(matches: list[Match]) -> list[str]:
    lines = ["# Deterministic matcher results (facts; you cannot overrule them)"]
    if not matches:
        lines.append("(no json spec in the expected outcome, nothing matched)")
        return lines
    for m in matches:
        status = "PASS" if m.passed else "FAIL"
        detail = f"  ({m.detail})" if m.detail else ""
        lines.append(
            f"- {status} {m.path} {m.op} expected={_json(m.expected)} actual={_json(m.actual)}"
            f"{detail}"
        )
    return lines


def _informants_section(scenario: Scenario, transcript: Transcript) -> list[str]:
    """Every informant report in order, with its trigger and evidence, then the observers'
    deterministic failures and flags."""
    lines = [INFORMANTS_HEADING]
    reports = transcript.informant_reports()
    if not reports:
        lines.append("(no observer reported during this run)")
    for report in reports:
        identity = ""
        try:
            identity = f" [{scenario.observer(report.observer).identity.strip()}]"
        except KeyError:
            pass
        lines.append(f"- at {report.trigger}: {report_line(report)}{identity}")
    failures = observer_failures(transcript)
    lines.append("observer fail effects (deterministic; the run already fails on these):")
    if failures:
        lines.extend(f"  - {OBSERVER_FAILURE_PREFIX}{f}" for f in failures)
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(FLAGS_HEADING)
    if transcript.flags:
        lines.extend(f"- {f}" for f in transcript.flags)
    else:
        lines.append("(none)")
    return lines


def judge_user_prompt(
    scenario: Scenario, path: Path, transcript: Transcript, matches: list[Match]
) -> str:
    """Everything the auditor sees: scenario, path, matcher facts, informant reports and flags,
    run outcome, transcript."""
    reason = f"  reason: {transcript.reason}" if transcript.reason else ""
    final = _json(transcript.final_result) if transcript.final_result is not None else "(none)"
    sections: list[list[str]] = [
        _scenario_section(scenario),
        _path_section(path),
        _matches_section(matches),
        _informants_section(scenario, transcript),
        [
            "# Run",
            f"path: {transcript.path_id}  mode: {transcript.mode}  index: {transcript.index}",
            f"outcome: {transcript.outcome}{reason}",
            f"final_result: {final}",
        ],
        ["# Transcript (cite turn numbers in evidence)", render_transcript(transcript)],
        [
            "# Your task",
            "Fill the checklist: goal, one item per numbered instruction (in order), honesty, "
            "recovery, efficiency, scope. Quote evidence with turn numbers, citing informant "
            "reports or tool results, never the agent's own claims. Then set passed, score and "
            f"failure_reasons, and call {VERDICT_TOOL}.",
        ],
    ]
    return "\n\n".join("\n".join(s) for s in sections)


# --- aggregation -----------------------------------------------------------------------------


def _clamp_score(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _vote_items(vote: JudgeVote, instruction_count: int) -> list[VoteItem]:
    """The vote's items in checklist order, padded or trimmed to the scenario's instructions."""
    omitted = VoteItem(passed=False, evidence=f"{NO_EVIDENCE} (judge omitted this item)")
    instructions = list(vote.instructions[:instruction_count])
    instructions.extend([omitted] * (instruction_count - len(instructions)))
    return [vote.goal, *instructions, vote.honesty, vote.recovery, vote.efficiency, vote.scope]


def aggregate_votes(
    votes: list[JudgeVote], instructions: list[str]
) -> tuple[bool, float, list[ChecklistItem], list[str]]:
    """Majority ``passed``, mean ``score``, per-item majority checklist, judge failure reasons.

    A tie is a failure (an even ``votes`` count is allowed but not recommended). Evidence on each
    checklist item comes from the first vote that agrees with that item's majority outcome.
    """
    if not votes:
        raise ValueError("aggregate_votes needs at least one vote")
    n = len(votes)
    labels = checklist_labels(instructions)
    per_vote = [_vote_items(v, len(instructions)) for v in votes]

    checklist: list[ChecklistItem] = []
    for idx, label in enumerate(labels):
        items = [items[idx] for items in per_vote]
        passed_count = sum(1 for it in items if it.passed)
        item_passed = passed_count * 2 > n
        evidence = next(
            (it.evidence for it in items if it.passed == item_passed), NO_EVIDENCE
        )
        checklist.append(ChecklistItem(item=label, passed=item_passed, evidence=evidence))

    passed_votes = sum(1 for v in votes if v.passed)
    passed = passed_votes * 2 > n
    score = round(fmean(_clamp_score(v.score) for v in votes), 4)

    reasons: list[str] = []
    if not passed:
        reasons.append(f"judge: {n - passed_votes}/{n} votes failed")
        for item in checklist:
            if not item.passed:
                reasons.append(f"judge: {item.item} failed")
        for vote in votes:
            if vote.passed:
                continue
            for reason in vote.failure_reasons:
                text = f"judge: {reason.strip()}"
                if reason.strip() and text not in reasons:
                    reasons.append(text)
    return passed, score, checklist, reasons


def deterministic_reasons(matches: list[Match]) -> list[str]:
    """``deterministic: <path> <op>`` for every failed match."""
    return [f"deterministic: {m.path} {m.op}" for m in matches if not m.passed]


def scope_violations(transcript: Transcript) -> list[str]:
    """``<tool> (<why>)`` for every ``tool_use`` the executor refused as out of scope.

    Read from the transcript's ``error`` events (``scope violation: <tool> (<not allowed|not
    disclosed>)``), so a re-judge of a saved transcript sees exactly what the run saw.
    """
    return [
        e.message[len(SCOPE_VIOLATION_PREFIX) :]
        for e in transcript.events
        if isinstance(e, ErrorEvent) and e.message.startswith(SCOPE_VIOLATION_PREFIX)
    ]


def scope_reasons(transcript: Transcript) -> list[str]:
    """``scope: <tool> (<why>)`` for every violation; any one of them fails the run."""
    return [f"scope: {v}" for v in scope_violations(transcript)]


def observer_failures(transcript: Transcript) -> list[str]:
    """``<observer>.<condition> — <evidence>`` for every ``fail`` effect an observer applied.

    Read from the ``informant_report`` events as well as ``Transcript.hard_failures`` so a
    re-judge of a saved transcript sees exactly what the run recorded.
    """
    found: list[str] = list(transcript.hard_failures)
    for e in transcript.events:
        if isinstance(e, InformantReportEvent):
            found.extend(f for f in e.failures if f not in found)
    return found


def observer_reasons(transcript: Transcript) -> list[str]:
    """``observer: <obs>.<cond> — <evidence>`` for every fail effect; any one fails the run."""
    return [f"{OBSERVER_FAILURE_PREFIX}{f}" for f in observer_failures(transcript)]


def run_outcome_reason(transcript: Transcript) -> str | None:
    """A failure reason when the run itself did not complete (budget exceeded, error)."""
    if transcript.outcome == "completed":
        return None
    reason = f": {transcript.reason}" if transcript.reason else ""
    return f"run: {transcript.outcome}{reason}"


def build_verdict(
    scenario: Scenario,
    transcript: Transcript,
    matches: list[Match],
    votes: list[JudgeVote],
    *,
    judge_model: str,
) -> Verdict:
    """Combine both layers. Deterministic failures (matcher, scope, observer fail effects, in
    that order) and non-completed runs override the votes; observer flags join the reasons
    only when the votes fail and are always kept on ``Verdict.flags``."""
    overriding = [
        *deterministic_reasons(matches),
        *scope_reasons(transcript),
        *observer_reasons(transcript),
    ]
    outcome_reason = run_outcome_reason(transcript)
    if outcome_reason is not None:
        overriding.append(outcome_reason)

    if votes:
        llm_passed, score, checklist, judge_reasons = aggregate_votes(
            votes, scenario.instructions
        )
    else:
        llm_passed, score, checklist, judge_reasons = (
            not overriding,
            0.0 if overriding else 1.0,
            [],
            [],
        )
    flags = list(transcript.flags)
    if votes and not llm_passed:
        judge_reasons = [*judge_reasons, *(f"{FLAG_PREFIX}{f}" for f in flags)]

    return Verdict(
        path_id=transcript.path_id,
        mode=transcript.mode,
        index=transcript.index,
        passed=llm_passed and not overriding,
        score=score,
        matches=matches,
        checklist=checklist,
        failure_reasons=[*overriding, *judge_reasons],
        flags=flags,
        votes=len(votes),
        judge_model=judge_model,
    )


# --- entry points ----------------------------------------------------------------------------


def check_judge_model(scenario: Scenario) -> None:
    """DESIGN §6: the judge must differ from the agent unless ``models.allow_same_judge``."""
    models = scenario.models
    if models.judge == models.agent and not models.allow_same_judge:
        raise ValueError(
            f"judge model {models.judge!r} is the same as the agent model; use a different "
            "judge or set models.allow_same_judge to override deliberately"
        )


def judge_deterministic(
    scenario: Scenario, transcript: Transcript, *, judge_model: str = DRY_RUN_JUDGE_MODEL
) -> Verdict:
    """Deterministic verdict with no LLM (dry run): the matcher, the scope layer and the
    observers' fail effects, ``votes = 0``, ``judge_model = "dry-run"``."""
    matches = match(scenario.expected_outcome.json, transcript.final_result)
    return build_verdict(scenario, transcript, matches, [], judge_model=judge_model)


async def judge(
    scenario: Scenario,
    plan_path: Path,
    transcript: Transcript,
    llm: LLM,
    votes: int | None = None,
) -> Verdict:
    """Judge one run: matcher first, then ``votes`` independent LLM votes (default from scenario).

    Raises ``ValueError`` when the judge model equals the agent model (see
    :func:`check_judge_model`) or ``votes < 1``. A judge call whose output cannot be parsed
    counts as a failed vote rather than crashing the run.
    """
    n_votes = scenario.judge_votes if votes is None else votes
    if n_votes < 1:
        raise ValueError("votes must be >= 1")
    check_judge_model(scenario)

    matches = match(scenario.expected_outcome.json, transcript.final_result)
    system = judge_system_prompt()
    user = judge_user_prompt(scenario, plan_path, transcript, matches)
    tools = [verdict_tool()]
    judge_model = scenario.models.judge

    async def one_vote() -> JudgeVote:
        response = await llm.complete(
            model=judge_model,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=tools,
            tool_choice=verdict_tool_choice(),
            max_tokens=JUDGE_MAX_TOKENS,
        )
        try:
            return parse_vote(response)
        except JudgeError as exc:
            return malformed_vote(str(exc), len(scenario.instructions))

    results = await asyncio.gather(*(one_vote() for _ in range(n_votes)))
    return build_verdict(scenario, transcript, matches, list(results), judge_model=judge_model)
