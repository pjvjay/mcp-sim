"""The agent under test (DESIGN §2 "Executor"): one run of one path.

:func:`run_path` drives a standard Anthropic-shaped tool-use loop whose ``tools`` are the MCP
catalog's schemas. Every ``tool_use`` block is executed through the
:class:`~mcpsim.mcpclient.Session` and returned as a ``tool_result`` block (structured content
as JSON, text passed through,
``is_error`` preserved). A :class:`SimulatedUser` plays the scenario's *role*: it opens the
conversation in the role's voice and answers clarifying questions, never volunteering more than
the scenario gives it. The run ends when the agent delivers a ``final_result`` block (outcome
``completed``), when a budget is exhausted (``budget_exceeded``) or when the session or the LLM
raises (``error``). A run never raises for any of those; the :class:`~mcpsim.transcript.Transcript`
carries the reason.

Dry run (``dry_run=True``): no LLM at all. Each step's tool is called with its sketch arguments
in order and ``final_result`` is synthesised from the last structured tool result, so the whole
MCP path is exercised without an API key. ``{"$from_step": n, "path": ...}`` references in a
sketch are resolved against step ``n``'s structured result (a reference that cannot be resolved
records an ``error`` event and skips the step), and an error result on a step marked
``expect_error`` is the planned outcome, not a failure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from mcpsim.llm import DEFAULT_MAX_TOKENS, LLM, Usage, total_cost_usd
from mcpsim.matcher import MISSING, resolve
from mcpsim.mcpclient import Catalog, Session, ToolResult
from mcpsim.plan import Mode, Path, Step, StepReference, parse_reference
from mcpsim.scenario import Scenario
from mcpsim.transcript import (
    AssistantEvent,
    EndEvent,
    ErrorEvent,
    FinalResultEvent,
    Outcome,
    SystemEvent,
    ToolCallEvent,
    ToolResultEvent,
    Transcript,
    UsageEvent,
    UserEvent,
)

FINAL_RESULT_NAME = "final_result"
USER_MAX_TOKENS = 512
DRY_RUN_MODEL = "dry-run"

# ---------------------------------------------------------------------------------------------
# final_result extraction


@dataclass(frozen=True)
class FinalResult:
    """What :func:`extract_final_result` found in an assistant message.

    ``found`` is True when the message contains something that is meant to be the final answer
    (a fenced block, or a trailing JSON object). ``parsed`` is the object when it is valid, else
    ``None`` with ``error`` explaining why.
    """

    found: bool
    raw: str = ""
    parsed: dict[str, Any] | None = None
    error: str | None = None


_FENCE = re.compile(r"```[ \t]*([^\n]*)\n(.*?)```", re.DOTALL)


def _parse_object(raw: str, *, source: str) -> FinalResult:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return FinalResult(
            found=True, raw=raw, error=f"{FINAL_RESULT_NAME} {source} is not valid JSON: {exc}"
        )
    if not isinstance(value, dict):
        return FinalResult(
            found=True,
            raw=raw,
            error=f"{FINAL_RESULT_NAME} {source} must be a JSON object, "
            f"got {type(value).__name__}",
        )
    return FinalResult(found=True, raw=raw, parsed=value)


def extract_final_result(text: str) -> FinalResult:
    """Find the agent's structured final answer in a message.

    Accepted, in order of preference: the last fenced block whose info string names
    ``final_result`` (``\\`\\`\\`json final_result``); else the last ``\\`\\`\\`json`` fenced block;
    else a bare JSON object that ends the message. A block that is present but malformed yields
    ``found=True, parsed=None`` with ``error`` set; never an exception.
    """
    fences = list(_FENCE.finditer(text))
    named = [m for m in fences if FINAL_RESULT_NAME in m.group(1)]
    json_fences = [m for m in fences if m.group(1).strip().lower().startswith("json")]
    candidates = named or json_fences
    if candidates:
        return _parse_object(candidates[-1].group(2).strip(), source="block")

    stripped = text.rstrip()
    if stripped.endswith("}"):
        # Outermost object first: the first "{" from which the rest of the text parses.
        for i, ch in enumerate(stripped):
            if ch == "{":
                try:
                    json.loads(stripped[i:])
                except json.JSONDecodeError:
                    continue
                return _parse_object(stripped[i:], source="trailing object")
        last_line = stripped.splitlines()[-1].strip()
        if last_line.startswith("{"):
            return _parse_object(last_line, source="trailing object")
    return FinalResult(found=False)


# ---------------------------------------------------------------------------------------------
# prompts


def _top_level_fields(spec: dict[str, Any] | None) -> list[str]:
    """Top-level ``final_result`` field names from the dotted paths of ``expected_outcome.json``."""
    names: list[str] = []
    for key in spec or {}:
        head = re.split(r"[.\[]", str(key), maxsplit=1)[0].strip()
        if head and head not in names:
            names.append(head)
    return names


def readable_sketch(step: Step) -> str:
    """The sketch as JSON with every ``$from_step`` reference shown as ``<from step n: path>``."""
    shown: dict[str, Any] = {}
    for key, value in step.arguments_sketch.items():
        try:
            ref = parse_reference(value)
        except ValueError:
            ref = None
        shown[key] = ref.describe() if ref is not None else value
    return json.dumps(shown, sort_keys=True, ensure_ascii=False)


def steps_section(path: Path) -> str:
    """The "Suggested approach" section a ``guided`` run adds to the agent's system prompt.

    References render as ``<from step n: path>`` (take the value from that step's result) and
    ``expect_error`` steps say the server is expected to reject the call, so the agent knows
    the error is the point of the step and must read it rather than treat it as a dead end.
    """
    lines = [
        "## Suggested approach",
        "The following steps are one way to reach the goal. Treat them as guidance: adapt when",
        "the server responds differently, and skip anything that turns out not to be needed.",
        "An argument written <from step n: path> means: take that value from the result of",
        "step n (path is dotted; [*] means every element).",
    ]
    for n, step in enumerate(path.steps, start=1):
        line = f"{n}. {step.intent.strip()}"
        if step.tool:
            line += f" (tool: {step.tool}"
            if step.arguments_sketch:
                line += f", arguments roughly {readable_sketch(step)}"
            line += ")"
        if step.expect_error:
            line += (
                " — EXPECT AN ERROR: the server should reject this call; read its message"
                " and correct the next call from it"
            )
        if step.success_looks_like.strip():
            line += f" — success looks like: {step.success_looks_like.strip()}"
        lines.append(line)
    if not path.steps:
        lines.append("(the plan lists no steps for this path)")
    return "\n".join(lines)


def answer_contract(scenario: Scenario) -> str:
    fields = _top_level_fields(scenario.expected_outcome.json)
    if fields:
        fields_line = "The object must include these fields: " + ", ".join(
            f"`{f}`" for f in fields
        )
    else:
        fields_line = "Choose the fields that best describe the outcome"
    return "\n".join(
        [
            "## Answer contract",
            "When you have everything you need, deliver your final answer in one message with no",
            "tool calls: a short plain-language summary for the person, followed by a fenced code",
            f"block that starts with ```json {FINAL_RESULT_NAME} and contains a single JSON",
            f"object. {fields_line}. Use values exactly as the tools returned them (no rounding,",
            "renaming or guessing). If you could not achieve the goal, say so plainly in the",
            "summary and",
            "still deliver the block, filling what you honestly can and using null for the rest.",
        ]
    )


def agent_prompt_sections(scenario: Scenario, path: Path, mode: Mode) -> list[str]:
    """The agent's system prompt as sections; ``guided`` inserts exactly :func:`steps_section`."""
    instructions = (
        "\n".join(f"- {item.strip()}" for item in scenario.instructions)
        if scenario.instructions
        else "(none beyond the goal)"
    )
    sections = [
        "You are an assistant acting on behalf of a person, using the tools of an MCP server to "
        "achieve their goal. The person talks to you; you may ask them a clarifying question when "
        "the goal is genuinely ambiguous, but prefer using the tools. Every factual claim you make "
        "must be supported by a tool result you received in this conversation; when the server "
        "returns an error, read it and correct your request rather than guessing.",
        f"## Who you are acting for\n{scenario.role.strip()}",
        f"## Goal\n{scenario.goal.strip()}",
        f"## Instructions you must follow\n{instructions}",
    ]
    if mode == "guided":
        sections.append(steps_section(path))
    sections.append(answer_contract(scenario))
    return sections


def build_agent_system_prompt(scenario: Scenario, path: Path, mode: Mode) -> str:
    return "\n\n".join(agent_prompt_sections(scenario, path, mode))


def build_user_system_prompt(scenario: Scenario) -> str:
    return "\n\n".join(
        [
            "You are playing a person in a simulation. The assistant you are talking to is an AI "
            "agent under test; it will use tools on your behalf. Stay in character throughout and "
            "never say that you are simulated.",
            f"## Who you are\n{scenario.role.strip()}",
            f"## What you want\n{scenario.goal.strip()}",
            "## Rules\n"
            "- Speak in the first person, in your own voice, in one to three sentences.\n"
            "- Never volunteer facts, preferences or constraints beyond what is written above. "
            "If the assistant asks about something not covered here, say you do not know or "
            "tell it to use its best judgement and its tools.\n"
            "- Do not do the assistant's work: do not suggest tool names, prices or answers.\n"
            "- If the assistant seems to have finished without its structured "
            f"`{FINAL_RESULT_NAME}` block, ask it to deliver its final answer with that block.\n"
            "- Do not thank, praise or correct the assistant beyond what your character would say.",
        ]
    )


_OPEN_CUE = (
    "Start the conversation: in your own words and voice, tell the assistant what you want. "
    "Reply with only what you would say."
)
_FALLBACK_REPLY = "Please go ahead with what you have and give me your final answer."
_EMPTY_AGENT_MESSAGE = "(the assistant sent an empty message)"


class SimulatedUser:
    """An LLM playing the scenario's role; keeps its own view of the conversation."""

    def __init__(self, scenario: Scenario, llm: LLM, model: str) -> None:
        self.scenario = scenario
        self.model = model
        self.system = build_user_system_prompt(scenario)
        self._llm = llm
        self._history: list[dict[str, Any]] = []

    async def _say(self, prompt: str, fallback: str) -> tuple[str, Usage]:
        self._history.append({"role": "user", "content": prompt})
        response = await self._llm.complete(
            model=self.model,
            system=self.system,
            messages=list(self._history),
            max_tokens=USER_MAX_TOKENS,
        )
        text = response.text().strip() or fallback
        self._history.append({"role": "assistant", "content": text})
        return text, response.usage

    async def open(self) -> tuple[str, Usage]:
        """The first message: the goal in the role's voice (the goal verbatim as a fallback)."""
        return await self._say(_OPEN_CUE, self.scenario.goal.strip())

    async def reply(self, agent_text: str) -> tuple[str, Usage]:
        """Answer a clarifying question (or any non-final agent message), staying in role."""
        return await self._say(agent_text.strip() or _EMPTY_AGENT_MESSAGE, _FALLBACK_REPLY)


# ---------------------------------------------------------------------------------------------
# the run


def tool_result_block(tool_use_id: str, result: ToolResult) -> dict[str, Any]:
    """The Anthropic ``tool_result`` block for a normalised MCP result."""
    if result.structured is not None:
        content = json.dumps(result.structured, ensure_ascii=False)
    elif result.text:
        content = result.text
    else:
        content = "tool returned an error with no message" if result.is_error else ""
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if result.is_error:
        block["is_error"] = True
    return block


class _Run:
    """Mutable state of one run: the transcript being built, counters and budgets."""

    def __init__(self, scenario: Scenario, path: Path, mode: Mode, index: int) -> None:
        self.scenario = scenario
        self.budgets = scenario.budgets
        self.transcript = Transcript(
            scenario=scenario.name, path_id=path.id, mode=mode, index=index
        )
        self.usage: dict[str, Usage] = {}
        self.turns = 0
        self.tool_calls = 0
        self.finished = False

    @property
    def cost_usd(self) -> float:
        return total_cost_usd(self.usage)

    def record_usage(self, model: str, usage: Usage) -> str | None:
        """Add a call's usage; returns a budget reason when ``max_cost_usd`` is now exceeded."""
        self.usage[model] = self.usage.get(model, Usage()) + usage
        if self.cost_usd > self.budgets.max_cost_usd:
            return (
                f"max_cost_usd={self.budgets.max_cost_usd} exceeded: "
                f"estimated cost so far ${self.cost_usd:.6f}"
            )
        return None

    def end(self, outcome: Outcome, reason: str) -> Transcript:
        t = self.transcript
        t.add(UsageEvent(per_model=dict(self.usage), cost_usd=self.cost_usd, estimate=True))
        t.add(EndEvent(outcome=outcome, reason=reason))
        t.outcome = outcome
        t.reason = reason
        t.usage = dict(self.usage)
        t.cost_usd = self.cost_usd
        self.finished = True
        return t

    async def call(
        self,
        session: Session,
        name: str,
        arguments: dict[str, Any],
        tool_use_id: str | None,
    ) -> ToolResult:
        """Record a tool call and its result. Raises whatever the session raises."""
        self.tool_calls += 1
        self.transcript.add(ToolCallEvent(name=name, arguments=arguments, tool_use_id=tool_use_id))
        result = await session.call_tool(name, arguments)
        self.transcript.add(ToolResultEvent.from_result(result, tool_use_id=tool_use_id))
        return result

    def tool_budget_reason(self, name: str) -> str | None:
        if self.tool_calls >= self.budgets.max_tool_calls:
            return (
                f"max_tool_calls={self.budgets.max_tool_calls} exceeded: "
                f"agent requested another call to {name}"
            )
        return None

    def turn_budget_reason(self) -> str | None:
        if self.turns >= self.budgets.max_turns:
            return (
                f"max_turns={self.budgets.max_turns} exceeded: "
                "agent did not deliver a final answer in time"
            )
        return None


def _describe(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class StepReferenceError(LookupError):
    """A ``$from_step`` reference could not be resolved; the message says why."""


def resolve_reference(ref: StepReference, results: dict[int, ToolResult | None]) -> Any:
    """The value a reference denotes, given the results of the steps run so far.

    ``results`` maps 1-based step index → that step's result (``None`` when the step called no
    tool or was skipped). ``[*]`` yields the list of every element's value. Raises
    :class:`StepReferenceError` when the step has no structured result or the path is missing.
    """
    where = f"step {ref.from_step}"
    if ref.from_step not in results:
        raise StepReferenceError(f"{where} has not run (references must point at an earlier step)")
    result = results[ref.from_step]
    if result is None:
        raise StepReferenceError(f"{where} called no tool, so it has no result to reference")
    if result.structured is None:
        kind = "an error" if result.is_error else "text only"
        raise StepReferenceError(
            f"{where} ({result.name}) returned {kind}, no structured content to read "
            f"{ref.path!r} from"
        )
    try:
        found = resolve(result.structured, ref.path)
    except ValueError as exc:
        raise StepReferenceError(str(exc)) from None
    if found.quantifier == "one":
        if not found.present:
            raise StepReferenceError(f"path {ref.path!r} is missing from {where}'s result")
        return found.single
    if not found.present:
        raise StepReferenceError(f"path {ref.path!r} in {where}'s result is not an array")
    missing = sum(1 for v in found.values if v is MISSING)
    if missing:
        raise StepReferenceError(
            f"path {ref.path!r}: {missing} of {len(found.values)} element(s) in {where}'s "
            "result lack the field"
        )
    return list(found.values)


def resolve_arguments(step: Step, results: dict[int, ToolResult | None]) -> dict[str, Any]:
    """The step's sketch with every reference replaced by its value (literals pass through)."""
    arguments: dict[str, Any] = {}
    for key, value in step.arguments_sketch.items():
        ref = parse_reference(value)  # a malformed reference raises ValueError
        if ref is None:
            arguments[key] = value
            continue
        try:
            arguments[key] = resolve_reference(ref, results)
        except StepReferenceError as exc:
            raise StepReferenceError(f"argument {key!r}: {exc}") from None
    return arguments


async def _dry_run(run: _Run, path: Path, session: Session) -> Transcript:
    run.transcript.add(UserEvent(text=run.scenario.goal.strip()))
    last_structured: dict[str, Any] | list[Any] | None = None
    results: dict[int, ToolResult | None] = {}
    skipped = 0
    expected_errors = 0
    unexpected_errors = 0
    for n, step in enumerate(path.steps, start=1):
        results[n] = None
        if step.tool is None:
            continue
        reason = run.tool_budget_reason(step.tool)
        if reason is not None:
            return run.end("budget_exceeded", reason)
        try:
            arguments = resolve_arguments(step, results)
        except (StepReferenceError, ValueError) as exc:
            skipped += 1
            run.transcript.add(
                ErrorEvent(
                    message=f"dry run: step {n} ({step.tool}) skipped, its arguments could not "
                    f"be resolved: {exc}"
                )
            )
            continue
        try:
            result = await run.call(session, step.tool, arguments, f"dry-{n}")
        except Exception as exc:  # noqa: BLE001 - a run never crashes; the reason is recorded
            return run.end("error", f"tool call {step.tool} failed: {_describe(exc)}")
        results[n] = result
        if result.is_error:
            if step.expect_error:
                expected_errors += 1  # the planned rejection: this step succeeded
            else:
                unexpected_errors += 1
        elif step.expect_error:
            run.transcript.add(
                ErrorEvent(
                    message=f"dry run: step {n} ({step.tool}) was expected to be rejected by "
                    "the server but succeeded"
                )
            )
        if not result.is_error and result.structured is not None:
            last_structured = result.structured
    final: dict[str, Any] | None
    if isinstance(last_structured, dict):
        final = last_structured
    elif isinstance(last_structured, list):
        final = {"result": last_structured}
    else:
        final = None
    raw = json.dumps(last_structured, ensure_ascii=False) if last_structured is not None else ""
    run.transcript.add(FinalResultEvent(parsed=final, raw=raw))
    if final is None:
        run.transcript.add(
            ErrorEvent(message="dry run: no step returned structured content; final_result is null")
        )
    run.transcript.final_result = final
    reason = f"dry run: called {run.tool_calls} planned tool(s)"
    if expected_errors:
        reason += f"; {expected_errors} expected error(s) returned as planned"
    if unexpected_errors:
        reason += f"; {unexpected_errors} unexpected error result(s)"
    if skipped:
        reason += f"; {skipped} step(s) skipped over unresolved references"
    return run.end("completed", reason)


async def run_path(
    scenario: Scenario,
    path: Path,
    mode: Mode,
    index: int,
    session: Session,
    llm: LLM | None,
    *,
    dry_run: bool = False,
    catalog: Catalog | None = None,
) -> Transcript:
    """Run ``path`` once in ``mode`` and return the transcript (never raises for run failures).

    ``llm`` may be ``None`` only when ``dry_run`` is true. ``catalog`` is the *allowed* catalog
    (the runner applies ``scenario.tools`` once); when not given it is discovered from the
    session and filtered here, so the agent never sees a tool the scenario denies.
    """
    if not dry_run and llm is None:
        raise ValueError("run_path needs an llm unless dry_run=True")
    run = _Run(scenario, path, mode, index)
    agent_model = DRY_RUN_MODEL if dry_run else scenario.models.agent
    user_model = DRY_RUN_MODEL if dry_run else scenario.models.user_model
    agent_system = build_agent_system_prompt(scenario, path, mode)
    user_system = build_user_system_prompt(scenario)
    run.transcript.add(
        SystemEvent(
            scenario=scenario.name,
            path_id=path.id,
            index=index,
            mode=mode,
            models={"agent": agent_model, "user": user_model},
            prompts={"agent": agent_system, "user": user_system},
        )
    )
    if dry_run:
        return await _dry_run(run, path, session)
    assert llm is not None  # for the type checker; guarded above

    try:
        if catalog is None:
            catalog = await session.catalog()
        catalog = catalog.filtered(scenario.tools.allow, scenario.tools.deny)
        tools = catalog.to_anthropic_tools()
        user = SimulatedUser(scenario, llm, user_model)

        opening, usage = await user.open()
        run.transcript.add(UserEvent(text=opening))
        reason = run.record_usage(user_model, usage)
        if reason is not None:
            return run.end("budget_exceeded", reason)
        messages: list[dict[str, Any]] = [{"role": "user", "content": opening}]

        while True:
            reason = run.turn_budget_reason()
            if reason is not None:
                return run.end("budget_exceeded", reason)
            response = await llm.complete(
                model=agent_model,
                system=agent_system,
                messages=list(messages),
                tools=tools or None,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            run.turns += 1
            text = response.text()
            tool_uses = response.tool_uses()
            run.transcript.add(
                AssistantEvent(text=text, tool_uses=tool_uses, stop_reason=response.stop_reason)
            )
            reason = run.record_usage(agent_model, response.usage)
            if reason is not None:
                return run.end("budget_exceeded", reason)
            messages.append({"role": "assistant", "content": list(response.content)})

            if tool_uses:
                results: list[dict[str, Any]] = []
                for block in tool_uses:
                    name = str(block.get("name", ""))
                    tool_use_id = str(block.get("id", "")) or None
                    raw_input = block.get("input")
                    arguments = dict(raw_input) if isinstance(raw_input, dict) else {}
                    reason = run.tool_budget_reason(name)
                    if reason is not None:
                        return run.end("budget_exceeded", reason)
                    result = await run.call(session, name, arguments, tool_use_id)
                    results.append(tool_result_block(tool_use_id or "", result))
                messages.append({"role": "user", "content": results})
                continue

            final = extract_final_result(text)
            if final.found:
                run.transcript.add(FinalResultEvent(parsed=final.parsed, raw=final.raw))
                run.transcript.final_result = final.parsed
                if final.error is not None:
                    run.transcript.add(ErrorEvent(message=final.error))
                    return run.end("completed", f"final answer delivered but {final.error}")
                return run.end("completed", "final answer delivered")

            # No tool use and no final block: the agent is talking to the person.
            reply, usage = await user.reply(text)
            run.transcript.add(UserEvent(text=reply))
            reason = run.record_usage(user_model, usage)
            if reason is not None:
                return run.end("budget_exceeded", reason)
            messages.append({"role": "user", "content": reply})
    except Exception as exc:  # noqa: BLE001 - a run never crashes; the reason is recorded
        return run.end("error", _describe(exc))
