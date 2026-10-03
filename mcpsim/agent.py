"""The agent under test (DESIGN §2 "Executor"): one run of one path.

:func:`run_path` drives a standard Anthropic-shaped tool-use loop whose ``tools`` are the MCP
catalog's schemas. Every ``tool_use`` block is executed through the
:class:`~mcpsim.mcpclient.Session` and returned as a ``tool_result`` block (structured content
as JSON, text passed through,
``is_error`` preserved). A :class:`SimulatedUser` plays the scenario's ``user_instructions`` in
its ``context`` (device, location, language; DESIGN §3 "Scenario v2"): it opens the
conversation in that person's voice and answers clarifying questions, never volunteering more
than its instructions give it. The agent's system prompt carries role, goal and instructions,
the scenario's standard operating procedure (``agent.skill``, delimited) and ``agent.notes``,
and the context only when ``context.agent_visible``; never the user's instructions or the
judge's expected behaviour. The run ends when the agent delivers a ``final_result`` block (outcome
``completed``), when a budget is exhausted (``budget_exceeded``) or when the session or the LLM
raises (``error``). A run never raises for any of those; the :class:`~mcpsim.transcript.Transcript`
carries the reason.

Tool disclosure (DESIGN §2 "Tool scoping and disclosure"): the LLM only ever sees the tools in
:class:`ToolScope`'s ``offered`` list, built from the *allowed* catalog per
``scenario.tools.disclosure`` — ``all`` (every allowed tool from turn one), ``plan`` (the path's
tools when guided, everything when free) or ``progressive`` (the explicit ``initial`` globs or
:func:`mcpsim.scoping.initial_tools`, plus the framework's own ``discover_tools`` meta-tool that
adds up to three more per query without touching the server). Every change to the set is a
``tools_offered`` transcript event. A ``tool_use`` naming a tool that is not offered never
reaches the server: the agent gets an error ``tool_result`` and the transcript an ``error``
event ``scope violation: <tool> (not allowed|not disclosed)``. In guided mode under
``progressive`` disclosure the path's step tools join the initial set (reason
``initial:guided:path``), so a guided plan is executable without a ``discover_tools`` detour.

Observers (DESIGN §2b, :mod:`mcpsim.observers`): after every assistant turn (``turn``), every
batch of tool results (``tool_result``) and the final answer (``end``) the loop asks the
scenario's observers for reports and applies their effects through :class:`LiveRun` before the
next LLM call — the reports are recorded first (``informant_report``), then ``enable_tools`` /
``disable_tools`` change the offered set (``tools_offered`` with reason
``observer:<obs>.<cond>``), ``enable_goal`` records ``goal_enabled`` and the goal joins the
system prompt ("Goal enabled by observation (<obs>.<cond>): …") and the next user message,
and ``flag`` / ``fail`` land on ``Transcript.flags`` / ``hard_failures``. A hand-held
:class:`LiveRun` (``run_path(..., on_start=...)``) can do the same from outside.

Dry run (``dry_run=True``): no LLM at all. Each step's tool is called with its sketch arguments
in order and ``final_result`` is synthesised from the structured tool result whose top-level
keys cover the most ``expected_outcome.json`` keys (the last one on a tie; the last structured
result when none covers any), so the whole MCP path is exercised without an API key. Code and
group observers still run (LLM observers do not) and their effects are applied.
``{"$from_step": n, "path": ...}`` references in a sketch are resolved against step ``n``'s
structured result (a reference that cannot be resolved records an ``error`` event and skips the
step), and an error result on a step marked
``expect_error`` is the planned outcome, not a failure. The dry run follows the plan, so it is
offered exactly the path's allowed tools whatever the disclosure mode says; a step naming a
tool outside the allowed catalog is refused the same way as in a live run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from mcpsim.llm import DEFAULT_MAX_TOKENS, LLM, Usage, total_cost_usd
from mcpsim.matcher import MISSING, resolve
from mcpsim.mcpclient import Catalog, Session, ToolResult
from mcpsim.observers import ObserverRunner
from mcpsim.plan import Mode, Path, Step, StepReference, parse_reference
from mcpsim.scenario import Scenario, Trigger
from mcpsim.scoping import (
    DISCOVER_LIMIT,
    DISCOVER_TOOL_NAME,
    discover_tool_definition,
    expected_top_level_keys,
    first_sentence,
    initial_tools,
    rank_tools,
    tokens,
)
from mcpsim.transcript import (
    AssistantEvent,
    EndEvent,
    ErrorEvent,
    FinalResultEvent,
    GoalEnabledEvent,
    InformantReport,
    Outcome,
    SystemEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolsOfferedEvent,
    Transcript,
    UsageEvent,
    UserEvent,
)

FINAL_RESULT_NAME = "final_result"
USER_MAX_TOKENS = 512
DRY_RUN_MODEL = "dry-run"
NOW_AVAILABLE = "now available"
GOAL_PREFIX = "Goal enabled by observation"
__all__ = ["DISCOVER_LIMIT", "DISCOVER_TOOL_NAME", "discover_tool_definition"]  # re-exported
NOT_AVAILABLE = "tool {name} is not available in this conversation"
SCOPE_VIOLATION = "scope violation: {name} ({why})"

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
    fields = expected_top_level_keys(scenario.expected_outcome.json)
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


def goals_section(goals: list[str]) -> str:
    """The section the system prompt gains once observers have enabled goals (already prefixed
    ``Goal enabled by observation (<observer>.<condition>): …`` by :class:`LiveRun`)."""
    return "## Goals enabled by observation\n" + "\n".join(f"- {g}" for g in goals)


AGENT_INTRO = (
    "You are an assistant acting on behalf of a person, using the tools of an MCP server to "
    "achieve their goal. The person talks to you; you may ask them a clarifying question when "
    "the goal is genuinely ambiguous, but prefer using the tools. Every factual claim you make "
    "must be supported by a tool result you received in this conversation; when the server "
    "returns an error, read it and correct your request rather than guessing."
)
SOP_HEADING = "## Standard operating procedure (skill: {name})"
SOP_BEGIN = "<<<BEGIN SOP {name}>>>"
SOP_END = "<<<END SOP {name}>>>"
NOTES_HEADING = "## Notes on this environment"
AGENT_CONTEXT_HEADING = "## What you know about the person's situation"
NO_INSTRUCTIONS = "(none beyond the goal)"


def context_lines(scenario: Scenario) -> list[str]:
    """``- device: desktop web`` lines for the scenario's context (empty when none is set)."""
    return [f"- {label}: {value}" for label, value in scenario.context.items()]


def sop_section(scenario: Scenario) -> str:
    """The agent's standard operating procedure, delimited, or ``""`` without ``agent.skill``.

    The body is the SKILL.md with its frontmatter stripped, verbatim between
    :data:`SOP_BEGIN` / :data:`SOP_END` markers so the procedure cannot blur into the
    scenario's own text.
    """
    spec = scenario.agent
    if spec.skill_text is None:
        return ""
    name = spec.skill_name or "inline"
    return "\n".join(
        [
            SOP_HEADING.format(name=name),
            "This is how you are expected to work. Follow it step by step wherever this "
            "conversation and the tools you are offered allow; where it assumes something you "
            "do not have here (a script, a shell, a tool that is not offered), say so and do "
            "the nearest thing your tools allow, as the notes on this environment direct.",
            SOP_BEGIN.format(name=name),
            spec.skill_text.strip(),
            SOP_END.format(name=name),
        ]
    )


def notes_section(scenario: Scenario) -> str:
    """``agent.notes`` under its heading, or ``""``."""
    notes = scenario.agent.notes
    return f"{NOTES_HEADING}\n{notes.strip()}" if notes else ""


def agent_context_section(scenario: Scenario) -> str:
    """The context the agent is shown: only when ``context.agent_visible`` and set, else
    ``""``."""
    if not scenario.context.agent_visible:
        return ""
    lines = context_lines(scenario)
    return f"{AGENT_CONTEXT_HEADING}\n" + "\n".join(lines) if lines else ""


def agent_prompt_variables(
    scenario: Scenario, path: Path, mode: Mode, goals: list[str] | None = None
) -> dict[str, str]:
    """Every value the agent's system prompt is built from (an optional section is ``""``
    when absent). :func:`agent_prompt_sections` assembles them in the built-in order; a
    prompt template can place them itself."""
    instructions = (
        "\n".join(f"- {item.strip()}" for item in scenario.instructions)
        if scenario.instructions
        else NO_INSTRUCTIONS
    )
    return {
        "role": scenario.role.strip(),
        "goal": scenario.goal.strip(),
        "instructions": instructions,
        "skill_name": scenario.agent.skill_name or "",
        "skill_text": (scenario.agent.skill_text or "").strip(),
        "sop_section": sop_section(scenario),
        "notes_section": notes_section(scenario),
        "context_section": agent_context_section(scenario),
        "goals_section": goals_section(goals) if goals else "",
        "steps_section": steps_section(path) if mode == "guided" else "",
        "answer_contract": answer_contract(scenario),
    }


def agent_prompt_sections(
    scenario: Scenario, path: Path, mode: Mode, goals: list[str] | None = None
) -> list[str]:
    """The agent's system prompt as sections: intro, role, goal, instructions, then the SOP
    (:func:`sop_section`), the environment notes and the context (only when
    ``agent_visible``), the goals observers enabled so far (:func:`goals_section`), the
    path's steps when ``guided`` (exactly :func:`steps_section`) and the answer contract.
    Absent optional sections are left out. Never included: ``user_instructions``,
    ``expected_behavior`` or ``expected_outcome.text`` (the agent learns what the person
    wants from the person, and is not shown the judge's rubric)."""
    v = agent_prompt_variables(scenario, path, mode, goals)
    sections = [
        AGENT_INTRO,
        f"## Who you are acting for\n{v['role']}",
        f"## Goal\n{v['goal']}",
        f"## Instructions you must follow\n{v['instructions']}",
        v["sop_section"],
        v["notes_section"],
        v["context_section"],
        v["goals_section"],
        v["steps_section"],
        v["answer_contract"],
    ]
    return [s for s in sections if s]


def build_agent_system_prompt(
    scenario: Scenario, path: Path, mode: Mode, goals: list[str] | None = None
) -> str:
    return "\n\n".join(agent_prompt_sections(scenario, path, mode, goals))


USER_INTRO = (
    "You are playing a person in a simulation. The assistant you are talking to is an AI "
    "agent under test; it will use tools on your behalf. Stay in character throughout and "
    "never say that you are simulated."
)
USER_INSTRUCTIONS_HEADING = "## Your instructions"
USER_CONTEXT_HEADING = "## Your situation"
USER_RULES = (
    "## Rules\n"
    "- Speak in the first person, in your own voice, in one to three sentences.\n"
    "- Never volunteer facts, preferences or constraints beyond what your instructions and "
    "situation say. If the assistant asks about something not covered there, say you do not "
    "know or tell it to use its best judgement and its tools.\n"
    "- Do not do the assistant's work: do not suggest tool names, prices or answers.\n"
    "- If the assistant seems to have finished without its structured "
    f"`{FINAL_RESULT_NAME}` block, ask it to deliver its final answer with that block.\n"
    "- Do not thank, praise or correct the assistant beyond what your character would say."
)


def user_context_section(scenario: Scenario) -> str:
    """The simulated user's situation (device, location, language, details), or ``""``.

    The user always gets the context, whatever ``agent_visible`` says; a language adds the
    instruction to write every message in it.
    """
    lines = context_lines(scenario)
    if not lines:
        return ""
    text = f"{USER_CONTEXT_HEADING}\n" + "\n".join(lines)
    language = scenario.context.language
    if language:
        text += f"\nWrite every message in this language: {language.strip()}."
    return text


def user_prompt_variables(scenario: Scenario) -> dict[str, str]:
    """Every value the simulated user's system prompt is built from (``context_section`` is
    ``""`` when the scenario sets no context)."""
    return {
        "user_instructions": scenario.simulated_user_instructions.strip(),
        "context_section": user_context_section(scenario),
        "language": (scenario.context.language or "").strip(),
        "final_result_name": FINAL_RESULT_NAME,
    }


def build_user_system_prompt(scenario: Scenario) -> str:
    """The simulated user's system prompt: driven by ``user_instructions`` and the context
    (never by the agent's instructions, the SOP, the expected behaviour or the expected
    outcome)."""
    v = user_prompt_variables(scenario)
    sections = [
        USER_INTRO,
        f"{USER_INSTRUCTIONS_HEADING}\n{v['user_instructions']}",
        v["context_section"],
        USER_RULES,
    ]
    return "\n\n".join(s for s in sections if s)


OPEN_CUE = (
    "Start the conversation: in your own words and voice, tell the assistant what you want. "
    "Reply with only what you would say."
)
_FALLBACK_REPLY = "Please go ahead with what you have and give me your final answer."
_EMPTY_AGENT_MESSAGE = "(the assistant sent an empty message)"


class SimulatedUser:
    """An LLM playing the person in ``user_instructions`` and ``context``; keeps its own view of
    the conversation."""

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
        """The first message: what the person wants, in their voice (the goal verbatim as a
        fallback)."""
        return await self._say(OPEN_CUE, self.scenario.goal.strip())

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


# ---------------------------------------------------------------------------------------------
# tool disclosure


def goal_note(goals: list[str]) -> str:
    """The text block that carries goals an observer enabled mid-run to the agent."""
    bullets = "\n".join(f"- {g.strip()}" for g in goals)
    return (
        "## Additional goal\n"
        "The person's situation changed. In addition to the goal above, from now on also:\n"
        f"{bullets}"
    )


def user_content(primary: str | list[dict[str, Any]], goals: list[str]) -> Any:
    """A user message's ``content``: the reply or tool results, plus any pending goal note."""
    if not goals:
        return primary
    blocks = [{"type": "text", "text": primary}] if isinstance(primary, str) else list(primary)
    blocks.append({"type": "text", "text": goal_note(goals)})
    return blocks


class ToolScope:
    """Which allowed tools the agent is offered right now, and how the set grows.

    ``allowed`` is the scenario's allowed catalog (``scenario.tools`` allow/deny already
    applied); ``offered`` is the ordered subset whose definitions go to the LLM on every turn.
    Every change appends a ``tools_offered`` event to the transcript with a reason
    (``initial:<mode>:<disclosure>``, ``discover_tools:<query>``, or an observer's own).
    """

    def __init__(self, scenario: Scenario, allowed: Catalog, transcript: Transcript) -> None:
        self.scenario = scenario
        self.policy = scenario.tools
        self.allowed = allowed
        self.transcript = transcript
        self.offered: list[str] = []
        self.discoverable = self.policy.disclosure == "progressive" and self.policy.discover_tool

    def disclose_initial(self, path: Path, mode: Mode) -> list[str]:
        """Turn one's set per ``disclosure``; always recorded, even when empty."""
        disclosure = self.policy.disclosure
        if disclosure == "all":
            names = self.allowed.tool_names()
        elif disclosure == "plan":
            names = self.allowed.tool_names() if mode == "free" else path.tools_used()
        elif self.policy.initial is not None:
            names = [t.name for t in self.allowed.select(self.policy.initial)]
        else:
            names = [t.name for t in initial_tools(self.scenario, self.allowed)]
        added = self.offer(names, f"initial:{mode}:{disclosure}", always_record=True)
        if disclosure == "progressive" and mode == "guided":
            # A guided plan must be executable without a discover_tools detour.
            added += self.offer(path.tools_used(), "initial:guided:path")
        return added

    def offer(self, names: Iterable[str], reason: str, *, always_record: bool = False) -> list[str]:
        """Add the allowed tools among ``names`` that are not offered yet; returns the added."""
        added: list[str] = []
        for name in names:
            if name in self.offered or name in added or not self.allowed.has_tool(name):
                continue
            added.append(name)
        if added or always_record:
            self.offered.extend(added)
            self.transcript.add(ToolsOfferedEvent(added=added, reason=reason))
        return added

    def withdraw(
        self, names: Iterable[str], reason: str, *, always_record: bool = False
    ) -> list[str]:
        """Remove the offered tools among ``names`` (an observer's ``disable_tools``)."""
        removed = [n for n in names if n in self.offered]
        removed = list(dict.fromkeys(removed))
        if removed or always_record:
            self.offered = [n for n in self.offered if n not in removed]
            self.transcript.add(ToolsOfferedEvent(removed=removed, reason=reason))
        return removed

    def definitions(self) -> list[dict[str, Any]]:
        """The ``tools`` the LLM sees this turn: the offered tools, then ``discover_tools``."""
        defs = [self.allowed.tool(name).to_anthropic_tool() for name in self.offered]
        if self.discoverable:
            defs.append(discover_tool_definition())
        return defs

    def refusal(self, name: str) -> str | None:
        """Why a ``tool_use`` of ``name`` must not reach the server, or ``None`` when it may."""
        if name == DISCOVER_TOOL_NAME and self.discoverable:
            return None
        if not self.allowed.has_tool(name):
            return "not allowed"
        if name not in self.offered:
            return "not disclosed"
        return None

    def discover(self, query: str) -> str:
        """Answer ``discover_tools``: offer the top matches among the unoffered allowed tools.

        ``query`` is the only term source; the same relevance weights as the initial scoring
        apply; at most :data:`DISCOVER_LIMIT` tools with a score above zero are added.
        """
        unoffered = [t for t in self.allowed.tools if t.name not in self.offered]
        ranked = [(t, s) for t, s in rank_tools(tokens(query), unoffered) if s > 0]
        ranked = ranked[:DISCOVER_LIMIT]
        if not ranked:
            if not unoffered:
                return "Every tool of this server is already available to you."
            return (
                f"No further tool matches {query!r}; {len(unoffered)} undisclosed tool(s) "
                "remain. Try different words for what you need to do."
            )
        self.offer([t.name for t, _ in ranked], f"{DISCOVER_TOOL_NAME}:{query}")
        lines = [
            f"- {t.name}: {first_sentence(t.description) or '(no description)'} ({NOW_AVAILABLE})"
            for t, _ in ranked
        ]
        return f"{len(ranked)} tool(s) {NOW_AVAILABLE}:\n" + "\n".join(lines)


class LiveRun:
    """The handle on a run in flight: how observer effects reach the agent.

    ``offer_tools`` / ``withdraw_tools`` change the offered set (names or globs; only allowed
    tools, the return value says which); ``enable_goal`` records a ``goal_enabled`` event, adds
    the goal to the system prompt for every later turn (``goals``) and to the next user message
    (``take_goals``). :meth:`apply` records a batch of informant reports and then applies their
    effects in that order. A hand-held handle (``run_path(..., on_start=...)``) can call the same
    methods; nothing here can offer a tool the scenario denies.
    """

    def __init__(
        self, scope: ToolScope, transcript: Transcript, observers: ObserverRunner | None = None
    ) -> None:
        self.scope = scope
        self.transcript = transcript
        self.observers = observers
        self.goals: list[str] = []
        self._pending_goals: list[str] = []

    @property
    def offered(self) -> list[str]:
        return list(self.scope.offered)

    def offer_tools(
        self, names: Iterable[str], reason: str, *, always_record: bool = False
    ) -> list[str]:
        patterns = list(names)
        selected = [t.name for t in self.scope.allowed.select(patterns)]
        return self.scope.offer(selected, reason, always_record=always_record)

    def withdraw_tools(
        self, names: Iterable[str], reason: str, *, always_record: bool = False
    ) -> list[str]:
        patterns = list(names)
        selected = [t.name for t in self.scope.allowed.select(patterns)]
        return self.scope.withdraw(selected, reason, always_record=always_record)

    def enable_goal(
        self, text: str, reason: str, *, observer: str = "", condition: str = ""
    ) -> str:
        """Record the goal; returns the line the agent reads (prefixed with its provenance)."""
        self.transcript.add(
            GoalEnabledEvent(text=text, reason=reason, observer=observer, condition=condition)
        )
        source = f"{observer}.{condition}" if observer else reason
        line = f"{GOAL_PREFIX} ({source}): {text.strip()}"
        self.goals.append(line)
        self._pending_goals.append(line)
        return line

    def take_goals(self) -> list[str]:
        """The goals enabled since the last call (they go into the next user message)."""
        goals, self._pending_goals = self._pending_goals, []
        return goals

    def apply(self, trigger: Trigger, reports: list[InformantReport]) -> None:
        """Record ``reports`` (with the flags / failures / notes they trigger), THEN apply the
        tool and goal effects, so the transcript always shows the report before the change."""
        if self.observers is None or not reports:
            return
        effects = self.observers.effects(reports)
        self.transcript.add_reports(
            trigger,
            reports,
            flags=[e.effect.flag for e in effects if e.effect.flag],
            failures=[e.failure for e in effects if e.effect.fail],
            notes=[
                f"{e.observer}.{e.condition}: {e.effect.note}" for e in effects if e.effect.note
            ],
        )
        for e in effects:
            if e.effect.enable_tools:
                self.offer_tools(e.effect.enable_tools, e.reason, always_record=True)
            if e.effect.disable_tools:
                self.withdraw_tools(e.effect.disable_tools, e.reason, always_record=True)
            if e.effect.enable_goal:
                self.enable_goal(
                    e.effect.enable_goal, e.reason, observer=e.observer, condition=e.condition
                )

    async def observe(self, trigger: Trigger) -> dict[str, Usage]:
        """Ask the observers for reports at ``trigger`` and apply them; returns the LLM usage
        the observers spent (the caller folds it into the run and checks the cost budget)."""
        if self.observers is None or not self.observers.fires_at(trigger):
            return {}
        reports = await self.observers.report(trigger, self.transcript)
        self.apply(trigger, reports)
        return self.observers.take_usage()


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

    def record_usages(self, usages: dict[str, Usage]) -> str | None:
        """Add several models' usage (observers); the first budget reason, if any."""
        reason: str | None = None
        for model, usage in usages.items():
            reason = self.record_usage(model, usage) or reason
        return reason

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

    def refuse(self, name: str, why: str, tool_use_id: str | None) -> dict[str, Any]:
        """Record a scope violation; the ``tool_result`` block the agent gets instead."""
        self.transcript.add(ErrorEvent(message=SCOPE_VIOLATION.format(name=name, why=why)))
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id or "",
            "content": NOT_AVAILABLE.format(name=name),
            "is_error": True,
        }

    def discover(
        self, scope: ToolScope, arguments: dict[str, Any], tool_use_id: str | None
    ) -> ToolResult:
        """Answer ``discover_tools`` in-process: recorded like a call, never sent, not budgeted."""
        self.transcript.add(
            ToolCallEvent(name=DISCOVER_TOOL_NAME, arguments=arguments, tool_use_id=tool_use_id)
        )
        query = str(arguments.get("query", "") or "").strip()
        if query:
            text, is_error = scope.discover(query), False
        else:
            text = f"{DISCOVER_TOOL_NAME} needs a 'query': a few words on what you need to do"
            is_error = True
        result = ToolResult(
            name=DISCOVER_TOOL_NAME,
            is_error=is_error,
            text=text,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            chars=len(text),
        )
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


def best_final_result(
    structured_results: list[dict[str, Any] | list[Any]], expected_keys: list[str]
) -> dict[str, Any] | None:
    """The dry run's ``final_result``: the structured result whose top-level keys cover the
    most ``expected_keys`` (the last on a tie), else the last structured result; a list is
    wrapped as ``{"result": [...]}``; ``None`` when nothing structured came back."""
    if not structured_results:
        return None
    wrapped: list[dict[str, Any]] = [
        r if isinstance(r, dict) else {"result": r} for r in structured_results
    ]
    wanted = set(expected_keys)
    best = wrapped[-1]
    best_score = 0
    for candidate in wrapped:
        score = len(wanted & set(candidate))
        if score >= best_score and score > 0:
            best, best_score = candidate, score
    return best


async def _dry_run(
    run: _Run, path: Path, mode: Mode, session: Session, catalog: Catalog | None
) -> Transcript:
    if catalog is None:
        catalog = await session.catalog()
    catalog = catalog.filtered(run.scenario.tools.allow, run.scenario.tools.deny)
    scope = ToolScope(run.scenario, catalog, run.transcript)
    scope.offer(path.tools_used(), f"initial:{mode}:dry-run", always_record=True)
    live = LiveRun(scope, run.transcript, ObserverRunner(run.scenario, None, include_llm=False))
    run.transcript.add(UserEvent(text=run.scenario.goal.strip()))
    structured_results: list[dict[str, Any] | list[Any]] = []
    results: dict[int, ToolResult | None] = {}
    skipped = 0
    refused = 0
    expected_errors = 0
    unexpected_errors = 0
    for n, step in enumerate(path.steps, start=1):
        results[n] = None
        if step.tool is None:
            continue
        why = scope.refusal(step.tool)
        if why is not None:
            refused += 1
            run.refuse(step.tool, why, f"dry-{n}")
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
            structured_results.append(result.structured)
        await live.observe("tool_result")
    expected_keys = expected_top_level_keys(run.scenario.expected_outcome.json)
    final = best_final_result(structured_results, expected_keys)
    raw = json.dumps(final, ensure_ascii=False) if final is not None else ""
    run.transcript.add(FinalResultEvent(parsed=final, raw=raw))
    if final is None:
        run.transcript.add(
            ErrorEvent(message="dry run: no step returned structured content; final_result is null")
        )
    run.transcript.final_result = final
    await live.observe("end")
    reason = f"dry run: called {run.tool_calls} planned tool(s)"
    if expected_errors:
        reason += f"; {expected_errors} expected error(s) returned as planned"
    if unexpected_errors:
        reason += f"; {unexpected_errors} unexpected error result(s)"
    if skipped:
        reason += f"; {skipped} step(s) skipped over unresolved references"
    if refused:
        reason += f"; {refused} step(s) refused as out of scope"
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
    on_start: Callable[[LiveRun], None] | None = None,
    observer_llm: LLM | None = None,
) -> Transcript:
    """Run ``path`` once in ``mode`` and return the transcript (never raises for run failures).

    ``llm`` may be ``None`` only when ``dry_run`` is true. ``catalog`` is the *allowed* catalog
    (the runner applies ``scenario.tools`` once); when not given it is discovered from the
    session and filtered here, so the agent never sees a tool the scenario denies. ``on_start``
    is called with the :class:`LiveRun` handle after the initial disclosure and before the
    first turn (a hand-held observer keeps it to call ``offer_tools`` / ``enable_goal``).
    ``observer_llm`` serves the scenario's LLM observers (default: ``llm``); their usage counts
    against ``max_cost_usd`` like the agent's.
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
        try:
            return await _dry_run(run, path, mode, session, catalog)
        except Exception as exc:  # noqa: BLE001 - a run never crashes; the reason is recorded
            return run.end("error", _describe(exc))
    assert llm is not None  # for the type checker; guarded above

    try:
        if catalog is None:
            catalog = await session.catalog()
        catalog = catalog.filtered(scenario.tools.allow, scenario.tools.deny)
        scope = ToolScope(scenario, catalog, run.transcript)
        scope.disclose_initial(path, mode)
        observers = ObserverRunner(scenario, observer_llm if observer_llm is not None else llm)
        live = LiveRun(scope, run.transcript, observers)
        if on_start is not None:
            on_start(live)
        user = SimulatedUser(scenario, llm, user_model)

        opening, usage = await user.open()
        run.transcript.add(UserEvent(text=opening))
        reason = run.record_usage(user_model, usage)
        if reason is not None:
            return run.end("budget_exceeded", reason)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content(opening, live.take_goals())}
        ]

        while True:
            reason = run.turn_budget_reason()
            if reason is not None:
                return run.end("budget_exceeded", reason)
            response = await llm.complete(
                model=agent_model,
                system=build_agent_system_prompt(scenario, path, mode, live.goals),
                messages=list(messages),
                tools=scope.definitions() or None,
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
            reason = run.record_usages(await live.observe("turn"))
            if reason is not None:
                return run.end("budget_exceeded", reason)

            if tool_uses:
                results: list[dict[str, Any]] = []
                for block in tool_uses:
                    name = str(block.get("name", ""))
                    tool_use_id = str(block.get("id", "")) or None
                    raw_input = block.get("input")
                    arguments = dict(raw_input) if isinstance(raw_input, dict) else {}
                    why = scope.refusal(name)
                    if why is not None:
                        results.append(run.refuse(name, why, tool_use_id))
                        continue
                    if name == DISCOVER_TOOL_NAME:
                        result = run.discover(scope, arguments, tool_use_id)
                        results.append(tool_result_block(tool_use_id or "", result))
                        continue
                    reason = run.tool_budget_reason(name)
                    if reason is not None:
                        return run.end("budget_exceeded", reason)
                    result = await run.call(session, name, arguments, tool_use_id)
                    results.append(tool_result_block(tool_use_id or "", result))
                reason = run.record_usages(await live.observe("tool_result"))
                if reason is not None:
                    return run.end("budget_exceeded", reason)
                messages.append(
                    {"role": "user", "content": user_content(results, live.take_goals())}
                )
                continue

            final = extract_final_result(text)
            if final.found:
                run.transcript.add(FinalResultEvent(parsed=final.parsed, raw=final.raw))
                run.transcript.final_result = final.parsed
                if final.error is not None:
                    run.transcript.add(ErrorEvent(message=final.error))
                run.record_usages(await live.observe("end"))
                if final.error is not None:
                    return run.end("completed", f"final answer delivered but {final.error}")
                return run.end("completed", "final answer delivered")

            # No tool use and no final block: the agent is talking to the person.
            reply, usage = await user.reply(text)
            run.transcript.add(UserEvent(text=reply))
            reason = run.record_usage(user_model, usage)
            if reason is not None:
                return run.end("budget_exceeded", reason)
            messages.append({"role": "user", "content": user_content(reply, live.take_goals())})
    except Exception as exc:  # noqa: BLE001 - a run never crashes; the reason is recorded
        return run.end("error", _describe(exc))
