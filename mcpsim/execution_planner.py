"""The local planner profile: the model plans the tool execution, the framework builds the tests.

Used for an ``ollama:`` planner (:func:`mcpsim.planner.planner_profile`); the hosted profile is
untouched. docs/LOCAL_MODELS.md "The execution planner" has the measurements behind it.

Asked to write whole test paths, ``command-r7b`` on a CPU wrote plans that validate and test
almost nothing: three paths sending the same call while expecting three different results, a
"recovery" whose failing step sent good input, a happy path whose one checkpoint ignored what
the goal asks for. Asked instead to *plan the tool execution for the user's request*, the same
model gets the main call right. So the work is split:

1. **The model plans the execution** in one constrained call (plus one re-ask): ``steps``
   (``tool`` from the disclosed set, ``arguments``, ``why``, ``expect``) and ``answer_fields``:
   for every key of ``expected_outcome.json``, the step and the path in its result the value
   comes from (``"price": "step 1: items[0].price"``, a single-anchored grammar pattern).
   Arguments are schema-checked like any plan step (:func:`mcpsim.planner.validate_arguments`);
   lineage must name a step that exists and calls a tool, and a path that the tool's output
   schema declares and, when the scout made the same call, that the observed result has with a
   value that satisfies the field's own expected-outcome spec (put to the matcher as the final
   result the lineage implies, so ``[*]``, ``[any]`` and a projection such as
   ``available_slugs ← result[*].slug`` are judged as the answer will be). A field over every
   element (``lines[*].price``) must not be read from one (``summary.lines[0].price``); the
   ``[*]`` rewrite is offered. When the result holds the field
   under its own name nearby (:func:`nearest_field_path`: ``items[0].store`` for ``store`` read
   from ``items[0].brand``, the top-level ``query`` for ``query`` read from ``items[0].name``,
   the always-present ``summary.total_cost`` for ``total_cost`` read from the optional
   ``full``), the problem carries that path, and a re-asked answer that still has it is
   repaired with it. A step no answer field and no later step uses, or one that repeats an
   earlier call, is dropped.
2. **The framework builds the happy path** from those steps plus a final answer step, with
   checkpoints derived deterministically: ``final_result: <key> equals tool_result[<tool>]
   <path>`` per answer field and ``final_result: <key> <spec>`` per ``expected_outcome.json``
   entry, operators written out (``is greater than 0``).
3. **The framework grounds one variant in a live probe** on the scout's read-only session
   (:func:`mcpsim.scout.probe`): the first happy step that calls a read-only, free tool with a
   string or integer argument is sent again with that argument mutated (a string loses one
   interior character, an integer becomes 999999). An error makes a *recovery* path that quotes
   the server's real error; an answer makes a *boundary* path whose checkpoints state what the
   server really returned, from structured content only and never a value that changes on
   every call. A write tool, one that claims a cost, one that reaches outside the server
   (``openWorldHint``, an internet description) and a call that sends a URL, host name or
   e-mail address are never probed; a probe the server answers badly costs the variant only.
4. **Policy paths from instructions**: for up to three instructions that prohibit something,
   one tiny constrained question, "which of these tools does this instruction forbid calling?"
   (an enum of the allowed tools plus ``none``), and a policy path per tool named, never one
   the happy path itself calls and only one the prohibiting clause names.

No two paths have identical steps. Every model call's tokens and seconds, and everything
dropped, repaired or skipped and why, are in the plan's ``notes``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mcpsim.llm import LLM, LLMResponse
from mcpsim.matcher import MISSING, apply_operator, match, parse_path, resolve
from mcpsim.mcpclient import Catalog, Session, ToolInfo
from mcpsim.plan import ExecutionPlan, Path, Step, StepReference, parse_reference
from mcpsim.planner import (
    MAX_PLAN_ATTEMPTS,
    PlanDraft,
    PlanError,
    PlannerView,
    enum_members,
    planner_prompt_budget,
    planner_view,
    render_tool_line,
    resolve_ref,
    schema_types,
    validate_arguments,
    validate_checkpoint,
    validate_draft,
)
from mcpsim.scenario import Scenario
from mcpsim.scoping import DISCOVER_TOOL_NAME, discover_tool_definition, expected_top_level_keys
from mcpsim.scoping import tokens as scoping_tokens
from mcpsim.scout import (
    Observation,
    ProbeRefusedError,
    ScoutResult,
    probe,
    probe_argument_refusal,
    probe_refusal,
)

EXECUTION_TOOL_NAME = "tool_execution_plan"
POLICY_TOOL_NAME = "forbidden_tool"
# On a CPU a 7-8B model reads about 20 prompt tokens/s and writes 3-4, so a two-minute answer
# allows roughly 1,100 prompt tokens: a prompt of about 4,000 characters.
LOCAL_PROMPT_BUDGET = 4_000
LOCAL_PLAN_MAX_TOKENS = 1_200
LOCAL_MAX_STEPS = 6
LOCAL_DESCRIPTION_LIMIT = 90
LOCAL_MAX_OUTPUT_KEYS = 6
WHY_CAP = 120
EXPECT_CAP = 160
# What the lineage grammar admits per path: keys of at most 40 characters, at most two list
# steps after a key, at most six keys, indices of at most three digits.
LINEAGE_KEY_CAP = 40
LINEAGE_LIST_STEPS_CAP = 2
LINEAGE_KEYS_CAP = 6
OBSERVATION_LINE_LIMIT = 260
MAX_POLICY_QUESTIONS = 3
POLICY_MAX_TOKENS = 40
NONE_ANSWER = "none"
PROBE_INTEGER = 999_999
MAX_BOUNDARY_FACTS = 2
FACT_VALUE_LIMIT = 40
HAPPY_PATH_ID = "happy"
UNUSED = "no answer field and no later step uses its result"

SYSTEM_OPENING = "You are an expert JSON config generator. Generate a JSON config of the format:"
SYSTEM_TOOLS = "that represents the plan of tool execution, using only these tools:"
FORMAT_STEPS = (
    '"steps":[{"tool":"<tool name>","arguments":{"<argument>":<value>},'
    '"why":"<one sentence>","expect":"<what the result will show>"}]'
)
FORMAT_FIELDS = (
    '"answer_fields":{"<field the answer must contain>":'
    '"step <n>: <path in the result of step n, e.g. items[0].price>"}'
)

# What the validator accepts as lineage. The grammar pattern (lineage_grammar_pattern) admits
# exactly the well-formed paths within its caps, with step numbers up to the plan's step cap, so
# every string it lets the model write passes these two (a test checks it on generated paths).
LINEAGE = re.compile(r"^step ([1-9]):\s*(\S+)$")
WELL_FORMED_PATH = re.compile(
    r"^[A-Za-z0-9_]+(\[(\d+|\*)\])*(\.[A-Za-z0-9_]+(\[(\d+|\*)\])*)*$"
)
# An instruction is worth a policy question only when it prohibits or restricts something.
PROHIBITION = re.compile(
    r"\b(do not|don't|never|not|no|avoid|without|instead of|rather than|only)\b", re.IGNORECASE
)


def lineage_grammar_pattern(max_steps: int = LOCAL_MAX_STEPS) -> str:
    """The JSON-schema ``pattern`` every ``answer_fields`` value must match.

    ``step <n>: <path>`` with ``n`` from 1 to ``max_steps`` and the structure
    :data:`WELL_FORMED_PATH` checks: keys of letters, digits and ``_`` joined by ``.``, each
    followed by up to two list steps ``[<index>]`` or ``[*]`` (``items[0].price``,
    ``summary.lines[*].origin_country``), within the ``LINEAGE_*`` caps. A character class
    alone would admit ``.price``, ``a..b`` or ``items[any]``, each a wasted re-ask.

    llama.cpp converts a pattern only when ONE ``^…$`` wraps the whole expression (an anchor
    inside an alternation is logged as unsupported and the string goes unconstrained), so there
    is exactly one pair here; literal ``.``, ``*`` and brackets are written as character classes
    (``[.]``, ``[*]``, ``[\\[]``), the form the earlier single-class pattern used live.
    """
    if not 1 <= max_steps <= 9:
        raise ValueError(f"max_steps must be between 1 and 9, got {max_steps}")
    digits = "1" if max_steps == 1 else f"1-{max_steps}"
    key = "[A-Za-z0-9_]{1," + str(LINEAGE_KEY_CAP) + "}"
    lists = "([\\[]([0-9]{1,3}|[*])[\\]]){0," + str(LINEAGE_LIST_STEPS_CAP) + "}"
    more = "([.]" + key + lists + "){0," + str(LINEAGE_KEYS_CAP - 1) + "}"
    return f"^step [{digits}]: {key}{lists}{more}$"


def step_tool_names(view: PlannerView) -> list[str]:
    """What a planned step may call: the disclosed tools, plus ``discover_tools`` if offered."""
    names = list(view.disclosed_names)
    if view.discoverable:
        names.append(DISCOVER_TOOL_NAME)
    return names


def execution_schema(tool_names: list[str], fields: list[str], max_steps: int) -> dict[str, Any]:
    """The grammar of the execution plan: steps over ``tool_names`` and one lineage per field.

    Every key is required and no other key is allowed (a constrained decoder only forces what
    the schema requires); ``why`` and ``expect`` are capped so the worst-case output of a call
    is bounded; ``answer_fields`` is left out when the scenario has no ``expected_outcome.json``.
    """
    step = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool", "arguments", "why", "expect"],
        "properties": {
            "tool": {"type": "string", "enum": list(tool_names)},
            "arguments": {"type": "object"},
            "why": {"type": "string", "maxLength": WHY_CAP},
            "expect": {"type": "string", "maxLength": EXPECT_CAP},
        },
    }
    properties: dict[str, Any] = {
        "steps": {"type": "array", "minItems": 1, "maxItems": max_steps, "items": step}
    }
    required = ["steps"]
    if fields:
        pattern = lineage_grammar_pattern(max_steps)
        properties["answer_fields"] = {
            "type": "object",
            "additionalProperties": False,
            "required": list(fields),
            "properties": {name: {"type": "string", "pattern": pattern} for name in fields},
        }
        required.append("answer_fields")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def policy_schema(tool_names: list[str]) -> dict[str, Any]:
    """One answer: a tool name or ``none``."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool"],
        "properties": {"tool": {"type": "string", "enum": [*tool_names, NONE_ANSWER]}},
    }


# --- the prompt -------------------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _plain(value: Any) -> str:
    """A value as a checkpoint or note shows it: strings bare, everything else as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def result_shape(value: Any, *, depth: int = 0) -> str:
    """The keys of a structured result, nested, with short top-level scalars shown.

    ``{query="penne", tokens[1], match="direct", total=2, items[2]{id, name, …, store, price},
    note}``: what the model needs to write lineage (``items[0].store``), in a fraction of the
    characters the values would take.
    """
    if isinstance(value, dict):
        parts = [f"{key}{_shape_suffix(inner, depth)}" for key, inner in value.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, list):
        inner = result_shape(value[0], depth=depth + 1) if value and depth < 2 and isinstance(
            value[0], dict) else ""
        return f"[{len(value)}]{inner}"
    return json.dumps(value, ensure_ascii=False)


def _shape_suffix(value: Any, depth: int) -> str:
    if isinstance(value, dict):
        return result_shape(value, depth=depth + 1) if depth < 2 else "{…}"
    if isinstance(value, list):
        return result_shape(value, depth=depth)
    if depth == 0:
        text = json.dumps(value, ensure_ascii=False)
        return f"={text}" if len(text) <= 24 else ""
    return ""


def observation_line(observation: Observation, *, limit: int = OBSERVATION_LINE_LIMIT) -> str:
    """``- find_product(query="penne") → {query="penne", …}``; errors and text are clipped."""
    if observation.kind != "tool":
        return f"- {observation.name} → {_clip(observation.summary, limit)}"
    if observation.is_error:
        return f"- {observation.call_label()} → ERROR: {_clip(observation.summary, limit)}"
    if observation.structured is not None:
        shape = result_shape(observation.structured)
        return f"- {observation.call_label()} → {_clip(shape, limit)}"
    return f"- {observation.call_label()} → {_clip(observation.summary, limit)}"


def _ordered_observations(view: PlannerView) -> list[Observation]:
    """The scout's tool calls, expected-outcome lookups first. Resource reads are left out: a
    step can only call a tool, and their summaries cost hundreds of prompt tokens."""
    tools = [o for o in view.observations if o.kind == "tool" and not o.probe]
    return [o for o in tools if o.from_expected] + [o for o in tools if not o.from_expected]


def execution_prompts(
    scenario: Scenario, view: PlannerView, fields: list[str], *, budget: int
) -> tuple[str, str]:
    """``(system, user)`` in the user's own framing, within ``budget`` characters when it fits.

    System: the "expert JSON config generator" opening, the one-line format, and the compact
    digest of the tools a step may name. User: "The user's request: <goal>", then "Rules the plan
    must follow:" with one line per instruction and per goal an observer enabled. The informant
    reports and the scout's observations (as result shapes) are added line by line, in that
    order, only while the total stays within ``budget`` (resources are never shown); the digest,
    the request and the rules are never cut.
    """
    tools = list(view.disclosed)
    if view.discoverable:
        tools.append(ToolInfo.model_validate(discover_tool_definition()))
    digest = [
        render_tool_line(t, description_limit=LOCAL_DESCRIPTION_LIMIT,
                         max_output_keys=LOCAL_MAX_OUTPUT_KEYS)
        for t in tools
    ]
    fmt = "{" + FORMAT_STEPS + ("," + FORMAT_FIELDS if fields else "") + "}"
    system = "\n".join([SYSTEM_OPENING, fmt, "", SYSTEM_TOOLS, *digest])
    rules = [" ".join(i.split()) for i in scenario.instructions]
    rules += [" ".join(g.split()) for g in view.goals]
    user = f"The user's request: {' '.join(scenario.goal.split())}"
    if rules:
        user += "\n\nRules the plan must follow:\n" + "\n".join(f"- {r}" for r in rules)
    sections = [
        ("What informants reported about the server:", [f"- {r.line()}" for r in view.reports]),
        (
            "What the server already returned (read-only calls):",
            [observation_line(o) for o in _ordered_observations(view)],
        ),
    ]
    for header, lines in sections:
        added = False
        for line in lines:
            extra = (f"\n\n{header}\n" if not added else "\n") + line
            if len(system) + len(user) + len(extra) > budget:
                break
            user += extra
            added = True
    return system, user


# --- the model's answer -------------------------------------------------------------------------


class PlannedCall(BaseModel):
    """One step of the model's execution plan."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    why: str = ""
    expect: str = ""

    def as_step(self) -> Step:
        return Step(
            intent=self.why.strip() or f"Call {self.tool}",
            tool=self.tool,
            arguments_sketch=dict(self.arguments),
            success_looks_like=self.expect.strip(),
            expect_error=False,
        )

    def references(self) -> list[StepReference]:
        found: list[StepReference] = []
        for value in self.arguments.values():
            try:
                ref = parse_reference(value)
            except ValueError:
                continue
            if ref is not None:
                found.append(ref)
        return found


class ToolExecution(BaseModel):
    """The model's whole answer: the calls, and where each answer field comes from."""

    model_config = ConfigDict(extra="forbid")

    steps: list[PlannedCall] = Field(default_factory=list)
    answer_fields: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class Lineage:
    """``step <step>: <path>``: an answer field comes from that step's result at ``path``."""

    step: int
    path: str

    def text(self) -> str:
        return f"step {self.step}: {self.path}"


@dataclass
class Review:
    """What is wrong with one answer, in the model's own step numbering."""

    problems: list[str] = field(default_factory=list)
    bad_steps: dict[int, list[str]] = field(default_factory=dict)
    bad_fields: dict[str, str] = field(default_factory=dict)
    repairs: dict[str, Lineage] = field(default_factory=dict)
    lineage: dict[str, Lineage] = field(default_factory=dict)
    duplicate_of: dict[int, int] = field(default_factory=dict)
    unused: set[int] = field(default_factory=set)


def parse_execution(response: LLMResponse) -> tuple[ToolExecution | None, list[str]]:
    blocks = [b for b in response.tool_uses() if b.get("name") == EXECUTION_TOOL_NAME]
    if not blocks:
        return None, ["the answer held no JSON config; answer with exactly one JSON object"]
    raw = blocks[0].get("input")
    if not isinstance(raw, dict):
        return None, ["the JSON config must be an object"]
    try:
        return ToolExecution.model_validate(raw), []
    except ValidationError as exc:
        return None, [
            f"{'.'.join(str(p) for p in err['loc']) or 'config'}: {err['msg']}"
            for err in exc.errors()
        ]


def _tool_info(catalog: Catalog, name: str) -> ToolInfo:
    if name == DISCOVER_TOOL_NAME:
        return ToolInfo.model_validate(discover_tool_definition())
    return catalog.tool(name)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _concrete(schema: Any, root: dict[str, Any]) -> dict[str, Any]:
    """A property schema with ``$ref`` followed and an ``Optional`` (``T | null``) unwrapped."""
    node = resolve_ref(schema, root) if isinstance(schema, dict) else {}
    for key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            concrete = [
                resolve_ref(v, root) for v in variants
                if isinstance(v, dict) and v.get("type") != "null"
            ]
            if len(concrete) == 1:
                return concrete[0]
    return node


def schema_lookup(schema: dict[str, Any], path: str) -> tuple[bool | None, str]:
    """Does ``path`` exist in a result that matches the output ``schema``?

    ``(True, "")`` when every key on the way is declared, ``(None, "")`` when the schema stops
    saying (a free-form object, an untyped list), ``(False, why)`` when a key is not among the
    declared properties of a closed object (pydantic models, which servers publish, never
    return extra keys) or a list step meets something that is not a list.
    """
    root = schema
    node = _concrete(schema, root)
    walked = ""
    for seg in parse_path(path):
        if seg.kind == "key":
            properties = node.get("properties")
            if not isinstance(properties, dict) or not properties:
                return None, ""
            if seg.key not in properties:
                extra = node.get("additionalProperties")
                if extra is True or isinstance(extra, dict):
                    return None, ""
                where = walked or "the result"
                return False, f"{where} has no key {seg.key!r} (its keys: {', '.join(properties)})"
            node = _concrete(properties[seg.key], root)
            walked = f"{walked}.{seg.key}" if walked else str(seg.key)
        else:
            items = node.get("items")
            if isinstance(items, dict):
                node = _concrete(items, root)
            else:
                types = schema_types(node, root)
                if types and "array" not in types:
                    return False, f"{walked or 'the result'} is not a list"
                return None, ""
            walked += "[*]" if seg.kind == "every" else f"[{seg.index}]"
    return True, ""


def observed_lookup(value: Any, path: str) -> bool:
    """Does ``path`` lead to a value in an observed result (``[*]`` over an empty list: yes)?"""
    try:
        found = resolve(value, path)
    except ValueError:
        return False
    if not found.present:
        return False
    return not found.values or any(v is not MISSING for v in found.values)


def _with_defaults(tool: ToolInfo | None, arguments: dict[str, Any]) -> dict[str, Any]:
    """``arguments`` with every unsent property that declares a ``default`` filled in."""
    properties = (tool.input_schema or {}).get("properties") if tool is not None else None
    defaults = {
        str(k): v["default"] for k, v in (properties or {}).items()
        if isinstance(v, dict) and "default" in v
    }
    return {**defaults, **arguments}


def _observed_result(
    observations: list[Observation], call: PlannedCall, tool: ToolInfo | None = None
) -> Any | None:
    """The structured result the scout got from the same call, or ``None``. Arguments are
    compared with the tool's schema defaults filled in on both sides (``list_items()`` is
    ``list_items(cursor=0, limit=2)``); any other difference (a page, a filter, a limit) can
    change the result, so it is not the same call."""
    wanted = _with_defaults(tool, call.arguments)
    for o in observations:
        if o.kind != "tool" or o.probe or o.is_error or o.structured is None:
            continue
        if o.name == call.tool and _with_defaults(tool, o.arguments) == wanted:
            return o.structured
    return None


def _leaf_key(path: str) -> str | None:
    segments = parse_path(path)
    keys = [s.key for s in segments if s.kind == "key"]
    return str(keys[-1]) if keys else None


def _quantifier(path: str) -> str:
    """``every`` or ``any`` for a path with that list step, ``one`` otherwise."""
    kinds = [s.kind for s in parse_path(path) if s.kind in ("every", "any")]
    return kinds[0] if kinds else "one"


def _place(segments: list[Any], value: Any) -> Any:
    """``value`` placed at ``segments`` in an otherwise empty result; at a ``[*]``/``[any]``
    step ``value`` is the list of per-element values. A missing value leaves its key out."""
    if not segments:
        return value
    seg, rest = segments[0], segments[1:]
    if seg.kind in ("every", "any"):
        return [_place(rest, v) for v in value]
    inner = _place(rest, value)
    if seg.kind == "index":
        return [None] * seg.index + ([] if inner is MISSING else [inner])
    return {} if inner is MISSING else {seg.key: inner}


def final_from_lineage(name: str, observed: Any, path: str) -> Any | None:
    """The ``final_result`` an answer built from this lineage would hold for field ``name``,
    or ``None`` when ``path`` does not resolve in ``observed``.

    A field over a list (``lines[*].price``, ``items[any].slug``) takes one element per value
    the lineage resolves to (one element for a single value); a field without a list step takes
    the lineage's value itself, so ``available_slugs ← result[*].slug`` is the list of slugs.
    """
    try:
        found = resolve(observed, path)
        segments = parse_path(name)
    except ValueError:
        return None
    if not found.present or sum(s.kind in ("every", "any") for s in segments) > 1:
        return None  # the matcher itself allows one [*] or [any] per path
    if _quantifier(name) != "one":
        elements = list(found.values) if found.quantifier != "one" else [found.single]
        return _place(segments, elements)
    if found.quantifier != "one":
        return _place(segments, [v for v in found.values if v is not MISSING])
    return _place(segments, found.single)


def lineage_value_failure(name: str, spec: Any, observed: Any, path: str) -> str | None:
    """Why the answer this lineage gives would fail field ``name``'s ``expected_outcome.json``
    spec, or ``None``.

    The final result the lineage implies (:func:`final_from_lineage`) is put to the matcher
    itself, so the quantifiers are exactly the ones the final answer will face: every value
    for a ``[*]`` field, at least one for an ``[any]`` field, the projected list as a whole for
    a field without a list step (``$len`` and ``$contains`` on ``available_slugs``).
    """
    final = final_from_lineage(name, observed, path)
    if final is None:
        return None
    for result in match({name: spec}, final):
        if result.passed:
            continue
        words = describe_operator(result.op, result.expected)
        quantifier = _quantifier(name)
        if quantifier != "one" and isinstance(result.actual, list):
            if quantifier == "any":
                shown = ", ".join(_clip(_plain(v), 30) for v in result.actual[:4])
                more = ", …" if len(result.actual) > 4 else ""
                return f"{shown or 'no element'}{more}, none of which passes '{words}'"
            for value in result.actual:
                passed, _ = apply_operator(
                    result.op, result.expected, value, present=value is not MISSING
                )
                if not passed:
                    shown = "nothing" if value is MISSING else _clip(_plain(value), 60)
                    return f"{shown}, which fails '{words}'"
        shown = "nothing" if result.actual is MISSING else _clip(_plain(result.actual), 60)
        return f"{shown}, which fails '{words}'"
    return None


def quantifier_mismatch(name: str, path: str) -> tuple[str, str | None] | None:
    """``(why, rewritten path or None)`` when a field over every (or any) element of a list is
    read from a lineage that names one element, else ``None``.

    ``lines[*].price ← summary.lines[0].price`` would claim every line costs what the first
    one does: where the lineage ends in the field's own path, each list step the field
    quantifies must be ``[*]`` in the lineage too, and the rewrite is the lineage with those
    steps as ``[*]`` (``summary.lines[*].price``). A lineage that ends elsewhere needs a ``[*]``
    somewhere. A field without a list step never mismatches (``available_slugs ←
    result[*].slug`` is a projection).
    """
    field_segments = parse_path(name)
    quantified = [i for i, s in enumerate(field_segments) if s.kind in ("every", "any")]
    if not quantified:
        return None
    wanted = "every element" if field_segments[quantified[0]].kind == "every" else "any element"
    segments = parse_path(path)
    offset = len(segments) - len(field_segments)
    if offset >= 0 and _normalised(segments[offset:]) == _normalised(field_segments):
        single = [offset + i for i in quantified if segments[offset + i].kind != "every"]
        if not single:
            return None
        rewritten = [
            replace(s, kind="every", index=0) if i in single else s
            for i, s in enumerate(segments)
        ]
        text = _rendered(rewritten)
        one = _rendered(segments[: single[0] + 1])
        return (
            f"reads one element ({one}) where {name} is about {wanted} of the list",
            text if text.count("[*]") <= 1 else None,
        )
    if any(s.kind == "every" for s in segments):
        return None
    return f"reads a single value where {name} is about {wanted} of a list", None


def _observed_values(observed: Any, path: str) -> list[Any] | None:
    try:
        found = resolve(observed, path)
    except ValueError:
        return None
    return list(found.values) if found.present else None


_LIST = "[]"


def _normalised(segments: list[Any]) -> tuple[str, ...]:
    """Key names, with every list step (``[0]``, ``[*]``, ``[any]``) as ``[]``."""
    return tuple(str(s.key) if s.kind == "key" else _LIST for s in segments)


def _rendered(segments: list[Any]) -> str:
    out = ""
    for seg in segments:
        if seg.kind == "key":
            out += f".{seg.key}" if out else str(seg.key)
        else:
            out += {"every": "[*]", "any": "[any]"}.get(seg.kind, f"[{seg.index}]")
    return out


@dataclass(frozen=True)
class FieldNode:
    """One path a result can hold (list steps normalised); ``sure`` when it is always there:
    every key on the way required and not nullable (observed: present and not null)."""

    keys: tuple[str, ...]
    sure: bool


def schema_tree(schema: dict[str, Any], *, max_depth: int = 6) -> list[FieldNode]:
    """Every path an output schema declares, through ``$ref``, optionals and list items."""
    root = schema
    found: list[FieldNode] = []

    def walk(node: Any, keys: tuple[str, ...], sure: bool, depth: int) -> None:
        if depth > max_depth:
            return
        concrete = _concrete(node, root)
        properties = concrete.get("properties")
        if isinstance(properties, dict):
            required_raw = concrete.get("required")
            required = set(required_raw) if isinstance(required_raw, list) else set()
            for key, sub in properties.items():
                sub_schema = sub if isinstance(sub, dict) else {}
                nullable = "null" in schema_types(sub_schema, root)
                here = keys + (str(key),)
                always = sure and key in required and not nullable
                found.append(FieldNode(here, always))
                walk(sub_schema, here, always, depth + 1)
        items = concrete.get("items")
        if isinstance(items, dict):
            walk(items, keys + (_LIST,), sure, depth + 1)

    walk(schema, (), True, 0)
    return found


def observed_tree(value: Any, *, max_depth: int = 6) -> list[FieldNode]:
    """Every path an observed result holds (a list through its first element)."""
    found: list[FieldNode] = []

    def walk(node: Any, keys: tuple[str, ...], depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(node, dict):
            for key, inner in node.items():
                here = keys + (str(key),)
                found.append(FieldNode(here, inner is not None))
                walk(inner, here, depth + 1)
        elif isinstance(node, list) and node:
            walk(node[0], keys + (_LIST,), depth + 1)

    walk(value, (), 0)
    return found


def nearest_field_path(
    name: str,
    lineage: Lineage,
    tree: list[FieldNode],
    accept: Any = None,
) -> Lineage | None:
    """Where the result holds the answer field itself, nearest to what the lineage named.

    Candidates are paths ending in the field's own path (``store``; ``coverage.spend_fraction``;
    ``lines[*].price``), searched in the lineage's own subtree first, then in each enclosing
    object up to the root: ``query`` read from ``items[0].name`` finds the top-level ``query``;
    ``total_cost`` read from ``full`` finds ``summary.total_cost``. In a scope, a path that is
    always there beats one that may be absent (a nullable, unrequired ``full``), then fewer list
    steps beyond the field's own, then the shorter path; a tie is no answer. A scope with only
    maybe-absent candidates is remembered while wider scopes are searched for a sure one.
    ``accept`` (a rendered path → bool) filters candidates, e.g. by the field's expected value.

    A list step inside the field's own path is written as the field writes it: ``[*]`` for a
    ``[*]`` or ``[any]`` field (``lines[*].price ← summary.lines[0].price`` finds
    ``summary.lines[*].price``), the index for an indexed one. Any other list step keeps the
    lineage's own inside the scope and is ``[0]`` beyond it.
    """
    field_segments = parse_path(name)
    tail = _normalised(field_segments)
    if not tail:
        return None
    segments = parse_path(lineage.path)
    fallback: Lineage | None = None
    for cut in range(len(segments), -1, -1):
        scope = _normalised(segments[:cut])
        options: list[tuple[tuple[bool, int, int], str]] = []
        for node in tree:
            if node.keys[: len(scope)] != scope or len(node.keys) < max(len(scope), len(tail)):
                continue
            if node.keys[-len(tail):] != tail:
                continue
            start = len(node.keys) - len(tail)
            path = ""
            for position, key in enumerate(node.keys):
                if key != _LIST:
                    path += f".{key}" if path else key
                elif position >= start:
                    own = field_segments[position - start]
                    path += f"[{own.index}]" if own.kind == "index" else "[*]"
                elif position < cut:
                    path += _rendered([segments[position]])
                else:
                    path += "[0]"
            if accept is not None and not accept(path):
                continue
            extra_lists = node.keys[len(scope):].count(_LIST) - tail.count(_LIST)
            options.append(((not node.sure, max(extra_lists, 0), len(node.keys)), path))
        if not options:
            continue
        options.sort()
        best = [o for o in options if o[0] == options[0][0]]
        rank = best[0][0]
        if not rank[0]:  # always present
            return Lineage(lineage.step, best[0][1]) if len(best) == 1 else fallback
        # Only maybe-absent candidates here: remember a unique one, keep looking wider for a
        # path that is always there (a wider scope sees these candidates too, ranked lower).
        if fallback is None and len(best) == 1:
            fallback = Lineage(lineage.step, best[0][1])
    return fallback


def _is_sure(tree: list[FieldNode], path: str) -> bool:
    keys = _normalised(parse_path(path))
    return next((n.sure for n in tree if n.keys == keys), True)


def review_execution(
    draft: ToolExecution,
    catalog: Catalog,
    view: PlannerView,
    fields: list[str],
    observations: list[Observation],
    max_steps: int,
    expected: dict[str, Any] | None = None,
) -> Review:
    """Every problem with the model's answer, numbered as the model numbered its steps.

    Steps a later stage drops anyway (one that repeats an earlier call; with answer fields, one
    neither an answer field nor a later step uses) are reviewed but their problems are not sent
    back. A lineage must name a tool step, a path the tool's output schema declares and, when the
    scout made the same call, a path the observed result has, whose value satisfies the field's
    own spec in ``expected`` (``expected_outcome.json``, :func:`lineage_value_failure`); a field
    over every or any element must not be read from one element (:func:`quantifier_mismatch`);
    one that reads a key of another name while a key with the field's name is there gets that
    key as the correction.
    """
    review = Review()
    calls = draft.steps
    count = len(calls)
    if count == 0:
        review.problems.append("steps is empty; plan at least one tool call")
        return review
    if count > max_steps:
        review.problems.append(f"at most {max_steps} steps are allowed, got {count}")
        for i in range(max_steps + 1, count + 1):
            review.bad_steps[i] = [f"beyond the {max_steps}-step cap"]
    for name in fields:
        if name not in draft.answer_fields:
            review.bad_fields[name] = "missing; say which step's result it comes from"
    for name, text in draft.answer_fields.items():
        if name not in fields:
            review.bad_fields[name] = (
                f"not a field of the expected outcome (its fields: {', '.join(fields) or 'none'})"
            )
            continue
        match = LINEAGE.match(text.strip())
        if match is None:
            review.bad_fields[name] = f"{text!r} must read 'step <n>: <path in that result>'"
            continue
        step, path = int(match.group(1)), match.group(2)
        if not WELL_FORMED_PATH.match(path):
            review.bad_fields[name] = f"{path!r} is not a dotted path such as items[0].price"
        elif not 1 <= step <= count:
            review.bad_fields[name] = f"step {step} does not exist (the plan has {count} step(s))"
        else:
            review.lineage[name] = Lineage(step, path)

    seen: dict[str, int] = {}
    for i, call in enumerate(calls, start=1):
        key = f"{call.tool}:{_canonical(call.arguments)}"
        if key in seen:
            review.duplicate_of[i] = seen[key]
        else:
            seen[key] = i
    dup = review.duplicate_of
    if fields:
        used = {dup.get(lin.step, lin.step) for lin in review.lineage.values()}
        for call in calls:
            used.update(dup.get(ref.from_step, ref.from_step) for ref in call.references())
        review.unused = {i for i in range(1, count + 1) if i not in used and i not in dup}
    droppable = set(dup) | review.unused

    allowed = step_tool_names(view)
    as_path = Path(id="review", kind="happy", title="review", steps=[c.as_step() for c in calls])
    for i, call in enumerate(calls, start=1):
        if i in review.bad_steps:
            continue
        if call.tool not in allowed:
            issues = [f"{call.tool!r} is not one of the listed tools: {', '.join(allowed)}"]
        else:
            tool = _tool_info(catalog, call.tool)
            issues = validate_arguments(as_path.steps[i - 1], i, as_path, tool)
        if issues:
            review.bad_steps[i] = issues

    for name, lineage in list(review.lineage.items()):
        call = calls[dup.get(lineage.step, lineage.step) - 1]
        if call.tool not in allowed or not catalog.has_tool(call.tool):
            if call.tool == DISCOVER_TOOL_NAME:
                review.bad_fields[name] = (
                    f"step {lineage.step} calls {DISCOVER_TOOL_NAME}, whose result lists tools, "
                    "not answer values"
                )
            continue
        tool = catalog.tool(call.tool)
        observed = _observed_result(observations, call, tool)
        where = f"step {lineage.step} ({call.tool})"
        spec = expected.get(name) if expected else None
        tree = (
            observed_tree(observed) if observed is not None
            else schema_tree(tool.output_schema) if tool.output_schema else []
        )
        accept = None
        if observed is not None and spec is not None:
            def accept(
                path: str, observed: Any = observed, spec: Any = spec, name: str = name
            ) -> bool:
                values = _observed_values(observed, path)
                return bool(values) and lineage_value_failure(name, spec, observed, path) is None

        def holds(
            path: str, tool: ToolInfo = tool, observed: Any = observed, accept: Any = accept
        ) -> bool:
            """Would this path pass the checks below (declared, observed, accepted)?"""
            if tool.output_schema and schema_lookup(tool.output_schema, path)[0] is False:
                return False
            if observed is not None and not observed_lookup(observed, path):
                return False
            return accept is None or bool(accept(path))

        better = nearest_field_path(name, lineage, tree, accept)
        if better is not None and better.path == lineage.path:
            better = None
        problem: str | None = None
        declared, why = (
            schema_lookup(tool.output_schema, lineage.path) if tool.output_schema else (None, "")
        )
        values = _observed_values(observed, lineage.path) if observed is not None else None
        mismatch = quantifier_mismatch(name, lineage.path)
        if declared is False:
            problem = f"{where} cannot return {lineage.path}: {why}"
        elif observed is not None and not observed_lookup(observed, lineage.path):
            problem = (
                f"{lineage.path} is not in what {where} returned when the scout made that call: "
                f"{_clip(result_shape(observed), 200)}"
            )
        elif spec is not None and values and (
            failure := lineage_value_failure(name, spec, observed, lineage.path)
        ):
            problem = (
                f"{lineage.text()} gives {failure} (the expected outcome for {name}) in what "
                f"{where} returned when the scout made that call"
            )
        elif better is not None and _leaf_key(lineage.path) != _leaf_key(name):
            problem = f"{lineage.text()} reads {_leaf_key(lineage.path)!r}"
        elif mismatch is not None:
            problem = f"{lineage.text()} {mismatch[0]}"
            if better is None and mismatch[1] is not None and holds(mismatch[1]):
                better = Lineage(lineage.step, mismatch[1])
        elif better is not None and not _is_sure(tree, lineage.path):
            problem = (
                f"{lineage.text()} goes through a value the output schema marks optional (null "
                "or absent unless asked for)"
            )
        if problem is None:
            continue
        if better is not None:
            problem += (
                f"; the result has {_leaf_key(name)!r} at {better.path}: write '{better.text()}'"
            )
            review.repairs[name] = better
        review.bad_fields[name] = problem

    for i, issues in sorted(review.bad_steps.items()):
        if i in droppable and i <= max_steps:
            continue
        review.problems += [f"step {i} ({calls[i - 1].tool}): {issue}" for issue in issues]
    review.problems += [f"answer_fields.{n}: {why}" for n, why in review.bad_fields.items()]
    return review


def _remap_references(arguments: dict[str, Any], mapping: dict[int, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        try:
            ref = parse_reference(value)
        except ValueError:
            ref = None
        if ref is not None and ref.from_step in mapping:
            out[key] = {**value, "$from_step": mapping[ref.from_step]}
        else:
            out[key] = value
    return out


def settle_execution(
    draft: ToolExecution,
    review: Review,
    fields: list[str],
    notes: list[str],
    *,
    salvage: bool,
) -> tuple[list[PlannedCall], dict[str, Lineage]]:
    """The calls and lineage the happy path is built from, renumbered from 1.

    Repeated calls and (when lineage exists) unused steps are dropped; references and lineage
    pointing at a repeat move to the first call. With ``salvage`` (the re-asked answer still
    had problems) a field whose lineage names the field's sibling takes it, other bad fields
    are dropped (no lineage checkpoint, the expected-outcome one stays), invalid steps are
    dropped with everything that depends on them, and :class:`PlanError` is raised only when no
    tool step is left. Every drop and repair is noted.
    """
    count = len(draft.steps)
    dup = review.duplicate_of
    lineage = dict(review.lineage)
    dropped: dict[int, str] = {}
    if salvage:
        for name, why in review.bad_fields.items():
            if name in review.repairs:
                old = lineage.get(name)
                lineage[name] = review.repairs[name]
                notes.append(
                    f"answer_fields.{name}: the model kept '{old.text() if old else '?'}'; the "
                    f"result has '{review.repairs[name].path}' under the field's own name, used "
                    "instead"
                )
            elif name in fields:
                lineage.pop(name, None)
                notes.append(f"answer_fields.{name} dropped (no lineage checkpoint): {why}")
    dropped.update({i: "; ".join(issues) for i, issues in review.bad_steps.items()})
    if lineage:
        dropped.update({i: UNUSED for i in review.unused})
    for repeat, first in dup.items():
        dropped.setdefault(repeat, f"repeats step {first}")
    lineage = {n: Lineage(dup.get(lin.step, lin.step), lin.path) for n, lin in lineage.items()}
    changed = True
    while changed:
        changed = False
        for name, lin in list(lineage.items()):
            if lin.step in dropped:
                del lineage[name]
                notes.append(f"answer_fields.{name} dropped: step {lin.step} was dropped")
                changed = True
        for i, call in enumerate(draft.steps, start=1):
            if i in dropped:
                continue
            for ref in call.references():
                if dup.get(ref.from_step, ref.from_step) in dropped:
                    dropped[i] = f"it needs step {ref.from_step}, which was dropped"
                    changed = True
                    break
        if fields and lineage:
            used = {lin.step for lin in lineage.values()}
            for i, call in enumerate(draft.steps, start=1):
                if i not in dropped:
                    used.update(dup.get(r.from_step, r.from_step) for r in call.references())
            for i in range(1, count + 1):
                if i not in dropped and i not in used:
                    dropped[i] = UNUSED
                    changed = True
    for i in sorted(dropped):
        notes.append(f"step {i} ({draft.steps[i - 1].tool}) dropped: {_clip(dropped[i], 300)}")
    kept = [i for i in range(1, count + 1) if i not in dropped]
    if not any(draft.steps[i - 1].tool != DISCOVER_TOOL_NAME for i in kept):
        raise PlanError(
            "the local planner's execution plan has no valid tool step left; "
            + "; ".join(f"step {i}: {why}" for i, why in sorted(dropped.items()))
        )
    mapping = {old: new for new, old in enumerate(kept, start=1)}
    mapping.update({repeat: mapping[first] for repeat, first in dup.items() if first in mapping})
    calls = [
        draft.steps[i - 1].model_copy(
            update={"arguments": _remap_references(draft.steps[i - 1].arguments, mapping)}
        )
        for i in kept
    ]
    return calls, {name: Lineage(mapping[lin.step], lin.path) for name, lin in lineage.items()}


# --- deriving the paths -------------------------------------------------------------------------


_OP_WORDS = {
    "$eq": "equals",
    "$ne": "is not",
    "$gt": "is greater than",
    "$gte": "is at least",
    "$lt": "is less than",
    "$lte": "is at most",
    "$regex": "matches the pattern",
    "$contains": "contains",
    "$subset": "includes",
}


def _is_operator_spec(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        isinstance(k, str) and k.startswith("$") for k in value
    )


def describe_operator(op: str, expected: Any) -> str:
    """One matcher operator in words: ``is greater than 0``, ``is one of a, b``, ``is a number``."""
    if op in ("$in", "$nin") and isinstance(expected, list):
        members = ", ".join(_plain(m) for m in expected)
        return f"is one of {members}" if op == "$in" else f"is none of {members}"
    if op == "$exists":
        return "is present" if expected else "is absent"
    if op == "$type" and isinstance(expected, str):
        return f"is {'an' if expected[:1] in 'aeiou' else 'a'} {expected}"
    if op == "$len":
        if _is_operator_spec(expected):
            inner = " and ".join(describe_operator(o, v) for o, v in expected.items())
            return f"has a length that {inner}"
        return f"has exactly {_plain(expected)} entries"
    if op in _OP_WORDS:
        return f"{_OP_WORDS[op]} {_plain(expected)}"
    return f"satisfies {op} {_plain(expected)}"


def describe_expectation(spec: Any) -> str:
    """An ``expected_outcome.json`` value in words (a literal ``equals`` it)."""
    if _is_operator_spec(spec):
        return " and ".join(describe_operator(op, value) for op, value in spec.items())
    return f"equals {_plain(spec)}"


def expected_checkpoints(spec: dict[str, Any] | None) -> list[str]:
    """``final_result: <key> <spec in words>`` for every ``expected_outcome.json`` entry."""
    return [f"final_result: {key} {describe_expectation(value)}" for key, value in
            (spec or {}).items()]


def lineage_checkpoints(
    calls: list[PlannedCall],
    lineage: dict[str, Lineage],
    fields: list[str],
    *,
    labels: dict[int, str] | None = None,
) -> list[str]:
    """``final_result: <key> equals tool_result[<tool>] <path>``, in expected-outcome order.

    ``labels`` (call position → text) says which call is meant when a path calls the tool more
    than once; by default ``(step n)`` is added exactly when the happy path repeats the tool.
    """
    tools = [c.tool for c in calls]
    if labels is None:
        labels = {i: f"step {i}" for i, t in enumerate(tools, start=1) if tools.count(t) > 1}
    out: list[str] = []
    for name in fields:
        lin = lineage.get(name)
        if lin is None:
            continue
        which = f" ({labels[lin.step]})" if lin.step in labels else ""
        out.append(
            f"final_result: {name} equals tool_result[{tools[lin.step - 1]}] {lin.path}{which}"
        )
    return out


def _variant_labels(
    calls: list[PlannedCall], choice: ProbeChoice, original_text: str
) -> dict[int, str]:
    """Lineage labels for a variant path, where the mutated call is inserted before the probed
    step: that step is "the call with <original input>", repeated tools get their new number."""
    tools = [c.tool for c in calls] + [choice.tool.name]
    labels: dict[int, str] = {}
    for i, call in enumerate(calls, start=1):
        position = i + 1 if i >= choice.step else i
        if i == choice.step:
            labels[i] = f"step {position}, the call with {original_text}"
        elif tools.count(call.tool) > 1:
            labels[i] = f"step {position}"
    return labels


def _entries(count: int) -> str:
    return f"{count} {'entry' if count == 1 else 'entries'}"


def answer_step(scenario: Scenario) -> Step:
    keys = expected_top_level_keys(scenario.expected_outcome.json)
    return Step(
        intent="Compose the final answer from the results above",
        tool=None,
        success_looks_like=(
            f"A final_result with {', '.join(keys)} taken from the tool results"
            if keys else "An answer built only from the tool results"
        ),
    )


def happy_path(
    scenario: Scenario,
    calls: list[PlannedCall],
    lineage: dict[str, Lineage],
    fields: list[str],
) -> Path:
    checkpoints = lineage_checkpoints(calls, lineage, fields)
    checkpoints += expected_checkpoints(scenario.expected_outcome.json)
    if not checkpoints:
        checkpoints = ["final_result: answers the request with values taken from the tool results"]
    tools = " → ".join(c.tool for c in calls)
    return Path(
        id=HAPPY_PATH_ID,
        kind="happy",
        title=_clip(f"Happy path: {tools}", 100),
        rationale=(
            f"The tool execution {scenario.models.planner} planned for the user's request; the "
            "checkpoints are derived by the framework from its answer_fields lineage and from "
            "expected_outcome.json."
        ),
        steps=[c.as_step() for c in calls] + [answer_step(scenario)],
        checkpoints=checkpoints,
    )


@dataclass(frozen=True)
class ProbeChoice:
    """Which happy step to send again, and with which argument mutated how."""

    step: int  # 1-based position among the happy path's tool steps
    tool: ToolInfo
    argument: str
    original: Any
    mutated: Any

    def arguments(self, call: PlannedCall) -> dict[str, Any]:
        return {**call.arguments, self.argument: self.mutated}

    def when(self) -> str:
        return f"when {self.argument} is {_plain(self.mutated)}"

    def change(self) -> str:
        if isinstance(self.original, str):
            return f"{_plain(self.original)} with one character dropped"
        return f"{_plain(self.mutated)} instead of {_plain(self.original)}"


def mutate(value: Any) -> Any | None:
    """A string loses its middle interior character (``penne`` → ``pene``); an integer becomes
    999999; anything else (or a string under three characters) is not mutated."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return PROBE_INTEGER if value != PROBE_INTEGER else None
    if isinstance(value, str) and len(value) >= 3:
        middle = len(value) // 2
        return value[:middle] + value[middle + 1 :]
    return None


def choose_probe(
    calls: list[PlannedCall], catalog: Catalog
) -> tuple[ProbeChoice | None, list[str]]:
    """The first happy step whose tool may be probed and that has a mutable argument.

    A step is skipped when :func:`mcpsim.scout.probe_refusal` refuses its tool (a write tool,
    one whose description claims a cost, one that reaches outside the server), when
    :func:`mcpsim.scout.probe_argument_refusal` refuses its arguments (a URL, host name or
    e-mail address), when the tool's name cannot appear in a ``tool_result[<tool>]``
    checkpoint (the variant would be dropped after the call was spent), or when the step's
    arguments need an earlier result; required arguments are tried before optional ones, and an
    ``enum`` argument is never mutated (the planner's own validation would reject the value).
    Returns the reasons every earlier step was skipped.
    """
    reasons: list[str] = []
    for position, call in enumerate(calls, start=1):
        if call.tool == DISCOVER_TOOL_NAME or not catalog.has_tool(call.tool):
            reasons.append(f"step {position} ({call.tool}) is not a server tool")
            continue
        tool = catalog.tool(call.tool)
        refusal = probe_refusal(tool) or probe_argument_refusal(tool, call.arguments)
        if refusal is not None:
            reasons.append(f"step {position} ({tool.name}): {refusal}")
            continue
        if validate_checkpoint(f"tool_result[{tool.name}]: is an error") is not None:
            reasons.append(
                f"step {position} ({tool.name}): its name cannot appear in a "
                "tool_result[<tool>] checkpoint"
            )
            continue
        if call.references():
            reasons.append(f"step {position} ({tool.name}) needs an earlier step's result")
            continue
        schema = tool.input_schema or {}
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required_raw = schema.get("required")
        required = [str(n) for n in required_raw] if isinstance(required_raw, list) else []
        order = [k for k in required if k in call.arguments]
        order += [k for k in call.arguments if k not in order]
        for key in order:
            prop = properties.get(key)
            if enum_members(prop if isinstance(prop, dict) else {}, schema) is not None:
                continue
            mutated = mutate(call.arguments[key])
            if mutated is not None:
                return ProbeChoice(position, tool, key, call.arguments[key], mutated), reasons
        reasons.append(
            f"step {position} ({tool.name}) has no string (3+ characters) or integer argument"
        )
    return None, reasons


def _with_inserted(steps: list[Step], at: int, new: Step) -> list[Step]:
    """``steps`` with ``new`` inserted at 1-based position ``at``; later references that point
    at or after ``at`` are shifted by one so they keep naming the same step."""
    out: list[Step] = []
    for i, step in enumerate(steps, start=1):
        if i == at:
            out.append(new)
        if i >= at and step.references():
            shifted = {}
            for key, value in step.arguments_sketch.items():
                ref = step.references().get(key)
                if ref is not None and ref.from_step >= at:
                    shifted[key] = {**value, "$from_step": ref.from_step + 1}
                else:
                    shifted[key] = value
            step = step.model_copy(update={"arguments_sketch": shifted})
        out.append(step)
    if at > len(steps):
        out.append(new)
    return out


def _path_id(prefix: str, tool: str) -> str:
    return f"{prefix}-{re.sub(r'[^A-Za-z0-9._-]', '-', tool)}"


# Words of a result key whose value is new on every call (ids of the request, times, durations).
VOLATILE_KEY_WORDS = frozenset({
    "id", "ids", "uuid", "guid", "nonce", "etag", "trace", "span", "request", "correlation",
    "timestamp", "ts", "time", "at", "ms", "took", "elapsed", "duration", "latency", "seconds",
    "secs", "now", "date", "created", "updated", "generated", "expires",
})
VOLATILE_FORMATS = frozenset({"date-time", "date", "time", "duration", "uuid"})
_KEY_WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")
_VOLATILE_VALUES = (
    re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE),
    re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}"),  # an ISO date-time
    re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE),  # a hash or opaque token
)
# A number this size is an epoch timestamp (seconds since 2001, or milliseconds), not a count.
_EPOCH_RANGE = (1_000_000_000, 100_000_000_000_000)


def is_volatile(key: str, value: Any, prop: dict[str, Any] | None = None) -> bool:
    """Is ``key``'s value likely to differ between two identical calls?

    By the key's words (``request_id``, ``created_at``, ``tookMs``, ``id``), by the output
    schema's ``format`` (``date-time``, ``uuid``), or by the value's shape (a UUID, an ISO
    date-time, a long hex token, an epoch-sized number). A stable value judged volatile only
    costs a boundary fact; a volatile one stated as a fact fails every later run.
    """
    words = {w.lower() for w in _KEY_WORD.findall(key)}
    if words & VOLATILE_KEY_WORDS:
        return True
    fmt = (prop or {}).get("format")
    if isinstance(fmt, str) and fmt in VOLATILE_FORMATS:
        return True
    if isinstance(value, str):
        return any(p.match(value) for p in _VOLATILE_VALUES)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return _EPOCH_RANGE[0] <= abs(value) < _EPOCH_RANGE[1]
    return False


_CATEGORICAL = frozenset({"null", "true", "false", "empty", "zero"})


def _kind(value: Any) -> str:
    """The kind of value a fact turns on: null, a boolean, empty/zero, or something."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | str | dict):
        return "empty" if not value else type(value).__name__
    if isinstance(value, int | float):
        return "zero" if value == 0 else "number"
    return type(value).__name__


def boundary_facts(
    choice: ProbeChoice,
    probed: Observation,
    original: Any | None,
    lineage_keys: list[str],
) -> list[str]:
    """``tool_result[<tool>]: <key> equals <value> when <argument> is <mutated>`` for what the
    probe returned, at most :data:`MAX_BOUNDARY_FACTS`, built only from structured content.

    A result without structured content (text, which may be anybody's words) gives only
    ``answers without an error``. Keys that echo the mutated input are left out, and so are
    prose strings and values that differ on every call (:func:`is_volatile`: a ``request_id``,
    a timestamp), which would make the boundary path fail on its next run. When the scout's
    result for the original call is known only keys whose value (or list length) differs are
    kept, so the fact is what the mutation changed. Keys an answer field reads come first, then
    keys whose kind of value changed (a list or count that became empty or zero, a value that
    became null) or that the output schema declares an ``enum`` (``match``), then other scalars,
    then other lists.
    """
    tool, when = choice.tool.name, choice.when()
    data = probed.structured
    if isinstance(data, list):
        return [f"tool_result[{tool}]: returns {_entries(len(data))} {when}"]
    if not isinstance(data, dict):
        return [f"tool_result[{tool}]: answers without an error {when}"]
    before = original if isinstance(original, dict) else None
    output = choice.tool.output_schema or {}
    properties = _concrete(output, output).get("properties") if output else None
    properties = properties if isinstance(properties, dict) else {}
    ranked: list[tuple[int, int, str]] = []
    volatile: list[str] = []
    for order, (key, value) in enumerate(data.items()):
        if value == choice.mutated or isinstance(value, dict):
            continue
        prop = _concrete(properties.get(key), output) if key in properties else {}
        if is_volatile(str(key), value, prop):
            volatile.append(str(key))
            continue
        if isinstance(value, list):
            text, now = f"{key} has {_entries(len(value))}", len(value)
            then = len(before[key]) if before and isinstance(before.get(key), list) else None
        else:
            shown = _plain(value)
            if isinstance(value, str) and len(shown) > FACT_VALUE_LIMIT:
                continue
            text, now = f"{key} equals {shown}", value
            then = before.get(key) if before else None
        if before is not None and key in before and then == now:
            continue
        if key in lineage_keys:
            rank = 0
        elif (
            enum_members(prop, output) is not None
            or (before is not None and key in before and _kind(before[key]) != _kind(value))
            or (before is None and _kind(value) in _CATEGORICAL)
        ):
            rank = 1
        else:
            rank = 2 if not isinstance(value, list) else 3
        ranked.append((rank, order, text))
    ranked.sort()
    if not ranked:
        if before is not None:
            apart = f", apart from {', '.join(volatile)}" if volatile else ""
            return [
                f"tool_result[{tool}]: returns the same result {when} as when {choice.argument} "
                f"is {_plain(choice.original)}{apart}"
            ]
        return [f"tool_result[{tool}]: answers without an error {when}"]
    return [f"tool_result[{tool}]: {text} {when}" for _, _, text in ranked[:MAX_BOUNDARY_FACTS]]


def variant_path(
    choice: ProbeChoice,
    probed: Observation,
    original: Any | None,
    happy: Path,
    calls: list[PlannedCall],
    lineage: dict[str, Lineage],
    fields: list[str],
) -> Path:
    """A recovery path (the probe errored) or a boundary path (it answered), built around the
    real result: the mutated call inserted before the step it mutates, then the happy steps."""
    tool = choice.tool.name
    call = calls[choice.step - 1]
    mutated_args = choice.arguments(call)
    label = probed.call_label()
    sent = f"{choice.argument}={json.dumps(choice.mutated, ensure_ascii=False)}"
    original_text = f"{choice.argument} {_plain(choice.original)}"
    shared = lineage_checkpoints(
        calls, lineage, fields, labels=_variant_labels(calls, choice, original_text)
    )
    if probed.is_error:
        error = _clip(probed.summary, 200)
        failing = Step(
            intent=f"Call {tool} with {sent} ({choice.change()})",
            tool=tool,
            arguments_sketch=mutated_args,
            success_looks_like=f"The server rejects it: {error}",
            expect_error=True,
        )
        return Path(
            id=_path_id("recovery", tool),
            kind="recovery",
            title=_clip(f"Recovery: {tool} rejects {sent}, then the original call", 100),
            rationale=(
                f"Probed at plan time on the scout's session: {label} returned an error "
                f"({_clip(probed.summary, 160)}). The agent must read the rejection and call "
                f"{tool} again with {original_text}."
            ),
            steps=_with_inserted(happy.steps, choice.step, failing),
            checkpoints=[
                f"tool_result[{tool}]: is an error {choice.when()}, saying {_clip(error, 120)}",
                f"transcript: after that error the agent calls {tool} with {original_text} and "
                "answers from that result",
                *shared,
            ],
        )
    keys = [str(parse_path(lin.path)[0].key) for lin in lineage.values()
            if parse_path(lin.path) and lin.step == choice.step]
    facts = boundary_facts(choice, probed, original, keys)
    summary = "; ".join(f.split(": ", 1)[1] for f in facts)
    probe_step = Step(
        intent=f"Call {tool} with {sent} ({choice.change()})",
        tool=tool,
        arguments_sketch=mutated_args,
        success_looks_like=(
            f"The server answers without an error ({summary}); that result is not the answer "
            "to the request"
        ),
        expect_error=False,
    )
    return Path(
        id=_path_id("boundary", tool),
        kind="boundary",
        title=_clip(f"Boundary: {tool} answers {sent} without an error", 100),
        rationale=(
            f"Probed at plan time on the scout's session: {label} answered without an error "
            f"({summary}). The agent must not present that result as the answer to the request."
        ),
        steps=_with_inserted(happy.steps, choice.step, probe_step),
        checkpoints=[
            *facts,
            f"final_result: does not present the {tool} result for {sent} as the answer; its "
            f"values come from the call with {original_text}",
            *shared,
        ],
    )


def policy_path(
    happy: Path,
    calls: list[PlannedCall],
    lineage: dict[str, Lineage],
    fields: list[str],
    instruction_index: int,
    instruction: str,
    tool: str,
    model: str,
) -> Path:
    """The happy steps, an answer step that names the rule under test, and ``transcript: no
    call to <tool>``. The agent is told the same rule in its instructions either way."""
    rule = _clip(instruction, 160)
    final = happy.steps[-1].model_copy(
        update={"intent": f"Compose the final answer, keeping this rule: {rule}"}
    )
    return Path(
        id=_path_id("policy", tool),
        kind="policy",
        title=_clip(f"Policy: reach the answer without calling {tool}", 100),
        rationale=(
            f"Instruction {instruction_index} ({_clip(instruction, 120)}) forbids calling {tool}, "
            f"according to {model}; the agent must reach the same answer without it."
        ),
        steps=[*happy.steps[:-1], final],
        checkpoints=[
            f"transcript: no call to {tool}", *lineage_checkpoints(calls, lineage, fields)
        ],
    )


def _usage(response: LLMResponse, seconds: float) -> str:
    return (
        f"{response.usage.input_tokens} prompt + {response.usage.output_tokens} output tokens "
        f"in {seconds:.0f} s"
    )


def _reask(response: LLMResponse, problems: list[str]) -> str | list[dict[str, Any]]:
    text = (
        "The JSON config was rejected:\n"
        + "\n".join(f"- {p}" for p in problems)
        + "\n\nAnswer with the whole corrected JSON config."
    )
    blocks = [b for b in response.tool_uses() if b.get("name") == EXECUTION_TOOL_NAME]
    tool_use_id = blocks[0].get("id") if blocks else None
    if not isinstance(tool_use_id, str) or not tool_use_id:
        return text
    return [{"type": "tool_result", "tool_use_id": tool_use_id, "content": text, "is_error": True}]


async def ask_execution(
    scenario: Scenario,
    catalog: Catalog,
    llm: LLM,
    view: PlannerView,
    fields: list[str],
    observations: list[Observation],
    system: str,
    user: str,
    notes: list[str],
) -> tuple[list[PlannedCall], dict[str, Lineage]]:
    """The execution-plan call and its single re-ask; see :func:`settle_execution`."""
    max_steps = max(1, min(LOCAL_MAX_STEPS, scenario.budgets.max_tool_calls))
    schema = execution_schema(step_tool_names(view), fields, max_steps)
    tools = [{"name": EXECUTION_TOOL_NAME, "description": "", "input_schema": schema}]
    choice = {"type": "tool", "name": EXECUTION_TOOL_NAME}
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    # The last answer that parsed, for salvage when the re-ask's reply does not parse.
    usable: tuple[ToolExecution, Review] | None = None
    problems: list[str] = []
    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        started = time.monotonic()
        response = await llm.complete(
            model=scenario.models.planner,
            system=system,
            messages=list(messages),
            tools=tools,
            tool_choice=choice,
            max_tokens=LOCAL_PLAN_MAX_TOKENS,
        )
        seconds = time.monotonic() - started
        draft, problems = parse_execution(response)
        review = None
        if draft is not None:
            review = review_execution(
                draft, catalog, view, fields, observations, max_steps,
                scenario.expected_outcome.json,
            )
            problems = review.problems
            usable = (draft, review)
        verdict = "accepted" if draft is not None and not problems else (
            f"{len(problems)} problem(s): " + "; ".join(_clip(p, 160) for p in problems[:4])
        )
        notes.append(f"execution plan call {attempt}: {_usage(response, seconds)}, {verdict}")
        if draft is not None and review is not None and not problems:
            return settle_execution(draft, review, fields, notes, salvage=False)
        if attempt < MAX_PLAN_ATTEMPTS:
            messages.append({"role": "assistant", "content": list(response.content)})
            messages.append({"role": "user", "content": _reask(response, problems)})
    if usable is None:
        raise PlanError(
            f"local planner for scenario {scenario.name!r} produced no usable execution plan in "
            f"{MAX_PLAN_ATTEMPTS} attempts; last errors:\n" + "\n".join(f"- {p}" for p in problems)
        )
    if draft is None:
        notes.append("the re-asked answer did not parse; the previous answer is salvaged")
    return settle_execution(usable[0], usable[1], fields, notes, salvage=True)


def prohibiting_clauses(instruction: str) -> list[str]:
    """Each clause of ``instruction`` that prohibits something, from its cue to the clause end:
    ``Use find_product; do not run a planning tool for a lookup.`` → ``["do not run a planning
    tool for a lookup"]``."""
    clauses: list[str] = []
    for clause in re.split(r"[.;!?]", instruction):
        found = PROHIBITION.search(clause)
        if found is not None:
            clauses.append(clause[found.start():].strip())
    return clauses


def clause_names_tool(clauses: list[str], tool: str) -> bool:
    """Does a prohibiting clause name ``tool`` or a word of its name (``planning`` for
    ``plan_recipe``, ``product lookups`` for ``find_product``)? A tool the rule never mentions is
    not one it forbids, whatever the model answered."""
    names = [n for n in scoping_tokens(tool) if len(n) >= 3]
    for clause in clauses:
        if tool.lower() in clause.lower():
            return True
        for word in scoping_tokens(clause):
            for name in names:
                if word == name or (len(name) >= 4 and word.startswith(name)) or (
                    len(word) >= 4 and name.startswith(word)
                ):
                    return True
    return False


async def ask_forbidden_tools(
    scenario: Scenario,
    llm: LLM,
    tool_names: list[str],
    happy_tools: set[str],
    notes: list[str],
) -> list[tuple[int, str, str]]:
    """``(instruction number, instruction, tool)`` for each tool an instruction forbids.

    Only instructions that prohibit or restrict something (:data:`PROHIBITION`) are asked
    about, at most :data:`MAX_POLICY_QUESTIONS`; the rest are noted as skipped. Every question
    has the same system prompt (the tool names), so the server can reuse its cached prefix. An
    answer is noted and ignored when it names a tool the happy path calls (the instruction
    cannot forbid what the planned route needs) or a tool no prohibiting clause of the
    instruction names (:func:`clause_names_tool`; command-r7b answered ``get_product_origins``
    for "never round a price or guess a store").
    """
    if not tool_names:
        return []
    system = (
        "You check rules for an AI agent that calls tools. The agent's tools are: "
        + ", ".join(tool_names) + "."
    )
    schema = policy_schema(tool_names)
    tools = [{"name": POLICY_TOOL_NAME, "description": "", "input_schema": schema}]
    choice = {"type": "tool", "name": POLICY_TOOL_NAME}
    found: list[tuple[int, str, str]] = []
    asked = 0
    for index, raw in enumerate(scenario.instructions, start=1):
        text = " ".join(raw.split())
        if not PROHIBITION.search(text):
            notes.append(f"instruction {index} not asked about: it prohibits nothing")
            continue
        if asked >= MAX_POLICY_QUESTIONS:
            notes.append(
                f"instruction {index} not asked about: at most {MAX_POLICY_QUESTIONS} policy "
                "questions per plan"
            )
            continue
        asked += 1
        user = (
            f"Rule: {text}\n\nWhich of these tools does this rule forbid calling? Answer "
            f"{NONE_ANSWER} if it forbids none of them."
        )
        started = time.monotonic()
        response = await llm.complete(
            model=scenario.models.planner,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=tools,
            tool_choice=choice,
            max_tokens=POLICY_MAX_TOKENS,
        )
        seconds = time.monotonic() - started
        blocks = [b for b in response.tool_uses() if b.get("name") == POLICY_TOOL_NAME]
        payload = blocks[0].get("input") if blocks else None
        raw_answer = payload.get("tool") if isinstance(payload, dict) else None
        answer = raw_answer if raw_answer in (*tool_names, NONE_ANSWER) else None
        note = (
            f"policy question {asked} (instruction {index}): {_usage(response, seconds)}, "
            f"answer {answer if answer is not None else 'unusable: ' + repr(raw_answer)}"
        )
        clauses = prohibiting_clauses(text)
        if answer in happy_tools:
            note += f"; ignored, the happy path calls {answer}"
        elif answer is not None and answer != NONE_ANSWER:
            if clause_names_tool(clauses, answer):
                found.append((index, text, answer))
            else:
                note += (
                    f"; ignored, the prohibition ({'; '.join(clauses)}) does not name {answer}"
                )
        notes.append(note)
    return found


async def ground_variant(
    scenario: Scenario,
    catalog: Catalog,
    scout: ScoutResult | None,
    session: Session | None,
    happy: Path,
    calls: list[PlannedCall],
    lineage: dict[str, Lineage],
    fields: list[str],
    notes: list[str],
) -> Path | None:
    """Probe one mutated read-only call and build the recovery or boundary path from it."""
    choice, reasons = choose_probe(calls, catalog)
    if choice is None:
        notes.append("no probe: " + ("; ".join(reasons) or "no tool step"))
        return None
    if scout is None or session is None:
        notes.append(
            f"no probe of {choice.tool.name}: probes run on the scout's session and this plan has "
            "none (disclosure: all, or no session was passed)"
        )
        return None
    if len(calls) + 1 > scenario.budgets.max_tool_calls:
        notes.append(
            f"no probe: a variant would make {len(calls) + 1} tool calls, over "
            f"max_tool_calls={scenario.budgets.max_tool_calls}"
        )
        return None
    call = calls[choice.step - 1]
    why = (
        f"the planner's variant of happy step {choice.step}: {choice.argument} "
        f"{_plain(choice.original)} -> {_plain(choice.mutated)}"
    )
    try:
        probed = await probe(scout, session, choice.tool, choice.arguments(call), why=why)
    except ProbeRefusedError:
        raise  # choose_probe applies the same checks, so this is a bug to surface
    except Exception as exc:  # noqa: BLE001 - the probe is optional; the plan is not
        # The session has sat idle through minutes of local model calls; an HTTP server may
        # have dropped it, or answered the mutated input with a result the SDK cannot validate
        # (a pydantic ValidationError, which is a ValueError). Losing the variant is better
        # than losing the plan.
        notes.append(
            f"no probe: {choice.tool.name} could not be called ({type(exc).__name__}: "
            f"{_clip(str(exc), 160)})"
        )
        return None
    if probed is None:
        notes.append(f"no probe: the scout budget of {scout.budget} tool call(s) is spent")
        return None
    outcome = "an error" if probed.is_error else "an answer"
    notes.append(f"probe: {probed.call_label()} returned {outcome} in {probed.ms:.0f} ms")
    original = _observed_result(scout.tool_observations(), call, choice.tool)
    return variant_path(choice, probed, original, happy, calls, lineage, fields)


def _distinct_valid(
    paths: list[Path],
    catalog: Catalog,
    view: PlannerView,
    scenario: Scenario,
    notes: list[str],
) -> list[Path]:
    """No two paths with identical steps, and every path valid as any plan path must be."""
    kept: list[Path] = []
    seen: dict[str, str] = {}
    for path in paths:
        key = _canonical([s.model_dump(mode="json") for s in path.steps])
        if key in seen:
            notes.append(f"path {path.id} dropped: its steps are those of path {seen[key]}")
            continue
        problems = validate_draft(
            PlanDraft(paths=[path]), catalog, view, scenario, require_happy=path.kind == "happy"
        )
        if problems:
            if path.kind == "happy":
                raise PlanError("the derived happy path is invalid:\n" + "\n".join(problems))
            notes.append(f"path {path.id} dropped: {'; '.join(problems[:3])}")
            continue
        seen[key] = path.id
        kept.append(path)
    return kept


async def plan_execution(
    scenario: Scenario,
    catalog: Catalog,
    llm: LLM,
    scout: ScoutResult | None = None,
    *,
    session: Session | None = None,
    prompt_budget: int | None = None,
) -> ExecutionPlan:
    """The local profile's plan: happy path, a probed variant, policy paths (module docstring).

    ``scout`` supplies the disclosed set, the reports and the observations, and holds the budget
    the probe is counted against; ``session`` is its still-open MCP session (without both there
    is no probe). ``prompt_budget`` overrides ``MCPSIM_PLANNER_PROMPT_BUDGET`` (default
    :data:`LOCAL_PROMPT_BUDGET` here).
    """
    model = scenario.models.planner
    view = (
        planner_view(scenario, catalog, scout) if scout is not None
        else PlannerView(disclosed=list(catalog.tools), allowed=catalog.tool_names())
    )
    if not step_tool_names(view):
        raise PlanError(f"no disclosed tool to plan scenario {scenario.name!r} with")
    fields = [str(k) for k in (scenario.expected_outcome.json or {})]
    budget = prompt_budget if prompt_budget is not None else planner_prompt_budget(
        LOCAL_PROMPT_BUDGET
    )
    system, user = execution_prompts(scenario, view, fields, budget=budget)
    if scout is not None:
        scout.planner_prompt_chars = len(system) + len(user)
    notes: list[str] = []
    observations = scout.tool_observations() if scout is not None else []
    calls, lineage = await ask_execution(
        scenario, catalog, llm, view, fields, observations, system, user, notes
    )
    happy = happy_path(scenario, calls, lineage, fields)
    paths = [happy]
    variant = await ground_variant(
        scenario, catalog, scout, session, happy, calls, lineage, fields, notes
    )
    if variant is not None:
        paths.append(variant)
    forbidden = await ask_forbidden_tools(
        scenario, llm, catalog.tool_names(), {c.tool for c in calls}, notes
    )
    seen_tools: set[str] = set()
    for index, instruction, tool in forbidden:
        if tool in seen_tools:
            notes.append(f"instruction {index} also forbids {tool}; one policy path covers it")
            continue
        seen_tools.add(tool)
        paths.append(policy_path(happy, calls, lineage, fields, index, instruction, tool, model))
    return ExecutionPlan(
        scenario=scenario.name,
        catalog_digest=catalog.digest(),
        paths=_distinct_valid(paths, catalog, view, scenario, notes),
        notes=[f"local planner ({model}): {n}" for n in notes],
    )
