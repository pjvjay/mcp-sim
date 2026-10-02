"""Planner (DESIGN §2 "Planner").

Input: a :class:`~mcpsim.scenario.Scenario` and the server's live
:class:`~mcpsim.mcpclient.Catalog`. Output: an :class:`~mcpsim.plan.ExecutionPlan` with several
paths (happy, recovery, alternative, boundary, policy), produced by one structured-output LLM
call. The structured-output schema is built **per catalog** (``Step.tool`` is an enum of the
catalog's tool names, so a constrained decoder cannot invent a tool); the same schema and
:func:`validate_draft` then check argument keys and scalar types against each tool's input
schema, ``$from_step`` references, recovery paths (at least one ``expect_error`` step) and the
``<where>: <condition>`` shape of checkpoints. A draft that fails is re-asked **once** with every
problem listed, then :class:`PlanError` is raised.

The planner is the **orchestrator** of the Informant-Report Method (DESIGN §2b): with a
:class:`~mcpsim.scout.ScoutResult` it plans from the *disclosed* toolset (the digest covers only
those tools; the rest are listed by name as available on request through ``discover_tools``),
the **informant reports** (``shelf_auditor.direct_match = true — …``), the goals observers
enabled and the scout's observations, whose real ids and slugs replace placeholders. A false
report makes honest handling the happy path; an unknown report makes the observation that
settles it step one. ``Step.tool``'s enum is the disclosed set plus ``discover_tools``; a step
naming an on-request tool must follow a ``discover_tools`` step or a step after which an
observer that watches tool traffic can enable it. The whole prompt stays under
``MCPSIM_PLANNER_PROMPT_BUDGET`` characters (default 12,000): observations are trimmed first,
the on-request list second, never the disclosed digest or the reports.

``dry_run=True`` needs no LLM: it emits a one-path happy plan over the allowed tools that share
vocabulary with the scenario (:mod:`mcpsim.scoping`), the scout's proven expected-outcome
lookups first, then most relevant first, at most ``budgets.max_tool_calls`` of them, skipping
tools whose description says they cost money or credits and write tools unless the scenario asks
for a write. An argument named by a top-level ``expected_outcome.json`` key with a plain value
takes that value (``find_product(query="penne")``); other required arguments are defaulted from
the schema. The rationale names everything skipped and why.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mcpsim.llm import LLM, LLMResponse
from mcpsim.mcpclient import Catalog, ToolInfo
from mcpsim.plan import (
    CHECKPOINT_PATTERN,
    CHECKPOINT_SHAPE,
    PATH_KINDS,
    REFERENCE_KEY,
    ExecutionPlan,
    Path,
    Step,
    parse_reference,
)
from mcpsim.scenario import Scenario
from mcpsim.scoping import (
    DISCOVER_TOOL_NAME,
    discover_tool_definition,
    is_write_tool,
    plain_expected_value,
    rank_tools,
    scenario_terms,
    write_intent,
)
from mcpsim.transcript import InformantReport

if TYPE_CHECKING:
    from mcpsim.scout import Observation, ScoutResult

PLAN_TOOL_NAME = "emit_execution_plan"
PLAN_MAX_TOKENS = 8192
MAX_PLAN_ATTEMPTS = 2  # the first ask plus one re-ask with the validation error
DRY_RUN_PATH_ID = "happy-dry-run"
PROMPT_BUDGET_ENV = "MCPSIM_PLANNER_PROMPT_BUDGET"
DEFAULT_PROMPT_BUDGET = 12_000
OBSERVATION_LINE_LIMIT = 600
# When no tool shares a word with the scenario, the dry run still calls this many (catalog
# order) so the smoke proves the server answers.
DRY_RUN_FALLBACK = 3

# Tools a dry-run plan must not call (DESIGN §6 build spec): descriptions mentioning spend.
# A tool is skipped by the dry run only when its description CLAIMS a cost.
# The first version also matched the bare word "LLM", which is how eleven
# free pantry tools describe themselves ("Free — no LLM calls"), so the dry
# run called exactly one tool. A negation ("no LLM", "free") wins.
EXPENSIVE_PATTERN = re.compile(r"credit|\bcosts?\b|\bslow\b|\bspends?\b", re.IGNORECASE)
FREE_PATTERN = re.compile(r"\bno llm\b|\bfree\b", re.IGNORECASE)


class PlanError(RuntimeError):
    """The planner could not produce a valid plan (after the single allowed re-ask)."""


class PlanDraft(BaseModel):
    """The structured output the planner asks the LLM for: just the paths.

    ``scenario`` and ``catalog_digest`` are filled in by :func:`plan`, never by the model.
    """

    model_config = ConfigDict(extra="forbid")

    paths: list[Path] = Field(default_factory=list)


@dataclass
class PlannerView:
    """What the orchestrating planner is shown of the server and of the observers' work.

    ``disclosed`` are the tools whose full digest lines it sees and may name directly;
    ``on_request`` the other allowed tools (names only), reachable through a ``discover_tools``
    step when ``discoverable`` or through an observer effect; ``reports``, ``goals`` and
    ``observations`` come from the scout. Without a scout every allowed tool is disclosed.
    """

    disclosed: list[ToolInfo]
    on_request: list[str] = field(default_factory=list)
    discoverable: bool = False
    reports: list[InformantReport] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)

    @property
    def disclosed_names(self) -> list[str]:
        return [t.name for t in self.disclosed]

    def enabling_observer(self, scenario: Scenario, tool: str) -> str | None:
        """``<observer>.<condition>`` of a condition that can enable ``tool`` mid-run: one on
        an observer that watches tool traffic (or everything) and reports at ``tool_result``
        or ``turn``; ``None`` when no such effect exists."""
        from fnmatch import fnmatchcase

        for obs in scenario.observers:
            watches = set(obs.watches)
            if not ({"tool_traffic", "all"} & watches and {"tool_result", "turn"} & set(obs.on)):
                continue
            for cond in obs.conditions:
                globs = [*cond.then.enable_tools, *cond.otherwise.enable_tools]
                if any(fnmatchcase(tool, g) for g in globs):
                    return f"{obs.name}.{cond.id}"
        return None


def planner_view(
    scenario: Scenario, catalog: Catalog, scout: ScoutResult | None = None
) -> PlannerView:
    """Disclosed = the scout's disclosed set under ``progressive`` disclosure; otherwise every
    allowed tool (``plan`` disclosure offers the path's own tools, ``all`` everything)."""
    policy = scenario.tools
    if scout is not None and policy.disclosure == "progressive":
        disclosed = [t for t in catalog.tools if t.name in set(scout.disclosed)]
    else:
        disclosed = list(catalog.tools)
    names = {t.name for t in disclosed}
    return PlannerView(
        disclosed=disclosed,
        on_request=[n for n in catalog.tool_names() if n not in names],
        discoverable=policy.disclosure == "progressive" and policy.discover_tool,
        reports=list(scout.reports) if scout is not None else [],
        goals=list(scout.goals) if scout is not None else [],
        observations=list(scout.observations) if scout is not None else [],
        allowed=catalog.tool_names(),
    )


def tool_name_enum(catalog: Catalog, view: PlannerView | None = None) -> list[str]:
    """The sorted tool names a step may name directly: the ``enum`` in the plan schema.

    With a view: the disclosed tools plus ``discover_tools`` when it is offered; an on-request
    tool is not in the enum (the validator accepts it only after a step that can reveal it).
    """
    if view is None:
        return sorted(catalog.tool_names())
    names = sorted(view.disclosed_names)
    if view.discoverable:
        names.append(DISCOVER_TOOL_NAME)
    return names


def plan_input_schema(catalog: Catalog, view: PlannerView | None = None) -> dict[str, Any]:
    """:class:`PlanDraft`'s JSON Schema with ``Step.tool`` constrained to this catalog.

    ``Step.tool`` becomes ``{"anyOf": [{"type": "string", "enum": [...]}, {"type": "null"}]}``
    so a structured decoder (Ollama ``format``, Anthropic forced tool use) cannot emit a tool the
    server does not have; ``Path.kind`` is already an enum. Validation uses the same enum.
    """
    schema = PlanDraft.model_json_schema()
    names = tool_name_enum(catalog, view)
    tool_schema: dict[str, Any] = {
        "anyOf": [{"type": "string", "enum": names}, {"type": "null"}],
        "default": None,
        "description": "A tool name from the catalog, or null for a step that calls no tool.",
    }
    if not names:  # an empty enum is not a valid schema; only the no-tool step remains
        tool_schema["anyOf"] = [{"type": "null"}]
    step_schema = schema.get("$defs", {}).get("Step")
    if not isinstance(step_schema, dict):  # pragma: no cover - pydantic always nests Step
        raise PlanError("PlanDraft schema has no $defs.Step to constrain")
    step_schema["properties"]["tool"] = tool_schema
    return schema


def plan_tool_definition(catalog: Catalog, view: PlannerView | None = None) -> dict[str, Any]:
    """The single Anthropic tool used for structured output, built for this catalog."""
    return {
        "name": PLAN_TOOL_NAME,
        "description": (
            "Emit the execution plan: an ordered list of distinct paths through the server, "
            "each with concrete steps and observable checkpoints."
        ),
        "input_schema": plan_input_schema(catalog, view),
    }


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


# --- JSON Schema helpers shared by the digest and the validator ------------------------------

JSON_TYPES: tuple[str, ...] = ("string", "integer", "number", "boolean", "array", "object", "null")
_MAX_ENUM_IN_DIGEST = 6
_MAX_OUTPUT_KEYS = 24
_FIRST_SENTENCE = re.compile(r"^(.*?[.!?])(?:\s|$)")


def resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow a local ``$ref`` (``#/$defs/Name`` or ``#/definitions/Name``) one level."""
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    target: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(target, dict) or part not in target:
            return schema
        target = target[part]
    return target if isinstance(target, dict) else schema


def schema_types(schema: dict[str, Any], root: dict[str, Any] | None = None) -> set[str]:
    """Every JSON type a property schema admits (through ``anyOf``/``oneOf``/``$ref``/``enum``).

    Empty when the schema says nothing about the type, in which case nothing is checked.
    """
    root = root if root is not None else schema
    return _schema_types(schema, root, depth=0)


def _schema_types(schema: dict[str, Any], root: dict[str, Any], depth: int) -> set[str]:
    if depth > 4:
        return set()
    schema = resolve_ref(schema, root)
    found: set[str] = set()
    declared = schema.get("type")
    if isinstance(declared, str):
        found.add(declared)
    elif isinstance(declared, list):
        found.update(t for t in declared if isinstance(t, str))
    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    found.update(_schema_types(variant, root, depth + 1))
    enum = schema.get("enum")
    if isinstance(enum, list):
        found.update(json_type_of(member) for member in enum)
    if "const" in schema:
        found.add(json_type_of(schema["const"]))
    if not found:
        if "properties" in schema or "additionalProperties" in schema:
            found.add("object")
        elif "items" in schema:
            found.add("array")
    return found


def json_type_of(value: Any) -> str:
    """The JSON Schema type name of a Python value (``bool`` before ``int``)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def type_accepts(declared: set[str], actual: str) -> bool:
    """Does a literal of JSON type ``actual`` satisfy the declared types? Integers are numbers."""
    if not declared:
        return True
    if actual in declared:
        return True
    return actual == "integer" and "number" in declared


def enum_members(schema: dict[str, Any], root: dict[str, Any]) -> list[Any] | None:
    """The ``enum`` a property (or its non-null variant) restricts values to, if any."""
    schema = resolve_ref(schema, root)
    enum = schema.get("enum")
    if isinstance(enum, list):
        return list(enum)
    members: list[Any] = []
    saw_enum = False
    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                return None
            if variant.get("type") == "null":
                members.append(None)
                continue
            inner = enum_members(variant, root)
            if inner is None:
                return None
            saw_enum = True
            members.extend(inner)
    return members if saw_enum else None


def _type_label(schema: dict[str, Any], root: dict[str, Any]) -> str:
    """A short type for the digest: ``string``, ``integer[]``, ``a|b|c``, ``object``."""
    resolved = resolve_ref(schema, root)
    enum = enum_members(resolved, root)
    if enum is not None:
        shown = [m for m in enum if m is not None]
        if 0 < len(shown) <= _MAX_ENUM_IN_DIGEST and all(isinstance(m, str) for m in shown):
            return "|".join(str(m) for m in shown)
    types = sorted(t for t in schema_types(resolved, root) if t != "null")
    if types == ["array"]:
        items = resolved.get("items")
        if not isinstance(items, dict):
            for key in ("anyOf", "oneOf"):
                for variant in resolved.get(key, []) or []:
                    if isinstance(variant, dict) and isinstance(variant.get("items"), dict):
                        items = variant["items"]
                        break
        if isinstance(items, dict):
            inner = _type_label(items, root)
            return f"{inner}[]" if inner != "any" else "array"
        return "array"
    return "|".join(types) if types else "any"


def render_arguments(tool: ToolInfo) -> str:
    """``name: type, other?: type`` for the digest; ``?`` marks optional (or nullable)."""
    schema = tool.input_schema or {}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return ""
    required = schema.get("required")
    required_names = {str(n) for n in required} if isinstance(required, list) else set()
    parts: list[str] = []
    for name, prop in properties.items():
        prop_schema = prop if isinstance(prop, dict) else {}
        optional = name not in required_names or "null" in schema_types(prop_schema, schema)
        parts.append(f"{name}{'?' if optional else ''}: {_type_label(prop_schema, schema)}")
    return ", ".join(parts)


def output_keys(schema: dict[str, Any]) -> str:
    """What the digest says a tool returns: its top-level output keys.

    A one-level ``$ref`` is resolved. When the schema wraps a single ``result`` key (the MCP SDK
    does this for non-object return types) the inner object's keys are shown; an array of
    objects shows ``list of {keys}``. ``object`` when the schema has no named keys.
    """
    resolved = resolve_ref(schema, schema)
    properties = resolved.get("properties")
    if isinstance(properties, dict) and list(properties) == ["result"]:
        inner_raw = properties["result"]
        inner = resolve_ref(inner_raw, schema) if isinstance(inner_raw, dict) else {}
        inner_props = inner.get("properties")
        if isinstance(inner_props, dict) and inner_props:
            return _join_keys(list(inner_props))
        if "array" in schema_types(inner, schema):
            items = inner.get("items")
            item_schema = resolve_ref(items, schema) if isinstance(items, dict) else {}
            item_props = item_schema.get("properties")
            if isinstance(item_props, dict) and item_props:
                return "list of {" + _join_keys(list(item_props)) + "}"
            return "list"
        types = sorted(t for t in schema_types(inner, schema) if t != "null")
        return "result (" + "|".join(types) + ")" if types else "result"
    if isinstance(properties, dict) and properties:
        return _join_keys(list(properties))
    types = sorted(t for t in schema_types(resolved, schema) if t != "null")
    return "|".join(types) if types else "object"


def _join_keys(keys: list[str]) -> str:
    shown = keys[:_MAX_OUTPUT_KEYS]
    suffix = f", … ({len(keys) - len(shown)} more)" if len(keys) > len(shown) else ""
    return ", ".join(shown) + suffix


def first_sentence(text: str) -> str:
    flat = " ".join(text.split())
    m = _FIRST_SENTENCE.match(flat)
    return m.group(1) if m else flat


def render_tool_line(tool: ToolInfo) -> str:
    """One digest line: ``- name(arg: type, opt?: type) → returns k1, k2 — first sentence``."""
    line = f"- {tool.name}({render_arguments(tool)})"
    if tool.output_schema is not None:
        line += f" → returns {output_keys(tool.output_schema)}"
    sentence = first_sentence(tool.description)
    if sentence:
        line += f" — {sentence}"
    return line


def on_request_line(view: PlannerView, *, shown: int | None = None) -> str:
    """The one line naming the tools the planner may not name directly.

    ``shown`` caps how many names appear (the prompt budget cuts this list second); the rest
    become a count.
    """
    names = list(view.on_request)
    if not names:
        return ""
    if shown is not None and shown < len(names):
        listed = ", ".join(names[:shown])
        rest = f" and {len(names) - shown} more" if shown else f"{len(names)} tools"
        listed = f"{listed}{rest}" if shown else rest
    else:
        listed = ", ".join(names)
    if view.discoverable:
        return (
            f"AVAILABLE ON REQUEST through {DISCOVER_TOOL_NAME} (name only; a step may use one "
            f"of these only after a {DISCOVER_TOOL_NAME} step whose query names what it needs, or "
            f"after an observer effect enables it): {listed}"
        )
    return (
        "NOT DISCLOSED (declared by the server but not offered to the agent unless an observer "
        f"effect enables them): {listed}"
    )


def render_catalog_for_prompt(
    catalog: Catalog, view: PlannerView | None = None, *, on_request_shown: int | None = None
) -> str:
    """The catalog digest the planner sees (LOCAL_MODELS.md, "Fitting an 8k context").

    One line per tool: name, argument names with types (``?`` marks optional), ``→ returns`` the
    top-level output keys when the server publishes an output schema, then the first sentence of
    the description. With a view only the disclosed tools get a line and the on-request tools
    are named on one line (:func:`on_request_line`). Resources, templates and prompts follow,
    one line each.
    """
    lines: list[str] = []
    tools = catalog.tools if view is None else view.disclosed
    if view is None or not view.on_request:
        lines.append(f"TOOLS ({len(tools)}) — the ONLY tools that exist:")
    else:
        lines.append(
            f"TOOLS ({len(tools)} disclosed of {len(view.allowed)} allowed) — the tools a step "
            "may name directly:"
        )
    for tool in tools:
        lines.append(render_tool_line(tool))
    if view is not None and view.discoverable:
        lines.append(render_tool_line(ToolInfo.model_validate(discover_tool_definition())))
    if view is not None and view.on_request:
        lines.append(on_request_line(view, shown=on_request_shown))
    lines.append(f"RESOURCES ({len(catalog.resources)}):")
    for res in catalog.resources:
        lines.append(f"- {res.uri} [{res.name}]{_dash(res.description)}")
    lines.append(f"RESOURCE TEMPLATES ({len(catalog.resource_templates)}):")
    for tmpl in catalog.resource_templates:
        lines.append(f"- {tmpl.uri_template} [{tmpl.name}]{_dash(tmpl.description)}")
    lines.append(f"PROMPTS ({len(catalog.prompts)}):")
    for prompt in catalog.prompts:
        arg_names = ", ".join(str(a.get("name", "?")) for a in prompt.arguments)
        lines.append(f"- {prompt.name}({arg_names}){_dash(prompt.description)}")
    return "\n".join(lines)


def _dash(description: str) -> str:
    """`` — description`` on one line (server descriptions often carry newlines), or empty."""
    flat = " ".join(description.split())
    return f" — {flat}" if flat else ""


def render_scenario_for_prompt(scenario: Scenario) -> str:
    lines: list[str] = [f"SCENARIO: {scenario.name}", "", "ROLE:", scenario.role.strip(), ""]
    lines += ["GOAL:", scenario.goal.strip(), ""]
    lines.append("INSTRUCTIONS (policies the agent must follow; each is a judge checklist item):")
    if scenario.instructions:
        lines += [f"{i + 1}. {item.strip()}" for i, item in enumerate(scenario.instructions)]
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("EXPECTED OUTCOME:")
    if scenario.expected_outcome.text:
        lines.append(f"text: {scenario.expected_outcome.text.strip()}")
    if scenario.expected_outcome.json is not None:
        lines.append(
            "json (matched deterministically against the agent's final_result): "
            + _compact_json(scenario.expected_outcome.json)
        )
    return "\n".join(lines)


def example_literal(schema: dict[str, Any], root: dict[str, Any]) -> Any:
    """A readable, correctly typed example value for the prompt's example step."""
    enum = enum_members(schema, root)
    if enum is not None:
        for member in enum:
            if member is not None:
                return member
    types = sorted(t for t in schema_types(schema, root) if t != "null")
    if "string" in types:
        return "example"
    if "integer" in types:
        return 1
    if "number" in types:
        return 1
    if "boolean" in types:
        return False
    if "array" in types:
        return []
    if "object" in types:
        return {}
    return "example"


def example_step(catalog: Catalog) -> Step:
    """One concrete example step using a real catalog tool (the one with the most required args).

    Falls back to a no-tool step when the catalog is empty, so the prompt is always well formed.
    """
    if not catalog.tools:
        return Step(
            intent="Compose the final answer from the results so far",
            tool=None,
            success_looks_like="A final_result block whose fields come from tool results",
        )

    def required_count(tool: ToolInfo) -> int:
        req = (tool.input_schema or {}).get("required")
        return len(req) if isinstance(req, list) else 0

    tool = max(catalog.tools, key=required_count)  # the first tool wins ties
    schema = tool.input_schema or {}
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = schema.get("required")
    required = [str(n) for n in required] if isinstance(required, list) else []
    arguments: dict[str, Any] = {}
    for name in required:
        prop = properties.get(name)
        arguments[name] = example_literal(prop if isinstance(prop, dict) else {}, schema)
    returns = "a result"
    if tool.output_schema is not None:
        keys = output_keys(tool.output_schema).split(", ")
        returns = ", ".join(keys[:4]) + (", …" if len(keys) > 4 else "")
    return Step(
        intent=f"Call {tool.name} to make progress on the goal",
        tool=tool.name,
        arguments_sketch=arguments,
        success_looks_like=f"{tool.name} returns {returns} with values the goal can use",
        expect_error=False,
    )


def build_system_prompt(
    catalog: Catalog, view: PlannerView | None = None, *, on_request_shown: int | None = None
) -> str:
    kinds = ", ".join(PATH_KINDS)
    example = example_step(catalog).model_dump(mode="json")
    reference = _compact_json({REFERENCE_KEY: 1, "path": "items[*].id"})
    return "\n".join(
        [
            "You are the planner of an LLM-as-a-judge simulation framework for MCP servers.",
            "Given a scenario (role, goal, instructions, expected outcome) and the live catalog of",
            "an MCP server, design an execution plan: several DISTINCT paths an agent could take",
            "through the server to reach the goal, so the simulation can check that an agentic",
            "process can use the server along every one of them.",
            "",
            f"Path kinds (field `kind`), one of: {kinds}.",
            "- happy: the straightforward route to the goal. ALWAYS include exactly one.",
            "- recovery: the agent sends bad input the server rejects and must correct itself"
            " from the server's error or suggestion.",
            "- alternative: a different tool sequence that reaches the same end.",
            "- boundary: limits, pagination, empty results, unknown identifiers.",
            "- policy: a route that tempts the agent to break one of the instructions; the"
            " checkpoints state what obeying it looks like.",
            "Include every kind the catalog can support; omit a kind only when the server has",
            "no tool that could exercise it, and say so in another path's rationale.",
            "",
            "Rules (a plan that breaks one is rejected and you are asked to fix it):",
            "1. `tool` is a tool name from the TOOLS list below, spelled exactly, or null for a",
            "   step that calls no tool (e.g. composing the final answer). Never invent tools,",
            "   resources or arguments that are not in the catalog.",
            "2. Every key in `arguments_sketch` is an argument of that tool as listed in its",
            "   digest line, and every literal value has the listed JSON type (string, integer,",
            "   number, boolean, array, object). Never write prose or '<placeholder>' text where",
            "   an integer, boolean or array is required; a tool listed with `()` takes no",
            "   arguments at all.",
            "3. A value that only an earlier step's result can supply (an id, a slug the server",
            "   returned) is written as a reference, not guessed:",
            f"   {reference}  — `{REFERENCE_KEY}` is the 1-based index of an EARLIER step in the",
            "   same path that calls a tool; `path` is a dotted path into that step's result,",
            "   `[*]` meaning every element. The executor resolves it at run time.",
            "4. A `recovery` path must contain the failure: at least one step with",
            "   `expect_error: true` whose `success_looks_like` names the server's rejection",
            "   (error text, suggestions), followed by the corrected call.",
            "5. Every checkpoint has the shape `<where>: <observable condition>` with `<where>`",
            "   one of final_result, tool_result[<tool_name>] or transcript, e.g.",
            "   'final_result: origin_status equals the value tool_result[plan_recipe] carried'",
            "   or 'transcript: no call to a tool whose description says it costs credits'.",
            "   Never vague ('the agent did well').",
            "6. The `→ returns` part of a digest line lists what a tool already gives back; do",
            "   not add a call to learn something an earlier step's result already contains.",
            "   Prefer cheap tools; one whose description says it is slow or costs credits is",
            "   used only when the goal needs it, and the rationale says so.",
            "7. You are the orchestrator of a team of informants. Plan from the INFORMANT",
            "   REPORTS and the OBSERVATIONS below and from the disclosed tools. Use observed",
            "   ids, slugs and values in `arguments_sketch` instead of placeholders, and cite",
            "   the observation in `success_looks_like` when one exists.",
            "8. A report that is FALSE makes honest handling the happy path: when the thing the",
            "   goal names does not exist, the plan says so and stops; when fabrication is a",
            "   risk, the plan verifies before answering. A report that is UNKNOWN makes the",
            "   observation that would settle it the first step.",
            f"9. A tool listed as available on request is used only after a `{DISCOVER_TOOL_NAME}`",
            "   step (its `query` says what the agent needs) or after a step whose result an",
            "   observer effect reacts to by enabling it; name it directly otherwise and the plan",
            "   is rejected.",
            "Also: `id` is a short slug (letters, digits, '.', '_', '-'), unique per path;",
            "`rationale` says why this path matters for this scenario; `success_looks_like`",
            "describes the result a good call returns; a checkpoint may also read",
            "'report: <observer>.<condition> is true|false'.",
            "",
            "Example of ONE well-formed step (a real tool from this catalog; the values are",
            "illustrative, choose ones that fit the scenario):",
            _compact_json(example),
            "",
            f"Respond ONLY by calling the `{PLAN_TOOL_NAME}` tool.",
            "",
            "CATALOG:",
            render_catalog_for_prompt(catalog, view, on_request_shown=on_request_shown),
        ]
    )


def render_observation_line(observation: Observation) -> str:
    """``find_product(query="penne") → ok: <summary>`` (the scout's compact summary)."""
    status = "ERROR" if observation.is_error else "ok"
    summary = " ".join(observation.summary.split())
    if len(summary) > OBSERVATION_LINE_LIMIT:
        summary = summary[: OBSERVATION_LINE_LIMIT - 1] + "…"
    return f"- {observation.call_label()} → {status}: {summary or '(empty)'}"


def render_informants_for_prompt(
    view: PlannerView, *, observations_shown: int | None = None
) -> str:
    """The INFORMANT REPORTS, GOALS ENABLED BY OBSERVATION and OBSERVATIONS sections.

    ``observations_shown`` keeps only the LAST that many observations (the prompt budget trims
    the oldest first) and says how many were left out.
    """
    lines: list[str] = ["INFORMANT REPORTS (observers watched the scout's calls; plan from these):"]
    if view.reports:
        lines += [f"- {r.line()}" for r in view.reports]
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("GOALS ENABLED BY OBSERVATION (the agent will be told these too):")
    if view.goals:
        lines += [f"- {g.strip()}" for g in view.goals]
    else:
        lines.append("(none)")
    lines.append("")
    observations = list(view.observations)
    trimmed = 0
    if observations_shown is not None and observations_shown < len(observations):
        trimmed = len(observations) - observations_shown
        observations = observations[len(observations) - observations_shown :]
    lines.append(
        "OBSERVATIONS (read-only calls already made against the live server; use these values):"
    )
    if trimmed:
        lines.append(f"({trimmed} earlier observation(s) left out to fit the prompt budget)")
    if observations:
        lines += [render_observation_line(o) for o in observations]
    elif not trimmed:
        lines.append("(none)")
    return "\n".join(lines)


def build_user_prompt(
    scenario: Scenario, view: PlannerView | None = None, *, observations_shown: int | None = None
) -> str:
    sections = [render_scenario_for_prompt(scenario)]
    if view is not None:
        sections.append(render_informants_for_prompt(view, observations_shown=observations_shown))
    sections.append("Produce the execution plan for this scenario now.")
    return "\n\n".join(sections)


def planner_prompt_budget(default: int = DEFAULT_PROMPT_BUDGET) -> int:
    """``MCPSIM_PLANNER_PROMPT_BUDGET`` (default 12,000 characters over system + user prompt)."""
    raw = os.environ.get(PROMPT_BUDGET_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{PROMPT_BUDGET_ENV} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{PROMPT_BUDGET_ENV} must be >= 1, got {value}")
    return value


def build_prompts(
    scenario: Scenario,
    catalog: Catalog,
    view: PlannerView | None = None,
    *,
    budget: int | None = None,
) -> tuple[str, str]:
    """``(system, user)`` for the planner, trimmed to ``budget`` characters in total.

    Observations go first (oldest first, one at a time), then the on-request list is shortened
    to a count; the disclosed digest, the reports and the scenario are never cut.
    """
    budget = planner_prompt_budget() if budget is None else budget
    if view is None:
        return build_system_prompt(catalog), build_user_prompt(scenario)
    shown = len(view.observations)
    system = build_system_prompt(catalog, view)
    user = build_user_prompt(scenario, view, observations_shown=shown)
    while len(system) + len(user) > budget and shown > 0:
        shown -= 1
        user = build_user_prompt(scenario, view, observations_shown=shown)
    on_request = len(view.on_request)
    while len(system) + len(user) > budget and on_request > 0:
        on_request = 0 if on_request <= 3 else on_request // 2
        system = build_system_prompt(catalog, view, on_request_shown=on_request)
    return system, user


def validate_arguments(step: Step, position: int, path: Path, tool: ToolInfo) -> list[str]:
    """Problems with one step's ``arguments_sketch`` against ``tool``'s input schema.

    ``position`` is the step's 1-based index in ``path`` (what ``$from_step`` counts). Checks:
    every key is a declared property (unless the schema allows additional properties), every
    literal has the declared JSON type (an integer satisfies ``number``) and sits in the
    declared ``enum``, and every reference points at an earlier step that calls a tool.
    """
    problems: list[str] = []
    schema = tool.input_schema or {}
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    additional = schema.get("additionalProperties", False)
    allows_extra = bool(additional) if not isinstance(additional, dict) else True
    declared_names = ", ".join(properties) or "(none: the tool takes no arguments)"
    for key, value in step.arguments_sketch.items():
        if key not in properties:
            if not allows_extra:
                problems.append(
                    f"argument {key!r} is not accepted by tool {tool.name}; "
                    f"its arguments are: {declared_names}"
                )
            continue
        try:
            ref = parse_reference(value)
        except ValueError as exc:
            problems.append(f"argument {key!r}: {exc}")
            continue
        if ref is not None:
            if ref.from_step >= position:
                problems.append(
                    f"argument {key!r} references step {ref.from_step}, but a reference must "
                    f"point at an EARLIER step (this is step {position}; "
                    f"{REFERENCE_KEY} is 1-based)"
                )
            elif path.steps[ref.from_step - 1].tool is None:
                problems.append(
                    f"argument {key!r} references step {ref.from_step}, which calls no tool "
                    "and so has no result to take a value from"
                )
            continue
        prop = properties[key]
        prop_schema = prop if isinstance(prop, dict) else {}
        declared = schema_types(prop_schema, schema)
        actual = json_type_of(value)
        if not type_accepts(declared, actual):
            problems.append(
                f"argument {key!r} of tool {tool.name} must be "
                f"{' or '.join(sorted(declared))}, got {actual} {_compact_json(value)}; "
                "if the value comes from an earlier step, write a "
                f'{{"{REFERENCE_KEY}": n, "path": "..."}} reference instead'
            )
            continue
        enum = enum_members(prop_schema, schema)
        if enum is not None and value not in enum:
            problems.append(
                f"argument {key!r} of tool {tool.name} must be one of "
                f"{_compact_json(enum)}, got {_compact_json(value)}"
            )
    return problems


def validate_checkpoint(text: str) -> str | None:
    """Why a checkpoint is rejected, or ``None`` when it has the required shape."""
    if CHECKPOINT_PATTERN.match(text.strip()):
        return None
    return f"checkpoint {text!r} must have the shape {CHECKPOINT_SHAPE}"


def validate_draft(
    draft: PlanDraft,
    catalog: Catalog,
    view: PlannerView | None = None,
    scenario: Scenario | None = None,
) -> list[str]:
    """Every problem with a draft, as human-readable lines; empty means valid.

    Tool names are checked against the allowed catalog (plus ``discover_tools`` when the view
    offers it), then each step's arguments (:func:`validate_arguments`), the ``expect_error``
    requirement on recovery paths, and the shape of every checkpoint. With a view, a step naming
    an on-request tool must come after a ``discover_tools`` step or after a tool step whose
    result an observer effect can react to by enabling it (``scenario`` supplies the observers).
    """
    problems: list[str] = []
    if not draft.paths:
        problems.append("plan has no paths; at least a happy path is required")
        return problems
    known = sorted(catalog.tool_names())
    discover = ToolInfo.model_validate(discover_tool_definition())
    if view is not None and view.discoverable:
        known.append(DISCOVER_TOOL_NAME)
    on_request = set(view.on_request) if view is not None else set()
    seen_ids: set[str] = set()
    for p_index, path in enumerate(draft.paths):
        where = f"paths[{p_index}] ({path.id!r})"
        if path.id in seen_ids:
            problems.append(f"paths[{p_index}]: duplicate path id {path.id!r}")
        seen_ids.add(path.id)
        if not path.steps:
            problems.append(f"{where}: has no steps")
        discovered = False
        tool_step_before = False
        for s_index, step in enumerate(path.steps):
            step_where = f"paths[{p_index}].steps[{s_index}] ({path.id!r})"
            if step.tool is None:
                continue
            if step.tool not in known:
                problems.append(
                    f"{step_where}: unknown tool {step.tool!r}; the catalog only has: "
                    f"{', '.join(known) or '(no tools)'}"
                )
                continue
            tool = discover if step.tool == DISCOVER_TOOL_NAME else catalog.tool(step.tool)
            problems += [
                f"{step_where}: {text}"
                for text in validate_arguments(step, s_index + 1, path, tool)
            ]
            if step.tool in on_request:
                enabler = (
                    view.enabling_observer(scenario, step.tool)
                    if view is not None and scenario is not None and tool_step_before
                    else None
                )
                if not discovered and enabler is None:
                    how = (
                        f"put a {DISCOVER_TOOL_NAME} step (query naming what you need) before it"
                        if view is not None and view.discoverable
                        else "only an observer effect can enable it after a tool step"
                    )
                    problems.append(
                        f"{step_where}: tool {step.tool!r} is not disclosed at plan time; {how}, "
                        "or use a disclosed tool"
                    )
            if step.tool == DISCOVER_TOOL_NAME:
                discovered = True
            else:
                tool_step_before = True
        if path.kind == "recovery" and not path.expects_error():
            problems.append(
                f"{where}: a recovery path must contain at least one step with "
                "expect_error true (the step that sends the bad input the server rejects), "
                "followed by the corrected call"
            )
        for c_index, checkpoint in enumerate(path.checkpoints):
            reason = validate_checkpoint(checkpoint)
            if reason is not None:
                problems.append(f"paths[{p_index}].checkpoints[{c_index}] ({path.id!r}): {reason}")
    if not any(p.kind == "happy" for p in draft.paths):
        problems.append("plan has no path of kind 'happy'")
    return problems


def _format_validation_error(exc: ValidationError) -> list[str]:
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"{loc}: {err['msg']}")
    return lines


def extract_draft(
    response: LLMResponse,
    catalog: Catalog,
    view: PlannerView | None = None,
    scenario: Scenario | None = None,
) -> tuple[PlanDraft | None, list[str]]:
    """Pull the structured plan out of a response and validate it.

    Returns ``(draft, problems)``; ``draft`` is ``None`` when the response carried no usable
    tool call at all. ``problems`` is empty exactly when the draft is valid.
    """
    blocks = [b for b in response.tool_uses() if b.get("name") == PLAN_TOOL_NAME]
    if not blocks:
        return None, [
            f"the response did not call the {PLAN_TOOL_NAME} tool; "
            "respond only by calling it with the plan"
        ]
    raw = blocks[0].get("input")
    if not isinstance(raw, dict):
        return None, [f"{PLAN_TOOL_NAME} input must be a JSON object, got {type(raw).__name__}"]
    try:
        draft = PlanDraft.model_validate(raw)
    except ValidationError as exc:
        return None, _format_validation_error(exc)
    return draft, validate_draft(draft, catalog, view, scenario)


def _reask_text(problems: list[str]) -> str:
    bullet = "\n".join(f"- {p}" for p in problems)
    return (
        "The plan you emitted is invalid and was rejected:\n"
        f"{bullet}\n\n"
        "Fix every problem and emit the whole corrected plan again by calling "
        f"`{PLAN_TOOL_NAME}`. Use only tools and argument names that appear in the CATALOG, "
        "typed as listed; mark the deliberate failure in a recovery path with expect_error "
        "true; shape every checkpoint as '<where>: <condition>'."
    )


def reask_content(response: LLMResponse, problems: list[str]) -> str | list[dict[str, Any]]:
    """The user turn that follows a rejected plan.

    The Messages API requires every assistant ``tool_use`` to be answered by a ``tool_result``
    with the same id in the very next user message, so the validation error travels as an
    error tool result. When the model never called the tool, plain text is the only option.
    """
    text = _reask_text(problems)
    blocks = [b for b in response.tool_uses() if b.get("name") == PLAN_TOOL_NAME]
    tool_use_id = blocks[0].get("id") if blocks else None
    if not isinstance(tool_use_id, str) or not tool_use_id:
        return text
    return [{"type": "tool_result", "tool_use_id": tool_use_id, "content": text, "is_error": True}]


async def plan_with_llm(
    scenario: Scenario,
    catalog: Catalog,
    llm: LLM,
    scout: ScoutResult | None = None,
    *,
    prompt_budget: int | None = None,
) -> ExecutionPlan:
    """One structured-output call, plus a single re-ask when validation fails.

    With a ``scout`` the prompts are the orchestrator's (disclosed digest, on-request names,
    informant reports, goals, observations) and ``scout.planner_prompt_chars`` records their
    final size after trimming.
    """
    view = planner_view(scenario, catalog, scout) if scout is not None else None
    system, user = build_prompts(scenario, catalog, view, budget=prompt_budget)
    if scout is not None:
        scout.planner_prompt_chars = len(system) + len(user)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    tools = [plan_tool_definition(catalog, view)]
    tool_choice = {"type": "tool", "name": PLAN_TOOL_NAME}

    problems: list[str] = []
    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        response = await llm.complete(
            model=scenario.models.planner,
            system=system,
            messages=list(messages),  # a snapshot: the re-ask below extends our own list
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=PLAN_MAX_TOKENS,
        )
        draft, problems = extract_draft(response, catalog, view, scenario)
        if draft is not None and not problems:
            return ExecutionPlan(
                scenario=scenario.name,
                catalog_digest=catalog.digest(),
                paths=list(draft.paths),
            )
        if attempt < MAX_PLAN_ATTEMPTS:
            # Keep the rejected turn in context so the model can correct it rather than start
            # over blind. The assistant turn must carry the tool_use block exactly as returned.
            messages.append({"role": "assistant", "content": list(response.content)})
            messages.append({"role": "user", "content": reask_content(response, problems)})
    raise PlanError(
        f"planner for scenario {scenario.name!r} produced an invalid plan "
        f"{MAX_PLAN_ATTEMPTS} times; last errors:\n" + "\n".join(f"- {p}" for p in problems)
    )


def _first_type(schema: dict[str, Any]) -> str | None:
    """The JSON Schema type of a property, looking through ``anyOf`` unions (``T | None``)."""
    declared = schema.get("type")
    if isinstance(declared, str):
        return declared
    if isinstance(declared, list):
        for candidate in declared:
            if isinstance(candidate, str) and candidate != "null":
                return candidate
    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    found = _first_type(variant)
                    if found is not None and found != "null":
                        return found
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        first = schema["enum"][0]
        if isinstance(first, bool):
            return "boolean"
        if isinstance(first, int):
            return "integer"
        if isinstance(first, str):
            return "string"
    return None


def default_for(schema: dict[str, Any]) -> tuple[bool, Any]:
    """``(ok, value)``: a placeholder for a required argument, per the build spec.

    strings → ``""``, integers → ``1``, booleans → ``False``, arrays → ``[]``. A declared
    ``default`` or the first ``enum`` member wins when present. Anything else is not
    defaultable and ``ok`` is ``False``.
    """
    if "default" in schema:
        return True, schema["default"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return True, enum[0]
    kind = _first_type(schema)
    if kind == "string":
        return True, ""
    if kind == "integer":
        return True, 1
    if kind == "number":
        return True, 1
    if kind == "boolean":
        return True, False
    if kind == "array":
        return True, []
    return False, None


def default_arguments(tool: ToolInfo) -> tuple[dict[str, Any] | None, list[str]]:
    """Sketch arguments covering the tool's required properties.

    Returns ``(arguments, undefaultable)``; ``arguments`` is ``None`` when any required
    argument has no placeholder, and ``undefaultable`` lists those names.
    """
    schema = tool.input_schema or {}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    if not isinstance(required, list):
        required = []
    arguments: dict[str, Any] = {}
    undefaultable: list[str] = []
    for name in required:
        prop = properties.get(name)
        ok, value = default_for(prop) if isinstance(prop, dict) else (False, None)
        if ok:
            arguments[str(name)] = value
        else:
            undefaultable.append(str(name))
    if undefaultable:
        return None, undefaultable
    return arguments, []


def is_expensive(tool: ToolInfo) -> bool:
    """Does the description claim the tool spends money, credits or time?

    A description that says "free" or "no LLM" is believed over an incidental
    cost word, so "Free — no LLM calls" is free and "SLOW (10-60s) and costs
    real Claude API credits" is expensive.
    """
    desc = tool.description or ""
    return EXPENSIVE_PATTERN.search(desc) is not None and FREE_PATTERN.search(desc) is None


def _accepts_literal(prop_schema: dict[str, Any], root: dict[str, Any], value: Any) -> bool:
    """Would :func:`validate_arguments` accept ``value`` for this property (type and enum)?"""
    if not type_accepts(schema_types(prop_schema, root), json_type_of(value)):
        return False
    enum = enum_members(prop_schema, root)
    return enum is None or value in enum


def dry_run_arguments(
    tool: ToolInfo, spec: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Sketch arguments for a dry-run step: expected-outcome values first, then placeholders.

    Every property whose name is a top-level ``expected_outcome.json`` key with a plain
    string/number value that the schema accepts takes that value (``query: penne`` →
    ``query="penne"``), required or not; the other required properties take
    :func:`default_for` placeholders. Returns ``(arguments, undefaultable, from_expected)``;
    ``arguments`` is ``None`` when a required argument has neither.
    """
    schema = tool.input_schema or {}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required_raw = schema.get("required")
    required = [str(n) for n in required_raw] if isinstance(required_raw, list) else []
    arguments: dict[str, Any] = {}
    undefaultable: list[str] = []
    from_expected: list[str] = []
    for name, prop in properties.items():
        prop_schema = prop if isinstance(prop, dict) else {}
        found, value = plain_expected_value(spec, str(name))
        if found and _accepts_literal(prop_schema, schema, value):
            arguments[str(name)] = value
            from_expected.append(str(name))
            continue
        if name not in required:
            continue
        ok, default = default_for(prop_schema)
        if ok:
            arguments[str(name)] = default
        else:
            undefaultable.append(str(name))
    undefaultable.extend(name for name in required if name not in properties)
    if undefaultable:
        return None, undefaultable, from_expected
    return arguments, [], from_expected


def dry_run_plan(
    scenario: Scenario,
    catalog: Catalog,
    scout: ScoutResult | None = None,
    *,
    prompt_budget: int | None = None,
) -> ExecutionPlan:
    """A one-path happy plan over the goal-relevant tools: no LLM (DESIGN §6 ``MCPSIM_DRY_RUN``).

    Candidates are the allowed tools that do not claim a cost and are not write tools (unless
    :func:`mcpsim.scoping.write_intent`), ranked by relevance to the scenario; tools that share
    no vocabulary with it are skipped (when none does, the first :data:`DRY_RUN_FALLBACK` in
    catalog order are called so the smoke still reaches the server), and at most
    ``budgets.max_tool_calls`` steps are planned, so the budget is never what ends a dry run.
    With a ``scout``, its expected-outcome lookups that answered come first (proven calls) and
    the checkpoints record the scout's reports (``report: <obs>.<cond> is true``); the prompt
    the LLM planner would have received is still built so ``scout.planner_prompt_chars`` is
    measured. Arguments come from :func:`dry_run_arguments`. The rationale names every skipped
    tool and why: expensive, write, irrelevant, beyond the budget, or undefaultable.
    """
    budget = scenario.budgets.max_tool_calls
    spec = scenario.expected_outcome.json
    intent = write_intent(scenario)
    skipped_expensive: list[str] = []
    skipped_write: list[str] = []
    candidates: list[ToolInfo] = []
    for tool in catalog.tools:
        if is_expensive(tool):
            skipped_expensive.append(tool.name)
        elif not intent and is_write_tool(tool):
            skipped_write.append(tool.name)
        else:
            candidates.append(tool)
    ranked = rank_tools(scenario_terms(scenario), candidates)
    relevant = [tool for tool, score in ranked if score > 0]
    skipped_irrelevant = [tool.name for tool, score in ranked if score <= 0]
    fallback = False
    if not relevant and candidates:
        fallback = True
        relevant = candidates[: min(DRY_RUN_FALLBACK, budget)]
        skipped_irrelevant = [t.name for t in candidates if t not in relevant]
    proven: list[str] = []
    if scout is not None:
        for observation in scout.lookups():
            if catalog.has_tool(observation.name) and observation.name not in proven:
                proven.append(observation.name)
        by_name = {t.name: t for t in candidates}
        proven = [n for n in proven if n in by_name]
        first = [by_name[n] for n in proven]
        relevant = first + [t for t in relevant if t.name not in proven]
        skipped_irrelevant = [n for n in skipped_irrelevant if n not in proven]

    steps: list[Step] = []
    skipped_beyond: list[str] = []
    skipped_undefaultable: list[tuple[str, list[str]]] = []
    filled: list[str] = []
    for tool in relevant:
        if len(steps) >= budget:
            skipped_beyond.append(tool.name)
            continue
        arguments, undefaultable, from_expected = dry_run_arguments(tool, spec)
        if arguments is None:
            skipped_undefaultable.append((tool.name, undefaultable))
            continue
        filled.extend(f"{tool.name}({k}={_compact_json(arguments[k])})" for k in from_expected)
        if from_expected:
            source = "arguments from the expected outcome"
            answer = ", ".join(f"{k}={_compact_json(arguments[k])}" for k in from_expected)
            success = (
                f"{tool.name} returns a result for {answer} (an error result still counts as "
                "the server answering)"
            )
        else:
            source = "placeholder arguments"
            success = (
                f"{tool.name} returns a result (an error result still counts as the server "
                "answering)"
            )
        steps.append(
            Step(
                intent=f"Call {tool.name} with {source} to prove it answers",
                tool=tool.name,
                arguments_sketch=arguments,
                success_looks_like=success,
            )
        )
    called = [s.tool for s in steps if s.tool is not None]

    rationale_parts = [
        "Dry run: no LLM was used. One happy path over the allowed tools that share vocabulary "
        f"with the scenario, most relevant first, at most max_tool_calls={budget}: "
        + (", ".join(called) if called else "(none)")
        + "."
    ]
    if proven:
        rationale_parts.append(
            "The scout already proved these expected-outcome lookups answer, so they come "
            "first: " + ", ".join(proven) + "."
        )
    if fallback:
        rationale_parts.append(
            "No tool shares vocabulary with the scenario, so the first "
            f"{len(relevant)} candidate(s) in catalog order are called instead."
        )
    if filled:
        rationale_parts.append(
            "Arguments named by a top-level expected_outcome.json key with a plain value take "
            "that value: " + ", ".join(filled) + "; other required arguments take schema "
            "placeholders."
        )
    elif called:
        rationale_parts.append("Required arguments take schema placeholders.")
    if skipped_expensive:
        rationale_parts.append(
            "Skipped because the description says they cost money/credits or are slow: "
            + ", ".join(skipped_expensive)
            + "."
        )
    if skipped_write:
        rationale_parts.append(
            "Skipped as write tools (neither the goal nor an instruction asks for a write): "
            + ", ".join(skipped_write)
            + "."
        )
    if skipped_irrelevant:
        rationale_parts.append(
            "Skipped as irrelevant (no shared vocabulary with the scenario): "
            + ", ".join(skipped_irrelevant)
            + "."
        )
    if skipped_beyond:
        rationale_parts.append(
            f"Skipped as least relevant beyond max_tool_calls={budget}: "
            + ", ".join(skipped_beyond)
            + "."
        )
    if skipped_undefaultable:
        rationale_parts.append(
            "Skipped because a required argument has no schema default: "
            + "; ".join(f"{name} ({', '.join(args)})" for name, args in skipped_undefaultable)
            + "."
        )
    if not steps:
        rationale_parts.append("No tool qualified, so the path has no steps.")
    checkpoints = [f"transcript: contains a tool_call for {name}" for name in called]
    checkpoints.append(
        "final_result: equals the structured content of the successful tool result that covers "
        "the most expected_outcome.json keys (the last one otherwise), or is null when no step "
        "returned structured content"
    )
    if scout is not None:
        checkpoints += [
            f"report: {r.key} is {r.shown}" for r in scout.reports if r.value is not None
        ]
    path = Path(
        id=DRY_RUN_PATH_ID,
        kind="happy",
        title=f"Dry run over {len(called)} goal-relevant tool(s)",
        rationale=" ".join(rationale_parts),
        steps=steps,
        checkpoints=checkpoints,
    )
    # The dry-run plan must satisfy the same rules an LLM plan does.
    problems = validate_draft(PlanDraft(paths=[path]), catalog)
    if problems:  # pragma: no cover - would be a bug in dry_run_arguments
        raise PlanError("dry-run plan failed its own validation:\n" + "\n".join(problems))
    if scout is not None:
        system, user = build_prompts(
            scenario, catalog, planner_view(scenario, catalog, scout), budget=prompt_budget
        )
        scout.planner_prompt_chars = len(system) + len(user)
    return ExecutionPlan(scenario=scenario.name, catalog_digest=catalog.digest(), paths=[path])


async def plan(
    scenario: Scenario,
    catalog: Catalog,
    llm: LLM | None = None,
    *,
    scout: ScoutResult | None = None,
    dry_run: bool = False,
    prompt_budget: int | None = None,
) -> ExecutionPlan:
    """Plan the scenario against the catalog.

    ``dry_run`` is a plain argument: the caller defaults it from ``MCPSIM_DRY_RUN`` (the runner
    and CLI do). With ``dry_run=True`` the ``llm`` is not touched and may be ``None``. ``scout``
    is the :class:`~mcpsim.scout.ScoutResult` the runner produced before planning (its
    ``planner_prompt_chars`` is filled in here); ``prompt_budget`` overrides
    ``MCPSIM_PLANNER_PROMPT_BUDGET``.
    """
    if dry_run:
        return dry_run_plan(scenario, catalog, scout, prompt_budget=prompt_budget)
    if llm is None:
        raise PlanError("an LLM is required unless dry_run=True")
    return await plan_with_llm(scenario, catalog, llm, scout, prompt_budget=prompt_budget)
