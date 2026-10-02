"""Planner (DESIGN §2 "Planner").

Input: a :class:`~mcpsim.scenario.Scenario` and the server's live
:class:`~mcpsim.mcpclient.Catalog`. Output: an :class:`~mcpsim.plan.ExecutionPlan` with several
paths (happy, recovery, alternative, boundary, policy), produced by one structured-output LLM
call. Every ``tool`` named in a step must exist in the catalog; a plan that fails validation is
re-asked **once** with the error appended, then :class:`PlanError` is raised.

``dry_run=True`` needs no LLM: it emits a one-path happy plan whose steps call every catalog tool
whose required arguments can all be defaulted from the schema, skipping tools whose description
says they cost money or credits, and says so in the path's rationale.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mcpsim.llm import LLM, LLMResponse
from mcpsim.mcpclient import Catalog, ToolInfo
from mcpsim.plan import PATH_KINDS, ExecutionPlan, Path, Step
from mcpsim.scenario import Scenario

PLAN_TOOL_NAME = "emit_execution_plan"
PLAN_MAX_TOKENS = 8192
MAX_PLAN_ATTEMPTS = 2  # the first ask plus one re-ask with the validation error
DRY_RUN_PATH_ID = "happy-dry-run"

# Tools a dry-run plan must not call (DESIGN §6 build spec): descriptions mentioning spend.
EXPENSIVE_PATTERN = re.compile(r"credit|costs|SLOW|LLM")


class PlanError(RuntimeError):
    """The planner could not produce a valid plan (after the single allowed re-ask)."""


class PlanDraft(BaseModel):
    """The structured output the planner asks the LLM for: just the paths.

    ``scenario`` and ``catalog_digest`` are filled in by :func:`plan`, never by the model.
    """

    model_config = ConfigDict(extra="forbid")

    paths: list[Path] = Field(default_factory=list)


def plan_tool_definition() -> dict[str, Any]:
    """The single Anthropic tool used for structured output."""
    return {
        "name": PLAN_TOOL_NAME,
        "description": (
            "Emit the execution plan: an ordered list of distinct paths through the server, "
            "each with concrete steps and observable checkpoints."
        ),
        "input_schema": PlanDraft.model_json_schema(),
    }


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def render_catalog_for_prompt(catalog: Catalog) -> str:
    """The catalog as the planner sees it: names, descriptions, schemas, resources, prompts."""
    lines: list[str] = []
    lines.append(f"TOOLS ({len(catalog.tools)}) — the ONLY tools that exist:")
    for tool in catalog.tools:
        lines.append(f"- {tool.name}")
        if tool.description.strip():
            lines.append(f"  description: {tool.description.strip()}")
        lines.append(f"  input_schema: {_compact_json(tool.input_schema)}")
        if tool.output_schema is not None:
            lines.append(f"  output_schema: {_compact_json(tool.output_schema)}")
    lines.append(f"RESOURCES ({len(catalog.resources)}):")
    for res in catalog.resources:
        desc = f" — {res.description.strip()}" if res.description.strip() else ""
        lines.append(f"- {res.uri} [{res.name}]{desc}")
    lines.append(f"RESOURCE TEMPLATES ({len(catalog.resource_templates)}):")
    for tmpl in catalog.resource_templates:
        desc = f" — {tmpl.description.strip()}" if tmpl.description.strip() else ""
        lines.append(f"- {tmpl.uri_template} [{tmpl.name}]{desc}")
    lines.append(f"PROMPTS ({len(catalog.prompts)}):")
    for prompt in catalog.prompts:
        arg_names = ", ".join(str(a.get("name", "?")) for a in prompt.arguments)
        desc = f" — {prompt.description.strip()}" if prompt.description.strip() else ""
        lines.append(f"- {prompt.name}({arg_names}){desc}")
    return "\n".join(lines)


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


def build_system_prompt(catalog: Catalog) -> str:
    kinds = ", ".join(PATH_KINDS)
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
            "Rules:",
            "- A step's `tool` MUST be a tool name from the TOOLS list below, spelled exactly;",
            "  never invent tools, resources or arguments that are not in the catalog. A step",
            "  that needs no tool (e.g. composing the final answer) sets `tool` to null.",
            "- `arguments_sketch` is a JSON object of plausible arguments matching the tool's",
            "  input_schema (values may be placeholders the agent will refine).",
            "- `success_looks_like` describes the result a good call returns.",
            "- `checkpoints` are OBSERVABLE FACTS a judge can verify from the transcript, phrased",
            "  concretely, e.g. 'origin_status in the final answer equals the value the plan",
            "  tool returned', never vague ('the agent did well').",
            "- `id` is a short slug (letters, digits, '.', '_', '-'), unique per path;",
            "  `rationale` says why this path matters for this scenario.",
            "- Prefer tools that are cheap; if a tool's description says it is slow or costs",
            "  credits, use it only when the goal needs it and say so in the rationale.",
            f"Respond ONLY by calling the `{PLAN_TOOL_NAME}` tool.",
            "",
            "CATALOG:",
            render_catalog_for_prompt(catalog),
        ]
    )


def build_user_prompt(scenario: Scenario) -> str:
    return render_scenario_for_prompt(scenario) + (
        "\n\nProduce the execution plan for this scenario now."
    )


def validate_draft(draft: PlanDraft, catalog: Catalog) -> list[str]:
    """Every problem with a draft, as human-readable lines; empty means valid."""
    problems: list[str] = []
    if not draft.paths:
        problems.append("plan has no paths; at least a happy path is required")
        return problems
    known = catalog.tool_names()
    seen_ids: set[str] = set()
    for p_index, path in enumerate(draft.paths):
        if path.id in seen_ids:
            problems.append(f"paths[{p_index}]: duplicate path id {path.id!r}")
        seen_ids.add(path.id)
        if not path.steps:
            problems.append(f"paths[{p_index}] ({path.id!r}): has no steps")
        for s_index, step in enumerate(path.steps):
            if step.tool is not None and step.tool not in known:
                problems.append(
                    f"paths[{p_index}].steps[{s_index}] ({path.id!r}): unknown tool "
                    f"{step.tool!r}; the catalog only has: {', '.join(known) or '(no tools)'}"
                )
    if not any(p.kind == "happy" for p in draft.paths):
        problems.append("plan has no path of kind 'happy'")
    return problems


def _format_validation_error(exc: ValidationError) -> list[str]:
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"{loc}: {err['msg']}")
    return lines


def extract_draft(response: LLMResponse, catalog: Catalog) -> tuple[PlanDraft | None, list[str]]:
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
    return draft, validate_draft(draft, catalog)


def _reask_text(problems: list[str]) -> str:
    bullet = "\n".join(f"- {p}" for p in problems)
    return (
        "The plan you emitted is invalid and was rejected:\n"
        f"{bullet}\n\n"
        "Fix every problem and emit the whole corrected plan again by calling "
        f"`{PLAN_TOOL_NAME}`. Use only tools that appear in the CATALOG."
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


async def plan_with_llm(scenario: Scenario, catalog: Catalog, llm: LLM) -> ExecutionPlan:
    """One structured-output call, plus a single re-ask when validation fails."""
    system = build_system_prompt(catalog)
    messages: list[dict[str, Any]] = [{"role": "user", "content": build_user_prompt(scenario)}]
    tools = [plan_tool_definition()]
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
        draft, problems = extract_draft(response, catalog)
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
    """Does the description say the tool spends money, credits or an LLM call?"""
    return EXPENSIVE_PATTERN.search(tool.description) is not None


def dry_run_plan(scenario: Scenario, catalog: Catalog) -> ExecutionPlan:
    """A one-path happy plan from the catalog alone: no LLM (DESIGN §6 ``MCPSIM_DRY_RUN``)."""
    steps: list[Step] = []
    skipped_expensive: list[str] = []
    skipped_undefaultable: list[tuple[str, list[str]]] = []
    for tool in catalog.tools:
        if is_expensive(tool):
            skipped_expensive.append(tool.name)
            continue
        arguments, undefaultable = default_arguments(tool)
        if arguments is None:
            skipped_undefaultable.append((tool.name, undefaultable))
            continue
        steps.append(
            Step(
                intent=f"Call {tool.name} with placeholder arguments to prove it answers",
                tool=tool.name,
                arguments_sketch=arguments,
                success_looks_like=(
                    f"{tool.name} returns a result (an error result still counts as the "
                    "server answering)"
                ),
            )
        )
    rationale_parts = [
        "Dry run: no LLM was used. One happy path that calls every catalog tool whose required "
        "arguments can be defaulted from its schema, in catalog order."
    ]
    if skipped_expensive:
        rationale_parts.append(
            "Skipped because the description says they cost money/credits or are slow: "
            + ", ".join(skipped_expensive)
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
    called = [s.tool for s in steps if s.tool is not None]
    checkpoints = [f"The transcript contains a tool_call for {name}" for name in called]
    checkpoints.append(
        "The final_result is the structured content of the last tool result, or null when "
        "the last tool returned no structured content"
    )
    path = Path(
        id=DRY_RUN_PATH_ID,
        kind="happy",
        title=f"Dry run over {len(called)} catalog tool(s)",
        rationale=" ".join(rationale_parts),
        steps=steps,
        checkpoints=checkpoints,
    )
    return ExecutionPlan(scenario=scenario.name, catalog_digest=catalog.digest(), paths=[path])


async def plan(
    scenario: Scenario,
    catalog: Catalog,
    llm: LLM | None = None,
    *,
    dry_run: bool = False,
) -> ExecutionPlan:
    """Plan the scenario against the catalog.

    ``dry_run`` is a plain argument: the caller defaults it from ``MCPSIM_DRY_RUN`` (the runner
    and CLI do), this module never reads the environment. With ``dry_run=True`` the ``llm`` is
    not touched and may be ``None``.
    """
    if dry_run:
        return dry_run_plan(scenario, catalog)
    if llm is None:
        raise PlanError("an LLM is required unless dry_run=True")
    return await plan_with_llm(scenario, catalog, llm)
