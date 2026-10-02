"""Tool scoping: which of a server's tools a scenario needs (DESIGN §2 "Tool scoping").

Pure functions, no LLM. The agent loop uses them to choose the *initial* set a ``progressive``
scenario discloses and to answer the ``discover_tools`` meta-tool; the dry run uses them to plan
goal-relevant tools instead of walking the catalog.

* :func:`scenario_terms` — the stemmed tokens of the goal, the instructions, the expected
  outcome's prose, the dotted keys of ``expected_outcome.json`` and its plain string values.
* :func:`tool_terms` / :func:`relevance` — the same tokens for a tool (name, description,
  input property names, top-level output keys) and a weighted overlap: a name token counts 3,
  an output key or input property 2, a description word 1.
* :func:`is_write_tool` — the server's ``read_only_hint``/``destructive_hint`` annotations, or
  a ``submit_``/``review_``/``approve_``/... name.
* :func:`write_intent` — does the goal or an instruction ask the agent to submit, record,
  review, approve, reject, write, register or add something (an imperative, not the noun
  "the product record")?
* :func:`initial_tools` — the top-``k`` tools by relevance, plus every tool whose output keys
  cover a top-level key of ``expected_outcome.json``, write tools only with write intent, never
  fewer than ``min(3, len(allowed))``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from mcpsim.mcpclient import Catalog, ToolInfo
from mcpsim.scenario import Scenario

NAME_WEIGHT = 3.0
OUTPUT_WEIGHT = 2.0
INPUT_WEIGHT = 2.0
DESCRIPTION_WEIGHT = 1.0
INITIAL_K = 5
INITIAL_FLOOR = 3

_WORD = re.compile(r"[a-z0-9]+")
_FIRST_SENTENCE = re.compile(r"^(.*?[.!?])(?:\s|$)")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_']*")
_SENTENCE_SPLIT = re.compile(r"[.;:!?\n]")

# Words that carry no signal about which tool a scenario needs. English function words, the
# verbs every instruction uses ("use", "report", "say") and the words every MCP tool
# description uses ("tool", "call", "returns", "free", "LLM").
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then else of to in on at by for from with without as is are was
    were be been being it its this that these those there here which who whom what when where
    why how do does did done not no yes me my i you your we our they them their he she his her
    one ones any all each every some such than too very can could should would may might must
    will shall just only also about into over under again further once so up down out off more
    most other same own both few get got tell say said give take make made use used using let
    know want need please exactly exact plainly never always instead rather still already
    tool tools server call calls called return returns returned result results value values
    free llm first back come comes way like per via report reports reported reporting
    """.split()
)

WRITE_TOOL_NAME = re.compile(
    r"^(submit|review|approve|reject|create|update|delete|write|post|set)_"
)
_WRITE_VERB = re.compile(
    r"^(submit(s|ted|ting)?|record(s|ed|ing)?|review(s|ed|ing)?|approv(e|es|ed|ing)"
    r"|reject(s|ed|ing)?|writ(e|es|ing|ten)|register(s|ed|ing)?|add(s|ed|ing)?)$"
)
_WRITE_VERB_STEM = re.compile(r"^(submit|record|review|approve|reject|write|register|add)$")
_REPORT_A_LABEL = re.compile(
    r"\breport(s|ed|ing)?\s+(a|an|the|this|that|my|each|every|one|another)?\s*label", re.I
)
# A write verb is read as an instruction to the agent when it opens a sentence or follows one
# of these words ("then submit", "must record", "after submitting"); "the product record" and
# "the server rejects it" are nouns and third parties, not intent.
_IMPERATIVE_LEAD: frozenset[str] = frozenset(
    """
    to and then must should please can will may not never do also or just first now you i we
    always go still after before finally next
    """.split()
)


# --- tokens -----------------------------------------------------------------------------------


def stem(token: str) -> str:
    """A deliberately simple stem: strip a trailing ``s`` from words longer than three letters.

    ``prices`` → ``price``, ``origins`` → ``origin``, ``status`` → ``statu`` (both sides are
    stemmed the same way, so overlap still works). The pantry server's own matcher uses the
    same rule; nothing smarter is needed for a relevance ranking.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokens(text: str) -> set[str]:
    """Lowercase, stemmed, stopword-free word tokens of ``text`` (``_`` and ``.`` split too)."""
    found: set[str] = set()
    for raw in _WORD.findall(text.lower()):
        if len(raw) < 2 or raw in STOPWORDS:
            continue
        stemmed = stem(raw)
        if stemmed in STOPWORDS:
            continue
        found.add(stemmed)
    return found


def first_sentence(text: str) -> str:
    """The first sentence of a description, whitespace collapsed (the whole text if none ends)."""
    flat = " ".join(text.split())
    m = _FIRST_SENTENCE.match(flat)
    return m.group(1) if m else flat


# --- the scenario side ------------------------------------------------------------------------


def expected_top_level_keys(spec: dict[str, Any] | None) -> list[str]:
    """Top-level field names from the dotted paths of ``expected_outcome.json``, in order."""
    names: list[str] = []
    for key in spec or {}:
        head = re.split(r"[.\[]", str(key), maxsplit=1)[0].strip()
        if head and head not in names:
            names.append(head)
    return names


def _is_operator_spec(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        isinstance(k, str) and k.startswith("$") for k in value
    )


def plain_expected_value(spec: dict[str, Any] | None, key: str) -> tuple[bool, Any]:
    """``(found, value)`` when ``expected_outcome.json`` pins top-level ``key`` to a plain
    string or number (``query: penne``), not to an operator object (``{$gt: 0}``)."""
    if not spec or key not in spec:
        return False, None
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return False, None
    return True, value


def scenario_terms(scenario: Scenario) -> set[str]:
    """Every token that says what the scenario is about (the role's persona is left out)."""
    parts: list[str] = [scenario.goal, *scenario.instructions]
    outcome = scenario.expected_outcome
    if outcome.text:
        parts.append(outcome.text)
    for key, value in (outcome.json or {}).items():
        parts.append(re.sub(r"\[[^\]]*\]", " ", str(key)).replace(".", " ").replace("_", " "))
        if isinstance(value, str):
            parts.append(value)
    return tokens(" ".join(parts))


def write_intent(scenario: Scenario) -> bool:
    """Does the goal or an instruction tell the agent to perform a write?

    True when a write verb (submit, record, review, approve, reject, write, register, add) is
    used as an instruction — opening a sentence, after "to"/"then"/"must"/"after"/... — when a
    tool-like identifier starts with one (``submit_origin_evidence``), or when the text says
    "report a label". "Quote the product record" and "when the server rejects it" are not.
    """
    for text in (scenario.goal, *scenario.instructions):
        if _REPORT_A_LABEL.search(text):
            return True
        for sentence in _SENTENCE_SPLIT.split(text):
            previous: str | None = None
            for word in _IDENTIFIER.findall(sentence):
                lowered = word.lower()
                if "_" in lowered:
                    if _WRITE_VERB_STEM.match(lowered.split("_", 1)[0]):
                        return True
                elif _WRITE_VERB.match(lowered) and (
                    previous is None or previous in _IMPERATIVE_LEAD
                ):
                    return True
                previous = lowered
    return False


# --- the tool side ----------------------------------------------------------------------------


def _resolve_local_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow a local ``$ref`` (``#/$defs/Name``) one level; anything else is returned as is."""
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    target: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(target, dict) or part not in target:
            return schema
        target = target[part]
    return target if isinstance(target, dict) else schema


def output_key_names(schema: dict[str, Any] | None) -> list[str]:
    """The top-level keys a tool returns, through a single ``result`` wrapper.

    The MCP SDK wraps non-object return types as ``{"result": ...}``; for an object inside it
    the object's keys are returned, for an array of objects the element keys (what the agent
    reads per item). A one-level ``$ref`` is resolved. Empty when the schema names no keys.
    """
    if not schema:
        return []
    resolved = _resolve_local_ref(schema, schema)
    properties = resolved.get("properties")
    if not isinstance(properties, dict):
        return []
    if list(properties) == ["result"]:
        inner_raw = properties["result"]
        inner = _resolve_local_ref(inner_raw, schema) if isinstance(inner_raw, dict) else {}
        inner_props = inner.get("properties")
        if isinstance(inner_props, dict):
            return [str(k) for k in inner_props]
        items = inner.get("items")
        item_schema = _resolve_local_ref(items, schema) if isinstance(items, dict) else {}
        item_props = item_schema.get("properties")
        if isinstance(item_props, dict):
            return [str(k) for k in item_props]
        return []
    return [str(k) for k in properties]


def input_property_names(tool: ToolInfo) -> list[str]:
    properties = (tool.input_schema or {}).get("properties")
    return [str(k) for k in properties] if isinstance(properties, dict) else []


def tool_term_sources(tool: ToolInfo) -> dict[str, set[str]]:
    """The tool's tokens by origin: ``name``, ``output``, ``input``, ``description``."""
    output = " ".join(output_key_names(tool.output_schema))
    inputs = " ".join(input_property_names(tool))
    return {
        "name": tokens(tool.name.replace("_", " ")),
        "output": tokens(output.replace("_", " ")),
        "input": tokens(inputs.replace("_", " ")),
        "description": tokens(tool.description),
    }


def tool_terms(tool: ToolInfo) -> set[str]:
    """Every token of the tool: name, description, input property names, output keys."""
    found: set[str] = set()
    for source in tool_term_sources(tool).values():
        found |= source
    return found


def relevance_to_terms(terms: set[str], tool: ToolInfo) -> float:
    """Weighted overlap of ``terms`` with the tool: name 3, output key 2, input property 2,
    description word 1 (each distinct token counted once per source)."""
    sources = tool_term_sources(tool)
    return (
        NAME_WEIGHT * len(terms & sources["name"])
        + OUTPUT_WEIGHT * len(terms & sources["output"])
        + INPUT_WEIGHT * len(terms & sources["input"])
        + DESCRIPTION_WEIGHT * len(terms & sources["description"])
    )


def relevance(scenario: Scenario, tool: ToolInfo) -> float:
    """How much ``tool`` has to do with ``scenario`` (0 means no shared vocabulary at all)."""
    return relevance_to_terms(scenario_terms(scenario), tool)


def rank_tools(terms: set[str], tools: Iterable[ToolInfo]) -> list[tuple[ToolInfo, float]]:
    """``(tool, score)`` by descending :func:`relevance_to_terms`; ties keep catalog order."""
    scored = [(tool, relevance_to_terms(terms, tool)) for tool in tools]
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][1], i))
    return [scored[i] for i in order]


def is_write_tool(tool: ToolInfo) -> bool:
    """Does the tool change state? From the server's annotations, else from its name.

    ``read_only_hint`` explicitly ``False`` or ``destructive_hint`` ``True`` (the MCP tool
    annotations, as the SDK dumps them, camelCase accepted too), or a name starting with
    ``submit_``, ``review_``, ``approve_``, ``reject_``, ``create_``, ``update_``, ``delete_``,
    ``write_``, ``post_`` or ``set_``.
    """
    annotations = tool.annotations or {}
    read_only = annotations.get("read_only_hint", annotations.get("readOnlyHint"))
    if read_only is False:
        return True
    destructive = annotations.get("destructive_hint", annotations.get("destructiveHint"))
    if destructive is True:
        return True
    return WRITE_TOOL_NAME.match(tool.name) is not None


def covers_expected_key(tool: ToolInfo, expected_keys: Iterable[str]) -> bool:
    """Does one of the tool's top-level output keys name a top-level expected-outcome key?"""
    wanted = set(expected_keys)
    return any(key in wanted for key in output_key_names(tool.output_schema))


def initial_tools(scenario: Scenario, catalog: Catalog, k: int = INITIAL_K) -> list[ToolInfo]:
    """The tools a ``progressive`` scenario offers on turn one, most relevant first.

    From the allowed ``catalog``: write tools are left out unless :func:`write_intent`; the
    top-``k`` by :func:`relevance` (score above zero, ties by catalog order) are taken; every
    tool whose output keys cover a top-level ``expected_outcome.json`` key is forced in; and
    the set is padded in catalog order to ``min(3, len(allowed))`` so an agent always has
    something to call. The result is ordered by score, then catalog order.
    """
    terms = scenario_terms(scenario)
    intent = write_intent(scenario)
    candidates = [t for t in catalog.tools if intent or not is_write_tool(t)]
    ranked = rank_tools(terms, candidates)
    score_of = {tool.name: score for tool, score in ranked}
    chosen: list[ToolInfo] = [tool for tool, score in ranked if score > 0][:k]
    names = {t.name for t in chosen}
    expected_keys = expected_top_level_keys(scenario.expected_outcome.json)
    for tool in candidates:
        if tool.name not in names and covers_expected_key(tool, expected_keys):
            chosen.append(tool)
            names.add(tool.name)
    floor = min(INITIAL_FLOOR, len(catalog.tools))
    for tool in candidates:
        if len(chosen) >= floor:
            break
        if tool.name not in names:
            chosen.append(tool)
            names.add(tool.name)
    position = {t.name: i for i, t in enumerate(catalog.tools)}
    chosen.sort(key=lambda t: (-score_of[t.name], position[t.name]))
    return chosen
