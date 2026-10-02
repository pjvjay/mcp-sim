"""Observers: the Informant-Report Method (DESIGN §2b).

Never ask the *subject* (the agent under test) to report its own status: its answers are shaped
by alignment training and meta-knowledge. Instead *observers* — informants, each with a social
identity or relationship to the subject (an independent auditor, a clerk, a customer) — watch
the conversation and the tool traffic and report, per declared **condition**, true / false /
unknown with a verbatim quote as evidence. Conditions are declared the way Sierra declares them,
as ``when(...)`` clauses whose **effects** enable a goal and its toolset, flag the run for the
judge, or fail it outright.

The declaration models live in :mod:`mcpsim.scenario` (they are part of the scenario file) and
are re-exported here; this module adds the Python DSL that builds the same models::

    from mcpsim.observers import observer

    auditor = observer(
        "shelf_auditor",
        identity="An independent auditor who trusts only what the store's records say.",
        watches=["tool_traffic", "final_answer"],
    )
    auditor.when("find_product has returned a DIRECT match for penne", id="direct_match").then(
        enable_tools=["get_product"], enable_goal="Quote the cheapest direct hit."
    )
    scenario = load_scenario("x.yaml").with_observers([auditor])

which mirrors the TS-style one-liner ``observer.when("See a chat with the word bear in it")
{ /* enable this goal and its toolset */ }``.
"""

from __future__ import annotations

from typing import Any

from mcpsim.scenario import (
    CHECK_SLICES,
    COUNT_OPERATORS,
    DEFAULT_TRIGGERS,
    DEFAULT_WATCHES,
    TRIGGERS,
    WATCHES,
    Check,
    Condition,
    Effect,
    Observer,
    ObserverKind,
    Trigger,
    Watch,
)
from mcpsim.transcript import EVIDENCE_LIMIT, NO_EVIDENCE, InformantReport, InformantReportEvent

__all__ = [
    "CHECK_SLICES",
    "COUNT_OPERATORS",
    "DEFAULT_TRIGGERS",
    "DEFAULT_WATCHES",
    "EVIDENCE_LIMIT",
    "NO_EVIDENCE",
    "TRIGGERS",
    "WATCHES",
    "Check",
    "Condition",
    "ConditionBuilder",
    "Effect",
    "InformantReport",
    "InformantReportEvent",
    "Observer",
    "ObserverBuilder",
    "ObserverKind",
    "Trigger",
    "Watch",
    "observer",
]


# --- the Python DSL ---------------------------------------------------------------------------


def _effect_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate effect keyword arguments eagerly so a typo fails at the call site."""
    return Effect.model_validate(kwargs).model_dump(mode="python", exclude_defaults=True)


class ConditionBuilder:
    """One ``when(...)`` clause under construction; every method returns ``self`` so effects
    chain (``.then(...).otherwise(...)``) and ``.when(...)`` opens the next clause."""

    def __init__(self, parent: ObserverBuilder, data: dict[str, Any]) -> None:
        self._parent = parent
        self._data = data

    def then(self, **effect: Any) -> ConditionBuilder:
        """The effect applied when the condition is reported true."""
        self._data["then"] = _effect_kwargs(effect)
        return self

    def otherwise(self, **effect: Any) -> ConditionBuilder:
        """The effect applied when the condition is reported false."""
        self._data["otherwise"] = _effect_kwargs(effect)
        return self

    def check(self, **check: Any) -> ConditionBuilder:
        """The code validator (``kind: code``): ``word_count=``, ``regex=``, ``tool_result=`` or
        ``tool_called=``."""
        self._data["check"] = Check.model_validate(check).model_dump(
            mode="python", exclude_none=True
        )
        return self

    def all_of(self, *refs: str) -> ConditionBuilder:
        """``kind: group``: every ``<observer>.<condition>`` (``!`` negates) must be true."""
        self._data["all_of"] = list(refs)
        return self

    def any_of(self, *refs: str) -> ConditionBuilder:
        """``kind: group``: at least one ``<observer>.<condition>`` must be true."""
        self._data["any_of"] = list(refs)
        return self

    def when(self, text: str, *, id: str, **fields: Any) -> ConditionBuilder:  # noqa: A002
        """Open the next condition on the same observer."""
        return self._parent.when(text, id=id, **fields)

    def build(self) -> Observer:
        return self._parent.build()

    @property
    def observer(self) -> ObserverBuilder:
        return self._parent


class ObserverBuilder:
    """An observer under construction; :meth:`build` validates it into an :class:`Observer`.

    ``Scenario.with_observers`` accepts builders directly (it calls ``build``).
    """

    def __init__(
        self,
        name: str,
        *,
        identity: str,
        kind: ObserverKind = "llm",
        watches: list[Watch] | None = None,
        on: list[Trigger] | None = None,
        model: str | None = None,
    ) -> None:
        self._data: dict[str, Any] = {
            "name": name,
            "identity": identity,
            "kind": kind,
            "watches": list(watches) if watches is not None else list(DEFAULT_WATCHES),
            "on": list(on) if on is not None else list(DEFAULT_TRIGGERS),
            "conditions": [],
        }
        if model is not None:
            self._data["model"] = model

    @property
    def name(self) -> str:
        return str(self._data["name"])

    def when(self, text: str, *, id: str, **fields: Any) -> ConditionBuilder:  # noqa: A002
        """Declare a condition: the natural-language conditional and its ``id``.

        Extra keyword arguments are the same fields the YAML takes (``check``, ``all_of``,
        ``any_of``, ``then``, ``otherwise``); the chained methods set them one at a time.
        """
        data: dict[str, Any] = {"id": id, "when": text, **fields}
        self._data["conditions"].append(data)
        return ConditionBuilder(self, data)

    def to_dict(self) -> dict[str, Any]:
        """The declaration as a scenario file would hold it (unvalidated)."""
        return {
            **{k: v for k, v in self._data.items() if k != "conditions"},
            "conditions": [dict(c) for c in self._data["conditions"]],
        }

    def build(self) -> Observer:
        return Observer.model_validate(self.to_dict())


def observer(
    name: str,
    *,
    identity: str,
    kind: ObserverKind = "llm",
    watches: list[Watch] | None = None,
    on: list[Trigger] | None = None,
    model: str | None = None,
) -> ObserverBuilder:
    """Start declaring an observer (see the module docstring for the shape)."""
    return ObserverBuilder(name, identity=identity, kind=kind, watches=watches, on=on, model=model)
