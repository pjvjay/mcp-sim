"""Deterministic matcher (DESIGN §4). Pure functions, no LLM.

``spec`` is ``expected_outcome.json``: keys are dotted paths into the agent's ``final_result``
(``lines[*].origin_country``, ``items[any].id``, ``lines[0].price``), values are either a
literal (``$eq``) or a mapping of operators (``{"$gt": 0}``). Comparisons never coerce types:
``"5"`` and ``5`` are different and the :class:`~mcpsim.verdict.Match` says so.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mcpsim.verdict import Match

MISSING = "<missing>"

OPERATORS: tuple[str, ...] = (
    "$eq",
    "$ne",
    "$in",
    "$nin",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$regex",
    "$exists",
    "$contains",
    "$len",
    "$subset",
    "$type",
)
JSON_TYPES: tuple[str, ...] = ("string", "number", "boolean", "array", "object", "null")

_SEGMENT = re.compile(r"\[(\*|any|\d+)\]|([^.\[\]]+)")


def json_type(value: Any) -> str:
    """JSON type name of a Python value (``bool`` is not a number, ``int`` and ``float`` are)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list | tuple):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def json_equal(expected: Any, actual: Any) -> bool:
    """Strict structural equality: same JSON type at every level, no coercion."""
    te, ta = json_type(expected), json_type(actual)
    if te != ta:
        return False
    if te == "array":
        return len(expected) == len(actual) and all(
            json_equal(e, a) for e, a in zip(expected, actual, strict=True)
        )
    if te == "object":
        return set(expected.keys()) == set(actual.keys()) and all(
            json_equal(expected[k], actual[k]) for k in expected
        )
    return bool(expected == actual)


@dataclass(frozen=True)
class _Segment:
    kind: str  # "key" | "index" | "every" | "any"
    key: str = ""
    index: int = 0


def parse_path(path: str) -> list[_Segment]:
    segments: list[_Segment] = []
    for m in _SEGMENT.finditer(path):
        bracket, key = m.group(1), m.group(2)
        if key is not None:
            segments.append(_Segment("key", key=key))
        elif bracket == "*":
            segments.append(_Segment("every"))
        elif bracket == "any":
            segments.append(_Segment("any"))
        else:
            segments.append(_Segment("index", index=int(bracket)))
    return segments


@dataclass(frozen=True)
class _Resolved:
    """Where a path landed: one value, a quantified list of values, or nothing."""

    quantifier: str  # "one" | "every" | "any"
    values: list[Any]  # each is a value or MISSING
    present: bool  # for "one": the leaf exists; for quantified: the array exists

    @property
    def single(self) -> Any:
        return self.values[0] if self.values else MISSING


def _descend(value: Any, segments: list[_Segment]) -> Any:
    """Follow key/index segments only; returns MISSING when the path does not exist."""
    for seg in segments:
        if seg.kind == "key":
            if isinstance(value, Mapping) and seg.key in value:
                value = value[seg.key]
            else:
                return MISSING
        elif seg.kind == "index":
            if isinstance(value, list) and 0 <= seg.index < len(value):
                value = value[seg.index]
            else:
                return MISSING
        else:  # pragma: no cover - guarded by resolve()
            raise ValueError("wildcard inside _descend")
    return value


def resolve(actual: Any, path: str) -> _Resolved:
    segments = parse_path(path)
    wild = [i for i, s in enumerate(segments) if s.kind in ("every", "any")]
    if not wild:
        value = _descend(actual, segments)
        return _Resolved("one", [value], value is not MISSING)
    if len(wild) > 1:
        raise ValueError(f"path {path!r}: at most one [*] or [any] segment is supported")
    i = wild[0]
    head, tail = segments[:i], segments[i + 1 :]
    container = _descend(actual, head)
    if not isinstance(container, list):
        return _Resolved(segments[i].kind, [], False)
    return _Resolved(segments[i].kind, [_descend(elem, tail) for elem in container], True)


def _compare(op: str, expected: Any, actual: Any) -> tuple[bool, str]:
    te, ta = json_type(expected), json_type(actual)
    if te != ta or te not in ("number", "string"):
        return False, f"cannot compare {ta} with {te} using {op}"
    if op == "$gt":
        return bool(actual > expected), ""
    if op == "$gte":
        return bool(actual >= expected), ""
    if op == "$lt":
        return bool(actual < expected), ""
    return bool(actual <= expected), ""


def apply_operator(op: str, expected: Any, actual: Any, *, present: bool) -> tuple[bool, str]:
    """Evaluate one operator on one leaf. Returns ``(passed, detail)``."""
    if op == "$exists":
        if not isinstance(expected, bool):
            return False, "$exists expects true or false"
        return present == expected, "" if present == expected else f"present={present}"
    if not present:
        return False, "path missing"

    if op == "$eq":
        if json_equal(expected, actual):
            return True, ""
        te, ta = json_type(expected), json_type(actual)
        if te != ta:
            return False, f"type mismatch: expected {te}, actual {ta} (no coercion)"
        return False, "not equal"
    if op == "$ne":
        return (not json_equal(expected, actual)), ""
    if op in ("$in", "$nin"):
        if not isinstance(expected, list):
            return False, f"{op} expects a list"
        member = any(json_equal(e, actual) for e in expected)
        if op == "$in":
            return member, "" if member else "not a member (types compared strictly)"
        return (not member), "" if not member else "is a member"
    if op in ("$gt", "$gte", "$lt", "$lte"):
        return _compare(op, expected, actual)
    if op == "$regex":
        if not isinstance(expected, str):
            return False, "$regex expects a pattern string"
        if not isinstance(actual, str):
            return False, f"$regex needs a string, actual is {json_type(actual)}"
        try:
            found = re.search(expected, actual) is not None
        except re.error as exc:
            return False, f"invalid regex: {exc}"
        return found, "" if found else "no match"
    if op == "$contains":
        if isinstance(actual, str):
            if not isinstance(expected, str):
                return False, f"$contains on a string needs a string, got {json_type(expected)}"
            return (expected in actual), ""
        if isinstance(actual, list):
            return any(json_equal(expected, a) for a in actual), ""
        return False, f"$contains needs a string or array, actual is {json_type(actual)}"
    if op == "$len":
        if json_type(actual) not in ("string", "array", "object"):
            return False, f"$len needs a string, array or object, actual is {json_type(actual)}"
        length = len(actual)
        if isinstance(expected, bool):
            return False, "$len expects an integer or nested operators"
        if isinstance(expected, int):
            return length == expected, f"len={length}"
        if isinstance(expected, Mapping):
            details: list[str] = []
            ok = True
            for nested_op, nested_expected in expected.items():
                if nested_op not in ("$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"):
                    return False, f"$len does not support nested {nested_op}"
                passed, detail = apply_operator(nested_op, nested_expected, length, present=True)
                ok = ok and passed
                details.append(f"{nested_op} {nested_expected!r}: {'pass' if passed else 'fail'}")
            return ok, f"len={length}; " + "; ".join(details)
        return False, "$len expects an integer or nested operators"
    if op == "$subset":
        if not isinstance(expected, Mapping):
            return False, "$subset expects an object"
        if not isinstance(actual, Mapping):
            return False, f"$subset needs an object, actual is {json_type(actual)}"
        problems = []
        for key, value in expected.items():
            if key not in actual:
                problems.append(f"{key}: missing")
            elif not json_equal(value, actual[key]):
                problems.append(f"{key}: expected {value!r}, actual {actual[key]!r}")
        return (not problems), "; ".join(problems)
    if op == "$type":
        if expected not in JSON_TYPES:
            return False, f"unknown type {expected!r}; use one of {', '.join(JSON_TYPES)}"
        actual_type = json_type(actual)
        if actual_type == expected:
            return True, ""
        return False, f"actual is {actual_type}"
    return False, f"unknown operator {op}"


def _ops_for(value: Any) -> dict[str, Any]:
    """Turn a spec value into ``{op: expected}``; a literal becomes ``{"$eq": literal}``."""
    if isinstance(value, Mapping) and value:
        keys = list(value.keys())
        dollar = [k for k in keys if isinstance(k, str) and k.startswith("$")]
        if dollar and len(dollar) == len(keys):
            return dict(value)
        if dollar:
            raise ValueError("mixes operators and literal keys")
    return {"$eq": value}


def match_one(path: str, op: str, expected: Any, actual: Any) -> Match:
    """Evaluate one ``(path, op, expected)`` against ``actual``."""
    try:
        resolved = resolve(actual, path)
    except ValueError as exc:
        return Match(
            path=path, op=op, expected=expected, actual=MISSING, passed=False, detail=str(exc)
        )

    if resolved.quantifier == "one":
        passed, detail = apply_operator(op, expected, resolved.single, present=resolved.present)
        return Match(
            path=path,
            op=op,
            expected=expected,
            actual=resolved.single,
            passed=passed,
            detail=detail,
        )

    if not resolved.present:
        passed, detail = apply_operator(op, expected, MISSING, present=False)
        return Match(
            path=path,
            op=op,
            expected=expected,
            actual=MISSING,
            passed=passed,
            detail=detail or "array missing",
        )

    results = [
        apply_operator(op, expected, v, present=v is not MISSING) for v in resolved.values
    ]
    failed = [i for i, (ok, _) in enumerate(results) if not ok]
    n = len(results)
    if resolved.quantifier == "every":
        passed = not failed
        if n == 0:
            detail = "empty array: [*] is vacuously true"
        elif passed:
            detail = f"{n}/{n} elements pass"
        else:
            shown = "; ".join(f"[{i}] {results[i][1] or 'fail'}" for i in failed[:5])
            detail = f"{n - len(failed)}/{n} elements pass; failed: {shown}"
    else:  # any
        passed = len(failed) < n
        if n == 0:
            detail = "empty array: [any] has no element to satisfy"
        elif passed:
            detail = f"{n - len(failed)}/{n} elements pass"
        else:
            detail = f"0/{n} elements pass"
    return Match(
        path=path,
        op=op,
        expected=expected,
        actual=resolved.values,
        passed=passed,
        detail=detail,
    )


def match(spec: Mapping[str, Any] | None, actual: Any) -> list[Match]:
    """Evaluate a whole ``expected_outcome.json`` spec. One :class:`Match` per path × operator."""
    if not spec:
        return []
    matches: list[Match] = []
    for path, value in spec.items():
        try:
            ops = _ops_for(value)
        except ValueError as exc:
            matches.append(
                Match(
                    path=path,
                    op="?",
                    expected=value,
                    actual=MISSING,
                    passed=False,
                    detail=f"invalid spec: {exc}",
                )
            )
            continue
        for op, expected in ops.items():
            matches.append(match_one(path, op, expected, actual))
    return matches


def all_passed(matches: list[Match]) -> bool:
    return all(m.passed for m in matches)
