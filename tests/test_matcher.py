from __future__ import annotations

from typing import Any

import pytest

from mcpsim.matcher import MISSING, OPERATORS, all_passed, json_equal, json_type, match, match_one

ACTUAL: dict[str, Any] = {
    "recipe_slug": "tomato_penne",
    "origin_status": "verified",
    "total_cost": 12.5,
    "count": 5,
    "flag": True,
    "nothing": None,
    "tags": ["pasta", "dinner"],
    "store": {"name": "Fake Mart", "city": "Vancouver"},
    "lines": [
        {"name": "penne", "origin_country": "Italy", "price": 2.49},
        {"name": "tomato", "origin_country": "Canada", "price": 3.99},
        {"name": "basil", "price": 1.5},
    ],
    "empty": [],
}


def one(path: str, op: str, expected: Any, actual: Any = ACTUAL) -> tuple[bool, str, Any]:
    m = match_one(path, op, expected, actual)
    return m.passed, m.detail, m.actual


# --- every operator: a passing and a failing case -------------------------------------------

OPERATOR_TABLE: list[tuple[str, str, Any, bool]] = [
    ("recipe_slug", "$eq", "tomato_penne", True),
    ("recipe_slug", "$eq", "penne", False),
    ("recipe_slug", "$ne", "penne", True),
    ("recipe_slug", "$ne", "tomato_penne", False),
    ("origin_status", "$in", ["verified", "unverified"], True),
    ("origin_status", "$in", ["unknown"], False),
    ("origin_status", "$nin", ["unknown"], True),
    ("origin_status", "$nin", ["verified"], False),
    ("total_cost", "$gt", 0, True),
    ("total_cost", "$gt", 12.5, False),
    ("total_cost", "$gte", 12.5, True),
    ("total_cost", "$gte", 13, False),
    ("count", "$lt", 6, True),
    ("count", "$lt", 5, False),
    ("count", "$lte", 5, True),
    ("count", "$lte", 4, False),
    ("recipe_slug", "$regex", r"^tomato_", True),
    ("recipe_slug", "$regex", r"^penne", False),
    ("recipe_slug", "$exists", True, True),
    ("recipe_slug", "$exists", False, False),
    ("recipe_slug", "$contains", "penne", True),
    ("recipe_slug", "$contains", "rice", False),
    ("tags", "$contains", "pasta", True),
    ("tags", "$contains", "breakfast", False),
    ("lines", "$len", 3, True),
    ("lines", "$len", 2, False),
    ("lines", "$len", {"$gte": 3}, True),
    ("lines", "$len", {"$gte": 5}, False),
    ("store", "$subset", {"name": "Fake Mart"}, True),
    ("store", "$subset", {"name": "Fake Mart", "country": "CA"}, False),
    ("total_cost", "$type", "number", True),
    ("total_cost", "$type", "string", False),
    ("flag", "$type", "boolean", True),
    ("nothing", "$type", "null", True),
    ("tags", "$type", "array", True),
    ("store", "$type", "object", True),
    ("store", "$type", "array", False),
]


@pytest.mark.parametrize(("path", "op", "expected", "want"), OPERATOR_TABLE)
def test_operator_table(path: str, op: str, expected: Any, want: bool) -> None:
    m = match_one(path, op, expected, ACTUAL)
    assert m.passed is want, m.detail
    assert m.path == path and m.op == op and m.expected == expected


def test_every_operator_is_covered_by_the_table() -> None:
    covered = {op for _, op, _, _ in OPERATOR_TABLE}
    assert covered == set(OPERATORS)
    for op in OPERATORS:
        outcomes = {want for _, o, _, want in OPERATOR_TABLE if o == op}
        assert outcomes == {True, False}, f"{op} needs both a passing and a failing case"


# --- type strictness ---------------------------------------------------------------------------


def test_eq_never_coerces_string_to_number() -> None:
    passed, detail, actual = one("count", "$eq", "5")
    assert passed is False
    assert "type mismatch" in detail and "expected string" in detail and "actual number" in detail
    assert actual == 5


def test_eq_bool_is_not_a_number() -> None:
    assert one("flag", "$eq", 1)[0] is False
    assert one("count", "$eq", True)[0] is False
    assert json_type(True) == "boolean" and json_type(1) == "number"


def test_int_and_float_are_both_numbers() -> None:
    assert json_equal(5, 5.0) is True
    assert one("count", "$eq", 5.0)[0] is True


def test_in_is_type_strict() -> None:
    passed, detail, _ = one("count", "$in", ["5", 6])
    assert passed is False and "strictly" in detail


def test_comparison_across_types_fails_loudly() -> None:
    passed, detail, _ = one("recipe_slug", "$gt", 0)
    assert passed is False and "cannot compare string with number" in detail
    passed, detail, _ = one("flag", "$gt", 0)
    assert passed is False and "boolean" in detail


def test_strings_compare_lexicographically() -> None:
    assert one("recipe_slug", "$gt", "a")[0] is True
    assert one("recipe_slug", "$lt", "a")[0] is False


def test_regex_needs_string_actual_and_valid_pattern() -> None:
    passed, detail, _ = one("count", "$regex", "5")
    assert passed is False and "needs a string" in detail
    passed, detail, _ = one("recipe_slug", "$regex", "(")
    assert passed is False and "invalid regex" in detail


def test_subset_reports_each_problem() -> None:
    passed, detail, _ = one("store", "$subset", {"name": "Other", "country": "CA"})
    assert passed is False
    assert "name: expected 'Other'" in detail and "country: missing" in detail
    assert one("recipe_slug", "$subset", {"a": 1})[1].startswith("$subset needs an object")


def test_len_nested_operators_and_bad_inputs() -> None:
    passed, detail, _ = one("lines", "$len", {"$gte": 2, "$lt": 10})
    assert passed is True and "len=3" in detail
    passed, detail, _ = one("recipe_slug", "$len", 12)
    assert passed is True
    passed, detail, _ = one("count", "$len", 1)
    assert passed is False and "needs a string, array or object" in detail
    passed, detail, _ = one("lines", "$len", {"$regex": "x"})
    assert passed is False and "nested $regex" in detail
    passed, detail, _ = one("lines", "$len", True)
    assert passed is False


def test_type_unknown_name() -> None:
    passed, detail, _ = one("count", "$type", "integer")
    assert passed is False and "unknown type" in detail


def test_contains_on_object_fails() -> None:
    passed, detail, _ = one("store", "$contains", "name")
    assert passed is False and "needs a string or array" in detail


def test_unknown_operator() -> None:
    passed, detail, _ = one("count", "$between", [1, 2])
    assert passed is False and "unknown operator $between" in detail


# --- missing paths -----------------------------------------------------------------------------


@pytest.mark.parametrize("op", [op for op in OPERATORS if op != "$exists"])
def test_missing_path_fails_every_operator_except_exists(op: str) -> None:
    passed, detail, actual = one("no.such.path", op, "x")
    assert passed is False
    assert actual == MISSING
    assert detail == "path missing"


def test_missing_path_exists_false_passes() -> None:
    assert one("no.such.path", "$exists", False)[0] is True
    assert one("no.such.path", "$exists", True)[0] is False


def test_null_value_is_present() -> None:
    assert one("nothing", "$exists", True)[0] is True
    assert one("nothing", "$eq", None)[0] is True


def test_no_final_result_means_everything_missing() -> None:
    results = match({"a": 1, "b": {"$exists": False}}, None)
    assert [m.passed for m in results] == [False, True]
    assert results[0].actual == MISSING


# --- quantifiers and indexes -------------------------------------------------------------------


def test_every_element_passes() -> None:
    passed, detail, actual = one("lines[*].name", "$ne", "rice")
    assert passed is True
    assert actual == ["penne", "tomato", "basil"]
    assert detail == "3/3 elements pass"


def test_every_element_missing_field_fails_even_for_ne() -> None:
    # A line with no origin_country cannot be shown to be non-US: missing fails every operator
    # except $exists (DESIGN §4), and the detail names the offending element.
    passed, detail, actual = one("lines[*].origin_country", "$ne", "United States")
    assert passed is False
    assert actual == ["Italy", "Canada", MISSING]
    assert detail == "2/3 elements pass; failed: [2] path missing"


def test_every_element_fails_names_the_index() -> None:
    passed, detail, _ = one("lines[*].origin_country", "$exists", True)
    assert passed is False
    assert "2/3 elements pass" in detail and "[2]" in detail


def test_every_on_empty_array_is_vacuously_true() -> None:
    passed, detail, actual = one("empty[*].x", "$eq", 1)
    assert passed is True and actual == [] and "vacuously" in detail


def test_any_element() -> None:
    assert one("lines[any].origin_country", "$eq", "Canada")[0] is True
    passed, detail, _ = one("lines[any].origin_country", "$eq", "France")
    assert passed is False and detail == "0/3 elements pass"
    passed, detail, _ = one("empty[any]", "$eq", 1)
    assert passed is False and "no element" in detail


def test_wildcard_on_non_array_is_missing() -> None:
    passed, detail, actual = one("store[*].name", "$eq", "x")
    assert passed is False and actual == MISSING and detail == "path missing"
    assert one("store[*].name", "$exists", False)[0] is True


def test_numeric_index() -> None:
    assert one("lines[0].name", "$eq", "penne")[0] is True
    assert one("lines[1].price", "$gt", 3)[0] is True
    assert one("lines[9].name", "$exists", False)[0] is True
    assert one("tags[1]", "$eq", "dinner")[0] is True


def test_two_wildcards_are_rejected() -> None:
    passed, detail, _ = one("lines[*].x[*]", "$eq", 1)
    assert passed is False and "at most one" in detail


# --- whole-spec behaviour ----------------------------------------------------------------------


def test_match_spec_from_design() -> None:
    spec = {
        "recipe_slug": "tomato_penne",
        "origin_status": {"$in": ["verified", "unverified"]},
        "lines[*].origin_country": {"$ne": "United States"},
        "total_cost": {"$gt": 0},
    }
    complete = dict(ACTUAL)
    complete["lines"] = [dict(line, origin_country="Canada") for line in ACTUAL["lines"]]
    results = match(spec, complete)
    assert [(m.path, m.op, m.passed) for m in results] == [
        ("recipe_slug", "$eq", True),
        ("origin_status", "$in", True),
        ("lines[*].origin_country", "$ne", True),
        ("total_cost", "$gt", True),
    ]
    assert all_passed(results)

    us = dict(complete)
    us["lines"] = complete["lines"] + [{"name": "ketchup", "origin_country": "United States"}]
    results = match(spec, us)
    failed = [m for m in results if not m.passed]
    assert [m.path for m in failed] == ["lines[*].origin_country"]
    assert "[3]" in failed[0].detail


def test_multiple_operators_on_one_path_produce_one_match_each() -> None:
    results = match({"total_cost": {"$gt": 0, "$lt": 10}}, ACTUAL)
    assert [(m.op, m.passed) for m in results] == [("$gt", True), ("$lt", False)]
    assert not all_passed(results)


def test_literal_object_is_deep_equality_not_operators() -> None:
    results = match({"store": {"name": "Fake Mart", "city": "Vancouver"}}, ACTUAL)
    assert results[0].op == "$eq" and results[0].passed is True
    results = match({"store": {"name": "Fake Mart"}}, ACTUAL)
    assert results[0].passed is False  # not a subset: $eq on objects is exact


def test_mixed_operator_and_literal_keys_is_invalid() -> None:
    results = match({"store": {"$subset": {}, "name": "x"}}, ACTUAL)
    assert results[0].passed is False and "invalid spec" in results[0].detail


def test_empty_spec() -> None:
    assert match(None, ACTUAL) == []
    assert match({}, ACTUAL) == []
