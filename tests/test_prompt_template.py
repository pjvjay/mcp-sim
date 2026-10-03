"""The template language of the role files (mcpsim.prompt_template)."""

from __future__ import annotations

import pytest

from mcpsim.prompt_template import TemplateError, parse, split_parts


def render(source: str, **values: object) -> str:
    return parse(source).render(values)  # type: ignore[arg-type]


def test_placeholders_are_replaced_verbatim_and_never_reparsed() -> None:
    template = "Hello {{ name }}, {{name}}!"
    assert render(template, name="{{ name }} {% if x %}") == (
        "Hello {{ name }} {% if x %}, {{ name }} {% if x %}!"
    )


def test_a_line_holding_only_tags_disappears_with_its_line_break() -> None:
    source = "A\n{% if flag %}\nB\n{% endif %}\nC"
    assert render(source, flag="yes") == "A\nB\nC"
    assert render(source, flag="") == "A\nC"


def test_blocks_keep_paragraph_spacing() -> None:
    source = "Intro\n\n{% if notes %}\n## Notes\n{{ notes }}\n\n{% endif %}\n## Contract\nend"
    assert render(source, notes="n1") == "Intro\n\n## Notes\nn1\n\n## Contract\nend"
    assert render(source, notes="") == "Intro\n\n## Contract\nend"


def test_inline_if_elif_else_and_not() -> None:
    source = "{% if a %}A{% elif b %}B{% elif not c %}not C{% else %}C{% endif %}."
    assert render(source, a="1", b="1", c="") == "A."
    assert render(source, a="", b="1", c="") == "B."
    assert render(source, a="", b="", c="") == "not C."
    assert render(source, a="", b="", c="1") == "C."


def test_blocks_nest() -> None:
    source = "{% if a %}\n{% if b %}\nboth\n{% else %}\nonly a\n{% endif %}\n{% endif %}\nend"
    assert render(source, a="1", b="1") == "both\nend"
    assert render(source, a="1", b="") == "only a\nend"
    assert render(source, a="", b="1") == "end"


def test_booleans_are_conditions_and_render_as_words() -> None:
    assert render("{% if on %}yes{% else %}no{% endif %} {{ on }}", on=True) == "yes true"
    assert render("{% if on %}yes{% else %}no{% endif %} {{ on }}", on=False) == "no false"


def test_auto_numbering_skips_items_a_false_block_drops() -> None:
    source = "Rules:\n#. one\n{% if extra %}\n#. extra\n{% endif %}\n#. last\nAfter\n#. again"
    assert render(source, extra="x") == "Rules:\n1. one\n2. extra\n3. last\nAfter\n1. again"
    assert render(source, extra="") == "Rules:\n1. one\n2. last\nAfter\n1. again"
    # A value that starts with "#. " is data, never numbered.
    assert render("{{ v }}", v="#. not a rule") == "#. not a rule"


def test_comments_are_removed() -> None:
    assert render("A {# inline #}B\n{# whole line #}\nC") == "A B\nC"


def test_the_rendered_prompt_never_starts_or_ends_with_a_line_break() -> None:
    assert render("{% if x %}\nX\n{% endif %}\nbody\n{{ tail }}", x="", tail="") == "body"


def test_names_and_inserted_placeholders() -> None:
    template = parse("{{ a }} {% if b %}{{ c }}{% endif %} {% if not d %}x{% endif %}")
    assert template.names == {"a", "b", "c", "d"}
    assert template.inserted == {"a", "c"}


def test_rendering_without_a_value_names_it() -> None:
    with pytest.raises(KeyError, match="no value for b"):
        parse("{{ a }}{{ b }}", origin="t.md").render({"a": "x"})


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("ok\n{{ name", "line 6: '{{' is never closed"),
        ("{% if a %}\nx", "line 5: '{% if a %}' has no endif"),
        ("x\n{% endif %}", "line 6: 'endif' without an open 'if'"),
        ("{% else %}", "line 5: 'else' without an open 'if'"),
        ("{% if a %}{% else %}{% else %}{% endif %}", "'else' without an open 'if'"),
        ("{% if a %}{% else %}{% elif b %}{% endif %}", "'elif' without an open 'if'"),
        ("{% for x in y %}", "unknown tag '{% for x in y %}'"),
        ("{% if a b %}{% endif %}", "must be 'if NAME' or 'if not NAME'"),
        ("{{ Bad-Name }}", "'{{ Bad-Name }}' is not a placeholder name"),
        ("{% %}", "empty '{% %}' tag"),
        ("{# open", "'{#' is never closed"),
        ("x {% prompt user %}", "must stand alone on its line"),
    ],
)
def test_syntax_errors_name_the_line(source: str, message: str) -> None:
    with pytest.raises(TemplateError) as info:
        parse(source, origin="roles/x.md", first_line=5)
    assert str(info.value).startswith("roles/x.md, line ")
    assert message in str(info.value)


def test_split_parts_by_marker_lines() -> None:
    body = (
        "{# A comment that documents the file\n   and spans lines. #}\n\n"
        "{% prompt system %}\n\nSystem text\n\n{% prompt user %}\nUser {{ x }}\n\n"
    )
    parts = split_parts(body, origin="f.md", first_line=10)
    assert parts == {"system": ("System text", 15), "user": ("User {{ x }}", 18)}


def test_split_parts_refuses_text_before_the_first_marker_and_duplicates() -> None:
    with pytest.raises(TemplateError, match="text before the first"):
        split_parts("stray\n{% prompt system %}\nx", origin="f.md")
    with pytest.raises(TemplateError, match="prompt 'system' is defined twice"):
        split_parts("{% prompt system %}\na\n{% prompt system %}\nb", origin="f.md")
