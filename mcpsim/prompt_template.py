"""The small template language of the role files (``skills/simulate/roles/*.md``).

A role file's body is split into named prompts by marker lines ``{% prompt NAME %}``; each
prompt is a template:

* ``{{ name }}`` — the value the code computed for ``name`` (a catalog digest, a transcript, a
  list of checkpoints). Values are inserted verbatim and never re-read as template syntax.
* ``{% if name %}`` … ``{% elif other %}`` … ``{% else %}`` … ``{% endif %}`` — a block kept
  only when the value is non-empty (``if not name`` inverts it). Blocks nest.
* ``{# comment #}`` — removed. Inside a prompt it must close on the same line; before the first
  ``{% prompt %}`` marker of a file a comment may span lines (the place to document the file).
* ``#. `` at the very start of a line — an auto-numbered item: ``1. ``, ``2. ``, … The count
  restarts after any line that is not such an item (a line holding only a tag does not count),
  so a rule inside a false ``if`` simply drops out of the numbering.

A line that holds nothing but ``{% … %}`` tags and comments disappears entirely, its line break
included, so block tags can sit on their own lines without leaving blank lines behind. A
rendered prompt never starts or ends with a line break.

:func:`parse` checks the syntax (unclosed or unknown tags, an ``endif`` without its ``if``) and
reports the line; :attr:`Template.names` and :attr:`Template.inserted` let the role loader check
the placeholders against what each prompt accepts and requires.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

Value = str | bool | None

NAME_PATTERN = r"[a-z_][a-z0-9_]*"
_NAME = re.compile(rf"^{NAME_PATTERN}$")
_TOKEN = re.compile(r"\{\{(.*?)\}\}|\{%(.*?)%\}|\{#(.*?)#\}")
_COMMENT_BLOCK = re.compile(r"\{#.*?#\}", re.DOTALL)
_PART = re.compile(rf"^\{{%\s*prompt\s+({NAME_PATTERN})\s*%\}}\s*$")
_OPENERS = ("{{", "{%", "{#")
NUMBERED = "#. "


class TemplateError(ValueError):
    """A template that cannot be parsed; the message names the line."""


@dataclass
class _Text:
    text: str


@dataclass
class _Var:
    name: str
    line: int


@dataclass
class _Number:
    series: int


@dataclass
class _If:
    name: str
    negate: bool
    line: int
    then: list[_Node] = field(default_factory=list)
    otherwise: list[_Node] = field(default_factory=list)


_Node = _Text | _Var | _Number | _If


@dataclass
class Template:
    """A parsed prompt: render it with :meth:`render`."""

    source: str
    nodes: list[_Node]
    origin: str = "<template>"

    @property
    def names(self) -> set[str]:
        """Every name the template reads (in ``{{ }}`` or in a condition)."""
        found: set[str] = set()
        _collect(self.nodes, found, conditions=True)
        return found

    @property
    def inserted(self) -> set[str]:
        """The names the template inserts with ``{{ name }}`` (anywhere, conditional or not)."""
        found: set[str] = set()
        _collect(self.nodes, found, conditions=False)
        return found

    def first_line(self, name: str) -> int | None:
        """The line (1-based, within the source) where ``name`` is first read."""
        return _first_line(self.nodes, name)

    def render(self, values: Mapping[str, Value]) -> str:
        """The prompt for ``values``; every name the template reads must be present."""
        missing = sorted(n for n in self.names if n not in values)
        if missing:
            raise KeyError(f"{self.origin}: no value for {', '.join(missing)}")
        out: list[str] = []
        counters: dict[int, int] = {}
        _render(self.nodes, values, out, counters)
        return "".join(out).strip("\n")


def _collect(nodes: list[_Node], found: set[str], *, conditions: bool) -> None:
    for node in nodes:
        if isinstance(node, _Var):
            found.add(node.name)
        elif isinstance(node, _If):
            if conditions:
                found.add(node.name)
            _collect(node.then, found, conditions=conditions)
            _collect(node.otherwise, found, conditions=conditions)


def _first_line(nodes: list[_Node], name: str) -> int | None:
    for node in nodes:
        if isinstance(node, _Var) and node.name == name:
            return node.line
        if isinstance(node, _If):
            if node.name == name:
                return node.line
            inner = _first_line(node.then, name) or _first_line(node.otherwise, name)
            if inner is not None:
                return inner
    return None


def truthy(value: Value) -> bool:
    """A condition holds for ``True`` and for a non-empty string."""
    if isinstance(value, bool):
        return value
    return bool(value)


def _render(
    nodes: list[_Node], values: Mapping[str, Value], out: list[str], counters: dict[int, int]
) -> None:
    for node in nodes:
        if isinstance(node, _Text):
            out.append(node.text)
        elif isinstance(node, _Var):
            value = values[node.name]
            if isinstance(value, bool):
                out.append("true" if value else "false")
            elif value is not None:
                out.append(value)
        elif isinstance(node, _Number):
            counters[node.series] = counters.get(node.series, 0) + 1
            out.append(f"{counters[node.series]}. ")
        else:
            holds = truthy(values[node.name]) != node.negate
            _render(node.then if holds else node.otherwise, values, out, counters)


# --- parsing ----------------------------------------------------------------------------------


@dataclass
class _Frame:
    node: _If
    in_else: bool = False
    chained: bool = False  # opened by an elif: closed by the same endif as its parent


class _Parser:
    def __init__(self, origin: str) -> None:
        self.origin = origin
        self.root: list[_Node] = []
        self.stack: list[_Frame] = []

    def fail(self, line: int, message: str) -> TemplateError:
        return TemplateError(f"{self.origin}, line {line}: {message}")

    @property
    def target(self) -> list[_Node]:
        if not self.stack:
            return self.root
        frame = self.stack[-1]
        return frame.node.otherwise if frame.in_else else frame.node.then

    def tag(self, body: str, number: int) -> None:
        words = body.split()
        if not words:
            raise self.fail(number, "empty '{% %}' tag")
        keyword = words[0]
        if keyword in ("if", "elif"):
            negate = len(words) == 3 and words[1] == "not"
            name = words[2] if negate else (words[1] if len(words) == 2 else "")
            if not _NAME.match(name):
                raise self.fail(
                    number,
                    f"'{{% {body.strip()} %}}' must be '{keyword} NAME' or '{keyword} not NAME'",
                )
            node = _If(name, negate, number)
            if keyword == "elif":
                if not self.stack or self.stack[-1].in_else:
                    raise self.fail(number, "'elif' without an open 'if'")
                self.stack[-1].in_else = True
                self.target.append(node)
                self.stack.append(_Frame(node, chained=True))
            else:
                self.target.append(node)
                self.stack.append(_Frame(node))
            return
        if keyword == "else" and len(words) == 1:
            if not self.stack or self.stack[-1].in_else:
                raise self.fail(number, "'else' without an open 'if'")
            self.stack[-1].in_else = True
            return
        if keyword == "endif" and len(words) == 1:
            if not self.stack:
                raise self.fail(number, "'endif' without an open 'if'")
            while self.stack and self.stack.pop().chained:
                pass
            return
        if keyword == "prompt":
            raise self.fail(
                number, "'{% prompt NAME %}' must stand alone on its line, outside any if block"
            )
        raise self.fail(number, f"unknown tag '{{% {body.strip()} %}}'")


def parse(source: str, *, origin: str = "<template>", first_line: int = 1) -> Template:
    """Parse one prompt template (see the module docstring); ``first_line`` is the line number
    of the template's first line within its file, for error messages."""
    parser = _Parser(origin)
    series = 0
    numbered_before = False
    lines = source.split("\n")
    for index, raw in enumerate(lines):
        number = first_line + index
        tokens = list(_TOKEN.finditer(raw))
        rest = _TOKEN.sub("", raw)
        for opener in _OPENERS:
            if opener in rest:
                raise parser.fail(number, f"'{opener}' is never closed on this line")
        if tokens and not rest.strip() and all(t.group(1) is None for t in tokens):
            # Only tags and comments: the whole line, its break included, disappears.
            for token in tokens:
                if token.group(2) is not None:
                    parser.tag(token.group(2), number)
            continue
        line = raw
        if line.startswith(NUMBERED):
            if not numbered_before:
                series += 1
            parser.target.append(_Number(series))
            line = line[len(NUMBERED) :]
            numbered_before = True
        else:
            numbered_before = False
        position = 0
        for token in _TOKEN.finditer(line):
            if token.start() > position:
                parser.target.append(_Text(line[position : token.start()]))
            position = token.end()
            if token.group(1) is not None:
                name = token.group(1).strip()
                if not _NAME.match(name):
                    raise parser.fail(
                        number, f"'{{{{{token.group(1)}}}}}' is not a placeholder name"
                    )
                parser.target.append(_Var(name, number))
            elif token.group(2) is not None:
                parser.tag(token.group(2), number)
        if position < len(line):
            parser.target.append(_Text(line[position:]))
        if index < len(lines) - 1:
            parser.target.append(_Text("\n"))
    if parser.stack:
        frame = parser.stack[-1]
        raise parser.fail(frame.node.line, f"'{{% if {frame.node.name} %}}' has no endif")
    return Template(source=source, nodes=_merge(parser.root), origin=origin)


def _merge(nodes: list[_Node]) -> list[_Node]:
    merged: list[_Node] = []
    for node in nodes:
        if isinstance(node, _If):
            node.then = _merge(node.then)
            node.otherwise = _merge(node.otherwise)
        if isinstance(node, _Text) and merged and isinstance(merged[-1], _Text):
            merged[-1] = _Text(merged[-1].text + node.text)
        else:
            merged.append(node)
    return merged


def split_parts(body: str, *, origin: str, first_line: int = 1) -> dict[str, tuple[str, int]]:
    """``{name: (template source, line number of its first line)}`` from a role file's body,
    split at ``{% prompt NAME %}`` marker lines. Blank lines around each prompt are dropped.
    Text before the first marker must be blank (or comments)."""
    parts: dict[str, tuple[str, int]] = {}
    current: str | None = None
    buffer: list[str] = []
    start = first_line

    def close() -> None:
        if current is None:
            leftover = _COMMENT_BLOCK.sub("", "\n".join(buffer))
            if leftover.strip():
                raise TemplateError(
                    f"{origin}, line {start}: text before the first '{{% prompt NAME %}}' marker"
                )
            return
        lines = list(buffer)
        offset = 0
        while lines and not lines[0].strip():
            lines.pop(0)
            offset += 1
        while lines and not lines[-1].strip():
            lines.pop()
        parts[current] = ("\n".join(lines), start + offset)

    for index, line in enumerate(body.split("\n")):
        found = _PART.match(line)
        if found is None:
            buffer.append(line)
            continue
        close()
        name = found.group(1)
        if name in parts:
            raise TemplateError(
                f"{origin}, line {first_line + index}: prompt {name!r} is defined twice"
            )
        current = name
        buffer = []
        start = first_line + index + 1
    close()
    return parts
