from __future__ import annotations

import hashlib

import pytest
from mcp import types as mcp_types

from mcpsim.mcpclient import TEXT_LIMIT, ToolResult, normalise_result
from tests.conftest import open_session
from tests.fake_server import SERVER_NAME, TOOL_NAMES, build_server


async def test_catalog_lists_everything() -> None:
    async with open_session() as session:
        catalog = await session.catalog()
    assert catalog.server_name == SERVER_NAME
    assert sorted(catalog.tool_names()) == sorted(TOOL_NAMES)
    lookup = catalog.tool("lookup")
    assert lookup.description.startswith("Look up a product by slug")
    assert lookup.input_schema["type"] == "object"
    assert lookup.input_schema["required"] == ["slug"]
    assert lookup.input_schema["properties"]["slug"]["type"] == "string"
    assert lookup.output_schema is not None and lookup.output_schema["type"] == "object"
    echo = catalog.tool("echo")
    assert echo.output_schema is None
    assert [r.uri for r in catalog.resources] == ["fake://about"]
    assert [t.uri_template for t in catalog.resource_templates] == ["fake://item/{slug}"]
    assert [p.name for p in catalog.prompts] == ["shopping_prompt"]
    assert catalog.prompts[0].arguments[0]["name"] == "goal"
    assert catalog.has_tool("lookup") and not catalog.has_tool("nope")
    with pytest.raises(KeyError):
        catalog.tool("nope")


async def test_anthropic_tool_shape() -> None:
    async with open_session() as session:
        catalog = await session.catalog()
    tools = catalog.to_anthropic_tools()
    assert {t["name"] for t in tools} == set(TOOL_NAMES)
    for t in tools:
        assert set(t) == {"name", "description", "input_schema"}
        assert t["input_schema"]["type"] == "object"


async def test_catalog_digest_is_stable_and_changes_with_the_catalog() -> None:
    async with open_session() as session:
        first = await session.catalog()
        second = await session.catalog()
    assert first.digest() == second.digest()
    assert len(first.digest()) == 64

    bigger = build_server()

    @bigger.tool()
    def extra(x: int) -> dict[str, int]:
        """An extra tool."""
        return {"x": x}

    async with open_session(bigger) as other:
        changed = await other.catalog()
    assert "extra" in changed.tool_names()
    assert changed.digest() != first.digest()


async def test_call_tool_structured_result() -> None:
    async with open_session() as session:
        result = await session.call_tool("lookup", {"slug": "penne"})
        assert session.tool_calls == 1
    assert isinstance(result, ToolResult)
    assert result.name == "lookup"
    assert result.is_error is False
    assert result.structured == {
        "slug": "penne",
        "price": 2.49,
        "store": "Fake Mart",
        "origin_status": "verified",
    }
    assert '"slug": "penne"' in result.text
    assert result.ms >= 0
    assert result.chars == len(result.text) and not result.truncated
    assert result.sha256 == hashlib.sha256(result.text.encode()).hexdigest()
    assert result.payload() == result.structured


async def test_call_tool_error_result() -> None:
    async with open_session() as session:
        result = await session.call_tool("fail", {"reason": "bad input"})
    assert result.is_error is True
    assert result.structured is None
    assert "fail tool invoked: bad input" in result.text
    assert result.payload() == result.text


async def test_server_side_validation_error_is_an_error_result() -> None:
    async with open_session() as session:
        result = await session.call_tool("lookup", {"slug": 42})
    assert result.is_error is True
    assert "slug" in result.text


async def test_unknown_tool_is_an_error_result_not_an_exception() -> None:
    async with open_session() as session:
        result = await session.call_tool("no_such_tool", {})
    assert result.is_error is True
    assert "no_such_tool" in result.text


async def test_tool_error_message_reaches_the_client() -> None:
    async with open_session() as session:
        result = await session.call_tool("lookup", {"slug": "caviar"})
    assert result.is_error is True
    assert "unknown slug 'caviar'" in result.text and "penne" in result.text


async def test_text_only_result() -> None:
    async with open_session() as session:
        result = await session.call_tool("echo", {"text": "HI"})
    assert result.is_error is False
    assert result.structured is None
    assert result.text == "HI"
    assert result.payload() == "HI"


async def test_text_is_truncated_with_full_digest() -> None:
    long_text = "x" * (TEXT_LIMIT + 500) + "END"
    async with open_session() as session:
        result = await session.call_tool("echo", {"text": long_text})
    assert result.chars == len(long_text)
    assert len(result.text) == TEXT_LIMIT
    assert result.truncated is True
    assert result.sha256 == hashlib.sha256(long_text.encode()).hexdigest()
    assert not result.text.endswith("END")


async def test_pagination_tool_round_trip() -> None:
    seen: list[str] = []
    cursor: int | None = 0
    pages = 0
    async with open_session() as session:
        while cursor is not None:
            result = await session.call_tool("list_items", {"cursor": cursor, "limit": 2})
            assert result.is_error is False and isinstance(result.structured, dict)
            seen.extend(item["slug"] for item in result.structured["items"])
            cursor = result.structured["next_cursor"]
            pages += 1
    assert pages == 3 and len(seen) == 5 and seen == sorted(seen)


def test_normalise_unexpected_result_type() -> None:
    result = normalise_result("x", mcp_types.Result(), 1.5)
    assert result.is_error is True
    assert "unexpected result type Result" in result.text
    assert result.ms == 1.5


def test_normalise_non_text_content_is_labelled() -> None:
    raw = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type="text", text="hello"),
            mcp_types.ImageContent(type="image", data="AAAA", mime_type="image/png"),
        ],
        structured_content=None,
        is_error=False,
    )
    result = normalise_result("pic", raw, 0.0)
    assert result.text == "hello\n[image content omitted]"


async def test_second_server_instance_is_independent() -> None:
    async with open_session(build_server("other")) as s:
        assert s.server_name == "other"
        result = await s.call_tool("lookup", {"slug": "basil"})
    assert result.structured == {
        "slug": "basil",
        "price": 1.5,
        "store": "Corner Shop",
        "origin_status": "verified",
    }
