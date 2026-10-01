"""An in-process MCP server with toy tools for the framework's own tests (DESIGN §8).

Tools: one returning structured content (``lookup``), one that always errors (``fail``), one that
paginates (``list_items``), one text-only (``echo``), and one whose description says it costs
credits (``expensive_report``) so dry-run planning has something to skip.

Run as a module (``python -m tests.fake_server``) it serves over stdio, which the CLI tests use.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SERVER_NAME = "fake-pantry"

ITEMS: dict[str, dict[str, Any]] = {
    "penne": {"slug": "penne", "price": 2.49, "store": "Fake Mart", "origin_status": "verified"},
    "tomato": {
        "slug": "tomato",
        "price": 3.99,
        "store": "Fake Mart",
        "origin_status": "unverified",
    },
    "basil": {"slug": "basil", "price": 1.5, "store": "Corner Shop", "origin_status": "verified"},
    "garlic": {"slug": "garlic", "price": 0.8, "store": "Corner Shop", "origin_status": "verified"},
    "olive_oil": {
        "slug": "olive_oil",
        "price": 9.0,
        "store": "Fake Mart",
        "origin_status": "unverified",
    },
}

TOOL_NAMES: tuple[str, ...] = ("lookup", "fail", "list_items", "echo", "expensive_report")
# Tools a dry-run planner should call: every tool whose description does not mention cost.
FREE_TOOL_NAMES: tuple[str, ...] = ("lookup", "fail", "list_items", "echo")
EXPENSIVE_TOOL_NAMES: tuple[str, ...] = ("expensive_report",)


def build_server(name: str = SERVER_NAME) -> MCPServer:
    server = MCPServer(name)

    @server.tool()
    def lookup(slug: str) -> dict[str, Any]:
        """Look up a product by slug. Returns price, store and origin_status."""
        item = ITEMS.get(slug)
        if item is None:
            raise ToolError(f"unknown slug {slug!r}; try one of: {', '.join(sorted(ITEMS))}")
        return dict(item)

    @server.tool()
    def fail(reason: str = "") -> dict[str, Any]:
        """Always fails with the given reason."""
        raise ToolError(f"fail tool invoked: {reason or 'no reason given'}")

    @server.tool()
    def list_items(cursor: int = 0, limit: int = 2) -> dict[str, Any]:
        """List products a page at a time. Follow next_cursor until it is null."""
        slugs = sorted(ITEMS)
        page = slugs[cursor : cursor + limit]
        next_cursor = cursor + limit if cursor + limit < len(slugs) else None
        return {"items": [ITEMS[s] for s in page], "next_cursor": next_cursor, "total": len(slugs)}

    @server.tool(structured_output=False)
    def echo(text: str) -> str:
        """Return the text unchanged, as plain text content (no structured content)."""
        return text

    @server.tool()
    def expensive_report(topic: str) -> dict[str, Any]:
        """SLOW: generates a report with an LLM; costs credits. Avoid unless asked."""
        return {"topic": topic, "report": "lorem ipsum"}

    @server.resource("fake://about")
    def about() -> str:
        """What this server is."""
        return "A fake pantry server for mcp-sim tests."

    @server.resource("fake://item/{slug}")
    def item(slug: str) -> str:
        """One product as text."""
        return str(ITEMS.get(slug, {}))

    @server.prompt()
    def shopping_prompt(goal: str) -> str:
        """Frame a shopping goal for an agent."""
        return f"You are shopping. Goal: {goal}"

    return server


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the CLI tests
    build_server().run(transport="stdio")
