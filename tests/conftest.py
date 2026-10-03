from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml
from mcp.client._memory import InMemoryTransport
from mcp.server.mcpserver import MCPServer

from mcpsim.mcpclient import Session, connect_streams
from tests.fake_server import build_server

REPO_ROOT = Path(__file__).resolve().parent.parent


def fake_server_stdio_spec() -> dict[str, Any]:
    """A ``server`` section that launches ``tests/fake_server.py`` as a stdio subprocess."""
    return {
        "stdio": {
            "command": sys.executable,
            "args": ["-m", "tests.fake_server"],
            "env": {"PYTHONPATH": str(REPO_ROOT)},
        }
    }


@pytest.fixture
def fake_server() -> MCPServer:
    return build_server()


@asynccontextmanager
async def open_session(server: MCPServer | None = None) -> AsyncIterator[Session]:
    """A :class:`Session` on an in-process fake server over the SDK's in-memory transport.

    This is a context manager rather than an async fixture on purpose: the transport runs an
    anyio task group, and pytest-asyncio tears async fixtures down in a different task, which
    anyio refuses ("Attempted to exit cancel scope in a different task").
    """
    async with InMemoryTransport(server or build_server()) as (read, write):
        async with connect_streams(read, write) as s:
            yield s


@pytest.fixture
def scenario_data() -> dict[str, Any]:
    return {
        "name": "fake-lookup",
        "role": "A shopper who wants the price of penne and will not accept guesses.",
        "goal": "Find the price and store of penne.",
        "instructions": [
            "Use the lookup tool; do not invent prices.",
            "Report origin_status exactly as returned.",
        ],
        "expected_outcome": {
            "text": "The price and store of penne, with its origin_status.",
            "json": {"slug": "penne", "price": {"$gt": 0}, "origin_status": "verified"},
        },
        "server": fake_server_stdio_spec(),
    }


@pytest.fixture
def scenario_path(tmp_path: Path, scenario_data: dict[str, Any]) -> Path:
    path = tmp_path / "fake-lookup.yaml"
    path.write_text(yaml.safe_dump(scenario_data, sort_keys=False), encoding="utf-8")
    return path
