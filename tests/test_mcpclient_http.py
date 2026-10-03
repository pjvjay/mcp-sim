"""An HTTP MCP server's refusals (a gateway's rate limit, an auth failure) are named, not hidden.

The SDK's streamable HTTP client turns a non-2xx answer without a JSON-RPC error body into
"Server returned an error response" and drops the status and the body; its task groups then
wrap that in exception groups. A ContextForge rate-limit lockout (HTTP 429, body "Account
locked") used to reach the transcripts and the suite table as "ExceptionGroup: unhandled errors
in a TaskGroup (1 sub-exception)". These tests serve the fake pantry over real streamable HTTP
behind a switch that refuses chosen JSON-RPC methods, the way ContextForge's limiter does.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import uvicorn
import yaml
from mcp.shared.exceptions import MCPError

from mcpsim import runner
from mcpsim.agent import run_path
from mcpsim.mcpclient import MCPClientError, connect, describe_exception, sole_leaf
from mcpsim.plan import Path as PlanPath
from mcpsim.plan import Step
from mcpsim.scenario import ServerSpec, parse_scenario
from tests.fake_server import ITEMS, build_server

# What ContextForge 1.0.11 answers once its limiter has locked a client out.
LOCKED = json.dumps(
    {
        "error": "Account locked",
        "message": "Too many rate limit violations. Account locked for 15 minutes.",
        "lockout_duration_minutes": 15,
        "reset_in_seconds": 900,
    },
    separators=(",", ":"),
)


@dataclass
class Refusals:
    """JSON-RPC method -> (HTTP status, body, content type) to answer instead of the server."""

    rules: dict[str, tuple[int, str, str]] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)


class RefusingProxy:
    """ASGI middleware: answers a POST whose JSON-RPC ``method`` has a rule with that rule."""

    def __init__(self, app: Any, refusals: Refusals) -> None:
        self.app = app
        self.refusals = refusals

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []
        more = True
        while more:
            message = await receive()
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)
        try:
            method = json.loads(body).get("method")
        except (ValueError, AttributeError):
            method = None
        self.refusals.seen.append(str(method))
        rule = self.refusals.rules.get(str(method))
        if rule is not None:
            status, text, content_type = rule
            data = text.encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", content_type.encode()),
                        (b"content-length", str(len(data)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": data})
            return
        replayed = False

        async def replay() -> Any:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


@dataclass
class Served:
    url: str
    refusals: Refusals

    @property
    def spec(self) -> ServerSpec:
        return ServerSpec.model_validate({"http": {"url": self.url}})

    def refuse(self, method: str, status: int = 429, body: str = LOCKED) -> None:
        self.refusals.rules[method] = (status, body, "application/json")


@pytest.fixture
def served() -> Iterator[Served]:
    refusals = Refusals()
    app = RefusingProxy(build_server().streamable_http_app(stateless_http=True), refusals)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "the test server did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield Served(url=f"http://127.0.0.1:{port}/mcp", refusals=refusals)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def locked_message(url: str) -> str:
    return f"HTTP 429 Too Many Requests from {url}: {LOCKED}"


# --- unwrapping ---------------------------------------------------------------------------------


def test_describe_exception_names_the_failure_inside_the_sdk_groups() -> None:
    leaf = MCPError(-32603, "Server returned an error response")
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [leaf])])
    assert sole_leaf(nested) is leaf
    assert describe_exception(nested) == "MCPError: Server returned an error response"
    assert describe_exception(ExceptionGroup("g", [MCPClientError("HTTP 429 from x")])) == (
        "HTTP 429 from x"
    ), "an MCPClientError is already written for the reader"
    two = ExceptionGroup("g", [ValueError("a"), KeyError("b"), ValueError("a")])
    assert sole_leaf(two) is two
    assert describe_exception(two) == "ValueError: a; KeyError: 'b'"
    assert describe_exception(RuntimeError()) == "RuntimeError"


# --- the client ---------------------------------------------------------------------------------


async def test_a_refused_open_says_what_the_server_answered(served: Served) -> None:
    served.refuse("initialize")
    with pytest.raises(MCPClientError) as info:
        async with connect(served.spec):
            pass  # pragma: no cover - the open fails
    assert str(info.value) == locked_message(served.url)
    assert served.refusals.seen == ["initialize"]


async def test_a_refused_tool_call_says_what_the_server_answered(served: Served) -> None:
    async with connect(served.spec) as session:
        catalog = await session.catalog()
        assert catalog.has_tool("lookup")
        served.refuse("tools/call")
        with pytest.raises(MCPClientError) as info:
            await session.call_tool("lookup", {"slug": "penne"})
        assert str(info.value) == locked_message(served.url)
        served.refusals.rules.clear()
        result = await session.call_tool("lookup", {"slug": "penne"})
    assert result.structured == ITEMS["penne"], "the session is still usable after a refusal"


async def test_a_tool_error_is_still_a_result_not_an_http_failure(served: Served) -> None:
    async with connect(served.spec) as session:
        result = await session.call_tool("lookup", {"slug": "nope"})
    assert result.is_error and "unknown slug 'nope'" in result.text


async def test_a_rate_limited_listing_fails_the_catalog_instead_of_dropping_it(
    served: Served,
) -> None:
    """A tools-only server may not implement resources or prompts, so those listings are
    optional; but a 429 is a refusal, and a catalog without the server's resources would be
    a wrong catalog (and a different digest), not a smaller one."""
    served.refuse("resources/list")
    async with connect(served.spec) as session:
        with pytest.raises(MCPClientError) as info:
            await session.catalog()
    assert str(info.value) == locked_message(served.url)


async def test_an_unimplemented_listing_still_leaves_it_empty(served: Served) -> None:
    served.refusals.rules["prompts/list"] = (400, "prompts are not supported", "text/plain")
    async with connect(served.spec) as session:
        catalog = await session.catalog()
    assert catalog.prompts == []
    assert [r.uri for r in catalog.resources] == ["fake://about"]
    assert catalog.has_tool("lookup")


# --- what the runs and the suite record -------------------------------------------------------


def _scenario(url: str) -> dict[str, Any]:
    return {
        "name": "http-lookup",
        "category": "Lookup",
        "role": "A shopper who wants the price of penne.",
        "goal": "Find the price of penne.",
        "instructions": ["Use the lookup tool; do not invent prices."],
        "expected_outcome": {"text": "The price of penne."},
        "server": {"http": {"url": url}},
    }


async def test_a_dry_run_names_the_refused_tool_call(served: Served) -> None:
    scenario = parse_scenario(_scenario(served.url))
    path = PlanPath(
        id="happy",
        kind="happy",
        title="Look up penne",
        steps=[Step(intent="look up penne", tool="lookup", arguments_sketch={"slug": "penne"})],
    )
    async with connect(served.spec) as session:
        served.refuse("tools/call")
        transcript = await run_path(scenario, path, "guided", 0, session, None, dry_run=True)
    assert transcript.outcome == "error"
    assert transcript.reason == f"tool call lookup failed: {locked_message(served.url)}"


async def test_a_run_that_cannot_open_its_session_names_the_answer(served: Served) -> None:
    scenario = parse_scenario(_scenario(served.url))
    path = PlanPath(id="happy", kind="happy", title="h")
    served.refuse("initialize")
    catalog_stub: Any = None
    transcript = await runner._one_run(
        scenario, path, "guided", 0, catalog=catalog_stub, agent_llm=None, dry_run=True
    )
    assert transcript.reason == f"could not open MCP session: {locked_message(served.url)}"


def test_the_suite_row_names_the_answer_not_an_exception_group(
    served: Served, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "http-lookup.yaml").write_text(yaml.safe_dump(_scenario(served.url)), "utf-8")
    served.refuse("initialize")
    code = runner.run_suite(scenarios, tmp_path / "runs", dry_run=True)
    out = capsys.readouterr().out
    assert code == 1
    row = next(line for line in out.splitlines() if line.startswith("http-lookup  "))
    assert row.split()[:3] == ["http-lookup", "Lookup", "-"]
    assert row.endswith(f"error: {locked_message(served.url)}")
    assert "ExceptionGroup" not in out
    # The session the planning phase opened after the refusal is cleared works again.
    served.refusals.rules.clear()
    assert runner.run_suite(scenarios, tmp_path / "runs", dry_run=True) == 0
