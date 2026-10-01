"""MCP client wrapper: connect to a server, discover its catalog, call tools.

``connect(server_spec)`` is an async context manager yielding a :class:`Session`; the fakes and
tests use ``connect_streams(read, write)`` over the SDK's in-memory transport. Tool results are
normalised to :class:`ToolResult` so the transcript, the agent and the judge never see SDK types.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from pydantic import BaseModel, ConfigDict, Field

from mcpsim.scenario import ServerSpec

TEXT_LIMIT = 4000


class MCPClientError(RuntimeError):
    """Connection or configuration failure (not a tool error; those are ``ToolResult``s)."""


class ToolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None

    def to_anthropic_tool(self) -> dict[str, Any]:
        """The Anthropic ``tools`` entry; the MCP input schema is already JSON Schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ResourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    uri: str
    description: str = ""
    mime_type: str | None = None


class ResourceTemplateInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    uri_template: str
    description: str = ""
    mime_type: str | None = None


class PromptInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=list)


class Catalog(BaseModel):
    """What the server exposes, discovered live over MCP."""

    model_config = ConfigDict(extra="forbid")

    server_name: str = ""
    tools: list[ToolInfo] = Field(default_factory=list)
    resources: list[ResourceInfo] = Field(default_factory=list)
    resource_templates: list[ResourceTemplateInfo] = Field(default_factory=list)
    prompts: list[PromptInfo] = Field(default_factory=list)

    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    def tool(self, name: str) -> ToolInfo:
        for t in self.tools:
            if t.name == name:
                return t
        raise KeyError(f"tool {name!r} is not in the catalog")

    def has_tool(self, name: str) -> bool:
        return any(t.name == name for t in self.tools)

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_tool() for t in self.tools]

    def digest(self) -> str:
        """sha256 over the canonical JSON of everything that affects planning."""
        payload = self.model_dump(mode="json", exclude={"server_name"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ToolResult(BaseModel):
    """A normalised ``tools/call`` result.

    ``text`` is the joined text content truncated to :data:`TEXT_LIMIT` characters; ``sha256``
    and ``chars`` describe the full text so truncation is always visible.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    is_error: bool
    structured: dict[str, Any] | list[Any] | None = None
    text: str = ""
    ms: float = 0.0
    sha256: str = ""
    chars: int = 0

    @property
    def truncated(self) -> bool:
        return self.chars > len(self.text)

    def payload(self) -> Any:
        """What an agent should see: structured content when present, else the text."""
        return self.structured if self.structured is not None else self.text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_text(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            kind = getattr(block, "type", type(block).__name__)
            parts.append(f"[{kind} content omitted]")
    return "\n".join(parts)


def normalise_result(
    name: str, result: Any, ms: float, *, text_limit: int = TEXT_LIMIT
) -> ToolResult:
    """Turn whatever ``ClientSession.call_tool`` returned into a :class:`ToolResult`."""
    if isinstance(result, mcp_types.CallToolResult):
        full_text = _content_text(list(result.content))
        structured = result.structured_content
        is_error = bool(result.is_error)
    else:
        full_text = (
            f"unexpected result type {type(result).__name__}: "
            f"{json.dumps(result.model_dump(mode='json'), default=str)[:text_limit]}"
        )
        structured = None
        is_error = True
    return ToolResult(
        name=name,
        is_error=is_error,
        structured=structured,
        text=full_text[:text_limit],
        ms=round(ms, 3),
        sha256=_sha256(full_text),
        chars=len(full_text),
    )


class Session:
    """Thin wrapper over :class:`ClientSession` with catalog discovery and normalised calls."""

    def __init__(self, session: ClientSession, *, server_name: str = "") -> None:
        self._session = session
        self.server_name = server_name
        self.tool_calls = 0

    @property
    def raw(self) -> ClientSession:
        return self._session

    async def catalog(self) -> Catalog:
        tools: list[ToolInfo] = []
        cursor: str | None = None
        while True:
            params = mcp_types.PaginatedRequestParams(cursor=cursor) if cursor else None
            page = await self._session.list_tools(params=params)
            for t in page.tools:
                tools.append(
                    ToolInfo(
                        name=t.name,
                        description=t.description or "",
                        input_schema=dict(t.input_schema or {}),
                        output_schema=dict(t.output_schema) if t.output_schema else None,
                    )
                )
            cursor = page.next_cursor
            if not cursor:
                break

        resources: list[ResourceInfo] = []
        templates: list[ResourceTemplateInfo] = []
        prompts: list[PromptInfo] = []
        try:
            cursor = None
            while True:
                params = mcp_types.PaginatedRequestParams(cursor=cursor) if cursor else None
                rpage = await self._session.list_resources(params=params)
                for r in rpage.resources:
                    resources.append(
                        ResourceInfo(
                            name=r.name,
                            uri=str(r.uri),
                            description=r.description or "",
                            mime_type=r.mime_type,
                        )
                    )
                cursor = rpage.next_cursor
                if not cursor:
                    break
            cursor = None
            while True:
                params = mcp_types.PaginatedRequestParams(cursor=cursor) if cursor else None
                tpage = await self._session.list_resource_templates(params=params)
                for rt in tpage.resource_templates:
                    templates.append(
                        ResourceTemplateInfo(
                            name=rt.name,
                            uri_template=rt.uri_template,
                            description=rt.description or "",
                            mime_type=rt.mime_type,
                        )
                    )
                cursor = tpage.next_cursor
                if not cursor:
                    break
        except Exception:  # noqa: BLE001 - a tools-only server may not implement resources
            pass
        try:
            cursor = None
            while True:
                params = mcp_types.PaginatedRequestParams(cursor=cursor) if cursor else None
                ppage = await self._session.list_prompts(params=params)
                for p in ppage.prompts:
                    prompts.append(
                        PromptInfo(
                            name=p.name,
                            description=p.description or "",
                            arguments=[a.model_dump(mode="json") for a in (p.arguments or [])],
                        )
                    )
                cursor = ppage.next_cursor
                if not cursor:
                    break
        except Exception:  # noqa: BLE001 - a tools-only server may not implement prompts
            pass

        return Catalog(
            server_name=self.server_name,
            tools=tools,
            resources=resources,
            resource_templates=templates,
            prompts=prompts,
        )

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Call a tool; server-side errors come back as ``is_error`` results, never exceptions.

        Transport failures (closed session, broken pipe) do raise; the executor turns those
        into an ``error`` outcome.
        """
        self.tool_calls += 1
        started = time.perf_counter()
        result = await self._session.call_tool(name, arguments or {})
        ms = (time.perf_counter() - started) * 1000
        return normalise_result(name, result, ms)


@asynccontextmanager
async def connect_streams(read: Any, write: Any) -> AsyncIterator[Session]:
    """Wrap already-open transport streams (stdio, HTTP or the SDK's in-memory transport)."""
    async with ClientSession(read, write) as session:
        init = await session.initialize()
        server_name = ""
        info = getattr(init, "server_info", None)
        if info is not None:
            server_name = str(getattr(info, "name", "") or "")
        yield Session(session, server_name=server_name)


def stdio_params(spec: ServerSpec) -> StdioServerParameters:
    if spec.stdio is None:
        raise MCPClientError("server spec has no stdio section")
    env = {**get_default_environment(), **spec.stdio.env}
    return StdioServerParameters(command=spec.stdio.command, args=list(spec.stdio.args), env=env)


def http_headers(spec: ServerSpec) -> dict[str, str]:
    if spec.http is None:
        raise MCPClientError("server spec has no http section")
    headers: dict[str, str] = {}
    if spec.http.bearer_env:
        token = os.environ.get(spec.http.bearer_env, "")
        if not token:
            raise MCPClientError(
                f"bearer token environment variable {spec.http.bearer_env} is not set"
            )
        headers["Authorization"] = f"Bearer {token}"
    return headers


@asynccontextmanager
async def connect(spec: ServerSpec) -> AsyncIterator[Session]:
    """Connect per the scenario's ``server`` section and yield a :class:`Session`."""
    if spec.stdio is not None:
        # The subprocess inherits the real stderr (the SDK default); passing ``sys.stderr``
        # breaks under pytest's capture, whose replacement stream has no fileno.
        async with stdio_client(stdio_params(spec)) as (read, write):
            async with connect_streams(read, write) as session:
                yield session
        return
    if spec.http is None:  # pragma: no cover - ServerSpec validation prevents this
        raise MCPClientError("server spec has neither stdio nor http")
    headers = http_headers(spec)
    # mcp 2.x speaks HTTP through the ``httpx2`` package, not ``httpx``; the SDK's factory
    # builds the right client type with MCP's long SSE read timeout and merges our headers.
    async with create_mcp_http_client(headers=headers or None) as http:
        async with streamable_http_client(spec.http.url, http_client=http) as streams:
            read, write = streams[0], streams[1]
            async with connect_streams(read, write) as session:
                yield session
