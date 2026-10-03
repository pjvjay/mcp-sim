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
from fnmatch import fnmatchcase
from typing import Any

from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from pydantic import BaseModel, ConfigDict, Field

from mcpsim.scenario import ServerSpec

TEXT_LIMIT = 4000
# How much of a non-2xx HTTP answer's body a failure message quotes.
HTTP_BODY_LIMIT = 300
# What the SDK's streamable HTTP client says in place of a non-2xx answer that carries no
# JSON-RPC error (it drops the status and the body): the generic fallback, and its two 404s.
SDK_HTTP_FALLBACKS = frozenset(
    {"Server returned an error response", "Session terminated", "Not Found"}
)


class MCPClientError(RuntimeError):
    """Connection or configuration failure (not a tool error; those are ``ToolResult``s)."""


def _leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [exc]


def sole_leaf(exc: BaseException) -> BaseException:
    """The one exception inside ``exc`` when it is an exception group around a single failure
    (the SDK's task groups wrap every failure in one, often twice); ``exc`` itself otherwise."""
    leaves = _leaves(exc)
    return leaves[0] if len(leaves) == 1 else exc


def describe_exception(exc: BaseException) -> str:
    """``Type: message`` of what actually went wrong, exception groups unwrapped to their
    leaves (distinct leaves joined with '; '). An :class:`MCPClientError` is its message alone:
    it is already written for the reader (e.g. the HTTP status a server answered)."""
    parts: list[str] = []
    for leaf in _leaves(exc):
        text = str(leaf).strip()
        if isinstance(leaf, MCPClientError) and text:
            part = text
        else:
            part = f"{type(leaf).__name__}: {text}" if text else type(leaf).__name__
        if part not in parts:
            parts.append(part)
    return "; ".join(parts)


class HttpErrorLog:
    """The non-2xx answers an HTTP MCP endpoint gave to the requests of one connection.

    The SDK turns a non-2xx answer without a JSON-RPC error body into "Server returned an error
    response" and drops the status and the body, so a gateway's rate limit (ContextForge answers
    429 with an "Account locked" body) reads like a broken server. This keeps the last such
    answer so a failure can say what the server actually said.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.seq = 0  # bumped on every recorded answer
        self.status = 0
        self.reason = ""
        self.body = ""

    async def on_response(self, response: Any) -> None:
        """An ``httpx`` response hook; records POSTs (the JSON-RPC requests) answered >= 400."""
        if response.status_code < 400 or response.request.method != "POST":
            return
        try:
            raw = await response.aread()
            body = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        except Exception:  # noqa: BLE001 - the status alone is still worth reporting
            body = ""
        self.status = response.status_code
        self.reason = str(getattr(response, "reason_phrase", "") or "")
        self.body = " ".join(body.split())[:HTTP_BODY_LIMIT]
        self.seq += 1

    def describe(self) -> str:
        head = f"HTTP {self.status}" + (f" {self.reason}" if self.reason else "")
        return f"{head} from {self.url}" + (f": {self.body}" if self.body else "")

    def explains(self, exc: BaseException, since: int) -> bool:
        """Whether ``exc`` is the SDK's stand-in for an answer recorded after mark ``since``."""
        leaf = sole_leaf(exc)
        message = str(getattr(leaf, "message", "") or leaf).strip()
        return self.seq > since and message in SDK_HTTP_FALLBACKS


def _is_refusal(status: int) -> bool:
    """Statuses that mean the server refused or failed, never "method not implemented"."""
    return status in (401, 403, 429) or status >= 500


class ToolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    # The server's MCP tool annotations (``read_only_hint``, ``destructive_hint``, ...) as the
    # SDK dumps them; ``None`` when the server sent none. Scoping reads the write hints.
    annotations: dict[str, Any] | None = None

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

    def select(self, patterns: list[str]) -> list[ToolInfo]:
        """The tools whose name matches any of the ``fnmatch`` globs, in catalog order."""
        return [t for t in self.tools if matches_any(t.name, patterns)]

    def filtered(self, allow: list[str], deny: list[str]) -> Catalog:
        """The *allowed catalog*: tools matching an ``allow`` glob and no ``deny`` glob.

        Resources, templates and prompts are untouched; order is the server's. Globs are
        case-sensitive ``fnmatch`` patterns (``submit_*``, ``*_origin*``, ``*``).
        """
        kept = [
            t
            for t in self.tools
            if matches_any(t.name, allow) and not matches_any(t.name, deny)
        ]
        return self.model_copy(update={"tools": kept})

    def unmatched_patterns(self, patterns: list[str]) -> list[str]:
        """The globs among ``patterns`` that match none of this catalog's tools."""
        names = self.tool_names()
        return [p for p in patterns if not any(fnmatchcase(n, p) for n in names)]

    def digest(self) -> str:
        """sha256 over the canonical JSON of everything that affects planning."""
        payload = self.model_dump(mode="json", exclude={"server_name"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def matches_any(name: str, patterns: list[str]) -> bool:
    """Does ``name`` match at least one glob (``fnmatchcase``: case-sensitive, ``*``/``?``)?"""
    return any(fnmatchcase(name, p) for p in patterns)


class ResourceContent(BaseModel):
    """A normalised ``resources/read`` result: the joined text (binary parts noted, not
    decoded), the mime type the server declared and the full length."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    text: str = ""
    mime_type: str | None = None
    chars: int = 0
    ms: float = 0.0


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

    def __init__(
        self,
        session: ClientSession,
        *,
        server_name: str = "",
        http_errors: HttpErrorLog | None = None,
    ) -> None:
        self._session = session
        self.server_name = server_name
        self.tool_calls = 0
        self._http_errors = http_errors

    @property
    def raw(self) -> ClientSession:
        return self._session

    def _mark(self) -> int:
        return self._http_errors.seq if self._http_errors is not None else 0

    def _http_failure(self, exc: BaseException, since: int) -> MCPClientError | None:
        """An :class:`MCPClientError` naming the HTTP answer behind ``exc``, when the SDK's
        generic message stands in for a non-2xx answer recorded since mark ``since``."""
        log = self._http_errors
        if log is None or not log.explains(exc, since):
            return None
        return MCPClientError(log.describe())

    def _refused(self, exc: BaseException, since: int) -> MCPClientError | None:
        """Like :meth:`_http_failure`, but only for a refusal (401/403/429/5xx): the optional
        listings treat any other failure as "not implemented"."""
        failure = self._http_failure(exc, since)
        log = self._http_errors
        if failure is None or log is None or not _is_refusal(log.status):
            return None
        return failure

    async def catalog(self) -> Catalog:
        tools: list[ToolInfo] = []
        cursor: str | None = None
        while True:
            params = mcp_types.PaginatedRequestParams(cursor=cursor) if cursor else None
            since = self._mark()
            try:
                page = await self._session.list_tools(params=params)
            except Exception as exc:
                failure = self._http_failure(exc, since)
                if failure is not None:
                    raise failure from exc
                raise
            for t in page.tools:
                annotations = getattr(t, "annotations", None)
                tools.append(
                    ToolInfo(
                        name=t.name,
                        description=t.description or "",
                        input_schema=dict(t.input_schema or {}),
                        output_schema=dict(t.output_schema) if t.output_schema else None,
                        annotations=(
                            annotations.model_dump(mode="json", exclude_none=True)
                            if annotations is not None
                            else None
                        ),
                    )
                )
            cursor = page.next_cursor
            if not cursor:
                break

        resources: list[ResourceInfo] = []
        templates: list[ResourceTemplateInfo] = []
        prompts: list[PromptInfo] = []
        since = self._mark()
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
        except Exception as exc:  # noqa: BLE001 - a tools-only server may not implement resources
            refused = self._refused(exc, since)
            if refused is not None:
                raise refused from exc
        since = self._mark()
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
        except Exception as exc:  # noqa: BLE001 - a tools-only server may not implement prompts
            refused = self._refused(exc, since)
            if refused is not None:
                raise refused from exc

        return Catalog(
            server_name=self.server_name,
            tools=tools,
            resources=resources,
            resource_templates=templates,
            prompts=prompts,
        )

    async def read_resource(
        self, uri: str, *, text_limit: int = TEXT_LIMIT * 25
    ) -> ResourceContent:
        """Read a static resource; text parts are joined, blobs noted by size, never decoded.

        ``text`` is truncated to ``text_limit`` characters (``chars`` has the full length).
        Transport failures raise, like :meth:`call_tool`.
        """
        started = time.perf_counter()
        since = self._mark()
        try:
            result = await self._session.read_resource(uri)
        except Exception as exc:
            failure = self._http_failure(exc, since)
            if failure is not None:
                raise failure from exc
            raise
        ms = (time.perf_counter() - started) * 1000
        parts: list[str] = []
        mime_type: str | None = None
        for item in getattr(result, "contents", []) or []:
            if mime_type is None:
                mime_type = getattr(item, "mime_type", None)
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
            else:
                blob = getattr(item, "blob", None)
                size = len(blob) if isinstance(blob, str | bytes) else 0
                parts.append(f"[binary content omitted: {size} bytes base64]")
        full = "\n".join(parts)
        return ResourceContent(
            uri=uri, text=full[:text_limit], mime_type=mime_type, chars=len(full), ms=round(ms, 3)
        )

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Call a tool; server-side errors come back as ``is_error`` results, never exceptions.

        Transport failures (closed session, broken pipe) do raise; the executor turns those
        into an ``error`` outcome. An HTTP server's non-2xx answer (a gateway's rate limit, an
        expired token) raises :class:`MCPClientError` with the status and the body's start.
        """
        self.tool_calls += 1
        started = time.perf_counter()
        since = self._mark()
        try:
            result = await self._session.call_tool(name, arguments or {})
        except Exception as exc:
            failure = self._http_failure(exc, since)
            if failure is not None:
                raise failure from exc
            raise
        ms = (time.perf_counter() - started) * 1000
        return normalise_result(name, result, ms)


@asynccontextmanager
async def connect_streams(
    read: Any, write: Any, *, http_errors: HttpErrorLog | None = None
) -> AsyncIterator[Session]:
    """Wrap already-open transport streams (stdio, HTTP or the SDK's in-memory transport).

    ``http_errors`` is the HTTP transport's log of non-2xx answers, which the session's calls
    use to say what the server answered when the SDK reports only a generic error.
    """
    async with ClientSession(read, write) as session:
        init = await session.initialize()
        server_name = ""
        info = getattr(init, "server_info", None)
        if info is not None:
            server_name = str(getattr(info, "name", "") or "")
        yield Session(session, server_name=server_name, http_errors=http_errors)


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
    http_errors = HttpErrorLog(spec.http.url)
    opened = False
    # mcp 2.x speaks HTTP through the ``httpx2`` package, not ``httpx``; the SDK's factory
    # builds the right client type with MCP's long SSE read timeout and merges our headers.
    async with create_mcp_http_client(headers=headers or None) as http:
        http.event_hooks = {"request": [], "response": [http_errors.on_response]}
        try:
            async with streamable_http_client(spec.http.url, http_client=http) as streams:
                read, write = streams[0], streams[1]
                async with connect_streams(read, write, http_errors=http_errors) as session:
                    opened = True
                    yield session
        except Exception as exc:
            # Only a failure to open is rewritten here; one raised by the caller's own block
            # (after the yield) is the caller's to report and passes through unchanged.
            if opened or not http_errors.explains(exc, 0):
                raise
            raise MCPClientError(http_errors.describe()) from exc
