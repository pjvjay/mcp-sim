"""The one LLM boundary (DESIGN §6, docs/LOCAL_MODELS.md).

Everything that talks to a model goes through the :class:`LLM` protocol so tests can substitute
a scripted fake (``tests/fake_llm.py``). Two real implementations:

* :class:`AnthropicLLM` — the Messages API; retries with exponential backoff on 429/5xx (max 5
  attempts). Nothing outside this module imports ``anthropic``.
* :class:`OllamaLLM` — a local Ollama server over ``POST {host}/api/chat`` with the mapping
  table from docs/LOCAL_MODELS.md; retries on connection errors and 5xx (max 3 attempts); a
  404 for an unknown model fails at once with the ``ollama pull`` hint. Cost is always 0.

A model is addressed as ``provider:model`` (:func:`parse_model_spec`); a bare name means
``anthropic``. :func:`make_llm` builds (and caches) one client per provider; the runner's
``make_llm_for`` picks the provider from ``scenario.models.<role>``. Both clients record usage
per model and estimate cost from :data:`RATE_TABLE` (0 for anything not in the table).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any, Protocol, TypedDict

import anthropic
import httpx
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MAX_TOKENS = 4096

ANTHROPIC = "anthropic"
OLLAMA = "ollama"
PROVIDERS: tuple[str, ...] = (ANTHROPIC, OLLAMA)
DEFAULT_PROVIDER = ANTHROPIC

API_KEY_ENV = "ANTHROPIC_API_KEY"
OLLAMA_HOST_ENV = "OLLAMA_HOST"
OLLAMA_NUM_CTX_ENV = "MCPSIM_OLLAMA_NUM_CTX"
OLLAMA_DEADLINE_ENV = "MCPSIM_OLLAMA_DEADLINE_S"
OLLAMA_KEEP_ALIVE_ENV = "MCPSIM_OLLAMA_KEEP_ALIVE"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_NUM_CTX = 8192
# The longest one local call may take, retries included, and the ceiling for the env override:
# a local answer that needs more than ten minutes means the prompt or the machine is wrong, not
# that the caller should wait longer (docs/LOCAL_MODELS.md "Speed").
MAX_OLLAMA_DEADLINE_S = 600.0
# Keep the model loaded between calls: Ollama's own default (5 minutes) unloads it while the
# hosted agent and judge run, and every scenario then pays the load again with a cold cache.
DEFAULT_OLLAMA_KEEP_ALIVE = "30m"
# Rendered after "Cost is an ..." in report.md, hence the leading noun.
LOCAL_COST_NOTE = "estimate; local model(s) via Ollama cost 0 (no API spend)"

# USD per million tokens (input, output): Anthropic's first-party list prices (as of 2026-09),
# matched by longest prefix (:func:`rate_for`), so ``claude-opus-5-5`` has its own row and a
# dated id such as ``claude-haiku-4-5-20251001`` finds its family. Cache and batch discounts
# are not modelled; these are estimates and the report says so. A model missing here costs 0
# and its budget is not enforced (:func:`unpriced_models` names it so the runner can warn).
RATE_TABLE: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
KNOWN_MODELS: tuple[str, ...] = (
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-fable-5-1",
    "claude-haiku-4-5-20251001",
)


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """A provider-neutral assistant turn: Anthropic-shaped content blocks as plain dicts."""

    model_config = ConfigDict(extra="forbid")

    content: list[dict[str, Any]]
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str = ""

    def text(self) -> str:
        """Concatenated ``text`` blocks."""
        return "\n".join(
            str(block.get("text", "")) for block in self.content if block.get("type") == "text"
        )

    def tool_uses(self) -> list[dict[str, Any]]:
        return [block for block in self.content if block.get("type") == "tool_use"]


class LLM(Protocol):
    """``temperature`` is passed only when a role file sets one (see :func:`sampling`), so an
    implementation that never needs it may leave it out of its signature."""

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
    ) -> LLMResponse: ...


# Anthropic models that answer 400 to a sampling parameter (temperature / top_p / top_k): the
# Claude Opus 4.7+ and 5 family, Sonnet 5 / 5.5 (non-default values), Fable and Mythos.
NO_SAMPLING_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-",
    "claude-mythos-",
)


def rejects_sampling(model: str) -> bool:
    """Does this Anthropic model id reject ``temperature``?"""
    _, name = parse_model_spec(model)
    return name.startswith(NO_SAMPLING_PREFIXES)


class Sampling(TypedDict, total=False):
    """The optional sampling keyword arguments of :meth:`LLM.complete`."""

    temperature: float


def sampling(temperature: float | None) -> Sampling:
    """``{"temperature": t}`` when a role sets one, else nothing: the keyword arguments a call
    site adds to :meth:`LLM.complete`."""
    return Sampling() if temperature is None else Sampling(temperature=float(temperature))


def parse_model_spec(spec: str) -> tuple[str, str]:
    """``provider:model`` -> ``(provider, model)``; a bare name is an Anthropic model.

    Only a known provider counts as a prefix, so ``ollama:llama3.2:3b`` splits once into
    ``("ollama", "llama3.2:3b")`` while ``claude-sonnet-5-5`` is ``("anthropic", ...)``.
    """
    text = spec.strip()
    provider, sep, rest = text.partition(":")
    if sep and provider.strip().lower() in PROVIDERS:
        provider, name = provider.strip().lower(), rest.strip()
    else:
        provider, name = DEFAULT_PROVIDER, text
    if not name:
        raise ValueError(f"model spec {spec!r} has no model name (expected provider:model)")
    return provider, name


def is_local_model(spec: str) -> bool:
    """True when the spec names a provider that runs on this machine (Ollama)."""
    return parse_model_spec(spec)[0] == OLLAMA


def rate_for(model: str) -> tuple[float, float] | None:
    """Longest-prefix lookup in :data:`RATE_TABLE` (so dated ids like ``-20251001`` resolve).

    The provider prefix is stripped first; a local model has no rate and returns ``None``.
    """
    provider, name = parse_model_spec(model)
    if provider != ANTHROPIC:
        return None
    best: tuple[float, float] | None = None
    best_len = -1
    for prefix, rates in RATE_TABLE.items():
        if name.startswith(prefix) and len(prefix) > best_len:
            best, best_len = rates, len(prefix)
    return best


def estimate_cost_usd(model: str, usage: Usage) -> float:
    """Cost estimate in USD; 0.0 for a model missing from the rate table."""
    rates = rate_for(model)
    if rates is None:
        return 0.0
    rate_in, rate_out = rates
    return (usage.input_tokens * rate_in + usage.output_tokens * rate_out) / 1_000_000


def unpriced_models(specs: Iterable[str]) -> list[str]:
    """The Anthropic models among ``specs`` that :data:`RATE_TABLE` has no price for: their
    calls are costed at 0, so the reported cost is short and ``budgets.max_cost_usd`` never
    trips on them. Local models are free by design and are not listed."""
    found: list[str] = []
    for spec in specs:
        provider, name = parse_model_spec(spec)
        if provider == ANTHROPIC and rate_for(spec) is None and name not in found:
            found.append(name)
    return found


def total_cost_usd(usage_by_model: dict[str, Usage]) -> float:
    return round(sum(estimate_cost_usd(m, u) for m, u in usage_by_model.items()), 6)


SleepFn = Callable[[float], Awaitable[None]]


def is_retryable(exc: BaseException) -> bool:
    """429, any 5xx, or a connection-level failure before a status was received."""
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return isinstance(exc, anthropic.APIConnectionError)


FORCED_CHOICE_TYPES = ("tool", "any")


def translate_tool_choice(tool_choice: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Turn a forced tool choice into ``auto`` plus an instruction the model follows.

    Claude Sonnet 5.5, Opus 5.5 and Fable 5.1 answer ``tool_choice`` ``{"type": "tool"}`` and
    ``{"type": "any"}`` with a 400 (``tool_choice: type "tool" and "any" are not supported for
    this model``); the supported pattern is ``auto`` with an explicit instruction naming the tool,
    the caller validating that the call happened. Planner, judge and observers already validate
    and re-ask, so the translation is applied to every Anthropic model (Haiku 4.5 still accepts
    forced choice, but one code path is better than two). ``auto``/``none`` pass through.
    """
    kind = tool_choice.get("type")
    if kind == "tool" and tool_choice.get("name"):
        name = tool_choice["name"]
        return {"type": "auto"}, (
            f"Respond ONLY by calling the tool named `{name}` exactly once, with every "
            "required argument filled in. Do not write any other text."
        )
    if kind == "any":
        return {"type": "auto"}, (
            "Respond ONLY by calling exactly one of the provided tools, with every required "
            "argument filled in. Do not write any other text."
        )
    return tool_choice, None


class AnthropicLLM:
    """Anthropic Messages API behind the :class:`LLM` protocol."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        *,
        api_key: str | None = None,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        sleep: SleepFn = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        # Our own backoff governs; disable the SDK's built-in retries on a client we create.
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._jitter = jitter
        self.usage: dict[str, Usage] = {}
        self.calls = 0
        self.retries = 0
        self.forced_choice_translations = 0

    def _record(self, model: str, usage: Usage) -> None:
        self.usage[model] = self.usage.get(model, Usage()) + usage

    def cost_usd(self) -> float:
        return total_cost_usd(self.usage)

    def _delay(self, attempt: int) -> float:
        # attempt is 1-based: 1s, 2s, 4s, 8s ... capped, plus up to one second of jitter.
        return min(self._base_delay * (2 ** (attempt - 1)), self._max_delay) + self._jitter()

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
    ) -> LLMResponse:
        _, model_id = parse_model_spec(model)
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            choice, note = translate_tool_choice(tool_choice)
            kwargs["tool_choice"] = choice
            if note:
                kwargs["system"] = f"{system.rstrip()}\n\n{note}"
                self.forced_choice_translations += 1

        attempt = 0
        while True:
            attempt += 1
            self.calls += 1
            try:
                message = await self._client.messages.create(**kwargs)
            except Exception as exc:
                if attempt >= self._max_attempts or not is_retryable(exc):
                    raise
                self.retries += 1
                await self._sleep(self._delay(attempt))
                continue
            usage = Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            )
            self._record(model, usage)
            content = [
                block.model_dump(mode="json", exclude_none=True) for block in message.content
            ]
            return LLMResponse(
                content=content,
                stop_reason=message.stop_reason,
                usage=usage,
                model=message.model,
            )


# --- Ollama -------------------------------------------------------------------------------------


class OllamaError(RuntimeError):
    """A user-facing Ollama failure (unknown model, bad request, server down after retries)."""


def ollama_host(env: Mapping[str, str] | None = None) -> str:
    """``OLLAMA_HOST`` (default ``http://localhost:11434``), normalised to an ``http(s)://`` URL."""
    source = os.environ if env is None else env
    host = source.get(OLLAMA_HOST_ENV, "").strip() or DEFAULT_OLLAMA_HOST
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def ollama_num_ctx(env: Mapping[str, str] | None = None) -> int:
    """``MCPSIM_OLLAMA_NUM_CTX`` (default 8192); a bad value raises ``ValueError``."""
    source = os.environ if env is None else env
    raw = source.get(OLLAMA_NUM_CTX_ENV, "").strip()
    if not raw:
        return DEFAULT_OLLAMA_NUM_CTX
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{OLLAMA_NUM_CTX_ENV} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{OLLAMA_NUM_CTX_ENV} must be >= 1, got {value}")
    return value


def ollama_deadline_s(env: Mapping[str, str] | None = None) -> float:
    """``MCPSIM_OLLAMA_DEADLINE_S`` (default and ceiling 600): seconds one call may take in all.

    A value above the ceiling raises rather than being clamped, so a configuration that asks for
    a longer wait is visibly refused instead of silently shortened.
    """
    source = os.environ if env is None else env
    raw = source.get(OLLAMA_DEADLINE_ENV, "").strip()
    if not raw:
        return MAX_OLLAMA_DEADLINE_S
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{OLLAMA_DEADLINE_ENV} must be a number of seconds, got {raw!r}") from exc
    if not 0 < value <= MAX_OLLAMA_DEADLINE_S:
        raise ValueError(
            f"{OLLAMA_DEADLINE_ENV} must be > 0 and <= {MAX_OLLAMA_DEADLINE_S:.0f}, got {raw}"
        )
    return value


def ollama_keep_alive(env: Mapping[str, str] | None = None) -> str:
    """``MCPSIM_OLLAMA_KEEP_ALIVE`` (default ``30m``), passed to Ollama verbatim."""
    source = os.environ if env is None else env
    return source.get(OLLAMA_KEEP_ALIVE_ENV, "").strip() or DEFAULT_OLLAMA_KEEP_ALIVE


def _block_text(content: Any) -> str:
    """The text of a ``content`` that is a string or a list of Anthropic ``text`` blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def to_ollama_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic ``{name, description, input_schema}`` -> Ollama function definitions.

    The MCP input schema is JSON Schema already and is passed through untouched.
    """
    out: list[dict[str, Any]] = []
    for tool in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object"},
                },
            }
        )
    return out


def to_ollama_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-shaped messages -> Ollama chat messages (docs/LOCAL_MODELS.md mapping table).

    * ``text`` blocks become ``user`` / ``assistant`` messages (consecutive text blocks in one
      message are joined).
    * assistant ``tool_use`` blocks become ``tool_calls`` on the assistant message.
    * user ``tool_result`` blocks become ``{role: tool, tool_name, content}``; the tool name is
      recovered from the ``tool_use`` block with the same id earlier in the conversation.
    """
    out: list[dict[str, Any]] = []
    if system.strip():
        out.append({"role": "system", "content": system})
    names_by_id: dict[str, str] = {}
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if isinstance(content, str) or content is None:
            out.append({"role": role, "content": content or ""})
            continue
        if role == "assistant":
            texts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    texts.append(str(block.get("text", "")))
                elif kind == "tool_use":
                    names_by_id[str(block.get("id", ""))] = str(block["name"])
                    tool_calls.append(
                        {
                            "function": {
                                "name": block["name"],
                                "arguments": block.get("input") or {},
                            }
                        }
                    )
            entry: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue
        pending: list[str] = []
        for block in content:
            kind = block.get("type")
            if kind == "tool_result":
                if pending:
                    out.append({"role": role, "content": "\n".join(pending)})
                    pending = []
                tool_use_id = str(block.get("tool_use_id", ""))
                name = names_by_id.get(tool_use_id, tool_use_id)
                text = _block_text(block.get("content"))
                if block.get("is_error"):
                    text = f"ERROR: {text}" if text else "ERROR"
                out.append({"role": "tool", "tool_name": name, "content": text})
            elif kind == "text":
                pending.append(str(block.get("text", "")))
            else:
                pending.append(json.dumps(block, ensure_ascii=False))
        if pending:
            out.append({"role": role, "content": "\n".join(pending)})
    return out


def forced_tool(
    tools: list[dict[str, Any]] | None, tool_choice: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The tool definition named by ``tool_choice = {type: tool, name: X}``; else ``None``."""
    if not tool_choice or tool_choice.get("type") != "tool":
        return None
    name = tool_choice.get("name")
    for tool in tools or []:
        if tool.get("name") == name:
            return tool
    raise ValueError(f"tool_choice names {name!r}, which is not in tools")


def forced_history_as_text(messages: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Earlier answers of a structured-output conversation, replayed as the JSON text they were.

    In a ``format`` call Ollama is sent no tools, so an assistant ``tool_use`` for the forced
    tool would reach the chat template as a tool call it re-renders in its own syntax. The
    model actually wrote compact JSON; replaying exactly that keeps the conversation a strict
    extension of what the server has cached (the next call evaluates only the new turn) and
    shows the model its own words. The matching ``tool_result`` becomes plain user text.
    """
    ids: set[str] = set()
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        blocks: list[dict[str, Any]] = []
        for block in content:
            if block.get("type") == "tool_use" and block.get("name") == name:
                ids.add(str(block.get("id", "")))
                text = json.dumps(block.get("input") or {}, ensure_ascii=False,
                                  separators=(",", ":"))
                blocks.append({"type": "text", "text": text})
            elif block.get("type") == "tool_result" and str(block.get("tool_use_id", "")) in ids:
                text = _block_text(block.get("content"))
                if block.get("is_error"):
                    text = f"ERROR: {text}" if text else "ERROR"
                blocks.append({"type": "text", "text": text})
            else:
                blocks.append(block)
        out.append({**message, "content": blocks})
    return out


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    return {}


def is_retryable_ollama(exc: BaseException) -> bool:
    """Connection-level failures and 5xx responses; never a 4xx, never a read timeout (the model
    is slow, and asking again would spend the rest of the deadline on the same prompt)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    if isinstance(exc, httpx.ReadTimeout):
        return False
    return isinstance(exc, httpx.TransportError)


class OllamaLLM:
    """A local Ollama server behind the :class:`LLM` protocol (``POST {host}/api/chat``)."""

    def __init__(
        self,
        host: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        num_ctx: int | None = None,
        deadline: float | None = None,
        keep_alive: str | None = None,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        sleep: SleepFn = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.host = (host or ollama_host()).rstrip("/")
        self.num_ctx = num_ctx if num_ctx is not None else ollama_num_ctx()
        self.deadline = deadline if deadline is not None else ollama_deadline_s()
        if not 0 < self.deadline <= MAX_OLLAMA_DEADLINE_S:
            raise ValueError(
                f"deadline must be > 0 and <= {MAX_OLLAMA_DEADLINE_S:.0f} s, got {self.deadline}"
            )
        self.keep_alive = keep_alive or ollama_keep_alive()
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.deadline, connect=min(30.0, self.deadline))
        )
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._jitter = jitter
        self._next_call_id = 0
        self.usage: dict[str, Usage] = {}
        self.calls = 0
        self.retries = 0

    def _record(self, model: str, usage: Usage) -> None:
        self.usage[model] = self.usage.get(model, Usage()) + usage

    def cost_usd(self) -> float:
        """Always 0: nothing local is billed (the report says "local")."""
        return 0.0

    def _delay(self, attempt: int) -> float:
        return min(self._base_delay * (2 ** (attempt - 1)), self._max_delay) + self._jitter()

    def _call_id(self) -> str:
        self._next_call_id += 1
        return f"call_{self._next_call_id}"

    def build_request(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | None,
        max_tokens: int,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """The ``/api/chat`` body for one call (pure; tests inspect it). Temperature is 0
        unless a role sets one."""
        forced = forced_tool(tools, tool_choice)
        if forced is not None:
            hint = (
                f"Answer with exactly one JSON object matching the `{forced['name']}` schema, "
                "written compactly on one line; no prose before or after it."
            )
            description = str(forced.get("description", "")).strip()
            if description:
                hint = f"{hint}\n{description}"
            system = f"{system.rstrip()}\n\n{hint}" if system.strip() else hint
        if forced is not None:
            messages = forced_history_as_text(messages, forced["name"])
        body: dict[str, Any] = {
            "model": model,
            "messages": to_ollama_messages(system, messages),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0 if temperature is None else temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        if forced is not None:
            body["format"] = forced.get("input_schema") or {"type": "object"}
        elif tools:
            body["tools"] = to_ollama_tools(tools)
        return body

    def parse_response(
        self, data: dict[str, Any], *, model: str, forced: dict[str, Any] | None
    ) -> LLMResponse:
        """``/api/chat`` JSON -> provider-neutral :class:`LLMResponse`."""
        message = data.get("message") or {}
        text = str(message.get("content") or "")
        raw_calls = message.get("tool_calls") or []
        done_reason = data.get("done_reason")
        content: list[dict[str, Any]] = []
        truncated = done_reason == "length"
        if forced is not None:
            payload: dict[str, Any] | None = None
            if text.strip() and not truncated:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    payload = parsed
            if payload is not None:
                content.append(
                    {
                        "type": "tool_use",
                        "id": self._call_id(),
                        "name": forced["name"],
                        "input": payload,
                    }
                )
            elif text:
                content.append({"type": "text", "text": text})
        else:
            if text:
                content.append({"type": "text", "text": text})
            for call in raw_calls:
                function = call.get("function") or {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": self._call_id(),
                        "name": str(function.get("name", "")),
                        "input": _parse_arguments(function.get("arguments")),
                    }
                )
        has_tool_use = any(block["type"] == "tool_use" for block in content)
        if truncated:
            stop_reason = "max_tokens"
        elif has_tool_use:
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"
        usage = Usage(
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
        )
        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            usage=usage,
            model=str(data.get("model") or model),
        )

    def _error_for(self, model: str, response: httpx.Response) -> OllamaError | None:
        """A user-facing error for a 4xx; ``None`` for anything retryable (5xx)."""
        status = response.status_code
        if status >= 500:
            return None
        try:
            detail = str(response.json().get("error", "")).strip()
        except (ValueError, AttributeError):
            detail = response.text.strip()
        if status == 404:
            return OllamaError(
                f"Ollama at {self.host} has no model {model!r}"
                + (f" ({detail})" if detail else "")
                + f"; run `ollama pull {model}` (or `ollama list` to see what is installed)"
            )
        return OllamaError(f"Ollama at {self.host} rejected the request ({status}): {detail}")

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
    ) -> LLMResponse:
        _, model_id = parse_model_spec(model)
        forced = forced_tool(tools, tool_choice)
        body = self.build_request(
            model=model_id,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            async with asyncio.timeout(self.deadline):
                return await self._post_with_retries(model, model_id, body, forced)
        except TimeoutError as exc:
            prompt_chars = sum(len(str(m.get("content") or "")) for m in body["messages"])
            raise OllamaError(
                f"{model_id} did not answer within {self.deadline:g} s "
                f"({prompt_chars:,} prompt characters) and the call was cancelled; shrink the "
                "prompt or free the machine (docs/LOCAL_MODELS.md \"Speed\"). "
                f"{OLLAMA_DEADLINE_ENV} can only lower this limit."
            ) from exc

    async def _post_with_retries(
        self,
        model: str,
        model_id: str,
        body: dict[str, Any],
        forced: dict[str, Any] | None,
    ) -> LLMResponse:
        url = f"{self.host}/api/chat"
        attempt = 0
        while True:
            attempt += 1
            self.calls += 1
            try:
                response = await self._client.post(url, json=body)
                if response.status_code >= 400:
                    error = self._error_for(model_id, response)
                    if error is not None:
                        raise error
                    response.raise_for_status()
            except OllamaError:
                raise
            except httpx.ReadTimeout as exc:  # the deadline, reached by the socket first
                raise TimeoutError from exc
            except Exception as exc:
                if attempt >= self._max_attempts or not is_retryable_ollama(exc):
                    if isinstance(exc, httpx.TransportError):
                        raise OllamaError(
                            f"cannot reach Ollama at {self.host} ({type(exc).__name__}: {exc}); "
                            f"is `ollama serve` running? ({OLLAMA_HOST_ENV} selects the server)"
                        ) from exc
                    if isinstance(exc, httpx.HTTPStatusError):
                        raise OllamaError(
                            f"Ollama at {self.host} failed with "
                            f"{exc.response.status_code} after {attempt} attempt(s): "
                            f"{exc.response.text.strip()[:300]}"
                        ) from exc
                    raise
                self.retries += 1
                await self._sleep(self._delay(attempt))
                continue
            data = response.json()
            if not isinstance(data, dict):
                raise OllamaError(f"Ollama at {self.host} returned a non-object reply: {data!r}")
            result = self.parse_response(data, model=model_id, forced=forced)
            self._record(model, result.usage)
            return result


# --- provider factory ---------------------------------------------------------------------------


class RoutingLLM:
    """Dispatches each call to the provider named by its ``model`` spec.

    Used when one component talks to two providers at once (the agent under test and the
    simulated user share one client in ``agent.run_path``).
    """

    def __init__(self, factory: Callable[[str], LLM] | None = None) -> None:
        self._factory = factory or make_llm

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
    ) -> LLMResponse:
        provider, _ = parse_model_spec(model)
        return await self._factory(provider).complete(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            **sampling(temperature),
        )


_CLIENTS: dict[str, LLM] = {}


def clear_llm_cache() -> None:
    """Forget cached provider clients (tests; or after changing ``OLLAMA_HOST``)."""
    _CLIENTS.clear()


def missing_api_key_message(purpose: str | None = None) -> str:
    return (
        f"{API_KEY_ENV} is not set; export it for a real run, use --dry-run (MCPSIM_DRY_RUN=1) "
        "for the no-LLM smoke mode, or address the model as ollama:<model> to run locally"
        + (f" (needed for the {purpose})" if purpose else "")
    )


def make_llm(provider: str, *, purpose: str | None = None) -> LLM:
    """One cached client per provider: ``anthropic`` -> :class:`AnthropicLLM`, ``ollama`` ->
    :class:`OllamaLLM`. ``purpose`` names the role for the error message.

    Raises ``RuntimeError`` when ``anthropic`` is asked for without ``ANTHROPIC_API_KEY`` (the
    key is checked on every call so a cache cannot hide a missing key) and ``ValueError`` for an
    unknown provider.
    """
    key = provider.strip().lower()
    if key == ANTHROPIC:
        if not os.environ.get(API_KEY_ENV):
            raise RuntimeError(missing_api_key_message(purpose))
    elif key != OLLAMA:
        raise ValueError(
            f"unknown model provider {provider!r}; use one of "
            + ", ".join(f"{p}:<model>" for p in PROVIDERS)
        )
    cached = _CLIENTS.get(key)
    if cached is None:
        cached = AnthropicLLM() if key == ANTHROPIC else OllamaLLM()
        _CLIENTS[key] = cached
    return cached
