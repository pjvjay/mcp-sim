"""The ``mcpsim ui`` test runner: a Starlette app over scenario files and run directories.

Security model (contract D):

* The server binds 127.0.0.1 by default; :func:`serve` refuses a non-loopback host unless
  ``allow_remote`` is set. Without it, a request whose ``Host`` is not a loopback name is
  refused too, which stops DNS-rebinding pages from reading the API through the browser.
* A random token is generated per process and embedded in the page; every POST must carry it
  in ``X-MCPSim-Token`` (a custom header, so a cross-site form or fetch cannot send it
  without a CORS preflight this server never answers). A POST with a foreign ``Origin`` is
  refused as well.
* Path parameters are looked up in listings (known scenarios, run directories, transcript
  files) and every resolved path must stay inside the runs directory; nothing else on disk is
  served. The page renders all text with ``textContent`` and ships a strict CSP.
"""

from __future__ import annotations

import contextlib
import html
import ipaddress
import json
import os
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path as FsPath
from typing import Any
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from mcpsim import __version__
from mcpsim.plan import MODES
from mcpsim.scenario import MODEL_ROLES
from mcpsim.ui.jobs import JobConflictError, JobManager, parse_run_options
from mcpsim.ui.scenario_view import ScenarioView
from mcpsim.ui.skill_config import (
    KNOWN_MODELS,
    SKILL_ENV,
    SkillConfig,
    default_skill_dir,
    load_skill_config,
    resolve_roles,
)
from mcpsim.ui.store import Store, now_iso

TOKEN_HEADER = "X-MCPSim-Token"
TOKEN_PLACEHOLDER = "__MCPSIM_TOKEN__"
STATIC_DIR = FsPath(__file__).resolve().parent / "static"
STATIC_FILES: dict[str, str] = {
    "app.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
}
MAX_BODY_BYTES = 64 * 1024
MAX_RUN_SCENARIOS = 500
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
SECURITY_HEADERS: dict[str, str] = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "content-security-policy": CSP,
    "cache-control": "no-store",
}


def is_loopback(host: str) -> bool:
    """``127.0.0.1``, ``::1``, ``localhost`` (any loopback address or that name)."""
    name = host.strip().strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def host_name(header: str) -> str:
    """The host part of a ``Host`` header (``[::1]:8765`` -> ``::1``)."""
    header = header.strip()
    if header.startswith("["):
        return header[1 : header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


@dataclass
class UISettings:
    runs_dir: FsPath
    scenario_sources: list[str]
    skill_dir: FsPath | None = None
    skill: SkillConfig = field(default_factory=SkillConfig)
    allow_remote: bool = False
    command: list[str] | None = None
    cwd: FsPath | None = None
    bases: list[FsPath] = field(default_factory=list)


def resolve_settings(
    *,
    skill: str | None = None,
    runs: str | None = None,
    scenarios: list[str] | None = None,
    allow_remote: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: FsPath | None = None,
) -> UISettings:
    """CLI flags over ``config.yaml`` over built-in defaults (``scenarios/``, ``runs/``)."""
    source = os.environ if env is None else env
    base = (cwd or FsPath.cwd()).resolve()
    skill_dir = default_skill_dir(skill, source)
    config = load_skill_config(skill_dir)
    bases = [base] + ([skill_dir.resolve()] if skill_dir is not None else [])
    sources = list(scenarios) if scenarios else list(config.scenarios) or ["scenarios"]
    runs_value = runs or (os.path.expandvars(config.runs_dir) if config.runs_dir else "runs")
    runs_path = FsPath(runs_value).expanduser()
    if not runs_path.is_absolute():
        runs_path = base / runs_path
    return UISettings(
        runs_dir=runs_path,
        scenario_sources=sources,
        skill_dir=skill_dir,
        skill=config,
        allow_remote=allow_remote,
        cwd=base,
        bases=bases,
    )


async def _read_body(request: Request) -> bytes | None:
    """The request body, or ``None`` once it exceeds :data:`MAX_BODY_BYTES`."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        return None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


class SecurityMiddleware:
    """Refuse non-loopback ``Host`` headers (unless remote is allowed); add security headers."""

    def __init__(self, app: Any, *, allow_remote: bool) -> None:
        self.app = app
        self.allow_remote = allow_remote

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if not self.allow_remote:
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
            if not is_loopback(host_name(headers.get("host", ""))):
                response = _error(421, "this server only answers requests for a loopback host")
                await response(scope, receive, send)
                return

        async def send_with_headers(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                existing = {k.lower() for k, _ in message.get("headers", [])}
                extra = [
                    (k.encode(), v.encode())
                    for k, v in SECURITY_HEADERS.items()
                    if k.encode() not in existing
                ]
                message["headers"] = [*message.get("headers", []), *extra]
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RunnerUI:
    """The endpoints; one instance per app, holding the token, the store and the jobs."""

    def __init__(
        self, settings: UISettings, *, token: str | None = None, jobs: JobManager | None = None
    ) -> None:
        self.settings = settings
        self.token = token or secrets.token_urlsafe(32)
        self.store = Store(
            scenario_sources=settings.scenario_sources,
            runs_dir=settings.runs_dir,
            skill=settings.skill,
            bases=settings.bases or None,
        )
        env: dict[str, str] = {}
        if settings.skill_dir is not None and settings.skill_dir.exists():
            env[SKILL_ENV] = str(settings.skill_dir.resolve())
        self.jobs = jobs or JobManager(
            runs_dir=settings.runs_dir.resolve(),
            command=settings.command,
            env=env,
            cwd=settings.cwd,
            status_lookup=self._run_status,
        )
        if self.jobs.status_lookup is None:
            self.jobs.status_lookup = self._run_status
        self._static = {
            name: (STATIC_DIR / name).read_bytes()
            for name in STATIC_FILES
            if (STATIC_DIR / name).is_file()
        }
        page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        self._page = page.replace(TOKEN_PLACEHOLDER, html.escape(self.token, quote=True))

    # --- helpers ----------------------------------------------------------------------------

    def _run_status(self, name: str, run_id: str) -> str | None:
        run_dir = self.store.run_dir(name, run_id)
        if run_dir is None:
            return None
        status = self.store.run_summary(name, run_id, run_dir)["status"]
        return str(status) if status in ("passed", "failed", "partial") else None

    def _in_progress(self, name: str, since: dict[str, str]) -> set[str]:
        return self.store.in_progress(name, since.get(name)) if name in since else set()

    def _known(self, name: str) -> tuple[ScenarioView, FsPath] | None:
        index = self.store.scan()
        view = index.views.get(name)
        return (view, index.paths[name]) if view is not None else None

    def _scenario_status(self, name: str, last: dict[str, Any] | None, active: set[str]) -> str:
        if name in active:
            return "running"
        if last is None:
            return "never"
        status = str(last["status"])
        return status if status in ("passed", "failed", "partial") else "failed"

    def _check_post(self, request: Request) -> JSONResponse | None:
        supplied = request.headers.get(TOKEN_HEADER, "")
        if not supplied or not secrets.compare_digest(supplied, self.token):
            # "bad_token" lets the page tell a stale tab (server restarted) from other refusals.
            return _error(403, f"missing or wrong {TOKEN_HEADER} header", code="bad_token")
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host", ""):
            return _error(403, "cross-origin request refused")
        return None

    # --- pages ------------------------------------------------------------------------------

    def index(self, request: Request) -> Response:
        return HTMLResponse(self._page)

    def static(self, request: Request) -> Response:
        name = request.url.path.lstrip("/")
        body = self._static.get(name)
        if body is None:
            return _error(404, "not found")
        return Response(body, media_type=STATIC_FILES[name])

    # --- read API ---------------------------------------------------------------------------

    def config(self, request: Request) -> Response:
        skill = self.settings.skill
        roles = resolve_roles(skill)
        return JSONResponse(
            {
                "version": __version__,
                "skill": skill.to_json(),
                "roles": [r.to_json() for r in roles.values()],
                "run": skill.run,
                "scenario_sources": self.settings.scenario_sources,
                "runs_dir": str(self.settings.runs_dir),
                "known_models": list(KNOWN_MODELS),
                "model_roles": list(MODEL_ROLES),
                "modes": list(MODES),
                "allow_remote": self.settings.allow_remote,
                "warnings": list(skill.warnings),
            }
        )

    def scenarios(self, request: Request) -> Response:
        index = self.store.scan(fresh=True)
        active = self.jobs.active_scenarios()
        since = self.jobs.running_since()
        rows: list[dict[str, Any]] = []
        categories: dict[str, dict[str, Any]] = {}
        for name, view in sorted(index.views.items(), key=lambda kv: (kv[1].category, kv[1].title)):
            last, count = self.store.last_judged(name, self._in_progress(name, since))
            status = self._scenario_status(name, last, active)
            rows.append(
                {
                    "name": name,
                    "title": view.title,
                    "category": view.category,
                    "file": view.file,
                    "error": view.error,
                    "user_instructions": view.user_instructions,
                    "status": status,
                    "pass_rate": last["pass_rate"] if last else None,
                    "pass_k": last["pass_k"] if last else None,
                    "last_run": last,
                    "run_count": count,
                }
            )
            group = categories.setdefault(
                view.category,
                {
                    "name": view.category,
                    "total": 0,
                    "counts": dict.fromkeys(("passed", "failed", "partial", "running", "never"), 0),
                },
            )
            group["total"] += 1
            group["counts"][status] += 1
        return JSONResponse(
            {
                "scenarios": rows,
                "categories": sorted(categories.values(), key=lambda g: g["name"]),
                "warnings": index.warnings,
                "runs_dir": str(self.settings.runs_dir),
                "generated_at": now_iso(),
            }
        )

    def scenario(self, request: Request) -> Response:
        name = request.path_params["name"]
        known = self._known(name)
        if known is None:
            return _error(404, "unknown scenario")
        view, _ = known
        last, count = self.store.last_judged(
            name, self._in_progress(name, self.jobs.running_since())
        )
        return JSONResponse(
            {
                "scenario": view.to_json(),
                "status": self._scenario_status(name, last, self.jobs.active_scenarios()),
                "last_run": last,
                "run_count": count,
                "models": self.store.effective_models(view),
            }
        )

    def scenario_runs(self, request: Request) -> Response:
        name = request.path_params["name"]
        if self._known(name) is None:
            return _error(404, "unknown scenario")
        running = self._in_progress(name, self.jobs.running_since())
        return JSONResponse({"scenario": name, "runs": self.store.history(name, running)})

    def run(self, request: Request) -> Response:
        name, run_id = request.path_params["name"], request.path_params["run_id"]
        if self._known(name) is None:
            return _error(404, "unknown scenario")
        run_dir = self.store.run_dir(name, run_id)
        if run_dir is None:
            return _error(404, "unknown run")
        return JSONResponse(
            {"scenario": name, "run_id": run_id, **self.store.run_detail(name, run_id, run_dir)}
        )

    def transcript(self, request: Request) -> Response:
        name, run_id = request.path_params["name"], request.path_params["run_id"]
        if self._known(name) is None:
            return _error(404, "unknown scenario")
        run_dir = self.store.run_dir(name, run_id)
        if run_dir is None:
            return _error(404, "unknown run")
        data = self.store.transcript(run_dir, request.path_params["file"])
        if data is None:
            return _error(404, "unknown transcript")
        return JSONResponse({"scenario": name, "run_id": run_id, **data})

    def list_jobs(self, request: Request) -> Response:
        return JSONResponse({"jobs": self.jobs.list(tail=0)})

    def job(self, request: Request) -> Response:
        state = self.jobs.get(request.path_params["job_id"])
        if state is None:
            return _error(404, "unknown job")
        return JSONResponse(state)

    # --- write API --------------------------------------------------------------------------

    async def start_run(self, request: Request) -> Response:
        refused = self._check_post(request)
        if refused is not None:
            return refused
        raw = await _read_body(request)
        if raw is None:
            return _error(413, f"body larger than {MAX_BODY_BYTES} bytes")
        try:
            body = json.loads(raw)
        except ValueError:
            return _error(400, "body must be a JSON object")
        if not isinstance(body, dict):
            return _error(400, "body must be a JSON object")
        unknown = sorted(
            set(body) - {"scenarios", "models", "repeat", "modes", "dry_run", "allow_same_judge"}
        )
        if unknown:
            return _error(400, f"unknown field(s): {', '.join(unknown)}")
        try:
            options = parse_run_options(body)
        except ValueError as exc:
            return _error(400, str(exc))
        index = await run_in_threadpool(self.store.scan)
        requested = body.get("scenarios")
        if requested == "all":
            names = [n for n, v in sorted(index.views.items()) if v.error is None]
        elif (
            isinstance(requested, list)
            and requested
            and len(requested) <= MAX_RUN_SCENARIOS
            and all(isinstance(n, str) for n in requested)
        ):
            names = list(dict.fromkeys(requested))
        else:
            return _error(
                400, f'scenarios must be "all" or a list of 1-{MAX_RUN_SCENARIOS} scenario names'
            )
        missing = [n for n in names if n not in index.views]
        if missing:
            return _error(404, "unknown scenario(s)", scenarios=missing[:20])
        broken = [n for n in names if index.views[n].error is not None]
        if broken:
            return _error(400, "scenario file(s) do not validate", scenarios=broken[:20])
        if not names:
            return _error(400, "no runnable scenarios")
        try:
            job_id = self.jobs.submit([(n, index.paths[n]) for n in names], options)
        except JobConflictError as exc:
            return _error(409, str(exc), scenarios=exc.names)
        return JSONResponse({"job_id": job_id, "scenarios": names}, status_code=202)

    async def cancel_job(self, request: Request) -> Response:
        refused = self._check_post(request)
        if refused is not None:
            return refused
        if not self.jobs.cancel(request.path_params["job_id"]):
            return _error(404, "unknown job")
        return JSONResponse({"job_id": request.path_params["job_id"], "cancel_requested": True})


def create_app(
    settings: UISettings, *, token: str | None = None, jobs: JobManager | None = None
) -> Starlette:
    ui = RunnerUI(settings, token=token, jobs=jobs)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        yield
        ui.jobs.shutdown()

    routes = [
        Route("/", ui.index, methods=["GET"]),
        *(Route(f"/{name}", ui.static, methods=["GET"]) for name in STATIC_FILES),
        Route("/api/config", ui.config, methods=["GET"]),
        Route("/api/scenarios", ui.scenarios, methods=["GET"]),
        Route("/api/scenarios/{name}", ui.scenario, methods=["GET"]),
        Route("/api/scenarios/{name}/runs", ui.scenario_runs, methods=["GET"]),
        Route("/api/runs/{name}/{run_id}", ui.run, methods=["GET"]),
        Route("/api/runs/{name}/{run_id}/transcripts/{file}", ui.transcript, methods=["GET"]),
        Route("/api/run", ui.start_run, methods=["POST"]),
        Route("/api/jobs", ui.list_jobs, methods=["GET"]),
        Route("/api/jobs/{job_id}", ui.job, methods=["GET"]),
        Route("/api/jobs/{job_id}/cancel", ui.cancel_job, methods=["POST"]),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.add_middleware(SecurityMiddleware, allow_remote=settings.allow_remote)
    app.state.ui = ui
    return app


def serve(settings: UISettings, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Run the app with uvicorn until interrupted. Refuses a non-loopback host unless allowed."""
    if not settings.allow_remote and not is_loopback(host):
        raise ValueError(
            f"refusing to bind {host!r}: not a loopback address (pass --allow-remote to expose "
            "the runner, its transcripts and its run button to the network)"
        )
    import uvicorn

    app = create_app(settings)
    shown = f"[{host}]" if ":" in host else host
    print(f"mcpsim ui: http://{shown}:{port}/", flush=True)
    print(f"  scenarios: {', '.join(settings.scenario_sources)}", flush=True)
    print(f"  runs:      {settings.runs_dir}", flush=True)
    if settings.skill_dir is not None:
        print(f"  skill:     {settings.skill_dir}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning", server_header=False)
