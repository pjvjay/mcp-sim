"""The ``mcpsim ui`` test runner: a Starlette app over scenario files and run directories.

Security model (contract D):

* The server binds 127.0.0.1 by default; :func:`serve` refuses a non-loopback host unless
  ``allow_remote`` is set. Every request's ``Host`` is checked too, which stops DNS-rebinding
  pages (a site whose name the attacker points at this machine) from loading the page, its
  token or the API through the browser: without ``allow_remote`` only loopback names pass;
  with it, also IP addresses (a rebinding page always uses a name), this machine's host names,
  the bound name and every ``--allow-host`` name (:func:`host_allowed`). Anything else is 421.
* A random token is generated per process and embedded in the page; every POST must carry it
  in ``X-MCPSim-Token`` (a custom header, so a cross-site form or fetch cannot send it
  without a CORS preflight this server never answers). A POST with a foreign ``Origin`` is
  refused as well.
* Path parameters are looked up in listings (known scenarios, run directories, transcript
  files) and every resolved path must stay inside the runs directory; nothing else on disk is
  served. The page renders all text with ``textContent`` and ships a strict CSP.

What the page shows comes from the same code the CLI runs: the simulate skill through
:func:`mcpsim.skill.load_skill` (``--skill``, else ``MCPSIM_SKILL``, else the packaged copy), the
scenario files through the scenario model, and runs through ``mcpsim run --skill``.
"""

from __future__ import annotations

import contextlib
import html
import ipaddress
import json
import secrets
import socket
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping, Sequence
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
from mcpsim.judge import HONESTY_ITEM
from mcpsim.llm import KNOWN_MODELS
from mcpsim.plan import MODES
from mcpsim.scenario import MODEL_ROLES
from mcpsim.skill import SKILL_ENV, Skill, SkillError, load_skill, skill_dir
from mcpsim.ui.jobs import JobConflictError, JobManager, parse_run_options
from mcpsim.ui.scenario_view import ScenarioView
from mcpsim.ui.skill_view import SkillSource, overrides_json, role_rows, scenario_settings
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


def is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip().strip("[]"))
    except ValueError:
        return False
    return True


def remote_host_names(bind_host: str | None = None, extra: Sequence[str] = ()) -> frozenset[str]:
    """The host names a remote request may use under ``--allow-remote``: this machine's name
    (as ``gethostname`` reports it, and its first label), the bound name when it is one, and
    ``extra`` (``--allow-host``). No DNS lookup: ``getfqdn`` can stall for half a minute, so a
    fully qualified name the hostname does not show needs ``--allow-host``."""
    names = {socket.gethostname(), *extra}
    if bind_host and not is_ip_literal(bind_host):
        names.add(bind_host)
    out: set[str] = set()
    for name in names:
        name = name.strip().lower().rstrip(".")
        if name:
            out.add(name)
            out.add(name.split(".", 1)[0])
    return frozenset(out)


def host_allowed(name: str, *, allow_remote: bool, names: frozenset[str] = frozenset()) -> bool:
    """May a request with this ``Host`` name be served? Loopback always; with
    ``allow_remote``, also any IP address and the ``names`` (:func:`remote_host_names`).
    A DNS-rebinding page sends its own domain name, which is none of these."""
    if is_loopback(name):
        return True
    if not allow_remote:
        return False
    return is_ip_literal(name) or name.strip().lower().rstrip(".") in names


def host_name(header: str) -> str:
    """The host part of a ``Host`` header (``[::1]:8765`` -> ``::1``)."""
    header = header.strip()
    if header.startswith("["):
        return header[1 : header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


@dataclass
class UISettings:
    """``skill`` is the simulate skill as loaded at start; ``scenario_sources`` is ``None`` to
    follow its ``config.yaml`` (re-read on every scan) or the ``--scenarios`` entries."""

    skill: Skill
    runs_dir: FsPath
    scenario_sources: list[str] | None = None
    allow_remote: bool = False
    command: list[str] | None = None
    cwd: FsPath = field(default_factory=FsPath.cwd)
    # Under allow_remote: extra Host names to answer (--allow-host) and the bound --host.
    allow_hosts: list[str] = field(default_factory=list)
    bind_host: str | None = None

    @property
    def skill_dir(self) -> FsPath:
        return self.skill.path


def resolve_settings(
    *,
    skill: str | None = None,
    runs: str | None = None,
    scenarios: list[str] | None = None,
    allow_remote: bool = False,
    cwd: FsPath | None = None,
    allow_hosts: list[str] | None = None,
    bind_host: str | None = None,
) -> UISettings:
    """The skill (``--skill``, else ``$MCPSIM_SKILL``, else the packaged copy), loaded and
    validated as every mcpsim command loads it; the runs directory (``--runs``, else the
    config's ``runs_dir``) and the scenario sources (``--scenarios``, else the config's), both
    relative to the working directory. Raises :class:`~mcpsim.skill.SkillError`."""
    base = (cwd or FsPath.cwd()).resolve()
    loaded = load_skill(skill_dir(skill))
    if runs:
        runs_path = FsPath(runs).expanduser()
        runs_path = runs_path if runs_path.is_absolute() else base / runs_path
    else:
        runs_path = loaded.runs_dir(base)
    return UISettings(
        skill=loaded,
        runs_dir=runs_path,
        scenario_sources=list(scenarios) if scenarios else None,
        allow_remote=allow_remote,
        cwd=base,
        allow_hosts=list(allow_hosts or []),
        bind_host=bind_host,
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
    """Refuse a ``Host`` that :func:`host_allowed` rejects (421); add security headers."""

    def __init__(
        self, app: Any, *, allow_remote: bool, names: frozenset[str] = frozenset()
    ) -> None:
        self.app = app
        self.allow_remote = allow_remote
        self.names = names

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        name = host_name(headers.get("host", ""))
        if not host_allowed(name, allow_remote=self.allow_remote, names=self.names):
            message = (
                "this server answers loopback, IP-address and allowed host names only "
                "(mcpsim ui --allow-host NAME adds one)"
                if self.allow_remote
                else "this server only answers requests for a loopback host"
            )
            response = _error(421, message)
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
        self.skill_source = SkillSource(settings.skill)
        self.store = Store(
            scenario_sources=self._scenario_sources,
            runs_dir=settings.runs_dir,
            cwd=settings.cwd,
        )
        skill_path = settings.skill_dir.resolve()
        self.jobs = jobs or JobManager(
            runs_dir=settings.runs_dir.resolve(),
            skill_dir=skill_path,
            command=settings.command,
            # --skill names the skill; MCPSIM_SKILL makes library defaults agree with it.
            env={SKILL_ENV: str(skill_path)},
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

    def _scenario_sources(self) -> list[str]:
        if self.settings.scenario_sources is not None:
            return list(self.settings.scenario_sources)
        skill, _ = self.skill_source.load()
        return list(skill.config.scenarios)

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
        """The skill as ``mcpsim config`` resolves it (:meth:`Skill.describe`): every role
        file's model and the layer it came from, its settings and prompts, the run defaults,
        the overrides, the scenario sources and the runs directory."""
        skill, error = self.skill_source.load()
        warnings: list[str] = []
        if error is not None:
            warnings.append(f"the skill no longer loads (showing the last one that did): {error}")
        try:
            described = skill.describe(base=self.settings.cwd)
        except SkillError as exc:  # runs_dir names an unset environment variable
            return _error(500, str(exc))
        warnings += described.get("problems", [])
        _, notes = self.store.files()
        warnings += notes
        return JSONResponse(
            {
                "version": __version__,
                "skill": {
                    "name": skill.name,
                    "description": skill.description,
                    "path": str(skill.path),
                    "config_file": described["config"],
                    "overrides": overrides_json(skill),
                    "error": error,
                },
                "roles": role_rows(skill, described),
                "run": {k: v["value"] for k, v in described["run"].items()},
                "run_sources": {k: v["source"] for k, v in described["run"].items()},
                "scenario_sources": self._scenario_sources(),
                "scenario_sources_from": (
                    "--scenarios" if self.settings.scenario_sources is not None else "config.yaml"
                ),
                "runs_dir": str(self.settings.runs_dir),
                "known_models": list(KNOWN_MODELS),
                "model_roles": list(MODEL_ROLES),
                "modes": list(MODES),
                "honesty_item": HONESTY_ITEM,
                "allow_remote": self.settings.allow_remote,
                "warnings": warnings,
            }
        )

    def scenarios(self, request: Request) -> Response:
        index = self.store.scan(fresh=True)
        active = self.jobs.active_scenarios()
        since = self.jobs.running_since()
        rows: list[dict[str, Any]] = []
        categories: dict[str, dict[str, Any]] = {}
        for name, view in sorted(index.views.items(), key=lambda kv: (kv[1].category, kv[1].title)):
            writing = self._in_progress(name, since)
            last, count = self.store.last_judged(name, writing)
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
                    # A newer run that was cancelled or crashed: last_run is still the result.
                    "latest_incomplete": self.store.latest_incomplete(name, last, writing),
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
        index = self.store.scan()
        view = index.views.get(name)
        if view is None:
            return _error(404, "unknown scenario")
        writing = self._in_progress(name, self.jobs.running_since())
        last, count = self.store.last_judged(name, writing)
        models: dict[str, Any] | None = None
        run: dict[str, Any] | None = None
        warnings: list[str] = []
        scenario = index.scenarios.get(name)
        if scenario is not None:
            skill, error = self.skill_source.load()
            if error is not None:
                warnings.append(f"the skill no longer loads: {error}")
            try:
                models, run = scenario_settings(skill, scenario)
                skill.apply(scenario)  # what `mcpsim run` checks before it starts
            except (SkillError, ValueError) as exc:
                warnings.append(f"this scenario will not run with this skill: {exc}")
        return JSONResponse(
            {
                "scenario": view.to_json(),
                "status": self._scenario_status(name, last, self.jobs.active_scenarios()),
                "last_run": last,
                "latest_incomplete": self.store.latest_incomplete(name, last, writing),
                "run_count": count,
                "models": models,
                "run_settings": run,
                "warnings": warnings,
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
    names = (
        remote_host_names(settings.bind_host, settings.allow_hosts)
        if settings.allow_remote
        else frozenset()
    )
    app.add_middleware(SecurityMiddleware, allow_remote=settings.allow_remote, names=names)
    app.state.ui = ui
    return app


def check_bind(host: str, *, allow_remote: bool) -> None:
    """Raise ``ValueError`` for a non-loopback ``host`` unless ``allow_remote``."""
    if not allow_remote and not is_loopback(host):
        raise ValueError(
            f"refusing to bind {host!r}: not a loopback address (pass --allow-remote to expose "
            "the runner, its transcripts and its run button to the network)"
        )


def serve(settings: UISettings, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Run the app with uvicorn until interrupted. Refuses a non-loopback host unless allowed."""
    check_bind(host, allow_remote=settings.allow_remote)
    import uvicorn

    if settings.allow_remote and settings.bind_host is None:
        settings.bind_host = host
    app = create_app(settings)
    shown = f"[{host}]" if ":" in host else host
    print(f"mcpsim ui: http://{shown}:{port}/", flush=True)
    if settings.allow_remote:
        names = sorted(remote_host_names(settings.bind_host, settings.allow_hosts))
        print(
            "mcpsim ui: warning: --allow-remote: anyone who can reach this port can read every "
            "transcript and start runs that spend API credit. Requests are answered for "
            f"loopback, IP addresses and the host names {', '.join(names)} (add one with "
            "--allow-host NAME); any other Host gets 421.",
            file=sys.stderr,
            flush=True,
        )
    sources = settings.scenario_sources or settings.skill.config.scenarios
    print(f"  scenarios: {', '.join(sources) or '(none configured)'}", flush=True)
    print(f"  runs:      {settings.runs_dir}", flush=True)
    print(f"  skill:     {settings.skill_dir}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning", server_header=False)
