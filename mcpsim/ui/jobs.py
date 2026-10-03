"""Background runs for the runner UI: one job runs its scenarios one after another, each as an
``mcpsim run <file> --skill <dir> --out <runs_dir>`` subprocess (contract D), so a run from the
page resolves its models, prompts and run settings exactly as the same command typed in a shell.

A job lives in a daemon thread, so it behaves the same under uvicorn and under a test client
whose event loop comes and goes. Output (stdout and stderr merged) goes to a bounded log; the
CLI's ``run dir: <path>`` line names the run directory, whose report then decides the
scenario's status. Nothing here goes through a shell: the command is an argument list built
from validated values. Each subprocess leads its own process group, so stopping a job also
stops the MCP servers and other children the run started.
"""

from __future__ import annotations

import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path as FsPath
from typing import Any

from mcpsim.plan import MODES
from mcpsim.scenario import MODEL_ROLES

JOB_STATUSES: tuple[str, ...] = ("queued", "running", "done", "failed")
SCENARIO_STATUSES: tuple[str, ...] = (
    "queued",
    "running",
    "passed",
    "failed",
    "partial",
    "error",
    "cancelled",
)
MODEL_SPEC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
RUN_DIR_RE = re.compile(r"^run dir: (.+)$")
LOG_LINES = 2000
LOG_TAIL = 300
MAX_REPEAT = 100
MAX_JOBS_KEPT = 50
# How long to keep reading a finished subprocess's output before moving on.
OUTPUT_DRAIN_S = 3.0


def terminate(process: subprocess.Popen[str], *, kill: bool = False) -> None:
    """Signal the subprocess's whole process group (it was started as a session leader), or
    the process alone where process groups do not exist."""
    if process.poll() is not None:
        return
    sig = signal.SIGKILL if kill and hasattr(signal, "SIGKILL") else signal.SIGTERM
    killpg = getattr(os, "killpg", None)
    if killpg is not None:
        try:
            killpg(process.pid, sig)
            return
        except (ProcessLookupError, PermissionError):
            pass
    try:
        if kill:
            process.kill()
        else:
            process.terminate()
    except ProcessLookupError:
        pass


def default_command() -> list[str]:
    """``python -m mcpsim.cli``: the same interpreter (and import path) as this server."""
    return [sys.executable, "-m", "mcpsim.cli"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobConflictError(Exception):
    """A requested scenario is already queued or running in another job."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(f"already queued or running: {', '.join(names)}")
        self.names = names


@dataclass
class RunOptions:
    models: dict[str, str] = field(default_factory=dict)
    repeat: int | None = None
    modes: list[str] = field(default_factory=list)
    dry_run: bool = False
    allow_same_judge: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "models": dict(self.models),
            "repeat": self.repeat,
            "modes": list(self.modes),
            "dry_run": self.dry_run,
            "allow_same_judge": self.allow_same_judge,
        }


def parse_run_options(body: Mapping[str, Any]) -> RunOptions:
    """Validate the optional fields of ``POST /api/run``; raises ``ValueError`` with a reason."""
    models_raw = body.get("models")
    models: dict[str, str] = {}
    if models_raw is not None:
        if not isinstance(models_raw, dict):
            raise ValueError("models must be an object of role -> provider:model")
        for role, spec in models_raw.items():
            if role not in MODEL_ROLES:
                expected = ", ".join(MODEL_ROLES)
                raise ValueError(f"models: unknown role {role!r}; expected {expected}")
            if spec is None or spec == "":
                continue
            if not isinstance(spec, str) or not MODEL_SPEC_RE.match(spec):
                raise ValueError(f"models.{role}: not a model spec: {spec!r}")
            models[role] = spec
    repeat = body.get("repeat")
    if repeat is not None and (
        isinstance(repeat, bool) or not isinstance(repeat, int) or not 1 <= repeat <= MAX_REPEAT
    ):
        raise ValueError(f"repeat must be an integer from 1 to {MAX_REPEAT}")
    modes_raw = body.get("modes")
    modes: list[str] = []
    if modes_raw is not None:
        if not isinstance(modes_raw, list) or not modes_raw:
            raise ValueError(f"modes must be a non-empty list of {', '.join(MODES)}")
        for mode in modes_raw:
            if mode not in MODES:
                raise ValueError(f"modes: unknown mode {mode!r}; expected {', '.join(MODES)}")
            if mode not in modes:
                modes.append(mode)
    flags: dict[str, bool] = {}
    for key in ("dry_run", "allow_same_judge"):
        value = body.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        flags[key] = value
    return RunOptions(models=models, repeat=repeat, modes=modes, **flags)


def build_command(
    prefix: Sequence[str],
    scenario_file: FsPath,
    runs_dir: FsPath,
    options: RunOptions,
    skill_dir: FsPath | None = None,
) -> list[str]:
    """The ``mcpsim run`` argument list for one scenario. The settings form maps to the
    command line's layer: ``--models`` (top of the model precedence), ``--repeat``, ``--mode``
    (one mode; both modes leave the choice to the skill and the scenario), ``--dry-run`` and
    ``--allow-same-judge``."""
    cmd = [*prefix, "run", str(scenario_file)]
    if skill_dir is not None:
        cmd += ["--skill", str(skill_dir)]
    cmd += ["--out", str(runs_dir)]
    if options.models:
        cmd += ["--models", ",".join(f"{k}={v}" for k, v in sorted(options.models.items()))]
    if options.repeat is not None:
        cmd += ["--repeat", str(options.repeat)]
    if len(options.modes) == 1:
        cmd += ["--mode", options.modes[0]]
    if options.dry_run:
        cmd.append("--dry-run")
    if options.allow_same_judge:
        cmd.append("--allow-same-judge")
    return cmd


@dataclass
class ScenarioTask:
    name: str
    file: FsPath
    status: str = "queued"
    exit_code: int | None = None
    run_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "exit_code": self.exit_code,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class Job:
    id: str
    tasks: list[ScenarioTask]
    options: RunOptions
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    cancel_requested: bool = False
    log: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_LINES))
    log_total: int = 0
    process: subprocess.Popen[str] | None = None

    def to_json(self, tail: int = LOG_TAIL) -> dict[str, Any]:
        lines = list(self.log)[-tail:] if tail > 0 else []
        return {
            "job_id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "options": self.options.to_json(),
            "scenarios": [t.to_json() for t in self.tasks],
            "log_tail": lines,
            "log_lines": self.log_total,
        }


# A run directory's status once its subprocess has finished: (scenario name, run id) ->
# passed / failed / partial, or None when the directory has no judged runs.
StatusLookup = Callable[[str, str], str | None]


class JobManager:
    def __init__(
        self,
        *,
        runs_dir: FsPath,
        skill_dir: FsPath | None = None,
        command: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: FsPath | None = None,
        status_lookup: StatusLookup | None = None,
        output_drain_s: float = OUTPUT_DRAIN_S,
    ) -> None:
        self.runs_dir = runs_dir
        self.skill_dir = skill_dir
        self.output_drain_s = output_drain_s
        self.command = list(command) if command else default_command()
        self.env_extra = dict(env or {})
        self.cwd = cwd
        self.status_lookup = status_lookup
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    # --- queries ----------------------------------------------------------------------------

    def get(self, job_id: str, tail: int = LOG_TAIL) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_json(tail) if job is not None else None

    def list(self, tail: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.to_json(tail) for j in jobs]

    def active_scenarios(self) -> set[str]:
        with self._lock:
            return {
                t.name
                for j in self._jobs.values()
                for t in j.tasks
                if t.status in ("queued", "running")
            }

    def running_since(self) -> dict[str, str]:
        """Scenario name -> when its subprocess started (ISO), for tasks running right now."""
        with self._lock:
            return {
                t.name: t.started_at
                for j in self._jobs.values()
                for t in j.tasks
                if t.status == "running" and t.started_at is not None
            }

    # --- commands ---------------------------------------------------------------------------

    def submit(self, scenarios: Sequence[tuple[str, FsPath]], options: RunOptions) -> str:
        if not scenarios:
            raise ValueError("no scenarios to run")
        with self._lock:
            busy = {
                t.name
                for j in self._jobs.values()
                for t in j.tasks
                if t.status in ("queued", "running")
            }
            clash = sorted({name for name, _ in scenarios} & busy)
            if clash:
                raise JobConflictError(clash)
            job_id = secrets.token_hex(8)
            job = Job(
                id=job_id,
                tasks=[ScenarioTask(name=name, file=path) for name, path in scenarios],
                options=options,
            )
            self._jobs[job_id] = job
            self._prune()
        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        self._threads.append(thread)
        thread.start()
        return job_id

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in ("done", "failed"):
                return True
            job.cancel_requested = True
            process = job.process
        if process is not None:
            terminate(process)
        return True

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop every running subprocess (server exit); SIGKILL whatever outlives ``timeout``."""
        with self._lock:
            jobs = list(self._jobs.values())
            for job in jobs:
                job.cancel_requested = True
        for job in jobs:
            if job.process is not None:
                terminate(job.process)
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        for job in jobs:
            if job.process is not None:
                terminate(job.process, kill=True)

    def wait(self, job_id: str, timeout: float = 30.0) -> dict[str, Any] | None:
        """Block until the job finishes (tests and scripts); returns its final state."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.get(job_id)
            if state is None or state["status"] in ("done", "failed"):
                return state
            time.sleep(0.02)
        return self.get(job_id)

    # --- internals --------------------------------------------------------------------------

    def _prune(self) -> None:
        finished = sorted(
            (j for j in self._jobs.values() if j.status in ("done", "failed")),
            key=lambda j: j.created_at,
        )
        while len(self._jobs) > MAX_JOBS_KEPT and finished:
            self._jobs.pop(finished.pop(0).id, None)

    def _append(self, job: Job, line: str) -> None:
        with self._lock:
            job.log.append(line)
            job.log_total += 1

    def _run_job(self, job: Job) -> None:
        with self._lock:
            job.status = "running"
            job.started_at = _now()
        try:
            for task in job.tasks:
                with self._lock:
                    cancelled = job.cancel_requested
                if cancelled:
                    with self._lock:
                        task.status = "cancelled"
                    continue
                self._run_task(job, task)
        except Exception as exc:  # a job thread records the failure; it must not raise
            with self._lock:
                job.error = f"{type(exc).__name__}: {exc}"
                for task in job.tasks:
                    if task.status in ("queued", "running"):
                        task.status = "error"
        with self._lock:
            failed = job.error is not None or any(t.status == "error" for t in job.tasks)
            job.status = "failed" if failed else "done"
            job.finished_at = _now()
            job.process = None

    def _run_task(self, job: Job, task: ScenarioTask) -> None:
        cmd = build_command(self.command, task.file, self.runs_dir, job.options, self.skill_dir)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **self.env_extra}
        with self._lock:
            task.status = "running"
            task.started_at = _now()
        self._append(job, f"[{task.name}] $ mcpsim {' '.join(cmd[len(self.command) :])}")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                cwd=str(self.cwd) if self.cwd else None,
                start_new_session=True,
            )
        except OSError as exc:
            self._append(job, f"[{task.name}] cannot start: {exc}")
            with self._lock:
                task.status = "error"
                task.finished_at = _now()
            return
        with self._lock:
            job.process = process
            cancelled = job.cancel_requested
        if cancelled:
            terminate(process)
        found: list[str] = []
        stdout = process.stdout
        assert stdout is not None

        def pump() -> None:
            for raw in stdout:
                line = raw.rstrip("\n")
                m = RUN_DIR_RE.match(line.strip())
                if m:
                    found.append(m.group(1).strip())
                self._append(job, f"[{task.name}] {line}")

        # The output is read on its own thread: a process the run started in another session
        # (the MCP SDK starts stdio servers that way) can hold the pipe open after the CLI has
        # exited, and the job must not wait on it.
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        code = process.wait()
        reader.join(self.output_drain_s)
        if reader.is_alive():
            self._append(job, f"[{task.name}] (output still open after exit; not waiting for it)")
        run_id = self._run_id(task.name, found[-1] if found else None)
        status = self._status(job, task, code, run_id)
        self._append(job, f"[{task.name}] exit {code}: {status}")
        with self._lock:
            task.exit_code = code
            task.run_id = run_id
            task.status = status
            task.finished_at = _now()
            job.process = None

    def _run_id(self, name: str, run_dir: str | None) -> str | None:
        if not run_dir:
            return None
        path = FsPath(run_dir)
        if not path.is_absolute():
            path = (self.cwd or FsPath.cwd()) / path
        try:
            if path.resolve().parent != (self.runs_dir / name).resolve():
                return None
        except OSError:
            return None
        return path.name

    def _status(self, job: Job, task: ScenarioTask, code: int, run_id: str | None) -> str:
        """The run directory's verdicts decide; without them the run did not pass.

        * a judged run directory -> its status (passed / failed / partial);
        * a run directory with nothing judged -> failed;
        * no run directory: cancelled when the job was stopped, else error (the CLI failed
          before it wrote results, whatever its exit code).
        """
        if run_id is not None:
            looked_up = self.status_lookup(task.name, run_id) if self.status_lookup else None
            if looked_up in ("passed", "failed", "partial"):
                return looked_up
            return "failed"
        if job.cancel_requested and code != 0:
            return "cancelled"
        return "error"
