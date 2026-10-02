"""The runner (DESIGN §2 "Runner"): ``plan -> runs -> judge -> report`` with every artefact on disk.

Implements exactly the synchronous contract documented at the top of :mod:`mcpsim.cli`
(``plan_scenario``, ``run_scenario``, ``judge_run_dir``, ``report_run_dir``, ``run_suite``);
each wraps ``asyncio.run`` around an async body. A run directory is
``<out_dir>/<scenario.name>/<timestamp>/`` holding ``scenario.json`` (the validated scenario),
``plan.json``, ``transcripts/<path>-<mode>-<i>.jsonl``, ``verdicts/<same>.json``,
``report.json`` and ``report.md``.

Runs within a scenario execute concurrently up to ``scenario.concurrency``
(``asyncio.Semaphore``), each with its own MCP session (a stdio server is launched per run;
HTTP shares the URL). A stdio ``setup`` command runs once per ``run_scenario`` (and once per
``plan_scenario``, which also has to connect) with the stdio ``env`` applied. Every path runs in
``guided`` mode; a ``happy`` path additionally runs in ``free`` mode.

Dry run (``dry_run=True``): :func:`mcpsim.planner.plan` with ``dry_run=True``,
:func:`mcpsim.agent.run_path` with ``dry_run=True`` and the matcher-only
:func:`mcpsim.judge.judge_deterministic`; no LLM is constructed. Otherwise every LLM comes from
:func:`make_llm_for`, which reads ``scenario.models.<role>`` (``provider:model``, see
docs/LOCAL_MODELS.md) and asks :func:`mcpsim.llm.make_llm` for that provider's cached client.

``model_overrides`` / ``allow_same_judge`` (the CLI's ``--models`` and ``--allow-same-judge``)
are applied with :func:`apply_model_overrides` right after the scenario is loaded, so scenario
files stay provider-neutral and the ``scenario.json`` written to the run directory records the
models that actually ran.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path as FsPath
from typing import Literal

from mcpsim.agent import DRY_RUN_MODEL, run_path
from mcpsim.judge import check_judge_model, judge, judge_deterministic
from mcpsim.llm import (
    API_KEY_ENV,
    LLM,
    LOCAL_COST_NOTE,
    RoutingLLM,
    is_local_model,
    make_llm,
    parse_model_spec,
)
from mcpsim.mcpclient import Catalog, connect
from mcpsim.plan import MODES, ExecutionPlan, Mode, Path
from mcpsim.planner import plan as plan_paths
from mcpsim.report import (
    Report,
    SuiteReport,
    aggregate,
    aggregate_suite,
    render_markdown,
    render_suite_markdown,
    summary_line,
)
from mcpsim.report import exit_code as report_exit_code
from mcpsim.scenario import MODEL_ROLES, Models, Scenario, load_scenario
from mcpsim.transcript import EndEvent, SystemEvent, Transcript
from mcpsim.verdict import Verdict

Role = Literal["planner", "agent", "judge", "user"]

SCENARIO_FILE = "scenario.json"
PLAN_FILE = "plan.json"
TRANSCRIPTS_DIR = "transcripts"
VERDICTS_DIR = "verdicts"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
SUITE_JSON = "suite.json"
SUITE_MD = "suite.md"
SCENARIO_SUFFIXES = (".yaml", ".yml", ".json")
SETUP_TIMEOUT_S = 600

__all__ = ["API_KEY_ENV"]  # re-exported for callers that check the key through the runner


# --- LLM construction -------------------------------------------------------------------------


def make_llm_for(scenario: Scenario, role: Role) -> LLM:
    """The one place an LLM client is built for a role (planner / agent / judge / user).

    The provider comes from ``scenario.models.<role>`` (``provider:model``; a bare name is an
    Anthropic model) and :func:`mcpsim.llm.make_llm` returns that provider's cached client. The
    ``agent`` client also serves the simulated user (``agent.run_path`` takes one ``llm``), so
    when the user's provider differs from the agent's a :class:`~mcpsim.llm.RoutingLLM` is
    returned that picks the provider per call.
    """
    provider, _ = parse_model_spec(scenario.models.for_role(role))
    if role == "agent":
        user_provider, _ = parse_model_spec(scenario.models.user_model)
        if user_provider != provider:
            # Build both now so a missing key or unknown provider fails before any run.
            make_llm(provider, purpose="agent")
            make_llm(user_provider, purpose="simulated user")
            return RoutingLLM()
    return make_llm(provider, purpose=role)


def apply_model_overrides(
    scenario: Scenario,
    overrides: Mapping[str, str] | None = None,
    *,
    allow_same_judge: bool = False,
) -> Scenario:
    """``--models`` / ``--allow-same-judge`` applied to a loaded scenario (a validated copy).

    ``overrides`` maps a role (``planner`` / ``agent`` / ``judge`` / ``user``) to a model spec;
    ``allow_same_judge=True`` sets ``models.allow_same_judge`` (it never resets a scenario's own
    ``true``). Unknown roles raise ``ValueError``.
    """
    overrides = dict(overrides or {})
    if not overrides and not allow_same_judge:
        return scenario
    unknown = sorted(set(overrides) - set(MODEL_ROLES))
    if unknown:
        raise ValueError(
            f"unknown model role(s) {', '.join(unknown)}; expected one of {', '.join(MODEL_ROLES)}"
        )
    for spec in overrides.values():
        parse_model_spec(spec)  # raises ValueError for an empty name
    data = {**scenario.models.model_dump(), **overrides}
    if allow_same_judge:
        data["allow_same_judge"] = True
    models = Models.model_validate(data)
    return scenario.model_copy(update={"models": models})


def uses_local_models(scenario: Scenario) -> bool:
    """True when any role (including the simulated user) runs on a local provider."""
    return any(is_local_model(scenario.models.for_role(role)) for role in MODEL_ROLES)


# --- paths and timestamps ---------------------------------------------------------------------


def timestamp() -> str:
    """UTC, filesystem-safe, sortable: ``20260101T120000Z``."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def new_run_dir(out_dir: str | os.PathLike[str], scenario_name: str) -> FsPath:
    """``<out_dir>/<scenario>/<timestamp>/``, with a ``-N`` suffix when that second is taken."""
    base = FsPath(out_dir) / scenario_name
    stamp = timestamp()
    candidate = base / stamp
    n = 1
    while candidate.exists():
        candidate = base / f"{stamp}-{n}"
        n += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _log(message: str) -> None:
    print(f"mcpsim: {message}", file=sys.stderr, flush=True)


# --- setup command ----------------------------------------------------------------------------


def run_setup(scenario: Scenario) -> None:
    """Run the stdio ``setup`` command once, with the stdio ``env`` applied; no-op otherwise.

    Raises ``RuntimeError`` with the tail of stderr when the command fails.
    """
    stdio = scenario.server.stdio
    if stdio is None or not stdio.setup or not stdio.setup.strip():
        return
    argv = shlex.split(stdio.setup)
    env = {**os.environ, **stdio.env}
    _log(f"setup: {stdio.setup}")
    try:
        completed = subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=SETUP_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"setup command not found: {argv[0]} ({exc})") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"setup command timed out after {SETUP_TIMEOUT_S}s: {stdio.setup}"
        ) from exc
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout or "").strip().splitlines()[-10:])
        raise RuntimeError(
            f"setup command failed with exit {completed.returncode}: {stdio.setup}"
            + (f"\n{tail}" if tail else "")
        )


# --- discovery and planning -------------------------------------------------------------------


async def discover_catalog(scenario: Scenario) -> Catalog:
    async with connect(scenario.server) as session:
        return await session.catalog()


def allowed_catalog(scenario: Scenario, catalog: Catalog) -> Catalog:
    """The catalog after ``scenario.tools`` allow/deny globs, warning once per unmatched glob.

    This is applied once per scenario; the planner, every run (the agent and the dry run) and
    ``plan.catalog_digest`` all work from the result, never from the server's full catalog.
    """
    policy = scenario.tools
    for field_name, patterns in (("allow", policy.allow), ("deny", policy.deny)):
        for pattern in catalog.unmatched_patterns(patterns):
            _log(
                f"warning: tools.{field_name} glob {pattern!r} matches no tool of this server "
                f"(tools: {', '.join(catalog.tool_names()) or '(none)'})"
            )
    allowed = catalog.filtered(policy.allow, policy.deny)
    if not allowed.tools:
        _log(
            f"warning: tools.allow/deny leave no tool for {scenario.name!r}; the agent will have "
            "nothing to call"
        )
    return allowed


async def _plan(scenario: Scenario, catalog: Catalog, *, dry_run: bool) -> ExecutionPlan:
    llm = None if dry_run else make_llm_for(scenario, "planner")
    return await plan_paths(scenario, catalog, llm, dry_run=dry_run)


def _write_scenario(scenario: Scenario, run_dir: FsPath) -> FsPath:
    path = run_dir / SCENARIO_FILE
    path.write_text(scenario.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _load_scenario_json(run_dir: FsPath) -> Scenario:
    path = run_dir / SCENARIO_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir}: no {SCENARIO_FILE}; is this a run directory?")
    return Scenario.model_validate_json(path.read_text(encoding="utf-8"))


def _load_plan_json(run_dir: FsPath) -> ExecutionPlan:
    path = run_dir / PLAN_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir}: no {PLAN_FILE}; is this a run directory?")
    return ExecutionPlan.load(path)


# --- the run matrix ---------------------------------------------------------------------------


def modes_for(path: Path) -> list[Mode]:
    """Every path runs ``guided``; the happy path also runs ``free`` (DESIGN §1 "Mode")."""
    return list(MODES) if path.kind == "happy" else ["guided"]


def select_paths(plan: ExecutionPlan, only_path: str | None) -> list[Path]:
    if only_path is None:
        return list(plan.paths)
    for p in plan.paths:
        if p.id == only_path:
            return [p]
    known = ", ".join(p.id for p in plan.paths) or "(none)"
    raise ValueError(f"no path with id {only_path!r} in the plan; available: {known}")


def run_matrix(
    plan: ExecutionPlan,
    *,
    only_path: str | None,
    repeat: int,
    mode: Mode | None,
) -> list[tuple[Path, Mode, int]]:
    """Every (path, mode, index) the scenario asks for, after narrowing."""
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if mode is not None and mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
    cells: list[tuple[Path, Mode, int]] = []
    for path in select_paths(plan, only_path):
        for m in modes_for(path):
            if mode is not None and m != mode:
                continue
            cells.extend((path, m, i) for i in range(repeat))
    return cells


def _connection_failure(
    scenario: Scenario, path: Path, mode: Mode, index: int, exc: BaseException
) -> Transcript:
    """A transcript standing in for a run whose MCP session could not be opened."""
    text = str(exc).strip()
    reason = f"could not open MCP session: {type(exc).__name__}" + (f": {text}" if text else "")
    transcript = Transcript(scenario=scenario.name, path_id=path.id, mode=mode, index=index)
    transcript.add(
        SystemEvent(scenario=scenario.name, path_id=path.id, index=index, mode=mode)
    )
    transcript.add(EndEvent(outcome="error", reason=reason))
    transcript.outcome = "error"
    transcript.reason = reason
    return transcript


async def _one_run(
    scenario: Scenario,
    path: Path,
    mode: Mode,
    index: int,
    *,
    catalog: Catalog,
    agent_llm: LLM | None,
    dry_run: bool,
) -> Transcript:
    """One MCP session, one run of one path; never raises for run failures."""
    try:
        async with connect(scenario.server) as session:
            return await run_path(
                scenario,
                path,
                mode,
                index,
                session,
                agent_llm,
                dry_run=dry_run,
                catalog=catalog,
            )
    except Exception as exc:  # noqa: BLE001 - a broken server is a failed run, not a crash
        return _connection_failure(scenario, path, mode, index, exc)


async def _judge_one(
    scenario: Scenario,
    plan: ExecutionPlan,
    transcript: Transcript,
    *,
    judge_llm: LLM | None,
    votes: int | None,
) -> Verdict:
    if judge_llm is None or _is_dry_run_transcript(transcript):
        return judge_deterministic(scenario, transcript)
    return await judge(scenario, plan.path(transcript.path_id), transcript, judge_llm, votes)


def _is_dry_run_transcript(transcript: Transcript) -> bool:
    system = next((e for e in transcript.events if isinstance(e, SystemEvent)), None)
    return system is not None and system.models.get("agent") == DRY_RUN_MODEL


def _write_report(
    run_dir: FsPath,
    scenario: Scenario,
    verdicts: Iterable[Verdict],
    transcripts: Iterable[Transcript],
) -> tuple[FsPath, FsPath]:
    report = aggregate(list(verdicts), list(transcripts), scenario=scenario.name, run_dir=run_dir)
    if uses_local_models(scenario):
        report.cost_note = f"{LOCAL_COST_NOTE}; hosted calls, if any, are an {report.cost_note}"
    json_path = report.save(run_dir / REPORT_JSON)
    md_path = run_dir / REPORT_MD
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


async def _run_scenario_async(
    scenario: Scenario,
    run_dir: FsPath,
    *,
    plan_path: str | os.PathLike[str] | None,
    only_path: str | None,
    repeat: int | None,
    mode: Mode | None,
    dry_run: bool,
) -> FsPath:
    if not dry_run:
        check_judge_model(scenario)
        if scenario.models.judge == scenario.models.agent:
            _log(
                f"warning: judge model {scenario.models.judge!r} equals the agent model "
                "(allowed by models.allow_same_judge)"
            )

    catalog = allowed_catalog(scenario, await discover_catalog(scenario))
    if plan_path is not None:
        plan = ExecutionPlan.load(FsPath(plan_path))
        if plan.scenario != scenario.name:
            _log(
                f"warning: plan {plan_path} was made for scenario {plan.scenario!r}, "
                f"running it against {scenario.name!r}"
            )
        if plan.catalog_digest != catalog.digest():
            _log(
                f"warning: the allowed catalog changed since plan {plan_path} was made "
                "(digest differs: the server's tools or the scenario's tools policy); "
                "review the plan"
            )
    else:
        plan = await _plan(scenario, catalog, dry_run=dry_run)
    plan.save(run_dir / PLAN_FILE)

    cells = run_matrix(
        plan, only_path=only_path, repeat=scenario.repeat if repeat is None else repeat, mode=mode
    )
    agent_llm = None if dry_run else make_llm_for(scenario, "agent")
    judge_llm = None if dry_run else make_llm_for(scenario, "judge")
    semaphore = asyncio.Semaphore(scenario.concurrency)
    transcripts_dir = run_dir / TRANSCRIPTS_DIR
    verdicts_dir = run_dir / VERDICTS_DIR
    _log(f"{scenario.name}: {len(cells)} run(s), concurrency {scenario.concurrency}")

    async def cell(path: Path, m: Mode, index: int) -> tuple[Transcript, Verdict]:
        async with semaphore:
            transcript = await _one_run(
                scenario, path, m, index, catalog=catalog, agent_llm=agent_llm, dry_run=dry_run
            )
            transcript.write_jsonl(transcripts_dir / f"{transcript.stem}.jsonl")
            _log(f"{scenario.name}: {transcript.stem}: {transcript.outcome} ({transcript.reason})")
            verdict = await _judge_one(
                scenario, plan, transcript, judge_llm=judge_llm, votes=None
            )
            verdict.save(verdicts_dir / f"{transcript.stem}.json")
            return transcript, verdict

    results = await asyncio.gather(*(cell(p, m, i) for p, m, i in cells))
    transcripts = [t for t, _ in results]
    verdicts = [v for _, v in results]
    _write_report(run_dir, scenario, verdicts, transcripts)
    return run_dir


# --- public, synchronous contract -------------------------------------------------------------


def plan_scenario(
    scenario_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    dry_run: bool = False,
    model_overrides: Mapping[str, str] | None = None,
    allow_same_judge: bool = False,
) -> FsPath:
    """Load, discover, plan; write ``<out_dir>/<scenario.name>/<timestamp>/plan.json``."""
    scenario = apply_model_overrides(
        load_scenario(FsPath(scenario_path)), model_overrides, allow_same_judge=allow_same_judge
    )
    run_setup(scenario)

    async def body() -> ExecutionPlan:
        catalog = allowed_catalog(scenario, await discover_catalog(scenario))
        return await _plan(scenario, catalog, dry_run=dry_run)

    plan = asyncio.run(body())
    run_dir = new_run_dir(out_dir, scenario.name)
    _write_scenario(scenario, run_dir)
    return plan.save(run_dir / PLAN_FILE)


def run_scenario(
    scenario_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    plan_path: str | os.PathLike[str] | None = None,
    only_path: str | None = None,
    repeat: int | None = None,
    mode: Mode | None = None,
    dry_run: bool = False,
    model_overrides: Mapping[str, str] | None = None,
    allow_same_judge: bool = False,
) -> FsPath:
    """``plan -> runs -> judge -> report``; returns the run directory (see the module docstring)."""
    scenario = apply_model_overrides(
        load_scenario(FsPath(scenario_path)), model_overrides, allow_same_judge=allow_same_judge
    )
    if repeat is not None and repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if plan_path is not None and not FsPath(plan_path).is_file():
        raise FileNotFoundError(f"plan file not found: {plan_path}")
    run_setup(scenario)
    run_dir = new_run_dir(out_dir, scenario.name)
    _write_scenario(scenario, run_dir)
    return asyncio.run(
        _run_scenario_async(
            scenario,
            run_dir,
            plan_path=plan_path,
            only_path=only_path,
            repeat=repeat,
            mode=mode,
            dry_run=dry_run,
        )
    )


def _read_transcripts(run_dir: FsPath) -> list[Transcript]:
    folder = run_dir / TRANSCRIPTS_DIR
    if not folder.is_dir():
        raise FileNotFoundError(f"{run_dir}: no {TRANSCRIPTS_DIR}/ directory")
    return [Transcript.read_jsonl(p) for p in sorted(folder.glob("*.jsonl"))]


def _read_verdicts(run_dir: FsPath) -> list[Verdict]:
    folder = run_dir / VERDICTS_DIR
    if not folder.is_dir():
        return []
    return [Verdict.load(p) for p in sorted(folder.glob("*.json"))]


def judge_run_dir(
    run_dir: str | os.PathLike[str],
    *,
    votes: int | None = None,
) -> list[FsPath]:
    """Re-judge every saved transcript; rewrite verdicts and the report; return verdict paths.

    Transcripts recorded in dry run (agent model ``"dry-run"``) are judged by the matcher alone,
    as in the original run; every other transcript gets the LLM judge from :func:`make_llm_for`.
    """
    folder = FsPath(run_dir)
    scenario = _load_scenario_json(folder)
    plan = _load_plan_json(folder)
    transcripts = _read_transcripts(folder)
    if votes is not None and votes < 1:
        raise ValueError(f"votes must be >= 1, got {votes}")
    needs_llm = any(not _is_dry_run_transcript(t) for t in transcripts)
    judge_llm = make_llm_for(scenario, "judge") if needs_llm else None
    if needs_llm:
        check_judge_model(scenario)

    async def body() -> list[Verdict]:
        return list(
            await asyncio.gather(
                *(
                    _judge_one(scenario, plan, t, judge_llm=judge_llm, votes=votes)
                    for t in transcripts
                )
            )
        )

    verdicts = asyncio.run(body())
    written = [
        v.save(folder / VERDICTS_DIR / f"{t.stem}.json")
        for t, v in zip(transcripts, verdicts, strict=True)
    ]
    _write_report(folder, scenario, verdicts, transcripts)
    return written


def report_run_dir(
    run_dir: str | os.PathLike[str],
) -> tuple[FsPath, FsPath]:
    """Rebuild ``report.json`` and ``report.md`` from ``verdicts/`` and ``transcripts/``."""
    folder = FsPath(run_dir)
    scenario = _load_scenario_json(folder)
    transcripts = _read_transcripts(folder)
    verdicts = _read_verdicts(folder)
    return _write_report(folder, scenario, verdicts, transcripts)


def scenario_files(scenario_dir: str | os.PathLike[str]) -> list[FsPath]:
    """Every ``*.yaml`` / ``*.yml`` / ``*.json`` file in the directory, sorted by name."""
    folder = FsPath(scenario_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"scenario directory not found: {folder}")
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SCENARIO_SUFFIXES
    )
    if not files:
        raise ValueError(f"no scenario files (*.yaml, *.yml, *.json) in {folder}")
    return files


def run_suite(
    scenario_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    threshold: float = 1.0,
    dry_run: bool = False,
    model_overrides: Mapping[str, str] | None = None,
    allow_same_judge: bool = False,
) -> int:
    """Run every scenario in the directory, write ``suite.json`` / ``suite.md``, return exit."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    files = scenario_files(scenario_dir)
    reports: list[Report] = []
    for file in files:
        run_dir = run_scenario(
            file,
            out_dir,
            dry_run=dry_run,
            model_overrides=model_overrides,
            allow_same_judge=allow_same_judge,
        )
        report = Report.load(run_dir / REPORT_JSON)
        reports.append(report)
        print(f"{report.scenario}: {summary_line(report)}  ({run_dir})")
    suite: SuiteReport = aggregate_suite(reports)
    suite_dir = FsPath(out_dir) / f"suite-{timestamp()}"
    n = 1
    while suite_dir.exists():
        suite_dir = FsPath(out_dir) / f"suite-{timestamp()}-{n}"
        n += 1
    suite_dir.mkdir(parents=True)
    suite.save(suite_dir / SUITE_JSON)
    (suite_dir / SUITE_MD).write_text(render_suite_markdown(suite), encoding="utf-8")
    print(f"suite: {summary_line(suite)}")
    print(f"suite report: {suite_dir / SUITE_MD}")
    return report_exit_code(suite, threshold)
