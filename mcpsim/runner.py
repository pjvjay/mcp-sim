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

Before planning, whenever the scenario's disclosure is not ``all``, the :mod:`mcpsim.scout`
observes the server read-only on the same session that discovered the catalog, the observers
report at the ``scout`` trigger, and the result is saved as ``scout.json`` beside ``plan.json``
(``--plan`` reuse skips the scout; the runs' observers still fire). Planning happens on that
session too, so a local planner's probe (:mod:`mcpsim.execution_planner`) is recorded in the
same ``scout.json``.

Dry run (``dry_run=True``): :func:`mcpsim.planner.plan` with ``dry_run=True``,
:func:`mcpsim.agent.run_path` with ``dry_run=True`` and the matcher-only
:func:`mcpsim.judge.judge_deterministic`; no LLM is constructed (code and group observers still
run). Otherwise every LLM comes from :func:`make_llm_for`, which reads ``scenario.models.<role>``
(``provider:model``, see docs/LOCAL_MODELS.md) and asks :func:`mcpsim.llm.make_llm` for that
provider's cached client.

The simulate skill (:mod:`mcpsim.skill`: ``--skill DIR``, else ``MCPSIM_SKILL``, else the
packaged ``skills/simulate``) supplies every role's prompts and settings, and resolves each
scenario right after it is loaded (:meth:`mcpsim.skill.Skill.apply`): models by the precedence
built-in < role frontmatter < ``config.yaml`` defaults < matching overrides < the scenario file
< ``model_overrides`` (the CLI's ``--models``), and ``repeat`` / ``modes`` / ``judge_votes`` /
``concurrency`` by the same ladder. The ``scenario.json`` written to the run directory records
what actually ran, so a re-judge reads the resolved models and votes from it.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path as FsPath
from typing import Any, Literal

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
    unpriced_models,
)
from mcpsim.mcpclient import Catalog, Session, connect, describe_exception, sole_leaf
from mcpsim.observers import ObserverRunner
from mcpsim.plan import MODES, ExecutionPlan, Mode, Path
from mcpsim.planner import plan as plan_paths
from mcpsim.planner import planner_profile
from mcpsim.report import (
    Report,
    SuiteError,
    SuiteReport,
    aggregate,
    aggregate_suite,
    render_markdown,
    render_suite_markdown,
    summary_line,
)
from mcpsim.report import exit_code as report_exit_code
from mcpsim.scenario import MODEL_ROLES, Models, Observer, Scenario, ScenarioError, load_scenario
from mcpsim.scout import PROBE_RESERVE, SCOUT_FILE, ScoutResult, scout, should_scout
from mcpsim.skill import (
    Resolved,
    ScenarioEntry,
    Skill,
    SkillError,
    SkillRef,
    load_entries,
    load_skill,
    one_line,
    select_entries,
)
from mcpsim.transcript import EndEvent, SystemEvent, Transcript
from mcpsim.verdict import Verdict

Role = Literal["planner", "agent", "judge", "user", "observer"]

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


def llm_observers(scenario: Scenario) -> list[Observer]:
    """The observers that need a model (``kind: llm``)."""
    return [o for o in scenario.observers if o.kind == "llm"]


def observer_providers(scenario: Scenario) -> list[str]:
    """Distinct providers the LLM observers use (each observer's own ``model``, else
    ``models.observer``, else the agent's), in declaration order."""
    providers: list[str] = []
    for obs in llm_observers(scenario):
        provider, _ = parse_model_spec(scenario.models.model_for_observer(obs))
        if provider not in providers:
            providers.append(provider)
    return providers


def make_llm_for(scenario: Scenario, role: Role) -> LLM:
    """The one place an LLM client is built for a role (planner / agent / judge / user /
    observer).

    The provider comes from ``scenario.models.<role>`` (``provider:model``; a bare name is an
    Anthropic model) and :func:`mcpsim.llm.make_llm` returns that provider's cached client. The
    ``agent`` client also serves the simulated user (``agent.run_path`` takes one ``llm``), so
    when the user's provider differs from the agent's a :class:`~mcpsim.llm.RoutingLLM` is
    returned that picks the provider per call; likewise ``observer`` routes when the LLM
    observers' models span several providers.
    """
    provider, _ = parse_model_spec(scenario.models.for_role(role))
    if role == "agent":
        user_provider, _ = parse_model_spec(scenario.models.user_model)
        if user_provider != provider:
            # Build both now so a missing key or unknown provider fails before any run.
            make_llm(provider, purpose="agent")
            make_llm(user_provider, purpose="simulated user")
            return RoutingLLM()
    if role == "observer":
        providers = observer_providers(scenario) or [provider]
        if len(providers) > 1:
            for each in providers:
                make_llm(each, purpose="observer")
            return RoutingLLM()
        return make_llm(providers[0], purpose="observer")
    return make_llm(provider, purpose=role)


def apply_model_overrides(
    scenario: Scenario,
    overrides: Mapping[str, str] | None = None,
    *,
    allow_same_judge: bool = False,
) -> Scenario:
    """``--models`` / ``--allow-same-judge`` applied to a loaded scenario (a validated copy),
    without the simulate skill's layers.

    The runner itself resolves through :meth:`mcpsim.skill.Skill.apply` (:func:`prepare_scenario`),
    where ``--models`` is the top of the precedence; this helper stays for library callers that
    only want the overrides. ``overrides`` maps a role (``planner`` / ``agent`` / ``judge`` /
    ``user`` / ``observer``) to a model spec; ``allow_same_judge=True`` sets
    ``models.allow_same_judge`` (it never resets a scenario's own ``true``). Unknown roles raise
    ``ValueError``.
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


def scout_observers(
    scenario: Scenario, *, dry_run: bool, skill: SkillRef = None
) -> ObserverRunner | None:
    """The observer runner the scout uses: code and group observers only in dry run, every
    observer (with its LLM) otherwise; ``None`` when no observer reports at ``scout``."""
    if not any("scout" in o.on for o in scenario.observers):
        return None
    if dry_run:
        return ObserverRunner(scenario, None, include_llm=False, skill=skill)
    needs_llm = any("scout" in o.on for o in llm_observers(scenario))
    return ObserverRunner(
        scenario, make_llm_for(scenario, "observer") if needs_llm else None, skill=skill
    )


async def discover_scout_and_plan(
    scenario: Scenario, *, dry_run: bool, skill: SkillRef = None
) -> tuple[Catalog, ScoutResult | None, ExecutionPlan]:
    """One session: discover the catalog, apply the tool policy, scout (unless disclosure is
    ``all``), and plan while the session is still open, so a local planner can probe one
    mutated read-only call on it (counted against the scout's budget, of which the scout leaves
    :data:`~mcpsim.scout.PROBE_RESERVE` call(s) unspent for it)."""
    llm = None if dry_run else make_llm_for(scenario, "planner")  # a missing key fails first
    plan: ExecutionPlan | None = None
    failure: Exception | None = None
    async with connect(scenario.server) as session:
        catalog = allowed_catalog(scenario, await session.catalog())
        scout_result = None
        if should_scout(scenario):
            scout_result = await _scout(scenario, catalog, session, dry_run=dry_run, skill=skill)
        try:
            plan = await _plan(
                scenario, catalog, llm, scout_result=scout_result, session=session, skill=skill
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below, outside the session
            # Raised inside the session, it would leave the stdio client's task group as an
            # ExceptionGroup; callers (and the CLI's messages) expect the PlanError itself.
            failure = exc
    if failure is not None:
        raise failure
    assert plan is not None
    return catalog, scout_result, plan


async def _scout(
    scenario: Scenario, catalog: Catalog, session: Session, *, dry_run: bool, skill: SkillRef = None
) -> ScoutResult:
    probes = not dry_run and planner_profile(scenario) == "local"
    result = await scout(
        scenario,
        catalog,
        session,
        observers=scout_observers(scenario, dry_run=dry_run, skill=skill),
        reserve=PROBE_RESERVE if probes else 0,
    )
    _log(
        f"{scenario.name}: scouted {len(result.observations)} observation(s) "
        f"({result.tool_calls} tool call(s) of {result.budget}); "
        f"{len(result.disclosed)} tool(s) disclosed, {len(result.on_request)} on request"
        + (f"; {len(result.reports)} informant report(s)" if result.reports else "")
    )
    return result


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
    if policy.initial is not None:
        for pattern in allowed.unmatched_patterns(policy.initial):
            _log(
                f"warning: tools.initial glob {pattern!r} matches no allowed tool "
                f"(allowed: {', '.join(allowed.tool_names()) or '(none)'})"
            )
    if not allowed.tools:
        _log(
            f"warning: tools.allow/deny leave no tool for {scenario.name!r}; the agent will have "
            "nothing to call"
        )
    for obs in scenario.observers:
        for cond in obs.conditions:
            for branch, effect in (("then", cond.then), ("otherwise", cond.otherwise)):
                for field_name in ("enable_tools", "disable_tools"):
                    globs: list[str] = getattr(effect, field_name)
                    for pattern in allowed.unmatched_patterns(globs):
                        _log(
                            f"warning: observer {obs.name}.{cond.id} {branch}.{field_name} glob "
                            f"{pattern!r} matches no allowed tool "
                            f"(allowed: {', '.join(allowed.tool_names()) or '(none)'})"
                        )
    return allowed


async def _plan(
    scenario: Scenario,
    catalog: Catalog,
    llm: LLM | None,
    *,
    scout_result: ScoutResult | None = None,
    session: Session | None = None,
    skill: SkillRef = None,
) -> ExecutionPlan:
    """``llm`` is ``None`` exactly in a dry run."""
    plan = await plan_paths(
        scenario,
        catalog,
        llm,
        scout=scout_result,
        dry_run=llm is None,
        probe_session=session,
        skill=skill,
    )
    for note in plan.notes:
        _log(f"{scenario.name}: {note}")
    return plan


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
    mode: Mode | None = None,
    modes: Sequence[Mode] | None = None,
) -> list[tuple[Path, Mode, int]]:
    """Every (path, mode, index) the scenario asks for, after narrowing to ``only_path`` and to
    ``mode`` (one mode) or ``modes`` (the modes to keep; default both)."""
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    keep = list(modes) if modes is not None else list(MODES)
    if mode is not None:
        keep = [mode]
    for m in keep:
        if m not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {m!r}")
    cells: list[tuple[Path, Mode, int]] = []
    for path in select_paths(plan, only_path):
        for m in modes_for(path):
            if m not in keep:
                continue
            cells.extend((path, m, i) for i in range(repeat))
    return cells


def _connection_failure(
    scenario: Scenario, path: Path, mode: Mode, index: int, exc: BaseException
) -> Transcript:
    """A transcript standing in for a run whose MCP session could not be opened."""
    reason = f"could not open MCP session: {describe_exception(exc)}"
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
    observer_llm: LLM | None = None,
    skill: SkillRef = None,
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
                observer_llm=observer_llm,
                skill=skill,
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
    skill: SkillRef = None,
) -> Verdict:
    if judge_llm is None or _is_dry_run_transcript(transcript):
        return judge_deterministic(scenario, transcript)
    return await judge(
        scenario, plan.path(transcript.path_id), transcript, judge_llm, votes, skill=skill
    )


def _is_dry_run_transcript(transcript: Transcript) -> bool:
    system = next((e for e in transcript.events if isinstance(e, SystemEvent)), None)
    return system is not None and system.models.get("agent") == DRY_RUN_MODEL


def _write_report(
    run_dir: FsPath,
    scenario: Scenario,
    verdicts: Iterable[Verdict],
    transcripts: Iterable[Transcript],
    *,
    repeat: int | None = None,
) -> tuple[FsPath, FsPath]:
    """``report.json`` and ``report.md``; ``repeat`` is pass^k's ``k``: how many runs per
    path × mode were asked for (the run's resolved repeat, which ``scenario.json`` records for a
    re-judge or re-report). Without it ``k`` is inferred from the runs on disk."""
    report = aggregate(
        list(verdicts),
        list(transcripts),
        scenario=scenario.name,
        run_dir=run_dir,
        repeat=repeat,
    )
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
    modes: Sequence[Mode] | None = None,
    skill: Skill | None = None,
) -> FsPath:
    if not dry_run:
        check_judge_model(scenario)
        if scenario.models.judge == scenario.models.agent:
            _log(
                f"warning: judge model {scenario.models.judge!r} equals the agent model "
                "(allowed by models.allow_same_judge)"
            )

    if plan_path is not None:
        catalog = allowed_catalog(scenario, await discover_catalog(scenario))
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
        catalog, scout_result, plan = await discover_scout_and_plan(
            scenario, dry_run=dry_run, skill=skill
        )
        if scout_result is not None:
            scout_result.save(run_dir / SCOUT_FILE)
    plan.save(run_dir / PLAN_FILE)

    repeats = scenario.repeat if repeat is None else repeat
    cells = run_matrix(plan, only_path=only_path, repeat=repeats, mode=mode, modes=modes)
    agent_llm = None if dry_run else make_llm_for(scenario, "agent")
    judge_llm = None if dry_run else make_llm_for(scenario, "judge")
    observer_llm = (
        make_llm_for(scenario, "observer") if not dry_run and llm_observers(scenario) else None
    )
    semaphore = asyncio.Semaphore(scenario.concurrency)
    transcripts_dir = run_dir / TRANSCRIPTS_DIR
    verdicts_dir = run_dir / VERDICTS_DIR
    _log(f"{scenario.name}: {len(cells)} run(s), concurrency {scenario.concurrency}")

    async def cell(path: Path, m: Mode, index: int) -> tuple[Transcript, Verdict]:
        async with semaphore:
            transcript = await _one_run(
                scenario,
                path,
                m,
                index,
                catalog=catalog,
                agent_llm=agent_llm,
                dry_run=dry_run,
                observer_llm=observer_llm,
                skill=skill,
            )
            transcript.write_jsonl(transcripts_dir / f"{transcript.stem}.jsonl")
            _log(f"{scenario.name}: {transcript.stem}: {transcript.outcome} ({transcript.reason})")
            verdict = await _judge_one(
                scenario, plan, transcript, judge_llm=judge_llm, votes=None, skill=skill
            )
            verdict.save(verdicts_dir / f"{transcript.stem}.json")
            return transcript, verdict

    results = await asyncio.gather(*(cell(p, m, i) for p, m, i in cells))
    transcripts = [t for t, _ in results]
    verdicts = [v for _, v in results]
    _write_report(run_dir, scenario, verdicts, transcripts, repeat=repeats)
    return run_dir


# --- public, synchronous contract -------------------------------------------------------------


def prepare_scenario(
    scenario_path: str | os.PathLike[str],
    *,
    skill: SkillRef = None,
    model_overrides: Mapping[str, str] | None = None,
    allow_same_judge: bool = False,
    repeat: int | None = None,
    modes: Sequence[Mode] | None = None,
    check_sampling: bool = True,
) -> tuple[Scenario, Resolved, Skill]:
    """Load a scenario and resolve it against the skill (:meth:`mcpsim.skill.Skill.apply`):
    ``(the scenario as it will run, the resolution with its sources, the skill)``.
    ``check_sampling=False`` (a dry run, which calls no model) skips the refusal of a
    temperature the resolved model rejects."""
    loaded = load_skill(skill)
    if repeat is not None and repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    run_overrides: dict[str, Any] = {}
    if repeat is not None:
        run_overrides["repeat"] = repeat
    if modes is not None:
        run_overrides["modes"] = list(modes)
    for spec in (model_overrides or {}).values():
        parse_model_spec(spec)  # raises ValueError for an empty name
    scenario, resolved = loaded.apply(
        load_scenario(FsPath(scenario_path)),
        model_overrides=model_overrides,
        run_overrides=run_overrides,
        allow_same_judge=allow_same_judge,
        check_sampling=check_sampling,
    )
    return scenario, resolved, loaded


def _log_resolution(
    scenario: Scenario, resolved: Resolved, skill: Skill, *, dry_run: bool = False
) -> None:
    models = ", ".join(f"{role}={s.value}" for role, s in resolved.models.items())
    run = resolved.run
    _log(
        f"{scenario.name}: skill {skill.path}; {models}; repeat {run['repeat'].value}, "
        f"modes {'+'.join(run['modes'].value)}, judge_votes {run['judge_votes'].value}, "
        f"concurrency {run['concurrency'].value}"
    )
    if dry_run:
        return
    m = scenario.models
    specs = [m.planner, m.agent, m.user, m.judge]
    specs += [m.model_for_observer(o) for o in llm_observers(scenario)]
    unpriced = unpriced_models(specs)
    if unpriced:
        _log(
            f"warning: no price is known for {', '.join(unpriced)}: the report counts its calls "
            "as $0 and budgets.max_cost_usd does not limit them (mcpsim.llm.RATE_TABLE)"
        )


def plan_scenario(
    scenario_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    dry_run: bool = False,
    model_overrides: Mapping[str, str] | None = None,
    allow_same_judge: bool = False,
    skill: SkillRef = None,
) -> FsPath:
    """Load, discover, plan; write ``<out_dir>/<scenario.name>/<timestamp>/plan.json``."""
    scenario, resolved, loaded = prepare_scenario(
        scenario_path,
        skill=skill,
        model_overrides=model_overrides,
        allow_same_judge=allow_same_judge,
        check_sampling=not dry_run,
    )
    _log_resolution(scenario, resolved, loaded, dry_run=dry_run)
    run_setup(scenario)

    async def body() -> tuple[ExecutionPlan, ScoutResult | None]:
        _, scout_result, plan = await discover_scout_and_plan(
            scenario, dry_run=dry_run, skill=loaded
        )
        return plan, scout_result

    plan, scout_result = asyncio.run(body())
    run_dir = new_run_dir(out_dir, scenario.name)
    _write_scenario(scenario, run_dir)
    if scout_result is not None:
        scout_result.save(run_dir / SCOUT_FILE)
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
    skill: SkillRef = None,
    modes: Sequence[Mode] | None = None,
) -> FsPath:
    """``plan -> runs -> judge -> report``; returns the run directory (see the module docstring).

    ``repeat`` and ``mode`` / ``modes`` are the command line's layer of the run settings;
    without them the skill's configuration and the scenario file decide.
    """
    scenario, resolved, loaded = prepare_scenario(
        scenario_path,
        skill=skill,
        model_overrides=model_overrides,
        allow_same_judge=allow_same_judge,
        repeat=repeat,
        modes=[mode] if mode is not None else modes,
        check_sampling=not dry_run,
    )
    if plan_path is not None and not FsPath(plan_path).is_file():
        raise FileNotFoundError(f"plan file not found: {plan_path}")
    _log_resolution(scenario, resolved, loaded, dry_run=dry_run)
    run_setup(scenario)
    run_dir = new_run_dir(out_dir, scenario.name)
    _write_scenario(scenario, run_dir)
    return asyncio.run(
        _run_scenario_async(
            scenario,
            run_dir,
            plan_path=plan_path,
            only_path=only_path,
            repeat=scenario.repeat,
            mode=None,
            dry_run=dry_run,
            modes=resolved.modes,
            skill=loaded,
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
    skill: SkillRef = None,
) -> list[FsPath]:
    """Re-judge every saved transcript; rewrite verdicts and the report; return verdict paths.

    Transcripts recorded in dry run (agent model ``"dry-run"``) are judged by the matcher alone,
    as in the original run; every other transcript gets the LLM judge from :func:`make_llm_for`.
    The judge's model and votes come from the run's ``scenario.json`` (``votes`` overrides the
    count); its prompts and settings from ``skill``.
    """
    folder = FsPath(run_dir)
    scenario = _load_scenario_json(folder)
    plan = _load_plan_json(folder)
    transcripts = _read_transcripts(folder)
    if votes is not None and votes < 1:
        raise ValueError(f"votes must be >= 1, got {votes}")
    loaded = load_skill(skill)
    needs_llm = any(not _is_dry_run_transcript(t) for t in transcripts)
    judge_llm = make_llm_for(scenario, "judge") if needs_llm else None
    if needs_llm:
        check_judge_model(scenario)
        loaded.check_sampling(scenario)

    async def body() -> list[Verdict]:
        return list(
            await asyncio.gather(
                *(
                    _judge_one(scenario, plan, t, judge_llm=judge_llm, votes=votes, skill=loaded)
                    for t in transcripts
                )
            )
        )

    verdicts = asyncio.run(body())
    written = [
        v.save(folder / VERDICTS_DIR / f"{t.stem}.json")
        for t, v in zip(transcripts, verdicts, strict=True)
    ]
    # scenario.json records the repeat the run resolved (--repeat included): pass^k's k.
    _write_report(folder, scenario, verdicts, transcripts, repeat=scenario.repeat)
    return written


def report_run_dir(
    run_dir: str | os.PathLike[str],
) -> tuple[FsPath, FsPath]:
    """Rebuild ``report.json`` and ``report.md`` from ``verdicts/`` and ``transcripts/``.
    pass^k's ``k`` is the repeat ``scenario.json`` records (what the run asked for, ``--repeat``
    included), so a run that stopped early reports its missing repeats instead of a smaller k."""
    folder = FsPath(run_dir)
    scenario = _load_scenario_json(folder)
    transcripts = _read_transcripts(folder)
    verdicts = _read_verdicts(folder)
    return _write_report(folder, scenario, verdicts, transcripts, repeat=scenario.repeat)


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


# --- the suite --------------------------------------------------------------------------------

USER_ERRORS: tuple[type[Exception], ...] = (
    ScenarioError,
    SkillError,
    OSError,
    ValueError,
    KeyError,
    RuntimeError,
)


@dataclass
class SuiteRow:
    """One scenario of a suite: its report, or the error that stopped it."""

    entry: ScenarioEntry
    report: Report | None = None
    run_dir: FsPath | None = None
    error: str | None = None


def suite_entries(
    scenario_dir: str | os.PathLike[str] | None = None,
    *,
    skill: SkillRef = None,
    names: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
) -> list[ScenarioEntry]:
    """The scenarios a suite runs: the files in ``scenario_dir`` when given, else every source
    in the skill's ``config.yaml``; narrowed by the ``names`` / ``categories`` globs (any of
    each). Entries that fail to load are kept (with ``error``) unless a filter leaves them out.
    Raises ``ValueError`` when nothing is left."""
    if scenario_dir is not None:
        files = [(f, str(scenario_dir)) for f in scenario_files(scenario_dir)]
        entries = load_entries(files)
        where = str(scenario_dir)
    else:
        loaded = load_skill(skill)
        entries = loaded.entries()
        where = f"the scenarios of {loaded.config_path}"
    selected = select_entries(entries, names=names, categories=categories)
    if not selected:
        filters = [
            *(f"--name {n}" for n in names or []),
            *(f"--category {c}" for c in categories or []),
        ]
        raise ValueError(
            f"no scenario in {where}"
            + (f" matches {' '.join(filters)}" if filters else "")
            + (f" (known: {', '.join(e.name for e in entries)})" if entries else "")
        )
    return selected


def format_suite_table(rows: Sequence[SuiteRow]) -> str:
    """The summary table ``mcpsim suite`` prints: one line per scenario with its pass rate,
    pass^k, cost, run time and run directory (or the error that stopped it)."""
    header = ["scenario", "category", "passed", "rate", "pass^k", "cost", "time", "result"]
    table: list[list[str]] = []
    for row in rows:
        name, category = row.entry.name, row.entry.category or "-"
        if row.report is None:
            error = one_line(row.error or row.entry.error or "error")
            table.append([name, category, "-", "-", "-", "-", "-", f"error: {error}"])
            continue
        r = row.report
        pk = f"pass^{r.pass_k.k} {'yes' if r.pass_k.all_passed else 'no'}" if r.pass_k.k else "n/a"
        table.append(
            [
                name,
                category,
                f"{r.passed}/{r.runs}",
                f"{r.pass_rate * 100:.0f}%",
                pk,
                f"${r.cost_usd:.2f}",
                f"{r.wall_clock_s or r.duration_s:.0f}s",
                str(row.run_dir or r.run_dir),
            ]
        )
    widths = [max([len(header[i]), *(len(t[i]) for t in table)]) for i in range(len(header) - 1)]
    lines = []
    for cells in [header, *table]:
        lead = "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells[:-1]))
        lines.append(f"{lead}  {cells[-1]}".rstrip())
    return "\n".join(lines)


def run_suite(
    scenario_dir: str | os.PathLike[str] | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    *,
    threshold: float = 1.0,
    dry_run: bool = False,
    model_overrides: Mapping[str, str] | None = None,
    allow_same_judge: bool = False,
    skill: SkillRef = None,
    names: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    repeat: int | None = None,
    modes: Sequence[Mode] | None = None,
    on_row: Callable[[SuiteRow], None] | None = None,
) -> int:
    """Run every selected scenario (:func:`suite_entries`) one after another into ``out_dir``
    (default: the skill's ``runs_dir``), print the summary table (:func:`format_suite_table`),
    write ``<out_dir>/suite-<timestamp>/suite.json`` and ``suite.md``, and return the exit code:
    :func:`mcpsim.report.exit_code` over the suite, or 1 when a scenario could not run.

    A scenario that fails to load or to run (server unreachable, planner gave up) is reported
    and the suite goes on. ``on_row`` is called after each scenario (the UI's progress hook).
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    loaded = load_skill(skill)
    entries = suite_entries(scenario_dir, skill=loaded, names=names, categories=categories)
    out = FsPath(out_dir) if out_dir is not None else loaded.runs_dir()
    _log(f"suite: {len(entries)} scenario(s) into {out} (skill {loaded.path})")
    rows: list[SuiteRow] = []
    for entry in entries:
        row = SuiteRow(entry=entry)
        if entry.error is not None:
            row.error = entry.error
        else:
            try:
                run_dir = run_scenario(
                    entry.file,
                    out,
                    dry_run=dry_run,
                    model_overrides=model_overrides,
                    allow_same_judge=allow_same_judge,
                    skill=loaded,
                    repeat=repeat,
                    modes=modes,
                )
                row.run_dir = FsPath(run_dir)
                row.report = Report.load(row.run_dir / REPORT_JSON)
            except Exception as exc:  # noqa: BLE001 - one scenario's failure is a row, not the end
                # Besides the user-facing errors (a bad scenario, an unreachable server, a planner
                # that gave up) this catches API refusals (an exhausted credit balance answers 400)
                # so the other scenarios still run; MCPSIM_DEBUG=1 re-raises for the traceback.
                if os.environ.get("MCPSIM_DEBUG"):
                    raise
                # The SDK's task groups wrap a failure inside an MCP session in exception
                # groups; the row names the failure itself.
                leaf = sole_leaf(exc)
                if isinstance(leaf, BaseExceptionGroup):
                    row.error = describe_exception(leaf)
                else:
                    message = (
                        leaf.args[0] if isinstance(leaf, KeyError) and leaf.args else str(leaf)
                    )
                    prefix = "" if isinstance(leaf, USER_ERRORS) else f"{type(leaf).__name__}: "
                    row.error = f"{prefix}{message}" if str(message) else type(leaf).__name__
        if row.report is not None:
            print(f"{entry.name}: {summary_line(row.report)}  ({row.run_dir})", flush=True)
        else:
            print(f"{entry.name}: error: {one_line(row.error or 'error')}", flush=True)
        rows.append(row)
        if on_row is not None:
            on_row(row)
    reports = [r.report for r in rows if r.report is not None]
    suite: SuiteReport = aggregate_suite(reports)
    suite.errors = [
        SuiteError(scenario=r.entry.name, file=str(r.entry.file), error=r.error or "")
        for r in rows
        if r.report is None
    ]
    suite_dir = out / f"suite-{timestamp()}"
    n = 1
    while suite_dir.exists():
        suite_dir = out / f"suite-{timestamp()}-{n}"
        n += 1
    suite_dir.mkdir(parents=True)
    suite.save(suite_dir / SUITE_JSON)
    (suite_dir / SUITE_MD).write_text(render_suite_markdown(suite), encoding="utf-8")
    print()
    print(format_suite_table(rows))
    print()
    print(f"suite: {summary_line(suite)}")
    print(f"suite report: {suite_dir / SUITE_MD}")
    code = report_exit_code(suite, threshold)
    return 1 if suite.errors and code == 0 else code
