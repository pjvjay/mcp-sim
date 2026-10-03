"""``mcpsim`` command line (DESIGN §2 "CLI").

``catalog`` works on its own; ``config`` prints what the simulate skill resolves to
(:mod:`mcpsim.skill`). ``plan``, ``run``, ``judge``, ``report`` and ``suite`` call into
``mcpsim.runner``, which is imported lazily inside each subcommand so this module imports (and
``mcpsim catalog`` works) before the runner exists; until it does, those subcommands print
"not yet wired" and exit 2. ``ui`` serves the local test runner (:mod:`mcpsim.ui`), which
starts runs as ``mcpsim run`` subprocesses; it imports Starlette and uvicorn lazily too.

Runner contract
===============

``mcpsim/runner.py`` must define exactly these **synchronous** functions (they may wrap
``asyncio.run`` internally). The CLI calls them with keyword arguments for everything after
the positional path(s), so keyword names are part of the contract::

    def plan_scenario(
        scenario_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
        *,
        dry_run: bool = False,
        model_overrides: Mapping[str, str] | None = None,  # --models, only when given
        allow_same_judge: bool = False,                    # --allow-same-judge, only when given
        skill: str | None = None,                          # --skill, only when given
    ) -> pathlib.Path:
        '''Load the scenario, discover the catalog, plan (LLM; or the one-path dry-run plan
        when dry_run), write <out_dir>/<scenario.name>/<timestamp>/plan.json and return
        that file's path.'''

    def run_scenario(
        scenario_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
        *,
        plan_path: str | os.PathLike[str] | None = None,  # reuse this plan.json, do not plan
        only_path: str | None = None,                      # run only this path id
        repeat: int | None = None,                         # override scenario.repeat
        mode: Literal["guided", "free"] | None = None,     # run only this mode
        dry_run: bool = False,
        model_overrides: Mapping[str, str] | None = None,  # --models, only when given
        allow_same_judge: bool = False,                    # --allow-same-judge, only when given
        skill: str | None = None,                          # --skill, only when given
    ) -> pathlib.Path:
        '''plan -> runs -> judge -> report. Returns the run directory
        <out_dir>/<scenario.name>/<timestamp>/ holding plan.json, scenario.json (the validated
        scenario, so judge_run_dir/report_run_dir need only the directory),
        transcripts/<path>-<mode>-<i>.jsonl, verdicts/<same>.json, report.json and report.md
        (mcpsim.report.aggregate / render_markdown). In dry run the LLM judge is skipped and
        verdicts come from the matcher alone with judge_model "dry-run".'''

    def judge_run_dir(
        run_dir: str | os.PathLike[str],
        *,
        votes: int | None = None,                          # override scenario.judge_votes
        skill: str | None = None,                          # --skill, only when given
    ) -> list[pathlib.Path]:
        '''Re-judge every transcripts/*.jsonl in the run directory (scenario and plan are read
        from scenario.json and plan.json there), rewrite verdicts/*.json and report.json /
        report.md, and return the verdict paths written.'''

    def report_run_dir(
        run_dir: str | os.PathLike[str],
    ) -> tuple[pathlib.Path, pathlib.Path]:
        '''Rebuild report.json and report.md from verdicts/ and transcripts/; return
        (report.json path, report.md path).'''

    def run_suite(
        scenario_dir: str | os.PathLike[str] | None,       # positional, None: config.yaml's
        out_dir: str | os.PathLike[str] | None,            # --out, None: config.yaml's runs_dir
        *,
        threshold: float = 1.0,
        dry_run: bool = False,
        model_overrides: Mapping[str, str] | None = None,  # --models, only when given
        allow_same_judge: bool = False,                    # --allow-same-judge, only when given
        skill: str | None = None,                          # --skill, only when given
        names: list[str] | None = None,                    # --name GLOB (repeatable), when given
        categories: list[str] | None = None,               # --category GLOB (repeatable)
        repeat: int | None = None,                         # --repeat, only when given
        modes: list[str] | None = None,                    # --modes, only when given
    ) -> int:
        '''Run every selected scenario (the directory's *.yaml / *.yml / *.json files, else
        every scenarios entry of the skill's config.yaml, narrowed by the globs) with
        run_scenario, one after another, write <out_dir>/suite-<timestamp>/suite.json and
        suite.md, print one line per scenario and a summary table with pass^k, and return
        mcpsim.report.exit_code(suite_report, threshold) (1 when a scenario could not run).'''

Conventions the CLI relies on:

* ``dry_run`` is resolved here from ``--dry-run`` or ``MCPSIM_DRY_RUN=1`` (also ``true`` /
  ``yes``); the runner takes the argument and never reads the environment itself.
* The runner raises ``ScenarioError`` / ``MCPClientError`` / ``ValueError`` / ``RuntimeError``
  / ``KeyError`` / ``OSError`` for user-facing failures (bad scenario, server unreachable,
  unknown ``--only-path``, planner gave up, ...). The CLI prints them as one line on stderr and
  exits 1; set ``MCPSIM_DEBUG=1`` to get the traceback instead. Anything else propagates.
* ``run``, ``judge`` and ``report`` take ``--threshold`` (default 1.0) and exit with
  ``mcpsim.report.exit_code`` over the run directory's ``report.json``; ``suite`` returns the
  runner's exit code.
* ``plan``, ``run`` and ``suite`` take ``--models key=value[,key=value]`` (keys ``planner`` /
  ``agent`` / ``judge`` / ``user`` / ``observer``, values ``provider:model`` such as
  ``ollama:command-r7b``; see docs/LOCAL_MODELS.md) and ``--allow-same-judge``; ``plan``,
  ``run``, ``judge``, ``suite`` and ``config`` take ``--skill DIR``. The CLI passes these
  keywords **only when the flag was given**, so a plain call keeps the exact keyword set above;
  the runner resolves them through the skill (:meth:`mcpsim.skill.Skill.apply`: ``--models`` is
  the top of the model precedence) so scenario files stay provider-neutral.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path as FsPath
from typing import Any

from mcpsim import __version__
from mcpsim.mcpclient import Catalog, MCPClientError, connect
from mcpsim.report import Report, exit_code, render_markdown, summary_line
from mcpsim.scenario import MODEL_ROLES, Scenario, ScenarioError, load_scenario
from mcpsim.skill import SKILL_ENV, Skill, SkillError, load_skill, one_line

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

DRY_RUN_ENV = "MCPSIM_DRY_RUN"
DEBUG_ENV = "MCPSIM_DEBUG"
RUNNER_MODULE = "mcpsim.runner"

_USER_ERRORS: tuple[type[Exception], ...] = (
    ScenarioError,
    SkillError,
    MCPClientError,
    OSError,
    ValueError,
    KeyError,
    RuntimeError,
)


def render_catalog(catalog: Catalog) -> str:
    """Plain-text listing of tools, resources, templates and prompts."""
    lines: list[str] = []
    title = f"server: {catalog.server_name}" if catalog.server_name else "server"
    lines.append(f"{title}  (catalog digest {catalog.digest()[:12]})")
    lines.append(f"tools ({len(catalog.tools)}):")
    for tool in catalog.tools:
        props = tool.input_schema.get("properties", {}) if tool.input_schema else {}
        required = set(tool.input_schema.get("required", [])) if tool.input_schema else set()
        args = ", ".join(
            f"{name}{'' if name in required else '?'}: {schema.get('type', 'any')}"
            for name, schema in props.items()
            if isinstance(schema, dict)
        )
        first_line = tool.description.strip().splitlines()[0] if tool.description.strip() else ""
        lines.append(f"  - {tool.name}({args})")
        if first_line:
            lines.append(f"      {first_line}")
    lines.append(f"resources ({len(catalog.resources)}):")
    for res in catalog.resources:
        lines.append(f"  - {res.uri}  [{res.name}]")
    lines.append(f"resource templates ({len(catalog.resource_templates)}):")
    for tmpl in catalog.resource_templates:
        lines.append(f"  - {tmpl.uri_template}  [{tmpl.name}]")
    lines.append(f"prompts ({len(catalog.prompts)}):")
    for prompt in catalog.prompts:
        arg_names = ", ".join(str(a.get("name", "?")) for a in prompt.arguments)
        lines.append(f"  - {prompt.name}({arg_names})")
    return "\n".join(lines)


async def _discover(scenario: Scenario) -> Catalog:
    async with connect(scenario.server) as session:
        return await session.catalog()


def cmd_catalog(args: argparse.Namespace) -> int:
    try:
        scenario = load_scenario(args.scenario)
    except ScenarioError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FAILURE
    try:
        catalog = asyncio.run(_discover(scenario))
    except MCPClientError as exc:
        print(f"mcpsim: cannot connect to server: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    if args.json:
        print(json.dumps(catalog.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        print(render_catalog(catalog))
    return EXIT_OK


def dry_run_requested(flag: bool = False, env: Mapping[str, str] | None = None) -> bool:
    """``--dry-run`` or ``MCPSIM_DRY_RUN`` set to 1/true/yes (case-insensitive)."""
    source = os.environ if env is None else env
    value = source.get(DRY_RUN_ENV, "").strip().lower()
    return bool(flag) or value in {"1", "true", "yes"}


def parse_model_overrides(text: str) -> dict[str, str]:
    """``planner=ollama:command-r7b,user=ollama:llama3.2:3b`` -> ``{role: spec}``.

    Keys must be one of ``planner``, ``agent``, ``judge``, ``user``; values are split on the
    first ``=`` only so a ``provider:model:tag`` value survives. Raises ``ValueError``.
    """
    overrides: dict[str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise ValueError(f"--models expects key=value, got {item!r}")
        if key not in MODEL_ROLES:
            raise ValueError(
                f"--models: unknown key {key!r}; expected one of {', '.join(MODEL_ROLES)}"
            )
        overrides[key] = value
    if not overrides:
        raise ValueError("--models expects at least one key=value")
    return overrides


def _models_arg(text: str) -> dict[str, str]:
    try:
        return parse_model_overrides(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``model_overrides`` / ``allow_same_judge`` / ``skill`` keyword arguments, present only
    when given."""
    kwargs: dict[str, Any] = {}
    if getattr(args, "skill", None):
        kwargs["skill"] = args.skill
    given: list[dict[str, str]] = getattr(args, "models", None) or []
    if given:
        merged: dict[str, str] = {}
        for chunk in given:
            merged.update(chunk)
        kwargs["model_overrides"] = merged
    if getattr(args, "allow_same_judge", False):
        kwargs["allow_same_judge"] = True
    return kwargs


def _load_runner(command: str) -> Any | None:
    """Import ``mcpsim.runner`` lazily; ``None`` (after a message) when it does not exist yet."""
    try:
        return importlib.import_module(RUNNER_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == RUNNER_MODULE:
            print(f"mcpsim {command}: not yet wired ({RUNNER_MODULE} is missing)", file=sys.stderr)
            return None
        raise


def _guarded(command: str, body: Callable[[], int]) -> int:
    """Run a subcommand body, turning user-facing errors into one stderr line and exit 1."""
    try:
        return body()
    except _USER_ERRORS as exc:
        if os.environ.get(DEBUG_ENV):
            raise
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        print(f"mcpsim {command}: {message}", file=sys.stderr)
        return EXIT_FAILURE


def _exit_from_report(run_dir: FsPath, threshold: float) -> int:
    """Exit status for a run directory's ``report.json``; 0 when the runner wrote none."""
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        return EXIT_OK
    report = Report.load(report_path)
    print(summary_line(report))
    print(f"report: {run_dir / 'report.md'}")
    return exit_code(report, threshold)


def cmd_plan(args: argparse.Namespace) -> int:
    runner = _load_runner(args.command)
    if runner is None:
        return EXIT_USAGE

    def body() -> int:
        plan_path = runner.plan_scenario(
            args.scenario,
            args.out,
            dry_run=dry_run_requested(args.dry_run),
            **model_kwargs(args),
        )
        print(str(plan_path))
        return EXIT_OK

    return _guarded(args.command, body)


def cmd_run(args: argparse.Namespace) -> int:
    runner = _load_runner(args.command)
    if runner is None:
        return EXIT_USAGE

    def body() -> int:
        run_dir = FsPath(
            runner.run_scenario(
                args.scenario,
                args.out,
                plan_path=args.plan,
                only_path=args.only_path,
                repeat=args.repeat,
                mode=args.mode,
                dry_run=dry_run_requested(args.dry_run),
                **model_kwargs(args),
            )
        )
        print(f"run dir: {run_dir}")
        return _exit_from_report(run_dir, args.threshold)

    return _guarded(args.command, body)


def cmd_judge(args: argparse.Namespace) -> int:
    runner = _load_runner(args.command)
    if runner is None:
        return EXIT_USAGE

    def body() -> int:
        extra = {"skill": args.skill} if args.skill else {}
        written = list(runner.judge_run_dir(args.run_dir, votes=args.votes, **extra))
        print(f"judged {len(written)} transcript(s) in {args.run_dir}")
        return _exit_from_report(FsPath(args.run_dir), args.threshold)

    return _guarded(args.command, body)


def cmd_report(args: argparse.Namespace) -> int:
    runner = _load_runner(args.command)
    if runner is None:
        return EXIT_USAGE

    def body() -> int:
        json_path, md_path = runner.report_run_dir(args.run_dir)
        report = Report.load(json_path)
        if args.markdown:
            print(render_markdown(report), end="")
        else:
            print(summary_line(report))
        print(f"report: {md_path}")
        return exit_code(report, args.threshold)

    return _guarded(args.command, body)


def suite_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``names`` / ``categories`` / ``repeat`` / ``modes``, present only when given."""
    kwargs: dict[str, Any] = {}
    if args.name:
        kwargs["names"] = list(args.name)
    if args.category:
        kwargs["categories"] = list(args.category)
    if args.repeat is not None:
        kwargs["repeat"] = args.repeat
    if args.modes:
        kwargs["modes"] = list(args.modes)
    return kwargs


def cmd_suite(args: argparse.Namespace) -> int:
    if args.list:
        return _guarded(args.command, lambda: list_suite(args))
    runner = _load_runner(args.command)
    if runner is None:
        return EXIT_USAGE

    def body() -> int:
        code = runner.run_suite(
            args.scenario_dir,
            args.out,
            threshold=args.threshold,
            dry_run=dry_run_requested(args.dry_run),
            **model_kwargs(args),
            **suite_kwargs(args),
        )
        return int(code)

    return _guarded(args.command, body)


def list_suite(args: argparse.Namespace) -> int:
    """``suite --list``: the scenarios the suite would run, with their resolved models and run
    settings, without running anything."""
    from mcpsim.runner import suite_entries

    skill = load_skill(args.skill)
    entries = suite_entries(
        args.scenario_dir,
        skill=skill,
        names=args.name or None,
        categories=args.category or None,
    )
    if args.json:
        rows = []
        for entry in entries:
            row = entry.to_json()
            if entry.scenario is not None:
                row["resolved"] = skill.resolve(
                    entry.scenario,
                    model_overrides=_merged_models(args),
                    run_overrides=_run_overrides(args),
                ).to_json()
            rows.append(row)
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    width = max(len(e.name) for e in entries)
    for entry in entries:
        if entry.error is not None:
            print(f"{entry.name.ljust(width)}  error: {one_line(entry.error)}")
            continue
        assert entry.scenario is not None
        resolved = skill.resolve(
            entry.scenario, model_overrides=_merged_models(args), run_overrides=_run_overrides(args)
        )
        run = resolved.run
        print(
            f"{entry.name.ljust(width)}  [{entry.category}] repeat {run['repeat'].value}, "
            f"modes {'+'.join(run['modes'].value)}, judge_votes {run['judge_votes'].value}, "
            f"judge {resolved.models['judge'].value}  ({entry.file})"
        )
    return EXIT_OK


def _merged_models(args: argparse.Namespace) -> dict[str, str] | None:
    given: list[dict[str, str]] = getattr(args, "models", None) or []
    merged: dict[str, str] = {}
    for chunk in given:
        merged.update(chunk)
    return merged or None


def _run_overrides(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if getattr(args, "repeat", None) is not None:
        out["repeat"] = args.repeat
    if getattr(args, "modes", None):
        out["modes"] = list(args.modes)
    return out


def render_config(info: dict[str, Any]) -> str:
    """Plain-text ``mcpsim config``: the skill, each role's model (and where it came from),
    settings and prompts, the run settings, the scenario sources, runs_dir and overrides."""
    lines = [
        f"skill: {info['skill']['name']}  ({info['skill']['path']})",
        f"config: {info['config']}",
    ]
    if info.get("scenario"):
        lines.append(f"resolved for scenario: {info['scenario']}")
    lines += ["", "roles:"]
    for name, role in info["roles"].items():
        model = role.get("model")
        if model is not None:
            head = f"  {name}: {model}  (from {role['model_source']})"
        else:
            head = f"  {name}: {role['frontmatter_model']}  ({role['summary']})"
        lines.append(head)
        settings = [
            f"{key} {role[key]}"
            for key in ("max_tokens", "temperature", "votes", "policy_max_tokens")
            if role.get(key) is not None
        ]
        prompts = ", ".join(role["prompts"])
        lines.append(f"      {'; '.join(settings) or 'defaults'}; prompts: {prompts}")
        lines.append(f"      {role['file']}")
    lines += ["", "run:"]
    for key, setting in info["run"].items():
        value = setting["value"]
        shown = ", ".join(value) if isinstance(value, list) else value
        lines.append(f"  {key}: {shown}  (from {setting['source']})")
    lines += ["", "scenarios:"]
    for source in info["scenarios"]:
        where = source["path"] or "-"
        count = f"{len(source['files'])} file(s)"
        note = f"; {source['note']}" if source["note"] else ""
        lines.append(f"  {source['entry']} -> {where}: {count}{note}")
    lines += ["", f"runs_dir: {info['runs_dir']}"]
    if info["overrides"]:
        lines += ["", "overrides:"]
        for i, override in enumerate(info["overrides"], start=1):
            match = ", ".join(f"{k}={v}" for k, v in override["match"].items() if v)
            models = ", ".join(f"{k}={v}" for k, v in override["models"].items())
            run = ", ".join(f"{k}={v}" for k, v in override["run"].items() if v is not None)
            parts = [f"models {models}"] if models else []
            parts += [f"run {run}"] if run else []
            lines.append(f"  {i}. match {match}: {'; '.join(parts) or '(nothing)'}")
    return "\n".join(lines)


def cmd_config(args: argparse.Namespace) -> int:
    def body() -> int:
        skill = load_skill(args.skill)
        scenario = None
        if args.scenario:
            scenario = _find_scenario(skill, args.scenario)
        info = skill.describe(scenario=scenario)
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(render_config(info))
        return EXIT_OK

    return _guarded(args.command, body)


def _find_scenario(skill: Skill, name_or_file: str) -> Scenario:
    """A scenario file path, or the name of a configured scenario."""
    if FsPath(name_or_file).is_file():
        return load_scenario(name_or_file)
    for entry in skill.entries():
        if entry.name == name_or_file:
            if entry.scenario is None:
                raise ValueError(f"scenario {name_or_file!r} does not load: {entry.error}")
            return entry.scenario
    raise ValueError(
        f"no configured scenario named {name_or_file!r} (and no such file); "
        "`mcpsim suite --list` shows the names"
    )


def _skill_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skill",
        metavar="DIR",
        help=f"the simulate skill directory (default: ${SKILL_ENV}, else the packaged one)",
    )


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from mcpsim.ui.app import resolve_settings, serve
    except ModuleNotFoundError as exc:  # starlette / uvicorn come with the mcp SDK
        print(f"mcpsim ui: missing dependency: {exc.name}", file=sys.stderr)
        return EXIT_USAGE
    settings = resolve_settings(
        skill=args.skill,
        runs=args.runs,
        scenarios=args.scenarios,
        allow_remote=args.allow_remote,
    )
    try:
        serve(settings, host=args.host, port=args.port)
    except ValueError as exc:
        print(f"mcpsim ui: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def _model_args(parser: argparse.ArgumentParser) -> None:
    _skill_arg(parser)
    parser.add_argument(
        "--models",
        action="append",
        type=_models_arg,
        metavar="key=value[,key=value]",
        help=(
            "override the scenario's models for this run; keys planner|agent|judge|user|observer, "
            "values provider:model (e.g. ollama:command-r7b; a bare name is an Anthropic model)"
        ),
    )
    parser.add_argument(
        "--allow-same-judge",
        action="store_true",
        help="let the judge use the same model as the agent (read verdicts as a first pass)",
    )


def _threshold_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="minimum overall pass rate (0-1) for exit code 0 (default: 1.0)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcpsim",
        description="LLM-as-a-judge simulations for MCP servers.",
        epilog=f"Set {DRY_RUN_ENV}=1 for the no-LLM smoke mode (same as --dry-run).",
    )
    parser.add_argument("--version", action="version", version=f"mcpsim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("catalog", help="print what the scenario's server exposes")
    p.add_argument("scenario", help="scenario YAML/JSON file")
    p.add_argument("--json", action="store_true", help="emit the catalog as JSON")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("plan", help="plan paths for a scenario and save plan.json")
    p.add_argument("scenario", help="scenario YAML/JSON file")
    p.add_argument("--out", default="runs", help="output root (default: runs)")
    p.add_argument("--dry-run", action="store_true", help="no LLM; one happy path from catalog")
    _model_args(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="plan, run, judge and report a scenario")
    p.add_argument("scenario", help="scenario YAML/JSON file")
    p.add_argument("--out", default="runs", help="output root (default: runs)")
    p.add_argument("--plan", help="reuse an existing plan.json instead of planning")
    p.add_argument("--only-path", help="run only this path id")
    p.add_argument("--repeat", type=int, help="override the scenario's repeat count")
    p.add_argument("--mode", choices=["guided", "free"], help="run only this mode")
    p.add_argument("--dry-run", action="store_true", help="no LLM; call planned tools, match")
    _model_args(p)
    _threshold_arg(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("judge", help="re-judge the transcripts in a run directory")
    p.add_argument("run_dir", help="a run directory written by `mcpsim run`")
    p.add_argument("--votes", type=int, help="override judge_votes")
    _skill_arg(p)
    _threshold_arg(p)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("report", help="rebuild report.json/report.md for a run directory")
    p.add_argument("run_dir", help="a run directory written by `mcpsim run`")
    p.add_argument("--markdown", action="store_true", help="print report.md to stdout")
    _threshold_arg(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser(
        "suite",
        help="run every configured scenario (or every scenario in a directory)",
        description=(
            "Run the scenarios of the skill's config.yaml (or of SCENARIO_DIR), one after "
            "another, into runs_dir, and print a summary table with pass^k."
        ),
    )
    p.add_argument(
        "scenario_dir",
        nargs="?",
        help="a directory of scenario files (default: the scenarios in the skill's config.yaml)",
    )
    p.add_argument("--out", help="output root (default: runs_dir from the skill's config.yaml)")
    p.add_argument(
        "--name", action="append", metavar="GLOB", help="only scenarios whose name matches "
        "(repeatable; any matches)"
    )
    p.add_argument(
        "--category", action="append", metavar="GLOB", help="only scenarios whose category "
        "matches (repeatable; any matches)"
    )
    p.add_argument("--repeat", type=int, help="runs per path and mode (pass^k's k)")
    p.add_argument(
        "--modes",
        type=_modes_arg,
        metavar="guided,free",
        help="the run modes to keep (default: from config.yaml)",
    )
    p.add_argument(
        "--list", action="store_true", help="list the selected scenarios and their settings"
    )
    p.add_argument("--json", action="store_true", help="with --list: print JSON")
    _threshold_arg(p)
    p.add_argument("--dry-run", action="store_true", help="no LLM; see `run --dry-run`")
    _model_args(p)
    p.set_defaults(func=cmd_suite)

    p = sub.add_parser(
        "config", help="print the simulate skill's resolved roles, models and run settings"
    )
    _skill_arg(p)
    p.add_argument(
        "--scenario", metavar="NAME|FILE", help="resolve for this scenario (its overrides too)"
    )
    p.add_argument("--json", action="store_true", help="print JSON")
    p.set_defaults(func=cmd_config)
    p = sub.add_parser("ui", help="serve the local test runner (scenarios, runs, transcripts)")
    p.add_argument(
        "--skill",
        help="simulate skill directory (default: $MCPSIM_SKILL, else the packaged skill)",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    p.add_argument(
        "--runs", help="runs directory (default: the skill config's runs_dir, else runs/)"
    )
    p.add_argument(
        "--scenarios",
        action="append",
        metavar="DIR_OR_GLOB",
        help="scenario directory, file or glob; repeatable (default: the skill config's "
        "scenarios, else scenarios/)",
    )
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow a non-loopback --host (anyone who can reach it can read runs and start them)",
    )
    p.set_defaults(func=cmd_ui)
    return parser


def _modes_arg(text: str) -> list[str]:
    modes = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in modes if m not in ("guided", "free")]
    if not modes or bad:
        raise argparse.ArgumentTypeError(
            f"--modes expects guided, free or guided,free; got {text!r}"
        )
    return modes


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    return int(code)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
