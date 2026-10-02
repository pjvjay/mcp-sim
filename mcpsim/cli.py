"""``mcpsim`` command line (DESIGN §2 "CLI").

``catalog`` works on its own. ``plan``, ``run``, ``judge``, ``report`` and ``suite`` call into
``mcpsim.runner``, which is imported lazily inside each subcommand so this module imports (and
``mcpsim catalog`` works) before the runner exists; until it does, those subcommands print
"not yet wired" and exit 2.

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
        scenario_dir: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
        *,
        threshold: float = 1.0,
        dry_run: bool = False,
        model_overrides: Mapping[str, str] | None = None,  # --models, only when given
        allow_same_judge: bool = False,                    # --allow-same-judge, only when given
    ) -> int:
        '''Run every *.yaml / *.yml / *.json scenario in the directory (sorted by name) with
        run_scenario, write <out_dir>/suite-<timestamp>/suite.json and suite.md
        (mcpsim.report.aggregate_suite / render_suite_markdown), print one summary line per
        scenario, and return mcpsim.report.exit_code(suite_report, threshold).'''

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
  ``agent`` / ``judge`` / ``user``, values ``provider:model`` such as ``ollama:command-r7b``;
  see docs/LOCAL_MODELS.md) and ``--allow-same-judge``. The CLI passes ``model_overrides`` /
  ``allow_same_judge`` **only when the flag was given**, so a plain call keeps the exact keyword
  set above; the runner applies them with ``scenario.model_copy(update=...)`` so scenario files
  stay provider-neutral.
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

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

DRY_RUN_ENV = "MCPSIM_DRY_RUN"
DEBUG_ENV = "MCPSIM_DEBUG"
RUNNER_MODULE = "mcpsim.runner"

_USER_ERRORS: tuple[type[Exception], ...] = (
    ScenarioError,
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
    """``model_overrides`` / ``allow_same_judge`` keyword arguments, present only when given."""
    kwargs: dict[str, Any] = {}
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
        written = list(runner.judge_run_dir(args.run_dir, votes=args.votes))
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


def cmd_suite(args: argparse.Namespace) -> int:
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
        )
        return int(code)

    return _guarded(args.command, body)


def _model_args(parser: argparse.ArgumentParser) -> None:
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
    _threshold_arg(p)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("report", help="rebuild report.json/report.md for a run directory")
    p.add_argument("run_dir", help="a run directory written by `mcpsim run`")
    p.add_argument("--markdown", action="store_true", help="print report.md to stdout")
    _threshold_arg(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("suite", help="run every scenario in a directory")
    p.add_argument("scenario_dir", help="directory of scenario files")
    p.add_argument("--out", default="runs", help="output root (default: runs)")
    _threshold_arg(p)
    p.add_argument("--dry-run", action="store_true", help="no LLM; see `run --dry-run`")
    _model_args(p)
    p.set_defaults(func=cmd_suite)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    return int(code)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
