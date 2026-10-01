"""``mcpsim`` command line (DESIGN §2 "CLI").

Stage 1 ships the skeleton: ``catalog`` works end to end; ``plan``, ``run``, ``judge``,
``report`` and ``suite`` are parsed but exit 2 with "not yet wired" until the runner lands.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from mcpsim import __version__
from mcpsim.mcpclient import Catalog, MCPClientError, connect
from mcpsim.scenario import Scenario, ScenarioError, load_scenario

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


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


def _not_yet_wired(args: argparse.Namespace) -> int:
    print(f"mcpsim {args.command}: not yet wired", file=sys.stderr)
    return EXIT_USAGE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcpsim",
        description="LLM-as-a-judge simulations for MCP servers.",
    )
    parser.add_argument("--version", action="version", version=f"mcpsim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("catalog", help="print what the scenario's server exposes")
    p.add_argument("scenario", help="scenario YAML/JSON file")
    p.add_argument("--json", action="store_true", help="emit the catalog as JSON")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("plan", help="plan paths for a scenario")
    p.add_argument("scenario")
    p.add_argument("--out", default="runs", help="output root (default: runs)")
    p.add_argument("--dry-run", action="store_true", help="no LLM; one happy path from catalog")
    p.set_defaults(func=_not_yet_wired)

    p = sub.add_parser("run", help="plan, run, judge and report a scenario")
    p.add_argument("scenario")
    p.add_argument("--out", default="runs")
    p.add_argument("--plan", help="reuse an existing plan.json")
    p.add_argument("--only-path", help="run only this path id")
    p.add_argument("--repeat", type=int, help="override the scenario's repeat count")
    p.add_argument("--mode", choices=["guided", "free"], help="run only this mode")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_not_yet_wired)

    p = sub.add_parser("judge", help="re-judge the transcripts in a run directory")
    p.add_argument("run_dir")
    p.add_argument("--votes", type=int, help="override judge_votes")
    p.set_defaults(func=_not_yet_wired)

    p = sub.add_parser("report", help="rebuild report.json/report.md for a run directory")
    p.add_argument("run_dir")
    p.set_defaults(func=_not_yet_wired)

    p = sub.add_parser("suite", help="run every scenario in a directory")
    p.add_argument("scenario_dir")
    p.add_argument("--out", default="runs")
    p.add_argument("--threshold", type=float, default=1.0, help="min overall pass rate")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_not_yet_wired)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    return int(code)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
