"""A stand-in for ``python -m mcpsim.cli`` in the runner-UI job tests.

``fake_mcpsim.py run <scenario> [--skill DIR] --out <dir> [--models ..] [--repeat N]
[--mode M | --modes M,N] [--dry-run] [--allow-same-judge]`` behaves like ``mcpsim run`` as far
as the UI can tell: it prints a few progress lines, writes ``<out>/<name>/<stamp>/``
(scenario.json, transcripts, verdicts, report.json, plus ``argv.json`` so a test can see the
arguments), prints ``run dir: <path>`` and exits 0 when every run passed, else 1.

``FAKE_MCPSIM_PLAN`` (JSON: scenario name -> behaviour) picks what happens per scenario:
``pass``, ``fail``, ``partial``, ``error`` (exit 2, no run directory), ``silent`` (exit 0
without a ``run dir:`` line), ``unjudged`` (a run directory with a plan and no verdicts, exit 1)
or ``sleep``: write the run directory's ``plan.json``, start a grandchild process that sleeps
(its pid goes to ``<out>/<name>.child``), then print a line every 50 ms for up to 20 s and pass;
for the conflict, in-progress and cancel tests. ``stall`` writes ``scenario.json`` (repeat 3),
one judged, passing run (``happy-guided-0``) and then prints a line every 50 ms for up to
20 s without ever writing ``report.json``: a run that a cancel or a crash stops part-way.
``orphan`` passes after starting a sleeper in a session of its own that inherits stdout (as
the MCP SDK starts stdio servers); its pid goes to ``<out>/<name>.orphan``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml


def _stamp_dir(out: Path, name: str) -> Path:
    base = out / name
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate, n = base / stamp, 1
    while candidate.exists():
        candidate = base / f"{stamp}-{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="fake-mcpsim")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("scenario")
    p.add_argument("--skill")
    p.add_argument("--out", required=True)
    p.add_argument("--models")
    p.add_argument("--repeat", type=int)
    which = p.add_mutually_exclusive_group()
    which.add_argument("--mode")
    which.add_argument("--modes")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-same-judge", action="store_true")
    args = parser.parse_args(argv)

    data = yaml.safe_load(Path(args.scenario).read_text(encoding="utf-8"))
    name = data["name"]
    plan = json.loads(os.environ.get("FAKE_MCPSIM_PLAN", "{}"))
    behaviour = plan.get(name, "pass")
    print(f"mcpsim: planning {name}", flush=True)
    print(f"mcpsim: skill={args.skill} env={os.environ.get('MCPSIM_SKILL', '')}", flush=True)

    if behaviour == "error":
        print(f"mcpsim run: {args.scenario}: cannot connect to server", file=sys.stderr, flush=True)
        return 2
    if behaviour == "silent":
        return 0

    if behaviour == "orphan":
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        orphan = subprocess.Popen(sleeper, start_new_session=True)
        (Path(args.out) / f"{name}.orphan").write_text(str(orphan.pid), encoding="utf-8")
        behaviour = "pass"

    run_dir = _stamp_dir(Path(args.out), name)
    if behaviour == "unjudged":
        (run_dir / "plan.json").write_text(json.dumps({"paths": []}), encoding="utf-8")
        print(f"run dir: {run_dir}", flush=True)
        return 1
    if behaviour == "stall":
        (run_dir / "scenario.json").write_text(json.dumps({**data, "repeat": 3}), "utf-8")
        for folder in ("verdicts", "transcripts"):
            (run_dir / folder).mkdir()
        verdict = {
            "path_id": "happy", "mode": "guided", "index": 0, "passed": True, "score": 1.0,
            "matches": [], "checklist": [], "failure_reasons": [], "flags": [], "votes": 1,
            "judge_model": "fake-judge",
        }
        (run_dir / "transcripts" / "happy-guided-0.jsonl").write_text(
            json.dumps({"t": "2026-01-01T00:00:00+00:00", "kind": "user", "text": "hi"}) + "\n"
        )
        (run_dir / "verdicts" / "happy-guided-0.json").write_text(json.dumps(verdict))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            print("mcpsim: still judging", flush=True)
            time.sleep(0.05)
        return 1
    if behaviour == "sleep":
        (run_dir / "plan.json").write_text(json.dumps({"paths": []}), encoding="utf-8")
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        (Path(args.out) / f"{name}.child").write_text(str(child.pid), encoding="utf-8")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            print("mcpsim: still running", flush=True)
            time.sleep(0.05)
        child.kill()
        behaviour = "pass"
    (run_dir / "argv.json").write_text(json.dumps(argv), encoding="utf-8")
    (run_dir / "scenario.json").write_text(json.dumps(data), encoding="utf-8")
    repeat = args.repeat or 1
    if args.mode:
        modes = [args.mode]
    else:
        modes = args.modes.split(",") if args.modes else ["guided", "free"]
    verdicts = []
    for mode in modes:
        for i in range(repeat):
            stem = f"happy-{mode}-{i}"
            passed = behaviour == "pass" or (behaviour == "partial" and i == 0 and mode == modes[0])
            verdict = {
                "path_id": "happy",
                "mode": mode,
                "index": i,
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "matches": [],
                "checklist": [],
                "failure_reasons": [] if passed else ["fake failure"],
                "flags": [],
                "votes": 1,
                "judge_model": "fake-judge",
            }
            verdicts.append(verdict)
            (run_dir / "verdicts").mkdir(exist_ok=True)
            (run_dir / "verdicts" / f"{stem}.json").write_text(json.dumps(verdict))
            (run_dir / "transcripts").mkdir(exist_ok=True)
            events = [
                {"t": "2026-01-01T00:00:00+00:00", "kind": "user", "text": "hello"},
                {"t": "2026-01-01T00:00:01+00:00", "kind": "end", "outcome": "completed"},
            ]
            (run_dir / "transcripts" / f"{stem}.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events)
            )
            print(f"mcpsim: judged {stem}: {'pass' if passed else 'fail'}", flush=True)
    passed_n = sum(1 for v in verdicts if v["passed"])
    report = {
        "scenario": name,
        "runs": len(verdicts),
        "passed": passed_n,
        "pass_rate": passed_n / len(verdicts),
        "mean_score": passed_n / len(verdicts),
        "cost_usd": 0.01,
        "judge_models": ["fake-judge"],
        "pass_k": {"k": repeat, "all_passed": passed_n == len(verdicts)},
        "duration_s": 1.5,
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    print(f"run dir: {run_dir}", flush=True)
    return 0 if passed_n == len(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
