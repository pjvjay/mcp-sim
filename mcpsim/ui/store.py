"""Read-only access to scenario files and run directories for the runner UI (contract B).

Every lookup by name goes through a listing: a scenario name must be one the scan found, a run
id must be a directory listed under ``<runs_dir>/<scenario>/``, and a transcript must be a
``*.jsonl`` file listed under that run's ``transcripts/``. Each resolved path must also stay
inside the runs directory after symlinks are followed. Nothing else on disk is ever read
through the API.

Scenario files are loaded with the scenario model (:mod:`mcpsim.ui.scenario_view`) from the
sources the skill's ``config.yaml`` names (or ``mcpsim ui --scenarios``), expanded the way
``mcpsim suite`` expands them (:func:`mcpsim.skill.expand_source`). Run artefacts (reports,
verdicts, transcripts) are read as plain JSON, so a run directory written by an older or newer
runner still shows.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path as FsPath
from typing import Any

from mcpsim.scenario import Scenario
from mcpsim.skill import expand_source
from mcpsim.ui.scenario_view import ScenarioView, load_view, snapshot_view

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TRANSCRIPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,250}\.jsonl$")
VERDICT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,250}\.json$")
STAMP_RE = re.compile(r"^(\d{8}T\d{6}Z)(?:-(\d+))?$")
MAX_FILE_BYTES = 32 * 1024 * 1024
STATUSES: tuple[str, ...] = ("passed", "failed", "partial", "running", "never")


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def run_started(run_id: str) -> datetime | None:
    """``20261002T120836Z`` (optionally ``-N``) -> that UTC time; anything else -> ``None``."""
    m = STAMP_RE.match(run_id)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def run_sort_key(run_id: str) -> tuple[str, int, str]:
    """Order run ids by time: ``<stamp>`` < ``<stamp>-1`` < ``<stamp>-2`` < ``<stamp>-10``."""
    m = STAMP_RE.match(run_id)
    if not m:
        return (run_id, 0, run_id)
    return (m.group(1), int(m.group(2) or 0), run_id)


def read_json(path: FsPath) -> Any:
    """Parsed JSON, or ``None`` when the file is missing, too large or not JSON."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def read_events(path: FsPath) -> list[dict[str, Any]]:
    """A transcript's events; a line that is not a JSON object becomes ``kind: unparsed``."""
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        if isinstance(event, dict):
            events.append(event)
        else:
            events.append({"kind": "unparsed", "raw": line[:2000]})
    return events


def split_stem(stem: str) -> tuple[str, str, int | None]:
    """``happy-dry-run-guided-0`` -> ``("happy-dry-run", "guided", 0)``; path ids may hold ``-``."""
    parts = stem.rsplit("-", 2)
    if len(parts) == 3 and parts[2].isdigit():
        return parts[0], parts[1], int(parts[2])
    return stem, "", None


def status_of(runs: int, passed: int) -> str:
    if runs <= 0:
        return "incomplete"
    if passed >= runs:
        return "passed"
    if passed == 0:
        return "failed"
    return "partial"


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def transcript_facts(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Outcome, cost and wall time of one transcript, from its own events."""
    system = next((e for e in events if e.get("kind") == "system"), {})
    end = next((e for e in reversed(events) if e.get("kind") == "end"), None)
    usage = next((e for e in reversed(events) if e.get("kind") == "usage"), None)
    times = [t for t in (parse_time(e.get("t")) for e in events) if t is not None]
    flags: list[str] = []
    failures: list[str] = []
    for e in events:
        if e.get("kind") == "informant_report":
            flags += [str(f) for f in e.get("flags") or [] if str(f) not in flags]
            failures += [str(f) for f in e.get("failures") or [] if str(f) not in failures]
    return {
        "outcome": end.get("outcome") if end else None,
        "reason": end.get("reason", "") if end else "transcript has no end event",
        "cost_usd": _num(usage.get("cost_usd")) if usage else None,
        "duration_s": round((max(times) - min(times)).total_seconds(), 3) if times else None,
        "first_t": min(times).isoformat() if times else None,
        "last_t": max(times).isoformat() if times else None,
        "events": len(events),
        "models": system.get("models") if isinstance(system.get("models"), dict) else {},
        "flags": flags,
        "hard_failures": failures,
    }


@dataclass
class ScenarioIndex:
    views: dict[str, ScenarioView] = field(default_factory=dict)
    paths: dict[str, FsPath] = field(default_factory=dict)
    # The validated scenario of every view without an error (as freshly loaded from its file,
    # so Skill.resolve sees which models the file names itself).
    scenarios: dict[str, Scenario] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Store:
    """Scenario scan and run-directory reads, all confined to the configured directories."""

    def __init__(
        self,
        *,
        scenario_sources: Callable[[], list[str]],
        runs_dir: FsPath,
        cwd: FsPath | None = None,
    ) -> None:
        # Called on every scan, so an edit to config.yaml's scenarios shows without a restart.
        self.scenario_sources = scenario_sources
        self.runs_dir = runs_dir
        self.cwd = cwd or FsPath.cwd()
        # Loaded files, reused while the file's (mtime, size) is unchanged.
        self._views: dict[FsPath, tuple[tuple[int, int], ScenarioView, Scenario | None]] = {}
        self._views_lock = threading.Lock()

    # --- scenarios --------------------------------------------------------------------------

    def _view(self, path: FsPath, fresh: bool) -> tuple[ScenarioView, Scenario | None]:
        try:
            st = path.stat()
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            key = (-1, -1)
        with self._views_lock:
            cached = self._views.get(path)
        if not fresh and cached is not None and cached[0] == key:
            return dataclasses.replace(cached[1]), cached[2]
        view, scenario = load_view(path, display_path=self._display(path))
        with self._views_lock:
            self._views[path] = (key, view, scenario)
        return dataclasses.replace(view), scenario

    def files(self) -> tuple[list[FsPath], list[str]]:
        """The scenario files the sources name, expanded as ``mcpsim suite`` expands them
        (relative to the working directory; ``$NAME`` / ``${NAME:-default}`` read the
        environment), deduplicated; plus a note per source that matched nothing."""
        files: list[FsPath] = []
        seen: set[FsPath] = set()
        notes: list[str] = []
        for entry in self.scenario_sources():
            source = expand_source(entry, base=self.cwd)
            if source.note:
                where = f" ({source.path})" if source.path else ""
                notes.append(f"scenarios entry {entry!r}{where}: {source.note}")
            for file in source.files:
                resolved = file.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(resolved)
        return files, notes

    def scan(self, fresh: bool = False) -> ScenarioIndex:
        """Every scenario file the sources name. A file whose modification time and size are
        unchanged is not loaded again unless ``fresh`` (the scenario list always is: an
        ``agent.skill`` the file points at may have changed)."""
        index = ScenarioIndex()
        files, warnings = self.files()
        index.warnings.extend(warnings)
        for path in files:
            view, scenario = self._view(path, fresh)
            if not NAME_RE.match(view.name):
                safe = re.sub(r"[^A-Za-z0-9._-]", "-", path.stem).lstrip("-._") or "scenario"
                view.error = (view.error + "\n" if view.error else "") + (
                    f"scenario name {view.name!r} is not a valid name"
                )
                view.name = safe
            if view.name in index.views:
                index.warnings.append(
                    f"{view.file}: scenario name {view.name!r} is also used by "
                    f"{index.views[view.name].file}; the first one is shown"
                )
                continue
            index.views[view.name] = view
            index.paths[view.name] = path
            if view.error is None and scenario is not None:
                index.scenarios[view.name] = scenario
        return index

    def _display(self, path: FsPath) -> str:
        try:
            return str(path.relative_to(self.cwd.resolve()))
        except ValueError:
            return str(path)

    # --- runs -------------------------------------------------------------------------------

    def _root(self) -> FsPath:
        return self.runs_dir.resolve()

    def _inside(self, path: FsPath) -> bool:
        try:
            return path.resolve().is_relative_to(self._root())
        except OSError:
            return False

    def scenario_runs_dir(self, name: str) -> FsPath | None:
        if not NAME_RE.match(name):
            return None
        folder = self.runs_dir / name
        if not folder.is_dir() or not self._inside(folder):
            return None
        return folder

    def run_ids(self, name: str) -> list[str]:
        """Run ids for a scenario, newest first (ids are UTC stamps; see :func:`run_sort_key`)."""
        folder = self.scenario_runs_dir(name)
        if folder is None:
            return []
        ids = [
            p.name
            for p in folder.iterdir()
            if RUN_ID_RE.match(p.name) and p.is_dir() and self._inside(p)
        ]
        return sorted(ids, key=run_sort_key, reverse=True)

    def run_dir(self, name: str, run_id: str) -> FsPath | None:
        if not RUN_ID_RE.match(run_id) or run_id not in self.run_ids(name):
            return None
        folder = self.scenario_runs_dir(name)
        return folder / run_id if folder is not None else None

    def _listed(self, folder: FsPath, pattern: re.Pattern[str]) -> list[FsPath]:
        if not folder.is_dir() or not self._inside(folder):
            return []
        return sorted(
            p for p in folder.iterdir() if pattern.match(p.name) and p.is_file() and self._inside(p)
        )

    def transcript_files(self, run_dir: FsPath) -> list[FsPath]:
        return self._listed(run_dir / "transcripts", TRANSCRIPT_RE)

    def verdicts(self, run_dir: FsPath) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for path in self._listed(run_dir / "verdicts", VERDICT_RE):
            data = read_json(path)
            if isinstance(data, dict):
                found[path.stem] = data
        return found

    def _small_json(self, run_dir: FsPath, name: str) -> Any:
        path = run_dir / name
        return read_json(path) if self._inside(path) else None

    def run_summary(
        self,
        name: str,
        run_id: str,
        run_dir: FsPath,
        *,
        verdicts: dict[str, dict[str, Any]] | None = None,
        facts: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Status, pass rate, pass^k, cost and duration of one run directory."""
        report = self._small_json(run_dir, "report.json")
        report = report if isinstance(report, dict) else None
        verdicts = self.verdicts(run_dir) if verdicts is None else verdicts
        snapshot = self._small_json(run_dir, "scenario.json")
        snapshot = snapshot if isinstance(snapshot, dict) else {}

        if report is not None and isinstance(report.get("runs"), int):
            runs = int(report["runs"])
            passed = int(report.get("passed") or 0)
        else:
            runs = len(verdicts)
            passed = sum(1 for v in verdicts.values() if v.get("passed") is True)
        scores = [s for s in (_num(v.get("score")) for v in verdicts.values()) if s is not None]
        mean_score = _num(report.get("mean_score")) if report else None
        if mean_score is None and scores:
            mean_score = round(sum(scores) / len(scores), 4)

        pass_k = report.get("pass_k") if report else None
        if not (isinstance(pass_k, dict) and "k" in pass_k):
            pass_k = self._computed_pass_k(verdicts)

        cost = _num(report.get("cost_usd")) if report else None
        duration = _num(report.get("duration_s")) if report else None
        started = run_started(run_id)
        if duration is None and report is not None and started is not None:
            generated = parse_time(report.get("generated_at"))
            if generated is not None and generated >= started:
                duration = round((generated - started).total_seconds(), 3)
        if cost is None or duration is None:
            if facts is None:
                facts = {
                    p.stem: transcript_facts(read_events(p)) for p in self.transcript_files(run_dir)
                }
            if cost is None and facts:
                cost = round(sum(f["cost_usd"] or 0.0 for f in facts.values()), 6)
            if duration is None and facts:
                firsts = [parse_time(f["first_t"]) for f in facts.values()]
                lasts = [parse_time(f["last_t"]) for f in facts.values()]
                starts = [t for t in firsts if t is not None]
                ends = [t for t in lasts if t is not None]
                if starts and ends:
                    duration = round((max(ends) - min(starts)).total_seconds(), 3)

        judge_models = report.get("judge_models") if report else None
        if not isinstance(judge_models, list):
            judge_models = sorted({str(v.get("judge_model")) for v in verdicts.values()})
        models = snapshot.get("models") if isinstance(snapshot.get("models"), dict) else {}
        return {
            "run_id": run_id,
            "scenario": name,
            "started_at": started.isoformat() if started else None,
            "status": status_of(runs, passed),
            "runs": runs,
            "passed": passed,
            "pass_rate": round(passed / runs, 4) if runs else None,
            "mean_score": mean_score,
            "pass_k": pass_k,
            "cost_usd": cost,
            "duration_s": duration,
            "judge_models": judge_models,
            "models": models,
            "has_report": report is not None,
            "has_plan": (run_dir / "plan.json").is_file(),
            "dry_run": judge_models == ["dry-run"],
        }

    @staticmethod
    def _computed_pass_k(verdicts: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        """pass^k from the verdicts when ``report.json`` predates it, the way the report computes
        it: ``k`` is the most runs any path x mode cell has, ``all_passed`` that every cell has
        ``k`` runs and every one passed."""
        if not verdicts:
            return None
        cells: dict[tuple[str, str], list[bool]] = {}
        for stem, v in verdicts.items():
            path_id, mode, _ = split_stem(stem)
            key = (str(v.get("path_id", path_id)), str(v.get("mode", mode)))
            cells.setdefault(key, []).append(v.get("passed") is True)
        k = max(len(group) for group in cells.values())
        all_passed = all(len(group) == k and all(group) for group in cells.values())
        return {"k": k, "all_passed": all_passed, "computed": True}

    def in_progress(self, name: str, since: str | None) -> set[str]:
        """Run ids of ``name`` started at or after ``since`` (when its running subprocess
        started) that have no ``report.json`` yet: the directories that run is writing."""
        start = parse_time(since)
        if start is None:
            return set()
        start = start.replace(microsecond=0)
        found: set[str] = set()
        for run_id in self.run_ids(name):
            started = run_started(run_id)
            if started is None or started < start:
                continue
            run_dir = self.run_dir(name, run_id)
            if run_dir is not None and not (run_dir / "report.json").is_file():
                found.add(run_id)
        return found

    def history(self, name: str, running: set[str] | None = None) -> list[dict[str, Any]]:
        """Every run directory's summary, newest first; ``running`` ids get status running."""
        out: list[dict[str, Any]] = []
        for run_id in self.run_ids(name):
            run_dir = self.run_dir(name, run_id)
            if run_dir is not None:
                summary = self.run_summary(name, run_id, run_dir)
                if running and run_id in running:
                    summary["status"] = "running"
                out.append(summary)
        return out

    def last_judged(
        self, name: str, skip: set[str] | None = None
    ) -> tuple[dict[str, Any] | None, int]:
        """The newest run that has a report or verdicts (ignoring ``skip``, the runs still
        being written), and how many run directories exist."""
        ids = self.run_ids(name)
        for run_id in ids:
            run_dir = self.run_dir(name, run_id)
            if run_dir is None or (skip and run_id in skip):
                continue
            report = run_dir / "report.json"
            if (report.is_file() and self._inside(report)) or self._listed(
                run_dir / "verdicts", VERDICT_RE
            ):
                return self.run_summary(name, run_id, run_dir), len(ids)
        return None, len(ids)

    def run_detail(self, name: str, run_id: str, run_dir: FsPath) -> dict[str, Any]:
        verdicts = self.verdicts(run_dir)
        facts: dict[str, dict[str, Any]] = {}
        transcripts: list[dict[str, Any]] = []
        for path in self.transcript_files(run_dir):
            f = transcript_facts(read_events(path))
            facts[path.stem] = f
            transcripts.append(self._transcript_row(path.stem, f, verdicts.get(path.stem)))
        listed = {t["stem"] for t in transcripts}
        for stem, verdict in sorted(verdicts.items()):
            if stem not in listed:
                transcripts.append(self._transcript_row(stem, None, verdict))
        report = self._small_json(run_dir, "report.json")
        plan = self._small_json(run_dir, "plan.json")
        snapshot = self._small_json(run_dir, "scenario.json")
        as_run = None
        if isinstance(snapshot, dict):
            as_run = snapshot_view(snapshot, fallback_name=name).to_json()
        return {
            "summary": self.run_summary(name, run_id, run_dir, verdicts=verdicts, facts=facts),
            "report": report if isinstance(report, dict) else None,
            "plan": plan if isinstance(plan, dict) else None,
            "scenario": as_run,
            "verdicts": verdicts,
            "transcripts": transcripts,
            "has_scout": (run_dir / "scout.json").is_file(),
        }

    @staticmethod
    def _transcript_row(
        stem: str, facts: dict[str, Any] | None, verdict: dict[str, Any] | None
    ) -> dict[str, Any]:
        path_id, mode, index = split_stem(stem)
        row: dict[str, Any] = {
            "stem": stem,
            "file": f"{stem}.jsonl" if facts is not None else None,
            "path_id": verdict.get("path_id", path_id) if verdict else path_id,
            "mode": verdict.get("mode", mode) if verdict else mode,
            "index": verdict.get("index", index) if verdict else index,
            "judged": verdict is not None,
            "passed": verdict.get("passed") if verdict else None,
            "score": verdict.get("score") if verdict else None,
            "goal_achieved": verdict.get("goal_achieved") if verdict else None,
            "sop_followed": verdict.get("sop_followed") if verdict else None,
        }
        if facts is not None:
            row.update(
                {
                    "outcome": facts["outcome"],
                    "reason": facts["reason"],
                    "cost_usd": facts["cost_usd"],
                    "duration_s": facts["duration_s"],
                    "events": facts["events"],
                }
            )
        return row

    def transcript(self, run_dir: FsPath, file: str) -> dict[str, Any] | None:
        if not TRANSCRIPT_RE.match(file):
            return None
        match = next((p for p in self.transcript_files(run_dir) if p.name == file), None)
        if match is None:
            return None
        events = read_events(match)
        verdict_path = run_dir / "verdicts" / f"{match.stem}.json"
        verdict = read_json(verdict_path) if self._inside(verdict_path) else None
        return {
            "file": file,
            "stem": match.stem,
            "events": events,
            "facts": transcript_facts(events),
            "verdict": verdict if isinstance(verdict, dict) else None,
        }

    def runs_root_exists(self) -> bool:
        return self.runs_dir.is_dir()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
