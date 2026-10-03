"""Report aggregation and rendering (DESIGN §1 "Report", §2 "Runner").

:func:`aggregate` folds the verdicts and transcripts of one run directory into a
:class:`Report`: pass rate per path × mode, overall pass rate and mean score, **pass^k** (every
one of the ``k`` repeats of every path and mode passed), the expected-behaviour checklist
tallied across runs, cost (runs and judge) and run time, and the worst failures with pointers
to their transcripts. :func:`render_markdown` hand-renders it as
``report.md`` (no template engine, by design). :func:`exit_code` turns a report and a pass-rate
threshold into a process exit status. :func:`aggregate_suite` and
:func:`render_suite_markdown` do the same across scenarios for ``mcpsim suite``.

Nothing here calls an LLM or touches a server; everything is arithmetic over saved artefacts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path as FsPath

from pydantic import BaseModel, ConfigDict, Field

from mcpsim.llm import Usage
from mcpsim.transcript import Transcript, now_iso
from mcpsim.verdict import Verdict

EXIT_PASS = 0
EXIT_FAIL = 1
DEFAULT_MAX_FAILURES = 10
COST_NOTE = (
    "estimate from the rate table; covers the agent, simulated-user and observer calls recorded "
    "in transcripts and the judge calls recorded in verdicts, not the planner"
)


def run_stem(path_id: str, mode: str, index: int) -> str:
    """``<path>-<mode>-<i>``: the file stem shared by a transcript and its verdict."""
    return f"{path_id}-{mode}-{index}"


def transcript_relpath(stem: str) -> str:
    return f"transcripts/{stem}.jsonl"


def verdict_relpath(stem: str) -> str:
    return f"verdicts/{stem}.json"


class CellStats(BaseModel):
    """Pass statistics for one path × mode cell (``cost_usd`` covers its runs and their
    judging; ``duration_s`` sums its runs' durations)."""

    model_config = ConfigDict(extra="forbid")

    path_id: str
    mode: str
    runs: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    mean_score: float = 0.0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    outcomes: dict[str, int] = Field(default_factory=dict)


class PassK(BaseModel):
    """pass^k: ``k`` is the number of repeats per path × mode (the scenario's ``repeat``, or
    ``--repeat``), ``all_passed`` whether every repeat of every path and mode passed — the
    strict reliability measure next to the pass rate."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=0, ge=0)
    all_passed: bool = False


class Tally(BaseModel):
    """How many graded runs met an item."""

    model_config = ConfigDict(extra="forbid")

    passed: int = 0
    graded: int = 0


class BehaviorStats(BaseModel):
    """One checklist item (an expected behaviour, or honesty) tallied across the judged runs."""

    model_config = ConfigDict(extra="forbid")

    item: str
    passed: int = 0
    graded: int = 0


class Failure(BaseModel):
    """One failed run, with where to look."""

    model_config = ConfigDict(extra="forbid")

    path_id: str
    mode: str
    index: int
    score: float
    outcome: str = ""
    reasons: list[str] = Field(default_factory=list)
    transcript: str
    verdict: str

    @property
    def stem(self) -> str:
        return run_stem(self.path_id, self.mode, self.index)


class Report(BaseModel):
    """Aggregation over one scenario's run directory (``report.json``)."""

    model_config = ConfigDict(extra="forbid")

    scenario: str = ""
    generated_at: str = Field(default_factory=now_iso)
    run_dir: str = ""
    runs: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    mean_score: float = 0.0
    pass_k: PassK = Field(default_factory=PassK)
    # Totals: cost_usd = run_cost_usd (transcripts) + judge_cost_usd (verdicts); duration_s
    # sums every run's duration, wall_clock_s spans the first run's start to the last run's end.
    cost_usd: float = 0.0
    run_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    duration_s: float = 0.0
    wall_clock_s: float = 0.0
    cost_note: str = COST_NOTE
    usage: dict[str, Usage] = Field(default_factory=dict)
    judge_models: list[str] = Field(default_factory=list)
    outcomes: dict[str, int] = Field(default_factory=dict)
    goal_achieved: Tally = Field(default_factory=Tally)
    sop_followed: Tally = Field(default_factory=Tally)
    behavior: list[BehaviorStats] = Field(default_factory=list)
    cells: list[CellStats] = Field(default_factory=list)
    worst_failures: list[Failure] = Field(default_factory=list)
    unjudged: list[str] = Field(default_factory=list)

    def save(self, path: str | FsPath) -> FsPath:
        fs_path = FsPath(path)
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        fs_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return fs_path

    @classmethod
    def load(cls, path: str | FsPath) -> Report:
        return cls.model_validate(json.loads(FsPath(path).read_text(encoding="utf-8")))


class ScenarioSummary(BaseModel):
    """One row of a suite report."""

    model_config = ConfigDict(extra="forbid")

    scenario: str
    run_dir: str = ""
    runs: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    mean_score: float = 0.0
    pass_k: PassK = Field(default_factory=PassK)
    cost_usd: float = 0.0
    duration_s: float = 0.0
    worst: str = ""


class SuiteError(BaseModel):
    """A scenario a suite could not run (it did not load, or its run raised)."""

    model_config = ConfigDict(extra="forbid")

    scenario: str
    file: str = ""
    error: str = ""


class SuiteReport(BaseModel):
    """Aggregation across scenarios (``suite.json``); ``errors`` lists the scenarios that could
    not run."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str = Field(default_factory=now_iso)
    scenarios: list[ScenarioSummary] = Field(default_factory=list)
    runs: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    mean_score: float = 0.0
    # Scenarios whose pass^k held (every repeat of every path and mode passed).
    pass_k_scenarios: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    cost_note: str = COST_NOTE
    errors: list[SuiteError] = Field(default_factory=list)

    def save(self, path: str | FsPath) -> FsPath:
        fs_path = FsPath(path)
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        fs_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return fs_path

    @classmethod
    def load(cls, path: str | FsPath) -> SuiteReport:
        return cls.model_validate(json.loads(FsPath(path).read_text(encoding="utf-8")))


def _rate(passed: int, runs: int) -> float:
    return round(passed / runs, 4) if runs else 0.0


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _count(items: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


def failure_reasons(verdict: Verdict) -> list[str]:
    """The judge's reasons, or ones synthesised from failed matches and checklist items."""
    if verdict.failure_reasons:
        return list(verdict.failure_reasons)
    reasons: list[str] = []
    for m in verdict.matches:
        if not m.passed:
            detail = f" ({m.detail})" if m.detail else ""
            reasons.append(
                f"deterministic: {m.path} {m.op} expected {m.expected!r}, actual {m.actual!r}"
                + detail
            )
    for item in verdict.checklist:
        if not item.passed:
            evidence = f": {item.evidence}" if item.evidence else ""
            reasons.append(f"checklist: {item.item}{evidence}")
    if not reasons and not verdict.passed:
        reasons.append("judge failed the run without a recorded reason")
    return reasons


def pass_k(
    verdicts: Sequence[Verdict],
    transcripts: Sequence[Transcript] = (),
    *,
    repeat: int | None = None,
) -> PassK:
    """pass^k over one scenario's runs.

    ``k`` is ``repeat`` (the runs asked for per path × mode: the scenario's ``repeat`` or
    ``--repeat``) when the caller knows it, else the most runs any path × mode cell has.
    ``all_passed`` holds only when every cell (from the verdicts and the transcripts) has at
    least ``k`` runs and every one of them was judged and passed: a run that was recorded but
    never judged, or a repeat that never ran, is not a pass. Nothing run at all is ``k = 0``,
    not passed.
    """
    if repeat is not None and repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    cells: dict[tuple[str, str], dict[int, Verdict | None]] = {}
    for t in transcripts:
        cells.setdefault((t.path_id, t.mode), {}).setdefault(t.index, None)
    for v in verdicts:
        cells.setdefault((v.path_id, v.mode), {})[v.index] = v
    if not cells:
        return PassK(k=0, all_passed=False)
    k = repeat if repeat is not None else max(len(runs) for runs in cells.values())
    all_passed = all(
        len(runs) >= k and all(v is not None and v.passed for v in runs.values())
        for runs in cells.values()
    )
    return PassK(k=k, all_passed=all_passed)


def behavior_stats(verdicts: Sequence[Verdict]) -> list[BehaviorStats]:
    """Every checklist item across the verdicts, in first-seen order, with how many runs met
    it out of how many graded it."""
    stats: dict[str, BehaviorStats] = {}
    for v in verdicts:
        for item in v.checklist:
            row = stats.setdefault(item.item, BehaviorStats(item=item.item))
            row.graded += 1
            row.passed += 1 if item.passed else 0
    return list(stats.values())


def _tally(values: Iterable[bool | None]) -> Tally:
    graded = [v for v in values if v is not None]
    return Tally(passed=sum(1 for v in graded if v), graded=len(graded))


def wall_clock_s(transcripts: Sequence[Transcript]) -> float:
    """Seconds from the earliest run's first event to the latest run's last event."""
    spans = [span for t in transcripts if (span := t.span()) is not None]
    if not spans:
        return 0.0
    start = min(a for a, _ in spans)
    end = max(b for _, b in spans)
    return round(max(0.0, (end - start).total_seconds()), 3)


def aggregate(
    verdicts: Sequence[Verdict],
    transcripts: Sequence[Transcript] = (),
    *,
    scenario: str | None = None,
    run_dir: str | FsPath | None = None,
    max_failures: int = DEFAULT_MAX_FAILURES,
    repeat: int | None = None,
) -> Report:
    """Fold verdicts (and the transcripts they were judged from) into a :class:`Report`.

    Verdicts define the set of runs; transcripts add cost, token usage, duration and outcomes and
    are matched to verdicts by ``<path>-<mode>-<i>``. A transcript without a verdict is listed
    under ``unjudged``, still counts towards cost (the money was spent) and keeps pass^k from
    holding. The judge's cost and usage come from the verdicts. ``repeat`` is the ``k`` of
    pass^k (see :func:`pass_k`; inferred from the runs when not given). Transcript and verdict
    paths in ``worst_failures`` are relative to the run directory, or absolute when ``run_dir``
    is given.
    """
    by_stem: dict[str, Transcript] = {t.stem: t for t in transcripts}
    name = scenario or next((t.scenario for t in transcripts if t.scenario), "")
    base = FsPath(run_dir) if run_dir is not None else None

    def locate(rel: str) -> str:
        return str(base / rel) if base is not None else rel

    ordered = sorted(verdicts, key=lambda v: (v.path_id, v.mode, v.index))
    grouped: dict[tuple[str, str], list[Verdict]] = {}
    for v in ordered:
        grouped.setdefault((v.path_id, v.mode), []).append(v)

    cells: list[CellStats] = []
    for (path_id, mode), group in grouped.items():
        stems = [run_stem(v.path_id, v.mode, v.index) for v in group]
        present = [by_stem[s] for s in stems if s in by_stem]
        cells.append(
            CellStats(
                path_id=path_id,
                mode=mode,
                runs=len(group),
                passed=sum(1 for v in group if v.passed),
                pass_rate=_rate(sum(1 for v in group if v.passed), len(group)),
                mean_score=_mean([v.score for v in group]),
                cost_usd=round(
                    sum(t.cost_usd for t in present) + sum(v.judge_cost_usd for v in group), 6
                ),
                duration_s=round(sum(t.duration_s for t in present), 3),
                outcomes=_count(t.outcome for t in present),
            )
        )

    usage: dict[str, Usage] = {}
    for t in transcripts:
        for model, u in t.usage.items():
            usage[model] = usage.get(model, Usage()) + u
    for v in ordered:
        for model, u in v.judge_usage.items():
            usage[model] = usage.get(model, Usage()) + u

    judged = {run_stem(v.path_id, v.mode, v.index) for v in ordered}
    failures: list[Failure] = []
    failed = sorted(
        (v for v in ordered if not v.passed), key=lambda v: (v.score, v.path_id, v.mode, v.index)
    )
    for v in failed[:max_failures]:
        stem = run_stem(v.path_id, v.mode, v.index)
        tr = by_stem.get(stem)
        reasons = failure_reasons(v)
        if tr is not None and tr.outcome != "completed":
            note = f"outcome {tr.outcome}" + (f": {tr.reason}" if tr.reason else "")
            if note not in reasons:
                reasons.append(note)
        failures.append(
            Failure(
                path_id=v.path_id,
                mode=v.mode,
                index=v.index,
                score=v.score,
                outcome=tr.outcome if tr is not None else "",
                reasons=reasons,
                transcript=locate(transcript_relpath(stem)),
                verdict=locate(verdict_relpath(stem)),
            )
        )

    passed = sum(1 for v in ordered if v.passed)
    run_cost = round(sum(t.cost_usd for t in transcripts), 6)
    judge_cost = round(sum(v.judge_cost_usd for v in ordered), 6)
    return Report(
        scenario=name,
        run_dir=str(base) if base is not None else "",
        runs=len(ordered),
        passed=passed,
        pass_rate=_rate(passed, len(ordered)),
        mean_score=_mean([v.score for v in ordered]),
        pass_k=pass_k(ordered, transcripts, repeat=repeat),
        cost_usd=round(run_cost + judge_cost, 6),
        run_cost_usd=run_cost,
        judge_cost_usd=judge_cost,
        duration_s=round(sum(t.duration_s for t in transcripts), 3),
        wall_clock_s=wall_clock_s(transcripts),
        usage=dict(sorted(usage.items())),
        judge_models=sorted({v.judge_model for v in ordered}),
        outcomes=_count(t.outcome for t in transcripts),
        goal_achieved=_tally(v.goal_achieved for v in ordered),
        sop_followed=_tally(v.sop_followed for v in ordered),
        behavior=behavior_stats(ordered),
        cells=cells,
        worst_failures=failures,
        unjudged=sorted(stem for stem in by_stem if stem not in judged),
    )


def aggregate_suite(reports: Sequence[Report]) -> SuiteReport:
    """Combine per-scenario reports; rates and mean score are weighted by run count."""
    rows: list[ScenarioSummary] = []
    score_sum = 0.0
    for r in sorted(reports, key=lambda r: r.scenario):
        rows.append(
            ScenarioSummary(
                scenario=r.scenario,
                run_dir=r.run_dir,
                runs=r.runs,
                passed=r.passed,
                pass_rate=r.pass_rate,
                mean_score=r.mean_score,
                pass_k=r.pass_k,
                cost_usd=r.cost_usd,
                duration_s=r.duration_s,
                worst=r.worst_failures[0].stem if r.worst_failures else "",
            )
        )
        score_sum += r.mean_score * r.runs
    runs = sum(r.runs for r in reports)
    passed = sum(r.passed for r in reports)
    return SuiteReport(
        scenarios=rows,
        runs=runs,
        passed=passed,
        pass_rate=_rate(passed, runs),
        mean_score=round(score_sum / runs, 4) if runs else 0.0,
        pass_k_scenarios=sum(1 for r in reports if r.pass_k.all_passed),
        cost_usd=round(sum(r.cost_usd for r in reports), 6),
        duration_s=round(sum(r.duration_s for r in reports), 3),
    )


def exit_code(report: Report | SuiteReport, threshold: float = 1.0) -> int:
    """0 when ``passed / runs >= threshold``, else 1. No judged runs is always 1.

    The comparison is done on the counts (``passed >= threshold * runs``) so a threshold such
    as 0.8 with 4 of 5 passes is exactly on the boundary and passes.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    if report.runs == 0:
        return EXIT_FAIL
    return EXIT_PASS if report.passed + 1e-9 >= threshold * report.runs else EXIT_FAIL


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _cell(text: object) -> str:
    """Make a value safe inside a Markdown table cell."""
    return str(text).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def pass_k_text(pk: PassK) -> str:
    """``pass^3 yes`` / ``pass^3 no`` (``pass^k n/a`` with no judged runs)."""
    if pk.k == 0:
        return "pass^k n/a"
    return f"pass^{pk.k} {'yes' if pk.all_passed else 'no'}"


def summary_line(report: Report | SuiteReport) -> str:
    if report.runs == 0:
        return "no judged runs"
    head = f"{report.passed}/{report.runs} runs passed ({_pct(report.pass_rate)})"
    if isinstance(report, Report):
        reliability = pass_k_text(report.pass_k)
    else:
        reliability = (
            f"pass^k held in {report.pass_k_scenarios}/{len(report.scenarios)} scenario(s)"
        )
    return (
        f"{head}, {reliability}, mean score {report.mean_score:.2f}, "
        f"est. cost ${report.cost_usd:.4f}, run time {report.duration_s:.1f}s"
    )


def _pass_k_bullet(report: Report) -> str:
    pk = report.pass_k
    if pk.k == 0:
        return "- pass^k: no judged runs."
    if pk.all_passed:
        return (
            f"- pass^{pk.k}: **yes**: all {pk.k} repeat(s) of every path and mode passed."
        )
    held = sum(1 for c in report.cells if c.runs >= pk.k and c.passed == c.runs)
    return (
        f"- pass^{pk.k}: **no**: {held} of {len(report.cells)} path × mode cell(s) passed all "
        f"{pk.k} repeat(s)."
    )


def render_markdown(report: Report) -> str:
    """Hand-rendered ``report.md``."""
    lines: list[str] = []
    title = f"# mcp-sim report: {report.scenario}" if report.scenario else "# mcp-sim report"
    lines += [title, ""]
    lines.append(f"Generated {report.generated_at}. **{summary_line(report)}**.")
    if report.run_dir:
        lines.append(f"Run directory: `{report.run_dir}`.")
    if report.judge_models:
        lines.append(f"Judge model(s): {', '.join(f'`{m}`' for m in report.judge_models)}.")
    lines += ["", "## Summary", "", _pass_k_bullet(report)]
    if report.goal_achieved.graded:
        g = report.goal_achieved
        lines.append(f"- Goal achieved: {g.passed}/{g.graded} judged run(s).")
    if report.sop_followed.graded:
        o = report.sop_followed
        lines.append(
            f"- Standard operating procedure followed: {o.passed}/{o.graded} judged run(s)."
        )
    lines.append(
        f"- Run time: {report.duration_s:.1f}s across {report.runs} run(s) "
        f"(wall clock {report.wall_clock_s:.1f}s)."
    )
    lines.append(
        f"- Cost: ${report.cost_usd:.4f} (runs ${report.run_cost_usd:.4f}, judge "
        f"${report.judge_cost_usd:.4f})."
    )
    if report.behavior:
        lines += ["", "## Expected behavior", ""]
        lines.append("| item | met | graded | rate |")
        lines.append("| --- | ---: | ---: | ---: |")
        for b in report.behavior:
            lines.append(
                f"| {_cell(b.item)} | {b.passed} | {b.graded} | {_pct(_rate(b.passed, b.graded))} |"
            )
    lines += ["", "## Pass rate by path and mode", ""]
    if not report.cells:
        lines.append("No judged runs.")
    else:
        lines.append(
            "| path | mode | runs | passed | pass rate | mean score | outcomes | time (s) "
            "| cost (USD) |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
        for c in report.cells:
            outcomes = ", ".join(f"{k} {n}" for k, n in c.outcomes.items()) or "-"
            lines.append(
                f"| {_cell(c.path_id)} | {_cell(c.mode)} | {c.runs} | {c.passed} | "
                f"{_pct(c.pass_rate)} | {c.mean_score:.2f} | {_cell(outcomes)} | "
                f"{c.duration_s:.1f} | {c.cost_usd:.4f} |"
            )
        all_outcomes = ", ".join(f"{k} {n}" for k, n in report.outcomes.items()) or "-"
        lines.append(
            f"| **all** | | {report.runs} | {report.passed} | {_pct(report.pass_rate)} | "
            f"{report.mean_score:.2f} | {_cell(all_outcomes)} | {report.duration_s:.1f} | "
            f"{report.cost_usd:.4f} |"
        )
    lines += ["", "## Worst failures", ""]
    if not report.worst_failures:
        lines.append("None: every judged run passed.")
    for i, f in enumerate(report.worst_failures, start=1):
        head = f"{i}. `{f.stem}`: score {f.score:.2f}"
        if f.outcome:
            head += f", outcome `{f.outcome}`"
        lines.append(head)
        for reason in f.reasons:
            lines.append(f"   - {reason}")
        lines.append(f"   - transcript: `{f.transcript}`")
        lines.append(f"   - verdict: `{f.verdict}`")
    if report.unjudged:
        lines += ["", "## Unjudged transcripts", ""]
        for stem in report.unjudged:
            lines.append(f"- `{transcript_relpath(stem)}`")
    lines += ["", "## Cost and usage", ""]
    lines.append(f"Cost is an {report.cost_note}.")
    if report.usage:
        lines += ["", "| model | input tokens | output tokens |", "| --- | ---: | ---: |"]
        for model, u in report.usage.items():
            lines.append(f"| {_cell(model)} | {u.input_tokens} | {u.output_tokens} |")
    return "\n".join(lines) + "\n"


def render_suite_markdown(suite: SuiteReport) -> str:
    """Hand-rendered ``suite.md``."""
    lines: list[str] = ["# mcp-sim suite report", ""]
    lines.append(f"Generated {suite.generated_at}. **{summary_line(suite)}**.")
    lines += ["", "## Scenarios", ""]
    if not suite.scenarios:
        lines.append("No scenarios ran.")
    else:
        lines.append(
            "| scenario | runs | passed | pass rate | pass^k | mean score | worst failure "
            "| time (s) | cost (USD) |"
        )
        lines.append("| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |")
        for s in suite.scenarios:
            worst = f"`{_cell(s.worst)}`" if s.worst else "-"
            lines.append(
                f"| {_cell(s.scenario)} | {s.runs} | {s.passed} | {_pct(s.pass_rate)} | "
                f"{pass_k_text(s.pass_k)} | {s.mean_score:.2f} | {worst} | {s.duration_s:.1f} | "
                f"{s.cost_usd:.4f} |"
            )
        lines.append(
            f"| **all** | {suite.runs} | {suite.passed} | {_pct(suite.pass_rate)} | "
            f"{suite.pass_k_scenarios}/{len(suite.scenarios)} | {suite.mean_score:.2f} | | "
            f"{suite.duration_s:.1f} | {suite.cost_usd:.4f} |"
        )
        lines += ["", "Run directories:", ""]
        for s in suite.scenarios:
            if s.run_dir:
                lines.append(f"- {_cell(s.scenario)}: `{s.run_dir}`")
    if suite.errors:
        lines += ["", "## Scenarios that could not run", ""]
        for e in suite.errors:
            text = "; ".join(x.strip() for x in e.error.splitlines() if x.strip()) or "error"
            lines.append(f"- {_cell(e.scenario)} (`{e.file}`): {_cell(text)}")
    lines += ["", f"Cost is an {suite.cost_note}.", ""]
    return "\n".join(lines)
