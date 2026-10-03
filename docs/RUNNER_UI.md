# Test runner UI (`mcpsim ui`)

`mcpsim ui` serves a local page and a JSON API over the scenario files and the run directories
`mcpsim run` writes. It shows every scenario grouped by `category` with its latest status,
opens any past run, and starts new runs as background `mcpsim run --skill` subprocesses. There
is no build step: the page is `mcpsim/ui/static/index.html` with `app.js` and `app.css`, served
by Starlette and uvicorn (both come with the MCP SDK).

`skills/simulate/scripts/run.sh ui` starts it with the environment the gateway and recipe
scenarios need (the Anthropic key from `.env`, `CONTEXTFORGE_JWT` from `CF_JWT_FILE`,
`RECIPE_SHOPPER_SKILL`); runs started from the page inherit it.

```
mcpsim ui [--skill DIR] [--host 127.0.0.1] [--port 8765] [--runs DIR]
          [--scenarios DIR_OR_GLOB ...] [--allow-remote]
```

| Flag | Default |
| --- | --- |
| `--skill` | `$MCPSIM_SKILL`, else the packaged `skills/simulate` |
| `--runs` | the skill's `config.yaml` `runs_dir` |
| `--scenarios` | the skill's `config.yaml` `scenarios`, re-read on every refresh |
| `--host` | `127.0.0.1`; anything that is not a loopback address needs `--allow-remote` |

Relative paths resolve against the working directory, as for `mcpsim suite`. A skill that does
not load (`mcpsim config` shows why) stops `mcpsim ui` before it serves anything. An edit that
breaks it while the page is open shows as a warning in Settings, and the page keeps the last
version that loaded.

Everything the page shows comes from the code the CLI runs:

- **The skill** through `mcpsim.skill.load_skill`. Settings shows `Skill.describe()`: each role
  file's model and the layer that set it, its temperature, max tokens and prompts, the run
  defaults with their sources, the overrides and the scenario sources. A scenario's detail
  shows `Skill.resolve()` for that scenario: models and run settings, each with its source.
  It also says when the scenario will not run with this skill, for example a role
  temperature that the resolved model rejects.
- **Scenario files** through `mcpsim.scenario.load_scenario`. A file is valid exactly when
  `mcpsim run` would run it; an `agent.skill: env:VAR` whose variable is unset is an error
  here too. The sources expand as `mcpsim suite` expands them: `$NAME` / `${NAME:-default}`,
  and directories without recursion.
- **Runs** as `mcpsim run <file> --skill <dir> --out <runs_dir>`, plus the settings form's
  options: `--models`, `--repeat`, `--mode` (one mode; both modes leave the choice to the
  skill and the scenario), `--dry-run` and `--allow-same-judge`. A run from the page resolves
  exactly as the same command typed in a shell. A run's `scenario.json` is read back through
  the scenario model, as `mcpsim judge` reads it.

## What the page shows

- **Scenario list**: grouped by category, each scenario with a status (passed, failed,
  partial, running, never run), its pass count and pass^k, and per-group counts. Search
  matches the name, title, category and the text of the user instructions; a status filter
  narrows it further. **Run all** runs what the filter shows; each row has its own run button.
- **Detail**: the simulated user's instructions and context (device, location, language,
  details, and whether the agent sees them), the agent's SOP when the scenario has one, the
  expected behaviour with the selected transcript's verdict and quoted evidence per item (and
  how many of the run's graded transcripts met each item), the judge's standing honesty item
  apart from the scenario's own, the judge (verdict, score, goal
  achieved, SOP followed, rationale and failure reasons, matcher table, observer flags, cost),
  the plan's paths, the scenario as it was run (`scenario.json`), and the models and run
  settings the next run will use, each with its source.
- **Run history**: every run directory, newest first, with status, passed / runs, pass^k,
  cost and duration. Click one to open it; a run still being written shows as running.
- **Conversation**: user and agent turns, tool calls with their arguments and result
  (collapsed to a one-line summary), the final result, and the informant reports, tool-set
  changes, goals and usage shown inline in a quieter style.
- **Settings**: the resolved provider, model, temperature and max tokens per role, with the
  layer that set each one, and per-run options sent with every run started from the page:
  repeat, a single mode, dry run, and model overrides per role (`provider:model`).
- **Runs dock**: each job's scenarios with their status and the tail of the `mcpsim` output
  while it runs; a job can be stopped.

## API

| Method and path | Returns |
| --- | --- |
| `GET /api/config` | the skill as `mcpsim config` resolves it: role rows (model, source, settings, prompts), run defaults and sources, overrides, scenario sources, runs dir, warnings |
| `GET /api/scenarios` | every scenario (name, title, category, file, status, pass_rate, pass_k, last_run) and per-category counts |
| `GET /api/scenarios/{name}` | the scenario's v2 view, its resolved models and run settings with their sources, and warnings |
| `GET /api/scenarios/{name}/runs` | run history, newest first |
| `GET /api/runs/{name}/{run_id}` | summary, report, plan, verdicts, transcript list, the run's scenario snapshot |
| `GET /api/runs/{name}/{run_id}/transcripts/{file}` | the transcript's events, facts and verdict |
| `POST /api/run` | `{scenarios: [names] or "all", models?, repeat?, modes?, dry_run?, allow_same_judge?}` -> `{job_id}` |
| `GET /api/jobs`, `GET /api/jobs/{job_id}` | job status (queued, running, done, failed), per-scenario status, log tail |
| `POST /api/jobs/{job_id}/cancel` | stops the job's subprocess and everything it started |

A scenario's status comes from its newest run directory that has verdicts; a job's
per-scenario status comes from the run directory the CLI names (`run dir: <path>`). A run that
exits without naming one is an error, and a run directory without verdicts is a failure.

## Security

- Loopback only by default: `--host` must be a loopback address unless `--allow-remote` is
  given, and requests whose `Host` header is not a loopback name are refused (DNS rebinding).
- Each server start generates a random token, embedded in the page; every POST must send it
  in `X-MCPSim-Token`, and a POST with a foreign `Origin` is refused. A page left open across
  a server restart is told to reload.
- Path parameters are looked up, never joined: a scenario must be one the scan found, a run id
  a directory listed under that scenario, a transcript a listed `*.jsonl` file, and every
  resolved path (symlinks followed) must stay inside the runs directory.
- The page renders all text through text nodes and ships a strict Content-Security-Policy
  (no inline script or style), because transcripts carry arbitrary web content.
- Runs are argument lists built from validated values (scenario names from the scan, model
  specs matching `provider:model`, known modes), never a shell command.

## Modules

- `mcpsim/ui/app.py`: the endpoints, the security middleware and `serve`.
- `mcpsim/ui/scenario_view.py`: the view of a validated `Scenario`. A file that does not
  validate is read as raw YAML only to place it in the list next to its error.
- `mcpsim/ui/skill_view.py`: role rows and per-scenario resolution over `mcpsim.skill`, and
  the last skill that loaded.
- `mcpsim/ui/store.py`: the scenario scan and the run-directory reads, all confined to the
  runs directory.
- `mcpsim/ui/jobs.py`: the background `mcpsim run` subprocesses.
