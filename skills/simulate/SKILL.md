---
name: simulate
description: >
  Run mcp-sim's Sierra-style simulations end to end on this machine: preflight the services,
  generate the gateway scenarios, run the configured suite, open the test runner and read the
  report. Use it to test an MCP server (the pantry server, directly or through ContextForge)
  with simulated users and an independent judge, or to change which model or prompt a
  simulation role uses.
---

# Simulate

mcp-sim tests an MCP server the way Sierra tests an agent. In each scenario a simulated user,
briefed in the second person with a persona, a situation and a context (device, location,
language), talks to the agent under test. The agent works through the server's tools, following
its standard operating procedure when it has one. An independent judge then grades every item
of the scenario's expected behaviour, with a verbatim quote for each. Every path and mode is
repeated `repeat` times, and a scenario passes pass^k only when all of those runs pass.

Every LLM call of a simulation is configured here, not in code:

| file | what it holds |
| --- | --- |
| `config.yaml` | models and run defaults (repeat, modes, judge_votes, concurrency) that override the role files, the scenario sources, `runs_dir`, per-scenario overrides |
| `roles/planner.md` | the hosted planner: system prompt, user prompt, re-ask |
| `roles/planner-local.md` | the execution planner, used only for an `ollama:` planner |
| `roles/agent.md` | the agent under test: system prompt (rendered each turn), the note an observer-enabled goal adds |
| `roles/user.md` | the simulated user: system prompt, opening cue, fallbacks |
| `roles/observer.md` | the LLM observers (Informant-Report Method) |
| `roles/judge.md` | the judge: the rules, the evidence it is shown, `votes` (the default judge_votes) |
| `scripts/run.sh` | the steps below as one script |

Each role file has YAML frontmatter (`role`, `provider`, `model`, optional `temperature` and
`max_tokens`, plus `votes` for the judge and `policy_max_tokens` for the local planner) and a
body of prompts, each starting with a `{% prompt NAME %}` line. They are read at run time, so a
change takes effect on the next run, with no code edit and no reinstall.

The prompts, `temperature` and `max_tokens` come only from the role file. Its `provider` /
`model` and the judge's `votes` are the lowest configurable layer instead: they are the
default, and anything above them wins (see [Change a model](#change-a-model)). The bundled
`config.yaml` sets no models and no `judge_votes`, so the role files are live as shipped; once
you set `defaults.judge` or `run.judge_votes` there, an override, a scenario's own `models` or
`judge_votes`, or `--models`, the role file's value is shadowed. `mcpsim config` shows which
layer each value came from and prints a `note:` under every role whose value is shadowed.

## Run it end to end

Work from the mcp-sim checkout. `scripts/run.sh` takes care of the paths, and of the secrets: it
loads them into the environment without printing them.

```bash
cd ~/Documents/workspace/mcp-sim            # the checkout; run.sh finds it on its own too
export CF_JWT_FILE=/path/to/contextforge_jwt.txt   # the ContextForge admin JWT (a file, never the value)
skills/simulate/scripts/run.sh preflight
skills/simulate/scripts/run.sh scenarios
skills/simulate/scripts/run.sh suite --name 'cheapest-*'
skills/simulate/scripts/run.sh ui
skills/simulate/scripts/run.sh report
```

1. **Preflight** (`run.sh preflight`). This checks five things:
   - mcp-sim's `.venv` is installed.
   - `ANTHROPIC_API_KEY` is set. It comes from `mcp-sim/.env` (`set -a; . .env; set +a`); never echo it.
   - The Anthropic API accepts one Haiku token, which proves the key works and the account has
     credit. Set `SKIP_KEY_CHECK=1` to skip this check.
   - The pantry API answers `$PANTRY_API_URL/health` (default `:8000`).
   - ContextForge answers `$CF_URL/health` (default `:4444`). If it does not, start it with pantry-gateway's `scripts/run.sh`.
   - The fetch server answers `$FETCH_URL/healthz` (default `:9100`). If it does not, start it with `scripts/run_fetch.sh`.

   It then asks ContextForge to re-read the pantry tools (pantry-gateway's
   `scripts/register_fetch.sh` with `REFRESH_PANTRY=true`), so the gateway scenarios see the tools
   the server has today. Set `SKIP_REFRESH=1` to skip that step. Fix every `FAIL` line before you
   go on.
2. **Gateway scenarios** (`run.sh scenarios`). This looks up the `pantry-sim` and `pantry-recipes`
   virtual servers by name and runs pantry-gateway's `scripts/make_scenarios.py` twice: once on
   `scenarios/pantry`, and once with `--recipe`. Both write into
   `pantry-gateway/scenarios/generated`, which `config.yaml` reads through
   `${PANTRY_GATEWAY_SCENARIOS:-../pantry-gateway/scenarios/generated}`. The generated files
   have names ending in `-gateway`. The recipe-link scenario's SOP comes from
   `RECIPE_SHOPPER_SKILL`; run.sh defaults it to pantry-api's `skills/recipe-shopper/SKILL.md`.
3. **Suite** (`run.sh suite [ARGS]`, which runs `mcpsim suite --skill <this skill> ARGS`). It
   runs every configured scenario, one after another, into `runs_dir` (`runs/simulate`). Then it
   prints a table with each scenario's passed/runs, pass rate, pass^k, cost and time, and writes
   `suite-<timestamp>/suite.md`. You can narrow the run:
   - `--name GLOB` and `--category GLOB` select scenarios (both repeatable).
   - `--repeat N` and `--modes guided` change the run settings.
   - `--models judge=anthropic:claude-opus-5` changes a model.
   - `--list` shows what would run, with the resolved settings, and runs nothing.
   - `--dry-run` uses no LLM at all.

   Each run makes live Claude API calls. A judge vote on Opus 5.5 costs about $0.05 (list
   prices; `report.json` has the estimate per run), and every transcript gets `judge_votes`
   of them, so start with `--name` on one scenario.
4. **Test runner** (`run.sh ui`, which runs `mcpsim ui --skill <this skill>` on
   127.0.0.1:8765). The runner offers search, run all / run one / re-run, run history and
   settings. Its detail view shows the user instructions, the context, each expected-behaviour
   verdict and the transcript with its tool calls. Settings and the detail view show the models
   and run settings this skill resolves, each with the layer it came from. The Run buttons
   start `mcpsim run <file> --skill <this skill>`, so a run from the page is the same as one
   from the shell.
5. **Report** (`run.sh report [DIR]`). This prints the newest `suite.md`, or a run directory's
   `report.md`. A run directory holds `scenario.json` (the scenario as it ran: resolved models,
   repeat, votes), `plan.json`, `transcripts/`, `verdicts/` (the expected-behaviour checklist,
   `goal_achieved`, `sop_followed`), `report.json` (`pass_k`, `cost_usd`, `duration_s`) and
   `report.md`.

## Change a model

Change it in the skill, not in the scenarios. Model precedence for each role, lowest to
highest:

> built-in default < `roles/<role>.md` frontmatter < `config.yaml` `defaults` < every matching
> `config.yaml` override (in file order) < the scenario file's own `models` < `--models`

- **Every scenario:** edit the role file's `model` (for example `model: claude-opus-5` in
  `roles/judge.md`), or set it in `config.yaml` `defaults` (`judge: anthropic:claude-opus-5`),
  which then wins over the role file.
- **Some scenarios:** add an override, e.g.
  `{match: {category: "Recipe*"}, models: {judge: ...}, run: {judge_votes: 1}}`. `match` takes
  fnmatch globs on `name` and/or `category`.
- **One run:** pass `--models role=provider:model` (roles: planner, agent, user, observer, judge).
- **A local model:** use `ollama:<model>`. A local planner switches to `roles/planner-local.md`.

Run settings (`repeat`, `modes`, `judge_votes`, `concurrency`) follow the same ladder, with
`roles/judge.md`'s `votes` as the layer of `judge_votes` above the built-in default. A value
the scenario file sets itself (most pantry scenarios set `repeat` and `judge_votes`) wins over
`config.yaml` and the role files.

`mcpsim config` (or `run.sh config`) prints every role's model and the layer it came from, a
`note:` for every role-file value that a higher layer shadows, the run settings and the
scenario sources. `mcpsim config --scenario NAME` shows the resolution for one scenario, its
overrides included. `--json` gives the same as data.

Notes:

- Claude Opus 5.5, Sonnet 5.5, Fable 5.1 and Opus 4.7 or later reject `temperature`. The
  check runs whenever a scenario is resolved: `mcpsim config` (for the defaults, or for
  `--scenario`) lists the conflict under `problems` and exits 1, `suite --list` shows the
  scenario as an error row, and `run`, `plan` and `suite` refuse it before any call. A dry run
  calls no model and skips the check.
- The judge must differ from the agent unless the scenario sets `models.allow_same_judge`.

## Change a prompt

Edit the prompt in `roles/<role>.md`. The template language is small:

- **`{{ name }}`** inserts a value the code computed: a catalog digest, the transcript, the
  checklist. The value is inserted verbatim.
- **`{% if name %}` … `{% elif other %}` … `{% else %}` … `{% endif %}`** keeps a block only
  when the value is non-empty. `{% if not name %}` inverts the test.
- **`#. `** at the start of a line auto-numbers that line.
- **`{# … #}`** is a comment.

A line that holds only tags disappears entirely. The comment at the top of each role file lists
the placeholders each prompt accepts, and which ones it must insert. `mcpsim config --json`
lists them too, under `roles.<role>.prompts`.

The loader refuses these mistakes with the file and line:

- an unknown placeholder;
- a required placeholder that is missing (the planner's `catalog`, the agent's `steps`, the
  judge's `transcript` and `behaviors`, and so on);
- an unknown or missing prompt;
- an unknown frontmatter key.

Run `mcpsim config` after an edit; it loads every role and fails loudly on these errors (and
on a temperature the resolved model rejects).

To experiment without touching the bundled files, copy the skill with
`cp -R skills/simulate /tmp/my-skill`, edit the copy, and run with `--skill /tmp/my-skill` (or
`MCPSIM_SKILL=/tmp/my-skill`).

The bundled templates are pinned by `tests/test_prompt_golden.py`. They render byte for byte
the prompts captured in `tests/fixtures/prompts/golden.json.gz`. If you change a bundled prompt
on purpose, regenerate the capture with `.venv/bin/python -m tests.prompt_cases --write`, and
review the diff of what the models will now be told.

## Troubleshooting

- **`scenario '…': environment variable RECIPE_SHOPPER_SKILL is not set`**: export it, or use
  `run.sh`, which sets it. `suite --list` shows which scenarios do not load and why.
- **Gateway scenarios fail with 401**: `CONTEXTFORGE_JWT` is missing. Set `CF_JWT_FILE`;
  run.sh reads the file into it.
- **`credit balance is too low`** (HTTP 400 from Anthropic): the account is out of credit. Use
  `--dry-run` to test the wiring without LLM calls.
- **A scenario stops with a setup or connection error**: the suite records it, goes on with the
  other scenarios, and exits 1.
