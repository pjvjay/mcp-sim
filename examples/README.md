# Sample runs

Artefacts produced on 2026-10-02 against the real pantry MCP server (`pantry-mcp` over stdio,
`DEMO_MODE=1`, a seeded SQLite catalog of 62 products) for `scenarios/pantry/cheapest-penne.yaml`.
Run-directory paths inside the files are the plain `runs/…` prefix the CLI prints by default;
nothing else was edited. The `dry-run/` set was regenerated after the observers landed (the
scenario denies `submit_*`/`review_*`, discloses progressively and declares three observers)
from a copy of the scenario whose `DB_URL` points at a separate seeded SQLite file, so a live run
on `runs/pantry-sim.db` was not disturbed; the `db` field in its `pipeline_status` result is that
file's path on this machine, as the server returned it. `local-plan/` was regenerated on
2026-10-03 by the execution planner (the local profile, docs/LOCAL_MODELS.md "The execution
planner"); `local-run/` predates scoping and observers, is unchanged, and ran the earlier local
plan it holds a copy of.

```
cheapest-penne/
├── dry-run/        MCPSIM_DRY_RUN=1 mcpsim run scenarios/pantry/cheapest-penne.yaml
│   ├── scout.json                    8 read-only observations, 1 informant report, 5 tools disclosed / 8 on request
│   ├── plan.json                     one happy path, the ten goal-relevant free tools, the scout's lookup first
│   ├── transcripts/happy-dry-run-guided-0.jsonl
│   ├── verdicts/happy-dry-run-guided-0.json
│   ├── report.json
│   └── report.md
├── local-plan/     mcpsim plan scenarios/pantry/cheapest-penne.yaml --models planner=ollama:command-r7b
│   ├── scout.json                    4 scout calls, then the planner's probe find_product(query="pene")
│   └── plan.json                     happy, boundary and policy paths; 513 s wall, CPU limited to 24-80 %
└── local-run/      mcpsim run … --models agent=ollama:command-r7b,user=ollama:llama3.2:3b,judge=ollama:qwen2.5:7b
    ├── plan.json                     the earlier local plan: --allow-same-judge --repeat 1 --only-path 1
    └── transcripts/1-guided-0.jsonl  the guided cell; see below
```

## How to read a transcript

One JSON event per line, every event carrying `t` and `kind`:

| kind | what it records |
| --- | --- |
| `system` | mode, the models used per role, the prompts |
| `user` | what the simulated user said (the goal in the role's voice; later, answers to clarifying questions) |
| `assistant` | the agent's text and any `tool_use` blocks |
| `tools_offered` | the agent's tool set changed: `added`, `removed`, `reason` (`initial:<mode>:<disclosure>`, `initial:guided:path`, `discover_tools:<query>`, `observer:<observer>.<condition>`) |
| `tool_call` / `tool_result` | each MCP call and the server's answer: `is_error`, `structured` (parsed JSON), `text` (first 4,000 chars with `sha256` and `chars` of the full text), `ms` |
| `informant_report` | what the observers reported at a trigger (`scout`, `turn`, `tool_result`, `end`): per condition `value` true/false/null, `evidence`, `confidence`, and the `flags` / `failures` / `notes` those reports triggered |
| `goal_enabled` | an observer effect added a goal: `text`, `reason`, `observer`, `condition` |
| `error` | a non-fatal problem, including `scope violation: <tool> (not allowed|not disclosed)` for a call that never reached the server |
| `final_result` | the parsed ```json block the agent ended with (or `null`, with an `error` event saying why) |
| `usage` | tokens per model and the cost estimate (0 and "local" for Ollama) |
| `end` | `completed`, `budget_exceeded` or `error`, with the reason |

## What each set shows

**dry-run.** No model anywhere, and the whole observation → report → effect → plan chain is
visible in the files.

*Observation.* The scenario discloses progressively, so before planning the scout opened one
session and made eight read-only observations within its budget of five tool calls
(`max_tool_calls: 10 // 2`): the server's four static resources (`pantry://recipes`,
`pantry://catalog/categories`, `pantry://countries`, `pantry://origins/coverage`), then
`find_product(query="penne")` — the one disclosed tool whose string argument the expected
outcome pins — and then the zero-argument reads among the disclosed set, `get_product_origins`,
`origin_triage` and `list_products` (`get_product` needs an id, so it was not called).
`find_product` answered `match: "direct"`, `total: 2`, with Penne Rigate 500g at 1.97 from
GreenLeaf Grocers Kitsilano first.

*Report.* The scenario's `shelf_clerk` — "a stock clerk who reads the find_product result and
nothing else", a code observer — reported at the `scout` trigger:
`shelf_clerk.direct_match = true — find_product.match == 'direct', find_product.total == 2`.
The two LLM observers (`fabrication_auditor`, `shelf_auditor`) report at `end` only and are
not consulted in a dry run, so `scout.json` holds exactly that one report.

*Effect.* The report's `then` enabled `get_product` (already in the disclosed set, so the
`tools_offered` entry in `scout.json` adds nothing new but records that the effect fired) and
the goal "Quote the cheapest direct hit by exact name, price and store." (`goal_enabled`).
`scout.json` ends with five tools disclosed, eight on request, and
`planner_prompt_chars: 11461` — the size of the prompt the LLM planner would have received,
under the 12,000-character budget without trimming.

*Plan.* The dry-run planner put the scout's proven lookup first and then the other
goal-relevant free tools (the three costly planners skipped, the two write tools denied): ten
steps, exactly the scenario's `max_tool_calls`, with the scout's report as a checkpoint,
`report: shelf_clerk.direct_match is true`.

*Run.* The agent was offered the path's ten tools (`tools_offered`, `initial:guided:dry-run`)
and called all of them: `find_product(query="penne")` answered as at scout time; the
`shelf_clerk`, now at its `tool_result` trigger, reported `direct_match = true` again, and the
transcript shows the report first, then `tools_offered` (`reason:
observer:shelf_clerk.direct_match`, no new tool because `get_product` was already in the path)
and `goal_enabled` — the condition → toolset chain without any model. The clerk reports after
each of the nine later results too, but a value that merely repeats fires nothing again, so
those entries carry the report alone. Nine tools answered; the tenth, `get_recipe("")`, was
rejected by the server's own validation, which is the right answer to a placeholder slug.
Outcome `completed` ("called 10 planned tool(s); 1 unexpected error result(s)").

*Verdict.* `final_result` is now `find_product`'s result — the structured result whose
top-level keys cover the most expected keys — rather than the `list_recipes` page that came
last, so `query` and `match` pass the matcher; `product_id`, `product_name`, `store`, `price`
and `origin_status` are expected at the top level of an agent's answer and live inside
`find_product`'s `items`, so they fail honestly (`<missing>`). No observer flagged or failed
the run. Read it as proof that the scout, the informants, the planner and the agent loop reach
the server in the right order and record what it says, not as a verdict on the server.

**local-plan.** Written by the execution planner: the 8B Cohere model planned the tool
execution and said where each answer field comes from; the framework built the paths. The
model's first answer traced five fields into `items[2]` of a two-item result; the re-ask
listed what the scout's own `find_product(query="penne")` returned and where each field
actually is, and the second answer was accepted. `plan.json` holds:

* **happy**: `find_product(query="penne")` and an answer step; fourteen checkpoints, seven from
  the model's lineage (`final_result: store equals tool_result[find_product] items[0].store`,
  `query equals tool_result[find_product] query`, …) and seven from `expected_outcome.json`
  (`final_result: price is greater than 0`, `match equals direct`, …);
* **boundary-find_product**: the planner probed `find_product(query="pene")` on the scout's
  session (the last call of the scout's budget, recorded in `scout.json` with `probe: true`);
  the server answered `match: "none"` with no items, so the path sends `pene`, then `penne`,
  and checks `tool_result[find_product]: match equals none when query is pene` and that the
  answer does not come from that result;
* **policy-plan_recipe**: instruction 1 ("do not run a planning tool for a lookup") forbids
  `plan_recipe`, according to one 13-token answer from the model; the path checks
  `transcript: no call to plan_recipe`.

`notes` record every model call's tokens and seconds, and the two policy answers the framework
did not use: `get_product_origins` for "never round a price or guess a store" (the rule names
no tool) and `find_product` (the happy path calls it).

**local-run.** The simulated user (`llama3.2:3b`) opened in role. The agent (`command-r7b`,
guided by the plan, with all fifteen tool definitions in a 4,771-token prompt) **made no tool
call** and answered with a fabricated catalog: product ids 12345 and 67890, "Gluten Free Penne"
from a "Health Food Store" at $3.99, "Basmati Rice Penne" at $4.99, `match: "relaxed"`. Nothing
of that exists in the server. The deterministic matcher fails this run on `match` (expected
`direct`), and on `product_id`, `product_name`, `store` and `price` (expected at the top level,
present only inside a nested `items` list). A judge weaker than the agent could be talked into a
pass by the confident prose; the matcher cannot. The free-mode cell of the same run and the
`qwen2.5:7b` judge votes were still executing when these files were copied, so no verdict or
report is included for this set; the matcher outcome above was derived by hand from the
transcript and the scenario's `expected_outcome.json`.

## Reproduce

```bash
MCPSIM_DRY_RUN=1 .venv/bin/mcpsim run scenarios/pantry/cheapest-penne.yaml
.venv/bin/mcpsim plan scenarios/pantry/cheapest-penne.yaml --models planner=ollama:command-r7b   # the local profile, chosen explicitly
.venv/bin/mcpsim run scenarios/pantry/cheapest-penne.yaml \
  --models agent=ollama:command-r7b,user=ollama:llama3.2:3b,judge=ollama:qwen2.5:7b \
  --allow-same-judge --repeat 1 --only-path happy --plan runs/cheapest-penne/<timestamp>/plan.json
```

Local models are slow on a CPU-only laptop (minutes per LLM turn); see `docs/LOCAL_MODELS.md`.
