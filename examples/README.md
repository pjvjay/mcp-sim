# Sample runs

Artefacts produced on 2026-10-02 against the real pantry MCP server (`pantry-mcp` over stdio,
`DEMO_MODE=1`, a seeded SQLite catalog of 62 products) for `scenarios/pantry/cheapest-penne.yaml`.
Run-directory paths inside the files are the plain `runs/…` prefix the CLI printed; nothing else
was edited. The `dry-run/` set was regenerated after tool scoping landed (the scenario denies
`submit_*`/`review_*` and discloses progressively) from a copy of the scenario whose `DB_URL`
points at a separate seeded SQLite file, so a live run on `runs/pantry-sim.db` was not disturbed;
the `db` field in its `pipeline_status` result is that file's path on this machine, as the server
returned it. `local-plan/` and `local-run/` predate scoping and are unchanged.

```
cheapest-penne/
├── dry-run/        MCPSIM_DRY_RUN=1 mcpsim run scenarios/pantry/cheapest-penne.yaml
│   ├── plan.json                     one happy path, the ten goal-relevant free tools, most relevant first
│   ├── transcripts/happy-dry-run-guided-0.jsonl
│   ├── verdicts/happy-dry-run-guided-0.json
│   ├── report.json
│   └── report.md
├── local-plan/     mcpsim plan scenarios/pantry/cheapest-penne.yaml --models planner=ollama:command-r7b
│   └── plan.json                     2 paths; 1,438 s wall on a CPU shared with a test suite
└── local-run/      mcpsim run … --models agent=ollama:command-r7b,user=ollama:llama3.2:3b,judge=ollama:qwen2.5:7b
    ├── plan.json                     --allow-same-judge --repeat 1 --only-path 1 --plan local-plan/plan.json
    └── transcripts/1-guided-0.jsonl  the guided cell; see below
```

## How to read a transcript

One JSON event per line, every event carrying `t` and `kind`:

| kind | what it records |
| --- | --- |
| `system` | mode, the models used per role, the prompts |
| `user` | what the simulated user said (the goal in the role's voice; later, answers to clarifying questions) |
| `assistant` | the agent's text and any `tool_use` blocks |
| `tools_offered` | the agent's tool set changed: `added`, `removed`, `reason` (`initial:<mode>:<disclosure>`, `discover_tools:<query>`, an observer's reason) |
| `tool_call` / `tool_result` | each MCP call and the server's answer: `is_error`, `structured` (parsed JSON), `text` (first 4,000 chars with `sha256` and `chars` of the full text), `ms` |
| `error` | a non-fatal problem, including `scope violation: <tool> (not allowed|not disclosed)` for a call that never reached the server |
| `final_result` | the parsed ```json block the agent ended with (or `null`, with an `error` event saying why) |
| `usage` | tokens per model and the cost estimate (0 and "local" for Ollama) |
| `end` | `completed`, `budget_exceeded` or `error`, with the reason |

## What each set shows

**dry-run.** No model anywhere. The scenario's `tools` policy hides `submit_origin_evidence`
and `review_origin_submission`, so the planner saw thirteen tools; it skipped the three whose
descriptions claim a cost (`plan_recipe`, `plan_from_text`, `plan_week`) and ranked the other
ten by the vocabulary they share with the goal, which is exactly the scenario's
`max_tool_calls: 10`. The agent was offered those ten (`tools_offered`,
`initial:guided:dry-run`) and called all of them in that order: `find_product(query="penne")`
first — the argument comes from the expected outcome, and the server answered `match: "direct"`
with two penne products — then `get_product(1)`, `get_product_origins`, `origin_triage`,
`list_products`, `rank_products_by_origin`, `list_origin_submissions`, `pipeline_status` and
`list_recipes`, all of which answered; the tenth, `get_recipe("")`, was rejected by the server's
own validation, which is the right answer to a placeholder slug. Outcome `completed` ("called 10
planned tool(s); 1 unexpected error result(s)"); the matcher still fails every expectation,
because the dry run's `final_result` is the last structured result — the `list_recipes` page,
seven recipes under a `result` key — not the penne product `find_product` returned nine calls
earlier. Read it as proof that the pipeline reaches the server, starts with the right tool and
records what the server says, not as a verdict on the server.

**local-plan.** The 8B Cohere model, constrained to the catalog's tool names and
schema-checked arguments, produced a correct happy path (`find_product(query="penne", limit=1)`)
with a checkpoint in the required shape, plus a thin recovery path whose "expected error" step
repeats the happy call and would not actually fail. The validator enforces shape, not substance;
a stronger planner model or a judge over the plan itself is the next step.

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
.venv/bin/mcpsim plan scenarios/pantry/cheapest-penne.yaml   # models.planner is ollama:command-r7b in the file
.venv/bin/mcpsim run scenarios/pantry/cheapest-penne.yaml \
  --models agent=ollama:command-r7b,user=ollama:llama3.2:3b,judge=ollama:qwen2.5:7b \
  --allow-same-judge --repeat 1 --only-path 1 --plan runs/cheapest-penne/<timestamp>/plan.json
```

Local models are slow on a CPU-only laptop (minutes per LLM turn); see `docs/LOCAL_MODELS.md`.
