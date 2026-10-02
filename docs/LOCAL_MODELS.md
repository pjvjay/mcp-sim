# Running simulations on local models (Ollama)

The framework has one LLM boundary (`mcpsim/llm.py`, the `LLM` protocol). Anthropic models are
the default; an Ollama backend lets the planner, the agent under test, the simulated user and
(with caveats) the judge run on a local model such as Cohere's `command-r7b`, with zero API
spend. This is how the first plan executions are meant to run on a laptop before a key is
involved.

## Addressing a model

Every `models` entry in a scenario is `provider:model`; a bare name means `anthropic`.

```yaml
models:
  planner: ollama:command-r7b
  agent:   ollama:command-r7b
  user:    ollama:llama3.2:3b          # the simulated user needs little
  judge:   anthropic:claude-opus-5-5   # see "The judge" below
```

`OLLAMA_HOST` (default `http://localhost:11434`) selects the server. Nothing else is configured:
the catalog, prompts and transcripts are identical whichever provider answers.

## How the Ollama backend maps the protocol

| Protocol concept | Ollama `/api/chat` |
| --- | --- |
| `system`, `messages` with `text` blocks | `messages` with roles `system`/`user`/`assistant` |
| `tools` (MCP input schemas) | `tools: [{type: function, function: {name, description, parameters}}]` — the MCP `inputSchema` is already JSON Schema and is passed through |
| assistant `tool_use` block | `message.tool_calls[*].function` (`name`, `arguments` object); each gets a synthetic id `call_<n>` so transcripts keep the same shape as Anthropic runs |
| user `tool_result` block | `{role: tool, tool_name: <name>, content: <text or JSON>}` |
| `tool_choice = {type: tool, name: X}` (planner / judge structured output) | `format: <X's input_schema>` — Ollama's structured outputs constrain decoding to the schema; the reply is wrapped as a single `tool_use` block for `X` so callers do not know the difference |
| `stop_reason` | `tool_use` when `tool_calls` is non-empty, else `end_turn`; `max_tokens` when `done_reason == "length"` |
| `usage` | `prompt_eval_count` / `eval_count`; cost is 0 and the report says "local" |
| retries | 5xx and connection errors, exponential backoff; a 404 for an unknown model fails immediately with the `ollama pull` hint |

Options sent with every call: `temperature: 0` (simulations should vary by path and repeat, not
by sampling noise — repeats still differ because tool results and the simulated user differ),
`num_ctx` from the model's reported context length (8192 for `command-r7b`), `num_predict`
from `max_tokens`.

## Fitting an 8k context

`command-r7b` has an 8,192-token window; the planner prompt must respect it. The planner
therefore sends a **catalog digest** — one line per tool: name, argument names with types
(`?` marks optional), first sentence of the description — rather than full schemas. Fifteen
pantry tools digest to about 1.3k tokens. Tool results fed back to the agent are the
token-lean MCP shapes (summaries, pages) and the transcript truncation at 4,000 characters
applies before the result is shown to the model. Scenarios meant for local models should keep
`budgets.max_tool_calls` around 8 and `max_turns` around 8; the runner marks a run
`budget_exceeded` rather than overflowing the window.

## Constrain, don't just validate

With structured decoding the schema is the strongest lever. The planner's `Step.tool` field is
emitted as `{"enum": [<every catalog tool name>]} | null` rather than a free string, so a local
model *cannot* name a tool that does not exist; the catalog validation and single re-ask remain
as the safety net for hosted models. The lesson came from a smoke call on this machine: with
an unconstrained field, `command-r7b` wrote a sentence of prose into `tool`. Likewise
`Path.kind` is an enum and `arguments_sketch` keys are checked against the tool's schema.

## Lessons from the first local plan (command-r7b, pantry boycott scenario, 2026-10-01)

The first plan the local model wrote against the live 15-tool catalog validated (every tool
name real, thanks to the enum) and had the right spine — `plan_recipe` with
`exclude_origin: ["United States"]`, then a final JSON answer — but showed four weaknesses the
planner must close with structure rather than prose:

1. **Argument keys and types must be schema-checked, not just tool names.** It called
   `list_recipes` with `{"slug": …}` (the tool takes no arguments) and `get_product_origins`
   with string placeholders where integers are required. Validation: every `arguments_sketch`
   key must be a property of the tool's input schema and scalar values must match the declared
   type; placeholders are allowed only as `{"$from_step": n, "path": "summary.lines[*].product_id"}`
   references, which the executor resolves at run time.
2. **The digest must say what a tool returns.** The model added a redundant
   `get_product_origins` call to learn an `origin_status` that `plan_recipe`'s `PlanResult`
   already carries. Each digest line gains `→ {top-level output fields}` from the output
   schema when the server publishes one.
3. **Checkpoints need a shape.** "Priced shopping list for tomato_penne" is not checkable.
   Ask for `<where>: <observable condition>` with `<where>` ∈ {`final_result`, `tool_result[<tool>]`,
   `transcript`} and reject a plan whose checkpoints lack the prefix.
4. **A recovery path must contain the failure.** The model's recovery path was the happy path
   with `exclude_origin: ["server_suggestion"]`; a recovery path must have a step whose
   `success_looks_like` names the server's rejection (e.g. a ToolError with suggestions) before
   the corrected call. The validator checks that a `recovery` path has at least one step marked
   `expect_error: true`.

Timing under a heavily loaded machine: 1,451 prompt tokens, 918 output tokens, 20 minutes.
Unloaded, expect a few minutes. These rules are provider-independent and improve hosted plans
too; the local model is simply where their absence shows first.

## What to expect from a 7–8B model

* **Planner.** Produces usable happy paths and recovery paths; weaker at inventing boundary and
  policy paths. Plans are validated against the catalog and re-asked once on an unknown tool;
  with structured decoding the schema is always satisfied, so the failure mode is a thin plan,
  not a broken one. Review `plan.json` before a long run.
* **Agent under test.** Handles 2–4 tool-call paths reliably when `guided`; `free` mode is where
  it shows its limits, which is exactly what the simulation is for. Treat a `free`-mode failure
  on a local model as a hint about tool descriptions, not as a verdict on the server.
* **Simulated user.** Fine on any of the three models.
* **The judge.** A judge weaker than the agent is the one configuration the design warns
  against. Use an Anthropic model for the judge when a key is available; when it is not, set
  `allow_same_judge: true` and read the verdicts as a first pass — the deterministic matcher
  still runs and still overrides the votes, so JSON expectations are enforced exactly.
* **Speed.** Measured on this CPU-only Intel laptop with `command-r7b`: 45 s to load the model,
  then about 2.5 minutes for a 90-token structured answer to a one-line prompt; a full plan or an
  agent turn that reads a long tool result takes several minutes. `concurrency: 1` for local
  servers (Ollama serialises requests anyway); `llama3.2:3b` is the fast option when the task is
  simple, `qwen2.5:7b` a middle ground.

## Observers on local models

Observers (DESIGN §2b) are API calls by default: an LLM observer runs on `models.observer`,
which falls back to the agent's model, and a per-observer `model` can send one informant to a
local model (`model: ollama:qwen2.5:7b`) while the rest stay hosted. Cost control is built into
the design rather than the provider: `on` defaults to `[scout, end]` — one report at plan time
and one after the final answer — so a scenario with two LLM observers costs two observer calls
per run unless it asks for `turn` or `tool_result`; `MCPSIM_OBSERVER_MAX_CALLS` (default 12)
caps the calls per run and the informant then reports `unknown` with evidence "observer budget
exhausted" rather than being skipped; code and group observers never call a model and are the
ones that drive the dry run. On a CPU-only local model an observer call at every `tool_result`
costs as much as an agent turn (minutes), so keep local informants on `[end]`, keep their
`watches` narrow (a `tool_traffic`-only observer's prompt is a few hundred tokens), and prefer
a `code` observer with a `tool_result` check whenever the condition is a field in a structured
result. The scout's planner prompt is bounded by `MCPSIM_PLANNER_PROMPT_BUDGET` (12,000
characters, about 3k tokens) for the same 8k-window reason as the catalog digest: the pantry
cheapest-penne prompt with its eight observations and the informant report measures 11,461.

## Smoke sequence on this machine

```bash
ollama list                                   # command-r7b, llama3.2:3b, qwen2.5:7b
mcpsim catalog scenarios/pantry/tomato-penne-boycott.yaml
mcpsim plan scenarios/pantry/tomato-penne-boycott.yaml --models planner=ollama:command-r7b
mcpsim run  scenarios/pantry/cheapest-penne.yaml --models agent=ollama:command-r7b,user=ollama:llama3.2:3b,judge=ollama:qwen2.5:7b --allow-same-judge --repeat 1
```

`--models` overrides the scenario's `models` block for one run, so the same scenario files serve
the local and the hosted configuration.
