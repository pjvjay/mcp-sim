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
   the corrected call. The validator checks that a `recovery` path has a step marked
   `expect_error: true`, a later tool step that expects success, and that the two do not send
   the same tool the same arguments.

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
* **Speed.** See "Speed" below. `concurrency: 1` for local servers (Ollama serialises requests
  anyway); `llama3.2:3b` is the fast option when the task is simple, `qwen2.5:7b` a middle ground.

## Speed

No local call may take longer than ten minutes, and on a CPU that is not throttled each
planner call should answer in about two. What follows is how that is enforced and what was
measured getting there (`command-r7b`, Ollama 0.34.4, i7-9750H with 6 cores and 16 GB, CPU only,
2026-10-02).

**A local call has a deadline.** `MCPSIM_OLLAMA_DEADLINE_S` (default 600) bounds one call *in
total*, retries included; a call that reaches it is cancelled (Ollama stops generating when the
client goes away) and fails with the prompt size in the message. The setting can lower the
limit, never raise it: an answer that needs more than ten minutes means the prompt or the
machine is wrong. A read timeout is not retried, since asking again spends the rest of the
deadline on the same prompt. Requests carry `keep_alive` (`MCPSIM_OLLAMA_KEEP_ALIVE`, default
`30m`) so the model stays loaded while the hosted agent and judge run; Ollama's own five
minutes unloaded it between scenarios, and each one paid the load again with a cold cache.

**Where the time goes.** On this machine with the CPU cool, the model reads about 20 prompt
tokens per second and writes 3.5–4 (4.1 at a short context, 2.7 at 3,400 tokens). The first
planner prompt was 3,385 tokens, so 170 s passed before the first output token. Measured and
ruled out: the JSON-schema grammar (4.07 tok/s with the plan grammar, 4.17 without; Ollama does
not add the schema to the prompt) and the thread count (12 threads: 8.8 tok/s against 16.7 for
Ollama's default of 6).

**The local planner profile.** A planner addressed as `ollama:` gets a profile built for those
numbers:

* *A compact prompt* (`LOCAL_PROMPT_BUDGET`, 4,000 characters): the same rules said once and
  briefly; tools only (a step cannot name a resource or a prompt); descriptions cut to 90
  characters and output keys to six; short section headers; an example step drawn from the
  disclosed tools. The cheapest-penne prompt went from 11,702 characters to 4,750 (3,385 tokens
  to 1,616).
* *One path per call.* Call 1 asks for the happy path. Each later call continues the same
  conversation (the accepted answer replayed as the compact JSON the model wrote, then "one
  more path, of a kind not used yet"), so the server restores its cached prompt and evaluates
  only the new turn. A rejected path gets the usual re-ask; one that fails twice is dropped and
  its kind is not asked for again (at temperature 0 the same request fails the same way). The
  happy path is mandatory. `MCPSIM_LOCAL_PLAN_PATHS` (default 3) sets how many paths; a plan
  never makes more than two calls per path. `plan.json` `notes` record every call's tokens and
  seconds.
* *A grammar that bounds the output.* One path per call, at most six steps and three
  checkpoints, capped string lengths, and every checkpoint matched against a regex of the
  allowed shapes with the catalog's real tool names. Told the shape in prose, `command-r7b`
  wrote `find_product: match == 'direct'`; with the pattern it writes
  `tool_result[find_product]: match equals direct`. llama.cpp converts a pattern only when one
  `^…$` wraps the whole expression; an anchor inside an alternation is logged as unsupported
  and the string goes unconstrained, which a test guards against.

| cheapest-penne plan | prompt tokens | output tokens | wall |
| --- | --- | --- | --- |
| before, under memory pressure (browser open) | 3,385 | 226 | 27 min, invalid |
| before, no pressure | 3,385 | 236 | 4.5 min per call, invalid |
| local profile, call 1 (happy) | 1,616 | 248 | 290 s |
| local profile, call 2 (next path, cached prompt) | 1,845 | 181 | 107 s |
| local profile, call 3 (rejected: `list_products` takes `search`, not `query`) | 2,088 | 171 | 228 s |
| local profile, call 4 (the re-ask, accepted) | 2,377 | 171 | 218 s |

The local-profile rows ran with the CPU held at 22–33 % of its clock (see below): prompt 5.6–6.6
tok/s, output 0.9–2.4. No call came near the deadline. At the unthrottled rates the same calls
take about 2.4 minutes (call 1, whose answer the model pretty-prints; later calls copy the
compact JSON in the conversation and write a third fewer tokens) and one minute (each later
call). A second run, after the recovery rule below, took 134 s, 163 s, 140 s and 129 s.

**Speed is fixed; plan quality is not.** Both runs produced plans that validate and test almost
nothing. In the first, the "recovery" path sent good input marked `expect_error` and had no
corrected call (the validator now requires the failure, a later tool step that expects success,
and different input between the two). In the second, told so, the model changed the kind
instead of the path: all three paths call `find_product(query="penne")` and expect,
respectively, a direct match, a relaxed match and zero results; only the first is what the
server returns. The happy path's single checkpoint (`match equals direct`) never mentions the
price, store or origin status the goal asks for. The grammar guarantees the shape, not the
reasoning, and designing distinct test cases is reasoning a 7–8B model on a compact prompt does
poorly. Neither plan is committed as an example; `examples/cheapest-penne/local-plan` is still
the earlier two-path plan. The likely remedy is a smaller job for the local model: checkpoints
derived from `expected_outcome.json`, the happy path from the scout's proven calls, a fixed shape
per path kind, and short constrained questions to the model ("a bad value for `product_id`",
"the instruction most tempting to break") instead of whole paths.

**What the machine does to it.** Two things outside the framework dominated:

* *CPU speed limit.* `pmset -g therm` reported `CPU_Speed_Limit` between 22 and 37 during these
  runs (`sysctl machdep.xcpm.cpu_thermal_level` 144–244), even with Ollama idle: every rate
  above fell by about 3.5×. On this laptop a 65 W adapter was charging the battery from 10 %;
  the machine ships with an 87 W or 96 W one. Other steady CPU users (an animated wallpaper,
  `mediaanalysisd`, a busy browser or chat window) cost more than their share, because llama.cpp
  runs one thread per core and waits for the slowest.
* *Memory pressure.* Ollama 0.34 loads the weights (5.8 GB with the 8k context) into ordinary
  memory and rejects `use_mlock` as an invalid option, so when the system runs short macOS
  compresses or swaps them: with a browser holding many tabs, prompt evaluation fell to 3 tok/s
  and generation to 0.44.

Check both before a long local run: `pmset -g therm` (`CPU_Speed_Limit` should be 100),
`memory_pressure | tail -1`, and `ollama ps` (100 % CPU, the model loaded).

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
