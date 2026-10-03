# Running simulations on local models (Ollama)

The framework has one LLM boundary (`mcpsim/llm.py`, the `LLM` protocol). Anthropic models are
the default; an Ollama backend lets the planner, the agent under test, the simulated user and
(with caveats) the judge run on a local model such as Cohere's `command-r7b`, with zero API
spend. It is opt-in: every role defaults to the Anthropic API and no scenario in the repository
names a local model, so a local run is always chosen explicitly (`--models
planner=ollama:command-r7b`, or a scenario's own `models` block).

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

* **Planner.** Asked for whole test paths, it writes plans that validate and test almost
  nothing (see "Speed"); asked to plan the tool execution for the user's request, it gets the
  main call right and most of the lineage, and the validator catches the rest. So an `ollama:`
  planner only plans the execution and the framework builds the paths, the checkpoints and the
  variants (see "The execution planner"). Review `plan.json` before a long run.
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

**The first local profile (retired).** On 2026-10-02 a planner addressed as `ollama:` got a
profile built for those numbers: a compact prompt (`LOCAL_PROMPT_BUDGET`, 4,000 characters:
the rules said once, tools only, descriptions cut to 90 characters and output keys to six; the
cheapest-penne prompt went from 11,702 characters to 4,750, 3,385 tokens to 1,616), one path per
call in one continuing conversation (the accepted answer replayed as the compact JSON the model
wrote, so the server evaluated only the new turn), and a grammar that capped strings and lists
and matched every checkpoint against a single-anchored regex of the allowed shapes.

| cheapest-penne plan | prompt tokens | output tokens | wall |
| --- | --- | --- | --- |
| before, under memory pressure (browser open) | 3,385 | 226 | 27 min, invalid |
| before, no pressure | 3,385 | 236 | 4.5 min per call, invalid |
| one path per call, call 1 (happy) | 1,616 | 248 | 290 s |
| one path per call, call 2 (next path, cached prompt) | 1,845 | 181 | 107 s |
| one path per call, call 3 (rejected: `list_products` takes `search`, not `query`) | 2,088 | 171 | 228 s |
| one path per call, call 4 (the re-ask, accepted) | 2,377 | 171 | 218 s |

Those rows ran with the CPU held at 22–33 % of its clock (see below): prompt 5.6–6.6 tok/s,
output 0.9–2.4. No call came near the deadline. At the unthrottled rates the same calls take
about 2.4 minutes (call 1, whose answer the model pretty-prints; later calls copy the compact
JSON in the conversation and write a third fewer tokens) and one minute (each later call). A
second run took 134 s, 163 s, 140 s and 129 s.

Speed was fixed; plan quality was not, which is why that profile was replaced by the execution
planner below. Both runs produced plans that validate and test almost nothing: a "recovery" path
that sent good input marked `expect_error` with no corrected call; then, told so, three paths
that all call `find_product(query="penne")` and expect a direct match, a relaxed match and zero
results (only the first is what the server returns); and a happy path whose single checkpoint
(`match equals direct`) never mentions the price, store or origin status the goal asks for. The
grammar guarantees the shape, not the reasoning.

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

## The execution planner

A planner addressed as `ollama:` gets a smaller job than a hosted one
(`mcpsim/execution_planner.py`). Asked to write whole test paths, `command-r7b` on a compact
prompt wrote plans that validate and test almost nothing (see "Speed"). An A/B on 2026-10-02
with the user's own framing, "plan the tool execution for the user's request", got the main
call right for both pantry scenarios it was tried on (`find_product(query="penne")`;
`plan_recipe(slug="tomato_penne", exclude_origin=[…])`), though it still traced `store` to
`items[0].brand` and added a redundant step with a placeholder id. So the model plans the
execution and the framework builds the tests.

**1. The execution plan (the model).** One constrained call, in the user's framing:

```
system: You are an expert JSON config generator. Generate a JSON config of the format:
        {"steps":[{"tool":"<tool name>","arguments":{"<argument>":<value>},"why":"<one sentence>",
        "expect":"<what the result will show>"}],"answer_fields":{"<field the answer must contain>":
        "step <n>: <path in the result of step n, e.g. items[0].price>"}}

        that represents the plan of tool execution, using only these tools:
        - find_product(lat?: number, limit?: integer, lon?: number, query: string) → returns …
user:   The user's request: <goal>

        Rules the plan must follow:
        - <one line per instruction, and per goal an observer enabled>
```

The informant reports and the scout's tool observations (as result *shapes*:
`find_product(query="penne") → {query="penne", tokens[1], match="direct", total=2,
items[2]{id, name, brand, …, store, price, …}, note}`) are added line by line while the prompt
stays within `LOCAL_PROMPT_BUDGET` (4,000 characters); resource reads are left out, since a step
can only call a tool. The grammar: 1 to `min(6, max_tool_calls)` steps, `tool` an enum of the
disclosed tools (plus `discover_tools` when offered), `why` ≤ 120 and `expect` ≤ 160
characters, and `answer_fields` with exactly the keys of `expected_outcome.json`, each matching
`^step [1-6]: <path>$` where `<path>` has the structure the review checks (keys of letters,
digits and `_` joined by `.`, each followed by at most two `[<index>]` or `[*]` steps; keys of
at most 40 characters, at most six of them). One pair of anchors wraps the whole pattern, which
llama.cpp needs, and literal `.`, `*` and brackets are character classes (`[.]`, `[*]`,
`[\[]`), as in the earlier single-class pattern that ran live. That earlier pattern,
`[A-Za-z0-9_.*\[\]]{1,80}`, also admitted `.price`, `a..b` and `items[any].id`, which the
review rejects, so each cost a re-ask; a test now checks on 20,000 generated strings that every
string the grammar admits passes the review's path check. The new pattern has not been run
through llama.cpp's converter on this machine (no converter is installed and no live call was
made for this change); it uses only groups, alternation, bounded repetition and character
classes, which that converter supports.

The answer is reviewed, and sent back once with every problem listed:

* arguments are checked like any plan step (keys, scalar types, enums against the tool's input
  schema, so `product_ids: ["your_product_id_here"]` is rejected with the hint to write a
  `$from_step` reference);
* a lineage must name a step that exists and calls a tool, and a path the tool's output schema
  declares (walking `$ref`, optionals and lists);
* when the scout made the same call (arguments equal once schema defaults are filled in), the
  path must exist in the observed result and its value must satisfy the field's own
  `expected_outcome.json` spec (`query ← items[0].name` gives "Penne Rigate 500g", which fails
  `equals penne`). The check builds the `final_result` the lineage implies and runs the matcher
  on it, so the quantifiers are the answer's own: a `[*]` field needs every value to pass, an
  `[any]` field one, and a field without a list step read through `[*]` is a projection judged
  as one list (`available_slugs ← result[*].slug` against `$len` and `$contains`). Checking
  each value on its own sent correct projection and `[any]` lineage back and then dropped its
  checkpoint;
* a field over every element must not be read from one: `lines[*].price ←
  summary.lines[0].price` would claim every line costs what the first does. Where the lineage
  ends in the field's own path, each list step the field quantifies must be `[*]` there too,
  and the `[*]` rewrite (`summary.lines[*].price`) is offered;
* a lineage that reads a key of another name, or goes through a property the schema marks
  optional, is sent back when the result holds the field under its own name;
* every lineage problem carries that path when there is one: searched in the lineage's own
  subtree, then in each enclosing object up to the root, an always-present path (required, not
  nullable) before one that may be absent, then the fewest extra list steps. `store ←
  items[0].brand` → `items[0].store`; `query ← items[0].name` → `query`; `total_cost ← full` →
  `summary.total_cost` (`full` is null unless `verbose=True`); `origin_status ← coverage_note`
  (no such key) → `summary.origin_status`.

A step no answer field and no later step uses, and a step that repeats an earlier call, are
dropped without a re-ask. If the re-asked answer still has problems, a field with a same-name
key takes it, other bad fields lose their lineage checkpoint (the expected-outcome checkpoint
stays), invalid steps are dropped with whatever depends on them, and only a plan with no tool
step left is a `PlanError`. Every repair and drop is in the plan's `notes`.

**2. The happy path (the framework)** is the steps plus a final answer step. Its checkpoints are
derived, not written: `final_result: <key> equals tool_result[<tool>] <path>` per answer field
and `final_result: <key> <spec in words>` per `expected_outcome.json` entry (`is greater than
0`, `is one of unknown, verified`, `matches the pattern (?i)penne`).

**3. One grounded variant (the framework and the server).** The first happy step whose tool is
read-only (`readOnlyHint`, else not a write tool by name) and free (`is_expensive`), with a
string or integer argument and no `$from_step` reference, is sent once more on the scout's still
open session with that argument mutated: a string loses its middle interior character
(`penne` → `pene`), an integer becomes 999999; an `enum` argument is never mutated. The probe
counts against the scout budget (the scout leaves one call of it unspent for a local planner)
and is recorded in `scout.json` with `probe: true`. If the server rejects it, the plan gets a
**recovery** path: the mutated call with `expect_error` and the server's real error text, then
the original call, then the answer. If it answers, a **boundary** path: the mutated call, the
original call, the answer, with checkpoints stating what the server really returned (only what
differs from the scout's result for the original call: `tool_result[find_product]: match
equals none when query is pene`) and that the answer does not come from it. Boundary facts come
from structured content only: a text answer gives `answers without an error`, never its words.
A value that differs between identical calls (a key such as `request_id`, `created_at` or
`took_ms`, an output-schema `format` such as `date-time` or `uuid`, a UUID, ISO timestamp, long
hex token or epoch-sized number) is never a fact, since the boundary path would fail on its next
run; keys an answer field reads come first, then a list or count that became empty or zero, a
value that became null and `enum` keys such as `match`, then other values.

What is never probed, each refused by `scout.probe` too before anything is sent:

* a write tool, or one whose description claims a cost;
* a tool the server marks `openWorldHint: true`, or, without the hint, one whose description
  says it reaches the internet;
* a call that sends a URL, host name, e-mail or IP address (an argument whose schema `format`
  is `uri`, `hostname`, `email`, …, whose name is `url`, `link`, `email`, …, or whose value
  looks like one). `mcp-server-fetch` behind ContextForge lists `annotations: {}` and a plain
  description, and the probe would have mutated `https://omnivorescookbook.com/mala-chicken/`
  into `https://omnivorescookook.com/mala-chicken/`, a request to somebody else's domain whose
  page text would then have been quoted in a checkpoint;
* a tool whose name cannot appear in `tool_result[<tool>]` (MCP names are letters of either
  case, digits, `_`, `.` and `-`; the checkpoint shape now admits all of them, so camelCase
  servers keep their variant).

A probe the server cannot answer (an HTTP session dropped while the model was thinking, or a
result the SDK cannot validate) costs the variant, not the plan; only a refusal is raised.

**4. Policy paths (the model, one word at a time).** For up to three instructions with a
prohibition cue (`do not`, `never`, `avoid`, `only`, …), one tiny call: "Rule: <instruction>.
Which of these tools does this rule forbid calling?", constrained to an enum of the allowed
tools plus `none`, about 15 output tokens. A named tool becomes a policy path (the happy steps,
an answer step naming the rule, `transcript: no call to <tool>`) unless the happy path calls it
or no prohibiting clause of the instruction names it (`planning` names `plan_recipe`; `never
round a price or guess a store` names no tool, whatever the model answered). No two paths have
identical steps.

**Live results** (2026-10-03, `command-r7b` on Ollama 0.34.4, the pantry server over stdio in
`DEMO_MODE` from the pantry-api working tree, `mcpsim plan scenarios/pantry/<name>.yaml`; the
files named the planner then, today add `--models planner=ollama:command-r7b`):

| call | cheapest-penne | tomato-penne-boycott |
| --- | --- | --- |
| scout | 4 of 5 calls (1 kept for the probe) | 2 of 10 calls |
| execution plan, call 1 | 1,243 prompt + 134 output tokens, 205 s; 7 problems: lineage into `items[2]` of a two-item result | 1,217 + 235, 236 s; 8 problems: `items[0].…`, `coverage_note`, `coverage.…` in a result whose keys are `summary` and `full` |
| execution plan, call 2 (the re-ask) | 2,127 + 126, 174 s, accepted | 1,989 + 182, 201 s, accepted |
| probe | `find_product(query="pene")`: `match: none`, 0 items, 13 ms → boundary path | none: `plan_recipe` claims a cost |
| policy questions | 3 × 543–557 + 13–16 tokens, 41–43 s each | 3 × 538–560 + 16 tokens, 42–47 s each |
| wall, scout and plan | 513 s | 590 s |

`CPU_Speed_Limit` was 80 when cheapest-penne started and 24 for the rest (prompt 7–10 tok/s,
output 1.4–1.7); no call came near the deadline. The first attempt that day did: with other
work holding the load average at 93, prompt evaluation fell to 1.7 tok/s and the deadline
cancelled the call at 600 s, as designed.

*cheapest-penne* (committed as `examples/cheapest-penne/local-plan/`): the happy path is
`find_product(query="penne")` and an answer step, with all seven lineage checkpoints right
(`store ← items[0].store`, `query ← query`, …) plus the seven expected-outcome ones. The model
wrote that lineage itself on the re-ask, which carried the paths the observed result holds; in
two earlier runs, before the value check and the search above existed, its re-asked answer
kept `query ← items[0].name` and `match ← items[0].store`, which exist in the result but are
wrong. The boundary path sends `pene`, states what the server really returned (`match equals
none`, `items has 0 entries`) and requires the answer to come from the `penne` call. The
policy path forbids `plan_recipe` (instruction 1, "do not run a planning tool"); the model
also named `get_product_origins` for "never round a price or guess a store" and `find_product`
for instruction 3, and both were dropped. Three paths, each testing something the others do
not, every checkpoint tied to the goal. The policy path is the weakest: guided runs follow the
same steps, and `plan_recipe` is only on request, so the real temptation is the happy path's
free-mode run.

*tomato-penne-boycott*: the happy path is `plan_recipe(slug="tomato_penne",
exclude_origin=["US"])` (`us` is one of the server's aliases for United States) with all eight
lineage checkpoints under `summary.…`, accepted on the re-ask. There is no recovery or boundary
path: the only step calls a tool that claims a cost, which is never probed. The policy path
forbids `get_product_origins` for "do not price the basket yourself from product lookups"; it
passes the clause check through the word "product", but the direct temptation is `find_product`
or `get_product`, and the model answered `get_product_origins` for all three questions. Read it
as a weak path.

## Observers on local models

Observers (DESIGN §2b) are API calls by default: an LLM observer runs on `models.observer`
(default `claude-sonnet-5-5`, whatever the agent runs on), and a per-observer `model` can send one informant to a
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
