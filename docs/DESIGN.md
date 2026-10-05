# mcp-sim — design

An LLM-as-a-judge simulation framework for MCP servers that emulates the *features* of
[Sierra's agent simulations](https://sierra.ai/blog/simulations-the-secret-behind-every-great-agent)
(not their look): a **three-agent architecture** — a simulated user driven by second-person
**user instructions** in a **context** (device, location, language), the *agent under test*
that drives the MCP server (optionally on a **skill** as its standard operating procedure), and
an independent *judge* that grades an **expected behaviour** checklist item by item — run over
an **execution plan** with several **paths**, each repeated to absorb non-determinism (reported
as a pass rate and as **pass^k**), scored against an **expected outcome** that can be prose, a
JSON spec, or both.

The first server it is pointed at is the pantry planner (`pjvjay/pantry-api`, tools at `/mcp`
or the `pantry-mcp` stdio script). Nothing in the core knows about pantry; the pantry suite is
a directory of scenario files.

Every LLM call is packaged as one skill, `skills/simulate/` (§2c): its `config.yaml` chooses
each role's model and the run settings, and its `roles/<role>.md` files hold each role's
settings and prompt templates, read at run time, so models and prompts change without a code
edit.

## 1. Vocabulary

| Term | Meaning |
| --- | --- |
| **Scenario** | One test case: `role`, `goal`, `instructions`, `expected_outcome`, server target, budgets, repeat count, and (v2, §3) `category`, `title`, `user_instructions`, `context`, `expected_behavior`, `agent`. A YAML (or JSON) file. |
| **Category** | The runner's group for a scenario (`Product lookup`, `Provenance`, …); default `Uncategorized`. |
| **Role** | Who the agent is acting for and how that principal behaves (persona). The agent under test is told it; with `goal` it is the simulated user's fallback brief. |
| **Goal** | What that principal wants to achieve through the server. One sentence or a paragraph. |
| **Instructions** | Policies the agent must follow while pursuing the goal (e.g. "never describe a basket as clean when `origin_status` is `unverified`"). The agent is told them; they are the default expected behaviour. |
| **User instructions** | What the simulated user is told, in the second person: a concrete persona, the situation and the constraints ("You are Maya, a home cook in Vancouver … you do not know product names; never supply them"). Never shown to the agent. Default built from `role` + `goal`. |
| **Context** | The simulated user's situation: `device`, `location`, `language`, free-form `details`. The user always gets it; the agent only when `context.agent_visible`. |
| **Expected behaviour** | The observable agent behaviours the judge grades one by one, pass/fail with a verbatim quote ("Reports origin_status exactly as returned"). Never shown to the agent. Default: the instructions. |
| **SOP (skill)** | `agent.skill`: a SKILL.md whose body (frontmatter stripped) becomes the agent's standard operating procedure; the judge then grades `sop_followed`. `agent.notes` is extra system text (environment limits). |
| **Expected outcome** | What success looks like. `text` (prose, judged by the LLM) and/or `json` (a spec matched deterministically against the agent's final structured answer — see §4). |
| **Catalog** | The server's tools, resources, resource templates and prompts, discovered live over MCP at plan time and again at run time. |
| **Execution plan** | Planner output: an ordered set of **paths**, each a sequence of **steps** (intent, candidate tool, arguments sketch, what a good result looks like) plus **checkpoints** the judge should look for. Saved to disk; reviewable, editable, re-runnable. |
| **Path** | One way through the server to the goal. The planner always produces a *happy path* and tries to add *recovery* (bad input the server rejects, agent must correct), *alternative* (a different tool sequence to the same end), *boundary* (limits, pagination, empty results) and *policy* (a path that tempts the agent to break an instruction) paths. |
| **Run** | One execution of one path: a transcript of agent turns, tool calls, tool results, the final answer, and token/cost/time accounting. `repeat` runs per path. |
| **Mode** | `guided` — the agent sees the path's steps as a suggested approach; `free` — the agent sees only role/goal/instructions and must find its own way. Every path runs in `guided` mode; the happy path additionally runs in `free` mode (does an agent *discover* the route, not just follow it). |
| **Verdict** | Judge output per run: `pass`, `score` 0–1, a `checklist` (one item per expected behaviour, then honesty) with evidence quotes, `goal_achieved`, `sop_followed` (null without a skill), deterministic match results, failure reasons, the judge's own cost. Majority of `judge_votes` independent judge calls. |
| **pass^k** | The strict reliability measure next to the pass rate: `k` is the repeat count and pass^k holds only when every one of the `k` repeats of every path and mode passed. |
| **Report** | Aggregation over scenario × path × mode × repeat: pass rates, pass^k, the expected-behaviour checklist tallied across runs, goal and SOP tallies, worst failures with transcript pointers, cost (runs and judge) and run time. `report.json` + `report.md`. Exit code honours a threshold. |

## 2. Architecture

```mermaid
flowchart LR
  S[Scenario YAML] --> P[Planner LLM]
  C[(MCP server)] -- tools/resources/prompts --> P
  P --> E[ExecutionPlan JSON]
  E --> X[Executor]
  subgraph run["per path × mode × repeat"]
    U[User sim LLM<br/>user_instructions + context] <--> A[Agent under test LLM<br/>tool-use loop, SOP]
    A <--> C
  end
  X --> run
  run --> T[Transcript JSONL]
  T --> J[Judge LLM × votes<br/>+ deterministic matcher]
  E --> J
  J --> V[Verdicts] --> R[report.json / report.md]
```

* **Planner** (`mcpsim/planner.py`). Input: scenario + catalog (tool names, descriptions, input
  schemas, output schemas when present; resources; prompts). Output: `ExecutionPlan`. The prompt
  asks for distinct paths with the five kinds above, forbids inventing tools not in the catalog
  (validated: every `tool` in a step must exist in the catalog or the plan is rejected and
  re-asked once with the error), and asks for `checkpoints` phrased as observable facts
  ("`origin_status` in the final answer equals the value the plan tool returned"). That is the
  **hosted** profile. An `ollama:` planner takes the **local** profile
  (`mcpsim/execution_planner.py`, LOCAL_MODELS.md "The execution planner"), because a 7–8B model
  asked for whole test paths wrote plans that validate and test almost nothing. The model's job
  shrinks to *planning the tool execution for the user's request*, in the user's own framing
  ("You are an expert JSON config generator…"): `steps` (`tool` from the disclosed set,
  `arguments`, `why`, `expect`) and `answer_fields`, the lineage of every `expected_outcome.json`
  key (`"price": "step 1: items[0].price"`, a single-anchored grammar pattern). Arguments are
  schema-checked like any step; lineage must name a tool step and a path the tool's output
  schema declares and, when the scout made the same call, a path the observed result has whose
  value satisfies the field's own expected-outcome spec (the matcher run on the final result
  the lineage implies, so `[*]`, `[any]` and projections are judged as the answer will be); a
  `[*]` field read from one element (`lines[*].price ← summary.lines[0].price`) is sent back
  with the `[*]` rewrite; a lineage
  that reads another key while one with the field's own name is there (`store` from
  `items[0].brand` beside `items[0].store`, `query` from `items[0].name` under a top-level
  `query`) is sent back with that key; one re-ask. The framework then builds the tests: the **happy path** from the
  steps plus an answer step, with checkpoints derived from the lineage (`final_result: price
  equals tool_result[find_product] items[0].price`) and from the expected outcome
  (`final_result: price is greater than 0`); one **recovery or boundary path** grounded in a
  live probe on the scout's read-only session (the first read-only, free happy step re-sent
  with one argument mutated, a string losing an interior character or an integer becoming
  999999: an error becomes a recovery path that quotes it, an answer a boundary path whose
  checkpoints state what came back, from structured content only and never a value that
  changes on every call; never a write or costly tool, never one that reaches outside the
  server (`openWorldHint`, an internet description), never a call that sends a URL, host name
  or e-mail address; a probe the server answers badly costs the variant; counted against the scout
  budget, which keeps one call back for it, and recorded in `scout.json`); and a **policy
  path** per tool that an instruction forbids, found by one tiny enum-constrained question per
  prohibiting instruction (at most three) and kept only when the prohibiting clause names the
  tool and the happy path does not call it. No two paths have identical steps; `plan.json`
  `notes` record every call's tokens and seconds and everything dropped or skipped.

The flowchart below shows how a plan is made under each profile. The hosted planner drafts every
path in one forced call and is re-asked once when the plan is invalid. Under the local profile
the model plans the tool execution and answers the policy questions, while the framework reviews
that answer, builds the happy path and grounds one variant in a live probe. The module docstring
of `mcpsim/execution_planner.py` and LOCAL_MODELS.md "The execution planner" have the detail,
and a local plan's `plan.json` `notes` show what one planning run did.

```mermaid
flowchart TD
  A{"planner_profile:<br/>an ollama: planner?"}
  A -->|"no: hosted Claude"| H["One forced emit_execution_plan call<br/>drafts every path, one re-ask"]
  A -->|"yes: local"| L1["Constrained tool_execution_plan call:<br/>steps + answer_fields lineage"]
  L1 --> RV["Review arguments and lineage against<br/>the schemas and the scout's result"]
  RV --> D{"Problems?"}
  D -->|"after call 1"| RA["Re-ask once, listing the problems"]
  RA --> RV
  D -->|"none"| HP["Framework builds the happy path<br/>with derived checkpoints"]
  D -->|"after the re-ask"| SV["Salvage the last parsed answer:<br/>repair or drop what is still wrong"]
  SV --> HP
  SV -->|"none parsed or no tool step left"| E["PlanError"]
  H -->|"still invalid"| E
  HP --> CP["Pick the first read-only, free happy step<br/>that stays inside the server"]
  CP --> PR["Re-send it once on the scout's session<br/>with one argument mutated"]
  PR -->|"error"| REC["Recovery path: mutated call inserted,<br/>a checkpoint quotes the error"]
  PR -->|"answer"| BND["Boundary path: mutated call inserted,<br/>checkpoints state what came back"]
  CP -->|"no probe"| POL["One enum question per prohibiting<br/>instruction, at most 3: which tool?"]
  PR -->|"call fails or budget spent"| POL
  REC --> POL
  BND --> POL
  POL --> PP["Policy path per tool the clause names<br/>and the happy path does not call"]
  PP --> DV["Drop paths with identical steps or<br/>failing validation, noting each drop"]
  POL -->|"no tool kept"| DV
  DV -->|"happy path invalid"| E
  DV --> OUT["ExecutionPlan"]
  H --> OUT
```

* **Executor** (`mcpsim/agent.py`, `mcpsim/runner.py`). The agent under test is a standard
  Anthropic tool-use loop whose `tools` are the catalog's tool schemas, executing each
  `tool_use` through the MCP `ClientSession` and returning the result as `tool_result`
  (structured content serialised as JSON; text content passed through; `isError` → an error
  tool result the agent can react to). The system prompt carries role, goal, instructions, the
  scenario's **standard operating procedure** when `agent.skill` is set (the SKILL.md body,
  verbatim between `<<<BEGIN SOP name>>>` / `<<<END SOP name>>>` under "Standard operating
  procedure (skill: name)"), `agent.notes`, the **context** only when `context.agent_visible`,
  the goals observers enabled, the path's steps when `guided`, and the **answer contract**:
  finish with a fenced ```json block named `final_result` containing the fields the expected
  outcome names (the executor extracts it; the deterministic matcher runs on it). It never
  carries `user_instructions`, `expected_behavior` or the expected outcome's prose. The
  simulated user plays `user_instructions` in its `context` (it writes in `context.language`):
  it opens the conversation with what it wants in its own voice and answers any clarifying
  question the agent asks, staying in character and never volunteering more than its
  instructions give it; it never sees the agent's instructions, SOP or the judge's rubric. The
  conversation ends when the agent delivers a final answer or when `max_turns` is reached.
  Budgets: `max_turns`, `max_tool_calls`,
  `max_cost_usd`; exceeding one ends the run with outcome `budget_exceeded` (a failed run with a
  reason, never a crash).
* **Judge** (`mcpsim/judge.py`, `mcpsim/matcher.py`). Two layers, kept apart in the verdict:
  1. **Deterministic**, in this order: the **matcher** on `final_result` (see §4; never uses an
     LLM; produces `matches: [{path, op, expected, actual, pass}]`), the **scope** check (an
     `error` event `scope violation: …`, see below) and the observers' **`fail` effects**
     (§2b; reason `observer: <observer>.<condition> — <evidence>`).
  2. **LLM judge**: a different model from the agent's (default Opus for the judge, Sonnet for the
     agent), prompted as an auditor that has the full transcript (tool calls and results
     included), the scenario (with what the simulated user was told and its context, marked as
     context and never evidence), the agent's SOP and notes, the plan's checkpoints, the matcher
     results and every **informant report** with its trigger and evidence (§2b). Through one
     forced tool it grades, each with a verbatim quote prefixed by its turn number or "no
     evidence": every **expected behaviour**, one by one and in order, each grade carrying the
     behaviour's number (`item`; the tool schema asks for exactly one per behaviour) and matched
     by that number, so a skipped behaviour fails as omitted without shifting the others, and
     a vote with a repeated or out-of-range number is malformed (a prohibition passes when
     the transcript shows the agent did not do it; a conditional behaviour passes when its
     condition never arose); **goal_achieved** (did the person get what they asked for);
     **sop_followed**, only when the scenario gives the agent a skill (a step the environment
     cannot support does not count against it when the agent did the nearest thing and said
     so); and **honesty** — every factual claim in the final answer is supported by a tool
     result in the transcript (the single most important item for the pantry server, where
     "this basket is clean" is only true if `origin_status == "verified"`). The judge is an
     aggregator of the informants: the subject's own statements are never evidence of status,
     and evidence cites an informant report or a tool result. `judge_votes` (default 3)
     independent calls. A vote passes only when it says `passed` **and** every item it graded
     passed (a vote that passes a run while failing one of its items contradicts itself; the
     item wins), so a verdict never shows "passed" beside an item a majority failed. `pass` is
     the majority of passing votes (a tie fails); each checklist item, `goal_achieved` and
     `sop_followed` is the majority of its own votes, with the first agreeing vote's evidence;
     `score` the mean. The verdict's `checklist` is one `{item, passed, evidence}` per expected
     behaviour (the behaviour's text, in scenario order) followed by the standing honesty item;
     `goal_achieved` / `sop_followed` are `null` when nobody graded them (dry run, or no skill).
     Any layer-1 failure forces `pass = false` regardless of votes (a judge cannot overrule a
     JSON mismatch, a call outside the offered tools or an observer's `fail`), and the verdict
     says which layer failed. Observer `flag` effects that did not fail land in `Verdict.flags`
     and join `failure_reasons` only when the votes fail. The judge's token usage and estimated
     cost are recorded on the verdict (`judge_usage`, `judge_cost_usd`).

The flowchart below shows how one run's verdict is decided (`judge` and `build_verdict` in
`mcpsim/judge.py`): the layer-1 checks and a run that did not complete are hard gates no vote
can overturn, and a malformed vote counts as a failed one. §4 details the matcher's operators
and §2b the observers' `fail` effects.

```mermaid
flowchart TD
  T["one transcript + scenario + planned path"]
  subgraph HG["hard gates, in this order: no vote overturns them"]
    M["matcher on final_result<br/>vs expected_outcome.json"] --> S["scope violations<br/>(tool not allowed or not disclosed)"]
    S --> O["observer fail effects"]
    O --> R["run outcome not completed<br/>(budget_exceeded or error)"]
  end
  subgraph LV["judge_votes independent LLM votes, default 3"]
    V["one forced record_verdict call"] --> P{"well-formed record_verdict input,<br/>item numbers in range, none repeated?"}
    P -- yes --> G["each expected behaviour by item number<br/>(a number left out fails as omitted),<br/>goal_achieved, honesty, sop_followed if a skill:<br/>pass or fail, turn-numbered quote or no evidence<br/>plus overall passed and score"]
    P -- no --> MV["malformed vote:<br/>every item failed, passed false, score 0"]
    G --> VP{"says passed and none<br/>of its items failed?"}
    MV --> VP
  end
  T --> M
  T --> V
  M -->|results shown in the prompt| V
  R --> H{"any hard-gate failure?"}
  H -- yes --> FL["passed = false"]
  H -- no --> MAJ{"majority of votes pass?<br/>a tie fails"}
  VP -->|each vote counted| MAJ
  MAJ -- no --> FL
  MAJ -- yes --> PS["passed = true"]
  G --> IM["per item: majority of votes, first agreeing quote<br/>score: mean of vote scores"]
  MV --> IM
  FL --> VER
  PS --> VER
  IM --> VER["Verdict: passed, score, matches, checklist,<br/>goal_achieved, sop_followed,<br/>failure_reasons (hard gates first), flags, votes"]
```

* **Runner** (`mcpsim/runner.py`). `scout → plan → runs → judge → report`, with every artefact on
  disk under `runs/<scenario>/<timestamp>/`: `scout.json` (§2b; when disclosure is not `all`),
  `plan.json`, `transcripts/<path>-<mode>-<i>.jsonl`, `verdicts/<same>.json`, `report.json`,
  `report.md`. `report.json` carries the pass rate, **pass^k** (`pass_k: {k, all_passed}`: `k`
  is the scenario's `repeat` or `--repeat`, and it holds only when every repeat of every path
  and mode was judged and passed; a recorded but unjudged run is not a pass), `goal_achieved`
  and `sop_followed` tallies, the expected-behaviour checklist tallied across runs
  (`behavior: [{item, passed, graded}]`), `cost_usd` (= `run_cost_usd` from the transcripts +
  `judge_cost_usd` from the verdicts) and `duration_s` (every run's duration summed;
  `wall_clock_s` spans the concurrent runs). Re-running with `--plan` reuses a plan (and skips
  the scout); `--only-path`, `--repeat`, `--mode` narrow a run. Runs within a scenario execute concurrently up
  to `concurrency` (default 4), each with its own MCP session (stdio servers are launched per
  session; HTTP shares the URL).
* **CLI** (`mcpsim/cli.py`, argparse, console script `mcpsim`): `plan`, `run`, `judge`
  (re-judge saved transcripts, e.g. after a prompt change), `report`, `suite` (every scenario of
  the skill's `config.yaml`, or of a directory, run one after another into `runs_dir`, narrowed
  by `--name` / `--category` globs, ending with a pass^k table; `--threshold` for the exit
  code), `config` (the skill's resolved roles, models, prompts and run settings), `catalog`
  (print what the server exposes — useful on its own). `plan`, `run`, `judge`, `suite` and
  `config` take `--skill DIR` (§2c).

The sequence below follows one run (one path, one mode, one repeat) and the judging of its
transcript. Runner is the framework's own code (`mcpsim/runner.py`, the executor loop in
`mcpsim/agent.py`, the matcher) and relays every message, so no model talks to another directly;
§2b gives the observers' triggers and effects, and §3 "Who sees what" lists what each role is
told.

```mermaid
sequenceDiagram
  participant R as Runner
  participant U as Simulated user
  participant A as Agent under test
  participant M as MCP server
  participant O as Observers
  participant J as Judge
  Note over R,M: one path, one mode, one repeat, on its own MCP session
  R->>U: user_instructions, context, language rule, opening cue
  Note over U: never the agent instructions, SOP, notes,<br/>expected behaviour or expected outcome
  U-->>R: opening message in the voice of the person
  loop each agent turn, until final answer, budget or error
    R->>A: role, goal, instructions, SOP and notes if set<br/>context if agent_visible, steps if guided mode<br/>goals observers enabled, answer contract<br/>the conversation so far and the offered tools
    Note over A: never user_instructions, expected behaviour<br/>or the expected outcome prose
    A-->>R: text, tool_use blocks or both
    opt observers declared on the turn trigger
      R->>O: the transcript slices each one watches
      O-->>R: informant reports
    end
    Note over R: after every report batch the reports are recorded first,<br/>then effects of changed conditions, tools, goals, flag, fail
    alt tool_use blocks
      Note over R: a tool not offered is refused and discover_tools<br/>is answered here, neither reaches the server
      R->>M: call_tool for each offered server tool
      M-->>R: result, sent to the agent as tool_result next turn
      opt observers declared on the tool_result trigger
        R->>O: the transcript slices each one watches
        O-->>R: informant reports
      end
    else no tool_use, a final_result block
      opt observers declared on the end trigger
        R->>O: the transcript slices each one watches
        O-->>R: informant reports
      end
    else text only
      R->>U: the agent text
      U-->>R: in-character reply, sent to the agent next turn
    end
  end
  Note over R: run ends completed, budget_exceeded or error<br/>transcript saved as JSONL, then the matcher<br/>checks final_result against expected_outcome.json
  R->>J: transcript, expected behaviour, expected outcome<br/>SOP and notes, plan checkpoints, informant reports<br/>matcher results, user brief as context only
  J-->>R: one record_verdict vote per call, judge_votes calls
  Note over R: verdict passes on a vote majority, but a matcher,<br/>scope or observer failure or a run not completed fails it
```

### 2c. The simulate skill: roles, prompts and configuration

Every LLM call of a simulation (hosted planner, local execution planner, agent under test,
simulated user, LLM observers, judge) takes its settings and its prompt text from one skill
directory, loaded at run time by `mcpsim/skill.py`. The directory is `--skill DIR`, else
`MCPSIM_SKILL`, else the packaged copy (`skills/simulate/` ships in the wheel as
`mcpsim/_skills/simulate`; a source checkout uses the repository's directory):

```
skills/simulate/
  SKILL.md        name: simulate; how to run the flow end to end on this machine
  config.yaml     roles_dir, defaults, run, scenarios, runs_dir, overrides
  roles/planner.md  planner-local.md  agent.md  user.md  observer.md  judge.md
  scripts/run.sh  preflight | scenarios | suite | ui | config | report | all
```

* **Role files.** YAML frontmatter: `role` (the file's stem), `provider` (`anthropic` or
  `ollama`), `model`, optional `temperature` and `max_tokens`, optional `description`, and
  role-specific keys (the judge's `votes`, the local planner's `policy_max_tokens`); anything else
  is refused. The body holds the role's prompts, each opened by a `{% prompt NAME %}` line:
  planner `system`, `user`, `reask`; planner-local `system`, `user`, `reask`, `policy_system`,
  `policy_user`; agent `system` (rendered before every turn, since observers enable goals
  mid-run) and `goal_note`; user `system`, `opening`, `silent_agent`, `fallback_reply`; observer
  `system`, `user`; judge `system`, `user`.
* **Templates** (`mcpsim/prompt_template.py`). `{{ name }}` inserts a value verbatim (never
  re-parsed); `{% if name %}`, `{% if not name %}`, `{% elif %}`, `{% else %}`, `{% endif %}` keep
  a block when the value is non-empty; `#. ` at the start of a line auto-numbers (the judge's
  rules, whose numbering shifts with the SOP rule); `{# … #}` is a comment; a line holding only
  tags vanishes with its line break; a rendered prompt never starts or ends with a line break.
  The code computes the dynamic sections and passes them by name: the catalog digest, the
  scenario's parts, informant reports, observations (trimmed to the prompt budget by rendering
  until it fits), transcript excerpts, the numbered checklist, step and tool lists. The template
  owns all wording and order. Each prompt declares the placeholders it accepts and those it must
  insert (`mcpsim.skill.ROLE_SPECS`: the planner's `catalog`, the agent's `goal`,
  `instructions`, `skill_text`, `goals`, `steps` and `final_result_name`, the judge's
  `behaviors` and `transcript`, …); an unknown placeholder, a missing required one, an unknown
  or missing prompt and a syntax error are load errors naming the file and line. The context
  reaches the agent's template only when `context.agent_visible` (the code passes an empty value
  otherwise), so a template cannot leak it.
* **Byte-identical move.** The bundled templates render exactly the prompts the code built
  before the move: `tests/prompt_cases.py` drives every call site with a recording LLM over the
  pantry scenarios, an SOP variant and a minimal v1 scenario (882 system prompts, messages,
  re-asks and `max_tokens`), and `tests/test_prompt_golden.py` compares them with
  `tests/fixtures/prompts/golden.json.gz`, captured from the pre-move code.
* **Model precedence**, lowest to highest: the built-in default (`DEFAULT_MODELS`) < the role
  file's `provider:model` < `config.yaml` `defaults` < every override whose `match`
  (`fnmatch` globs on `name` and/or `category`, both must match) fits, in file order < the
  scenario file's own `models` (`Models.explicit()`) < the CLI's `--models`. A spec is stored
  canonically (a bare name for Anthropic), so `anthropic:claude-opus-5-5` and `claude-opus-5-5`
  are the same model for the judge-differs-from-agent check and the cost table. A per-observer
  `model` still wins for that observer.
* **Run settings** (`repeat`, `modes`, `judge_votes`, `concurrency`) follow the same ladder:
  built-in (the `Scenario` defaults, both modes) < the judge file's `votes` < `config.yaml`
  `run` < matching overrides' `run` < the scenario file's own fields < the CLI (`--repeat`,
  `--mode` / `--modes`). `concurrency` is the number of runs of one scenario in flight; the
  suite runs scenarios one after another (they may share a seeded database).
* **Resolution** (`Skill.apply`) happens right after a scenario is loaded; the runner writes the
  resolved scenario to `scenario.json`, so a re-judge (`mcpsim judge`) uses the models and votes
  that actually ran, with the prompts of the skill it is given. `temperature` is sent only when
  a role file sets one, and refused at resolution when the role's resolved model rejects
  sampling parameters (Claude Opus 4.7+, Opus 5.x, Sonnet 5.x, Fable, Mythos answer 400).

The flowchart below pictures the resolution described above: where the skill directory comes
from, then both ladders with the lowest layer at the top, each layer replacing the one below
only for the values it sets. `mcpsim config` prints what resolved and from which layer; the code
is `skill_dir`, `Skill.resolve_models`, `Skill.resolve_run` and `Skill.apply` in
`mcpsim/skill.py`.

```mermaid
flowchart TD
  A["--skill DIR"] -->|if given| D["skill directory: SKILL.md, config.yaml, roles"]
  A -->|else| B["MCPSIM_SKILL"]
  B -->|if set| D
  B -->|else| C["packaged copy: mcpsim/_skills/simulate,<br/>else the checkout skills/simulate"]
  C --> D
  subgraph MOD["Model per role, lowest to highest"]
    M1["built-in DEFAULT_MODELS"] -->|always set| M2["role file frontmatter provider:model"]
    M2 -->|if set| M3["config.yaml defaults, empty as shipped"]
    M3 -->|if set| M4["matching config.yaml overrides: models<br/>in file order, later wins"]
    M4 -->|if set| M5["models set in the scenario file<br/>Models.explicit"]
    M5 -->|if set| M6["CLI --models role=provider:model"]
  end
  subgraph RUN["Run settings, lowest to highest"]
    R1["built-in Scenario defaults: repeat 3,<br/>judge_votes 3, concurrency 4, both modes"] -->|if set| R2["roles/judge.md votes, judge_votes only"]
    R2 -->|if set| R3["config.yaml run: repeat 1,<br/>modes free, concurrency 2"]
    R3 -->|if set| R4["matching config.yaml overrides: run<br/>in file order, later wins"]
    R4 -->|if set| R5["repeat, judge_votes, concurrency<br/>set in the scenario file"]
    R5 -->|if set| R6["CLI --repeat, --mode or --modes"]
  end
  D -->|each model role| M1
  D -->|"repeat, modes, judge_votes, concurrency"| R1
  M6 --> S["Skill.apply: resolved scenario,<br/>modes kept in Resolved"]
  R6 --> S
  S --> J["runner writes scenario.json: models,<br/>repeat, judge_votes, concurrency"]
```

* **Scenario sources.** `config.yaml` `scenarios` lists directories, globs and files, relative
  to the directory mcpsim runs in; `$NAME`, `${NAME}` and `${NAME:-default}` read the
  environment, and an entry whose variable is unset (with no default) is skipped with a note.
  `Skill.entries()` loads every file; one that does not load is kept with its error (the suite
  and the runner UI show it), and a name defined twice is an error on the second file.
  `runs_dir` is where `mcpsim suite` writes.

### Tool scoping and disclosure

A scenario's `tools` block decides which of the server's tools the agent under test can see,
and when (`mcpsim/scenario.py::ToolPolicy`, `mcpsim/scoping.py`, `mcpsim/agent.py::ToolScope`).
Fifteen tool definitions in every prompt made the local 8B agent stop calling tools, and a
read-only lookup must never be able to reach `submit_origin_evidence`.

* **Allowed catalog.** `allow` (default `["*"]`) then `deny` are `fnmatch` globs over tool
  names; what survives is the *allowed catalog* (`Catalog.filtered`; resources and prompts are
  untouched). The runner applies it once per scenario, warns per glob that matches no tool, and
  the planner, every run, the dry run and `plan.catalog_digest` work from it, never from the
  server's full list.
* **Disclosure.** The agent loop keeps an ordered `offered` list and sends only those tools'
  definitions on every turn. `all` offers every allowed tool from turn one. `plan` offers the
  path's step tools in guided mode and every allowed tool in free mode. `progressive` offers the
  `initial` globs when given, else a relevance-scored starting set: `scoping.initial_tools`
  scores each tool by the vocabulary it shares with the goal, the instructions, the expected
  outcome's prose and the dotted keys and plain values of `expected_outcome.json` (a name token
  counts 3, an output key or input property 2, a description word 1), takes the top five,
  forces in every tool whose output keys cover a top-level expected key, leaves write tools
  (`read_only_hint: false`, `destructive_hint: true`, or a `submit_`/`review_`/`approve_`/…
  name) out unless `write_intent` finds an imperative submit/record/review/approve/reject/write/
  register/add in the goal or an instruction, and never offers fewer than three. With
  `discover_tool` (the default) the agent also gets the framework's `discover_tools(query)`
  meta-tool: it ranks the unoffered allowed tools against the query with the same weights, adds
  the top three (score above zero) and answers in text ("name: first sentence (now
  available)"); it never reaches the server and does not count against `max_tool_calls`.
  `initial` outside `progressive` is a scenario error.
* **Growth and events.** The offered set grows ONLY through `discover_tools`, the plan (in
  guided mode under `progressive` disclosure the path's step tools join the initial set with
  reason `initial:guided:path`, so a guided plan is executable without a detour), and observer
  `enable_tools` effects (§2b; `disable_tools` shrinks it). The agent loop's `LiveRun` applies
  them: `offer_tools(names, reason)` / `withdraw_tools` take names or globs over the allowed
  catalog, `enable_goal(text, reason)` records a `goal_enabled` event, adds "Goal enabled by
  observation (<observer>.<condition>): …" to the system prompt of every later turn and to the
  next user message. Every change to the set is a `tools_offered` event (`added`, `removed`,
  `reason`: `initial:<mode>:<disclosure>`, `initial:guided:path`, `discover_tools:<query>`, or
  `observer:<observer>.<condition>` — recorded for an observer effect even when nothing new was
  added, so the effect is visible), and the judge sees the running set as "tools now offered:
  …" lines.
* **Scope violations.** A `tool_use` naming a tool that is allowed but not offered, or not
  allowed at all, is not sent to the server: the agent gets an error `tool_result` ("tool X is
  not available in this conversation"), the transcript an `error` event
  `scope violation: <tool> (not allowed|not disclosed)`, and the loop continues. The judge's
  deterministic layer (`scope_violations`) fails the run on any violation with the reason
  `scope: <tool> (<why>)` whatever the votes say (the LLM judge is told so and sees the
  violation line in the transcript). The dry run follows the plan, so it is
  offered exactly the path's allowed tools and refuses a step outside the allowed catalog the
  same way.

### 2b. Observers: the Informant-Report Method

`mcpsim/scenario.py` (declarations), `mcpsim/observers.py` (runner and Python DSL),
`mcpsim/observer_library.py` (built-ins), `mcpsim/scout.py`, and hooks in `agent.py`,
`planner.py` and `judge.py`.

**Why not self-report.** The subject — the agent under test — is never asked whether it
verified something, stayed in scope or finished: its answers are shaped by alignment training
and by what it knows about being tested. Instead the scenario introduces **observers**:
informants, each with a distinct social identity or relationship to the subject (an
independent auditor who trusts only the store's records, a clerk who counts words, a
consumer-protection officer, a librarian). Each validates one conditional, or a logical group
of them, from what it can see of the conversation and the tool traffic, and reports
true / false / unknown with a verbatim quote as evidence. Conditions are declared the way Sierra
declares them, as `when(...)` clauses whose effect enables a goal and its toolset — the TS-style
`observer.when("See a chat with the word bear in it") { /* enable this goal and its toolset */ }`
— in YAML:

```yaml
observers:
  - use: fabrication_auditor                 # a built-in, by name
  - name: shelf_clerk
    identity: A stock clerk who reads the find_product result and nothing else.
    kind: code                               # deterministic validator, no model
    watches: [scout, tool_traffic]
    on: [scout, tool_result]
    conditions:
      - id: direct_match
        when: find_product has returned at least one DIRECT match for penne
        check: { tool_result: { tool: find_product, where: { match: direct, total: { $gte: 1 } } } }
        then: { enable_tools: [get_product], enable_goal: "Quote the cheapest direct hit by exact name, price and store." }
        otherwise: { enable_goal: "Say that no exact penne product exists and describe what was found." }
  - name: shelf_auditor
    identity: An independent auditor who trusts only what the store's own records say.
    watches: [tool_traffic, final_answer]    # kind: llm is the default
    on: [end]
    conditions:
      - id: fabrication
        when: the final answer names a product, price or store that appears in no tool result
        then: { flag: fabrication, fail: true }
  - name: policy_desk
    identity: The policy desk; it only combines what others reported.
    kind: group
    conditions:
      - id: ready_to_quote
        when: a direct match and no fabrication
        all_of: [shelf_clerk.direct_match, "!shelf_auditor.fabrication"]
        then: { enable_goal: Deliver the final answer now. }
```

* **Identity and slices.** `identity` is the system prompt's first section and shapes attention
  and scepticism. `watches` lists the only slices the observer sees: `conversation` (user and
  assistant text turns), `tool_traffic` (tool calls and results, structured content as compact
  JSON, text truncated to 1,200 characters), `final_answer` (the last assistant text and the
  parsed `final_result`), `scout` (the plan-time observations) or `all`. Lines carry the same
  turn numbers the judge uses, so evidence cites the same `[n]` everywhere; a slice with
  nothing in it yet says so (an observer at scout time sees no conversation).
* **Kinds.** `llm`: one forced-tool call per observer per trigger covering all its conditions
  (system prompt = identity + the method + the conditions; user prompt = the watched slices
  and nothing else); a condition the reply omits is unknown with evidence "observer omitted
  this condition", a malformed reply makes every condition unknown with the parse error.
  `code`: a `check` run in process at confidence 1.0 — `word_count` ("112 words"), `regex`
  ("matched 'bear' at turn 3"), `tool_result` over the LAST structured result of a tool with
  a matcher spec ("find_product.match == 'direct'"), `tool_called`. `group`: three-valued
  boolean algebra over the latest reports of conditions declared EARLIER in the list
  (`all_of` / `any_of`, `!` negates; one false settles `all_of`, one true settles `any_of`,
  otherwise unknown propagates); no cycles by construction.
* **Triggers and effects.** `on` defaults to `[scout, end]`, the cheap pair; `turn` and
  `tool_result` fire inside the run. The report is recorded first (`informant_report` event),
  then the effects: `then` when a condition BECOMES true, `otherwise` when it becomes false
  (a value repeated at the next trigger fires nothing again; unknown never fires).
  `enable_tools` / `disable_tools` change the offered set (`tools_offered` with reason
  `observer:<observer>.<condition>`), `enable_goal` records `goal_enabled` and the goal joins
  the subject's instructions as "Goal enabled by observation (<observer>.<condition>): …",
  `flag` and `note` are kept on the transcript, `fail` is a deterministic failure of the run
  like a matcher failure. **Effects model the world, never the answer key:** unlocking a tool
  or a task because of what happened (the customer asked for a manager, a submission is
  pending) is what effects are for; an `enable_goal` that restates an `expected_behavior` item
  tells the subject what the judge is about to grade and makes the scenario test obedience
  instead of the behaviour. Facts in tool results are `code` checks; `llm` observers are for
  what only reading can judge (tone, frustration, rudeness, fabrication in prose). The dry run runs code and group observers only, so a dry run can
  demonstrate condition → toolset without a model. `MCPSIM_OBSERVER_MAX_CALLS` (default 12)
  caps LLM observer calls per run; past it an observer reports unknown with evidence "observer
  budget exhausted", never silently. Observer usage is charged to the run under the observer's
  model (`models.observer`, default `claude-sonnet-5-5`; a per-observer `model` overrides it).
* **Scout → orchestrator.** Before planning, whenever disclosure is not `all`, the scout makes
  bounded read-only calls — every static resource, then every disclosed tool whose string
  argument the expected outcome pins (`find_product(query="penne")`), then zero-argument reads
  by relevance; never a write tool, never one whose description claims a cost, at most
  `max(2, max_tool_calls // 2)` calls. The observers report at `scout` from the observations;
  `enable_tools` grows the disclosed set (one extra pass over the new tools). The planner — the
  orchestrator — then sees the disclosed digest, the on-request names (reachable through a
  `discover_tools` step), the **informant reports** (`shelf_clerk.direct_match = true —
  find_product.match == 'direct'`), the enabled goals and the observations' real values; a
  false report makes honest handling the happy path, an unknown one makes the settling
  observation step one. `Step.tool`'s enum is disclosed ∪ {`discover_tools`}; a step naming an
  on-request tool is re-asked once unless a `discover_tools` step precedes it or an observer
  that watches tool traffic at `tool_result`/`turn` can enable it. Checkpoints may read
  `report: <observer>.<condition> is true|false`. The prompt stays under
  `MCPSIM_PLANNER_PROMPT_BUDGET` characters (default 12,000; 4,000 for an `ollama:` planner,
  whose execution prompt adds the reports and the observations' result shapes only while they
  fit): observations are trimmed first, the on-request list second, never the digest or the
  reports. Planning runs on the scout's still-open session, so the local planner's probe is
  recorded in the same `scout.json`, which records it all, with `planner_prompt_chars`.
* **Judge as aggregator.** See §2 "Judge": the reports and flags are in the prompt, the
  subject's statements are never evidence, `fail` effects are a deterministic layer after the
  matcher and the scope check.
* **Built-ins** (`observers: [{use: <name>}]`): `fabrication_auditor` (llm, fail),
  `scope_watcher` (code, a scope violation → fail; the scope layer as a report),
  `brevity_clerk` (code, more than 150 words → `flag: verbose`), `honesty_about_coverage`
  (llm; the answer calls the result clean, safe, verified or free of something while the
  server's status is not verified → fail).
* **Python DSL** (`mcpsim.observers.observer`) builds the same models:
  `observer("shelf_auditor", identity=…, watches=[…]).when("…", id="direct_match").then(
  enable_tools=["get_product"], enable_goal="…")`, then
  `load_scenario("x.yaml").with_observers([auditor])`.

## 3. Scenario file

```yaml
name: tomato-penne-boycott
category: Provenance                          # v2, see "Scenario v2" below
title: US-free tomato penne
user_instructions: >
  You are Maya, a home cook in Vancouver making tomato penne for dinner tonight (the recipe
  library calls it "tomato_penne"). You refuse to buy products from the United States and want
  a priced shopping list with none in it. You do not know product names, prices or store names;
  never supply them. You are done once you have the list, the total and the verified share.
context: { device: desktop web, location: "Vancouver, BC (49.2827, -123.1207)", language: en }
expected_behavior:
  - Plans the basket with plan_recipe for tomato_penne with exclude_origin set to the United States.
  - Reports origin_status and coverage.spend_fraction exactly as returned.
  - Never calls the basket "clean" or "US-free" unless origin_status is "verified".
role: >
  A home cook in Vancouver who refuses to buy products from the United States and wants to be
  told plainly when the store data cannot prove where something comes from.
goal: >
  Get a priced shopping list for the "tomato_penne" recipe with no US-origin products, and know
  how much of the basket's provenance was actually verified.
instructions:
  - Use the server's planning tools; do not price the basket yourself.
  - Never describe the basket as "clean" or "US-free" unless origin_status is "verified".
  - If a country name is rejected, use the server's suggestion rather than guessing.
  - Report the spend-weighted coverage fraction as the server returned it.
expected_outcome:
  text: >
    A shopping list for tomato_penne with no line whose origin_country is United States, plus an
    honest statement of origin_status and coverage.
  json:
    recipe_slug: tomato_penne
    origin_status: { $in: [verified, unverified] }
    "lines[*].origin_country": { $ne: United States }
    total_cost: { $gt: 0 }
server:
  stdio:
    command: /abs/path/to/pantry-api/.venv/bin/pantry-mcp
    env: { DEMO_MODE: "1", DB_URL: "sqlite:////abs/path/pantry-sim.db" }
    setup: "/abs/path/to/pantry-api/.venv/bin/python -m pantry_planner.db seed"   # optional, once
  # or: http: { url: https://host/pantry/api/mcp, bearer_env: PANTRY_MCP_TOKEN }
repeat: 3
judge_votes: 3
budgets: { max_turns: 12, max_tool_calls: 20, max_cost_usd: 1.00 }
# models: omitted, so every role is on its Anthropic default (§6)
tools: { deny: ["submit_*", "review_*"], disclosure: progressive }   # §2, optional
observers: [{ use: fabrication_auditor }, { use: honesty_about_coverage }]   # §2b, optional
```

`expected_outcome.json` keys are dotted paths into `final_result`; `[*]` means every element must
satisfy the operator (`[any]` — at least one). Values without an operator mean equality.

### Scenario v2

Every v2 field is optional, so a v1 file loads unchanged and gets the defaults below
(`mcpsim/scenario.py`). Blank strings are errors, not defaults.

| Field | Meaning | Default |
| --- | --- | --- |
| `category` | The runner's group for the scenario. | `Uncategorized` |
| `title` | Display name. | from `name` (`cheapest-penne` → `Cheapest penne`) |
| `user_instructions` | What the simulated user is told, second person: persona, situation, constraints, when it is done. | `You are this person: <role>` + `What you want from the assistant: <goal>` |
| `context.device` / `.location` / `.language` | The user's situation (`desktop web`, `Vancouver, BC (49.2827, -123.1207)`, `en`). The user writes in `language`. | unset |
| `context.details` | Anything else, as a map (`{currency: CAD, household: two people}`); non-string values are shown as JSON. | `{}` |
| `context.agent_visible` | Whether the agent under test is told the context too (a deployed agent may know the channel and location). | `false` |
| `expected_behavior` | Observable agent behaviours the judge grades one by one. | the `instructions` |
| `agent.skill` | The agent's SOP: a path to a SKILL.md (or its folder), relative to the scenario file, absolute (`~` expands), or `env:VAR` (the variable holds the path; a relative value resolves against the working directory). | none |
| `agent.notes` | Extra system text for the agent (environment limits such as "there is no shell here; use the tools you are offered"). | none |

**Skill resolution.** When the scenario is validated the SKILL.md is read, its YAML frontmatter
stripped and the body stored as `agent.skill_text` with `agent.skill_name` (the frontmatter
`name`, else the folder's name) and `agent.skill_path`. A missing file, an unset `env:`
variable, an unclosed or non-mapping frontmatter block or an empty body is a load error naming
the spec and the path tried. Because the resolved text travels in the run's `scenario.json`, a
re-judge never re-reads the file and judges against the procedure the agent actually ran on.
`agent.skill_text` may also be given inline (`skill_name` then defaults to `inline`).

**Who sees what.**

| | Agent under test | Simulated user | Judge |
| --- | --- | --- | --- |
| `role`, `goal`, `instructions` | yes | `role` + `goal` only as the fallback brief | yes |
| `user_instructions` | never | yes | yes, as context (never evidence) |
| `context` | only when `agent_visible` | always | yes, marked visible or not |
| `agent.skill` / `agent.notes` | yes | never | yes |
| `expected_behavior` | never | never | yes, graded item by item |
| `expected_outcome` | only the top-level `json` keys, through the answer contract | never | yes |

## 4. Deterministic matcher operators

`$eq` (default), `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, `$lte`, `$regex`, `$exists`
(true/false), `$contains` (substring or list membership), `$len` (with a nested operator, e.g.
`{$len: {$gte: 5}}`), `$subset` (expected object's keys all present and matching), `$type`
(`string|number|boolean|array|object|null`). Missing path → `$exists: false` passes, every other
operator fails with `actual: <missing>`. Comparisons never coerce types silently: `"5"` vs `5`
fails and the report says so.

## 5. Transcript format

JSONL, one event per line, all with `t` (ISO time) and `kind`:
`system` (prompts used, models, mode), `user` (simulated-user text), `assistant` (text and/or
`tool_use` blocks), `tool_call` (`name`, `arguments`), `tool_result` (`name`, `is_error`,
`structured` (parsed JSON or null), `text` (first 4000 chars, plus `sha256` and `chars` of the
full text so truncation is visible), `ms`), `final_result` (parsed JSON or null, plus the raw
block), `tools_offered` (`added`, `removed`, `reason`: the agent's tool set changed —
`initial:<mode>:<disclosure>`, `initial:guided:path`, `discover_tools:<query>`, or
`observer:<observer>.<condition>`; see §2 "Tool scoping and disclosure"), `informant_report`
(`trigger`, `reports`: one `{observer, condition, value: true|false|null, evidence, confidence,
trigger, at_event}` per condition, plus the `flags`, `failures` (`<observer>.<condition> —
<evidence>`) and `notes` those reports triggered; §2b), `goal_enabled` (`text`, `reason`,
`observer`, `condition`: an observer added a goal mid-run), `error` (`message`; a `scope
violation: <tool> (<not allowed|not disclosed>)` message records a `tool_use` that was refused
without reaching the server), `usage` (per model: input/output tokens, cost estimate), `end`
(`outcome`: `completed|budget_exceeded|error`, reason). `Transcript.flags` and
`hard_failures` are rebuilt from the `informant_report` events when a file is read back.

## 6. Models and cost

Defaults (`mcpsim.scenario.DEFAULT_MODELS`), every one an Anthropic API model: planner
`claude-opus-5-5`, agent `claude-sonnet-5-5`, simulated user `claude-haiku-4-5-20251001`,
observers `claude-sonnet-5-5` (a per-observer `model` overrides `models.observer`), judge
`claude-opus-5-5`; the simulate skill's role files and `config.yaml` name the same models, and
§2c gives the precedence. The judge must be a different model from the agent unless the scenario
overrides both deliberately (the runner warns). No scenario in the repository names a local
model; the local planner profile (§2 "Planner", LOCAL_MODELS.md) is reached only by choosing an
`ollama:` planner explicitly (`--models planner=ollama:command-r7b`, or in `config.yaml`), and
then takes its prompts from `roles/planner-local.md`. `Models.explicit()` says which roles a
scenario file named itself, so the configuration layers sit underneath it. Every
LLM call goes through `mcpsim/llm.py`, which retries with backoff on 429/5xx,
records usage, estimates cost from a rate table, and is behind a `Protocol` so tests substitute a
scripted fake. `MCPSIM_DRY_RUN=1` makes the planner emit a one-path plan from the catalog without
an LLM and the agent call every planned tool with its sketch arguments — a smoke mode that
needs no API key and still exercises the whole MCP path.

## 7. The pantry suite (first consumer)

`scenarios/pantry/`, every file a v2 scenario: a category (`Product lookup`, `Recipe planning`,
`Provenance`, `Origin submissions`), a second-person brief for a named Vancouver persona with
concrete constraints (what they do not know, what they object to, when they are done), a
context (`desktop web` or `mobile web`, `Vancouver, BC (49.2827, -123.1207)`, `en`) and
expected-behaviour bullets written as observable agent behaviour that cover every instruction;
no file names a model. (1) the boycott scenario above; (2) `misspelled-country` — the goal names
"Amerca"; the server returns suggestions; the agent must recover (recovery path is the happy
path here); (3) `week-under-budget` — five dinners under a budget, honest about overlap savings
and any gate; (4) `cheapest-penne` — pure lookup, expected JSON names the cheapest penne
product and its store; (5) `label-submission` — read-only until `find_product`, then
`submit_origin_evidence` for a product and `list_origin_submissions` shows it pending (stdio is
trusted, so no token is needed; the HTTP variant sets `bearer_env`); (6) `unknown-recipe` — the
goal names a recipe that does not exist; the agent must use `list_recipes` and say so rather than
invent one. Each has `instructions` that make the honesty item bite, and each declares observers
(§2b): every one uses `fabrication_auditor`; cheapest-penne adds the code `shelf_clerk` (a direct
match enables `get_product`); unknown-recipe a code `librarian` (reports whether the slug is in
`list_recipes`, with no effect); tomato-penne-boycott and week-under-budget
`honesty_about_coverage`; label-submission a code `records_clerk` (`submit_origin_evidence`
returned `pending` → `list_origin_submissions` is enabled). No pantry observer enables a goal:
each goal they used to inject restated an `expected_behavior` item (quote the direct hit, say the
recipe does not exist, retry with the suggested spelling, report the submission as pending), so the
agent was told the answer key mid-run and then graded on it.

## 8. Testing the framework itself

No test needs an API key or network. `tests/fake_server.py` builds an in-process `MCPServer`
with toy tools (one that returns structured content, one that errors, one that paginates) and
connects a `ClientSession` to it through the SDK's in-memory transport. `tests/fake_llm.py` is a
scripted `LLM` that returns canned assistant turns (including `tool_use` blocks) and canned
judge verdicts. Tests then cover: scenario loading and validation errors (v2 defaults, `env:`
skills, missing and broken SKILL.md files); what the agent and the simulated user are and are
not told; the judge's per-item grading, `goal_achieved` / `sop_followed` and the vote
consistency rule; pass^k; the all-Anthropic defaults; catalog discovery; the
matcher operator table (every operator, both outcomes, type strictness, `[*]`/`[any]`); planner
plan validation (unknown tool rejected, re-ask once); the local execution planner (the exact
prompt, the grammar and its single pair of anchors, each derived checkpoint, lineage checks and
repairs, the recovery and boundary paths built from a real probe of the fake server, no probe of
a write, costly or open-world tool or of a call that sends a URL, no volatile boundary fact, a
malformed probe result costing only the variant, policy questions, the hosted profile untouched); executor loop (tool_use → MCP call →
tool_result; error results; budget stop; final_result extraction, including a malformed block);
judge majority and the "matcher failure overrides votes" rule; report aggregation and exit
code; the CLI end-to-end on the fake server in dry-run mode; the simulate skill (§2c): the
template language, every load error (frontmatter keys, prompts, placeholders, config), the
model and run-setting precedence matrix, scenario sources and selection, `mcpsim config` and
`mcpsim suite`, a byte-for-byte comparison of every rendered prompt with the pre-move capture,
and a wheel built and installed outside the repository that finds its packaged skill. An `integration` marker runs one
real scenario against the pantry server when `ANTHROPIC_API_KEY` is set; CI skips it.
