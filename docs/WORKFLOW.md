# How an mcp-sim simulation flows

Two views of the same thing: the end-to-end pipeline from a scenario file to a report, and one run seen
as a sequence. Vocabulary is defined in [DESIGN.md](DESIGN.md); the observer model is the
Informant-Report Method (observers with their own identities report on the subject; the subject is never
asked to report on itself).

## End to end

```mermaid
flowchart TD
    subgraph S["1 · Scenario (YAML)"]
        S1["role · goal · instructions"]
        S2["expected_outcome<br/>prose for the judge + JSON spec for the matcher"]
        S3["tools: allow / deny · disclosure = all | plan | progressive"]
        S4["observers: identity · when(condition) → then{enable_tools, enable_goal, flag, fail}"]
        S5["models: planner = ollama:command-r7b (local) · agent, user, observers, judge = API"]
    end

    subgraph C["2 · Catalog"]
        MCP[("MCP server<br/>(pantry-mcp over stdio or /mcp over HTTP)")]
        CAT["tools · resources · prompts<br/>discovered over MCP"]
        ALLOW["allowed catalog<br/>allow/deny applied"]
    end

    subgraph O["3 · Scout and informants (plan time)"]
        SC["Scout — read-only, no LLM:<br/>resources · lookups from expected values · zero-arg reads"]
        OB{{"Observer LLMs with identities<br/>kind = llm | code | group"}}
        REP[/"informant reports<br/>condition → true / false / unknown + verbatim evidence"/]
        EFF["effects<br/>enable_tools · enable_goal · flag · fail"]
        DISC["disclosed toolset + enabled goals<br/>(the rest reachable via discover_tools)"]
    end

    subgraph P["4 · Orchestrating planner (local Cohere)"]
        PL["plan from the disclosed tools and the reports"]
        PLAN[("plan.json<br/>paths: happy · recovery · alternative · boundary · policy<br/>steps with tool + arguments · checkpoints")]
        VAL["validate: tool ∈ catalog enum · argument keys and types ·<br/>recovery path contains a failing step · checkpoint shape"]
    end

    subgraph R["5 · Runs (path × mode × repeat)"]
        US["Simulated user LLM<br/>plays the role"]
        AG["Subject agent LLM<br/>tool-use loop over the offered tools"]
        SESS["MCP ClientSession<br/>tools/call JSON-RPC"]
        OBR{{"Observers at turn / tool_result / end"}}
        EFF2["effects mid-run:<br/>reveal tools · add goals · flag · fail"]
        TR[/"transcript.jsonl<br/>user · assistant · tool_call · tool_result · tools_offered ·<br/>informant_report · goal_enabled · final_result · end"/]
    end

    subgraph J["6 · Judge"]
        DM["deterministic: JSON matcher on final_result"]
        SV["deterministic: scope violations<br/>(tool not allowed / not disclosed)"]
        OF["deterministic: observer fail effects"]
        JV["judge LLM × N votes<br/>reads the transcript and the informant reports,<br/>never the subject's own claims as proof"]
        VER[/"verdict.json<br/>passed · score · matches · checklist with evidence · reasons"/]
    end

    subgraph RP["7 · Report"]
        AGG["aggregate: pass rate by path × mode · worst failures · cost"]
        MD[/"report.md · report.json · exit code against a threshold"/]
    end

    S3 --> ALLOW
    MCP --> CAT --> ALLOW
    ALLOW --> SC --> OB --> REP --> EFF --> DISC
    S4 --> OB
    S1 --> PL
    DISC --> PL --> PLAN --> VAL
    VAL -->|"re-ask once on a violation"| PL
    PLAN -->|"guided: steps as guidance · free: goal only"| AG
    US <-->|"goal in the role's voice · clarifying answers"| AG
    AG <-->|"tool_use ↔ tool_result"| SESS
    SESS <-->|"JSON-RPC"| MCP
    AG --> OBR --> EFF2 --> AG
    AG --> TR
    OBR --> TR
    S2 --> DM
    TR --> DM
    TR --> SV
    TR --> OF
    TR --> JV
    REP -.->|"plan-time reports"| JV
    DM --> VER
    SV --> VER
    OF --> VER
    JV --> VER
    VER --> AGG --> MD
```

Reading it: the matcher, the scope check and observer `fail` effects are hard gates; a judge vote can
never overturn them. Disclosure only ever grows for a stated reason (`discover_tools`, a plan step, or an
observer effect), and every change is a transcript event the judge can read. In dry run the LLM boxes are
skipped: the scout and code observers still run, the agent walks the plan mechanically, and the verdict
comes from the deterministic layers alone.

## One run as a sequence

```mermaid
sequenceDiagram
    autonumber
    participant SC as Scout (no LLM)
    participant OBS as Observers<br/>(identities, API)
    participant PL as Planner<br/>(ollama:command-r7b)
    participant U as Simulated user<br/>(API)
    participant A as Subject agent<br/>(API)
    participant M as MCP server<br/>(pantry)
    participant J as Judge<br/>(API)

    SC->>M: read resources · find_product(query="penne") · pipeline_status
    M-->>SC: structured results
    SC->>OBS: observations (watches: scout)
    OBS-->>SC: reports: direct_match=true (evidence), write_enabled=false
    Note over SC,PL: effects → disclosed toolset + enabled goals
    SC->>PL: disclosed digest · on-request list · reports · scenario
    PL-->>PL: plan.json (validated: enum, schema, recovery, checkpoints)

    U->>A: "I'm looking for the cheapest penne…"
    loop until final answer or budget
        A->>M: tools/call (only offered tools)
        M-->>A: tool_result (structured + text)
        A->>OBS: turn / tool_result trigger
        OBS-->>A: reports → effects (reveal tool, add goal, flag, fail)
        opt agent asks the user
            A->>U: clarifying question
            U-->>A: answer, in role
        end
    end
    A-->>U: final answer + ```json final_result```
    A->>OBS: end trigger
    OBS-->>J: informant reports (all triggers)
    Note over J: matcher · scope · observer-fail gates first
    J->>J: N independent votes over transcript + reports
    J-->>J: verdict.json → report.md
```
