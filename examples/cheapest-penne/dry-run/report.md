# mcp-sim report: cheapest-penne

Generated 2026-10-02T05:56:26.930+00:00. **0/4 runs passed (0.0%), mean score 0.00, est. cost $0.0000**.
Run directory: `runs/cheapest-penne/20261002T055552Z`.
Judge model(s): `dry-run`.

## Pass rate by path and mode

| path | mode | runs | passed | pass rate | mean score | outcomes | cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| happy-dry-run | free | 2 | 0 | 0.0% | 0.00 | budget_exceeded 2 | 0.0000 |
| happy-dry-run | guided | 2 | 0 | 0.0% | 0.00 | budget_exceeded 2 | 0.0000 |
| **all** | | 4 | 0 | 0.0% | 0.00 | budget_exceeded 4 | 0.0000 |

## Worst failures

1. `happy-dry-run-free-0`: score 0.00, outcome `budget_exceeded`
   - deterministic: query $eq
   - deterministic: match $eq
   - deterministic: product_id $type
   - deterministic: product_name $regex
   - deterministic: store $type
   - deterministic: price $gt
   - deterministic: origin_status $in
   - run: budget_exceeded: max_tool_calls=10 exceeded: agent requested another call to review_origin_submission
   - outcome budget_exceeded: max_tool_calls=10 exceeded: agent requested another call to review_origin_submission
   - transcript: `runs/cheapest-penne/20261002T055552Z/transcripts/happy-dry-run-free-0.jsonl`
   - verdict: `runs/cheapest-penne/20261002T055552Z/verdicts/happy-dry-run-free-0.json`
2. `happy-dry-run-free-1`: score 0.00, outcome `budget_exceeded`
   - deterministic: query $eq
   - deterministic: match $eq
   - deterministic: product_id $type
   - deterministic: product_name $regex
   - deterministic: store $type
   - deterministic: price $gt
   - deterministic: origin_status $in
   - run: budget_exceeded: max_tool_calls=10 exceeded: agent requested another call to review_origin_submission
   - outcome budget_exceeded: max_tool_calls=10 exceeded: agent requested another call to review_origin_submission
   - transcript: `runs/cheapest-penne/20261002T055552Z/transcripts/happy-dry-run-free-1.jsonl`
   - verdict: `runs/cheapest-penne/20261002T055552Z/verdicts/happy-dry-run-free-1.json`
3. `happy-dry-run-guided-0`: score 0.00, outcome `budget_exceeded`
   - deterministic: query $eq
   - deterministic: match $eq
   - deterministic: product_id $type
   - deterministic: product_name $regex
   - deterministic: store $type
   - deterministic: price $gt
   - deterministic: origin_status $in
   - run: budget_exceeded: max_tool_calls=10 exceeded: agent requested another call to review_origin_submission
   - outcome budget_exceeded: max_tool_calls=10 exceeded: agent requested another call to review_origin_submission
   - transcript: `runs/cheapest-penne/20261002T055552Z/transcripts/happy-dry-run-guided-0.jsonl`
   - verdict: `runs/cheapest-penne/20261002T055552Z/verdicts/happy-dry-run-guided-0.json`
4. `happy-dry-run-guided-1`: score 0.00, outcome `budget_exceeded`
   - deterministic: query $eq
   - deterministic: match $eq
   - deterministic: product_id $type
   - deterministic: product_name $regex
   - deterministic: store $type
   - deterministic: price $gt
   - deterministic: origin_status $in
   - run: budget_exceeded: max_tool_calls=10 exceeded: agent requested another call to review_origin_submission
   - outcome budget_exceeded: max_tool_calls=10 exceeded: agent requested another call to review_origin_submission
   - transcript: `runs/cheapest-penne/20261002T055552Z/transcripts/happy-dry-run-guided-1.jsonl`
   - verdict: `runs/cheapest-penne/20261002T055552Z/verdicts/happy-dry-run-guided-1.json`

## Cost and usage

Cost is an estimate from the rate table; covers the agent and simulated-user calls recorded in transcripts, not the planner or judge.
