#!/bin/bash
# The simulate skill's flow, end to end on this machine (see ../SKILL.md).
#
#   run.sh preflight          the Anthropic key, the pantry API, ContextForge, the fetch server, and a
#                             refresh of the pantry tools ContextForge caches
#   run.sh scenarios          generate the gateway scenarios (pantry-gateway scripts/make_scenarios.py,
#                             the pantry copies and the --recipe ones)
#   run.sh suite [ARGS...]    mcpsim suite --skill <this skill> ARGS (e.g. --name 'cheapest-*')
#   run.sh ui [ARGS...]       mcpsim ui --skill <this skill> ARGS (the test runner, 127.0.0.1:8765)
#   run.sh config [ARGS...]   mcpsim config --skill <this skill> ARGS (models, prompts, run settings)
#   run.sh report [DIR]       print the newest suite report (or DIR's report.md / suite.md)
#   run.sh all [ARGS...]      preflight, scenarios, suite ARGS, report
#
# Inputs (env, all optional):
#   MCPSIM_HOME          the mcp-sim checkout (default: two levels above this skill)
#   WORKSPACE            where the other repositories live (default: the parent of MCPSIM_HOME)
#   PANTRY_GATEWAY_HOME  pantry-gateway checkout (default $WORKSPACE/pantry-gateway)
#   PANTRY_API_HOME      pantry-api checkout (default $WORKSPACE/pantry-platform/pantry-api)
#   PANTRY_API_URL       the pantry API (default http://127.0.0.1:8000)
#   CF_URL               ContextForge (default http://127.0.0.1:4444)
#   FETCH_URL            the fetch server (default http://127.0.0.1:9100)
#   CF_JWT_FILE          a file holding the ContextForge admin JWT (needed for the refresh, the
#                        scenario generation and the gateway scenarios, which read it as
#                        CONTEXTFORGE_JWT); without it the gateway steps are skipped
#   RECIPE_SHOPPER_SKILL the recipe-shopper SKILL.md (default: pantry-api's skills/recipe-shopper)
#   PANTRY_GATEWAY_SCENARIOS  where the generated scenarios go (default: pantry-gateway's
#                        scenarios/generated, which config.yaml reads)
#   SKIP_REFRESH=1       preflight without refreshing ContextForge's pantry tools
#   SKIP_KEY_CHECK=1     preflight without the one-token Claude call that proves the key works
# Secrets are read from files into the environment and never printed.
set -euo pipefail

SKILL_DIR=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${MCPSIM_HOME:-}" ]; then
  candidate=$(cd "$SKILL_DIR/../.." && pwd)
  if grep -qs '^name = "mcpsim"' "$candidate/pyproject.toml"; then
    MCPSIM_HOME=$candidate
  else
    echo "set MCPSIM_HOME to the mcp-sim checkout (this skill is not inside one)" >&2
    exit 2
  fi
fi
WORKSPACE=${WORKSPACE:-$(cd "$MCPSIM_HOME/.." && pwd)}
PANTRY_GATEWAY_HOME=${PANTRY_GATEWAY_HOME:-$WORKSPACE/pantry-gateway}
PANTRY_API_HOME=${PANTRY_API_HOME:-$WORKSPACE/pantry-platform/pantry-api}
PANTRY_API_URL=${PANTRY_API_URL:-http://127.0.0.1:8000}
CF_URL=${CF_URL:-http://127.0.0.1:4444}
FETCH_URL=${FETCH_URL:-http://127.0.0.1:9100}
MCPSIM=$MCPSIM_HOME/.venv/bin/mcpsim
PYTHON=$MCPSIM_HOME/.venv/bin/python
export PANTRY_GATEWAY_SCENARIOS=${PANTRY_GATEWAY_SCENARIOS:-$PANTRY_GATEWAY_HOME/scenarios/generated}

ok() { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; failed=1; }
skip() { printf '  skip  %s\n' "$1"; }

# The environment every mcpsim run gets: the Anthropic key from mcp-sim/.env, the ContextForge
# JWT as CONTEXTFORGE_JWT (the gateway scenarios' bearer_env), the recipe-shopper SOP.
load_env() {
  if [ -f "$MCPSIM_HOME/.env" ]; then
    set -a; . "$MCPSIM_HOME/.env"; set +a
  fi
  if [ -n "${CF_JWT_FILE:-}" ] && [ -s "$CF_JWT_FILE" ]; then
    CONTEXTFORGE_JWT=$(tr -d '[:space:]' < "$CF_JWT_FILE"); export CONTEXTFORGE_JWT
  fi
  local sop=$PANTRY_API_HOME/skills/recipe-shopper/SKILL.md
  if [ -z "${RECIPE_SHOPPER_SKILL:-}" ] && [ -f "$sop" ]; then
    export RECIPE_SHOPPER_SKILL=$sop
  fi
}

# The id of a ContextForge virtual server, by name (the JWT is read inside Python, never echoed).
server_id() {
  CF_URL=$CF_URL NAME=$1 "$PYTHON" - <<'PY'
import json, os, pathlib, sys, urllib.request
jwt = pathlib.Path(os.environ["CF_JWT_FILE"]).expanduser().read_text().strip()
req = urllib.request.Request(os.environ["CF_URL"].rstrip("/") + "/servers",
                             headers={"Authorization": f"Bearer {jwt}"})
data = json.load(urllib.request.urlopen(req, timeout=10))
items = data if isinstance(data, list) else data.get("servers", data.get("items", []))
found = [s["id"] for s in items if s.get("name") == os.environ["NAME"]]
if not found:
    sys.exit(f"no ContextForge virtual server named {os.environ['NAME']!r}")
print(found[0])
PY
}

preflight() {
  failed=0
  load_env
  echo "preflight"
  if [ -x "$MCPSIM" ]; then ok "mcpsim installed ($MCPSIM)"; else bad "no $MCPSIM (python -m venv .venv && .venv/bin/pip install -e '.[dev]')"; fi
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then ok "ANTHROPIC_API_KEY set (from $MCPSIM_HOME/.env)"; else bad "ANTHROPIC_API_KEY missing: put it in $MCPSIM_HOME/.env"; fi
  if [ -n "${ANTHROPIC_API_KEY:-}" ] && [ "${SKIP_KEY_CHECK:-0}" != 1 ]; then
    # One output token from Haiku: proves the key is accepted and the account has credit.
    if answer=$("$PYTHON" - <<'PY' 2>&1
import anthropic
try:
    anthropic.Anthropic().messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=1, messages=[{"role": "user", "content": "ok"}]
    )
except anthropic.APIStatusError as exc:
    body = exc.body if isinstance(exc.body, dict) else {}
    raise SystemExit(f"HTTP {exc.status_code}: {body.get('error', {}).get('message', exc.message)}")
PY
    ); then ok "the Anthropic API answers (one Haiku token)"; else bad "the Anthropic API refused a one-token call: ${answer:0:160}"; fi
  fi
  if curl -sf --max-time 5 "$PANTRY_API_URL/health" >/dev/null; then ok "pantry API answers ($PANTRY_API_URL/health)"; else bad "pantry API down at $PANTRY_API_URL (the gateway scenarios need it)"; fi
  local gateway_up=1
  if curl -sf --max-time 5 "$CF_URL/health" >/dev/null; then ok "ContextForge answers ($CF_URL/health)"; else bad "ContextForge down at $CF_URL (pantry-gateway scripts/run.sh)"; gateway_up=0; fi
  if curl -sf --max-time 5 "$FETCH_URL/healthz" >/dev/null; then ok "fetch server answers ($FETCH_URL/healthz)"; else bad "fetch server down at $FETCH_URL (pantry-gateway scripts/run_fetch.sh)"; gateway_up=0; fi
  if [ -z "${CF_JWT_FILE:-}" ] || [ ! -s "$CF_JWT_FILE" ]; then
    skip "CF_JWT_FILE not set: no tool refresh, and the gateway scenarios cannot run"
  elif [ "${SKIP_REFRESH:-0}" = 1 ]; then
    skip "pantry tool refresh (SKIP_REFRESH=1)"
  elif [ "$gateway_up" = 1 ]; then
    # Re-reads the pantry server's tools into ContextForge and keeps pantry-recipes in step.
    if REFRESH_PANTRY=true CF_URL=$CF_URL CF_JWT_FILE=$CF_JWT_FILE \
        FETCH_MCP_URL="$FETCH_URL/mcp" "$PANTRY_GATEWAY_HOME/scripts/register_fetch.sh" >/dev/null; then
      ok "ContextForge re-read the pantry tools (register_fetch.sh REFRESH_PANTRY=true)"
    else
      bad "refreshing the pantry tools failed (run pantry-gateway scripts/register_fetch.sh)"
    fi
  fi
  if [ -n "${RECIPE_SHOPPER_SKILL:-}" ] && [ -f "$RECIPE_SHOPPER_SKILL" ]; then ok "recipe-shopper SOP: $RECIPE_SHOPPER_SKILL"; else skip "RECIPE_SHOPPER_SKILL not found: the recipe-link scenarios will not load"; fi
  return "$failed"
}

scenarios() {
  load_env
  if [ -z "${CF_JWT_FILE:-}" ] || [ ! -s "$CF_JWT_FILE" ]; then
    echo "scenarios: CF_JWT_FILE is needed to look up the virtual servers' ids" >&2
    return 1
  fi
  local sim recipes
  sim=$(server_id pantry-sim)
  recipes=$(server_id pantry-recipes)
  echo "generating gateway scenarios into $PANTRY_GATEWAY_SCENARIOS"
  "$PYTHON" "$PANTRY_GATEWAY_HOME/scripts/make_scenarios.py" \
    "$MCPSIM_HOME/scenarios/pantry" "$sim" "$PANTRY_GATEWAY_SCENARIOS"
  "$PYTHON" "$PANTRY_GATEWAY_HOME/scripts/make_scenarios.py" \
    --recipe "$recipes" "$PANTRY_GATEWAY_SCENARIOS"
}

report() {
  local target=${1:-}
  cd "$MCPSIM_HOME"
  if [ -z "$target" ]; then
    local runs
    runs=$("$MCPSIM" config --skill "$SKILL_DIR" --json | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["runs_dir"])')
    target=$(ls -1d "$runs"/suite-* 2>/dev/null | sort | tail -1 || true)
    [ -n "$target" ] || { echo "no suite report under $runs yet; run: run.sh suite" >&2; return 1; }
  fi
  for f in suite.md report.md; do
    if [ -f "$target/$f" ]; then echo "== $target/$f"; cat "$target/$f"; return 0; fi
  done
  echo "no suite.md or report.md in $target" >&2
  return 1
}

command=${1:-help}
[ $# -gt 0 ] && shift
case "$command" in
  preflight) preflight ;;
  scenarios) scenarios ;;
  suite) load_env; cd "$MCPSIM_HOME"; exec "$MCPSIM" suite --skill "$SKILL_DIR" "$@" ;;
  ui) load_env; cd "$MCPSIM_HOME"; exec "$MCPSIM" ui --skill "$SKILL_DIR" "$@" ;;
  config) cd "$MCPSIM_HOME"; exec "$MCPSIM" config --skill "$SKILL_DIR" "$@" ;;
  report) report "$@" ;;
  all)
    preflight || { echo "preflight failed; fix the FAIL lines first" >&2; exit 1; }
    scenarios || echo "(gateway scenarios not regenerated; the suite runs what is there)" >&2
    load_env; cd "$MCPSIM_HOME"
    code=0; "$MCPSIM" suite --skill "$SKILL_DIR" "$@" || code=$?
    report || true
    exit "$code"
    ;;
  help|-h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown command '$command' (try: run.sh help)" >&2; exit 2 ;;
esac
