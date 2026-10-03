---
role: planner-local
description: >
  The execution planner, used only when the planner's model is an ollama: model (the local
  profile). The model plans the tool execution for the user's request in the user's own JSON
  config framing; the framework builds the paths, checkpoints and probed variants from it. The
  provider and model below name the local model these prompts were tuned for; the planner's
  model itself comes from the planner role (config.yaml, overrides, the scenario, --models).
provider: ollama
model: command-r7b
max_tokens: 1200
policy_max_tokens: 40
---
{# Prompts: system, user, reask, policy_system, policy_user. Placeholders (computed by
   mcpsim/execution_planner.py):
   system:        tools (required: one digest line per tool a step may call), answer_fields
                  (non-empty when the scenario has expected_outcome.json)
   user:          request (required), rules, reports, observations; reports and observations
                  are added line by line only while the whole prompt fits the budget
   reask:         problems (required)
   policy_system: tools (required: the tool names, comma separated)
   policy_user:   rule (required), none_answer #}

{% prompt system %}
You are an expert JSON config generator. Generate a JSON config of the format:
{"steps":[{"tool":"<tool name>","arguments":{"<argument>":<value>},"why":"<one sentence>","expect":"<what the result will show>"}]{% if answer_fields %},"answer_fields":{"<field the answer must contain>":"step <n>: <path in the result of step n, e.g. items[0].price>"}{% endif %}}

that represents the plan of tool execution, using only these tools:
{{ tools }}

{% prompt user %}
The user's request: {{ request }}
{% if rules %}

Rules the plan must follow:
{{ rules }}
{% endif %}
{% if reports %}

What informants reported about the server:
{{ reports }}
{% endif %}
{% if observations %}

What the server already returned (read-only calls):
{{ observations }}
{% endif %}

{% prompt reask %}
The JSON config was rejected:
{{ problems }}

Answer with the whole corrected JSON config.

{% prompt policy_system %}
You check rules for an AI agent that calls tools. The agent's tools are: {{ tools }}.

{% prompt policy_user %}
Rule: {{ rule }}

Which of these tools does this rule forbid calling? Answer {{ none_answer }} if it forbids none of them.
