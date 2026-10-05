"""The prompts the simulation sends are byte-identical to the ones captured in
``tests/fixtures/prompts/golden.json.gz`` (see :mod:`tests.prompt_cases`).

The golden file was written by the code *before* the prompt text moved into
``skills/simulate/roles/*.md``, so this test proves the bundled templates reproduce every system
prompt, user message, re-ask and ``max_tokens`` exactly, for every pantry scenario, a scenario
with a standard operating procedure and a minimal v1 scenario.
"""

from __future__ import annotations

import difflib

import pytest

from tests.prompt_cases import load_golden, render_all


@pytest.fixture(autouse=True)
def _default_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("MCPSIM_PLANNER_PROMPT_BUDGET", "MCPSIM_OBSERVER_MAX_CALLS", "MCPSIM_SKILL"):
        monkeypatch.delenv(var, raising=False)


def test_every_prompt_matches_the_capture() -> None:
    golden = load_golden()
    rendered = render_all()
    assert sorted(rendered) == sorted(golden), "the set of rendered cases changed"
    differ = [case for case in golden if rendered[case] != golden[case]]
    if differ:
        first = differ[0]
        diff = "\n".join(
            difflib.unified_diff(
                golden[first].splitlines(),
                rendered[first].splitlines(),
                "captured",
                "rendered",
                lineterm="",
                n=1,
            )
        )
        pytest.fail(f"{len(differ)} case(s) differ, first {first}:\n{diff[:4000]}")


def test_the_capture_covers_every_role_and_call() -> None:
    golden = load_golden()
    kinds = {case.split(".")[1] for case in golden}
    assert kinds == {"planner", "local", "agent", "user", "observer", "judge"}
    # The re-asks, the policy questions, the opening cue and the SOP variant are all in there.
    assert any(".planner.noscout.call2.message2" in c for c in golden)
    assert any(".local.policy.call1.message0" in c for c in golden)
    assert "Start the conversation" in golden["cheapest-penne.user.call1.message0"]
    assert "<<<BEGIN SOP price-check>>>" in golden[
        "cheapest-penne-sop.agent.live-plan.happy-find-penne.guided"
    ]
    sop_judge = golden["cheapest-penne-sop.judge.live.call1.system"]
    assert "sop_followed: the agent was given" in sop_judge
    assert golden["cheapest-penne.planner.scout.call1.max_tokens"] == "8192"
