from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from mcpsim.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, build_parser, main, render_catalog
from mcpsim.mcpclient import Catalog, PromptInfo, ResourceInfo, ResourceTemplateInfo, ToolInfo
from tests.fake_server import SERVER_NAME, TOOL_NAMES


def test_catalog_over_stdio_subprocess(
    scenario_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["catalog", str(scenario_path)])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert f"server: {SERVER_NAME}" in out
    for name in TOOL_NAMES:
        assert f"- {name}(" in out
    assert "lookup(slug: string)" in out
    assert "list_items(cursor?: integer, limit?: integer)" in out
    assert "fake://about" in out and "fake://item/{slug}" in out and "shopping_prompt(goal)" in out


def test_catalog_json(scenario_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["catalog", str(scenario_path), "--json"])
    assert code == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert sorted(t["name"] for t in data["tools"]) == sorted(TOOL_NAMES)
    assert data["tools"][0]["input_schema"]["type"] == "object"


def test_catalog_invalid_scenario(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    assert main(["catalog", str(path)]) == EXIT_FAILURE
    err = capsys.readouterr().err
    assert "invalid scenario" in err and "goal" in err


def test_catalog_missing_bearer_env(
    tmp_path: Path,
    scenario_data: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MCPSIM_TEST_TOKEN", raising=False)
    scenario_data["server"] = {
        "http": {"url": "http://127.0.0.1:9/mcp", "bearer_env": "MCPSIM_TEST_TOKEN"}
    }
    path = tmp_path / "http.yaml"
    path.write_text(yaml.safe_dump(scenario_data), encoding="utf-8")
    assert main(["catalog", str(path)]) == EXIT_FAILURE
    assert "MCPSIM_TEST_TOKEN is not set" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["plan", "s.yaml"],
        ["run", "s.yaml", "--only-path", "happy", "--mode", "free", "--repeat", "1"],
        ["judge", "runs/x"],
        ["report", "runs/x"],
        ["suite", "scenarios", "--threshold", "0.8"],
    ],
)
def test_unwired_subcommands_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == EXIT_USAGE
    assert f"mcpsim {argv[0]}: not yet wired" in capsys.readouterr().err


def test_no_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2


def test_parser_has_every_subcommand() -> None:
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    assert set(sub.choices) == {"catalog", "plan", "run", "judge", "report", "suite"}  # type: ignore[union-attr]


def test_render_catalog_without_server_name() -> None:
    catalog = Catalog(
        tools=[
            ToolInfo(
                name="t",
                description="First line.\nSecond.",
                input_schema={"type": "object", "properties": {"a": {"type": "string"}}},
            )
        ],
        resources=[ResourceInfo(name="r", uri="x://r")],
        resource_templates=[ResourceTemplateInfo(name="tm", uri_template="x://{id}")],
        prompts=[PromptInfo(name="p", arguments=[{"name": "q"}])],
    )
    text = render_catalog(catalog)
    assert text.startswith("server  (catalog digest ")
    assert "- t(a?: string)" in text and "First line." in text and "Second." not in text
    assert "x://r  [r]" in text and "x://{id}  [tm]" in text and "- p(q)" in text
