"""An installed mcpsim finds its default skill without the repository: the wheel carries
skills/simulate as package data (mcpsim/_skills/simulate) and mcpsim.skill.skill_dir picks it.
The runner UI ships too (mcpsim.ui with its static page) and serves that packaged skill."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SKILL_FILES = [
    "mcpsim/_skills/simulate/SKILL.md",
    "mcpsim/_skills/simulate/config.yaml",
    "mcpsim/_skills/simulate/scripts/run.sh",
    *(f"mcpsim/_skills/simulate/roles/{r}.md" for r in (
        "planner", "planner-local", "agent", "user", "observer", "judge"
    )),
]
UI_FILES = [
    "mcpsim/ui/__init__.py",
    "mcpsim/ui/app.py",
    "mcpsim/ui/static/index.html",
    "mcpsim/ui/static/app.js",
    "mcpsim/ui/static/app.css",
]

PROBE = """
import json, sys
site = sys.argv[1]
# Prefer the installed copy over the development install (an editable finder, a .pth path).
sys.meta_path = [f for f in sys.meta_path if "editable" not in repr(f).lower()]
sys.path.insert(0, site)
import mcpsim, mcpsim.skill
from mcpsim.agent import build_user_system_prompt
from mcpsim.scenario import parse_scenario
skill = mcpsim.skill.load_skill()
scenario = parse_scenario({
    "name": "s", "role": "A shopper.", "goal": "Find penne.",
    "expected_outcome": {"text": "Penne."}, "server": {"stdio": {"command": "true"}},
})
from starlette.testclient import TestClient
from mcpsim.ui.app import create_app, resolve_settings
settings = resolve_settings()
with TestClient(create_app(settings), base_url="http://127.0.0.1:8765") as client:
    page, script = client.get("/"), client.get("/app.js")
    config = client.get("/api/config").json()
print(json.dumps({
    "package": mcpsim.__file__,
    "skill": str(skill.path),
    "roles": sorted(skill.roles),
    "prompt": build_user_system_prompt(scenario),
    "ui": {
        "page": page.status_code,
        "script": script.status_code,
        "skill": config["skill"]["path"],
        "roles": [r["role"] for r in config["roles"]],
    },
}))
"""


def test_an_installed_wheel_finds_the_packaged_skill(tmp_path: Path) -> None:
    pytest.importorskip("setuptools", reason="the dev extra installs setuptools for this build")
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(REPO / name, source / name)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO / "mcpsim", source / "mcpsim", ignore=ignore)
    shutil.copytree(REPO / "skills", source / "skills", ignore=ignore)
    dist = tmp_path / "dist"
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check", "-q"]
    subprocess.run(
        [*pip, "wheel", str(source), "--no-deps", "--no-build-isolation", "-w", str(dist)],
        check=True,
    )
    [wheel] = dist.glob("mcpsim-*.whl")
    names = set(zipfile.ZipFile(wheel).namelist())
    assert set(SKILL_FILES) <= names
    assert set(UI_FILES) <= names
    assert not any(n.startswith("skills/") for n in names), "only inside the package"

    site = tmp_path / "site"
    subprocess.run([*pip, "install", "--no-deps", "--target", str(site), str(wheel)], check=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("MCPSIM_SKILL", "PYTHONPATH")}
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(site)],
        cwd=elsewhere,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    found = json.loads(done.stdout)
    assert found["package"] == str(site / "mcpsim" / "__init__.py")
    assert found["skill"] == str(site / "mcpsim" / "_skills" / "simulate")
    assert found["roles"] == sorted(
        ["planner", "planner-local", "agent", "user", "observer", "judge"]
    )
    assert found["prompt"].startswith("You are playing a person in a simulation.")
    assert "## Your instructions\nYou are this person: A shopper." in found["prompt"]
    # The installed runner UI serves its packaged page over the packaged skill.
    ui = found["ui"]
    assert (ui["page"], ui["script"]) == (200, 200)
    assert ui["skill"] == found["skill"]
    assert ui["roles"] == ["planner", "planner-local", "agent", "user", "observer", "judge"]
