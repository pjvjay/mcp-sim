"""A small reader for the simulate skill's ``config.yaml`` and role files (contract C).

The runner UI needs four things from the skill: which provider and model serve each role (and
where that came from), the run defaults, where the scenarios are, and where runs are written.
This module reads them without importing the skill loader, so the UI works on a branch where
that loader does not exist yet; switching to the real loader means replacing
:func:`load_skill_config` and :func:`resolve_roles`, whose return shapes are what the API
serves.

Model precedence, lowest to highest: built-in default < role frontmatter < ``config.yaml``
``defaults`` < matching ``config.yaml`` override < scenario file ``models`` < per-run
override (the CLI's ``--models``, the settings form here). Every role has a built-in Anthropic
default (:data:`BUILTIN_MODELS`, the scenario module's ``DEFAULT_MODELS`` when it has one).
When the planner resolves to an ``ollama:`` model, its role file is ``planner-local.md``.
"""

from __future__ import annotations

import fnmatch
import glob
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path as FsPath
from typing import Any

import yaml

from mcpsim import scenario as scenario_module

SKILL_ENV = "MCPSIM_SKILL"
CONFIG_FILE = "config.yaml"
ROLES: tuple[str, ...] = ("planner", "agent", "user", "observer", "judge")
PROVIDERS: tuple[str, ...] = ("anthropic", "ollama")
LOCAL_PLANNER_FILE = "planner-local"
# Every role's built-in default (all Anthropic API models). The scenario module's own table wins
# once it has one, so the UI shows what the runner will actually use.
BUILTIN_MODELS: dict[str, str] = {
    "planner": "claude-opus-5-5",
    "agent": "claude-sonnet-5-5",
    "user": "claude-haiku-4-5-20251001",
    "observer": "claude-sonnet-5-5",
    "judge": "claude-opus-5-5",
    **getattr(scenario_module, "DEFAULT_MODELS", {}),
}
KNOWN_MODELS: tuple[str, ...] = (
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-4-5-20251001",
    "claude-fable-5-1",
)
FRONTMATTER_KEYS: tuple[str, ...] = ("role", "provider", "model", "temperature", "max_tokens")


@dataclass
class RoleFile:
    role: str
    file: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedRole:
    role: str
    provider: str
    model: str
    source: str
    temperature: float | None = None
    max_tokens: int | None = None
    file: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "spec": self.spec}


@dataclass
class SkillConfig:
    skill_dir: str | None = None
    config_file: str | None = None
    roles_dir: str | None = None
    role_files: dict[str, RoleFile] = field(default_factory=dict)
    defaults: dict[str, str] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    scenarios: list[str] = field(default_factory=list)
    runs_dir: str | None = None
    overrides: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "skill_dir": self.skill_dir,
            "config_file": self.config_file,
            "roles_dir": self.roles_dir,
            "role_files": {k: {"file": v.file, **v.meta} for k, v in self.role_files.items()},
            "defaults": dict(self.defaults),
            "run": dict(self.run),
            "scenarios": list(self.scenarios),
            "runs_dir": self.runs_dir,
            "overrides": list(self.overrides),
            "warnings": list(self.warnings),
        }


def default_skill_dir(cli_value: str | None, env: Mapping[str, str] | None = None) -> FsPath | None:
    """``--skill DIR``, else ``$MCPSIM_SKILL``, else the packaged or in-repo ``skills/simulate``."""
    source = os.environ if env is None else env
    if cli_value:
        return FsPath(cli_value).expanduser()
    if source.get(SKILL_ENV):
        return FsPath(source[SKILL_ENV]).expanduser()
    package_dir = FsPath(__file__).resolve().parent.parent
    for candidate in (
        package_dir / "skills" / "simulate",
        package_dir.parent / "skills" / "simulate",
    ):
        if (candidate / CONFIG_FILE).is_file() or (candidate / "SKILL.md").is_file():
            return candidate
    return None


def split_spec(spec: str) -> tuple[str, str]:
    """``ollama:llama3.2:3b`` -> ``("ollama", "llama3.2:3b")``; a bare name is Anthropic's."""
    head, sep, rest = spec.partition(":")
    if sep and head in PROVIDERS and rest:
        return head, rest
    return "anthropic", spec


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """``---\\n<yaml>\\n---\\n<body>`` -> ``(meta, body)``; no frontmatter -> ``({}, text)``."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            meta = yaml.safe_load("".join(lines[1:i])) or {}
            if not isinstance(meta, dict):
                raise ValueError("frontmatter must be a mapping")
            return meta, "".join(lines[i + 1 :])
    raise ValueError("frontmatter has no closing '---'")


def _read_role_files(roles_dir: FsPath, warnings: list[str]) -> dict[str, RoleFile]:
    found: dict[str, RoleFile] = {}
    if not roles_dir.is_dir():
        return found
    for path in sorted(roles_dir.glob("*.md")):
        try:
            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
            warnings.append(f"{path.name}: cannot read frontmatter: {exc}")
            continue
        role = str(meta.get("role") or path.stem)
        found[path.stem] = RoleFile(
            role=role,
            file=str(path),
            meta={k: v for k, v in meta.items() if isinstance(v, str | int | float | bool)},
        )
    return found


def load_skill_config(skill_dir: FsPath | None) -> SkillConfig:
    """Read ``<skill_dir>/config.yaml`` and the role files' frontmatter; never raises."""
    config = SkillConfig()
    if skill_dir is None:
        config.warnings.append("no skill directory: built-in model defaults apply")
        return config
    config.skill_dir = str(skill_dir)
    data: dict[str, Any] = {}
    config_path = skill_dir / CONFIG_FILE
    if config_path.is_file():
        config.config_file = str(config_path)
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            config.warnings.append(f"{config_path}: cannot read: {exc}")
            loaded = {}
        if isinstance(loaded, dict):
            data = loaded
        else:
            config.warnings.append(f"{config_path}: top level must be a mapping")
    elif skill_dir.exists():
        config.warnings.append(f"{config_path} not found: built-in defaults apply")
    else:
        config.warnings.append(f"skill directory {skill_dir} does not exist")

    roles_dir = skill_dir / str(data.get("roles_dir") or "roles")
    config.roles_dir = str(roles_dir)
    config.role_files = _read_role_files(roles_dir, config.warnings)

    defaults = data.get("defaults")
    if isinstance(defaults, dict):
        config.defaults = {str(k): str(v) for k, v in defaults.items() if isinstance(v, str) and v}
    run = data.get("run")
    if isinstance(run, dict):
        config.run = {
            str(k): v for k, v in run.items() if isinstance(v, str | int | float | bool | list)
        }
    scenarios = data.get("scenarios")
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    if isinstance(scenarios, list):
        config.scenarios = [str(s) for s in scenarios if isinstance(s, str) and s.strip()]
    if isinstance(data.get("runs_dir"), str) and data["runs_dir"].strip():
        config.runs_dir = data["runs_dir"]
    overrides = data.get("overrides")
    if isinstance(overrides, list):
        config.overrides = [o for o in overrides if isinstance(o, dict)]
    return config


def matching_overrides(config: SkillConfig, name: str, category: str) -> list[dict[str, Any]]:
    """Overrides whose ``match`` globs (``name`` and/or ``category``) all match, in file order."""
    found: list[dict[str, Any]] = []
    for override in config.overrides:
        match = override.get("match")
        if not isinstance(match, dict) or not match:
            continue
        ok = True
        for key, value in (("name", name), ("category", category)):
            pattern = match.get(key)
            if pattern is not None and not fnmatch.fnmatchcase(value, str(pattern)):
                ok = False
        if ok and any(k in match for k in ("name", "category")):
            found.append(override)
    return found


def _role_from_frontmatter(role: str, rf: RoleFile) -> ResolvedRole | None:
    model = rf.meta.get("model")
    if not isinstance(model, str) or not model:
        return None
    provider = rf.meta.get("provider")
    if isinstance(provider, str) and provider:
        prov, mod = provider, model
    else:
        prov, mod = split_spec(model)
    temperature = rf.meta.get("temperature")
    max_tokens = rf.meta.get("max_tokens")
    return ResolvedRole(
        role=role,
        provider=prov,
        model=mod,
        source="frontmatter",
        temperature=float(temperature) if isinstance(temperature, int | float) else None,
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        file=rf.file,
        extra={k: v for k, v in rf.meta.items() if k not in FRONTMATTER_KEYS},
    )


def resolve_roles(
    config: SkillConfig,
    *,
    scenario_name: str | None = None,
    category: str | None = None,
    scenario_models: Mapping[str, str] | None = None,
    run_models: Mapping[str, str] | None = None,
) -> dict[str, ResolvedRole]:
    """Provider and model per role with the layer that set it (see the module docstring)."""
    layers: list[tuple[str, Mapping[str, str]]] = [("config defaults", config.defaults)]
    if scenario_name is not None:
        for override in matching_overrides(config, scenario_name, category or ""):
            models = override.get("models")
            if isinstance(models, dict):
                layers.append(("config override", {str(k): str(v) for k, v in models.items() if v}))
    if scenario_models:
        layers.append(("scenario", scenario_models))
    if run_models:
        layers.append(("run override", run_models))

    resolved: dict[str, ResolvedRole] = {}
    for role in ROLES:
        prov, mod = split_spec(BUILTIN_MODELS[role])
        current = ResolvedRole(role=role, provider=prov, model=mod, source="built-in")
        rf = config.role_files.get(role)
        if rf is not None:
            current.file = rf.file
            from_file = _role_from_frontmatter(role, rf)
            if from_file is not None:
                current = from_file
        for source, mapping in layers:
            spec = mapping.get(role)
            if spec:
                prov, mod = split_spec(spec)
                current = ResolvedRole(
                    role=role,
                    provider=prov,
                    model=mod,
                    source=source,
                    temperature=current.temperature,
                    max_tokens=current.max_tokens,
                    file=current.file,
                    extra=current.extra,
                )
        local = config.role_files.get(LOCAL_PLANNER_FILE)
        if role == "planner" and current.provider == "ollama" and local is not None:
            # The execution planner (planner-local.md) serves an Ollama planner.
            meta = _role_from_frontmatter(role, local)
            current.file = local.file
            if meta is not None:
                current.temperature = meta.temperature
                current.max_tokens = meta.max_tokens
                current.extra = meta.extra
        resolved[role] = current
    return resolved


def expand_sources(entries: list[str], bases: list[FsPath]) -> tuple[list[FsPath], list[str]]:
    """Scenario ``entries`` (directories, files or globs; ``$ENV`` expanded) -> files.

    A relative entry resolves against the first base where it exists (the working directory,
    then the skill directory). Directories are searched recursively for ``*.yaml``, ``*.yml``
    and ``*.json``. Returns ``(files, warnings)``; files are absolute, deduplicated and sorted.
    """
    files: set[FsPath] = set()
    warnings: list[str] = []
    for entry in entries:
        expanded = os.path.expanduser(os.path.expandvars(entry))
        if "$" in expanded:
            warnings.append(f"scenarios entry {entry!r}: environment variable not set")
            continue
        candidates: list[FsPath] = []
        if any(ch in expanded for ch in "*?["):
            for base in bases if not os.path.isabs(expanded) else [FsPath("/")]:
                pattern = expanded if os.path.isabs(expanded) else str(base / expanded)
                candidates = [FsPath(p) for p in glob.glob(pattern, recursive=True)]
                if candidates:
                    break
        else:
            for base in bases if not os.path.isabs(expanded) else [FsPath("/")]:
                path = FsPath(expanded) if os.path.isabs(expanded) else base / expanded
                if path.exists():
                    candidates = [path]
                    break
        if not candidates:
            warnings.append(f"scenarios entry {entry!r} matched nothing")
            continue
        for path in candidates:
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file() and child.suffix.lower() in (".yaml", ".yml", ".json"):
                        files.add(child.resolve())
            elif path.is_file() and path.suffix.lower() in (".yaml", ".yml", ".json"):
                files.add(path.resolve())
    return sorted(files), warnings
