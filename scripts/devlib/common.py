from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "dagzoo"
DOCS_DEP_MAP_PATH = REPO_ROOT / "docs" / "development" / "module-dependency-map.md"


@dataclass(frozen=True)
class CommandSpec:
    label: str
    argv: tuple[str, ...]


class DevToolError(RuntimeError):
    """Raised for developer tooling failures with user-facing messages."""


def repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def python_tool(tool_name: str) -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / tool_name
    if candidate.exists():
        return str(candidate)
    return tool_name


def venv_python() -> Path:
    return REPO_ROOT / ".venv" / "bin" / "python"


def run_command(command: CommandSpec) -> None:
    result = subprocess.run(command.argv, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise DevToolError(f"{command.label} failed with exit code {result.returncode}.")


def format_command(argv: tuple[str, ...]) -> str:
    return " ".join(argv)


def tool_exists(tool_name: str) -> bool:
    return shutil.which(tool_name) is not None


def normalize_files(files: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for file_str in files:
        if not file_str:
            continue
        path = Path(file_str)
        if path.is_absolute():
            normalized.append(repo_relative(path))
        else:
            normalized.append(path.as_posix())
    return tuple(normalized)


def run_git_capture(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed."
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def run_git_lines(*args: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in run_git_capture(*args).splitlines() if line.strip())


def git_common_dir() -> Path:
    return Path(run_git_capture("rev-parse", "--path-format=absolute", "--git-common-dir").strip())


def git_hook_path(hook_name: str) -> Path:
    return git_common_dir() / "hooks" / hook_name
