from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from .common import REPO_ROOT, git_hook_path, repo_relative, venv_python


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_doctor(mode: str) -> tuple[CheckResult, ...]:
    checks: list[CheckResult] = []
    if mode in {"code", "all"}:
        checks.extend(_code_checks())
    if mode in {"docs", "all"}:
        checks.extend(_docs_checks())
    return tuple(checks)


def render_doctor_results(results: tuple[CheckResult, ...]) -> str:
    lines = []
    for result in results:
        prefix = "ok" if result.ok else "missing"
        lines.append(f"{prefix}: {result.name} - {result.detail}")
    return "\n".join(lines) + "\n"


def doctor_passed(results: tuple[CheckResult, ...]) -> bool:
    return all(result.ok for result in results)


def _code_checks() -> list[CheckResult]:
    python_path = venv_python()
    checks = [
        CheckResult(
            name=".venv",
            ok=python_path.exists(),
            detail=f"expected interpreter at {repo_relative(python_path)}",
        ),
        CheckResult(
            name="uv",
            ok=shutil.which("uv") is not None,
            detail="install uv or ensure it is on PATH",
        ),
    ]
    if python_path.exists():
        result = subprocess.run(
            (
                str(python_path),
                "-c",
                "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)",
            ),
            cwd=REPO_ROOT,
            check=False,
        )
        checks.append(
            CheckResult(
                name="python>=3.13",
                ok=result.returncode == 0,
                detail="repo tooling expects Python 3.13+",
            )
        )
        import_result = subprocess.run(
            (str(python_path), "-c", "import dagzoo"),
            cwd=REPO_ROOT,
            check=False,
        )
        checks.append(
            CheckResult(
                name="import dagzoo",
                ok=import_result.returncode == 0,
                detail="run `uv sync --group dev` if the package is not installed into .venv",
            )
        )
        pre_commit_result = subprocess.run(
            (str(python_path), "-m", "pre_commit", "--version"),
            cwd=REPO_ROOT,
            check=False,
        )
        checks.append(
            CheckResult(
                name="pre-commit package",
                ok=pre_commit_result.returncode == 0,
                detail="run `uv sync --group dev` if pre-commit is missing from .venv",
            )
        )
    try:
        hook_path = git_hook_path("pre-commit")
        checks.append(
            CheckResult(
                name="pre-commit hook",
                ok=hook_path.exists(),
                detail=f"expected installed hook at {repo_relative(hook_path)}",
            )
        )
    except RuntimeError:
        checks.append(
            CheckResult(
                name="pre-commit hook",
                ok=False,
                detail="unable to resolve git hooks path",
            )
        )
    return checks


def _docs_checks() -> list[CheckResult]:
    return [
        CheckResult(
            name="node", ok=shutil.which("node") is not None, detail="required for docs build"
        ),
        CheckResult(
            name="npm", ok=shutil.which("npm") is not None, detail="required for docs dependencies"
        ),
        CheckResult(
            name="go", ok=shutil.which("go") is not None, detail="required for Hugo modules"
        ),
        CheckResult(
            name="hugo", ok=shutil.which("hugo") is not None, detail="required for docs site build"
        ),
    ]
