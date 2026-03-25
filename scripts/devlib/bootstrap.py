from __future__ import annotations

import subprocess

from .common import REPO_ROOT, DevToolError, format_command, repo_relative, tool_exists, venv_python


def bootstrap_environment() -> str:
    if not tool_exists("uv"):
        raise DevToolError(
            "`uv` is required for `./scripts/dev bootstrap`; install `uv` and retry."
        )

    sync_command = ("uv", "sync", "--group", "dev")
    sync_result = subprocess.run(sync_command, cwd=REPO_ROOT, check=False)
    if sync_result.returncode != 0:
        raise DevToolError(
            f"`{format_command(sync_command)}` failed with exit code {sync_result.returncode}."
        )

    python_path = venv_python()
    if not python_path.exists():
        raise DevToolError(
            f"bootstrap did not create `{repo_relative(python_path)}`; inspect `uv sync` output."
        )

    install_command = (str(python_path), "-m", "pre_commit", "install")
    install_result = subprocess.run(install_command, cwd=REPO_ROOT, check=False)
    if install_result.returncode != 0:
        raise DevToolError(
            f"`{format_command(install_command)}` failed with exit code {install_result.returncode}."
        )

    return (
        "bootstrap complete\n"
        f"ran: {format_command(sync_command)}\n"
        f"ran: {repo_relative(python_path)} -m pre_commit install\n"
    )
