#!/usr/bin/env python3
"""Validate that a pull request version bump is unchanged or a single patch/minor step."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from devlib.semver import (  # noqa: E402
    SemVer,
    allowed_pr_successors,
    read_pyproject_version,
    read_pyproject_version_text,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@dataclass(frozen=True)
class PRVersionDecision:
    ok: bool
    head_version: str
    base_version: str
    reason: str


def read_base_pyproject_version(*, base_ref: str, pyproject_path: str) -> str:
    repo_path = Path(pyproject_path).as_posix()
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{repo_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise ValueError(
            f"Could not read {repo_path} from {base_ref}. {stderr or 'Ensure the base ref exists.'}"
        )
    return read_pyproject_version_text(result.stdout, source=f"{base_ref}:{repo_path}")


def resolve_pr_version_decision(*, head_version: str, base_version: str) -> PRVersionDecision:
    head = SemVer.parse(head_version)
    base = SemVer.parse(base_version)
    if head == base:
        return PRVersionDecision(
            ok=True,
            head_version=str(head),
            base_version=str(base),
            reason="unchanged",
        )

    allowed = allowed_pr_successors(base)
    if head in allowed:
        reason = (
            "next_patch" if head.major == base.major and head.minor == base.minor else "next_minor"
        )
        return PRVersionDecision(
            ok=True,
            head_version=str(head),
            base_version=str(base),
            reason=reason,
        )

    allowed_list = ", ".join(str(version) for version in sorted(allowed))
    raise ValueError(
        f"Pull request version {head} must be unchanged or one allowed step after {base}. "
        f"Allowed bumped versions: {allowed_list}."
    )


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--pyproject", default="pyproject.toml", show_default=True)
@click.option("--base-ref", default="origin/main", show_default=True)
def cli(*, pyproject: str, base_ref: str) -> int:
    """Validate that a pull request version bump is unchanged or a single patch/minor step."""

    head_version = read_pyproject_version(pyproject)
    base_version = read_base_pyproject_version(base_ref=base_ref, pyproject_path=pyproject)
    decision = resolve_pr_version_decision(
        head_version=head_version,
        base_version=base_version,
    )
    print(
        f"PR version check passed: head={decision.head_version} "
        f"base={decision.base_version} reason={decision.reason}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        result = cli.main(args=argv, prog_name="check_pr_version_bump.py", standalone_mode=False)
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return int(exc.exit_code)
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.Abort:
        return 1
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
