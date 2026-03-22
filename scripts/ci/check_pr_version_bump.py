#!/usr/bin/env python3
"""Validate that a pull request version bump is unchanged or a single patch/minor step."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from devlib.semver import (  # noqa: E402
    SemVer,
    allowed_pr_successors,
    read_pyproject_version,
    read_pyproject_version_text,
)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--base-ref", default="origin/main")
    args = parser.parse_args(argv)

    head_version = read_pyproject_version(args.pyproject)
    base_version = read_base_pyproject_version(
        base_ref=args.base_ref, pyproject_path=args.pyproject
    )
    decision = resolve_pr_version_decision(
        head_version=head_version,
        base_version=base_version,
    )
    print(
        f"PR version check passed: head={decision.head_version} "
        f"base={decision.base_version} reason={decision.reason}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
