#!/usr/bin/env python3
"""Dry-run or remove ignored local artifact trees.

Targets are limited to known ignored runtime/docs outputs under the repo root.
Use ``--apply`` to actually delete them.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

TARGET_GROUPS: dict[str, tuple[Path, ...]] = {
    "runtime": (
        REPO_ROOT / "data",
        REPO_ROOT / "benchmarks" / "results",
        REPO_ROOT / "effective_config_artifacts",
    ),
    "docs": (
        REPO_ROOT / "public",
        REPO_ROOT / "site" / "public",
        REPO_ROOT / "site" / ".generated",
    ),
}


def _iter_targets(group: str) -> list[Path]:
    if group == "all":
        paths = TARGET_GROUPS["runtime"] + TARGET_GROUPS["docs"]
    else:
        paths = TARGET_GROUPS[group]
    return list(paths)


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--group",
    "group_name",
    type=click.Choice(("runtime", "docs", "all")),
    default="all",
    show_default=True,
    help="Artifact group to inspect or remove.",
)
@click.option(
    "--apply",
    is_flag=True,
    help="Actually delete the listed paths. Default is dry-run only.",
)
def cli(*, group_name: str, apply: bool) -> int:
    """Dry-run or remove ignored local artifact trees."""

    targets = _iter_targets(str(group_name))

    found = 0
    for path in targets:
        rel = _relpath(path, REPO_ROOT)
        if not path.exists():
            print(f"Skip missing: {rel}")
            continue
        found += 1
        if apply:
            _remove_path(path)
            print(f"Removed: {rel}")
        else:
            print(f"Would remove: {rel}")

    if not apply:
        print("Dry run only. Re-run with --apply to remove the listed paths.")
    elif found == 0:
        print("Nothing to remove.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    try:
        result = cli.main(
            args=list(argv) if argv is not None else None,
            prog_name="cleanup_local_artifacts.py",
            standalone_mode=False,
        )
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
