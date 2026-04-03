#!/usr/bin/env python3
"""Run the CI-only affected verification flow for pull requests."""

from __future__ import annotations

import sys
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from devlib.common import (  # noqa: E402
    CommandSpec,
    DevToolError,
    format_command,
    python_tool,
    run_command,
)
from devlib.impact import ImpactReport, build_impact_report, detect_changed_files  # noqa: E402
from devlib.test_selection import PytestSelection, is_docs_only_change_set  # noqa: E402

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _code_quick_commands(report: ImpactReport) -> tuple[CommandSpec, ...]:
    commands = [
        CommandSpec(
            label="ruff check",
            argv=(python_tool("ruff"), "check", "src", "tests", "scripts"),
        ),
        CommandSpec(
            label="ruff format",
            argv=(python_tool("ruff"), "format", "--check", "src", "tests", "scripts"),
        ),
        CommandSpec(
            label="mypy",
            argv=(python_tool("mypy"), "src"),
        ),
        CommandSpec(
            label="deptry",
            argv=(python_tool("deptry"), "."),
        ),
    ]
    if "architecture" in report.tags:
        commands.append(
            CommandSpec(
                label="import-linter",
                argv=(python_tool("lint-imports"),),
            )
        )
    return tuple(commands)


def _docs_commands() -> tuple[CommandSpec, ...]:
    python = python_tool("python")
    return (
        CommandSpec(
            label="docs repo paths",
            argv=(python, "scripts/docs/check_repo_paths.py"),
        ),
        CommandSpec(
            label="docs sync",
            argv=(python, "scripts/docs/sync_hugo_content.py"),
        ),
        CommandSpec(
            label="docs sync check",
            argv=(python, "scripts/docs/sync_hugo_content.py", "--check"),
        ),
        CommandSpec(
            label="docs links",
            argv=(python, "scripts/docs/check_links.py"),
        ),
        CommandSpec(
            label="docs build",
            argv=("hugo", "--source", "site", "--minify", "--gc", "--destination", "public"),
        ),
        CommandSpec(
            label="docs built links",
            argv=(python, "scripts/docs/check_built_output_links.py", "site/public"),
        ),
    )


def _adoption_surface_command() -> CommandSpec:
    return CommandSpec(
        label="docs adoption surface",
        argv=(python_tool("python"), "scripts/ci/check_adoption_surface.py"),
    )


def _build_pytest_command(
    *, targets: tuple[str, ...], incremental: bool, parallel: bool
) -> CommandSpec:
    argv: list[str] = [python_tool("pytest"), "-q"]
    if incremental:
        argv.append("--testmon")
    if parallel:
        argv.extend(("-n", "auto"))
    argv.extend(targets)
    return CommandSpec(label="pytest", argv=tuple(argv))


def _affected_pytest_commands(
    *, selection: PytestSelection, incremental: bool, parallel: bool
) -> tuple[CommandSpec, ...]:
    if selection.mode == "skip":
        return ()
    if selection.mode == "targeted":
        return (
            _build_pytest_command(
                targets=selection.targets,
                incremental=incremental,
                parallel=parallel,
            ),
        )
    return (
        _build_pytest_command(
            targets=(),
            # Full-suite runs are stable without testmon and avoid xdist collection drift.
            incremental=False,
            parallel=parallel,
        ),
    )


def build_affected_verification_commands(
    report: ImpactReport, *, incremental: bool, parallel: bool
) -> tuple[CommandSpec, ...]:
    if is_docs_only_change_set(report.changed_files):
        return (*_docs_commands(), _adoption_surface_command())
    return (
        *_code_quick_commands(report),
        _adoption_surface_command(),
        *_affected_pytest_commands(
            selection=report.pytest_selection,
            incremental=incremental,
            parallel=parallel,
        ),
    )


def run_affected_verification(
    *, base: str = "origin/main", incremental: bool = False, parallel: bool = False
) -> str:
    changed_files = detect_changed_files(source="base", base=base)
    report = build_impact_report(changed_files)
    commands = build_affected_verification_commands(
        report,
        incremental=incremental,
        parallel=parallel,
    )

    lines = [
        "affected verification",
        f"base ref: {base}",
        "recommended modes: "
        + (", ".join(report.recommended_modes) if report.recommended_modes else "none"),
        f"pytest selection: {report.pytest_selection.mode}",
        f"pytest selection reason: {report.pytest_selection.reason}",
    ]
    if report.pytest_selection.targets:
        lines.append("pytest selection targets: " + ", ".join(report.pytest_selection.targets))

    for command in commands:
        run_command(command)
        lines.append(f"ran: {format_command(command.argv)}")

    return "\n".join(lines) + "\n"


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--base", default="origin/main", show_default=True)
@click.option("--incremental", is_flag=True)
@click.option("--parallel", is_flag=True)
def cli(*, base: str, incremental: bool, parallel: bool) -> int:
    """Run the CI-only affected verification flow for pull requests."""

    try:
        print(
            run_affected_verification(
                base=base,
                incremental=incremental,
                parallel=parallel,
            ),
            end="",
        )
        return 0
    except DevToolError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    try:
        result = cli.main(
            args=argv,
            prog_name="run_affected_verification.py",
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
