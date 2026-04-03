#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from devlib.bootstrap import bootstrap_environment  # noqa: E402
from devlib.common import DevToolError  # noqa: E402
from devlib.contract import evaluate_release_contract, render_contract_result  # noqa: E402
from devlib.impact import (  # noqa: E402
    build_impact_report,
    detect_changed_files,
)
from devlib.impact import render_json as render_impact_json  # noqa: E402
from devlib.impact import render_text as render_impact_text  # noqa: E402
from devlib.review import (  # noqa: E402
    DEFAULT_BASE_REF,
    build_review_base_result,
    render_review_base_report,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
CHANGE_SOURCE_CHOICES = ("working-tree", "staged", "base")


@click.group(
    context_settings=CONTEXT_SETTINGS,
    help="Repo-local developer tooling for dagzoo.",
    epilog=(
        "Canonical local flow:\n"
        "  ./scripts/dev bootstrap\n"
        "  ./.venv/bin/nox -s quick\n"
        "  ./.venv/bin/nox -s bench_smoke\n"
        "  ./scripts/dev review-base\n"
        "  ./scripts/dev impact"
    ),
)
def cli() -> None:
    """Repo-local developer tooling for dagzoo."""


@cli.command("bootstrap")
def bootstrap_command() -> int:
    """Create or sync the local development environment."""

    print(bootstrap_environment(), end="")
    return 0


@cli.command("impact")
@click.option(
    "--source",
    type=click.Choice(CHANGE_SOURCE_CHOICES),
    default="working-tree",
    show_default=True,
)
@click.option("--base", default=None)
@click.option("--files", multiple=True, default=())
@click.option("--format", "output_format", type=click.Choice(("text", "json")), default="text")
def impact_command(
    *,
    source: str,
    base: str | None,
    files: tuple[str, ...],
    output_format: str,
) -> int:
    """Classify changed files and show dependency-aware downstream impact."""

    changed_files = detect_changed_files(
        source=source,
        base=base,
        files=files or None,
    )
    report = build_impact_report(changed_files)
    rendered = render_impact_json(report) if output_format == "json" else render_impact_text(report)
    print(rendered, end="")
    return 0


@cli.command("contract")
@click.option(
    "--source",
    type=click.Choice(CHANGE_SOURCE_CHOICES),
    default="working-tree",
    show_default=True,
)
@click.option("--base", default=None)
@click.option("--files", multiple=True, default=())
@click.option("--strict", is_flag=True)
def contract_command(
    *,
    source: str,
    base: str | None,
    files: tuple[str, ...],
    strict: bool,
) -> int:
    """Evaluate release-contract expectations for the current change set."""

    changed_files = detect_changed_files(
        source=source,
        base=base,
        files=files or None,
    )
    report = build_impact_report(changed_files)
    result = evaluate_release_contract(report, strict=strict)
    print(render_contract_result(result), end="")
    return 0 if result.ok else 1


@cli.command("review-base")
@click.option("--base-ref", default=DEFAULT_BASE_REF, show_default=True)
def review_base_command(*, base_ref: str) -> int:
    """Summarize the current review scope against the selected base ref."""

    result = build_review_base_result(base_ref)
    print(render_review_base_report(result), end="")
    return 0 if result.contract.ok else 1


def build_cli() -> click.Group:
    """Return the Click command group for the repo-local developer CLI."""

    return cli


def main(argv: list[str] | None = None) -> int:
    try:
        result = cli.main(args=argv, prog_name="./scripts/dev", standalone_mode=False)
    except DevToolError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
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
