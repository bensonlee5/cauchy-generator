#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./scripts/dev",
        description="Repo-local developer tooling for dagzoo.",
        epilog=(
            "Canonical local flow:\n"
            "  ./scripts/dev bootstrap\n"
            "  ./.venv/bin/nox -s quick\n"
            "  ./.venv/bin/nox -s bench_smoke\n"
            "  ./.venv/bin/nox -s bench_public_smoke\n"
            "  ./scripts/dev review-base\n"
            "  ./scripts/dev impact"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap")

    impact_parser = subparsers.add_parser("impact")
    _add_change_source_args(impact_parser)
    impact_parser.add_argument("--format", choices=("text", "json"), default="text")

    contract_parser = subparsers.add_parser("contract")
    _add_change_source_args(contract_parser)
    contract_parser.add_argument("--strict", action="store_true")

    review_parser = subparsers.add_parser("review-base")
    review_parser.add_argument("--base-ref", default=DEFAULT_BASE_REF)

    return parser


def _add_change_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source", choices=("working-tree", "staged", "base"), default="working-tree"
    )
    parser.add_argument("--base")
    parser.add_argument("--files", nargs="*")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "bootstrap":
            print(bootstrap_environment(), end="")
            return 0

        if args.command == "impact":
            changed_files = detect_changed_files(
                source=args.source,
                base=args.base,
                files=args.files,
            )
            report = build_impact_report(changed_files)
            rendered = (
                render_impact_json(report) if args.format == "json" else render_impact_text(report)
            )
            print(rendered, end="")
            return 0

        if args.command == "contract":
            changed_files = detect_changed_files(
                source=args.source,
                base=args.base,
                files=args.files,
            )
            report = build_impact_report(changed_files)
            result = evaluate_release_contract(report, strict=args.strict)
            print(render_contract_result(result), end="")
            return 0 if result.ok else 1

        if args.command == "review-base":
            result = build_review_base_result(args.base_ref)
            print(render_review_base_report(result), end="")
            return 0 if result.contract.ok else 1
    except DevToolError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.error("unreachable")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
