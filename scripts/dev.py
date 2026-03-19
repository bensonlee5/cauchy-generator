#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from devlib.common import DOCS_DEP_MAP_PATH, DevToolError, repo_relative  # noqa: E402
from devlib.contract import evaluate_release_contract, render_contract_result  # noqa: E402
from devlib.deps import (  # noqa: E402
    build_import_graph,
    dependency_docs_are_current,
    render_scope_json,
    render_scope_text,
    write_dependency_docs,
)
from devlib.doctor import doctor_passed, render_doctor_results, run_doctor  # noqa: E402
from devlib.impact import (  # noqa: E402
    build_impact_report,
    detect_changed_files,
)
from devlib.impact import render_json as render_impact_json  # noqa: E402
from devlib.impact import render_text as render_impact_text  # noqa: E402
from devlib.verify import build_verify_plan, execute_verify_plan  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="./scripts/dev")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("mode", choices=("code", "docs", "all"), nargs="?", default="all")

    deps_parser = subparsers.add_parser("deps")
    deps_parser.add_argument("--scope", choices=("package", "hybrid", "full"), default="hybrid")
    deps_parser.add_argument("--format", choices=("text", "json"), default="text")
    deps_parser.add_argument("--write-docs", action="store_true")
    deps_parser.add_argument("--check", action="store_true")

    impact_parser = subparsers.add_parser("impact")
    _add_change_source_args(impact_parser)
    impact_parser.add_argument("--format", choices=("text", "json"), default="text")

    contract_parser = subparsers.add_parser("contract")
    _add_change_source_args(contract_parser)
    contract_parser.add_argument("--strict", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "mode", choices=("quick", "code", "docs", "bench", "full", "affected")
    )
    _add_change_source_args(verify_parser)
    verify_parser.add_argument("--dry-run", action="store_true")
    verify_parser.add_argument("--incremental", action="store_true")
    verify_parser.add_argument("--parallel", action="store_true")

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
        if args.command == "doctor":
            results = run_doctor(args.mode)
            print(render_doctor_results(results), end="")
            return 0 if doctor_passed(results) else 1

        if args.command == "deps":
            graph = build_import_graph()
            if args.write_docs:
                write_dependency_docs(graph)
                print(f"wrote {repo_relative(DOCS_DEP_MAP_PATH)}")
            if args.check:
                if not dependency_docs_are_current(graph):
                    print(
                        "dependency map docs are stale; run `./scripts/dev deps --write-docs`.",
                        file=sys.stderr,
                    )
                    return 1
                if not args.write_docs:
                    print("dependency map docs are current.")
                return 0
            if args.write_docs:
                return 0
            rendered = (
                render_scope_json(graph, args.scope)
                if args.format == "json"
                else render_scope_text(graph, args.scope)
            )
            print(rendered, end="")
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

        if args.command == "verify":
            plan = build_verify_plan(
                mode=args.mode,
                source=args.source,
                base=args.base,
                files=args.files,
                incremental=args.incremental,
                parallel=args.parallel,
            )
            print(execute_verify_plan(plan, dry_run=args.dry_run), end="")
            return 0
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
