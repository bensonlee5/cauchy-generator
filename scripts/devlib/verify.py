from __future__ import annotations

from dataclasses import dataclass

from .common import CommandSpec, DevToolError, format_command, python_tool, run_command
from .contract import evaluate_release_contract, render_contract_result
from .deps import build_import_graph, dependency_docs_are_current
from .doctor import doctor_passed, render_doctor_results, run_doctor
from .impact import ImpactReport, build_impact_report, detect_changed_files
from .test_selection import PytestSelection, is_docs_only_change_set


@dataclass(frozen=True)
class VerifyPlan:
    mode: str
    headline: str
    commands: tuple[CommandSpec, ...]
    report: ImpactReport


def build_verify_plan(
    *,
    mode: str,
    source: str,
    base: str | None,
    files: list[str] | None,
    incremental: bool,
    parallel: bool,
) -> VerifyPlan:
    changed_files = detect_changed_files(source=source, base=base, files=files)
    graph = build_import_graph()
    report = build_impact_report(changed_files, graph=graph)
    docs_only = is_docs_only_change_set(report.changed_files)

    commands: list[CommandSpec] = []
    headline = f"verify {mode}"
    if mode in {"quick", "code", "full", "affected"} and not docs_only:
        commands.extend(_code_quick_commands(report, graph_changed=bool(report.changed_modules)))
    if mode in {"docs", "full"} or docs_only:
        commands.extend(_docs_commands())
    if mode in {"code", "full"} and not docs_only:
        commands.extend(_pytest_commands(incremental=incremental, parallel=parallel))
    if mode == "affected" and not docs_only:
        commands.extend(
            _affected_pytest_commands(
                selection=report.pytest_selection,
                incremental=incremental,
                parallel=parallel,
            )
        )
    if mode in {"bench", "full"}:
        commands.extend(_bench_commands())
    if mode == "quick" and docs_only:
        headline = "verify quick (docs-only change set)"
    if mode == "affected" and docs_only:
        headline = "verify affected (docs-only change set)"
    return VerifyPlan(mode=mode, headline=headline, commands=tuple(commands), report=report)


def execute_verify_plan(plan: VerifyPlan, *, dry_run: bool) -> str:
    lines = [
        plan.headline,
        "recommended modes: "
        + (", ".join(plan.report.recommended_modes) if plan.report.recommended_modes else "none"),
    ]
    if plan.report.suggested_pytest_targets:
        lines.append("suggested pytest targets: " + ", ".join(plan.report.suggested_pytest_targets))
    if plan.mode == "affected":
        selection = plan.report.pytest_selection
        lines.append(f"pytest selection: {selection.mode}")
        lines.append(f"pytest selection reason: {selection.reason}")
        if selection.targets:
            lines.append("pytest selection targets: " + ", ".join(selection.targets))
    code_results = run_doctor("code")
    needs_code_doctor = any(not command.label.startswith("docs") for command in plan.commands)
    if needs_code_doctor and not doctor_passed(code_results):
        raise DevToolError(render_doctor_results(code_results).strip())
    if "docs" in plan.report.tags:
        docs_results = run_doctor("docs")
        if any(command.label.startswith("docs") for command in plan.commands) and not doctor_passed(
            docs_results
        ):
            raise DevToolError(render_doctor_results(docs_results).strip())

    contract_result = evaluate_release_contract(plan.report)
    if contract_result.warnings:
        lines.extend(contract_result.warnings)
    if contract_result.errors:
        raise DevToolError(render_contract_result(contract_result).strip())

    if any(command.label == "dependency map freshness" for command in plan.commands):
        if not dependency_docs_are_current(build_import_graph()):
            raise DevToolError(
                "dependency map docs are stale; run `./scripts/dev deps --write-docs`."
            )

    if dry_run:
        lines.extend(f"dry-run: {format_command(command.argv)}" for command in plan.commands)
        return "\n".join(lines) + "\n"

    for command in plan.commands:
        run_command(command)
    lines.extend(f"ran: {format_command(command.argv)}" for command in plan.commands)
    return "\n".join(lines) + "\n"


def _code_quick_commands(report: ImpactReport, *, graph_changed: bool) -> list[CommandSpec]:
    commands = [
        CommandSpec(
            label="ruff check", argv=(python_tool("ruff"), "check", "src", "tests", "scripts")
        ),
        CommandSpec(
            label="ruff format",
            argv=(python_tool("ruff"), "format", "--check", "src", "tests", "scripts"),
        ),
        CommandSpec(label="mypy", argv=(python_tool("mypy"), "src")),
        CommandSpec(label="deptry", argv=(python_tool("deptry"), ".")),
    ]
    if "architecture" in report.tags:
        commands.append(CommandSpec(label="import-linter", argv=(python_tool("lint-imports"),)))
    if graph_changed:
        commands.append(
            CommandSpec(label="dependency map freshness", argv=("./scripts/dev", "deps", "--check"))
        )
    return commands


def _docs_commands() -> list[CommandSpec]:
    python = python_tool("python")
    return [
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
    ]


def _pytest_commands(*, incremental: bool, parallel: bool) -> list[CommandSpec]:
    return [_build_pytest_command(targets=(), incremental=incremental, parallel=parallel)]


def _affected_pytest_commands(
    *, selection: PytestSelection, incremental: bool, parallel: bool
) -> list[CommandSpec]:
    if selection.mode == "skip":
        return []
    return [
        _build_pytest_command(
            targets=selection.targets if selection.mode == "targeted" else (),
            incremental=incremental,
            parallel=parallel,
        )
    ]


def _build_pytest_command(
    *, targets: tuple[str, ...], incremental: bool, parallel: bool
) -> CommandSpec:
    argv = [python_tool("pytest")]
    argv.append("-q")
    if incremental:
        argv.append("--testmon")
    if parallel:
        argv.extend(("-n", "auto"))
    argv.extend(targets)
    return CommandSpec(label="pytest", argv=tuple(argv))


def _bench_commands() -> list[CommandSpec]:
    return [
        CommandSpec(
            label="bench smoke",
            argv=(
                python_tool("dagzoo"),
                "benchmark",
                "--suite",
                "smoke",
                "--preset",
                "cpu",
                "--baseline",
                "benchmarks/baselines/cpu_smoke.json",
                "--warn-threshold-pct",
                "10",
                "--fail-threshold-pct",
                "20",
                "--fail-on-regression",
                "--hardware-policy",
                "none",
                "--no-memory",
                "--out-dir",
                "benchmarks/results/dev_smoke",
            ),
        )
    ]
