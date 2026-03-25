from __future__ import annotations

from dataclasses import dataclass

from .common import normalize_files, run_git_capture, run_git_lines
from .contract import ContractResult, evaluate_release_contract
from .impact import ImpactReport, build_impact_report

DEFAULT_BASE_REF = "origin/main"


@dataclass(frozen=True)
class ReviewScope:
    base_ref: str
    merge_base: str
    changed_files: tuple[str, ...]


@dataclass(frozen=True)
class ReviewBaseResult:
    scope: ReviewScope
    report: ImpactReport
    contract: ContractResult


def read_merge_base(base_ref: str = DEFAULT_BASE_REF) -> str:
    return run_git_capture("merge-base", base_ref, "HEAD").strip()


def collect_review_scope(base_ref: str = DEFAULT_BASE_REF) -> ReviewScope:
    merge_base = read_merge_base(base_ref)
    changed_files = tuple(
        sorted(
            dict.fromkeys(
                normalize_files(
                    (
                        *run_git_lines("diff", "--name-only", f"{merge_base}..HEAD"),
                        *run_git_lines("diff", "--name-only", "--cached"),
                        *run_git_lines("diff", "--name-only"),
                        *run_git_lines("ls-files", "--others", "--exclude-standard"),
                    )
                )
            )
        )
    )
    return ReviewScope(base_ref=base_ref, merge_base=merge_base, changed_files=changed_files)


def build_review_base_result(base_ref: str = DEFAULT_BASE_REF) -> ReviewBaseResult:
    scope = collect_review_scope(base_ref)
    report = build_impact_report(scope.changed_files)
    contract = evaluate_release_contract(report)
    return ReviewBaseResult(scope=scope, report=report, contract=contract)


def render_review_base_report(result: ReviewBaseResult) -> str:
    lines = [
        "review-base",
        f"base ref: {result.scope.base_ref}",
        f"merge base: {result.scope.merge_base}",
        "changed files:",
    ]
    if result.scope.changed_files:
        lines.extend(f"- `{path}`" for path in result.scope.changed_files)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "tags:",
            "- "
            + (
                ", ".join(f"`{tag}`" for tag in result.report.tags)
                if result.report.tags
                else "none"
            ),
            "",
            "recommended verify modes:",
            "- "
            + (
                ", ".join(f"`{mode}`" for mode in result.report.recommended_modes)
                if result.report.recommended_modes
                else "none"
            ),
        ]
    )
    if result.report.suggested_pytest_targets:
        lines.extend(
            [
                "",
                "suggested pytest targets:",
                "- "
                + ", ".join(f"`{target}`" for target in result.report.suggested_pytest_targets),
            ]
        )
    lines.extend(
        [
            "",
            "pytest selection:",
            f"- mode: `{result.report.pytest_selection.mode}`",
            f"- reason: {result.report.pytest_selection.reason}",
        ]
    )
    if result.report.pytest_selection.targets:
        lines.append(
            "- targets: "
            + ", ".join(f"`{target}`" for target in result.report.pytest_selection.targets)
        )

    lines.extend(["", "release contract:"])
    if result.contract.warnings:
        lines.extend(f"- warning: {warning}" for warning in result.contract.warnings)
    if result.contract.errors:
        lines.extend(f"- error: {error}" for error in result.contract.errors)
    if not result.contract.warnings and not result.contract.errors:
        lines.append("- release contract checks passed.")

    return "\n".join(lines) + "\n"
