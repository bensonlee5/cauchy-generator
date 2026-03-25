from __future__ import annotations

import json
from dataclasses import dataclass

from .common import normalize_files, run_git_capture, run_git_lines
from .deps import ImportGraph, build_import_graph, module_to_package, path_to_module
from .review_policy import is_release_risk_path, suggested_pytest_targets
from .test_selection import PytestSelection, build_pytest_selection, is_docs_only_change_set

ARCHITECTURE_PATH_PREFIXES = (
    "src/dagzoo/core/",
    "src/dagzoo/functions/",
    "src/dagzoo/converters/",
    "src/dagzoo/sampling/",
    "src/dagzoo/io/",
    "src/dagzoo/filtering/",
)


@dataclass(frozen=True)
class ImpactModuleSummary:
    module: str
    direct_downstream: tuple[str, ...]
    transitive_downstream: tuple[str, ...]
    downstream_packages: tuple[str, ...]


@dataclass(frozen=True)
class ImpactReport:
    changed_files: tuple[str, ...]
    tags: tuple[str, ...]
    changed_modules: tuple[str, ...]
    module_summaries: tuple[ImpactModuleSummary, ...]
    pytest_selection: PytestSelection
    recommended_modes: tuple[str, ...]
    suggested_pytest_targets: tuple[str, ...]


def detect_changed_files(
    *, source: str = "working-tree", base: str | None = None, files: list[str] | None = None
) -> tuple[str, ...]:
    if files:
        return tuple(sorted(dict.fromkeys(normalize_files(files))))
    if source == "working-tree":
        changed_files: list[str] = []
        for line in run_git_capture("status", "--short").splitlines():
            if not line:
                continue
            changed_files.append(line[3:])
        return tuple(sorted(dict.fromkeys(normalize_files(changed_files))))
    if source == "staged":
        return tuple(
            sorted(dict.fromkeys(normalize_files(run_git_lines("diff", "--cached", "--name-only"))))
        )
    if source != "base":
        raise ValueError(f"Unsupported change source: {source!r}")
    if base is None:
        raise ValueError("base is required when source is not working-tree.")
    return tuple(
        sorted(
            dict.fromkeys(normalize_files(run_git_lines("diff", "--name-only", f"{base}...HEAD")))
        )
    )


def build_impact_report(
    changed_files: tuple[str, ...], graph: ImportGraph | None = None
) -> ImpactReport:
    graph = build_import_graph() if graph is None else graph
    tags = classify_tags(changed_files)
    changed_modules = tuple(
        sorted(module for path in changed_files if (module := path_to_module(path)) is not None)
    )
    module_summaries = tuple(
        ImpactModuleSummary(
            module=module,
            direct_downstream=graph.reverse_imports.get(module, ()),
            transitive_downstream=graph.transitive_importers(module),
            downstream_packages=tuple(
                sorted(
                    {
                        module_to_package(importer)
                        for importer in graph.transitive_importers(module)
                        if module_to_package(importer)
                        not in {module_to_package(module), "dagzoo", "dagzoo.__main__"}
                    }
                )
            ),
        )
        for module in changed_modules
    )
    impacted_packages = tuple(
        sorted({package for summary in module_summaries for package in summary.downstream_packages})
    )
    return ImpactReport(
        changed_files=changed_files,
        tags=tags,
        changed_modules=changed_modules,
        module_summaries=module_summaries,
        pytest_selection=build_pytest_selection(
            changed_files=changed_files,
            changed_modules=changed_modules,
            impacted_packages=impacted_packages,
        ),
        recommended_modes=recommend_modes(changed_files, tags, module_summaries),
        suggested_pytest_targets=suggested_pytest_targets(changed_files),
    )


def classify_tags(changed_files: tuple[str, ...]) -> tuple[str, ...]:
    tags: set[str] = set()
    for path in changed_files:
        if (
            path == "README.md"
            or path.startswith("docs/")
            or path.startswith("site/")
            or path.startswith("scripts/docs/")
        ):
            tags.add("docs")
        if (
            path.startswith("src/dagzoo/")
            or path.startswith("tests/")
            or path
            in (
                "pyproject.toml",
                ".pre-commit-config.yaml",
            )
        ):
            tags.add("code")
        if (
            path.startswith("src/dagzoo/bench/")
            or path.startswith("configs/benchmark")
            or path.startswith("benchmarks/baselines/")
            or path.startswith("scripts/benchmark")
        ):
            tags.add("bench")
        if path.startswith(ARCHITECTURE_PATH_PREFIXES):
            tags.add("architecture")
        if is_release_risk_path(path):
            tags.add("release-risk")
        if (
            path.startswith("scripts/")
            or path.startswith(".github/workflows/")
            or path == "AGENTS.md"
        ):
            tags.add("tooling")
    return tuple(sorted(tags))


def recommend_modes(
    changed_files: tuple[str, ...],
    tags: tuple[str, ...],
    module_summaries: tuple[ImpactModuleSummary, ...],
) -> tuple[str, ...]:
    recommended: list[str] = []
    impacted_packages = {
        package for summary in module_summaries for package in summary.downstream_packages
    }
    if is_docs_only_change_set(changed_files):
        return ("docs",)
    recommended.append("quick")
    if "code" in tags:
        recommended.append("code")
    if "docs" in tags:
        recommended.append("docs")
    if "bench" in tags or "dagzoo.bench" in impacted_packages:
        recommended.append("bench")
    return tuple(dict.fromkeys(recommended))


def render_text(report: ImpactReport) -> str:
    lines = [
        "Changed files:",
        *[f"- `{path}`" for path in report.changed_files],
        "",
        "Tags:",
        f"- {', '.join(f'`{tag}`' for tag in report.tags) if report.tags else 'none'}",
        "",
        "Recommended verify modes:",
        f"- {', '.join(f'`{mode}`' for mode in report.recommended_modes) if report.recommended_modes else 'none'}",
    ]
    if report.suggested_pytest_targets:
        lines.extend(
            [
                "",
                "Suggested pytest targets:",
                "- " + ", ".join(f"`{target}`" for target in report.suggested_pytest_targets),
            ]
        )
    lines.extend(
        [
            "",
            "Pytest selection:",
            f"- mode: `{report.pytest_selection.mode}`",
            f"- reason: {report.pytest_selection.reason}",
        ]
    )
    if report.pytest_selection.targets:
        lines.append(
            "- targets: " + ", ".join(f"`{target}`" for target in report.pytest_selection.targets)
        )
    if report.module_summaries:
        lines.extend(["", "Module impact:"])
        for summary in report.module_summaries:
            direct = ", ".join(f"`{module}`" for module in summary.direct_downstream) or "none"
            transitive = (
                ", ".join(f"`{module}`" for module in summary.transitive_downstream) or "none"
            )
            packages = (
                ", ".join(f"`{package}`" for package in summary.downstream_packages) or "none"
            )
            lines.extend(
                [
                    f"- `{summary.module}`",
                    f"  direct downstream: {direct}",
                    f"  transitive downstream: {transitive}",
                    f"  downstream packages: {packages}",
                ]
            )
    return "\n".join(lines) + "\n"


def render_json(report: ImpactReport) -> str:
    payload = {
        "changed_files": list(report.changed_files),
        "tags": list(report.tags),
        "recommended_modes": list(report.recommended_modes),
        "suggested_pytest_targets": list(report.suggested_pytest_targets),
        "pytest_selection": {
            "mode": report.pytest_selection.mode,
            "targets": list(report.pytest_selection.targets),
            "reason": report.pytest_selection.reason,
        },
        "modules": [
            {
                "module": summary.module,
                "direct_downstream": list(summary.direct_downstream),
                "transitive_downstream": list(summary.transitive_downstream),
                "downstream_packages": list(summary.downstream_packages),
            }
            for summary in report.module_summaries
        ],
    }
    return json.dumps(payload, indent=2) + "\n"
