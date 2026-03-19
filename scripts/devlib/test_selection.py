from __future__ import annotations

from dataclasses import dataclass

from .deps import module_to_package
from .review_policy import pytest_targets_for_path, requires_full_pytest

DOCS_ONLY_PATH_PREFIXES = ("docs/", "site/")

PACKAGE_PYTEST_TARGETS: dict[str, tuple[str, ...]] = {
    "dagzoo.cli": (
        "tests/test_cli_validation.py",
        "tests/test_cli_outputs.py",
        "tests/test_benchmark_cli.py",
        "tests/test_generate_handoff.py",
    ),
    "dagzoo.bench": (
        "tests/test_benchmark_suite.py",
        "tests/test_benchmark_cli.py",
        "tests/test_benchmark_stage_metrics.py",
        "tests/test_benchmark_throughput.py",
        "tests/test_benchmark_regression.py",
        "tests/test_h100_validation.py",
    ),
    "dagzoo.config": (
        "tests/test_config.py",
        "tests/test_config_resolution.py",
        "tests/test_generate_handoff.py",
    ),
    "dagzoo.core": (
        "tests/test_generate.py",
        "tests/test_dag_sampler.py",
        "tests/test_execution_semantics.py",
        "tests/test_fixed_layout_batched.py",
        "tests/test_hardware.py",
        "tests/test_node_pipeline.py",
        "tests/test_postprocess.py",
        "tests/test_rng.py",
        "tests/test_lineage_schema.py",
        "tests/test_lineage_artifact.py",
        "tests/test_generate_handoff.py",
        "tests/test_corpus_probe.py",
        "tests/test_trees.py",
    ),
    "dagzoo.functions": (
        "tests/test_activations.py",
        "tests/test_multi_function.py",
        "tests/test_random_functions.py",
    ),
    "dagzoo.converters": (
        "tests/test_numeric_converter.py",
        "tests/test_categorical_converter.py",
    ),
    "dagzoo.sampling": (
        "tests/test_sampling.py",
        "tests/test_noise_sampling.py",
        "tests/test_noise_config.py",
        "tests/test_correlated.py",
        "tests/test_missingness_sampling.py",
        "tests/test_random_points.py",
        "tests/test_random_matrices.py",
    ),
    "dagzoo.io": (
        "tests/test_lineage_artifact.py",
        "tests/test_lineage_schema.py",
        "tests/test_parquet_writer.py",
        "tests/test_generate_handoff.py",
    ),
    "dagzoo.filtering": (
        "tests/test_filtering_availability.py",
        "tests/test_extra_trees_filter.py",
        "tests/test_deferred_filter.py",
        "tests/test_filter_calibration.py",
    ),
    "dagzoo.diagnostics": (
        "tests/test_diagnostics_coverage.py",
        "tests/test_diagnostics_metrics.py",
        "tests/test_effective_diversity_audit.py",
        "tests/test_corpus_probe.py",
        "tests/test_diversity_audit_cli.py",
        "tests/test_filter_calibration.py",
    ),
    "dagzoo.graph": (
        "tests/test_dag_sampler.py",
        "tests/test_trees.py",
    ),
    "dagzoo.math": ("tests/test_random_matrices.py",),
    "dagzoo.postprocess": ("tests/test_postprocess.py",),
}


@dataclass(frozen=True)
class PytestSelection:
    mode: str
    targets: tuple[str, ...]
    reason: str


def build_pytest_selection(
    *,
    changed_files: tuple[str, ...],
    changed_modules: tuple[str, ...],
    impacted_packages: tuple[str, ...],
) -> PytestSelection:
    if changed_files and _is_docs_only_change_set(changed_files):
        return PytestSelection(mode="skip", targets=(), reason="docs-only change set")

    for path in changed_files:
        if requires_full_pytest(path):
            return PytestSelection(
                mode="full",
                targets=(),
                reason=f"full pytest required for root/tooling/test path `{path}`",
            )

    if not changed_modules:
        return PytestSelection(
            mode="full",
            targets=(),
            reason="no changed Python modules to narrow from",
        )

    changed_packages = {module_to_package(module) for module in changed_modules}
    if "dagzoo" in changed_packages or "dagzoo.__main__" in changed_packages:
        return PytestSelection(
            mode="full",
            targets=(),
            reason="root package or entrypoint change requires the full test suite",
        )

    targets: list[str] = []
    path_signals: list[str] = []
    package_signals: list[str] = []

    for path in changed_files:
        if not path.startswith("src/dagzoo/"):
            continue
        path_targets = pytest_targets_for_path(path)
        if path_targets:
            targets.extend(path_targets)
            path_signals.append(path)

    for package in sorted(changed_packages | set(impacted_packages)):
        package_targets = PACKAGE_PYTEST_TARGETS.get(package)
        if package_targets:
            targets.extend(package_targets)
            package_signals.append(package)

    unique_targets = tuple(dict.fromkeys(targets))
    if not unique_targets:
        return PytestSelection(
            mode="full",
            targets=(),
            reason="no safe narrowed pytest target set derived",
        )

    reason_parts: list[str] = []
    if path_signals:
        reason_parts.append("curated path prefixes: " + ", ".join(_compact(path_signals)))
    if package_signals:
        reason_parts.append("impacted packages: " + ", ".join(package_signals))
    return PytestSelection(
        mode="targeted",
        targets=unique_targets,
        reason="; ".join(reason_parts) if reason_parts else "narrowed pytest target set",
    )


def _is_docs_only_change_set(changed_files: tuple[str, ...]) -> bool:
    return all(
        path == "README.md" or path.startswith(DOCS_ONLY_PATH_PREFIXES) for path in changed_files
    )


def _compact(values: list[str], limit: int = 3) -> list[str]:
    unique = list(dict.fromkeys(values))
    if len(unique) <= limit:
        return unique
    return unique[:limit] + [f"+{len(unique) - limit} more"]
