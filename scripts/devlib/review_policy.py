from __future__ import annotations

RELEASE_RISK_EXACT_PATHS = frozenset(
    {
        "src/dagzoo/core/config_resolution.py",
        "src/dagzoo/core/dataset.py",
        "src/dagzoo/core/fixed_layout/runtime.py",
        "src/dagzoo/core/metadata.py",
    }
)
RELEASE_RISK_PREFIXES = (
    "src/dagzoo/cli/",
    "src/dagzoo/config/",
    "src/dagzoo/io/",
    "configs/",
)

FULL_PYTEST_EXACT_PATHS = frozenset(
    {
        "pyproject.toml",
        ".pre-commit-config.yaml",
        "AGENTS.md",
        "src/dagzoo/__init__.py",
        "src/dagzoo/__main__.py",
    }
)
FULL_PYTEST_PREFIXES = (
    "scripts/",
    ".github/workflows/",
    "tests/",
)

PYTEST_TARGETS_BY_PREFIX = (
    (
        "src/dagzoo/cli/",
        (
            "tests/test_cli_validation.py",
            "tests/test_cli_outputs.py",
            "tests/test_benchmark_cli.py",
            "tests/test_generate_handoff.py",
        ),
    ),
    (
        "src/dagzoo/bench/",
        (
            "tests/test_benchmark_suite.py",
            "tests/test_benchmark_cli.py",
            "tests/test_benchmark_stage_metrics.py",
            "tests/test_benchmark_throughput.py",
            "tests/test_benchmark_regression.py",
            "tests/scripts/test_h100_validation.py",
        ),
    ),
    (
        "src/dagzoo/config/",
        (
            "tests/test_config.py",
            "tests/test_config_resolution.py",
            "tests/test_generate_handoff.py",
        ),
    ),
    (
        "src/dagzoo/core/config_resolution.py",
        (
            "tests/test_config.py",
            "tests/test_config_resolution.py",
            "tests/test_generate_handoff.py",
        ),
    ),
    (
        "src/dagzoo/core/",
        (
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
    ),
    (
        "src/dagzoo/functions/",
        (
            "tests/test_activations.py",
            "tests/test_multi_function.py",
            "tests/test_random_functions.py",
        ),
    ),
    (
        "src/dagzoo/converters/",
        (
            "tests/test_numeric_converter.py",
            "tests/test_categorical_converter.py",
        ),
    ),
    (
        "src/dagzoo/sampling/",
        (
            "tests/test_sampling.py",
            "tests/test_noise_sampling.py",
            "tests/test_noise_config.py",
            "tests/test_correlated.py",
            "tests/test_missingness_sampling.py",
            "tests/test_random_points.py",
            "tests/test_random_matrices.py",
        ),
    ),
    (
        "src/dagzoo/io/",
        (
            "tests/test_lineage_artifact.py",
            "tests/test_lineage_schema.py",
            "tests/test_parquet_writer.py",
            "tests/test_generate_handoff.py",
        ),
    ),
    (
        "src/dagzoo/filtering/",
        ("tests/test_deferred_filter.py",),
    ),
    (
        "src/dagzoo/diagnostics/",
        (
            "tests/test_diagnostics_coverage.py",
            "tests/test_diagnostics_metrics.py",
            "tests/test_effective_diversity_audit.py",
            "tests/test_corpus_probe.py",
            "tests/test_diversity_audit_cli.py",
        ),
    ),
    (
        "src/dagzoo/graph/",
        (
            "tests/test_dag_sampler.py",
            "tests/test_trees.py",
        ),
    ),
    (
        "src/dagzoo/math/",
        ("tests/test_random_matrices.py",),
    ),
    (
        "src/dagzoo/postprocess/",
        ("tests/test_postprocess.py",),
    ),
    (
        "configs/",
        (
            "tests/test_config.py",
            "tests/test_config_resolution.py",
            "tests/test_generate_handoff.py",
            "tests/test_benchmark_cli.py",
        ),
    ),
    ("scripts/devlib/", ("tests/test_dev_tooling.py",)),
    (".pre-commit-config.yaml", ("tests/test_dev_tooling.py",)),
    (
        "scripts/docs/",
        (
            "tests/test_docs_scripts.py",
            "tests/test_dev_tooling.py",
        ),
    ),
    ("README.md", ("tests/test_docs_scripts.py",)),
    ("docs/", ("tests/test_docs_scripts.py",)),
    ("site/", ("tests/test_docs_scripts.py",)),
)


def is_release_risk_path(path: str) -> bool:
    return path in RELEASE_RISK_EXACT_PATHS or path.startswith(RELEASE_RISK_PREFIXES)


def requires_full_pytest(path: str) -> bool:
    return path in FULL_PYTEST_EXACT_PATHS or path.startswith(FULL_PYTEST_PREFIXES)


def pytest_targets_for_path(path: str) -> tuple[str, ...]:
    targets: list[str] = []
    for path_prefix, candidates in PYTEST_TARGETS_BY_PREFIX:
        if path == path_prefix or path.startswith(path_prefix):
            targets.extend(candidates)
    return tuple(dict.fromkeys(targets))


def suggested_pytest_targets(changed_files: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    suggested: list[str] = []
    for changed_file in changed_files:
        for path_prefix, targets in PYTEST_TARGETS_BY_PREFIX:
            if changed_file == path_prefix or changed_file.startswith(path_prefix):
                for target in targets:
                    if target in seen:
                        continue
                    seen.add(target)
                    suggested.append(target)
    return tuple(suggested)
