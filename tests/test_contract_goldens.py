from __future__ import annotations

import json
import re
from pathlib import Path

from golden_support import (
    assert_normalized_json_equal,
    assert_normalized_text_equal,
    load_golden_json,
    normalize_benchmark_summary,
    normalize_benchmark_summary_markdown,
    normalize_coverage_summary,
    normalize_handoff_manifest,
)

from dagzoo.bench.report import write_suite_json, write_suite_markdown
from dagzoo.config import GeneratorConfig
from dagzoo.core.dataset import generate_batch
from dagzoo.core.generate_handoff import write_generate_handoff_manifest
from dagzoo.diagnostics.coverage import (
    CoverageAggregationConfig,
    CoverageAggregator,
    write_coverage_summary_json,
)
from dagzoo.diagnostics.types import DatasetMetrics
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.io.shard_contract import DATASET_CATALOG_FILENAME

REPO_ROOT = Path(__file__).resolve().parents[1]


def _flatten_path_tokens(value: object, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    flattened: list[tuple[str, ...]] = []
    if prefix:
        flattened.append(prefix)
    if isinstance(value, dict):
        for key, child in value.items():
            flattened.extend(_flatten_path_tokens(child, prefix + (str(key),)))
    elif isinstance(value, list):
        for child in value:
            flattened.extend(_flatten_path_tokens(child, prefix + ("[]",)))
    return flattened


def _format_tokens(tokens: tuple[str, ...]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token == "[]":
            if parts:
                parts[-1] = f"{parts[-1]}[]"
            else:
                parts.append("[]")
            continue
        parts.append(token)
    return ".".join(parts)


def _benchmark_summary_fixture(tmp_path: Path) -> dict[str, object]:
    return {
        "suite": "smoke",
        "generated_at": "2026-03-19T12:00:00+00:00",
        "regression": {
            "status": "warn",
            "warn_threshold_pct": 10.0,
            "fail_threshold_pct": 20.0,
            "hard_fail": False,
            "issues": [
                {
                    "severity": "warn",
                    "preset": "cpu",
                    "metric": "datasets_per_minute",
                    "current": 100.0,
                    "baseline": 120.0,
                    "degradation_pct": 16.6666666667,
                }
            ],
        },
        "preset_results": [
            {
                "preset_key": "cpu",
                "dataset_rows_total": 1024,
                "generation_mode": "fixed_batched",
                "device": "cpu",
                "hardware_backend": "cpu",
                "datasets_per_minute": 120.5,
                "generation_datasets_per_minute": 121.0,
                "write_datasets_per_minute": 200.0,
                "filter_datasets_per_minute": None,
                "filter_accepted_datasets_per_minute": None,
                "reproducibility_match": True,
                "reproducibility_workload_match": False,
                "filter_rejection_rate_attempt_level": None,
                "filter_acceptance_rate_dataset_level": None,
                "filter_rejection_rate_dataset_level": None,
                "filter_retry_dataset_rate": None,
                "elapsed_seconds": 1.234,
                "latency_p95_ms": 12.34,
                "peak_rss_mb": 64.0,
                "diagnostics_enabled": True,
                "diagnostics_artifacts": {
                    "json": str(tmp_path / "diagnostics" / "coverage_summary.json"),
                    "markdown": str(tmp_path / "diagnostics" / "coverage_summary.md"),
                },
                "prepare_elapsed_seconds": 0.123,
                "prepare_cpu_time_seconds": 0.045,
                "prepare_cpu_busy_pct_of_wall": 0.365,
                "generation_elapsed_seconds": 0.456,
                "generation_cpu_time_seconds": 0.321,
                "generation_cpu_busy_pct_of_wall": 0.704,
                "raw_batch_elapsed_seconds": 0.222,
                "raw_batch_cpu_time_seconds": 0.111,
                "node_apply_elapsed_seconds": 0.05,
                "converter_elapsed_seconds": 0.04,
                "feature_materialization_elapsed_seconds": 0.03,
                "fixed_layout_target_cells_effective": 1024,
                "fixed_layout_per_dataset_cells": 32,
                "fixed_layout_realized_batch_size": 2,
                "fixed_layout_chunk_count": 1,
                "fixed_layout_tail_chunk_size": 0,
                "stage_sample_datasets": 2,
                "write_stage_elapsed_seconds": 0.789,
                "write_stage_cpu_time_seconds": 0.555,
                "write_stage_bytes_written": 4096,
                "write_stage_mib_per_second": 4.75,
                "filter_stage_elapsed_seconds": None,
                "filter_stage_cpu_time_seconds": None,
                "peak_cuda_reserved_mb": 128.0,
                "peak_cuda_reserved_pct_of_total_memory": 0.25,
                "peak_cuda_headroom_mb": 256.0,
                "scenarios": {
                    "baseline": {
                        "enabled": True,
                        "status": "pass",
                        "metrics": {},
                        "issues": [],
                    },
                    "throughput": {
                        "enabled": True,
                        "status": "pass",
                        "metrics": {},
                        "issues": [],
                    },
                    "filtering": {"enabled": False, "status": "off", "metrics": {}, "issues": []},
                    "missingness": {"enabled": False, "status": "off", "metrics": {}, "issues": []},
                    "shift": {"enabled": False, "status": "off", "metrics": {}, "issues": []},
                    "noise": {"enabled": False, "status": "off", "metrics": {}, "issues": []},
                },
            }
        ],
    }


def _write_generated_metadata(run_root: Path) -> None:
    generated_dir = run_root / "generated"
    shard_dir = generated_dir / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / DATASET_CATALOG_FILENAME).write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset_index": 0,
                        "dataset_id": "2" * 32,
                        "task": "classification",
                        "n_train": 16,
                        "n_test": 8,
                        "n_features": 8,
                        "feature_types": ["num"] * 8,
                        "n_classes": 3,
                        "group_ids": {
                            "request_run": "1" * 32,
                            "layout_plan": "4" * 32,
                        },
                        "intervention": {
                            "mode": "hard_interventional",
                            "signature": "a" * 32,
                        },
                        "target_derivation": "tabiclv2_latent_node",
                        "target_relevance": {
                            "feature_count": 5,
                            "feature_fraction": 0.625,
                        },
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "dataset_index": 1,
                        "dataset_id": "3" * 32,
                        "task": "classification",
                        "n_train": 16,
                        "n_test": 8,
                        "n_features": 8,
                        "feature_types": ["num"] * 8,
                        "n_classes": 3,
                        "group_ids": {
                            "request_run": "1" * 32,
                            "layout_plan": "4" * 32,
                        },
                        "intervention": {
                            "mode": "hard_interventional",
                            "signature": "a" * 32,
                        },
                        "target_derivation": "tabiclv2_latent_node",
                        "target_relevance": {
                            "feature_count": 6,
                            "feature_fraction": 0.75,
                        },
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_benchmark_summary_contract_golden(tmp_path: Path) -> None:
    summary = _benchmark_summary_fixture(tmp_path)
    path = write_suite_json(summary, tmp_path / "summary.json")
    actual = json.loads(path.read_text(encoding="utf-8"))

    assert_normalized_json_equal(
        actual,
        "benchmark_summary.json",
        normalizer=normalize_benchmark_summary,
    )


def test_benchmark_summary_markdown_contract_golden(tmp_path: Path) -> None:
    summary = _benchmark_summary_fixture(tmp_path)
    path = write_suite_markdown(summary, tmp_path / "summary.md")

    assert_normalized_text_equal(
        path.read_text(encoding="utf-8"),
        "benchmark_summary_excerpt.md",
        normalizer=normalize_benchmark_summary_markdown,
    )


def test_generate_handoff_manifest_contract_golden(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    generated_dir = run_root / "generated"
    _write_generated_metadata(run_root)
    effective_config = generated_dir / "effective_config.yaml"
    effective_trace = generated_dir / "effective_config_trace.yaml"
    effective_config.write_text(
        "seed: 7\noutput:\n  out_dir: /tmp/placeholder\n",
        encoding="utf-8",
    )
    effective_trace.write_text(
        "- source: generate.handoff_root\n"
        "  path: output.out_dir\n"
        "  old_value: /tmp/placeholder\n"
        "  new_value: /tmp/placeholder/generated\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 7\n", encoding="utf-8")

    manifest_path = write_generate_handoff_manifest(
        config_path=config_path,
        generate_invocation_overrides={
            "num_datasets": 2,
            "seed": 7,
            "rows": "1024..4096",
            "device": "cpu",
            "hardware_policy": "none",
            "missing_rate": None,
            "missing_mechanism": None,
            "missing_mar_observed_fraction": None,
            "missing_mar_logit_scale": None,
            "missing_mnar_logit_scale": None,
            "diagnostics": False,
            "diagnostics_out_dir": None,
            "handoff_root": str(run_root),
        },
        run_root=run_root,
        generated_dir=generated_dir,
        effective_config_path=effective_config,
        effective_config_trace_path=effective_trace,
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
    )
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert_normalized_json_equal(
        actual,
        "handoff_manifest.json",
        normalizer=normalize_handoff_manifest,
    )


def _coverage_metric_fixture() -> DatasetMetrics:
    return DatasetMetrics(
        task="classification",
        n_rows=64,
        n_features=8,
        n_classes=3,
        n_categorical_features=2,
        categorical_ratio=0.25,
        graph_edge_density=0.4,
        graph_indegree_std=0.6,
        graph_outdegree_std=0.5,
        graph_depth_ratio=0.75,
        graph_reachability_ratio=0.5,
        graph_ancestor_overlap_mean=0.3,
        graph_target_ancestor_fraction=0.625,
        mechanism_family_cooccurrence_ratio=0.4,
        shift_enabled=0.0,
        shift_graph_scale=0.0,
        shift_mechanism_scale=0.0,
        shift_variance_scale=0.0,
        shift_edge_odds_multiplier=1.0,
        shift_mechanism_nonlinear_mass=0.25,
        shift_noise_variance_multiplier=1.0,
        linearity_proxy=0.25,
        nonlinearity_proxy=0.75,
        wins_ratio_proxy=0.75,
        pearson_abs_mean=0.2,
        pearson_abs_max=0.5,
        spearman_abs_mean=None,
        spearman_abs_max=None,
        class_entropy=1.0,
        majority_minority_ratio=1.2,
        snr_proxy_db=3.0,
        cat_cardinality_min=2,
        cat_cardinality_mean=2.5,
        cat_cardinality_max=4,
    )


def test_coverage_summary_contract_golden(tmp_path: Path) -> None:
    agg = CoverageAggregator(
        CoverageAggregationConfig(
            histogram_bins=4,
            quantiles=(0.25, 0.5, 0.75),
            target_bands={"pearson_abs_mean": (0.0, 1.0)},
        )
    )
    agg.update_metrics(_coverage_metric_fixture())
    summary = agg.build_summary()
    path = write_coverage_summary_json(summary, tmp_path / "coverage_summary.json")
    actual = json.loads(path.read_text(encoding="utf-8"))

    assert_normalized_json_equal(
        actual,
        "coverage_summary.json",
        normalizer=normalize_coverage_summary,
    )


def test_generated_metadata_record_paths_contract_golden(tmp_path: Path) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.dataset.task = "classification"
    cfg.dataset.n_train = 16
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.dataset.n_classes_min = 4
    cfg.dataset.n_classes_max = 4
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    cfg.dataset.missing_rate = 0.1
    cfg.dataset.missing_mechanism = "mcar"
    cfg.intervention.mode = "hard_interventional"
    cfg.intervention.targets = [{"target_kind": "target", "value": 1.0}]  # type: ignore[list-item]
    cfg.validate_generation_constraints()

    batch = generate_batch(cfg, num_datasets=1, seed=123, device="cpu")
    write_packed_parquet_shards_stream(batch, tmp_path, shard_size=8, compression="zstd")
    record = json.loads(
        (tmp_path / "shard_00000" / DATASET_CATALOG_FILENAME).read_text().splitlines()[0]
    )

    actual_paths = sorted({_format_tokens(tokens) for tokens in _flatten_path_tokens(record)})

    assert actual_paths == load_golden_json("generated_metadata_record_paths.json")


def test_public_docs_do_not_reference_removed_target_head_contract() -> None:
    forbidden_patterns = {
        "p(y | X_complete)": r"p\(y \| X_complete\)",
        "p(y | x_complete)": r"p\(y \| x_complete\)",
        "y|X_complete": r"y\|X_complete",
        "conditional target head": r"conditional target head",
        "latent_complete_x_conditional": r"latent_complete_x_conditional",
        "posterior_predictive": r"posterior_predictive",
        "teacher_conditionals": r"teacher_conditionals",
        "teacher-conditional": r"teacher-conditional",
        "metadata.ndjson": r"metadata\.ndjson",
        "tab-foundry": r"tab-foundry",
    }

    doc_paths = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    offenders: list[str] = []
    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        for label, pattern in forbidden_patterns.items():
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {label}")

    assert not offenders, "Found removed contract terminology in docs:\n" + "\n".join(offenders)
