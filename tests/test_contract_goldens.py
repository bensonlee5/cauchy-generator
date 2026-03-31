from __future__ import annotations

import json
from pathlib import Path

from golden_support import (
    assert_normalized_json_equal,
    assert_normalized_text_equal,
    normalize_benchmark_summary,
    normalize_benchmark_summary_markdown,
    normalize_coverage_summary,
    normalize_handoff_manifest,
)

from dagzoo.bench.report import write_suite_json, write_suite_markdown
from dagzoo.core.generate_handoff import write_generate_handoff_manifest
from dagzoo.diagnostics.coverage import (
    CoverageAggregationConfig,
    CoverageAggregator,
    write_coverage_summary_json,
)
from dagzoo.diagnostics.types import DatasetMetrics


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
    factorization = "independent_p_x_complete_and_p_y_given_x_complete"
    metric_definition = "label-target log loss per test cell"
    (shard_dir / "metadata.ndjson").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset_index": 0,
                        "metadata": {
                            "dataset_id": "2" * 32,
                            "config": {
                                "dataset": {
                                    "target_parent_prior": "near_max_mixture",
                                    "target_parent_near_max_band_min_fraction": 0.75,
                                    "target_parent_below_sqrt_prob": 0.05,
                                    "target_parent_midrange_prob": 0.20,
                                }
                            },
                            "lineage": {
                                "assignments": {
                                    "target_parent_count": 6,
                                    "target_parent_fraction": 0.75,
                                    "target_parent_regime": "near_max",
                                }
                            },
                            "posterior_predictive": {
                                "factorization": factorization,
                                "metric_definition": metric_definition,
                                "teacher_conditional_export_enabled": False,
                                "teacher_conditionals_available": False,
                            },
                            "prior": {
                                "factorization": factorization,
                            },
                            "split_groups": {"request_run": "1" * 32},
                        },
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "dataset_index": 1,
                        "metadata": {
                            "dataset_id": "3" * 32,
                            "config": {
                                "dataset": {
                                    "target_parent_prior": "near_max_mixture",
                                    "target_parent_near_max_band_min_fraction": 0.75,
                                    "target_parent_below_sqrt_prob": 0.05,
                                    "target_parent_midrange_prob": 0.20,
                                }
                            },
                            "lineage": {
                                "assignments": {
                                    "target_parent_count": 7,
                                    "target_parent_fraction": 0.8,
                                    "target_parent_regime": "midrange",
                                }
                            },
                            "posterior_predictive": {
                                "factorization": factorization,
                                "metric_definition": metric_definition,
                                "teacher_conditional_export_enabled": False,
                                "teacher_conditionals_available": False,
                            },
                            "prior": {
                                "factorization": factorization,
                            },
                            "split_groups": {"request_run": "1" * 32},
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
