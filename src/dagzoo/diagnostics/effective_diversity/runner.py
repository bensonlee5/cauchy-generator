"""Rewritten baseline-vs-variant diversity audit runner."""

from __future__ import annotations

import datetime as dt
from typing import Any

from dagzoo.bench.corpus_probe import (
    CorpusProbeResult,
    build_corpus_probe_coverage_config,
    resolve_corpus_probe_counts,
    run_corpus_probe,
)
from dagzoo.config import GeneratorConfig
from dagzoo.recipes import parse_recipe_reference, serialize_config_reference

from .compare import (
    CORE_DIVERSITY_METRICS,
    build_comparison_record,
    summarize_comparison_status,
)


def _probe_entry(result: CorpusProbeResult) -> dict[str, Any]:
    """Convert one probe dataclass into the persisted report entry shape."""

    mechanism_family_summary = None
    if isinstance(result.coverage_summary, dict):
        payload = result.coverage_summary.get("mechanism_family_summary")
        if isinstance(payload, dict):
            mechanism_family_summary = payload
    return {
        "label": result.label,
        "config_path": result.config_path,
        "suite": result.suite,
        "num_datasets": int(result.num_datasets),
        "warmup_datasets": int(result.warmup_datasets),
        "requested_device": result.requested_device,
        "resolved_device": result.resolved_device,
        "resolved_config": result.resolved_config,
        "datasets_per_minute": float(result.datasets_per_minute),
        "filter_datasets_per_minute": result.filter_datasets_per_minute,
        "filter_accepted_datasets_per_minute": result.filter_accepted_datasets_per_minute,
        "filter_accepted_datasets_measured": int(result.filter_accepted_datasets_measured),
        "filter_rejected_datasets_measured": int(result.filter_rejected_datasets_measured),
        "filter_acceptance_rate_dataset_level": result.filter_acceptance_rate_dataset_level,
        "filter_rejection_rate_dataset_level": result.filter_rejection_rate_dataset_level,
        "coverage_summary": result.coverage_summary,
        "mechanism_family_summary": mechanism_family_summary,
        "filter_summary": result.filter_summary,
    }


def _build_variant_labels(variant_config_paths: list[str]) -> list[str]:
    """Derive stable labels from variant config paths while avoiding duplicates."""

    seen: dict[str, int] = {}
    labels: list[str] = []
    for idx, path in enumerate(variant_config_paths, start=1):
        recipe_name = parse_recipe_reference(path)
        if recipe_name is not None:
            stem = recipe_name
        else:
            stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0] if path else f"variant_{idx}"
        count = seen.get(stem, 0)
        seen[stem] = count + 1
        labels.append(stem if count == 0 else f"{stem}_{count + 1}")
    return labels


def run_effective_diversity_audit(
    *,
    baseline_config: GeneratorConfig,
    baseline_config_path: str,
    variant_configs: list[GeneratorConfig],
    variant_config_paths: list[str],
    variant_labels: list[str] | None = None,
    suite: str,
    num_datasets: int | None,
    warmup: int | None,
    device: str | None,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
) -> dict[str, Any]:
    """Run the rewritten diversity audit against baseline and variant configs."""

    if len(variant_configs) != len(variant_config_paths):
        raise ValueError("variant_configs and variant_config_paths must have matching lengths.")
    if variant_labels is not None and len(variant_labels) != len(variant_configs):
        raise ValueError("variant_labels and variant_configs must have matching lengths.")

    probe_seed = int(baseline_config.seed)
    probe_coverage_config = build_corpus_probe_coverage_config(baseline_config)
    probe_num_datasets, probe_warmup_datasets = resolve_corpus_probe_counts(
        baseline_config,
        suite=suite,
        num_datasets_override=num_datasets,
        warmup_override=warmup,
    )
    baseline_probe = run_corpus_probe(
        baseline_config,
        label="baseline",
        config_path=serialize_config_reference(baseline_config_path),
        suite=suite,
        num_datasets=probe_num_datasets,
        warmup=probe_warmup_datasets,
        device=device,
        probe_seed=probe_seed,
        coverage_config=probe_coverage_config,
    )
    baseline_entry = _probe_entry(baseline_probe)

    variants: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for label, config_path, variant_config in zip(
        variant_labels
        if variant_labels is not None
        else _build_variant_labels(variant_config_paths),
        variant_config_paths,
        variant_configs,
        strict=True,
    ):
        variant_probe = run_corpus_probe(
            variant_config,
            label=label,
            config_path=serialize_config_reference(config_path),
            suite=suite,
            num_datasets=probe_num_datasets,
            warmup=probe_warmup_datasets,
            device=device,
            probe_seed=probe_seed,
            coverage_config=probe_coverage_config,
        )
        variant_entry = _probe_entry(variant_probe)
        variants.append(variant_entry)
        comparisons.append(
            build_comparison_record(
                baseline_entry=baseline_entry,
                variant_entry=variant_entry,
                warn_threshold_pct=warn_threshold_pct,
                fail_threshold_pct=fail_threshold_pct,
            )
        )

    status_counts = {
        "pass": 0,
        "warn": 0,
        "fail": 0,
        "insufficient_metrics": 0,
    }
    for comparison in comparisons:
        status = str(comparison.get("diversity_status", "insufficient_metrics"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "schema_name": "dagzoo_diversity_audit_report",
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "baseline": baseline_entry,
        "variants": variants,
        "comparisons": comparisons,
        "summary": {
            "overall_status": summarize_comparison_status(comparisons),
            "warn_threshold_pct": float(warn_threshold_pct),
            "fail_threshold_pct": float(fail_threshold_pct),
            "num_variants": len(variants),
            "probe_num_datasets": int(probe_num_datasets),
            "probe_warmup_datasets": int(probe_warmup_datasets),
            "status_counts": status_counts,
            "core_metrics": list(CORE_DIVERSITY_METRICS),
        },
    }
