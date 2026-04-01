"""Public shard contract helpers and internal sidecar layout."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

DATASET_CATALOG_FILENAME = "dataset_catalog.ndjson"
INTERNAL_DIRNAME = "internal"
REPLAY_CATALOG_FILENAME = "replay_catalog.ndjson"
RUN_CONTEXT_FILENAME = "run_context.json"


def infer_task_from_metadata(metadata: Mapping[str, Any]) -> str:
    """Infer task label from metadata payloads."""

    config = metadata.get("config")
    if isinstance(config, Mapping):
        dataset = config.get("dataset")
        if isinstance(dataset, Mapping):
            task = dataset.get("task")
            if isinstance(task, str) and task.strip():
                normalized = task.strip().lower()
                if normalized in {"classification", "regression"}:
                    return normalized
    return "classification" if metadata.get("n_classes") is not None else "regression"


def build_dataset_catalog_record(
    *,
    dataset_index: int,
    n_train: int,
    n_test: int,
    n_features: int,
    feature_types: list[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one minimal public dataset-catalog record."""

    record: dict[str, Any] = {
        "dataset_index": int(dataset_index),
        "dataset_id": metadata.get("dataset_id"),
        "task": infer_task_from_metadata(metadata),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_features": int(n_features),
        "feature_types": list(feature_types),
        "n_classes": metadata.get("n_classes"),
    }
    split_groups = metadata.get("split_groups")
    if isinstance(split_groups, Mapping):
        group_ids = {}
        request_run = split_groups.get("request_run")
        if request_run is not None:
            group_ids["request_run"] = request_run
        cohort = split_groups.get("cohort")
        if cohort is not None:
            group_ids["cohort"] = cohort
        layout_plan = split_groups.get("layout_plan")
        if layout_plan is not None:
            group_ids["layout_plan"] = layout_plan
        if group_ids:
            record["group_ids"] = group_ids
    prior = metadata.get("prior")
    if isinstance(prior, Mapping):
        target_derivation = prior.get("target_derivation")
        if isinstance(target_derivation, str) and target_derivation.strip():
            record["target_derivation"] = target_derivation
    lineage = metadata.get("lineage")
    if isinstance(lineage, Mapping):
        assignments = lineage.get("assignments")
        if isinstance(assignments, Mapping):
            target_relevant_feature_count = assignments.get("target_relevant_feature_count")
            target_relevant_feature_fraction = assignments.get("target_relevant_feature_fraction")
            if (
                not isinstance(target_relevant_feature_count, bool)
                and isinstance(target_relevant_feature_count, int)
                and not isinstance(target_relevant_feature_fraction, bool)
                and isinstance(target_relevant_feature_fraction, (int, float))
            ):
                record["target_relevance"] = {
                    "feature_count": int(target_relevant_feature_count),
                    "feature_fraction": float(target_relevant_feature_fraction),
                }
    return record


def resolve_internal_root(
    public_root: str | Path,
    *,
    explicit_internal_root: str | Path | None = None,
) -> Path:
    """Resolve the internal sidecar root for one public shard root."""

    if explicit_internal_root is not None:
        return Path(explicit_internal_root)

    resolved_public_root = Path(public_root)
    direct_candidate = resolved_public_root / INTERNAL_DIRNAME
    if direct_candidate.exists():
        return direct_candidate
    sibling_candidate = resolved_public_root.parent / INTERNAL_DIRNAME
    if sibling_candidate.exists():
        return sibling_candidate
    return direct_candidate


def internal_shard_dir(
    *,
    internal_root: str | Path,
    shard_name: str,
) -> Path:
    """Return the internal sidecar directory for one public shard."""

    return Path(internal_root) / str(shard_name)


def iter_ndjson_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON-object NDJSON records from one file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid NDJSON record in {path}:{line_number}: expected object.")
            yield payload
