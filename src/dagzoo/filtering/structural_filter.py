"""Structural-only deferred filtering over lineage metadata."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dagzoo.filtering.structural_validity import (
    StructuralValidityConfig,
    evaluate_lineage_structural_validity,
)

STRUCTURAL_FILTER_MODE = "structural_v1"
_BACKEND = "lineage_structural_validity"
_MISSING_LINEAGE_ERROR = "Structural filtering requires metadata.lineage in shard metadata."


def apply_structural_filter(
    *,
    lineage_payload: Mapping[str, Any] | None,
    lineage_base_dir: Path | None,
    min_target_indegree: int = 1,
    min_target_relevant_feature_count: int = 2,
    min_target_relevant_feature_fraction: float = 0.05,
) -> tuple[bool, dict[str, Any]]:
    """Apply structural-only filtering from lineage metadata."""

    if not isinstance(lineage_payload, Mapping):
        raise ValueError(_MISSING_LINEAGE_ERROR)

    structural_result = evaluate_lineage_structural_validity(
        lineage_payload=lineage_payload,
        lineage_base_dir=lineage_base_dir,
        checks=StructuralValidityConfig(
            min_target_indegree=int(min_target_indegree),
            min_target_relevant_feature_count=int(min_target_relevant_feature_count),
            min_target_relevant_feature_fraction=float(min_target_relevant_feature_fraction),
        ),
    )

    details: dict[str, Any] = {
        "backend": _BACKEND,
        "filter_mode": STRUCTURAL_FILTER_MODE,
        "structural_filter_applied": True,
        "min_target_indegree": int(min_target_indegree),
        "min_target_relevant_feature_count": int(min_target_relevant_feature_count),
        "min_target_relevant_feature_fraction": float(min_target_relevant_feature_fraction),
        "target_indegree": int(structural_result.target_indegree),
        "feature_target_path_exists": bool(structural_result.feature_target_path_exists),
        "target_relevant_feature_count": int(structural_result.target_relevant_feature_count),
        "target_relevant_feature_fraction": float(
            structural_result.target_relevant_feature_fraction
        ),
    }
    if not structural_result.valid:
        details["reason"] = str(structural_result.reason)
        return False, details
    return True, details


__all__ = ["STRUCTURAL_FILTER_MODE", "apply_structural_filter"]
