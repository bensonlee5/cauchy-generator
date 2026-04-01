"""Shared structural-validity checks for obvious target degeneracy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dagzoo.core.layout_types import LayoutPlan
from dagzoo.io.lineage_artifact import (
    resolve_lineage_path,
    sha256_hex,
    unpack_upper_triangle_adjacency,
)
from dagzoo.io.lineage_schema import validate_lineage_payload

TARGET_ROOT_REASON = "target_root"
NO_FEATURE_TARGET_PATH_REASON = "no_feature_target_path"
INSUFFICIENT_TARGET_RELEVANT_FEATURE_COUNT_REASON = "insufficient_target_relevant_feature_count"
INSUFFICIENT_TARGET_RELEVANT_FEATURE_FRACTION_REASON = (
    "insufficient_target_relevant_feature_fraction"
)

_LINEAGE_BLOB_PATH_ERROR = (
    "metadata.lineage.graph.adjacency_ref.blob_path must be a relative path "
    "that resolves inside the shard lineage directory."
)
_LINEAGE_BLOB_SHA256_ERROR = (
    "metadata.lineage.graph.adjacency_ref.sha256 must match the resolved adjacency blob slice."
)


@dataclass(slots=True, frozen=True)
class StructuralValidityConfig:
    """Thresholds for graph-level target validity checks."""

    min_target_indegree: int
    min_target_relevant_feature_count: int
    min_target_relevant_feature_fraction: float


@dataclass(slots=True, frozen=True)
class StructuralValidityResult:
    """Resolved structural-validity verdict plus supporting summary metrics."""

    valid: bool
    reason: str | None
    target_indegree: int
    feature_target_path_exists: bool
    target_relevant_feature_count: int
    target_relevant_feature_fraction: float


def _resolve_safe_lineage_blob_path(*, lineage_base_dir: Path, blob_path_hint: str) -> Path:
    hinted = Path(blob_path_hint)
    if hinted.is_absolute():
        raise ValueError(_LINEAGE_BLOB_PATH_ERROR)

    lineage_root = (lineage_base_dir / "lineage").resolve()
    resolved = resolve_lineage_path(lineage_base_dir, blob_path_hint).resolve()
    try:
        resolved.relative_to(lineage_root)
    except ValueError as exc:
        raise ValueError(_LINEAGE_BLOB_PATH_ERROR) from exc
    return resolved


def _resolve_lineage_adjacency(
    *,
    lineage_payload: Mapping[str, Any],
    lineage_base_dir: Path | None,
) -> np.ndarray:
    validate_lineage_payload(dict(lineage_payload))
    graph = lineage_payload["graph"]
    assert isinstance(graph, Mapping)
    n_nodes = int(graph["n_nodes"])

    dense_adjacency = graph.get("adjacency")
    if isinstance(dense_adjacency, list):
        return np.asarray(dense_adjacency, dtype=np.uint8)

    adjacency_ref = graph.get("adjacency_ref")
    if not isinstance(adjacency_ref, Mapping):
        raise ValueError("metadata.lineage.graph must contain adjacency or adjacency_ref.")
    if lineage_base_dir is None:
        raise ValueError(
            "Compact lineage payload requires lineage_base_dir to resolve adjacency artifacts."
        )
    blob_path = _resolve_safe_lineage_blob_path(
        lineage_base_dir=lineage_base_dir,
        blob_path_hint=str(adjacency_ref["blob_path"]),
    )
    bit_offset = int(adjacency_ref["bit_offset"])
    bit_length = int(adjacency_ref["bit_length"])
    byte_offset = bit_offset // 8
    byte_length = (bit_length + 7) // 8
    with blob_path.open("rb") as handle:
        handle.seek(byte_offset)
        payload = handle.read(byte_length)
    if len(payload) == byte_length and sha256_hex(payload) != str(adjacency_ref["sha256"]):
        raise ValueError(_LINEAGE_BLOB_SHA256_ERROR)
    return unpack_upper_triangle_adjacency(payload, n_nodes=n_nodes, bit_length=bit_length)


def _ancestor_nodes_for_target(*, adjacency: np.ndarray, target_to_node: int) -> set[int]:
    ancestors = {int(target_to_node)}
    frontier = [int(target_to_node)]
    while frontier:
        node_index = int(frontier.pop())
        for parent_index in range(int(adjacency.shape[0])):
            if int(adjacency[parent_index, node_index]) == 0 or parent_index in ancestors:
                continue
            ancestors.add(parent_index)
            frontier.append(parent_index)
    return ancestors


def _evaluate_from_graph(
    *,
    feature_to_node: list[int],
    target_to_node: int,
    adjacency: np.ndarray,
    checks: StructuralValidityConfig,
) -> StructuralValidityResult:
    target_node = int(target_to_node)
    target_indegree = int(np.count_nonzero(adjacency[:, target_node]))
    relevant_nodes = _ancestor_nodes_for_target(adjacency=adjacency, target_to_node=target_node)
    target_relevant_feature_count = int(
        sum(1 for node_index in feature_to_node if int(node_index) in relevant_nodes)
    )
    total_features = int(len(feature_to_node))
    target_relevant_feature_fraction = (
        float(target_relevant_feature_count) / float(total_features) if total_features > 0 else 0.0
    )
    feature_target_path_exists = bool(target_relevant_feature_count > 0)

    reason: str | None = None
    if target_indegree < int(checks.min_target_indegree):
        reason = TARGET_ROOT_REASON
    elif not feature_target_path_exists:
        reason = NO_FEATURE_TARGET_PATH_REASON
    elif target_relevant_feature_count < int(checks.min_target_relevant_feature_count):
        reason = INSUFFICIENT_TARGET_RELEVANT_FEATURE_COUNT_REASON
    elif (
        target_relevant_feature_fraction
        < float(checks.min_target_relevant_feature_fraction) - 1e-12
    ):
        reason = INSUFFICIENT_TARGET_RELEVANT_FEATURE_FRACTION_REASON

    return StructuralValidityResult(
        valid=reason is None,
        reason=reason,
        target_indegree=int(target_indegree),
        feature_target_path_exists=bool(feature_target_path_exists),
        target_relevant_feature_count=int(target_relevant_feature_count),
        target_relevant_feature_fraction=float(target_relevant_feature_fraction),
    )


def evaluate_layout_structural_validity(
    layout: LayoutPlan,
    *,
    checks: StructuralValidityConfig,
) -> StructuralValidityResult:
    """Evaluate structural validity for one sampled in-memory layout."""

    adjacency = torch.as_tensor(layout.adjacency, dtype=torch.uint8, device="cpu").numpy()
    feature_to_node = [int(node_index) for node_index in layout.feature_node_assignment]
    return _evaluate_from_graph(
        feature_to_node=feature_to_node,
        target_to_node=int(layout.target_to_node),
        adjacency=adjacency,
        checks=checks,
    )


def evaluate_lineage_structural_validity(
    *,
    lineage_payload: Mapping[str, Any],
    lineage_base_dir: Path | None,
    checks: StructuralValidityConfig,
) -> StructuralValidityResult:
    """Evaluate structural validity for one persisted lineage payload."""

    validate_lineage_payload(dict(lineage_payload))
    assignments = lineage_payload["assignments"]
    assert isinstance(assignments, Mapping)
    feature_to_node = [int(value) for value in list(assignments["feature_to_node"])]
    target_to_node = int(assignments["target_to_node"])
    adjacency = _resolve_lineage_adjacency(
        lineage_payload=lineage_payload,
        lineage_base_dir=lineage_base_dir,
    )
    return _evaluate_from_graph(
        feature_to_node=feature_to_node,
        target_to_node=target_to_node,
        adjacency=adjacency,
        checks=checks,
    )


__all__ = [
    "INSUFFICIENT_TARGET_RELEVANT_FEATURE_COUNT_REASON",
    "INSUFFICIENT_TARGET_RELEVANT_FEATURE_FRACTION_REASON",
    "NO_FEATURE_TARGET_PATH_REASON",
    "TARGET_ROOT_REASON",
    "StructuralValidityConfig",
    "StructuralValidityResult",
    "evaluate_layout_structural_validity",
    "evaluate_lineage_structural_validity",
]
