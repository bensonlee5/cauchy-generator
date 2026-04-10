from __future__ import annotations

import torch

from dagzoo.core.layout_types import LayoutPlan
from dagzoo.filtering.structural_validity import (
    StructuralValidityConfig,
    evaluate_layout_structural_validity,
    evaluate_lineage_structural_validity,
)
from dagzoo.graph import dag_longest_path_to_target_nodes
from dagzoo.io.lineage_schema import LINEAGE_SCHEMA_VERSION_DENSE


def _layout(
    *,
    adjacency: list[list[int]],
    feature_to_node: list[int],
    target_to_node: int,
) -> LayoutPlan:
    graph_nodes = len(adjacency)
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.bool)
    graph_edges = int(adjacency_tensor.to(dtype=torch.int64).sum().item())
    density_denominator = graph_nodes * max(graph_nodes - 1, 1)
    return LayoutPlan(
        n_features=len(feature_to_node),
        n_cat=0,
        cat_idx=[],
        cardinalities=[],
        card_by_feature={},
        n_classes=2,
        feature_types=["num"] * len(feature_to_node),
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        graph_depth_nodes=graph_nodes,
        target_depth_nodes=dag_longest_path_to_target_nodes(
            adjacency_tensor,
            int(target_to_node),
        ),
        graph_edge_density=(
            float(graph_edges) / float(density_denominator) if density_denominator > 0 else 0.0
        ),
        adjacency=adjacency_tensor,
        feature_node_assignment=list(feature_to_node),
        target_to_node=int(target_to_node),
    )


def _lineage(
    *,
    adjacency: list[list[int]],
    feature_to_node: list[int],
    target_to_node: int,
) -> dict[str, object]:
    return {
        "schema_name": "dagzoo.dag_lineage",
        "schema_version": LINEAGE_SCHEMA_VERSION_DENSE,
        "graph": {
            "n_nodes": len(adjacency),
            "adjacency": adjacency,
        },
        "assignments": {
            "feature_to_node": list(feature_to_node),
            "target_to_node": int(target_to_node),
            "target_relevant_features": [],
            "target_relevant_feature_count": 0,
            "target_relevant_feature_fraction": 0.0,
        },
    }


def test_structural_validity_matches_layout_and_lineage_for_valid_graph() -> None:
    checks = StructuralValidityConfig(
        min_target_indegree=1,
        min_target_relevant_feature_count=2,
        min_target_relevant_feature_fraction=0.25,
    )
    adjacency = [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0],
    ]
    feature_to_node = [0, 1, 0, 1]
    target_to_node = 2

    layout_result = evaluate_layout_structural_validity(
        _layout(
            adjacency=adjacency,
            feature_to_node=feature_to_node,
            target_to_node=target_to_node,
        ),
        checks=checks,
    )
    lineage_result = evaluate_lineage_structural_validity(
        lineage_payload=_lineage(
            adjacency=adjacency,
            feature_to_node=feature_to_node,
            target_to_node=target_to_node,
        ),
        lineage_base_dir=None,
        checks=checks,
    )

    assert layout_result == lineage_result
    assert layout_result.valid is True


def test_structural_validity_matches_layout_and_lineage_for_no_path_graph() -> None:
    checks = StructuralValidityConfig(
        min_target_indegree=0,
        min_target_relevant_feature_count=0,
        min_target_relevant_feature_fraction=0.0,
    )
    adjacency = [
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ]
    feature_to_node = [1, 1, 1, 1]
    target_to_node = 2

    layout_result = evaluate_layout_structural_validity(
        _layout(
            adjacency=adjacency,
            feature_to_node=feature_to_node,
            target_to_node=target_to_node,
        ),
        checks=checks,
    )
    lineage_result = evaluate_lineage_structural_validity(
        lineage_payload=_lineage(
            adjacency=adjacency,
            feature_to_node=feature_to_node,
            target_to_node=target_to_node,
        ),
        lineage_base_dir=None,
        checks=checks,
    )

    assert layout_result == lineage_result
    assert layout_result.valid is False
    assert layout_result.reason == "no_feature_target_path"
