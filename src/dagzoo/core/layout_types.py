"""Internal typed layout contracts for generation modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

FeatureType = Literal["num", "cat"]
ConverterKind = Literal["cat", "num", "target_cls", "target_reg"]
MechanismFamily = Literal[
    "nn",
    "tree",
    "discretization",
    "gp",
    "linear",
    "quadratic",
    "em",
    "product",
    "piecewise",
]
AggregationKind = Literal["sum", "product", "max", "logsumexp"]


@dataclass(slots=True)
class LayoutPlan:
    """Typed sampled layout payload shared across generation modules."""

    n_features: int
    n_cat: int
    cat_idx: list[int]
    cardinalities: list[int]
    card_by_feature: dict[int, int]
    n_classes: int
    feature_types: list[FeatureType]
    graph_nodes: int
    graph_edges: int
    graph_depth_nodes: int
    target_depth_nodes: int
    graph_edge_density: float
    adjacency: torch.Tensor
    feature_node_assignment: list[int]
    target_to_node: int


__all__ = [
    "AggregationKind",
    "ConverterKind",
    "FeatureType",
    "LayoutPlan",
    "MechanismFamily",
]
