"""Typed payloads for dataset-level diagnostics metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DatasetMetrics:
    """Stable dataset-level metrics payload used by diagnostics extractors."""

    task: str
    n_rows: int
    n_features: int
    n_classes: int | None
    n_categorical_features: int
    categorical_ratio: float
    graph_edge_density: float | None
    graph_indegree_std: float | None
    graph_outdegree_std: float | None
    graph_depth_ratio: float | None
    graph_reachability_ratio: float | None
    graph_ancestor_overlap_mean: float | None
    graph_target_ancestor_fraction: float | None
    mechanism_family_cooccurrence_ratio: float | None
    shift_enabled: float
    shift_graph_scale: float
    shift_mechanism_scale: float
    shift_variance_scale: float
    shift_edge_odds_multiplier: float
    shift_mechanism_nonlinear_mass: float
    shift_noise_variance_multiplier: float

    linearity_proxy: float | None
    nonlinearity_proxy: float | None
    wins_ratio_proxy: float | None

    pearson_abs_mean: float | None
    pearson_abs_max: float | None
    spearman_abs_mean: float | None
    spearman_abs_max: float | None

    class_entropy: float | None
    majority_minority_ratio: float | None
    snr_proxy_db: float | None

    cat_cardinality_min: int | None
    cat_cardinality_mean: float | None
    cat_cardinality_max: int | None
