"""Dataset-level meta-feature metrics extractors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import fields
from typing import Any

import torch

from dagzoo.core.metrics_torch import extract_torch_metrics
from dagzoo.types import DatasetBundle

from .types import DatasetMetrics


def _to_cpu_value(value: Any) -> Any:
    """Move tensors to CPU for diagnostics extraction, leaving non-tensors unchanged."""

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    return value


def _bundle_on_cpu(bundle: DatasetBundle) -> DatasetBundle:
    """Return a CPU-normalized copy of a bundle for device-agnostic diagnostics."""

    return DatasetBundle(
        X_train=_to_cpu_value(bundle.X_train),
        y_train=_to_cpu_value(bundle.y_train),
        X_test=_to_cpu_value(bundle.X_test),
        y_test=_to_cpu_value(bundle.y_test),
        feature_types=list(bundle.feature_types),
        metadata=dict(bundle.metadata),
    )


def extract_metrics_batch(
    bundles: Sequence[DatasetBundle], *, include_spearman: bool = False
) -> list[DatasetMetrics]:
    """Extract dataset-level metrics for a sequence of bundles in order."""

    return [
        extract_dataset_metrics(bundle, include_spearman=include_spearman) for bundle in bundles
    ]


def extract_dataset_metrics(
    bundle: DatasetBundle, *, include_spearman: bool = False
) -> DatasetMetrics:
    """Extract deterministic dataset-level metrics from one generated bundle."""
    cpu_bundle = _bundle_on_cpu(bundle)

    # Derive all possible numeric metric names from the DatasetMetrics dataclass.
    metric_names = {f.name for f in fields(DatasetMetrics) if f.name != "task"}
    if not include_spearman:
        metric_names -= {"spearman_abs_mean", "spearman_abs_max"}

    raw = extract_torch_metrics(
        cpu_bundle, target_metric_names=metric_names, include_spearman=include_spearman
    )

    # Convert numeric results to expected types (int where appropriate)
    def _int_or_none(v: Any) -> int | None:
        return int(v) if v is not None else None

    def _float_or_none(v: Any) -> float | None:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        as_float = float(v)
        return as_float if math.isfinite(as_float) else None

    def _float_or_default(v: Any, default: float) -> float:
        value = _float_or_none(v)
        return default if value is None else value

    return DatasetMetrics(
        task=str(raw["task"]),
        n_rows=int(raw["n_rows"]),
        n_features=int(raw["n_features"]),
        n_classes=_int_or_none(raw.get("n_classes")),
        n_categorical_features=int(raw["n_categorical_features"]),
        categorical_ratio=float(raw["categorical_ratio"]),
        graph_edge_density=_float_or_none(raw.get("graph_edge_density")),
        graph_indegree_std=_float_or_none(raw.get("graph_indegree_std")),
        graph_outdegree_std=_float_or_none(raw.get("graph_outdegree_std")),
        graph_depth_ratio=_float_or_none(raw.get("graph_depth_ratio")),
        graph_target_depth_ratio=_float_or_none(raw.get("graph_target_depth_ratio")),
        graph_reachability_ratio=_float_or_none(raw.get("graph_reachability_ratio")),
        graph_ancestor_overlap_mean=_float_or_none(raw.get("graph_ancestor_overlap_mean")),
        graph_target_ancestor_fraction=_float_or_none(raw.get("graph_target_ancestor_fraction")),
        mechanism_family_cooccurrence_ratio=_float_or_none(
            raw.get("mechanism_family_cooccurrence_ratio")
        ),
        shift_enabled=_float_or_default(raw.get("shift_enabled"), 0.0),
        shift_graph_scale=_float_or_default(raw.get("shift_graph_scale"), 0.0),
        shift_mechanism_scale=_float_or_default(raw.get("shift_mechanism_scale"), 0.0),
        shift_variance_scale=_float_or_default(raw.get("shift_variance_scale"), 0.0),
        shift_edge_odds_multiplier=_float_or_default(raw.get("shift_edge_odds_multiplier"), 1.0),
        shift_mechanism_nonlinear_mass=_float_or_default(
            raw.get("shift_mechanism_nonlinear_mass"),
            0.0,
        ),
        shift_noise_variance_multiplier=_float_or_default(
            raw.get("shift_noise_variance_multiplier"),
            1.0,
        ),
        linearity_proxy=raw.get("linearity_proxy"),
        nonlinearity_proxy=raw.get("nonlinearity_proxy"),
        wins_ratio_proxy=raw.get("wins_ratio_proxy"),
        pearson_abs_mean=raw.get("pearson_abs_mean"),
        pearson_abs_max=raw.get("pearson_abs_max"),
        spearman_abs_mean=raw.get("spearman_abs_mean"),
        spearman_abs_max=raw.get("spearman_abs_max"),
        class_entropy=raw.get("class_entropy"),
        majority_minority_ratio=raw.get("majority_minority_ratio"),
        snr_proxy_db=raw.get("snr_proxy_db"),
        cat_cardinality_min=_int_or_none(raw.get("cat_cardinality_min")),
        cat_cardinality_mean=raw.get("cat_cardinality_mean"),
        cat_cardinality_max=_int_or_none(raw.get("cat_cardinality_max")),
    )
