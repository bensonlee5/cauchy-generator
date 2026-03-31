"""Metadata builders for dataset lineage and shift diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch

from dagzoo.core.layout_types import LayoutPlan, MechanismFamily
from dagzoo.core.shift import ShiftRuntimeParams, mechanism_nonlinear_mass
from dagzoo.io.lineage_schema import (
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION,
    validate_lineage_payload,
)


def _build_lineage_metadata(
    layout: LayoutPlan,
    *,
    feature_index_map: list[int],
) -> dict[str, Any]:
    """Build a validated DAG lineage payload from sampled layout internals."""

    n_nodes = int(layout.graph_nodes)
    raw_adjacency = layout.adjacency
    if isinstance(raw_adjacency, torch.Tensor):
        adjacency_rows = raw_adjacency.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        adjacency_rows = torch.as_tensor(raw_adjacency, dtype=torch.int64, device="cpu").tolist()
    adjacency = [[int(value) for value in row] for row in adjacency_rows]

    raw_feature_to_node = [int(node_index) for node_index in list(layout.feature_node_assignment)]
    feature_to_node = [raw_feature_to_node[int(src_col)] for src_col in feature_index_map]
    source_to_emitted = {
        int(src_col): int(emitted_index) for emitted_index, src_col in enumerate(feature_index_map)
    }
    target_parent_features = sorted(
        int(source_to_emitted[src_col])
        for src_col in list(layout.target_parent_features)
        if int(src_col) in source_to_emitted
    )
    target_parent_count = int(len(target_parent_features))
    emitted_feature_count = int(len(feature_index_map))
    target_parent_fraction = (
        float(target_parent_count) / float(emitted_feature_count)
        if emitted_feature_count > 0
        else 0.0
    )

    payload = {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "graph": {
            "n_nodes": n_nodes,
            "adjacency": adjacency,
        },
        "assignments": {
            "feature_to_node": feature_to_node,
            "target_mode": "latent_complete_x_conditional",
            "target_parent_features": target_parent_features,
            "target_parent_count": target_parent_count,
            "target_parent_fraction": float(target_parent_fraction),
            "target_parent_prior": str(layout.target_parent_prior),
            "target_parent_regime": str(layout.target_parent_regime),
            "target_parent_sqrt_threshold": int(layout.target_parent_sqrt_threshold),
        },
    }
    validate_lineage_payload(payload)
    return payload


def _build_shift_metadata(
    *,
    shift_params: ShiftRuntimeParams,
    function_family_mix: dict[MechanismFamily, float] | None = None,
) -> dict[str, Any]:
    """Build resolved shift metadata for one emitted dataset bundle."""

    nonlinear_mass = mechanism_nonlinear_mass(
        mechanism_logit_tilt=float(shift_params.mechanism_logit_tilt),
        family_weights=function_family_mix,
    )
    edge_odds_multiplier = float(math.exp(shift_params.edge_logit_bias_shift))
    noise_variance_multiplier = float(shift_params.variance_sigma_multiplier**2)
    return {
        "enabled": bool(shift_params.enabled),
        "mode": str(shift_params.mode),
        "graph_scale": float(shift_params.graph_scale),
        "mechanism_scale": float(shift_params.mechanism_scale),
        "variance_scale": float(shift_params.variance_scale),
        "edge_logit_bias_shift": float(shift_params.edge_logit_bias_shift),
        "mechanism_logit_tilt": float(shift_params.mechanism_logit_tilt),
        "variance_sigma_multiplier": float(shift_params.variance_sigma_multiplier),
        "edge_odds_multiplier": float(edge_odds_multiplier),
        "noise_variance_multiplier": float(noise_variance_multiplier),
        "mechanism_nonlinear_mass": float(nonlinear_mass),
    }
