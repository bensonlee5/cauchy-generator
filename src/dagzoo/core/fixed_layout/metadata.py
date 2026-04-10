"""Fixed-layout plan models and metadata helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import torch

from dagzoo.core.layout_types import LayoutPlan
from dagzoo.types import DatasetBundle

from .interventions import FixedLayoutResolvedInterventionPlan
from .plan_types import (
    FixedLayoutExecutionPlan,
    execution_plan_family_counts,
    execution_plan_variant_counts,
    fixed_layout_signature_payloads,
)

_FIXED_LAYOUT_METADATA_SCHEMA_VERSION = 12
_PARITY_SURFACE_SCHEMA_NAME = "dagzoo_fixed_layout_parity_surface"
_PARITY_SURFACE_SCHEMA_VERSION = 1


@dataclass(slots=True)
class _FixedLayoutPlan:
    """Internal pre-sampled layout bundle for canonical fixed-layout generation."""

    layout: LayoutPlan
    requested_device: str
    resolved_device: str
    plan_seed: int
    n_train: int
    n_test: int
    layout_signature: str
    candidate_attempt: int = 0
    execution_plan: FixedLayoutExecutionPlan = field(default_factory=FixedLayoutExecutionPlan)
    plan_signature: str | None = None
    layout_root_path: list[object] | None = None
    execution_plan_root_path: list[object] | None = None
    steering_layout_root_path: list[object] | None = None
    steering_execution_plan_root_path: list[object] | None = None
    stress_profile_name: str | None = None
    intervention_plan: FixedLayoutResolvedInterventionPlan | None = None
    prepared_execution_context: Any | None = field(default=None, repr=False)


def _layout_to_dict(layout: LayoutPlan) -> dict[str, Any]:
    adjacency = layout.adjacency
    if isinstance(adjacency, torch.Tensor):
        adjacency_payload = adjacency.to(device="cpu", dtype=torch.int64).tolist()
    else:
        adjacency_payload = torch.as_tensor(adjacency, dtype=torch.int64, device="cpu").tolist()
    return {
        "n_features": int(layout.n_features),
        "n_cat": int(layout.n_cat),
        "cat_idx": [int(value) for value in layout.cat_idx],
        "cardinalities": [int(value) for value in layout.cardinalities],
        "card_by_feature": {
            str(int(key)): int(value) for key, value in layout.card_by_feature.items()
        },
        "n_classes": int(layout.n_classes),
        "feature_types": [str(value) for value in layout.feature_types],
        "graph_nodes": int(layout.graph_nodes),
        "graph_edges": int(layout.graph_edges),
        "graph_depth_nodes": int(layout.graph_depth_nodes),
        "target_depth_nodes": int(layout.target_depth_nodes),
        "graph_edge_density": float(layout.graph_edge_density),
        "adjacency": adjacency_payload,
        "feature_node_assignment": [int(value) for value in layout.feature_node_assignment],
        "target_to_node": int(layout.target_to_node),
    }


def _layout_signature(layout: LayoutPlan) -> str:
    encoded = json.dumps(
        _layout_to_dict(layout),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=16).hexdigest()


def _annotate_fixed_layout_metadata(
    bundle: DatasetBundle,
    *,
    plan: _FixedLayoutPlan,
    layout_mode: str = "fixed",
) -> None:
    bundle.metadata["layout_mode"] = str(layout_mode)
    bundle.metadata["layout_plan_seed"] = int(plan.plan_seed)
    bundle.metadata["layout_signature"] = str(plan.layout_signature)
    bundle.metadata["layout_plan_schema_version"] = int(_FIXED_LAYOUT_METADATA_SCHEMA_VERSION)
    bundle.metadata["layout_execution_contract"] = str(plan.execution_plan.execution_contract)
    if plan.stress_profile_name is not None:
        bundle.metadata["layout_stress_profile_name"] = str(plan.stress_profile_name)
    else:
        bundle.metadata.pop("layout_stress_profile_name", None)
    keyed_replay = bundle.metadata.get("keyed_replay")
    if not isinstance(keyed_replay, dict):
        keyed_replay = {}
    keyed_replay["layout_root_path"] = (
        list(plan.layout_root_path)
        if plan.layout_root_path is not None
        else ["plan_candidate", int(plan.candidate_attempt), "layout"]
    )
    keyed_replay["execution_plan_root_path"] = (
        list(plan.execution_plan_root_path)
        if plan.execution_plan_root_path is not None
        else ["plan_candidate", int(plan.candidate_attempt), "execution_plan"]
    )
    if plan.steering_layout_root_path is not None:
        keyed_replay["steering_layout_root_path"] = list(plan.steering_layout_root_path)
    else:
        keyed_replay.pop("steering_layout_root_path", None)
    if plan.steering_execution_plan_root_path is not None:
        keyed_replay["steering_execution_plan_root_path"] = list(
            plan.steering_execution_plan_root_path
        )
    else:
        keyed_replay.pop("steering_execution_plan_root_path", None)
    bundle.metadata["keyed_replay"] = keyed_replay
    if plan.plan_signature is not None:
        bundle.metadata["layout_plan_signature"] = str(plan.plan_signature)
    family_counts = execution_plan_family_counts(plan.execution_plan)
    variant_counts = execution_plan_variant_counts(plan.execution_plan)
    total_function_plans = int(sum(family_counts.values()))
    bundle.metadata["mechanism_families"] = {
        "sampled_family_counts": dict(family_counts),
        "families_present": [family for family in sorted(family_counts)],
        "sampled_variant_counts": dict(variant_counts),
        "variants_present": [label for label in sorted(variant_counts)],
        "total_function_plans": int(total_function_plans),
    }
    bundle.metadata["parity_surface"] = _build_parity_surface_metadata(plan)


def _increment_count(counter: dict[str, int], label: object, count: int = 1) -> None:
    if isinstance(label, str) and label:
        counter[str(label)] = int(counter.get(str(label), 0)) + int(count)


def _summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "total": 0.0}
    total = float(sum(float(value) for value in values))
    return {
        "count": int(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(total / len(values)),
        "total": float(total),
    }


def _parent_arity_label(parent_count: int) -> str:
    if int(parent_count) >= 3:
        return "3plus"
    return str(int(parent_count))


def _source_shape_policy_label(
    *, source_kind: str, combine_kind: str | None, parent_count: int
) -> str:
    if source_kind == "random_points":
        return "random_points"
    assert combine_kind is not None
    return f"{combine_kind}_parent{_parent_arity_label(parent_count)}"


def _walk_matrix_payload(
    payload: object,
    *,
    matrix_kind_counts: dict[str, int],
    activation_base_kind_counts: dict[str, int],
    kernel_gamma_values: list[float],
    kernel_signed_counts: dict[str, int],
) -> None:
    if not isinstance(payload, dict):
        return
    kind = payload.get("kind")
    if not isinstance(kind, str):
        return
    _increment_count(matrix_kind_counts, kind)
    if kind == "activation":
        _increment_count(activation_base_kind_counts, payload.get("base_kind"))
        return
    if kind == "kernel":
        gamma = payload.get("gamma")
        if isinstance(gamma, (int, float)) and not isinstance(gamma, bool):
            kernel_gamma_values.append(float(gamma))
        signed = payload.get("signed")
        if isinstance(signed, bool):
            _increment_count(kernel_signed_counts, "signed" if signed else "unsigned")


def _walk_function_payload(
    payload: object,
    *,
    matrix_kind_counts: dict[str, int],
    activation_base_kind_counts: dict[str, int],
    gp_variant_counts: dict[str, int],
    kernel_gamma_values: list[float],
    kernel_signed_counts: dict[str, int],
) -> None:
    if not isinstance(payload, dict):
        return
    family = payload.get("family")
    if not isinstance(family, str):
        return
    if family in {"linear", "quadratic"}:
        _walk_matrix_payload(
            payload.get("matrix"),
            matrix_kind_counts=matrix_kind_counts,
            activation_base_kind_counts=activation_base_kind_counts,
            kernel_gamma_values=kernel_gamma_values,
            kernel_signed_counts=kernel_signed_counts,
        )
        return
    if family == "nn":
        layer_matrices = payload.get("layer_matrices")
        if isinstance(layer_matrices, list):
            for matrix_payload in layer_matrices:
                _walk_matrix_payload(
                    matrix_payload,
                    matrix_kind_counts=matrix_kind_counts,
                    activation_base_kind_counts=activation_base_kind_counts,
                    kernel_gamma_values=kernel_gamma_values,
                    kernel_signed_counts=kernel_signed_counts,
                )
        return
    if family in {"discretization", "em"}:
        _walk_matrix_payload(
            payload.get("linear_matrix"),
            matrix_kind_counts=matrix_kind_counts,
            activation_base_kind_counts=activation_base_kind_counts,
            kernel_gamma_values=kernel_gamma_values,
            kernel_signed_counts=kernel_signed_counts,
        )
        return
    if family == "gp":
        variant = payload.get("variant")
        if isinstance(variant, str):
            _increment_count(gp_variant_counts, f"gp.{variant}")
        return
    if family == "piecewise":
        _walk_matrix_payload(
            payload.get("gate_matrix"),
            matrix_kind_counts=matrix_kind_counts,
            activation_base_kind_counts=activation_base_kind_counts,
            kernel_gamma_values=kernel_gamma_values,
            kernel_signed_counts=kernel_signed_counts,
        )
        _walk_function_payload(
            payload.get("lhs"),
            matrix_kind_counts=matrix_kind_counts,
            activation_base_kind_counts=activation_base_kind_counts,
            gp_variant_counts=gp_variant_counts,
            kernel_gamma_values=kernel_gamma_values,
            kernel_signed_counts=kernel_signed_counts,
        )
        _walk_function_payload(
            payload.get("rhs"),
            matrix_kind_counts=matrix_kind_counts,
            activation_base_kind_counts=activation_base_kind_counts,
            gp_variant_counts=gp_variant_counts,
            kernel_gamma_values=kernel_gamma_values,
            kernel_signed_counts=kernel_signed_counts,
        )
        return
    if family == "product":
        _walk_function_payload(
            payload.get("lhs"),
            matrix_kind_counts=matrix_kind_counts,
            activation_base_kind_counts=activation_base_kind_counts,
            gp_variant_counts=gp_variant_counts,
            kernel_gamma_values=kernel_gamma_values,
            kernel_signed_counts=kernel_signed_counts,
        )
        _walk_function_payload(
            payload.get("rhs"),
            matrix_kind_counts=matrix_kind_counts,
            activation_base_kind_counts=activation_base_kind_counts,
            gp_variant_counts=gp_variant_counts,
            kernel_gamma_values=kernel_gamma_values,
            kernel_signed_counts=kernel_signed_counts,
        )


def _build_parity_surface_metadata(plan: _FixedLayoutPlan) -> dict[str, Any]:
    payloads = fixed_layout_signature_payloads(plan.execution_plan)
    converter_method_counts: dict[str, int] = {}
    converter_variant_counts: dict[str, int] = {}
    converter_method_variant_counts: dict[str, int] = {}
    gp_variant_counts: dict[str, int] = {}
    kernel_signed_counts: dict[str, int] = {}
    matrix_kind_counts: dict[str, int] = {}
    activation_base_kind_counts: dict[str, int] = {}
    root_base_kind_counts: dict[str, int] = {}
    source_kind_counts: dict[str, int] = {}
    combine_kind_counts: dict[str, int] = {}
    aggregation_kind_counts: dict[str, int] = {}
    parent_arity_counts: dict[str, int] = {}
    source_shape_policy_counts: dict[str, int] = {}
    kernel_gamma_values: list[float] = []
    categorical_cardinality_values = [
        float(value) for value in plan.layout.cardinalities if int(value) > 0
    ]

    for node_payload in payloads:
        if not isinstance(node_payload, dict):
            continue
        source_kind = node_payload.get("source_kind")
        if source_kind == "random_points":
            _increment_count(source_kind_counts, "random_points")
            _increment_count(root_base_kind_counts, node_payload.get("base_kind"))
            _increment_count(
                source_shape_policy_counts,
                _source_shape_policy_label(
                    source_kind="random_points",
                    combine_kind=None,
                    parent_count=0,
                ),
            )
            _walk_function_payload(
                node_payload.get("function"),
                matrix_kind_counts=matrix_kind_counts,
                activation_base_kind_counts=activation_base_kind_counts,
                gp_variant_counts=gp_variant_counts,
                kernel_gamma_values=kernel_gamma_values,
                kernel_signed_counts=kernel_signed_counts,
            )
        elif source_kind == "multi":
            parent_indices = node_payload.get("parent_indices")
            parent_count = len(parent_indices) if isinstance(parent_indices, list) else 0
            combine_kind = node_payload.get("combine_kind")
            if isinstance(combine_kind, str):
                _increment_count(source_kind_counts, combine_kind)
                _increment_count(combine_kind_counts, combine_kind)
                _increment_count(
                    source_shape_policy_counts,
                    _source_shape_policy_label(
                        source_kind="multi",
                        combine_kind=combine_kind,
                        parent_count=parent_count,
                    ),
                )
            _increment_count(parent_arity_counts, _parent_arity_label(parent_count))
            if node_payload.get("combine_kind") == "stack":
                _increment_count(aggregation_kind_counts, node_payload.get("aggregation_kind"))
                parent_functions = node_payload.get("parent_functions")
                if isinstance(parent_functions, list):
                    for function_payload in parent_functions:
                        _walk_function_payload(
                            function_payload,
                            matrix_kind_counts=matrix_kind_counts,
                            activation_base_kind_counts=activation_base_kind_counts,
                            gp_variant_counts=gp_variant_counts,
                            kernel_gamma_values=kernel_gamma_values,
                            kernel_signed_counts=kernel_signed_counts,
                        )
            else:
                _walk_function_payload(
                    node_payload.get("function"),
                    matrix_kind_counts=matrix_kind_counts,
                    activation_base_kind_counts=activation_base_kind_counts,
                    gp_variant_counts=gp_variant_counts,
                    kernel_gamma_values=kernel_gamma_values,
                    kernel_signed_counts=kernel_signed_counts,
                )

        converter_plans = node_payload.get("converter_plans")
        if not isinstance(converter_plans, list):
            continue
        for converter_payload in converter_plans:
            if not isinstance(converter_payload, dict):
                continue
            method = converter_payload.get("method")
            variant = converter_payload.get("variant")
            if isinstance(method, str):
                _increment_count(converter_method_counts, method)
            if isinstance(variant, str):
                _increment_count(converter_variant_counts, variant)
            if isinstance(method, str) and isinstance(variant, str):
                _increment_count(converter_method_variant_counts, f"{method}.{variant}")
            _walk_function_payload(
                converter_payload.get("function"),
                matrix_kind_counts=matrix_kind_counts,
                activation_base_kind_counts=activation_base_kind_counts,
                gp_variant_counts=gp_variant_counts,
                kernel_gamma_values=kernel_gamma_values,
                kernel_signed_counts=kernel_signed_counts,
            )

    return {
        "schema_name": _PARITY_SURFACE_SCHEMA_NAME,
        "schema_version": int(_PARITY_SURFACE_SCHEMA_VERSION),
        "converter_method_counts": dict(sorted(converter_method_counts.items())),
        "converter_variant_counts": dict(sorted(converter_variant_counts.items())),
        "converter_method_variant_counts": dict(sorted(converter_method_variant_counts.items())),
        "gp_variant_counts": dict(sorted(gp_variant_counts.items())),
        "kernel_gamma": _summarize_values(kernel_gamma_values),
        "kernel_signed_counts": dict(sorted(kernel_signed_counts.items())),
        "matrix_kind_counts": dict(sorted(matrix_kind_counts.items())),
        "activation_base_kind_counts": dict(sorted(activation_base_kind_counts.items())),
        "root_base_kind_counts": dict(sorted(root_base_kind_counts.items())),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "combine_kind_counts": dict(sorted(combine_kind_counts.items())),
        "aggregation_kind_counts": dict(sorted(aggregation_kind_counts.items())),
        "parent_arity_counts": dict(sorted(parent_arity_counts.items())),
        "source_shape_policy_counts": dict(sorted(source_shape_policy_counts.items())),
        "categorical_cardinality": _summarize_values(categorical_cardinality_values),
    }


def _extract_emitted_schema_signature(
    bundle: DatasetBundle,
) -> tuple[int, tuple[str, ...], tuple[int, ...], int]:
    n_features = int(bundle.metadata.get("n_features", int(bundle.X_train.shape[1])))
    feature_types = tuple(str(t) for t in bundle.feature_types)
    if len(feature_types) != n_features:
        raise ValueError(
            "Fixed-layout bundle emitted inconsistent feature schema metadata: "
            f"n_features={n_features}, feature_types_len={len(feature_types)}."
        )

    lineage = bundle.metadata.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("Fixed-layout bundle is missing lineage metadata.")
    assignments = lineage.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Fixed-layout bundle is missing lineage assignments metadata.")
    raw_feature_to_node = assignments.get("feature_to_node")
    if not isinstance(raw_feature_to_node, list):
        raise ValueError("Fixed-layout bundle is missing lineage assignments.feature_to_node.")
    feature_to_node = tuple(int(value) for value in raw_feature_to_node)
    if len(feature_to_node) != n_features:
        raise ValueError(
            "Fixed-layout bundle emitted inconsistent lineage feature mapping: "
            f"n_features={n_features}, feature_to_node_len={len(feature_to_node)}."
        )
    raw_target_to_node = assignments.get("target_to_node")
    if isinstance(raw_target_to_node, bool) or not isinstance(raw_target_to_node, int):
        raise ValueError("Fixed-layout bundle is missing lineage assignments.target_to_node.")

    return n_features, feature_types, feature_to_node, int(raw_target_to_node)


__all__ = [
    "_FixedLayoutPlan",
    "_annotate_fixed_layout_metadata",
    "_extract_emitted_schema_signature",
    "_layout_signature",
]
