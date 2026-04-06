"""Fixed-layout hard-intervention resolution helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dagzoo.config import (
    INTERVENTION_MODE_HARD_INTERVENTIONAL,
    INTERVENTION_MODE_OBSERVATIONAL,
    INTERVENTION_TARGET_KIND_FEATURE_NODE,
    INTERVENTION_TARGET_KIND_LATENT_NODE,
    INTERVENTION_TARGET_KIND_TARGET,
    GeneratorConfig,
)
from dagzoo.core.layout_types import LayoutPlan


@dataclass(frozen=True, slots=True)
class FixedLayoutNodeIntervention:
    """One resolved hard intervention on a sampled latent node."""

    node_index: int
    value: float


@dataclass(frozen=True, slots=True)
class FixedLayoutResolvedInterventionPlan:
    """Resolved hard-intervention selectors for one sampled fixed-layout plan."""

    node_interventions: tuple[FixedLayoutNodeIntervention, ...] = ()
    target_value: float | None = None


def _merge_resolved_value(
    *,
    label: str,
    current: float | None,
    candidate: float,
) -> float:
    """Merge one resolved selector value, rejecting conflicting collisions."""

    if current is None:
        return float(candidate)
    if math.isclose(float(current), float(candidate), rel_tol=0.0, abs_tol=1e-12):
        return float(current)
    raise ValueError(
        "Resolved hard-intervention selectors collide on "
        f"{label}: existing value {float(current)!r}, new value {float(candidate)!r}."
    )


def resolve_fixed_layout_intervention_plan(
    config: GeneratorConfig,
    layout: LayoutPlan,
) -> FixedLayoutResolvedInterventionPlan | None:
    """Resolve authored hard-intervention selectors against one sampled layout."""

    mode = str(config.intervention.mode)
    if mode == INTERVENTION_MODE_OBSERVATIONAL:
        return None
    if mode != INTERVENTION_MODE_HARD_INTERVENTIONAL:
        raise ValueError(f"Unsupported fixed-layout intervention mode {mode!r}.")

    target_value: float | None = None
    node_values: dict[int, float] = {}
    feature_to_node = [int(node_index) for node_index in layout.feature_node_assignment]

    for target in config.intervention.targets:
        target_kind = str(target.target_kind)
        value = float(target.value)
        if target_kind == INTERVENTION_TARGET_KIND_TARGET:
            target_value = _merge_resolved_value(
                label="target output",
                current=target_value,
                candidate=value,
            )
            continue
        if target_kind == INTERVENTION_TARGET_KIND_FEATURE_NODE:
            if target.index is None:
                raise ValueError("feature_node interventions require a resolved feature index.")
            feature_index = int(target.index)
            if feature_index < 0 or feature_index >= len(feature_to_node):
                raise ValueError(
                    "Resolved hard-intervention feature selector is out of range for the sampled "
                    f"layout: feature_index={feature_index} n_features={len(feature_to_node)}."
                )
            node_index = int(feature_to_node[feature_index])
            label = f"latent node {node_index} (resolved from feature_node[{feature_index}])"
        elif target_kind == INTERVENTION_TARGET_KIND_LATENT_NODE:
            if target.index is None:
                raise ValueError("latent_node interventions require a resolved node index.")
            node_index = int(target.index)
            label = f"latent node {node_index}"
        else:
            raise ValueError(f"Unsupported fixed-layout intervention target kind {target_kind!r}.")
        node_values[node_index] = _merge_resolved_value(
            label=label,
            current=node_values.get(node_index),
            candidate=value,
        )

    if not node_values and target_value is None:
        return None
    return FixedLayoutResolvedInterventionPlan(
        node_interventions=tuple(
            FixedLayoutNodeIntervention(node_index=int(node_index), value=float(node_value))
            for node_index, node_value in sorted(node_values.items())
        ),
        target_value=None if target_value is None else float(target_value),
    )


__all__ = [
    "FixedLayoutNodeIntervention",
    "FixedLayoutResolvedInterventionPlan",
    "resolve_fixed_layout_intervention_plan",
]
