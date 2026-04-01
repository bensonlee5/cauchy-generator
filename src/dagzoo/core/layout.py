"""Layout and node-spec sampling helpers for dataset generation."""

from __future__ import annotations

import math

import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.fixed_layout.plan_types import FixedLayoutConverterSpec
from dagzoo.core.layout_types import FeatureType, LayoutPlan
from dagzoo.core.shift import resolve_shift_runtime_params
from dagzoo.functions._rng_helpers import randint_scalar
from dagzoo.graph import dag_edge_density, dag_longest_path_nodes, sample_dag
from dagzoo.rng import KeyedRng
from dagzoo.sampling import CorrelatedSampler


def _sample_log_uniform_int(generator: torch.Generator, device: str, low: int, high: int) -> int:
    """Sample an integer from a log-uniform range [low, high]."""

    log_low = math.log(float(low))
    log_high = math.log(float(high))
    u = torch.empty(1, device=device).uniform_(log_low, log_high, generator=generator)
    sampled = int(math.exp(u.item()))
    return max(low, min(high, sampled))


def _sample_node_count(
    n_nodes_min: int,
    n_nodes_max: int,
    generator: torch.Generator,
    device: str,
) -> int:
    """Sample graph node count using log-uniform bounds."""

    low = max(2, int(n_nodes_min))
    high = max(2, int(n_nodes_max))
    if low > high:
        raise ValueError(f"graph.n_nodes_min must be <= n_nodes_max, got {low} > {high}.")
    return _sample_log_uniform_int(generator, device, low, high)


def _sample_assignments(
    n_cols: int, n_nodes: int, generator: torch.Generator, device: str
) -> list[int]:
    """Assign columns to a random eligible subset of graph nodes."""

    eligible_count = int(randint_scalar(1, n_nodes + 1, generator))
    all_nodes = torch.randperm(n_nodes, generator=generator, device=device)
    eligible_nodes = all_nodes[:eligible_count]
    # Sample with replacement from eligible nodes
    indices = torch.randint(0, eligible_count, (n_cols,), generator=generator, device=device)
    return eligible_nodes[indices].tolist()


def _ancestor_nodes_for_target(adjacency: torch.Tensor, *, target_to_node: int) -> set[int]:
    adjacency_cpu = adjacency.to(device="cpu", dtype=torch.bool)
    ancestors = {int(target_to_node)}
    frontier = [int(target_to_node)]
    while frontier:
        node_index = int(frontier.pop())
        for parent_index in range(int(adjacency_cpu.shape[0])):
            if not bool(adjacency_cpu[parent_index, node_index]) or parent_index in ancestors:
                continue
            ancestors.add(parent_index)
            frontier.append(parent_index)
    return ancestors


def _eligible_target_nodes(
    *,
    adjacency: torch.Tensor,
    feature_to_node: list[int],
    config: GeneratorConfig,
) -> list[int]:
    adjacency_cpu = adjacency.to(device="cpu", dtype=torch.bool)
    min_target_indegree = int(config.filter.min_target_indegree)
    min_target_relevant_feature_count = int(config.filter.min_target_relevant_feature_count)
    min_target_relevant_feature_fraction = float(config.filter.min_target_relevant_feature_fraction)
    total_features = int(len(feature_to_node))
    eligible: list[int] = []

    for target_to_node in range(int(adjacency_cpu.shape[0])):
        target_indegree = int(adjacency_cpu[:, target_to_node].to(dtype=torch.int64).sum().item())
        if target_indegree < min_target_indegree:
            continue
        relevant_nodes = _ancestor_nodes_for_target(adjacency_cpu, target_to_node=target_to_node)
        relevant_feature_count = int(
            sum(1 for node_index in feature_to_node if int(node_index) in relevant_nodes)
        )
        if relevant_feature_count == 0:
            continue
        if relevant_feature_count < min_target_relevant_feature_count:
            continue
        relevant_feature_fraction = (
            float(relevant_feature_count) / float(total_features) if total_features > 0 else 0.0
        )
        if relevant_feature_fraction + 1e-12 < min_target_relevant_feature_fraction:
            continue
        eligible.append(int(target_to_node))
    return eligible


def _sample_target_node(
    *,
    adjacency: torch.Tensor,
    feature_to_node: list[int],
    config: GeneratorConfig,
    keyed_rng: KeyedRng,
    device: str,
) -> int:
    generator = keyed_rng.keyed("assignments", "target").torch_rng(device=device)
    eligible_target_nodes = _eligible_target_nodes(
        adjacency=adjacency,
        feature_to_node=feature_to_node,
        config=config,
    )
    if not eligible_target_nodes:
        return int(_sample_assignments(1, int(adjacency.shape[0]), generator, device)[0])
    target_index = int(
        torch.randint(
            0,
            len(eligible_target_nodes),
            (1,),
            generator=generator,
            device=device,
        ).item()
    )
    return int(eligible_target_nodes[target_index])


def _sample_layout(
    config: GeneratorConfig,
    keyed_rng: KeyedRng,
    device: str,
) -> LayoutPlan:
    """Sample dataset layout, graph, and node assignments for one dataset instance."""

    shift_params = resolve_shift_runtime_params(config)
    sampled_feature_min = int(config.dataset.n_features_min)
    sampled_feature_max = int(config.dataset.n_features_max)
    if sampled_feature_min > sampled_feature_max:
        raise ValueError(
            "dataset.n_features_min must be <= n_features_max, "
            f"got {sampled_feature_min} > {sampled_feature_max}."
        )
    num_features = int(
        torch.randint(
            sampled_feature_min,
            sampled_feature_max + 1,
            (1,),
            generator=keyed_rng.keyed("feature_count").torch_rng(device=device),
        ).item()
    )

    corr = CorrelatedSampler(keyed_rng.keyed("correlated"), device)
    cat_ratio = float(
        corr.sample_num(
            "categorical_ratio",
            config.dataset.categorical_ratio_min,
            config.dataset.categorical_ratio_max,
            log_scale=False,
            as_int=False,
        )
    )
    num_categorical_features = int(round(cat_ratio * num_features))
    num_categorical_features = max(0, min(num_features, num_categorical_features))
    if num_categorical_features > 0:
        cat_idx_t = torch.randperm(
            num_features,
            generator=keyed_rng.keyed("categorical_feature_indices").torch_rng(device=device),
            device=device,
        )[:num_categorical_features]
        cat_idx_t, _ = torch.sort(cat_idx_t)
        cat_idx = cat_idx_t.tolist()
    else:
        cat_idx = []

    max_card = max(2, config.dataset.max_categorical_cardinality)
    cardinalities = []
    for feature_index in cat_idx:
        generator = keyed_rng.keyed("cardinality", int(feature_index)).torch_rng(device=device)
        log_low = math.log(2.0)
        log_high = math.log(float(max_card))
        u = torch.empty(1, device=device).uniform_(log_low, log_high, generator=generator)
        cardinalities.append(max(2, int(math.exp(u.item()))))
    card_by_feature = {
        int(idx): int(card) for idx, card in zip(cat_idx, cardinalities, strict=True)
    }

    n_classes = int(
        torch.randint(
            config.dataset.n_classes_min,
            config.dataset.n_classes_max + 1,
            (1,),
            generator=keyed_rng.keyed("n_classes").torch_rng(device=device),
        ).item()
    )
    n_classes = max(2, n_classes)

    num_nodes = _sample_node_count(
        int(config.graph.n_nodes_min),
        int(config.graph.n_nodes_max),
        keyed_rng.keyed("graph_nodes").torch_rng(device=device),
        device,
    )
    structure_attempts = max(1, int(config.filter.max_attempts))
    adjacency: torch.Tensor | None = None
    feature_to_node: list[int] | None = None
    target_to_node: int | None = None
    graph_depth_nodes = 1
    graph_edge_density = 0.0
    for structure_attempt in range(structure_attempts):
        attempt_root = (
            keyed_rng
            if structure_attempt == 0
            else keyed_rng.keyed("structure_candidate", int(structure_attempt))
        )
        adjacency = sample_dag(
            num_nodes,
            attempt_root.keyed("graph").torch_rng(device="cpu"),
            edge_logit_bias=float(shift_params.edge_logit_bias_shift),
        )
        graph_depth_nodes = dag_longest_path_nodes(adjacency)
        graph_edge_density = dag_edge_density(adjacency)
        feature_to_node = _sample_assignments(
            num_features,
            num_nodes,
            attempt_root.keyed("assignments", "feature").torch_rng(device=device),
            device,
        )
        target_to_node = _sample_target_node(
            adjacency=adjacency,
            feature_to_node=feature_to_node,
            config=config,
            keyed_rng=attempt_root,
            device=device,
        )
        if _eligible_target_nodes(
            adjacency=adjacency,
            feature_to_node=feature_to_node,
            config=config,
        ):
            break
    assert adjacency is not None
    assert feature_to_node is not None
    assert target_to_node is not None

    feature_types: list[FeatureType] = ["num"] * num_features
    for i in cat_idx:
        feature_types[int(i)] = "cat"

    return LayoutPlan(
        n_features=num_features,
        n_cat=num_categorical_features,
        cat_idx=cat_idx,
        cardinalities=cardinalities,
        card_by_feature=card_by_feature,
        n_classes=n_classes,
        feature_types=feature_types,
        graph_nodes=num_nodes,
        graph_edges=int(adjacency.sum().item()),
        graph_depth_nodes=int(graph_depth_nodes),
        graph_edge_density=float(graph_edge_density),
        adjacency=adjacency,
        feature_node_assignment=feature_to_node,
        target_to_node=target_to_node,
    )


def _resample_layout_graph(
    layout: LayoutPlan,
    *,
    config: GeneratorConfig,
    keyed_rng: KeyedRng,
    edge_logit_bias: float,
) -> LayoutPlan:
    """Resample only the graph portion of a layout while preserving feature schema."""

    feature_to_node = [int(value) for value in layout.feature_node_assignment]
    structure_attempts = max(1, int(config.filter.max_attempts))
    adjacency: torch.Tensor | None = None
    graph_depth_nodes = 1
    graph_edge_density = 0.0
    base_adjacency = torch.as_tensor(layout.adjacency, dtype=torch.bool, device="cpu")
    for structure_attempt in range(structure_attempts):
        attempt_root = (
            keyed_rng
            if structure_attempt == 0
            else keyed_rng.keyed("structure_candidate", int(structure_attempt))
        )
        adjacency = sample_dag(
            int(layout.graph_nodes),
            attempt_root.keyed("graph").torch_rng(device="cpu"),
            edge_logit_bias=float(edge_logit_bias),
        )
        graph_depth_nodes = dag_longest_path_nodes(adjacency)
        graph_edge_density = dag_edge_density(adjacency)
        if int(layout.target_to_node) in _eligible_target_nodes(
            adjacency=adjacency,
            feature_to_node=feature_to_node,
            config=config,
        ) and (
            not torch.equal(adjacency.to(device="cpu", dtype=torch.bool), base_adjacency)
            or structure_attempt == (structure_attempts - 1)
        ):
            break
    assert adjacency is not None
    return LayoutPlan(
        n_features=int(layout.n_features),
        n_cat=int(layout.n_cat),
        cat_idx=[int(idx) for idx in layout.cat_idx],
        cardinalities=[int(value) for value in layout.cardinalities],
        card_by_feature={int(key): int(value) for key, value in layout.card_by_feature.items()},
        n_classes=int(layout.n_classes),
        feature_types=list(layout.feature_types),
        graph_nodes=int(layout.graph_nodes),
        graph_edges=int(adjacency.sum().item()),
        graph_depth_nodes=int(graph_depth_nodes),
        graph_edge_density=float(graph_edge_density),
        adjacency=adjacency,
        feature_node_assignment=feature_to_node,
        target_to_node=int(layout.target_to_node),
    )


def _feature_key(feature_index: int) -> str:
    """Return canonical feature extraction key for one feature column."""

    return f"feature_{int(feature_index)}"


def _build_node_specs(
    node_index: int,
    layout: LayoutPlan,
    keyed_rng: KeyedRng,
) -> list[FixedLayoutConverterSpec]:
    """Build converter specs for one node in the graph execution order."""

    specs: list[FixedLayoutConverterSpec] = []
    feature_to_node = layout.feature_node_assignment
    feature_types = list(layout.feature_types)
    card_by_feature: dict[int, int] = layout.card_by_feature
    column_cursor = 0

    def _append_spec(
        *,
        key: str,
        kind: str,
        dim: int,
        cardinality: int | None,
    ) -> None:
        nonlocal column_cursor
        width = max(1, int(dim))
        specs.append(
            FixedLayoutConverterSpec(
                key=key,
                kind=kind,  # type: ignore[arg-type]
                dim=width,
                cardinality=cardinality,
                column_start=column_cursor,
                column_end=column_cursor + width,
            )
        )
        column_cursor += width

    feature_indices = [
        index for index, assignment in enumerate(feature_to_node) if assignment == node_index
    ]
    for feature_index in feature_indices:
        if feature_types[feature_index] == "cat":
            feature_generator = keyed_rng.keyed("feature", feature_index).torch_rng(device="cpu")
            cardinality = int(card_by_feature[feature_index])
            if (
                cardinality > 2
                and torch.empty(1).uniform_(0, 1, generator=feature_generator).item() >= 0.5
            ):
                output_dim = int(randint_scalar(1, cardinality, feature_generator))
            else:
                output_dim = cardinality
            _append_spec(
                key=_feature_key(feature_index),
                kind="cat",
                dim=max(1, output_dim),
                cardinality=cardinality,
            )
        else:
            _append_spec(
                key=_feature_key(feature_index),
                kind="num",
                dim=1,
                cardinality=None,
            )

    return specs


def _build_target_specs(
    layout: LayoutPlan,
    task: str,
) -> list[FixedLayoutConverterSpec]:
    """Build converter specs for the selected latent target node."""

    if task == "classification":
        n_classes = int(layout.n_classes)
        return [
            FixedLayoutConverterSpec(
                key="target",
                kind="target_cls",
                dim=max(2, n_classes),
                cardinality=n_classes,
                column_start=0,
                column_end=max(2, n_classes),
            )
        ]
    return [
        FixedLayoutConverterSpec(
            key="target",
            kind="target_reg",
            dim=1,
            cardinality=None,
            column_start=0,
            column_end=1,
        )
    ]
