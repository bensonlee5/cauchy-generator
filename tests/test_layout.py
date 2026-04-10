import itertools

import torch
from conftest import load_repo_config

import dagzoo.core.layout as layout_mod
from dagzoo.core.fixed_layout.metadata import _layout_signature
from dagzoo.rng import KeyedRng

_GRAPH_BREADTH_STRESS_PROFILE = "anti_memorization_piecewise_classification_graph_breadth_slice_v1"
_COMPOSITIONAL_STRESS_PROFILE = "anti_memorization_piecewise_classification_compositional_slice_v1"
_HYBRID_STRESS_PROFILE = "anti_memorization_piecewise_classification_hybrid_slice_v1"


def _small_layout_config():
    cfg = load_repo_config()
    cfg.dataset.task = "classification"
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 4
    cfg.dataset.n_classes_min = 3
    cfg.dataset.n_classes_max = 3
    cfg.dataset.categorical_ratio_min = 0.0
    cfg.dataset.categorical_ratio_max = 0.0
    cfg.graph.n_nodes_min = 4
    cfg.graph.n_nodes_max = 4
    cfg.filter.max_attempts = 2
    cfg.filter.min_target_indegree = 1
    cfg.filter.min_target_relevant_feature_count = 1
    cfg.filter.min_target_relevant_feature_fraction = 0.0
    cfg.validate_generation_constraints()
    return cfg


def test_sample_layout_graph_breadth_policy_is_deterministic() -> None:
    cfg = _small_layout_config()

    first = layout_mod._sample_layout(
        cfg,
        KeyedRng(123).keyed("layout"),
        "cpu",
        stress_profile_name=_GRAPH_BREADTH_STRESS_PROFILE,
    )
    second = layout_mod._sample_layout(
        cfg,
        KeyedRng(123).keyed("layout"),
        "cpu",
        stress_profile_name=_GRAPH_BREADTH_STRESS_PROFILE,
    )

    assert _layout_signature(first) == _layout_signature(second)
    assert first.feature_node_assignment == second.feature_node_assignment
    assert int(first.target_to_node) == int(second.target_to_node)
    torch.testing.assert_close(first.adjacency, second.adjacency)


def test_sample_layout_graph_breadth_prefers_higher_scoring_candidate(
    monkeypatch,
) -> None:
    cfg = _small_layout_config()
    low_breadth = torch.tensor(
        [
            [False, False, False, True],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )
    high_breadth = torch.tensor(
        [
            [False, False, True, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )
    dag_iter = itertools.chain([low_breadth, high_breadth], itertools.repeat(high_breadth))

    monkeypatch.setattr(layout_mod, "sample_dag", lambda *_args, **_kwargs: next(dag_iter).clone())
    monkeypatch.setattr(
        layout_mod,
        "_sample_assignments",
        lambda n_cols, _n_nodes, _generator, _device: [int(index) for index in range(int(n_cols))],
    )
    monkeypatch.setattr(layout_mod, "_sample_target_node", lambda **_kwargs: 3)
    monkeypatch.setattr(
        layout_mod,
        "sample_correlated_choice",
        lambda *_args, **kwargs: (
            "ancestor_breadth"
            if kwargs["name"] == "graph_relationship_profile"
            else kwargs["values"][0]
        ),
    )

    sampled = layout_mod._sample_layout(
        cfg,
        KeyedRng(321).keyed("layout"),
        "cpu",
        stress_profile_name=_GRAPH_BREADTH_STRESS_PROFILE,
    )

    torch.testing.assert_close(sampled.adjacency, high_breadth)
    assert int(sampled.target_to_node) == 3


def test_sample_layout_only_enables_relationship_profile_for_graph_enabled_profiles(
    monkeypatch,
) -> None:
    cfg = _small_layout_config()
    observed_names: list[str] = []
    original_choice = layout_mod.sample_correlated_choice

    def _recording_choice(*args, **kwargs):
        observed_names.append(str(kwargs["name"]))
        return original_choice(*args, **kwargs)

    monkeypatch.setattr(layout_mod, "sample_correlated_choice", _recording_choice)

    _ = layout_mod._sample_layout(
        cfg,
        KeyedRng(111).keyed("layout", "baseline"),
        "cpu",
        stress_profile_name=None,
    )
    _ = layout_mod._sample_layout(
        cfg,
        KeyedRng(112).keyed("layout", "compositional"),
        "cpu",
        stress_profile_name=_COMPOSITIONAL_STRESS_PROFILE,
    )

    assert "graph_relationship_profile" not in observed_names

    _ = layout_mod._sample_layout(
        cfg,
        KeyedRng(113).keyed("layout", "graph_breadth"),
        "cpu",
        stress_profile_name=_GRAPH_BREADTH_STRESS_PROFILE,
    )
    _ = layout_mod._sample_layout(
        cfg,
        KeyedRng(114).keyed("layout", "hybrid"),
        "cpu",
        stress_profile_name=_HYBRID_STRESS_PROFILE,
    )

    assert "graph_relationship_profile" in observed_names


def test_mean_ancestor_overlap_returns_none_for_insufficient_or_empty_pairs() -> None:
    assert (
        layout_mod._mean_ancestor_overlap(
            feature_to_node=[0],
            ancestor_masks=torch.eye(1, dtype=torch.bool),
        )
        is None
    )
    assert (
        layout_mod._mean_ancestor_overlap(
            feature_to_node=[0, 1],
            ancestor_masks=torch.zeros((2, 2), dtype=torch.bool),
        )
        is None
    )


def test_sample_target_node_graph_breadth_tiebreak_is_seeded(monkeypatch) -> None:
    cfg = _small_layout_config()
    adjacency = torch.zeros((4, 4), dtype=torch.bool)
    feature_to_node = [0, 1, 2, 2]
    monkeypatch.setattr(layout_mod, "_eligible_target_nodes", lambda **_kwargs: [1, 2])
    monkeypatch.setattr(
        layout_mod,
        "_ancestor_nodes_for_target",
        lambda _adjacency, *, target_to_node: {0, 1} if int(target_to_node) == 1 else {0, 2},
    )

    first = layout_mod._sample_target_node(
        adjacency=adjacency,
        feature_to_node=feature_to_node,
        config=cfg,
        keyed_rng=KeyedRng(77).keyed("layout"),
        device="cpu",
        relationship_profile="mixed_breadth",
    )
    second = layout_mod._sample_target_node(
        adjacency=adjacency,
        feature_to_node=feature_to_node,
        config=cfg,
        keyed_rng=KeyedRng(77).keyed("layout"),
        device="cpu",
        relationship_profile="mixed_breadth",
    )

    assert first == second
    assert first in {1, 2}


def test_resample_layout_graph_graph_breadth_prefers_higher_scoring_candidate(monkeypatch) -> None:
    cfg = _small_layout_config()
    base_layout = layout_mod.LayoutPlan(
        n_features=4,
        n_cat=0,
        cat_idx=[],
        cardinalities=[],
        card_by_feature={},
        n_classes=3,
        feature_types=["num", "num", "num", "num"],
        graph_nodes=4,
        graph_edges=2,
        graph_depth_nodes=2,
        graph_edge_density=2.0 / 6.0,
        adjacency=torch.tensor(
            [
                [False, False, False, True],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        ),
        feature_node_assignment=[0, 1, 2, 2],
        target_to_node=3,
    )
    repeated_base = base_layout.adjacency.clone()
    higher_scoring = torch.tensor(
        [
            [False, False, True, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )
    lower_scoring = torch.tensor(
        [
            [False, False, False, True],
            [False, False, True, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )
    dag_iter = iter([repeated_base, lower_scoring, higher_scoring] + [higher_scoring] * 8)
    sampled_biases: list[float] = []

    def _fake_sample_dag(_num_nodes, _generator, *, edge_logit_bias):
        sampled_biases.append(float(edge_logit_bias))
        return next(dag_iter).clone()

    monkeypatch.setattr(layout_mod, "sample_dag", _fake_sample_dag)
    monkeypatch.setattr(
        layout_mod,
        "sample_correlated_choice",
        lambda *_args, **kwargs: (
            "ancestor_breadth"
            if kwargs["name"] == "graph_relationship_profile"
            else kwargs["values"][0]
        ),
    )

    resampled = layout_mod._resample_layout_graph(
        base_layout,
        config=cfg,
        keyed_rng=KeyedRng(999).keyed("layout"),
        edge_logit_bias=0.25,
        stress_profile_name=_GRAPH_BREADTH_STRESS_PROFILE,
    )

    torch.testing.assert_close(resampled.adjacency, higher_scoring)
    assert sampled_biases[0] == 0.70
