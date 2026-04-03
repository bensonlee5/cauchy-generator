import itertools

import torch
from conftest import load_repo_config

import dagzoo.core.layout as layout_mod
from dagzoo.core.fixed_layout.metadata import _layout_signature
from dagzoo.rng import KeyedRng

_GRAPH_BREADTH_STRESS_PROFILE = "anti_memorization_piecewise_classification_graph_breadth_slice_v1"
_COMPOSITIONAL_STRESS_PROFILE = "anti_memorization_piecewise_classification_compositional_slice_v1"


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


def test_sample_layout_only_enables_relationship_profile_for_graph_breadth(
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

    assert "graph_relationship_profile" in observed_names
