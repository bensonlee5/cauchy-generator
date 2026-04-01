from __future__ import annotations

import numpy as np
import pytest
import torch

from dagzoo.filtering import apply_extra_trees_filter
from dagzoo.filtering.extra_trees_filter import (
    _apply_extra_trees_filter_numpy,
    _lineage_has_feature_target_path,
)
from dagzoo.io.lineage_schema import LINEAGE_SCHEMA_VERSION_DENSE


def _make_regression_split(
    *,
    seed: int = 7,
    n_train: int = 128,
    n_test: int = 64,
    n_features: int = 8,
    kind: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x_train = torch.randn(n_train, n_features, generator=gen)
    x_test = torch.randn(n_test, n_features, generator=gen)
    if kind == "linear":
        y_train = (
            2.0 * x_train[:, 0] - 0.5 * x_train[:, 1] + 0.05 * torch.randn(n_train, generator=gen)
        )
        y_test = 2.0 * x_test[:, 0] - 0.5 * x_test[:, 1] + 0.05 * torch.randn(n_test, generator=gen)
    elif kind == "piecewise":

        def _target(x: torch.Tensor) -> torch.Tensor:
            bucket = torch.floor((x[:, 0] + 3.0) * 2.5).to(torch.int64)
            pattern = ((bucket % 6) - 2).to(torch.float32)
            return pattern + 0.05 * x[:, 1]

        y_train = _target(x_train)
        y_test = _target(x_test)
    else:
        raise ValueError(f"Unsupported kind '{kind}'.")
    return x_train, y_train, x_test, y_test


def _make_classification_split(
    *,
    seed: int = 11,
    n_train: int = 128,
    n_test: int = 64,
    n_features: int = 6,
    n_classes: int = 4,
    garbage: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    centroids = torch.randn(n_classes, n_features, generator=gen)
    y_train = torch.arange(n_train, dtype=torch.int64) % n_classes
    y_test = torch.arange(n_test, dtype=torch.int64) % n_classes
    y_train = y_train[torch.randperm(n_train, generator=gen)]
    y_test = y_test[torch.randperm(n_test, generator=gen)]
    x_train = centroids[y_train] + 0.2 * torch.randn(n_train, n_features, generator=gen)
    x_test = centroids[y_test] + 0.2 * torch.randn(n_test, n_features, generator=gen)
    if garbage:
        y_train = torch.randint(0, n_classes, (n_train,), generator=gen, dtype=torch.int64)
        y_test = torch.randint(0, n_classes, (n_test,), generator=gen, dtype=torch.int64)
    return x_train, y_train, x_test, y_test


def _dense_lineage(
    *,
    feature_to_node: list[int],
    target_to_node: int,
    adjacency: list[list[int]],
):
    def _ancestor_nodes_for_target() -> set[int]:
        ancestors = {int(target_to_node)}
        frontier = [int(target_to_node)]
        while frontier:
            node_index = int(frontier.pop())
            for parent_index, row in enumerate(adjacency):
                if int(row[node_index]) == 0 or int(parent_index) in ancestors:
                    continue
                ancestors.add(int(parent_index))
                frontier.append(int(parent_index))
        return ancestors

    relevant_nodes = _ancestor_nodes_for_target()
    relevant_features = [
        feature_index
        for feature_index, node_index in enumerate(feature_to_node)
        if int(node_index) in relevant_nodes
    ]
    assignments: dict[str, object] = {
        "feature_to_node": feature_to_node,
        "target_to_node": int(target_to_node),
        "target_relevant_features": relevant_features,
        "target_relevant_feature_count": len(relevant_features),
        "target_relevant_feature_fraction": (
            float(len(relevant_features)) / float(len(feature_to_node)) if feature_to_node else 0.0
        ),
    }
    return {
        "schema_name": "dagzoo.dag_lineage",
        "schema_version": LINEAGE_SCHEMA_VERSION_DENSE,
        "graph": {
            "n_nodes": len(adjacency),
            "adjacency": adjacency,
        },
        "assignments": assignments,
    }


def test_extra_trees_filter_is_deterministic_for_fixed_seed() -> None:
    x_train, y_train, x_test, y_test = _make_regression_split()
    accepted_a, details_a = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="regression",
        seed=123,
        n_estimators=8,
        max_depth=5,
        n_bootstrap=33,
        ease_k_small=8,
    )
    accepted_b, details_b = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="regression",
        seed=123,
        n_estimators=8,
        max_depth=5,
        n_bootstrap=33,
        ease_k_small=8,
    )

    assert accepted_a == accepted_b
    assert details_a == details_b
    assert details_a["backend"] == "extra_trees_cpu"
    assert details_a["filter_mode"] == "small_shot_ease_v1"
    assert int(details_a["n_jobs"]) == -1


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_extra_trees_filter_numpy_helper_matches_torch_wrapper(task: str) -> None:
    if task == "classification":
        x_train, y_train, x_test, y_test = _make_classification_split(seed=141)
    else:
        x_train, y_train, x_test, y_test = _make_regression_split(seed=141)

    accepted_torch, details_torch = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task=task,
        seed=123,
        n_estimators=8,
        max_depth=5,
        min_samples_leaf=2,
        n_bootstrap=24,
        ease_k_small=8,
        n_jobs=1,
    )
    accepted_numpy, details_numpy = _apply_extra_trees_filter_numpy(
        x_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy(),
        x_test.detach().cpu().numpy(),
        y_test.detach().cpu().numpy(),
        task=task,
        seed=123,
        n_estimators=8,
        max_depth=5,
        min_samples_leaf=2,
        n_bootstrap=24,
        ease_k_small=8,
        n_jobs=1,
    )

    assert accepted_numpy == accepted_torch
    assert details_numpy == details_torch


def test_extra_trees_filter_rejects_trivially_easy_small_shot_classification() -> None:
    x_train, y_train, x_test, y_test = _make_classification_split(seed=11, n_classes=4)
    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="classification",
        seed=44,
        n_estimators=32,
        max_depth=6,
        n_bootstrap=64,
        ease_k_small=16,
        easy_skill_threshold=0.8,
        easy_gain_threshold=0.2,
    )

    assert accepted is False
    assert details["reason"] == "too_easy_small_shot"
    assert float(details["skill_small"]) > 0.8
    assert float(details["skill_gain_ub95"]) < 0.2


def test_extra_trees_filter_accepts_task_when_full_data_adds_meaningful_skill() -> None:
    x_train, y_train, x_test, y_test = _make_classification_split(seed=15, n_classes=8)
    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="classification",
        seed=77,
        n_estimators=32,
        max_depth=6,
        n_bootstrap=64,
        ease_k_small=4,
        easy_skill_threshold=0.85,
        easy_gain_threshold=0.1,
    )

    assert accepted is True
    assert "reason" not in details
    assert float(details["skill_full"]) > float(details["skill_small"])
    assert float(details["skill_gain"]) > 0.5


def test_extra_trees_filter_uses_shared_baseline_for_skill_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_train = torch.zeros((4, 1), dtype=torch.float32)
    x_test = torch.zeros((2, 1), dtype=torch.float32)
    y_train = torch.tensor([2.0, 2.0, -2.0, -2.0], dtype=torch.float32)
    y_test = torch.tensor([1.0, 3.0], dtype=torch.float32)

    call_count = {"value": 0}

    def _stub_fit(*, x_test: np.ndarray, **_kwargs) -> np.ndarray:
        _ = x_test
        call_count["value"] += 1
        if call_count["value"] == 1:
            return np.asarray([[2.0], [2.0]], dtype=np.float32)
        return np.asarray([[2.1], [2.1]], dtype=np.float32)

    class _FixedSmallSampleRng:
        def __init__(self, seed: int) -> None:
            self._rng = np.random.Generator(np.random.PCG64(seed))

        def choice(self, a, size=None, replace=True):  # noqa: ANN001
            assert int(a) == 4
            assert int(size) == 2
            assert replace is False
            return np.asarray([0, 1], dtype=np.int64)

        def integers(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return self._rng.integers(*args, **kwargs)

    monkeypatch.setattr(
        "dagzoo.filtering.extra_trees_filter._fit_extra_trees_predictions",
        _stub_fit,
    )
    monkeypatch.setattr(
        "dagzoo.filtering.extra_trees_filter.np.random.default_rng",
        lambda seed: _FixedSmallSampleRng(int(seed)),
    )

    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="regression",
        seed=17,
        n_bootstrap=16,
        ease_k_small=2,
        use_lineage_veto=False,
    )

    assert accepted is True
    assert float(details["skill_full"]) > float(details["skill_small"])
    assert float(details["skill_gain"]) < 0.0


def test_extra_trees_filter_rejects_garbage_classification_on_hard_side() -> None:
    x_train, y_train, x_test, y_test = _make_classification_split(garbage=True)
    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="classification",
        seed=91,
        n_estimators=16,
        max_depth=6,
        n_bootstrap=64,
        ease_k_small=16,
        hard_skill_threshold=0.05,
    )

    assert accepted is False
    assert details["reason"] == "too_hard_garbage"
    assert float(details["skill_full_ub95"]) < 0.05


def test_extra_trees_filter_rejects_single_feature_shortcut_with_stump_veto() -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(9)
    x_train = torch.randn(128, 4, generator=gen)
    x_test = torch.randn(64, 4, generator=gen)
    x_train[:, 0] = (x_train[:, 0] > 0.0).to(torch.float32)
    x_test[:, 0] = (x_test[:, 0] > 0.0).to(torch.float32)
    y_train = x_train[:, 0].clone()
    y_test = x_test[:, 0].clone()

    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="regression",
        seed=12,
        n_estimators=16,
        max_depth=6,
        n_bootstrap=32,
        ease_k_small=16,
        stump_skill_threshold=0.7,
    )

    assert accepted is False
    assert details["reason"] == "too_easy_stump"
    assert float(details["stump_skill"]) == pytest.approx(1.0)


def test_extra_trees_filter_rejects_lineage_without_feature_target_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_train, y_train, x_test, y_test = _make_regression_split(kind="linear")
    lineage = _dense_lineage(
        feature_to_node=[0, 0, 1],
        target_to_node=3,
        adjacency=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
    )

    monkeypatch.setattr(
        "dagzoo.filtering.extra_trees_filter.ExtraTreesRegressor",
        lambda *_args, **_kwargs: pytest.fail("model fit should not run after no-path veto"),
    )

    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="regression",
        seed=88,
        lineage_payload=lineage,
        use_lineage_veto=True,
    )

    assert accepted is False
    assert details["reason"] == "no_feature_target_path"
    assert details["lineage_veto_applied"] is True
    assert details["feature_target_path_exists"] is False


def test_extra_trees_filter_rejects_target_root_before_model_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_train, y_train, x_test, y_test = _make_regression_split(kind="linear")
    lineage = _dense_lineage(
        feature_to_node=[0, 0, 1],
        target_to_node=3,
        adjacency=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    )

    monkeypatch.setattr(
        "dagzoo.filtering.extra_trees_filter.ExtraTreesRegressor",
        lambda *_args, **_kwargs: pytest.fail("model fit should not run after target-root veto"),
    )

    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="regression",
        seed=88,
        lineage_payload=lineage,
        use_lineage_veto=True,
        min_target_indegree=1,
    )

    assert accepted is False
    assert details["reason"] == "target_root"
    assert details["target_indegree"] == 0
    assert details["lineage_veto_applied"] is True


def test_extra_trees_filter_rejects_insufficient_target_relevant_feature_count_before_model_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_train, y_train, x_test, y_test = _make_regression_split(kind="linear")
    lineage = _dense_lineage(
        feature_to_node=[0, 1, 1, 1],
        target_to_node=2,
        adjacency=[
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
    )

    monkeypatch.setattr(
        "dagzoo.filtering.extra_trees_filter.ExtraTreesRegressor",
        lambda *_args, **_kwargs: pytest.fail(
            "model fit should not run after relevant-feature-count veto"
        ),
    )

    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="regression",
        seed=88,
        lineage_payload=lineage,
        use_lineage_veto=True,
        min_target_indegree=0,
        min_target_relevant_feature_count=2,
        min_target_relevant_feature_fraction=0.0,
    )

    assert accepted is False
    assert details["reason"] == "insufficient_target_relevant_feature_count"
    assert details["target_relevant_feature_count"] == 1


def test_extra_trees_filter_rejects_insufficient_target_relevant_feature_fraction_before_model_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_train, y_train, x_test, y_test = _make_regression_split(kind="linear")
    lineage = _dense_lineage(
        feature_to_node=[0, 0, 1, 1],
        target_to_node=2,
        adjacency=[
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
    )

    monkeypatch.setattr(
        "dagzoo.filtering.extra_trees_filter.ExtraTreesRegressor",
        lambda *_args, **_kwargs: pytest.fail(
            "model fit should not run after relevant-feature-fraction veto"
        ),
    )

    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="regression",
        seed=88,
        lineage_payload=lineage,
        use_lineage_veto=True,
        min_target_indegree=0,
        min_target_relevant_feature_count=0,
        min_target_relevant_feature_fraction=0.75,
    )

    assert accepted is False
    assert details["reason"] == "insufficient_target_relevant_feature_fraction"
    assert details["target_relevant_feature_fraction"] == pytest.approx(0.5)


def test_lineage_veto_accepts_when_feature_nodes_reach_target_node() -> None:
    lineage = _dense_lineage(
        feature_to_node=[0, 1, 1],
        target_to_node=2,
        adjacency=[
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
    )

    lineage_present, has_path = _lineage_has_feature_target_path(
        lineage_payload=lineage,
        lineage_base_dir=None,
    )

    assert lineage_present is True
    assert has_path is True


def test_extra_trees_filter_rejects_prediction_collapse_on_classification_full_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_train, y_train, x_test, y_test = _make_classification_split(seed=12, n_classes=3)
    call_count = {"value": 0}

    def _stub_fit(*, x_test: np.ndarray, **_kwargs) -> np.ndarray:
        call_count["value"] += 1
        pred = np.zeros((x_test.shape[0], 3), dtype=np.float32)
        pred[:, 0] = 1.0
        return pred

    monkeypatch.setattr(
        "dagzoo.filtering.extra_trees_filter._fit_extra_trees_predictions",
        _stub_fit,
    )

    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="classification",
        seed=77,
        use_lineage_veto=False,
    )

    assert accepted is False
    assert call_count["value"] == 2
    assert details["reason"] == "prediction_collapse_full"
    assert details["predicted_unique_classes_full"] == 1
    assert details["cohen_kappa_full"] is not None


def test_extra_trees_filter_rejects_chance_kappa_on_classification_full_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_train = torch.zeros((4, 2), dtype=torch.float32)
    x_test = torch.zeros((4, 2), dtype=torch.float32)
    y_train = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    y_test = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    call_count = {"value": 0}

    def _stub_fit(*, x_test: np.ndarray, **_kwargs) -> np.ndarray:
        call_count["value"] += 1
        pred = np.zeros((x_test.shape[0], 2), dtype=np.float32)
        pred[:, 0] = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        pred[:, 1] = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return pred

    monkeypatch.setattr(
        "dagzoo.filtering.extra_trees_filter._fit_extra_trees_predictions",
        _stub_fit,
    )

    accepted, details = apply_extra_trees_filter(
        x_train,
        y_train,
        x_test,
        y_test,
        task="classification",
        seed=19,
        use_lineage_veto=False,
        classification_kappa_threshold=0.0,
    )

    assert accepted is False
    assert call_count["value"] == 2
    assert details["reason"] == "chance_kappa_full"
    assert float(details["cohen_kappa_full"]) <= 0.0
    assert details["predicted_unique_classes_full"] == 2


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("easy_skill_threshold", {"easy_skill_threshold": 1.1}),
        ("easy_gain_threshold", {"easy_gain_threshold": -0.1}),
        ("hard_skill_threshold", {"hard_skill_threshold": 1.1}),
        ("stump_skill_threshold", {"stump_skill_threshold": -0.1}),
        (
            "min_target_relevant_feature_fraction",
            {"min_target_relevant_feature_fraction": 1.1},
        ),
        ("classification_kappa_threshold", {"classification_kappa_threshold": 1.1}),
    ],
)
def test_extra_trees_filter_rejects_invalid_public_thresholds(
    field_name: str,
    kwargs: dict[str, float],
) -> None:
    x_train, y_train, x_test, y_test = _make_regression_split()
    message = (
        r"classification_kappa_threshold must be a finite value in \[-1.0, 1.0\]\."
        if field_name == "classification_kappa_threshold"
        else rf"{field_name} must be a finite value in \[0.0, 1.0\]"
    )
    with pytest.raises(ValueError, match=message):
        apply_extra_trees_filter(
            x_train,
            y_train,
            x_test,
            y_test,
            task="regression",
            seed=42,
            **kwargs,
        )


@pytest.mark.parametrize("bad_n_jobs", [0, -2, True])
def test_extra_trees_filter_rejects_invalid_n_jobs(bad_n_jobs: int | bool) -> None:
    x_train, y_train, x_test, y_test = _make_regression_split()
    with pytest.raises(ValueError, match=r"n_jobs must be -1 or an integer >= 1"):
        apply_extra_trees_filter(
            x_train,
            y_train,
            x_test,
            y_test,
            task="regression",
            seed=42,
            n_jobs=bad_n_jobs,
        )
