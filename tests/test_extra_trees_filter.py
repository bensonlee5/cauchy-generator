import pytest
import torch

from dagzoo.filtering import apply_extra_trees_filter
from dagzoo.filtering.extra_trees_filter import _apply_extra_trees_filter_numpy


def _make_regression_data(
    seed: int = 7, n_rows: int = 256, n_features: int = 12
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(n_rows, n_features, generator=g)
    w = torch.randn(n_features, 1, generator=g)
    y = (x @ w).squeeze(1) + 0.1 * torch.randn(n_rows, generator=g)
    return x, y


def _make_classification_data(
    seed: int = 11,
    n_rows: int = 256,
    n_features: int = 10,
    n_classes: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    y = torch.arange(n_rows, dtype=torch.int64) % int(n_classes)
    y = y[torch.randperm(n_rows, generator=g)]
    centroids = torch.randn(n_classes, n_features, generator=g)
    x = centroids[y] + 0.25 * torch.randn(n_rows, n_features, generator=g)
    return x, y


def _make_sparse_label_classification_data(
    *,
    seed: int = 91,
    n_rows: int = 256,
    n_features: int = 10,
    labels: tuple[int, ...] = (10, 20, 30, 40),
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    class_ids = torch.arange(n_rows, dtype=torch.int64) % int(len(labels))
    class_ids = class_ids[torch.randperm(n_rows, generator=g)]
    label_values = torch.tensor(labels, dtype=torch.int64)
    y = label_values[class_ids]
    centroids = torch.randn(len(labels), n_features, generator=g)
    x = centroids[class_ids] + 0.25 * torch.randn(n_rows, n_features, generator=g)
    return x, y


def test_extra_trees_filter_is_deterministic_for_fixed_seed() -> None:
    x, y = _make_regression_data()
    accepted_a, details_a = apply_extra_trees_filter(
        x,
        y,
        task="regression",
        seed=123,
        n_estimators=8,
        max_depth=5,
        n_bootstrap=33,
        threshold=0.5,
    )
    accepted_b, details_b = apply_extra_trees_filter(
        x,
        y,
        task="regression",
        seed=123,
        n_estimators=8,
        max_depth=5,
        n_bootstrap=33,
        threshold=0.5,
    )

    assert accepted_a == accepted_b
    assert details_a["wins_ratio"] == details_b["wins_ratio"]
    assert details_a["n_valid_oob"] == details_b["n_valid_oob"]
    assert details_a["backend"] == "extra_trees_cpu"
    assert int(details_a["n_jobs"]) == -1


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_extra_trees_filter_numpy_helper_matches_torch_wrapper(task: str) -> None:
    if task == "classification":
        x, y = _make_classification_data(seed=141)
    else:
        x, y = _make_regression_data(seed=141)

    accepted_torch, details_torch = apply_extra_trees_filter(
        x,
        y,
        task=task,
        seed=123,
        n_estimators=8,
        max_depth=5,
        min_samples_leaf=2,
        n_bootstrap=24,
        threshold=0.5,
        n_jobs=1,
    )
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    accepted_numpy, details_numpy = _apply_extra_trees_filter_numpy(
        x_np,
        y_np,
        task=task,
        seed=123,
        n_estimators=8,
        max_depth=5,
        min_samples_leaf=2,
        n_bootstrap=24,
        threshold=0.5,
        n_jobs=1,
    )

    assert accepted_numpy == accepted_torch
    assert details_numpy == details_torch


def test_extra_trees_filter_handles_non_divisible_bootstrap_chunks() -> None:
    x, y = _make_regression_data(seed=17)
    accepted, details = apply_extra_trees_filter(
        x,
        y,
        task="regression",
        seed=44,
        n_estimators=6,
        max_depth=4,
        n_bootstrap=33,
        threshold=0.5,
    )

    assert isinstance(accepted, bool)
    assert "wins_ratio" in details
    assert "n_valid_oob" in details
    assert details["backend"] == "extra_trees_cpu"


def test_extra_trees_filter_enforces_max_leaf_nodes() -> None:
    x, y = _make_regression_data(seed=21)
    accepted, details = apply_extra_trees_filter(
        x,
        y,
        task="regression",
        seed=42,
        n_estimators=6,
        max_depth=6,
        max_leaf_nodes=1,
        n_bootstrap=16,
        threshold=0.5,
    )

    assert isinstance(accepted, bool)
    assert details["backend"] == "extra_trees_cpu"


def test_extra_trees_filter_can_report_insufficient_oob_predictions() -> None:
    x, y = _make_regression_data(seed=5, n_rows=16, n_features=8)
    accepted, details = apply_extra_trees_filter(
        x,
        y,
        task="regression",
        seed=99,
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=2,
        n_bootstrap=16,
        threshold=0.5,
    )

    assert accepted is False
    assert details["reason"] == "insufficient_oob_predictions"
    assert 0 <= int(details["n_valid_oob"]) < 16
    assert details["backend"] == "extra_trees_cpu"
    assert int(details["n_jobs"]) == -1
    assert details["threshold_requested"] == pytest.approx(0.5)
    assert details["threshold_effective"] == pytest.approx(0.5)
    assert details["threshold_policy"] == "class_aware_piecewise_v1"
    assert details["class_count"] is None
    assert details["class_bucket"] == "not_applicable"
    assert details["threshold_delta"] == pytest.approx(0.0)


def test_extra_trees_filter_excludes_rows_with_no_oob_votes() -> None:
    n_rows = 64
    x, y = _make_regression_data(seed=13, n_rows=n_rows, n_features=8)
    accepted, details = apply_extra_trees_filter(
        x,
        y,
        task="regression",
        seed=115,
        n_estimators=1,
        max_depth=6,
        min_samples_leaf=1,
        n_bootstrap=8,
        threshold=0.1,
    )

    assert accepted is False
    assert details["reason"] == "insufficient_oob_predictions"
    assert 0 <= int(details["n_valid_oob"]) < n_rows


def test_extra_trees_filter_zero_threshold_bypasses_model_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = _make_regression_data(seed=13, n_rows=64, n_features=8)

    def _fail_model_init(*_args, **_kwargs):
        raise AssertionError("ExtraTreesRegressor should not be instantiated in bypass mode")

    monkeypatch.setattr(
        "dagzoo.filtering.extra_trees_filter.ExtraTreesRegressor",
        _fail_model_init,
    )

    accepted, details = apply_extra_trees_filter(
        x,
        y,
        task="regression",
        seed=115,
        n_estimators=1,
        max_depth=6,
        min_samples_leaf=1,
        n_bootstrap=8,
        threshold=0.0,
    )

    assert accepted is True
    assert details["backend"] == "filter_threshold_bypass"
    assert details["bypass"] is True
    assert details["threshold_requested"] == pytest.approx(0.0)
    assert details["threshold_effective"] == pytest.approx(0.0)
    assert details["threshold_policy"] == "zero_bypass_v1"
    assert details["threshold_delta"] == pytest.approx(0.0)
    assert "wins_ratio" not in details
    assert "n_valid_oob" not in details


def test_extra_trees_filter_classification_smoke() -> None:
    x, y = _make_classification_data()
    accepted, details = apply_extra_trees_filter(
        x,
        y,
        task="classification",
        seed=1,
        n_estimators=8,
        max_depth=4,
        min_samples_leaf=2,
        n_bootstrap=24,
        threshold=0.5,
    )

    assert isinstance(accepted, bool)
    assert "wins_ratio" in details
    assert "n_valid_oob" in details
    assert details["backend"] == "extra_trees_cpu"
    assert details["threshold_requested"] == pytest.approx(0.5)
    assert details["threshold_effective"] == pytest.approx(0.5)
    assert details["threshold_policy"] == "class_aware_piecewise_v1"
    assert int(details["class_count"]) >= 2
    assert details["class_bucket"] == "<=8"
    assert details["threshold_delta"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("n_classes", "expected_bucket", "expected_effective"),
    [
        (4, "<=8", 0.95),
        (10, "9-16", 0.90),
        (18, "17-24", 0.85),
        (28, "25-32", 0.80),
        (32, "25-32", 0.80),
    ],
)
def test_extra_trees_filter_class_aware_threshold_sweep(
    n_classes: int,
    expected_bucket: str,
    expected_effective: float,
) -> None:
    x, y = _make_classification_data(
        seed=211 + n_classes,
        n_rows=512,
        n_features=12,
        n_classes=n_classes,
    )
    _, details = apply_extra_trees_filter(
        x,
        y,
        task="classification",
        seed=17,
        n_estimators=6,
        max_depth=4,
        min_samples_leaf=2,
        n_bootstrap=16,
        threshold=0.95,
    )

    assert details["class_bucket"] == expected_bucket
    assert details["threshold_requested"] == pytest.approx(0.95)
    assert details["threshold_effective"] == pytest.approx(expected_effective)
    assert details["threshold_policy"] == "class_aware_piecewise_v1"
    assert int(details["class_count"]) == n_classes
    assert details["threshold_delta"] == pytest.approx(0.95 - expected_effective)


def test_extra_trees_filter_regression_threshold_diagnostics_are_not_applicable() -> None:
    x, y = _make_regression_data(seed=123)
    _, details = apply_extra_trees_filter(
        x,
        y,
        task="regression",
        seed=88,
        n_estimators=6,
        max_depth=4,
        n_bootstrap=16,
        threshold=0.75,
    )

    assert details["threshold_requested"] == pytest.approx(0.75)
    assert details["threshold_effective"] == pytest.approx(0.75)
    assert details["threshold_policy"] == "class_aware_piecewise_v1"
    assert details["class_count"] is None
    assert details["class_bucket"] == "not_applicable"
    assert details["threshold_delta"] == pytest.approx(0.0)


def test_extra_trees_filter_rejected_scored_run_keeps_threshold_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = _make_classification_data(seed=77, n_rows=384, n_features=12, n_classes=32)
    monkeypatch.setattr(
        "dagzoo.filtering.extra_trees_filter._bootstrap_wins_ratio",
        lambda **_kwargs: 0.5,
    )
    accepted, details = apply_extra_trees_filter(
        x,
        y,
        task="classification",
        seed=31,
        n_estimators=6,
        max_depth=4,
        min_samples_leaf=2,
        n_bootstrap=16,
        threshold=1.0,
    )

    assert accepted is False
    assert "wins_ratio" in details
    assert details["threshold_requested"] == pytest.approx(1.0)
    assert details["threshold_effective"] == pytest.approx(0.85)
    assert details["threshold_policy"] == "class_aware_piecewise_v1"
    assert details["class_bucket"] == "25-32"
    assert int(details["class_count"]) == 32
    assert details["threshold_delta"] == pytest.approx(0.15)


def test_extra_trees_filter_many_class_acceptance_rate_is_not_pathological() -> None:
    seeds = [1701, 1702, 1703, 1704, 1705, 1706]
    acceptance_rates: dict[int, float] = {}

    for n_classes in (8, 16, 24, 32):
        accepted = 0
        for seed in seeds:
            x, y = _make_classification_data(
                seed=seed + (n_classes * 100),
                n_rows=512,
                n_features=12,
                n_classes=n_classes,
            )
            is_accepted, _details = apply_extra_trees_filter(
                x,
                y,
                task="classification",
                seed=seed,
                n_estimators=6,
                max_depth=4,
                min_samples_leaf=2,
                n_bootstrap=16,
                threshold=0.95,
            )
            if is_accepted:
                accepted += 1
        acceptance_rates[n_classes] = accepted / float(len(seeds))

    assert acceptance_rates[32] >= (acceptance_rates[8] - 0.25)


@pytest.mark.parametrize(
    ("labels", "expected_class_count", "expected_bucket", "expected_effective"),
    [
        ((10, 11), 2, "<=8", 0.95),
        ((10, 20, 30, 40), 4, "<=8", 0.95),
    ],
)
def test_extra_trees_filter_sparse_labels_use_realized_class_count(
    labels: tuple[int, ...],
    expected_class_count: int,
    expected_bucket: str,
    expected_effective: float,
) -> None:
    x, y = _make_sparse_label_classification_data(labels=labels)
    _, details = apply_extra_trees_filter(
        x,
        y,
        task="classification",
        seed=123,
        n_estimators=6,
        max_depth=4,
        min_samples_leaf=2,
        n_bootstrap=16,
        threshold=0.95,
    )

    assert int(details["class_count"]) == expected_class_count
    assert details["class_bucket"] == expected_bucket
    assert details["threshold_requested"] == pytest.approx(0.95)
    assert details["threshold_effective"] == pytest.approx(expected_effective)
    assert details["threshold_delta"] == pytest.approx(0.95 - expected_effective)


@pytest.mark.parametrize("bad_seed", [-1, 4294967296])
def test_extra_trees_filter_rejects_out_of_range_seed(bad_seed: int) -> None:
    x, y = _make_regression_data(seed=9)
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        apply_extra_trees_filter(
            x,
            y,
            task="regression",
            seed=bad_seed,
            n_estimators=4,
            max_depth=3,
            n_bootstrap=8,
            threshold=0.5,
        )


@pytest.mark.parametrize("seed", [0, 4294967295])
def test_extra_trees_filter_accepts_32bit_seed_boundaries(seed: int) -> None:
    x, y = _make_regression_data(seed=19)
    accepted, details = apply_extra_trees_filter(
        x,
        y,
        task="regression",
        seed=seed,
        n_estimators=4,
        max_depth=3,
        n_bootstrap=8,
        threshold=0.5,
    )
    assert isinstance(accepted, bool)
    assert details["backend"] == "extra_trees_cpu"


def test_extra_trees_filter_honors_explicit_n_jobs() -> None:
    x, y = _make_regression_data(seed=1234)
    accepted, details = apply_extra_trees_filter(
        x,
        y,
        task="regression",
        seed=99,
        n_estimators=4,
        max_depth=3,
        n_bootstrap=8,
        threshold=0.5,
        n_jobs=1,
    )

    assert isinstance(accepted, bool)
    assert int(details["n_jobs"]) == 1


@pytest.mark.parametrize("bad_n_jobs", [0, -2, True])
def test_extra_trees_filter_rejects_invalid_n_jobs(bad_n_jobs: int | bool) -> None:
    x, y = _make_regression_data(seed=123)
    with pytest.raises(ValueError, match=r"n_jobs must be -1 or an integer >= 1"):
        apply_extra_trees_filter(
            x,
            y,
            task="regression",
            seed=42,
            n_estimators=4,
            max_depth=3,
            n_bootstrap=8,
            threshold=0.5,
            n_jobs=bad_n_jobs,
        )


@pytest.mark.parametrize("bad_threshold", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_extra_trees_filter_rejects_invalid_thresholds(bad_threshold: float | bool) -> None:
    x, y = _make_regression_data(seed=123)
    with pytest.raises(ValueError, match=r"threshold must be a finite value in \[0.0, 1.0\]"):
        apply_extra_trees_filter(
            x,
            y,
            task="regression",
            seed=42,
            n_estimators=4,
            max_depth=3,
            n_bootstrap=8,
            threshold=bad_threshold,
        )
