import numpy as np
import pytest
import torch

from dagzoo.config import GeneratorConfig
from dagzoo.core.dataset import generate_batch, generate_one
from dagzoo.core.metrics_torch import _compute_wins_ratio_proxy
from dagzoo.diagnostics import extract_dataset_metrics, extract_metrics_batch
from dagzoo.types import DatasetBundle


def _tiny_config(task: str) -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = task
    cfg.dataset.n_train = 64
    cfg.dataset.n_test = 32
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.dataset.n_classes_min = 2
    cfg.dataset.n_classes_max = 4
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    return cfg


def _tiny_shift_config(*, profile: str, scale_field: str, scale_value: float) -> GeneratorConfig:
    cfg = _tiny_config("regression")
    cfg.filter.enabled = False
    cfg.graph.n_nodes_min = 20
    cfg.graph.n_nodes_max = 20
    cfg.shift.enabled = True
    cfg.shift.mode = profile
    setattr(cfg.shift, scale_field, scale_value)
    return cfg


def test_extract_dataset_metrics_classification_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.75,
    )
    bundle = generate_one(_tiny_config("classification"), seed=7, device="cpu")

    metrics = extract_dataset_metrics(bundle, include_spearman=True)
    assert metrics.task == "classification"
    assert metrics.n_rows == bundle.X_train.shape[0] + bundle.X_test.shape[0]
    assert metrics.n_features == bundle.X_train.shape[1]
    assert metrics.n_classes is not None and metrics.n_classes >= 2
    assert metrics.class_entropy is not None and metrics.class_entropy >= 0.0
    assert metrics.majority_minority_ratio is not None and metrics.majority_minority_ratio >= 1.0
    assert metrics.linearity_proxy is not None and 0.0 <= metrics.linearity_proxy <= 1.0
    assert metrics.wins_ratio_proxy == pytest.approx(0.75)
    assert metrics.nonlinearity_proxy == pytest.approx(
        max(0.0, float(metrics.wins_ratio_proxy) - float(metrics.linearity_proxy))
    )
    assert 0.0 <= metrics.categorical_ratio <= 1.0
    assert metrics.graph_edge_density is not None and 0.0 <= metrics.graph_edge_density <= 1.0
    assert metrics.shift_enabled in {0.0, 1.0}
    assert metrics.shift_edge_odds_multiplier >= 1.0
    assert 0.0 <= metrics.shift_mechanism_nonlinear_mass <= 1.0
    assert metrics.shift_noise_variance_multiplier >= 1.0
    if metrics.pearson_abs_mean is not None:
        assert 0.0 <= metrics.pearson_abs_mean <= 1.0
    if metrics.pearson_abs_max is not None:
        assert 0.0 <= metrics.pearson_abs_max <= 1.0
    if metrics.spearman_abs_mean is not None:
        assert 0.0 <= metrics.spearman_abs_mean <= 1.0
    if metrics.spearman_abs_max is not None:
        assert 0.0 <= metrics.spearman_abs_max <= 1.0


def test_extract_dataset_metrics_regression_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.55,
    )
    bundle = generate_one(_tiny_config("regression"), seed=17, device="cpu")

    metrics = extract_dataset_metrics(bundle)
    assert metrics.task == "regression"
    assert metrics.n_classes is None
    assert metrics.class_entropy is None
    assert metrics.majority_minority_ratio is None
    assert metrics.linearity_proxy is not None and 0.0 <= metrics.linearity_proxy <= 1.0
    assert metrics.wins_ratio_proxy == pytest.approx(0.55)
    assert metrics.nonlinearity_proxy == pytest.approx(
        max(0.0, float(metrics.wins_ratio_proxy) - float(metrics.linearity_proxy))
    )
    assert metrics.snr_proxy_db is None or np.isfinite(metrics.snr_proxy_db)
    assert metrics.shift_enabled == pytest.approx(0.0)
    assert metrics.shift_graph_scale == pytest.approx(0.0)
    assert metrics.shift_mechanism_scale == pytest.approx(0.0)
    assert metrics.shift_variance_scale == pytest.approx(0.0)
    assert metrics.shift_edge_odds_multiplier == pytest.approx(1.0)
    assert metrics.shift_noise_variance_multiplier == pytest.approx(1.0)


def test_extract_dataset_metrics_spearman_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.61,
    )
    bundle = DatasetBundle(
        X_train=np.asarray([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0]], dtype=np.float32),
        y_train=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        X_test=np.asarray([[4.0, 50.0], [5.0, 60.0]], dtype=np.float32),
        y_test=np.asarray([4.0, 5.0], dtype=np.float32),
        feature_types=["num", "num"],
        metadata={"config": {"dataset": {"task": "regression"}}},
    )

    disabled = extract_dataset_metrics(bundle, include_spearman=False)
    enabled = extract_dataset_metrics(bundle, include_spearman=True)
    assert disabled.spearman_abs_mean is None
    assert disabled.spearman_abs_max is None
    assert enabled.spearman_abs_mean is not None
    assert enabled.spearman_abs_max is not None
    assert enabled.spearman_abs_mean == pytest.approx(1.0)
    assert enabled.spearman_abs_max == pytest.approx(1.0)


def test_extract_dataset_metrics_reproducible_for_fixed_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.42,
    )
    cfg = _tiny_config("classification")
    bundle_a = generate_one(cfg, seed=123, device="cpu")
    bundle_b = generate_one(cfg, seed=123, device="cpu")

    metrics_1 = extract_dataset_metrics(bundle_a)
    metrics_2 = extract_dataset_metrics(bundle_a)
    metrics_3 = extract_dataset_metrics(bundle_b)
    assert metrics_1 == metrics_2
    assert metrics_1 == metrics_3


def test_extract_dataset_metrics_exposes_shift_metadata_fields() -> None:
    cfg = _tiny_shift_config(profile="custom", scale_field="graph_scale", scale_value=0.7)
    cfg.shift.mechanism_scale = 0.4
    cfg.shift.variance_scale = 0.3

    bundle = generate_one(cfg, seed=3123, device="cpu")
    metrics = extract_dataset_metrics(bundle)
    shift_payload = bundle.metadata["shift"]

    assert metrics.shift_enabled == pytest.approx(1.0)
    assert metrics.shift_graph_scale == pytest.approx(float(shift_payload["graph_scale"]))
    assert metrics.shift_mechanism_scale == pytest.approx(float(shift_payload["mechanism_scale"]))
    assert metrics.shift_variance_scale == pytest.approx(float(shift_payload["variance_scale"]))
    assert metrics.shift_edge_odds_multiplier == pytest.approx(
        float(shift_payload["edge_odds_multiplier"])
    )
    assert metrics.shift_noise_variance_multiplier == pytest.approx(
        float(shift_payload["noise_variance_multiplier"])
    )
    assert metrics.shift_mechanism_nonlinear_mass == pytest.approx(
        float(shift_payload["mechanism_nonlinear_mass"])
    )


def test_extract_dataset_metrics_shift_profiles_move_in_expected_directions() -> None:
    baseline = _tiny_config("regression")
    baseline.filter.enabled = False
    baseline.graph.n_nodes_min = 20
    baseline.graph.n_nodes_max = 20

    graph_cfg = _tiny_shift_config(
        profile="graph_drift", scale_field="graph_scale", scale_value=1.0
    )
    mechanism_cfg = _tiny_shift_config(
        profile="mechanism_drift",
        scale_field="mechanism_scale",
        scale_value=1.0,
    )
    noise_cfg = _tiny_shift_config(
        profile="noise_drift", scale_field="variance_scale", scale_value=1.0
    )

    base_metrics = extract_dataset_metrics(generate_one(baseline, seed=3188, device="cpu"))
    graph_metrics = extract_dataset_metrics(generate_one(graph_cfg, seed=3188, device="cpu"))
    mechanism_metrics = extract_dataset_metrics(
        generate_one(mechanism_cfg, seed=3188, device="cpu")
    )
    noise_metrics = extract_dataset_metrics(generate_one(noise_cfg, seed=3188, device="cpu"))

    assert graph_metrics.graph_edge_density is not None
    assert base_metrics.graph_edge_density is not None
    assert graph_metrics.graph_edge_density >= base_metrics.graph_edge_density
    assert (
        mechanism_metrics.shift_mechanism_nonlinear_mass
        > base_metrics.shift_mechanism_nonlinear_mass
    )
    assert (
        noise_metrics.shift_noise_variance_multiplier > base_metrics.shift_noise_variance_multiplier
    )


def test_extract_dataset_metrics_handles_degenerate_constant_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: None,
    )
    bundle = DatasetBundle(
        X_train=np.ones((8, 3), dtype=np.float32),
        y_train=np.linspace(0.0, 1.0, num=8, dtype=np.float32),
        X_test=np.ones((4, 3), dtype=np.float32),
        y_test=np.linspace(1.0, 2.0, num=4, dtype=np.float32),
        feature_types=["num", "cat", "cat"],
        metadata={"config": {"dataset": {"task": "regression"}}},
    )

    metrics = extract_dataset_metrics(bundle, include_spearman=True)
    assert metrics.pearson_abs_mean is None
    assert metrics.pearson_abs_max is None
    assert metrics.spearman_abs_mean is None
    assert metrics.spearman_abs_max is None
    assert metrics.wins_ratio_proxy is None
    assert metrics.nonlinearity_proxy is None
    assert metrics.n_categorical_features == 2
    assert metrics.cat_cardinality_min == 1
    assert metrics.cat_cardinality_mean == pytest.approx(1.0)
    assert metrics.cat_cardinality_max == 1


def test_extract_metrics_batch_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._compute_wins_ratio_proxy",
        lambda **_kwargs: 0.6,
    )
    cfg = _tiny_config("classification")
    bundles = generate_batch(cfg, num_datasets=2, seed=900, device="cpu")

    metrics_batch = extract_metrics_batch(bundles)
    expected = [extract_dataset_metrics(bundle) for bundle in bundles]
    assert metrics_batch == expected


def test_extract_dataset_metrics_normalizes_bundle_to_cpu_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = DatasetBundle(
        X_train=torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32),
        y_train=torch.tensor([0.0, 1.0], dtype=torch.float32),
        X_test=torch.tensor([[4.0, 5.0]], dtype=torch.float32),
        y_test=torch.tensor([2.0], dtype=torch.float32),
        feature_types=["num", "num"],
        metadata={"config": {"dataset": {"task": "regression"}}},
    )

    captured: dict[str, object] = {}

    def _stub_extract(
        bundle_arg: DatasetBundle,
        *,
        target_metric_names: set[str],
        include_spearman: bool = False,
    ) -> dict[str, float | None | str]:
        captured["bundle"] = bundle_arg
        captured["target_metric_names"] = set(target_metric_names)
        captured["include_spearman"] = include_spearman
        return {
            "task": "regression",
            "n_rows": 3.0,
            "n_features": 2.0,
            "n_classes": None,
            "n_categorical_features": 0.0,
            "categorical_ratio": 0.0,
            "graph_edge_density": 0.3,
            "shift_enabled": 1.0,
            "shift_graph_scale": 0.4,
            "shift_mechanism_scale": 0.2,
            "shift_variance_scale": 0.1,
            "shift_edge_odds_multiplier": 1.2,
            "shift_mechanism_nonlinear_mass": 0.7,
            "shift_noise_variance_multiplier": 1.1,
            "linearity_proxy": 0.25,
            "nonlinearity_proxy": 0.0,
            "wins_ratio_proxy": 0.25,
            "pearson_abs_mean": 0.1,
            "pearson_abs_max": 0.2,
            "spearman_abs_mean": None,
            "spearman_abs_max": None,
            "class_entropy": None,
            "majority_minority_ratio": None,
            "snr_proxy_db": 1.0,
            "cat_cardinality_min": None,
            "cat_cardinality_mean": None,
            "cat_cardinality_max": None,
        }

    monkeypatch.setattr("dagzoo.diagnostics.metrics.extract_torch_metrics", _stub_extract)

    metrics = extract_dataset_metrics(bundle, include_spearman=False)
    extracted_bundle = captured["bundle"]
    assert isinstance(extracted_bundle, DatasetBundle)
    assert extracted_bundle is not bundle
    assert isinstance(extracted_bundle.X_train, torch.Tensor)
    assert isinstance(extracted_bundle.y_train, torch.Tensor)
    assert isinstance(extracted_bundle.X_test, torch.Tensor)
    assert isinstance(extracted_bundle.y_test, torch.Tensor)
    assert extracted_bundle.X_train.device.type == "cpu"
    assert extracted_bundle.y_train.device.type == "cpu"
    assert extracted_bundle.X_test.device.type == "cpu"
    assert extracted_bundle.y_test.device.type == "cpu"
    assert isinstance(captured["target_metric_names"], set)
    assert "spearman_abs_mean" not in captured["target_metric_names"]
    assert captured["include_spearman"] is False
    assert metrics.n_rows == 3
    assert metrics.n_features == 2
    assert metrics.graph_edge_density == pytest.approx(0.3)
    assert metrics.shift_enabled == pytest.approx(1.0)
    assert metrics.shift_edge_odds_multiplier == pytest.approx(1.2)


def test_compute_wins_ratio_proxy_uses_bootstrap_wins_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
    y = x[:, 0].clone()

    def _stub_fit(*, x_test: np.ndarray, **_kwargs) -> np.ndarray:
        return np.asarray(x_test[:, :1], dtype=np.float32)

    monkeypatch.setattr(
        "dagzoo.core.metrics_torch._fit_extra_trees_predictions",
        _stub_fit,
    )

    wins_ratio = _compute_wins_ratio_proxy(x=x, y=y, task="regression")

    assert wins_ratio == pytest.approx(1.0)
