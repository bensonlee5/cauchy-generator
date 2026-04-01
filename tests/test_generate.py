import math
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import load_repo_config

import dagzoo
import dagzoo.core
from dagzoo.config import (
    NOISE_FAMILY_GAUSSIAN,
    NOISE_FAMILY_LAPLACE,
    NOISE_FAMILY_MIXTURE,
    NOISE_FAMILY_STUDENT_T,
    GeneratorConfig,
)
from dagzoo.core.dataset import (
    generate_batch,
    generate_batch_iter,
    generate_one,
)
from dagzoo.core.fixed_layout.metadata import (
    _FixedLayoutPlan,
    _layout_signature,
)
from dagzoo.core.fixed_layout.plan_types import FixedLayoutExecutionPlan
from dagzoo.core.fixed_layout.runtime import (
    _fixed_layout_plan_classification_attempt_plan,
    _fixed_layout_plan_supports_classification_run,
    _generate_batch_with_plan_iter,
    _generate_fixed_layout_bundle_with_retries,
    _replay_emitted_fixed_layout_plan,
    _resolve_fixed_layout_batch_size,
    _sample_fixed_layout,
    _sample_fixed_layout_candidate,
    prepare_canonical_fixed_layout_run,
)
from dagzoo.core.generation_runtime import (
    _attach_teacher_conditionals_for_test_split,
    _build_fixed_schema_finalization_context,
    _finalize_generated_chunk_preserve_schema,
    _finalize_generated_tensors,
    _remap_teacher_conditionals_to_emitted_labels,
    _resolve_split_indices,
    _teacher_conditionals_metadata,
)
from dagzoo.core.layout import _sample_layout
from dagzoo.core.layout_types import LayoutPlan
from dagzoo.core.noise_runtime import (
    NoiseRuntimeSelection,
    _build_noise_distribution_metadata,
    _resolve_noise_runtime_selection,
)
from dagzoo.core.shift import mechanism_nonlinear_mass, resolve_shift_runtime_params
from dagzoo.core.validation import (
    InfeasibleStratifiedSplitError,
    _classification_split_valid,
    _stratified_split_indices,
)
from dagzoo.io.lineage_schema import (
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION,
    validate_lineage_payload,
    validate_metadata_lineage,
)
from dagzoo.rng import KeyedRng
from dagzoo.types import DatasetBundle


def _tiny_config() -> GeneratorConfig:
    cfg = load_repo_config()
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    return cfg


def test_public_package_hides_explicit_fixed_layout_exports() -> None:
    assert "sample_fixed_layout" not in dagzoo.__all__
    assert "generate_batch_fixed_layout" not in dagzoo.__all__
    assert "generate_batch_fixed_layout_iter" not in dagzoo.__all__
    assert "FixedLayoutPlan" not in dagzoo.__all__
    assert "sample_fixed_layout" not in dagzoo.core.__all__
    assert "generate_batch_fixed_layout" not in dagzoo.core.__all__
    assert "generate_batch_fixed_layout_iter" not in dagzoo.core.__all__
    assert "FixedLayoutPlan" not in dagzoo.core.__all__


def _tiny_regression_config() -> GeneratorConfig:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.filter.enabled = False
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    return cfg


def _layout_stub(
    *,
    feature_types: list[str],
    graph_nodes: int,
    adjacency: torch.Tensor,
    feature_node_assignment: list[int],
    target_node_assignment: int | None = None,
) -> LayoutPlan:
    _ = target_node_assignment
    graph_edges = int(adjacency.to(dtype=torch.int64).sum().item())
    n_features = len(feature_types)
    cat_idx = [idx for idx, kind in enumerate(feature_types) if kind == "cat"]
    card_by_feature = {idx: 4 for idx in cat_idx}
    density_denominator = graph_nodes * max(graph_nodes - 1, 1)
    graph_edge_density = float(graph_edges) / float(density_denominator) if graph_nodes > 1 else 0.0
    return LayoutPlan(
        n_features=n_features,
        n_cat=len(cat_idx),
        cat_idx=cat_idx,
        cardinalities=[4 for _ in cat_idx],
        card_by_feature=card_by_feature,
        n_classes=3,
        feature_types=list(feature_types),
        graph_nodes=int(graph_nodes),
        graph_edges=graph_edges,
        graph_depth_nodes=int(graph_nodes),
        graph_edge_density=graph_edge_density,
        adjacency=adjacency,
        feature_node_assignment=list(feature_node_assignment),
        target_parent_features=list(range(n_features)),
        target_parent_prior="all_features",
        target_parent_regime="all_features",
        target_parent_sqrt_threshold=max(
            1,
            min(n_features, int(math.floor(math.sqrt(max(1, n_features))))),
        ),
    )


def test_generate_one_shapes() -> None:
    cfg = _tiny_regression_config()
    bundle = generate_one(cfg, seed=7, device="cpu")
    assert isinstance(bundle.X_train, torch.Tensor)
    assert bundle.X_train.shape[0] == cfg.dataset.n_train
    assert bundle.X_test.shape[0] == cfg.dataset.n_test
    assert bundle.X_train.shape[1] == bundle.X_test.shape[1]
    assert len(bundle.feature_types) == bundle.X_train.shape[1]


def test_generate_one_uses_fixed_dataset_rows_and_updates_metadata_config_split() -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.rows = 1024  # type: ignore[assignment]
    cfg.dataset.n_test = 256
    cfg.dataset.n_train = 32

    bundle = generate_one(cfg, seed=8, device="cpu")
    assert int(bundle.X_train.shape[0]) == 768
    assert int(bundle.X_test.shape[0]) == 256
    assert int(bundle.metadata["config"]["dataset"]["n_train"]) == 768
    assert int(bundle.metadata["config"]["dataset"]["n_test"]) == 256


def test_generate_one_omits_unset_fixed_layout_target_cells_from_metadata_config() -> None:
    cfg = _tiny_regression_config()
    assert cfg.runtime.fixed_layout_target_cells is None

    bundle = generate_one(cfg, seed=9, device="cpu")

    runtime_config = bundle.metadata["config"]["runtime"]
    assert "fixed_layout_target_cells" not in runtime_config


def test_generate_one_omits_steering_from_metadata_config() -> None:
    cfg = _tiny_regression_config()
    cfg.steering.enabled = True
    cfg.steering.preset = "anti_memorization_piecewise_v1"
    cfg.validate_generation_constraints()

    bundle = generate_one(cfg, seed=10, device="cpu")

    assert "steering" not in bundle.metadata["config"]


def test_generate_one_with_stress_profile_omits_stress_from_metadata_config() -> None:
    cfg = load_repo_config()
    cfg.stress.profile = "anti_memorization_piecewise_classification_slice_v1"

    bundle = generate_one(cfg, seed=10, device="cpu")

    assert bundle.metadata["config"]["dataset"]["task"] == "classification"
    assert int(bundle.metadata["config"]["dataset"]["n_train"]) == 768
    assert int(bundle.metadata["config"]["dataset"]["n_test"]) == 256
    assert "stress" not in bundle.metadata["config"]
    assert "steering" not in bundle.metadata["config"]


def test_generate_batch_dynamic_steering_changes_metadata_over_dataset_order() -> None:
    cfg = _tiny_regression_config()
    cfg.steering.enabled = True
    cfg.steering.preset = "anti_memorization_piecewise_v1"
    cfg.validate_generation_constraints()

    batch = generate_batch(cfg, num_datasets=5, seed=1234, device="cpu")

    assert [int(bundle.metadata["dataset_index"]) for bundle in batch] == [0, 1, 2, 3, 4]
    assert "missingness" not in batch[0].metadata
    assert batch[1].metadata["shift"]["mode"] == "graph_drift"
    assert batch[2].metadata["shift"]["mode"] == "mixed"
    assert batch[2].metadata["shift"]["graph_scale"] == pytest.approx(0.5)
    assert batch[2].metadata["shift"]["variance_scale"] == pytest.approx(0.0)
    assert batch[3].metadata["shift"]["graph_scale"] == pytest.approx(0.0)
    assert batch[3].metadata["shift"]["variance_scale"] == pytest.approx(0.5)
    assert batch[3].metadata["noise_distribution"]["family_requested"] == NOISE_FAMILY_MIXTURE
    assert batch[4].metadata["noise_distribution"]["family_requested"] == NOISE_FAMILY_MIXTURE
    assert batch[4].metadata["noise_distribution"]["mixture_weights"] == {
        "gaussian": pytest.approx(0.5),
        "laplace": pytest.approx(0.3),
        "student_t": pytest.approx(0.2),
    }


def test_generate_batch_dynamic_steering_graph_stage_can_change_layout_signatures() -> None:
    cfg = _tiny_regression_config()
    cfg.steering.enabled = True
    cfg.steering.preset = "anti_memorization_piecewise_v1"
    cfg.validate_generation_constraints()

    batch = generate_batch(cfg, num_datasets=5, seed=1, device="cpu")

    layout_signatures = [str(bundle.metadata["layout_signature"]) for bundle in batch]
    feature_type_signatures = [
        tuple(str(value) for value in bundle.feature_types) for bundle in batch
    ]
    feature_counts = [int(bundle.metadata["n_features"]) for bundle in batch]

    assert len(set(layout_signatures)) >= 2
    assert len(set(feature_type_signatures)) == 1
    assert len(set(feature_counts)) == 1


def test_generate_batch_with_plan_iter_batches_steering_missingness_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.steering.enabled = True
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=901,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )
    grouped_chunk_sizes: list[int] = []
    seen_missing_rates: list[list[float]] = []

    def _missingness_config(missing_rate: float) -> GeneratorConfig:
        effective = deepcopy(cfg)
        effective.steering.enabled = False
        effective.dataset.missing_rate = float(missing_rate)
        effective.dataset.missing_mechanism = "none" if missing_rate <= 0.0 else "mcar"
        effective.validate_generation_constraints()
        return effective

    def _make_bundle(seed: int, missing_rate: float) -> DatasetBundle:
        return DatasetBundle(
            X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
            y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.float32),
            X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
            y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.float32),
            feature_types=["num", "num"],
            metadata={
                "seed": int(seed),
                "n_features": 2,
                "config": {"dataset": {"missing_rate": float(missing_rate)}},
                "lineage": {
                    "assignments": {
                        "feature_to_node": [0, 1],
                        "target_to_node": 1,
                        "target_parent_features": [0, 1],
                        "target_parent_count": 2,
                        "target_parent_fraction": 1.0,
                        "target_parent_prior": "all_features",
                        "target_parent_regime": "all_features",
                        "target_parent_sqrt_threshold": 1,
                    }
                },
                "filter": {"mode": "deferred", "status": "not_run"},
                "generation_attempts": {
                    "total_attempts": 1,
                    "retry_count": 0,
                    "filter_attempts": 0,
                    "filter_rejections": 0,
                    "filter_rejection_rate": None,
                },
            },
        )

    def _stub_resolve_steered_dataset_descriptor(
        _config: GeneratorConfig,
        *,
        base_plan: _FixedLayoutPlan,
        dataset_index: int,
        num_datasets: int,
        dataset_root: KeyedRng,
    ):
        _ = num_datasets
        missing_rate = [0.0, 0.1, 0.2][dataset_index]
        effective_config = _missingness_config(missing_rate)
        return SimpleNamespace(
            dataset_index=dataset_index,
            dataset_root=dataset_root,
            effective_config=effective_config,
            effective_plan=base_plan,
            effective_shift=resolve_shift_runtime_params(effective_config),
            finalization_context=_build_fixed_schema_finalization_context(
                effective_config,
                base_plan.layout,
                n_train=cfg.dataset.n_train,
                n_test=cfg.dataset.n_test,
                shift_params=resolve_shift_runtime_params(effective_config),
            ),
        )

    def _stub_group_noise_runtime_chunk(
        _config: GeneratorConfig,
        *,
        dataset_roots: list[KeyedRng],
        attempts: list[int] | None = None,
    ):
        grouped_chunk_sizes.append(len(dataset_roots))
        assert attempts == [0] * len(dataset_roots)
        return [
            SimpleNamespace(
                chunk_offsets=list(range(len(dataset_roots))),
                generation_seeds=[
                    dataset_root.keyed("attempt", 0, "raw_generation").child_seed()
                    for dataset_root in dataset_roots
                ],
                selection=NoiseRuntimeSelection(
                    family_requested="gaussian",
                    family_sampled="gaussian",
                    sampling_strategy="dataset_level",
                    base_scale=1.0,
                    student_t_df=5.0,
                    mixture_weights=None,
                ),
                attempt=0,
            )
        ]

    def _stub_generate_grouped_raw_batches(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan: FixedLayoutExecutionPlan,
        grouped_noise_runtime,
        requested_device: str,
        resolved_device: str,
        noise_sigma_multiplier: float,
    ) -> list[SimpleNamespace]:
        _ = execution_plan
        _ = requested_device
        _ = resolved_device
        _ = noise_sigma_multiplier
        group = grouped_noise_runtime[0]
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return [
            SimpleNamespace(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=group.attempt,
                x_batch=torch.zeros((len(group.chunk_offsets), n_rows, 2), dtype=torch.float32),
                y_batch=torch.zeros((len(group.chunk_offsets), n_rows), dtype=torch.float32),
                aux_meta_batch=[
                    {"filter": {"mode": "deferred", "status": "not_run"}}
                    for _ in group.chunk_offsets
                ],
                effective_resolved_device="cpu",
                device_fallback_reason=None,
                runtime_metrics={},
            )
        ]

    def _stub_finalize_generated_chunk_preserve_schema(
        _config: GeneratorConfig,
        _layout,
        *,
        context,
        contexts_by_batch=None,
        configs_by_batch=None,
        dataset_roots: list[KeyedRng],
        attempt: int,
        attempts_used: int,
        device: str,
        n_train: int,
        n_test: int,
        requested_device: str,
        resolved_device: str,
        device_fallback_reason: str | None,
        x: torch.Tensor,
        y: torch.Tensor,
        aux_meta_batch: list[dict[str, object]],
        noise_runtime_selection: NoiseRuntimeSelection,
        dtype: torch.dtype,
        resolved_split_indices=None,
    ) -> list[DatasetBundle | None]:
        _ = context
        _ = contexts_by_batch
        _ = attempt
        _ = attempts_used
        _ = device
        _ = n_train
        _ = n_test
        _ = requested_device
        _ = resolved_device
        _ = device_fallback_reason
        _ = x
        _ = y
        _ = aux_meta_batch
        _ = noise_runtime_selection
        _ = dtype
        _ = resolved_split_indices
        assert configs_by_batch is not None
        missing_rates = [float(entry.dataset.missing_rate) for entry in configs_by_batch]
        seen_missing_rates.append(missing_rates)
        return [
            _make_bundle(
                dataset_root.child_seed(), float(configs_by_batch[index].dataset.missing_rate)
            )
            for index, dataset_root in enumerate(dataset_roots)
        ]

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._resolve_steered_dataset_descriptor",
        _stub_resolve_steered_dataset_descriptor,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._group_noise_runtime_chunk",
        _stub_group_noise_runtime_chunk,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_grouped_raw_batches",
        _stub_generate_grouped_raw_batches,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_chunk_preserve_schema",
        _stub_finalize_generated_chunk_preserve_schema,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_fixed_layout_bundle_with_retries",
        lambda *_args, **_kwargs: pytest.fail(
            "steering missingness-only batching should not fall back to scalar retries"
        ),
    )

    bundles = list(
        _generate_batch_with_plan_iter(
            cfg,
            plan=plan,
            num_datasets=3,
            seed=33,
            batch_size=3,
        )
    )

    assert grouped_chunk_sizes == [3]
    assert seen_missing_rates == [[0.0, 0.1, 0.2]]
    assert [float(bundle.metadata["config"]["dataset"]["missing_rate"]) for bundle in bundles] == [
        pytest.approx(0.0),
        pytest.approx(0.1),
        pytest.approx(0.2),
    ]


def test_generate_batch_with_plan_iter_batches_noise_only_steering_by_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.steering.enabled = True
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=902,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )
    grouped_chunk_sizes: list[int] = []
    grouped_families: list[str] = []

    def _noise_config(index: int) -> GeneratorConfig:
        effective = deepcopy(cfg)
        effective.steering.enabled = False
        if index < 2:
            effective.noise.family = "gaussian"
            effective.noise.mixture_weights = None
        else:
            effective.noise.family = "student_t"
            effective.noise.student_t_df = 7.0
            effective.noise.mixture_weights = None
        effective.validate_generation_constraints()
        return effective

    def _make_bundle(seed: int) -> DatasetBundle:
        return DatasetBundle(
            X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
            y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.float32),
            X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
            y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.float32),
            feature_types=["num", "num"],
            metadata={
                "seed": int(seed),
                "n_features": 2,
                "lineage": {
                    "assignments": {
                        "feature_to_node": [0, 1],
                        "target_to_node": 1,
                        "target_parent_features": [0, 1],
                        "target_parent_count": 2,
                        "target_parent_fraction": 1.0,
                        "target_parent_prior": "all_features",
                        "target_parent_regime": "all_features",
                        "target_parent_sqrt_threshold": 1,
                    }
                },
                "filter": {"mode": "deferred", "status": "not_run"},
                "generation_attempts": {
                    "total_attempts": 1,
                    "retry_count": 0,
                    "filter_attempts": 0,
                    "filter_rejections": 0,
                    "filter_rejection_rate": None,
                },
            },
        )

    def _stub_resolve_steered_dataset_descriptor(
        _config: GeneratorConfig,
        *,
        base_plan: _FixedLayoutPlan,
        dataset_index: int,
        num_datasets: int,
        dataset_root: KeyedRng,
    ):
        _ = num_datasets
        effective_config = _noise_config(dataset_index)
        return SimpleNamespace(
            dataset_index=dataset_index,
            dataset_root=dataset_root,
            effective_config=effective_config,
            effective_plan=base_plan,
            effective_shift=resolve_shift_runtime_params(effective_config),
            finalization_context=_build_fixed_schema_finalization_context(
                effective_config,
                base_plan.layout,
                n_train=cfg.dataset.n_train,
                n_test=cfg.dataset.n_test,
                shift_params=resolve_shift_runtime_params(effective_config),
            ),
        )

    def _stub_group_noise_runtime_chunk(
        config: GeneratorConfig,
        *,
        dataset_roots: list[KeyedRng],
        attempts: list[int] | None = None,
    ):
        grouped_chunk_sizes.append(len(dataset_roots))
        grouped_families.append(str(config.noise.family))
        assert attempts == [0] * len(dataset_roots)
        return [
            SimpleNamespace(
                chunk_offsets=list(range(len(dataset_roots))),
                generation_seeds=[
                    dataset_root.keyed("attempt", 0, "raw_generation").child_seed()
                    for dataset_root in dataset_roots
                ],
                selection=NoiseRuntimeSelection(
                    family_requested=config.noise.family,
                    family_sampled=config.noise.family,
                    sampling_strategy="dataset_level",
                    base_scale=float(config.noise.base_scale),
                    student_t_df=float(config.noise.student_t_df),
                    mixture_weights=None,
                ),
                attempt=0,
            )
        ]

    def _stub_generate_grouped_raw_batches(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan: FixedLayoutExecutionPlan,
        grouped_noise_runtime,
        requested_device: str,
        resolved_device: str,
        noise_sigma_multiplier: float,
    ) -> list[SimpleNamespace]:
        _ = execution_plan
        _ = requested_device
        _ = resolved_device
        _ = noise_sigma_multiplier
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return [
            SimpleNamespace(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=group.attempt,
                x_batch=torch.zeros((len(group.chunk_offsets), n_rows, 2), dtype=torch.float32),
                y_batch=torch.zeros((len(group.chunk_offsets), n_rows), dtype=torch.float32),
                aux_meta_batch=[
                    {"filter": {"mode": "deferred", "status": "not_run"}}
                    for _ in group.chunk_offsets
                ],
                effective_resolved_device="cpu",
                device_fallback_reason=None,
                runtime_metrics={},
            )
            for group in grouped_noise_runtime
        ]

    def _stub_finalize_generated_chunk_preserve_schema(
        _config: GeneratorConfig,
        _layout,
        *,
        context,
        contexts_by_batch=None,
        configs_by_batch=None,
        dataset_roots: list[KeyedRng],
        attempt: int,
        attempts_used: int,
        device: str,
        n_train: int,
        n_test: int,
        requested_device: str,
        resolved_device: str,
        device_fallback_reason: str | None,
        x: torch.Tensor,
        y: torch.Tensor,
        aux_meta_batch: list[dict[str, object]],
        noise_runtime_selection: NoiseRuntimeSelection,
        dtype: torch.dtype,
        resolved_split_indices=None,
    ) -> list[DatasetBundle | None]:
        _ = (
            context,
            contexts_by_batch,
            configs_by_batch,
            attempt,
            attempts_used,
            device,
            n_train,
            n_test,
            requested_device,
            resolved_device,
            device_fallback_reason,
            x,
            y,
            aux_meta_batch,
            noise_runtime_selection,
            dtype,
            resolved_split_indices,
        )
        return [_make_bundle(dataset_root.child_seed()) for dataset_root in dataset_roots]

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._resolve_steered_dataset_descriptor",
        _stub_resolve_steered_dataset_descriptor,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._group_noise_runtime_chunk",
        _stub_group_noise_runtime_chunk,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_grouped_raw_batches",
        _stub_generate_grouped_raw_batches,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_chunk_preserve_schema",
        _stub_finalize_generated_chunk_preserve_schema,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_fixed_layout_bundle_with_retries",
        lambda *_args, **_kwargs: pytest.fail(
            "noise-only steering cohorts should stay on the grouped batch path"
        ),
    )

    bundles = list(
        _generate_batch_with_plan_iter(
            cfg,
            plan=plan,
            num_datasets=4,
            seed=44,
            batch_size=4,
        )
    )

    assert len(bundles) == 4
    assert grouped_chunk_sizes == [2, 2]
    assert grouped_families == ["gaussian", "student_t"]


def test_generate_batch_with_plan_iter_batches_graph_steering_by_effective_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.steering.enabled = True
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    layout_a = _layout_stub(
        feature_types=["num", "num"],
        graph_nodes=2,
        adjacency=torch.zeros((2, 2), dtype=torch.bool),
        feature_node_assignment=[0, 1],
        target_node_assignment=1,
    )
    layout_b = _layout_stub(
        feature_types=["num", "num"],
        graph_nodes=2,
        adjacency=torch.tensor([[0, 1], [0, 0]], dtype=torch.bool),
        feature_node_assignment=[0, 1],
        target_node_assignment=1,
    )
    plan_a = _FixedLayoutPlan(
        layout=layout_a,
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=903,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature=_layout_signature(layout_a),
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig_a",
    )
    plan_b = _FixedLayoutPlan(
        layout=layout_b,
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=904,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature=_layout_signature(layout_b),
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig_b",
    )
    grouped_chunk_sizes: list[int] = []

    def _make_bundle(seed: int) -> DatasetBundle:
        return DatasetBundle(
            X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
            y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.float32),
            X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
            y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.float32),
            feature_types=["num", "num"],
            metadata={
                "seed": int(seed),
                "n_features": 2,
                "lineage": {
                    "assignments": {
                        "feature_to_node": [0, 1],
                        "target_to_node": 1,
                        "target_parent_features": [0, 1],
                        "target_parent_count": 2,
                        "target_parent_fraction": 1.0,
                        "target_parent_prior": "all_features",
                        "target_parent_regime": "all_features",
                        "target_parent_sqrt_threshold": 1,
                    }
                },
                "filter": {"mode": "deferred", "status": "not_run"},
                "generation_attempts": {
                    "total_attempts": 1,
                    "retry_count": 0,
                    "filter_attempts": 0,
                    "filter_rejections": 0,
                    "filter_rejection_rate": None,
                },
            },
        )

    def _stub_resolve_steered_dataset_descriptor(
        _config: GeneratorConfig,
        *,
        base_plan: _FixedLayoutPlan,
        dataset_index: int,
        num_datasets: int,
        dataset_root: KeyedRng,
    ):
        _ = base_plan
        _ = num_datasets
        effective_config = deepcopy(cfg)
        effective_config.steering.enabled = False
        effective_config.validate_generation_constraints()
        effective_plan = plan_a if dataset_index < 2 else plan_b
        return SimpleNamespace(
            dataset_index=dataset_index,
            dataset_root=dataset_root,
            effective_config=effective_config,
            effective_plan=effective_plan,
            effective_shift=resolve_shift_runtime_params(effective_config),
            finalization_context=_build_fixed_schema_finalization_context(
                effective_config,
                effective_plan.layout,
                n_train=cfg.dataset.n_train,
                n_test=cfg.dataset.n_test,
                shift_params=resolve_shift_runtime_params(effective_config),
            ),
        )

    def _stub_group_noise_runtime_chunk(
        _config: GeneratorConfig,
        *,
        dataset_roots: list[KeyedRng],
        attempts: list[int] | None = None,
    ):
        grouped_chunk_sizes.append(len(dataset_roots))
        assert attempts == [0] * len(dataset_roots)
        return [
            SimpleNamespace(
                chunk_offsets=list(range(len(dataset_roots))),
                generation_seeds=[
                    dataset_root.keyed("attempt", 0, "raw_generation").child_seed()
                    for dataset_root in dataset_roots
                ],
                selection=NoiseRuntimeSelection(
                    family_requested="gaussian",
                    family_sampled="gaussian",
                    sampling_strategy="dataset_level",
                    base_scale=1.0,
                    student_t_df=5.0,
                    mixture_weights=None,
                ),
                attempt=0,
            )
        ]

    def _stub_generate_grouped_raw_batches(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan: FixedLayoutExecutionPlan,
        grouped_noise_runtime,
        requested_device: str,
        resolved_device: str,
        noise_sigma_multiplier: float,
    ) -> list[SimpleNamespace]:
        _ = execution_plan
        _ = requested_device
        _ = resolved_device
        _ = noise_sigma_multiplier
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return [
            SimpleNamespace(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=group.attempt,
                x_batch=torch.zeros((len(group.chunk_offsets), n_rows, 2), dtype=torch.float32),
                y_batch=torch.zeros((len(group.chunk_offsets), n_rows), dtype=torch.float32),
                aux_meta_batch=[
                    {"filter": {"mode": "deferred", "status": "not_run"}}
                    for _ in group.chunk_offsets
                ],
                effective_resolved_device="cpu",
                device_fallback_reason=None,
                runtime_metrics={},
            )
            for group in grouped_noise_runtime
        ]

    def _stub_finalize_generated_chunk_preserve_schema(
        _config: GeneratorConfig,
        _layout,
        *,
        context,
        contexts_by_batch=None,
        configs_by_batch=None,
        dataset_roots: list[KeyedRng],
        attempt: int,
        attempts_used: int,
        device: str,
        n_train: int,
        n_test: int,
        requested_device: str,
        resolved_device: str,
        device_fallback_reason: str | None,
        x: torch.Tensor,
        y: torch.Tensor,
        aux_meta_batch: list[dict[str, object]],
        noise_runtime_selection: NoiseRuntimeSelection,
        dtype: torch.dtype,
        resolved_split_indices=None,
    ) -> list[DatasetBundle | None]:
        _ = (
            context,
            contexts_by_batch,
            configs_by_batch,
            attempt,
            attempts_used,
            device,
            n_train,
            n_test,
            requested_device,
            resolved_device,
            device_fallback_reason,
            x,
            y,
            aux_meta_batch,
            noise_runtime_selection,
            dtype,
            resolved_split_indices,
        )
        return [_make_bundle(dataset_root.child_seed()) for dataset_root in dataset_roots]

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._resolve_steered_dataset_descriptor",
        _stub_resolve_steered_dataset_descriptor,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._group_noise_runtime_chunk",
        _stub_group_noise_runtime_chunk,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_grouped_raw_batches",
        _stub_generate_grouped_raw_batches,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_chunk_preserve_schema",
        _stub_finalize_generated_chunk_preserve_schema,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_fixed_layout_bundle_with_retries",
        lambda *_args, **_kwargs: pytest.fail(
            "graph-steered cohorts should stay on the grouped batch path"
        ),
    )

    bundles = list(
        _generate_batch_with_plan_iter(
            cfg,
            plan=plan_a,
            num_datasets=4,
            seed=55,
            batch_size=4,
        )
    )

    assert grouped_chunk_sizes == [2, 2]
    assert [bundle.metadata["layout_signature"] for bundle in bundles] == [
        str(plan_a.layout_signature),
        str(plan_a.layout_signature),
        str(plan_b.layout_signature),
        str(plan_b.layout_signature),
    ]


def test_generate_batch_with_plan_iter_classification_steering_captures_split_failures_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.task = "classification"
    cfg.steering.enabled = True
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    cfg.dataset.n_classes_min = 2
    cfg.dataset.n_classes_max = 2
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=905,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )
    raw_metric_calls: list[dict[str, float]] = []
    fallback_start_attempts: list[int] = []

    def _make_bundle(seed: int) -> DatasetBundle:
        return DatasetBundle(
            X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
            y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.int64),
            X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
            y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.int64),
            feature_types=["num", "num"],
            metadata={
                "seed": int(seed),
                "n_features": 2,
                "lineage": {
                    "assignments": {
                        "feature_to_node": [0, 1],
                        "target_to_node": 1,
                        "target_parent_features": [0, 1],
                        "target_parent_count": 2,
                        "target_parent_fraction": 1.0,
                        "target_parent_prior": "all_features",
                        "target_parent_regime": "all_features",
                        "target_parent_sqrt_threshold": 1,
                    }
                },
                "filter": {"mode": "deferred", "status": "not_run"},
                "generation_attempts": {
                    "total_attempts": 1,
                    "retry_count": 0,
                    "filter_attempts": 0,
                    "filter_rejections": 0,
                    "filter_rejection_rate": None,
                },
            },
        )

    def _stub_resolve_steered_dataset_descriptor(
        _config: GeneratorConfig,
        *,
        base_plan: _FixedLayoutPlan,
        dataset_index: int,
        num_datasets: int,
        dataset_root: KeyedRng,
    ):
        _ = base_plan
        _ = num_datasets
        effective_config = deepcopy(cfg)
        effective_config.steering.enabled = False
        effective_config.validate_generation_constraints()
        return SimpleNamespace(
            dataset_index=dataset_index,
            dataset_root=dataset_root,
            effective_config=effective_config,
            effective_plan=plan,
            effective_shift=resolve_shift_runtime_params(effective_config),
            finalization_context=_build_fixed_schema_finalization_context(
                effective_config,
                plan.layout,
                n_train=cfg.dataset.n_train,
                n_test=cfg.dataset.n_test,
                shift_params=resolve_shift_runtime_params(effective_config),
            ),
        )

    def _stub_group_noise_runtime_chunk(
        _config: GeneratorConfig,
        *,
        dataset_roots: list[KeyedRng],
        attempts: list[int] | None = None,
    ):
        assert attempts == [0] * len(dataset_roots)
        return [
            SimpleNamespace(
                chunk_offsets=list(range(len(dataset_roots))),
                generation_seeds=[
                    dataset_root.keyed("attempt", 0, "raw_generation").child_seed()
                    for dataset_root in dataset_roots
                ],
                selection=NoiseRuntimeSelection(
                    family_requested="gaussian",
                    family_sampled="gaussian",
                    sampling_strategy="dataset_level",
                    base_scale=1.0,
                    student_t_df=5.0,
                    mixture_weights=None,
                ),
                attempt=0,
            )
        ]

    def _stub_generate_grouped_raw_batches(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan: FixedLayoutExecutionPlan,
        grouped_noise_runtime,
        requested_device: str,
        resolved_device: str,
        noise_sigma_multiplier: float,
    ) -> list[SimpleNamespace]:
        _ = execution_plan
        _ = requested_device
        _ = resolved_device
        _ = noise_sigma_multiplier
        group = grouped_noise_runtime[0]
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return [
            SimpleNamespace(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=group.attempt,
                x_batch=torch.zeros((2, n_rows, 2), dtype=torch.float32),
                y_batch=torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1],
                    ],
                    dtype=torch.int64,
                ),
                aux_meta_batch=[
                    {"filter": {"mode": "deferred", "status": "not_run"}}
                    for _ in group.chunk_offsets
                ],
                effective_resolved_device="cpu",
                device_fallback_reason=None,
                runtime_metrics={"grouped_batch_count": 1.0},
            )
        ]

    def _stub_resolve_split_indices(
        y: torch.Tensor,
        *,
        task: str,
        n_train: int,
        keyed_rng: KeyedRng,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = keyed_rng
        assert task == "classification"
        assert n_train == cfg.dataset.n_train
        if int(y[0].item()) == 1:
            raise InfeasibleStratifiedSplitError("invalid split")
        return (
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            torch.tensor([4, 5], dtype=torch.int64),
        )

    def _stub_finalize_generated_chunk_preserve_schema(
        _config: GeneratorConfig,
        _layout,
        *,
        context,
        contexts_by_batch=None,
        configs_by_batch=None,
        dataset_roots: list[KeyedRng],
        attempt: int,
        attempts_used: int,
        device: str,
        n_train: int,
        n_test: int,
        requested_device: str,
        resolved_device: str,
        device_fallback_reason: str | None,
        x: torch.Tensor,
        y: torch.Tensor,
        aux_meta_batch: list[dict[str, object]],
        noise_runtime_selection: NoiseRuntimeSelection,
        dtype: torch.dtype,
        resolved_split_indices=None,
    ) -> list[DatasetBundle | None]:
        _ = (
            context,
            contexts_by_batch,
            configs_by_batch,
            attempt,
            attempts_used,
            device,
            n_train,
            n_test,
            requested_device,
            resolved_device,
            device_fallback_reason,
            x,
            y,
            aux_meta_batch,
            noise_runtime_selection,
            dtype,
        )
        assert resolved_split_indices is not None
        assert resolved_split_indices[0] is not None
        assert resolved_split_indices[1] is None
        return [_make_bundle(dataset_roots[0].child_seed()), None]

    def _stub_generate_fixed_layout_bundle_with_retries(
        _config: GeneratorConfig,
        *,
        plan: _FixedLayoutPlan,
        dataset_root: KeyedRng,
        requested_device: str,
        resolved_device: str,
        preserve_feature_schema: bool,
        start_attempt: int,
        finalization_context,
        on_raw_batch_metrics,
    ) -> DatasetBundle:
        _ = (
            plan,
            requested_device,
            resolved_device,
            preserve_feature_schema,
            finalization_context,
            on_raw_batch_metrics,
        )
        fallback_start_attempts.append(int(start_attempt))
        return _make_bundle(dataset_root.child_seed())

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._resolve_steered_dataset_descriptor",
        _stub_resolve_steered_dataset_descriptor,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._group_noise_runtime_chunk",
        _stub_group_noise_runtime_chunk,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_grouped_raw_batches",
        _stub_generate_grouped_raw_batches,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._resolve_split_indices",
        _stub_resolve_split_indices,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_chunk_preserve_schema",
        _stub_finalize_generated_chunk_preserve_schema,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_fixed_layout_bundle_with_retries",
        _stub_generate_fixed_layout_bundle_with_retries,
    )

    bundles = list(
        _generate_batch_with_plan_iter(
            cfg,
            plan=plan,
            num_datasets=2,
            seed=56,
            batch_size=2,
            on_raw_batch_metrics=lambda metrics: raw_metric_calls.append(metrics),
        )
    )

    assert len(bundles) == 2
    assert raw_metric_calls == [{"grouped_batch_count": 1.0}]
    assert fallback_start_attempts == [1]


def test_generate_batch_with_plan_iter_dynamic_steering_uses_retry_attempt_plan_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.steering.enabled = True
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=906,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )
    retry_start_attempts: list[int] = []

    def _make_bundle(seed: int) -> DatasetBundle:
        return DatasetBundle(
            X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
            y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.float32),
            X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
            y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.float32),
            feature_types=["num", "num"],
            metadata={
                "seed": int(seed),
                "n_features": 2,
                "lineage": {
                    "assignments": {
                        "feature_to_node": [0, 1],
                        "target_to_node": 1,
                        "target_parent_features": [0, 1],
                        "target_parent_count": 2,
                        "target_parent_fraction": 1.0,
                        "target_parent_prior": "all_features",
                        "target_parent_regime": "all_features",
                        "target_parent_sqrt_threshold": 1,
                    }
                },
                "filter": {"mode": "deferred", "status": "not_run"},
                "generation_attempts": {
                    "total_attempts": 1,
                    "retry_count": 0,
                    "filter_attempts": 0,
                    "filter_rejections": 0,
                    "filter_rejection_rate": None,
                },
            },
        )

    def _stub_resolve_steered_dataset_descriptor(
        _config: GeneratorConfig,
        *,
        base_plan: _FixedLayoutPlan,
        dataset_index: int,
        num_datasets: int,
        dataset_root: KeyedRng,
    ):
        _ = base_plan
        _ = num_datasets
        effective_config = deepcopy(cfg)
        effective_config.steering.enabled = False
        effective_config.validate_generation_constraints()
        return SimpleNamespace(
            dataset_index=dataset_index,
            dataset_root=dataset_root,
            effective_config=effective_config,
            effective_plan=plan,
            effective_shift=resolve_shift_runtime_params(effective_config),
            finalization_context=_build_fixed_schema_finalization_context(
                effective_config,
                plan.layout,
                n_train=cfg.dataset.n_train,
                n_test=cfg.dataset.n_test,
                shift_params=resolve_shift_runtime_params(effective_config),
            ),
        )

    def _stub_group_noise_runtime_chunk(
        _config: GeneratorConfig,
        *,
        dataset_roots: list[KeyedRng],
        attempts: list[int] | None = None,
    ):
        assert len(dataset_roots) == 1
        assert attempts == [0]
        return [
            SimpleNamespace(
                chunk_offsets=[0],
                generation_seeds=[
                    dataset_roots[0].keyed("attempt", 0, "raw_generation").child_seed()
                ],
                selection=NoiseRuntimeSelection(
                    family_requested="gaussian",
                    family_sampled="gaussian",
                    sampling_strategy="dataset_level",
                    base_scale=1.0,
                    student_t_df=5.0,
                    mixture_weights=None,
                ),
                attempt=0,
            )
        ]

    def _stub_generate_grouped_raw_batches(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan: FixedLayoutExecutionPlan,
        grouped_noise_runtime,
        requested_device: str,
        resolved_device: str,
        noise_sigma_multiplier: float,
    ) -> list[SimpleNamespace]:
        _ = execution_plan
        _ = requested_device
        _ = resolved_device
        _ = noise_sigma_multiplier
        group = grouped_noise_runtime[0]
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return [
            SimpleNamespace(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=group.attempt,
                x_batch=torch.zeros((1, n_rows, 2), dtype=torch.float32),
                y_batch=torch.zeros((1, n_rows), dtype=torch.float32),
                aux_meta_batch=[{"filter": {"mode": "deferred", "status": "not_run"}}],
                effective_resolved_device="cpu",
                device_fallback_reason=None,
                runtime_metrics={},
            )
        ]

    def _stub_finalize_generated_chunk_preserve_schema(
        _config: GeneratorConfig,
        _layout,
        *,
        context,
        contexts_by_batch=None,
        configs_by_batch=None,
        dataset_roots: list[KeyedRng],
        attempt: int,
        attempts_used: int,
        device: str,
        n_train: int,
        n_test: int,
        requested_device: str,
        resolved_device: str,
        device_fallback_reason: str | None,
        x: torch.Tensor,
        y: torch.Tensor,
        aux_meta_batch: list[dict[str, object]],
        noise_runtime_selection: NoiseRuntimeSelection,
        dtype: torch.dtype,
        resolved_split_indices=None,
    ) -> list[DatasetBundle | None]:
        _ = (
            context,
            contexts_by_batch,
            configs_by_batch,
            attempt,
            attempts_used,
            device,
            n_train,
            n_test,
            requested_device,
            resolved_device,
            device_fallback_reason,
            x,
            y,
            aux_meta_batch,
            noise_runtime_selection,
            dtype,
            resolved_split_indices,
        )
        return [_make_bundle(dataset_roots[0].child_seed())]

    def _stub_generate_fixed_layout_bundle_with_retries(
        _config: GeneratorConfig,
        *,
        plan: _FixedLayoutPlan,
        dataset_root: KeyedRng,
        requested_device: str,
        resolved_device: str,
        preserve_feature_schema: bool,
        start_attempt: int,
        finalization_context,
        on_raw_batch_metrics,
    ) -> DatasetBundle:
        _ = (
            plan,
            requested_device,
            resolved_device,
            preserve_feature_schema,
            finalization_context,
            on_raw_batch_metrics,
        )
        retry_start_attempts.append(int(start_attempt))
        return _make_bundle(dataset_root.child_seed())

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._resolve_steered_dataset_descriptor",
        _stub_resolve_steered_dataset_descriptor,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._group_noise_runtime_chunk",
        _stub_group_noise_runtime_chunk,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_grouped_raw_batches",
        _stub_generate_grouped_raw_batches,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_chunk_preserve_schema",
        _stub_finalize_generated_chunk_preserve_schema,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_fixed_layout_bundle_with_retries",
        _stub_generate_fixed_layout_bundle_with_retries,
    )

    bundles = list(
        _generate_batch_with_plan_iter(
            cfg,
            plan=plan,
            num_datasets=2,
            seed=57,
            batch_size=2,
            classification_attempt_plan=(0, 1),
        )
    )

    assert len(bundles) == 2
    assert retry_start_attempts == [1]


def test_generate_batch_with_plan_iter_dynamic_steering_rejects_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.steering.enabled = True
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=907,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )

    def _make_bundle(seed: int, *, feature_types: list[str]) -> DatasetBundle:
        return DatasetBundle(
            X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
            y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.float32),
            X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
            y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.float32),
            feature_types=feature_types,
            metadata={
                "seed": int(seed),
                "n_features": 2,
                "lineage": {
                    "assignments": {
                        "feature_to_node": [0, 1],
                        "target_to_node": 1,
                        "target_parent_features": [0, 1],
                        "target_parent_count": 2,
                        "target_parent_fraction": 1.0,
                        "target_parent_prior": "all_features",
                        "target_parent_regime": "all_features",
                        "target_parent_sqrt_threshold": 1,
                    }
                },
                "filter": {"mode": "deferred", "status": "not_run"},
                "generation_attempts": {
                    "total_attempts": 1,
                    "retry_count": 0,
                    "filter_attempts": 0,
                    "filter_rejections": 0,
                    "filter_rejection_rate": None,
                },
            },
        )

    def _stub_resolve_steered_dataset_descriptor(
        _config: GeneratorConfig,
        *,
        base_plan: _FixedLayoutPlan,
        dataset_index: int,
        num_datasets: int,
        dataset_root: KeyedRng,
    ):
        _ = base_plan
        _ = num_datasets
        effective_config = deepcopy(cfg)
        effective_config.steering.enabled = False
        effective_config.validate_generation_constraints()
        return SimpleNamespace(
            dataset_index=dataset_index,
            dataset_root=dataset_root,
            effective_config=effective_config,
            effective_plan=plan,
            effective_shift=resolve_shift_runtime_params(effective_config),
            finalization_context=_build_fixed_schema_finalization_context(
                effective_config,
                plan.layout,
                n_train=cfg.dataset.n_train,
                n_test=cfg.dataset.n_test,
                shift_params=resolve_shift_runtime_params(effective_config),
            ),
        )

    def _stub_group_noise_runtime_chunk(
        _config: GeneratorConfig,
        *,
        dataset_roots: list[KeyedRng],
        attempts: list[int] | None = None,
    ):
        assert attempts == [0] * len(dataset_roots)
        return [
            SimpleNamespace(
                chunk_offsets=list(range(len(dataset_roots))),
                generation_seeds=[
                    dataset_root.keyed("attempt", 0, "raw_generation").child_seed()
                    for dataset_root in dataset_roots
                ],
                selection=NoiseRuntimeSelection(
                    family_requested="gaussian",
                    family_sampled="gaussian",
                    sampling_strategy="dataset_level",
                    base_scale=1.0,
                    student_t_df=5.0,
                    mixture_weights=None,
                ),
                attempt=0,
            )
        ]

    def _stub_generate_grouped_raw_batches(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan: FixedLayoutExecutionPlan,
        grouped_noise_runtime,
        requested_device: str,
        resolved_device: str,
        noise_sigma_multiplier: float,
    ) -> list[SimpleNamespace]:
        _ = execution_plan
        _ = requested_device
        _ = resolved_device
        _ = noise_sigma_multiplier
        group = grouped_noise_runtime[0]
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return [
            SimpleNamespace(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=group.attempt,
                x_batch=torch.zeros((2, n_rows, 2), dtype=torch.float32),
                y_batch=torch.zeros((2, n_rows), dtype=torch.float32),
                aux_meta_batch=[
                    {"filter": {"mode": "deferred", "status": "not_run"}}
                    for _ in group.chunk_offsets
                ],
                effective_resolved_device="cpu",
                device_fallback_reason=None,
                runtime_metrics={},
            )
        ]

    def _stub_finalize_generated_chunk_preserve_schema(
        _config: GeneratorConfig,
        _layout,
        *,
        context,
        contexts_by_batch=None,
        configs_by_batch=None,
        dataset_roots: list[KeyedRng],
        attempt: int,
        attempts_used: int,
        device: str,
        n_train: int,
        n_test: int,
        requested_device: str,
        resolved_device: str,
        device_fallback_reason: str | None,
        x: torch.Tensor,
        y: torch.Tensor,
        aux_meta_batch: list[dict[str, object]],
        noise_runtime_selection: NoiseRuntimeSelection,
        dtype: torch.dtype,
        resolved_split_indices=None,
    ) -> list[DatasetBundle | None]:
        _ = (
            context,
            contexts_by_batch,
            configs_by_batch,
            attempt,
            attempts_used,
            device,
            n_train,
            n_test,
            requested_device,
            resolved_device,
            device_fallback_reason,
            x,
            y,
            aux_meta_batch,
            noise_runtime_selection,
            dtype,
            resolved_split_indices,
        )
        return [
            _make_bundle(dataset_roots[0].child_seed(), feature_types=["num", "num"]),
            _make_bundle(dataset_roots[1].child_seed(), feature_types=["num", "cat"]),
        ]

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._resolve_steered_dataset_descriptor",
        _stub_resolve_steered_dataset_descriptor,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._group_noise_runtime_chunk",
        _stub_group_noise_runtime_chunk,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_grouped_raw_batches",
        _stub_generate_grouped_raw_batches,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_chunk_preserve_schema",
        _stub_finalize_generated_chunk_preserve_schema,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_fixed_layout_bundle_with_retries",
        lambda *_args, **_kwargs: pytest.fail("schema mismatch should fail before retry fallback"),
    )

    with pytest.raises(ValueError, match="Fixed-layout schema mismatch"):
        list(
            _generate_batch_with_plan_iter(
                cfg,
                plan=plan,
                num_datasets=2,
                seed=58,
                batch_size=2,
            )
        )


def test_generate_batch_with_plan_iter_dynamic_steering_requires_all_grouped_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.steering.enabled = True
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=908,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )

    def _make_bundle(seed: int) -> DatasetBundle:
        return DatasetBundle(
            X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
            y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.float32),
            X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
            y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.float32),
            feature_types=["num", "num"],
            metadata={
                "seed": int(seed),
                "n_features": 2,
                "lineage": {
                    "assignments": {
                        "feature_to_node": [0, 1],
                        "target_to_node": 1,
                        "target_parent_features": [0, 1],
                        "target_parent_count": 2,
                        "target_parent_fraction": 1.0,
                        "target_parent_prior": "all_features",
                        "target_parent_regime": "all_features",
                        "target_parent_sqrt_threshold": 1,
                    }
                },
                "filter": {"mode": "deferred", "status": "not_run"},
                "generation_attempts": {
                    "total_attempts": 1,
                    "retry_count": 0,
                    "filter_attempts": 0,
                    "filter_rejections": 0,
                    "filter_rejection_rate": None,
                },
            },
        )

    def _stub_resolve_steered_dataset_descriptor(
        _config: GeneratorConfig,
        *,
        base_plan: _FixedLayoutPlan,
        dataset_index: int,
        num_datasets: int,
        dataset_root: KeyedRng,
    ):
        _ = base_plan
        _ = num_datasets
        effective_config = deepcopy(cfg)
        effective_config.steering.enabled = False
        effective_config.validate_generation_constraints()
        return SimpleNamespace(
            dataset_index=dataset_index,
            dataset_root=dataset_root,
            effective_config=effective_config,
            effective_plan=plan,
            effective_shift=resolve_shift_runtime_params(effective_config),
            finalization_context=_build_fixed_schema_finalization_context(
                effective_config,
                plan.layout,
                n_train=cfg.dataset.n_train,
                n_test=cfg.dataset.n_test,
                shift_params=resolve_shift_runtime_params(effective_config),
            ),
        )

    def _stub_group_noise_runtime_chunk(
        _config: GeneratorConfig,
        *,
        dataset_roots: list[KeyedRng],
        attempts: list[int] | None = None,
    ):
        assert attempts == [0] * len(dataset_roots)
        return [
            SimpleNamespace(
                chunk_offsets=[0],
                generation_seeds=[
                    dataset_roots[0].keyed("attempt", 0, "raw_generation").child_seed()
                ],
                selection=NoiseRuntimeSelection(
                    family_requested="gaussian",
                    family_sampled="gaussian",
                    sampling_strategy="dataset_level",
                    base_scale=1.0,
                    student_t_df=5.0,
                    mixture_weights=None,
                ),
                attempt=0,
            )
        ]

    def _stub_generate_grouped_raw_batches(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan: FixedLayoutExecutionPlan,
        grouped_noise_runtime,
        requested_device: str,
        resolved_device: str,
        noise_sigma_multiplier: float,
    ) -> list[SimpleNamespace]:
        _ = execution_plan
        _ = requested_device
        _ = resolved_device
        _ = noise_sigma_multiplier
        group = grouped_noise_runtime[0]
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return [
            SimpleNamespace(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=group.attempt,
                x_batch=torch.zeros((1, n_rows, 2), dtype=torch.float32),
                y_batch=torch.zeros((1, n_rows), dtype=torch.float32),
                aux_meta_batch=[{"filter": {"mode": "deferred", "status": "not_run"}}],
                effective_resolved_device="cpu",
                device_fallback_reason=None,
                runtime_metrics={},
            )
        ]

    def _stub_finalize_generated_chunk_preserve_schema(
        _config: GeneratorConfig,
        _layout,
        *,
        context,
        contexts_by_batch=None,
        configs_by_batch=None,
        dataset_roots: list[KeyedRng],
        attempt: int,
        attempts_used: int,
        device: str,
        n_train: int,
        n_test: int,
        requested_device: str,
        resolved_device: str,
        device_fallback_reason: str | None,
        x: torch.Tensor,
        y: torch.Tensor,
        aux_meta_batch: list[dict[str, object]],
        noise_runtime_selection: NoiseRuntimeSelection,
        dtype: torch.dtype,
        resolved_split_indices=None,
    ) -> list[DatasetBundle | None]:
        _ = (
            context,
            contexts_by_batch,
            configs_by_batch,
            attempt,
            attempts_used,
            device,
            n_train,
            n_test,
            requested_device,
            resolved_device,
            device_fallback_reason,
            x,
            y,
            aux_meta_batch,
            noise_runtime_selection,
            dtype,
            resolved_split_indices,
        )
        return [_make_bundle(dataset_roots[0].child_seed())]

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._resolve_steered_dataset_descriptor",
        _stub_resolve_steered_dataset_descriptor,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._group_noise_runtime_chunk",
        _stub_group_noise_runtime_chunk,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_grouped_raw_batches",
        _stub_generate_grouped_raw_batches,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_chunk_preserve_schema",
        _stub_finalize_generated_chunk_preserve_schema,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_fixed_layout_bundle_with_retries",
        lambda *_args, **_kwargs: pytest.fail(
            "missing grouped offsets should fail before retry fallback"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Missing grouped raw batch entry for steering-enabled fixed-layout chunk offset",
    ):
        list(
            _generate_batch_with_plan_iter(
                cfg,
                plan=plan,
                num_datasets=2,
                seed=59,
                batch_size=2,
            )
        )


def test_generate_batch_rows_range_is_seed_reproducible() -> None:
    cfg = _tiny_config()
    cfg.dataset.rows = "1024..4096"  # type: ignore[assignment]
    cfg.dataset.n_test = 256

    batch_a = generate_batch(cfg, num_datasets=5, seed=19, device="cpu")
    batch_b = generate_batch(cfg, num_datasets=5, seed=19, device="cpu")

    train_sizes_a = {int(bundle.X_train.shape[0]) for bundle in batch_a}
    train_sizes_b = {int(bundle.X_train.shape[0]) for bundle in batch_b}
    assert len(train_sizes_a) == 1
    assert len(train_sizes_b) == 1
    for bundle_a, bundle_b in zip(batch_a, batch_b, strict=True):
        assert 768 <= int(bundle_a.X_train.shape[0]) <= 3840
        assert 768 <= int(bundle_b.X_train.shape[0]) <= 3840
        assert int(bundle_a.X_train.shape[0]) == int(bundle_b.X_train.shape[0])
        assert int(bundle_a.X_test.shape[0]) == 256
        assert int(bundle_b.X_test.shape[0]) == 256
        assert (
            bundle_a.metadata["layout_plan_signature"]
            == batch_a[0].metadata["layout_plan_signature"]
        )
        assert (
            bundle_b.metadata["layout_plan_signature"]
            == batch_b[0].metadata["layout_plan_signature"]
        )


def test_generate_one_emits_lineage_metadata_schema_fields() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=11, device="cpu")
    lineage = bundle.metadata["lineage"]
    assert lineage["schema_name"] == LINEAGE_SCHEMA_NAME
    assert lineage["schema_version"] == LINEAGE_SCHEMA_VERSION
    assert "graph" in lineage
    assert "assignments" in lineage
    validate_metadata_lineage(bundle.metadata, required=True)
    validate_lineage_payload(lineage)


def test_generate_one_lineage_shapes_match_graph_stats() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=12, device="cpu")
    lineage = bundle.metadata["lineage"]
    graph = lineage["graph"]
    adjacency = graph["adjacency"]
    n_nodes = int(bundle.metadata["graph_nodes"])

    assert int(graph["n_nodes"]) == n_nodes
    assert len(adjacency) == n_nodes
    for row in adjacency:
        assert len(row) == n_nodes

    edge_count = sum(sum(int(value) for value in row) for row in adjacency)
    assert edge_count == int(bundle.metadata["graph_edges"])


def test_generate_one_lineage_assignment_lengths_and_bounds() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=13, device="cpu")
    lineage = bundle.metadata["lineage"]
    assignments = lineage["assignments"]
    n_nodes = int(bundle.metadata["graph_nodes"])

    feature_to_node = assignments["feature_to_node"]
    assert len(feature_to_node) == int(bundle.metadata["n_features"])
    assert assignments["target_mode"] == "latent_complete_x_conditional"
    target_parent_features = assignments["target_parent_features"]
    assert assignments["target_parent_count"] == len(target_parent_features)
    assert assignments["target_parent_prior"] == "near_max_mixture"
    assert assignments["target_parent_regime"] in {
        "all_features",
        "sparse_tail",
        "midrange",
        "near_max",
    }
    assert target_parent_features == sorted(target_parent_features)
    assert assignments["target_parent_sqrt_threshold"] == int(
        math.floor(math.sqrt(int(bundle.metadata["n_features"])))
    )
    assert assignments["target_parent_fraction"] == pytest.approx(
        float(len(target_parent_features)) / float(len(feature_to_node))
    )
    for node_index in feature_to_node:
        assert 0 <= int(node_index) < n_nodes


def test_generate_one_emits_target_parent_summary_metadata() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=2026, device="cpu")

    lineage_assignments = bundle.metadata["lineage"]["assignments"]
    summary = bundle.metadata["target_parent_summary"]

    assert summary["count"] == lineage_assignments["target_parent_count"]
    assert summary["fraction"] == pytest.approx(lineage_assignments["target_parent_fraction"])
    assert summary["prior"] == lineage_assignments["target_parent_prior"]
    assert summary["regime"] == lineage_assignments["target_parent_regime"]
    assert summary["sqrt_threshold"] == lineage_assignments["target_parent_sqrt_threshold"]
    assert summary["features"] == lineage_assignments["target_parent_features"]


def test_generate_one_emits_latent_complete_prior_metadata() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=1313, device="cpu")
    prior = bundle.metadata["prior"]

    assert prior == {
        "factorization": "independent_p_x_complete_and_p_y_given_x_complete",
        "target_head": "latent_complete_x_conditional",
        "feature_generator": "latent_dag",
        "missingness_stage": "post_target_observation",
        "classification_validity_policy": "retry_only",
        "localization_mode": "none",
        "n_adaptation": "none",
    }


def test_generate_one_can_export_teacher_conditionals() -> None:
    cfg = _tiny_config()
    cfg.diagnostics.teacher_conditional_export = True
    bundle = generate_one(cfg, seed=1414, device="cpu")

    posterior_predictive = bundle.metadata["posterior_predictive"]
    teacher_conditionals = bundle.metadata["teacher_conditionals"]

    assert posterior_predictive == {
        "factorization": "independent_p_x_complete_and_p_y_given_x_complete",
        "teacher_conditional_export_enabled": True,
        "teacher_conditionals_available": True,
        "metric_definition": "label-target log loss per test cell",
    }
    assert teacher_conditionals["schema_version"] == 1
    assert teacher_conditionals["target_split"] == "test"
    assert teacher_conditionals["class_labels"] == list(range(bundle.metadata["n_classes"]))
    probs = np.asarray(teacher_conditionals["test_probs"], dtype=np.float64)
    assert probs.shape == (len(bundle.y_test), bundle.metadata["n_classes"])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
    chosen = probs[np.arange(len(bundle.y_test)), np.asarray(bundle.y_test, dtype=np.int64)]
    expected = float((-np.log(np.clip(chosen, 1e-12, 1.0))).mean())
    assert teacher_conditionals["optimal_log_loss_per_test_cell"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "aux_meta",
    [
        {
            "teacher_conditional_test_probs": [[0.7, 0.3], [0.2, 0.8]],
            "teacher_conditional_class_labels": [],
        },
        {
            "teacher_conditional_test_probs": [0.7, 0.3],
            "teacher_conditional_class_labels": [0, 1],
        },
        {
            "teacher_conditional_test_probs": [[0.7, 0.3]],
            "teacher_conditional_class_labels": [0, 1],
        },
        {
            "teacher_conditional_test_probs": [[0.7], [0.3]],
            "teacher_conditional_class_labels": [0, 1],
        },
    ],
)
def test_teacher_conditionals_metadata_rejects_malformed_aux_metadata(
    aux_meta: dict[str, object],
) -> None:
    y_test = torch.tensor([0, 1], dtype=torch.int64)

    assert _teacher_conditionals_metadata(y_test=y_test, aux_meta=aux_meta) is None


def test_remap_teacher_conditionals_reorders_columns_for_emitted_label_permutation() -> None:
    aux_meta = {
        "teacher_conditional_class_labels": [0, 1],
        "teacher_conditional_full_probs": [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.70, 0.30],
            [0.40, 0.60],
        ],
    }
    train_idx = torch.tensor([0, 1], dtype=torch.int64)
    test_idx = torch.tensor([2, 3], dtype=torch.int64)

    _remap_teacher_conditionals_to_emitted_labels(
        aux_meta=aux_meta,
        train_idx=train_idx,
        test_idx=test_idx,
        y_train_raw=torch.tensor([0, 1], dtype=torch.int64),
        y_test_raw=torch.tensor([1, 0], dtype=torch.int64),
        y_train_emitted=torch.tensor([1, 0], dtype=torch.int64),
        y_test_emitted=torch.tensor([0, 1], dtype=torch.int64),
    )
    _attach_teacher_conditionals_for_test_split(aux_meta, test_idx=test_idx)
    teacher_conditionals = _teacher_conditionals_metadata(
        y_test=torch.tensor([0, 1], dtype=torch.int64),
        aux_meta=aux_meta,
    )

    assert teacher_conditionals is not None
    assert teacher_conditionals["class_labels"] == [0, 1]
    np.testing.assert_allclose(
        np.asarray(aux_meta["teacher_conditional_full_probs"], dtype=np.float64),
        np.asarray(
            [
                [0.10, 0.90],
                [0.80, 0.20],
                [0.30, 0.70],
                [0.60, 0.40],
            ],
            dtype=np.float64,
        ),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(teacher_conditionals["test_probs"], dtype=np.float64),
        np.asarray([[0.30, 0.70], [0.60, 0.40]], dtype=np.float64),
        atol=1e-6,
        rtol=1e-6,
    )
    expected = float((-np.log(np.asarray([0.30, 0.40], dtype=np.float64).clip(1e-12, 1.0))).mean())
    assert teacher_conditionals["optimal_log_loss_per_test_cell"] == pytest.approx(expected)


def test_remap_teacher_conditionals_drops_absent_classes_and_renormalizes() -> None:
    aux_meta = {
        "teacher_conditional_class_labels": [0, 1, 2],
        "teacher_conditional_full_probs": [
            [0.10, 0.20, 0.70],
            [0.20, 0.50, 0.30],
            [0.70, 0.10, 0.20],
            [0.25, 0.25, 0.50],
        ],
    }
    train_idx = torch.tensor([0, 1], dtype=torch.int64)
    test_idx = torch.tensor([2, 3], dtype=torch.int64)

    _remap_teacher_conditionals_to_emitted_labels(
        aux_meta=aux_meta,
        train_idx=train_idx,
        test_idx=test_idx,
        y_train_raw=torch.tensor([0, 2], dtype=torch.int64),
        y_test_raw=torch.tensor([2, 0], dtype=torch.int64),
        y_train_emitted=torch.tensor([1, 0], dtype=torch.int64),
        y_test_emitted=torch.tensor([0, 1], dtype=torch.int64),
    )
    _attach_teacher_conditionals_for_test_split(aux_meta, test_idx=test_idx)
    teacher_conditionals = _teacher_conditionals_metadata(
        y_test=torch.tensor([0, 1], dtype=torch.int64),
        aux_meta=aux_meta,
    )

    assert teacher_conditionals is not None
    assert aux_meta["teacher_conditional_class_labels"] == [0, 1]
    full_probs = np.asarray(aux_meta["teacher_conditional_full_probs"], dtype=np.float64)
    np.testing.assert_allclose(full_probs.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        full_probs,
        np.asarray(
            [
                [0.875, 0.125],
                [0.600, 0.400],
                [0.2222222222, 0.7777777778],
                [0.6666666667, 0.3333333333],
            ],
            dtype=np.float64,
        ),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(teacher_conditionals["test_probs"], dtype=np.float64),
        full_probs[2:],
        atol=1e-6,
        rtol=1e-6,
    )
    expected = float(
        (
            -np.log(
                np.asarray(
                    [
                        full_probs[2, 0],
                        full_probs[3, 1],
                    ],
                    dtype=np.float64,
                ).clip(1e-12, 1.0)
            )
        ).mean()
    )
    assert teacher_conditionals["optimal_log_loss_per_test_cell"] == pytest.approx(expected)


def test_sample_layout_near_dense_target_parent_prior_has_mode_at_max() -> None:
    cfg = _tiny_config()
    cfg.dataset.n_features_min = 20
    cfg.dataset.n_features_max = 20

    counts: list[int] = []
    regimes: list[str] = []
    for seed in range(512):
        layout = _sample_layout(cfg, KeyedRng(seed), device="cpu")
        counts.append(len(layout.target_parent_features))
        regimes.append(layout.target_parent_regime)

    assert max(set(counts), key=counts.count) == 20
    assert sum(1 for count in counts if count >= 15) > int(0.60 * len(counts))
    assert sum(1 for count in counts if count < int(math.floor(math.sqrt(20)))) == pytest.approx(
        0.05 * len(counts),
        rel=0.40,
    )
    assert "near_max" in regimes


def test_generate_one_emits_graph_complexity_metadata() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=14, device="cpu")

    graph_nodes = int(bundle.metadata["graph_nodes"])
    graph_depth_nodes = int(bundle.metadata["graph_depth_nodes"])
    graph_edge_density = float(bundle.metadata["graph_edge_density"])

    assert 1 <= graph_depth_nodes <= graph_nodes
    assert 0.0 <= graph_edge_density <= 1.0


def test_generate_batch_reproducible_metadata() -> None:
    cfg = _tiny_config()
    batch_a = generate_batch(cfg, num_datasets=2, seed=123, device="cpu")
    batch_b = generate_batch(cfg, num_datasets=2, seed=123, device="cpu")
    assert batch_a[0].metadata["seed"] == batch_b[0].metadata["seed"]
    np.testing.assert_allclose(
        np.asarray(batch_a[0].X_train),
        np.asarray(batch_b[0].X_train),
        atol=1e-6,
        rtol=1e-6,
    )


def test_generate_batch_reproducible_lineage_for_fixed_seed() -> None:
    cfg = _tiny_config()
    batch_a = generate_batch(cfg, num_datasets=2, seed=678, device="cpu")
    batch_b = generate_batch(cfg, num_datasets=2, seed=678, device="cpu")

    assert len(batch_a) == len(batch_b)
    for bundle_a, bundle_b in zip(batch_a, batch_b, strict=True):
        assert bundle_a.metadata["lineage"] == bundle_b.metadata["lineage"]


def test_generate_one_shift_disabled_preserves_baseline_outputs() -> None:
    baseline = _tiny_regression_config()
    disabled = _tiny_regression_config()
    disabled.shift.enabled = False
    disabled.shift.mode = "off"

    bundle_base = generate_one(baseline, seed=1881, device="cpu")
    bundle_disabled = generate_one(disabled, seed=1881, device="cpu")

    torch.testing.assert_close(bundle_base.X_train, bundle_disabled.X_train)
    torch.testing.assert_close(bundle_base.X_test, bundle_disabled.X_test)
    torch.testing.assert_close(bundle_base.y_train, bundle_disabled.y_train)
    torch.testing.assert_close(bundle_base.y_test, bundle_disabled.y_test)
    assert bundle_base.metadata["graph_edges"] == bundle_disabled.metadata["graph_edges"]
    assert bundle_base.metadata["graph_edge_density"] == pytest.approx(
        bundle_disabled.metadata["graph_edge_density"]
    )


def test_generate_one_shift_metadata_emits_disabled_defaults() -> None:
    cfg = _tiny_regression_config()
    cfg.shift.enabled = False
    cfg.shift.mode = "off"

    bundle = generate_one(cfg, seed=1882, device="cpu")
    shift_metadata = bundle.metadata["shift"]
    assert shift_metadata["enabled"] is False
    assert shift_metadata["mode"] == "off"
    assert shift_metadata["graph_scale"] == pytest.approx(0.0)
    assert shift_metadata["mechanism_scale"] == pytest.approx(0.0)
    assert shift_metadata["variance_scale"] == pytest.approx(0.0)
    assert shift_metadata["edge_logit_bias_shift"] == pytest.approx(0.0)
    assert shift_metadata["mechanism_logit_tilt"] == pytest.approx(0.0)
    assert shift_metadata["variance_sigma_multiplier"] == pytest.approx(1.0)
    assert shift_metadata["edge_odds_multiplier"] == pytest.approx(1.0)
    assert shift_metadata["noise_variance_multiplier"] == pytest.approx(1.0)
    assert shift_metadata["mechanism_nonlinear_mass"] == pytest.approx(
        mechanism_nonlinear_mass(mechanism_logit_tilt=0.0)
    )


def test_generate_one_shift_metadata_matches_resolved_runtime_params() -> None:
    cfg = _tiny_regression_config()
    cfg.shift.enabled = True
    cfg.shift.mode = "custom"
    cfg.shift.graph_scale = 0.6
    cfg.shift.mechanism_scale = 0.3
    cfg.shift.variance_scale = 0.4
    runtime = resolve_shift_runtime_params(cfg)

    bundle = generate_one(cfg, seed=1883, device="cpu")
    shift_metadata = bundle.metadata["shift"]
    assert shift_metadata["enabled"] is True
    assert shift_metadata["mode"] == "custom"
    assert shift_metadata["graph_scale"] == pytest.approx(runtime.graph_scale)
    assert shift_metadata["mechanism_scale"] == pytest.approx(runtime.mechanism_scale)
    assert shift_metadata["variance_scale"] == pytest.approx(runtime.variance_scale)
    assert shift_metadata["edge_logit_bias_shift"] == pytest.approx(runtime.edge_logit_bias_shift)
    assert shift_metadata["mechanism_logit_tilt"] == pytest.approx(runtime.mechanism_logit_tilt)
    assert shift_metadata["variance_sigma_multiplier"] == pytest.approx(
        runtime.variance_sigma_multiplier
    )
    assert shift_metadata["edge_odds_multiplier"] == pytest.approx(
        math.exp(runtime.edge_logit_bias_shift)
    )
    assert shift_metadata["noise_variance_multiplier"] == pytest.approx(
        runtime.variance_sigma_multiplier**2
    )
    assert shift_metadata["mechanism_nonlinear_mass"] == pytest.approx(
        mechanism_nonlinear_mass(mechanism_logit_tilt=runtime.mechanism_logit_tilt)
    )


def test_generate_one_shift_metadata_respects_mechanism_family_mix() -> None:
    cfg = _tiny_regression_config()
    cfg.shift.enabled = True
    cfg.shift.mode = "mechanism_drift"
    cfg.shift.mechanism_scale = 1.0
    cfg.mechanism.function_family_mix = {"linear": 1.0}

    bundle = generate_one(cfg, seed=18835, device="cpu")
    shift_metadata = bundle.metadata["shift"]
    assert shift_metadata["mechanism_nonlinear_mass"] == pytest.approx(0.0)
    assert bundle.metadata["config"]["mechanism"]["function_family_mix"] == {"linear": 1.0}


def test_generate_one_emits_realized_mechanism_family_metadata() -> None:
    cfg = _tiny_regression_config()
    cfg.mechanism.function_family_mix = {"piecewise": 0.3, "linear": 0.7}

    bundle = generate_one(cfg, seed=18836, device="cpu")
    mechanism_families = bundle.metadata["mechanism_families"]

    assert mechanism_families["total_function_plans"] >= 1
    assert set(mechanism_families["sampled_family_counts"]).issubset({"piecewise", "linear"})
    assert set(mechanism_families["families_present"]).issubset({"piecewise", "linear"})
    assert mechanism_families["sampled_variant_counts"] == {}
    assert mechanism_families["variants_present"] == []
    assert "piecewise" in mechanism_families["sampled_family_counts"]
    assert "piecewise" in mechanism_families["families_present"]


def test_generate_one_emits_realized_gp_variant_metadata() -> None:
    cfg = _tiny_regression_config()
    cfg.mechanism.function_family_mix = {"gp": 1.0}

    bundle = generate_one(cfg, seed=18837, device="cpu")
    mechanism_families = bundle.metadata["mechanism_families"]

    assert mechanism_families["sampled_family_counts"]["gp"] >= 1
    assert set(mechanism_families["families_present"]) == {"gp"}
    assert mechanism_families["total_function_plans"] >= 1
    assert set(mechanism_families["sampled_variant_counts"]).issubset(
        {"gp.standard", "gp.periodic", "gp.multiscale"}
    )
    assert mechanism_families["variants_present"]


def test_generate_one_noise_metadata_emits_gaussian_defaults() -> None:
    cfg = _tiny_regression_config()
    cfg.noise.family = "gaussian"
    cfg.noise.base_scale = 1.0
    cfg.noise.student_t_df = 5.0

    bundle = generate_one(cfg, seed=1884, device="cpu")
    noise_metadata = bundle.metadata["noise_distribution"]
    assert noise_metadata["family_requested"] == "gaussian"
    assert noise_metadata["family_sampled"] == "gaussian"
    assert noise_metadata["sampling_strategy"] == "dataset_level"
    assert noise_metadata["base_scale"] == pytest.approx(1.0)
    assert noise_metadata["student_t_df"] == pytest.approx(5.0)
    assert noise_metadata["mixture_weights"] is None


@pytest.mark.parametrize("family", [NOISE_FAMILY_LAPLACE, NOISE_FAMILY_STUDENT_T])
def test_generate_one_nongaussian_noise_family_changes_outputs_for_same_seed(family: str) -> None:
    baseline = _tiny_regression_config()
    baseline.noise.family = NOISE_FAMILY_GAUSSIAN

    drifted = _tiny_regression_config()
    drifted.noise.family = family
    drifted.noise.base_scale = 1.0
    if family == NOISE_FAMILY_STUDENT_T:
        drifted.noise.student_t_df = 6.0

    bundle_base = generate_one(baseline, seed=1885, device="cpu")
    bundle_drifted = generate_one(drifted, seed=1885, device="cpu")
    assert not torch.allclose(bundle_base.X_train, bundle_drifted.X_train)
    assert bundle_drifted.metadata["noise_distribution"]["family_requested"] == family
    assert bundle_drifted.metadata["noise_distribution"]["family_sampled"] == family


def test_generate_one_mixture_noise_is_dataset_level_and_reproducible() -> None:
    cfg = _tiny_regression_config()
    cfg.noise.family = "mixture"
    cfg.noise.mixture_weights = {"gaussian": 0.7, "laplace": 0.2, "student_t": 0.1}

    bundle_a = generate_one(cfg, seed=1886, device="cpu")
    bundle_b = generate_one(cfg, seed=1886, device="cpu")
    noise_a = bundle_a.metadata["noise_distribution"]
    noise_b = bundle_b.metadata["noise_distribution"]

    assert noise_a == noise_b
    assert noise_a["family_requested"] == "mixture"
    assert noise_a["family_sampled"] in {"gaussian", "laplace", "student_t"}
    assert noise_a["sampling_strategy"] == "dataset_level"
    assert noise_a["mixture_weights"] is not None
    assert sum(noise_a["mixture_weights"].values()) == pytest.approx(1.0)


def test_generate_one_graph_drift_increases_edge_density_for_same_seed() -> None:
    baseline = _tiny_regression_config()
    baseline.graph.n_nodes_min = 20
    baseline.graph.n_nodes_max = 20

    shifted = _tiny_regression_config()
    shifted.graph.n_nodes_min = 20
    shifted.graph.n_nodes_max = 20
    shifted.shift.enabled = True
    shifted.shift.mode = "graph_drift"

    bundle_base = generate_one(baseline, seed=2203, device="cpu")
    bundle_shifted = generate_one(shifted, seed=2203, device="cpu")

    assert int(bundle_shifted.metadata["graph_edges"]) >= int(bundle_base.metadata["graph_edges"])
    assert float(bundle_shifted.metadata["graph_edge_density"]) >= float(
        bundle_base.metadata["graph_edge_density"]
    )


@pytest.mark.parametrize(
    ("profile", "override_field", "override_value"),
    [
        ("mechanism_drift", "mechanism_scale", 1.0),
        ("noise_drift", "variance_scale", 1.0),
    ],
)
def test_generate_one_shift_profiles_change_outputs_for_same_seed(
    profile: str, override_field: str, override_value: float
) -> None:
    baseline = _tiny_regression_config()
    baseline.graph.n_nodes_min = 10
    baseline.graph.n_nodes_max = 10

    shifted = _tiny_regression_config()
    shifted.graph.n_nodes_min = 10
    shifted.graph.n_nodes_max = 10
    shifted.shift.enabled = True
    shifted.shift.mode = profile
    setattr(shifted.shift, override_field, override_value)

    bundle_base = generate_one(baseline, seed=2204, device="cpu")
    bundle_shifted = generate_one(shifted, seed=2204, device="cpu")
    if bundle_base.X_train.shape == bundle_shifted.X_train.shape:
        assert not torch.allclose(bundle_base.X_train, bundle_shifted.X_train)
    else:
        assert bundle_base.X_train.shape != bundle_shifted.X_train.shape
    if bundle_base.X_test.shape == bundle_shifted.X_test.shape:
        assert not torch.allclose(bundle_base.X_test, bundle_shifted.X_test)
    else:
        assert bundle_base.X_test.shape != bundle_shifted.X_test.shape


@pytest.mark.parametrize(
    ("profile", "overrides"),
    [
        ("graph_drift", {"graph_scale": 1.0}),
        ("mechanism_drift", {"mechanism_scale": 1.0}),
        ("noise_drift", {"variance_scale": 1.0}),
        ("mixed", {}),
    ],
)
def test_generate_one_shift_profiles_are_seed_reproducible(
    profile: str, overrides: dict[str, float]
) -> None:
    cfg = _tiny_regression_config()
    cfg.graph.n_nodes_min = 10
    cfg.graph.n_nodes_max = 10
    cfg.shift.enabled = True
    cfg.shift.mode = profile
    for key, value in overrides.items():
        setattr(cfg.shift, key, value)

    bundle_a = generate_one(cfg, seed=2205, device="cpu")
    bundle_b = generate_one(cfg, seed=2205, device="cpu")

    torch.testing.assert_close(bundle_a.X_train, bundle_b.X_train)
    torch.testing.assert_close(bundle_a.X_test, bundle_b.X_test)
    torch.testing.assert_close(bundle_a.y_train, bundle_b.y_train)
    torch.testing.assert_close(bundle_a.y_test, bundle_b.y_test)
    assert bundle_a.metadata["graph_edges"] == bundle_b.metadata["graph_edges"]
    assert bundle_a.metadata["graph_edge_density"] == pytest.approx(
        bundle_b.metadata["graph_edge_density"]
    )


def test_generate_one_lineage_assignments_follow_postprocess_feature_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "regression"
    cfg.filter.enabled = False

    layout = _layout_stub(
        feature_types=["num", "cat", "num", "cat"],
        graph_nodes=3,
        adjacency=torch.tensor(
            [
                [0, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.bool,
        ),
        feature_node_assignment=[0, 1, 2, 1],
        target_node_assignment=2,
    )

    def _stub_postprocess_dataset(
        x_train,
        y_train,
        x_test,
        y_test,
        feature_types,
        _task,
        _keyed_rng,
        _device,
        *,
        return_feature_index_map=False,
        preserve_feature_schema=False,
    ):
        assert return_feature_index_map is True
        assert preserve_feature_schema is False
        index_map = [2, 0, 3]
        reordered_types = [feature_types[i] for i in index_map]
        return (
            x_train[:, index_map],
            y_train,
            x_test[:, index_map],
            y_test,
            reordered_types,
            index_map,
        )

    monkeypatch.setattr(
        "dagzoo.core.generation_runtime.postprocess_dataset",
        _stub_postprocess_dataset,
    )

    n_rows = int(cfg.dataset.n_train + cfg.dataset.n_test)
    bundle = _finalize_generated_tensors(
        cfg,
        layout,
        dataset_seed=777,
        attempt=0,
        attempts_used=1,
        dataset_root=KeyedRng(777),
        device="cpu",
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        requested_device="cpu",
        resolved_device="cpu",
        device_fallback_reason=None,
        x=torch.arange(n_rows * 4, dtype=torch.float32).reshape(n_rows, 4),
        y=torch.linspace(0.0, 1.0, n_rows, dtype=torch.float32),
        aux_meta={"filter": {"enabled": False}},
        shift_params=resolve_shift_runtime_params(cfg),
        noise_runtime_selection=NoiseRuntimeSelection(
            family_requested="gaussian",
            family_sampled="gaussian",
            sampling_strategy="global",
            base_scale=1.0,
            student_t_df=5.0,
            mixture_weights=None,
        ),
        dtype=torch.float32,
        preserve_feature_schema=False,
    )
    assert int(bundle.metadata["n_features"]) == 3
    assert bundle.metadata["lineage"]["assignments"]["feature_to_node"] == [2, 0, 1]


def test_generate_torch_forces_cpu_for_stratified_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.filter.enabled = False
    captured: dict[str, str] = {}

    class _SplitSentinel(Exception):
        pass

    def _stub_stratified_split_indices(
        y: torch.Tensor,
        n_train: int,
        generator: torch.Generator,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = n_train
        captured["split_device_arg"] = device
        captured["split_y_device"] = y.device.type
        captured["split_rng_device"] = str(generator.device)
        raise _SplitSentinel

    monkeypatch.setattr(
        "dagzoo.core.generation_runtime._stratified_split_indices",
        _stub_stratified_split_indices,
    )

    with pytest.raises(_SplitSentinel):
        _finalize_generated_tensors(
            cfg,
            layout=_layout_stub(
                feature_types=["num", "num", "num", "num"],
                graph_nodes=4,
                adjacency=torch.zeros((4, 4), dtype=torch.bool),
                feature_node_assignment=[0, 1, 2, 3],
                target_node_assignment=3,
            ),
            dataset_seed=111,
            attempt=0,
            attempts_used=1,
            dataset_root=KeyedRng(111),
            device="cuda",
            n_train=8,
            n_test=4,
            requested_device="cuda",
            resolved_device="cuda",
            device_fallback_reason=None,
            x=torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4),
            y=torch.arange(12, dtype=torch.int64) % 3,
            aux_meta={"filter": {"enabled": False}},
            shift_params=resolve_shift_runtime_params(cfg),
            noise_runtime_selection=SimpleNamespace(),
            dtype=torch.float32,
        )

    assert captured["split_device_arg"] == "cpu"
    assert captured["split_y_device"] == "cpu"
    assert captured["split_rng_device"] == "cpu"


def test_generate_torch_routes_postprocess_to_runtime_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    captured: dict[str, str] = {}

    class _PostprocessSentinel(Exception):
        pass

    def _stub_postprocess_dataset(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        _feature_types: list[str],
        _task: str,
        keyed_rng: KeyedRng,
        device: str,
        *,
        return_feature_index_map: bool = False,
        preserve_feature_schema: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]:
        _ = return_feature_index_map
        _ = preserve_feature_schema
        _ = y_train
        _ = y_test
        captured["postprocess_device_arg"] = device
        captured["postprocess_x_train_device"] = x_train.device.type
        captured["postprocess_x_test_device"] = x_test.device.type
        captured["postprocess_rng_device"] = str(keyed_rng.torch_rng(device="cpu").device)
        raise _PostprocessSentinel

    monkeypatch.setattr(
        "dagzoo.core.generation_runtime.postprocess_dataset",
        _stub_postprocess_dataset,
    )

    with pytest.raises(_PostprocessSentinel):
        _finalize_generated_tensors(
            cfg,
            layout=_layout_stub(
                feature_types=["num", "num", "num", "num"],
                graph_nodes=4,
                adjacency=torch.zeros((4, 4), dtype=torch.bool),
                feature_node_assignment=[0, 1, 2, 3],
                target_node_assignment=3,
            ),
            dataset_seed=222,
            attempt=0,
            attempts_used=1,
            dataset_root=KeyedRng(222),
            device="cuda",
            n_train=8,
            n_test=4,
            requested_device="cuda",
            resolved_device="cuda",
            device_fallback_reason=None,
            x=torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4),
            y=torch.linspace(0.0, 1.0, 12, dtype=torch.float32),
            aux_meta={"filter": {"enabled": False}},
            shift_params=resolve_shift_runtime_params(cfg),
            noise_runtime_selection=SimpleNamespace(),
            dtype=torch.float32,
        )

    assert captured["postprocess_device_arg"] == "cuda"
    assert captured["postprocess_x_train_device"] == "cpu"
    assert captured["postprocess_x_test_device"] == "cpu"
    assert captured["postprocess_rng_device"] == "cpu"


def test_generate_torch_routes_missingness_to_runtime_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.missing_rate = 0.25
    cfg.dataset.missing_mechanism = "mcar"
    captured: dict[str, str] = {}

    class _MissingnessSentinel(Exception):
        pass

    def _stub_postprocess_dataset(
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        feature_types: list[str],
        _task: str,
        _keyed_rng: KeyedRng,
        _device: str,
        *,
        return_feature_index_map: bool = False,
        preserve_feature_schema: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]:
        _ = return_feature_index_map
        _ = preserve_feature_schema
        feature_index_map = list(range(x_train.shape[1]))
        return x_train, y_train, x_test, y_test, feature_types, feature_index_map

    def _stub_inject_missingness(
        x_train: torch.Tensor,
        x_test: torch.Tensor,
        *,
        dataset_cfg,
        keyed_rng: KeyedRng,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object] | None]:
        _ = dataset_cfg
        _ = keyed_rng
        captured["missingness_device_arg"] = device
        captured["missingness_x_train_device"] = x_train.device.type
        captured["missingness_x_test_device"] = x_test.device.type
        raise _MissingnessSentinel

    monkeypatch.setattr(
        "dagzoo.core.generation_runtime.postprocess_dataset",
        _stub_postprocess_dataset,
    )
    monkeypatch.setattr(
        "dagzoo.core.generation_runtime.inject_missingness",
        _stub_inject_missingness,
    )

    with pytest.raises(_MissingnessSentinel):
        _finalize_generated_tensors(
            cfg,
            layout=_layout_stub(
                feature_types=["num", "num", "num", "num"],
                graph_nodes=4,
                adjacency=torch.zeros((4, 4), dtype=torch.bool),
                feature_node_assignment=[0, 1, 2, 3],
                target_node_assignment=3,
            ),
            dataset_seed=333,
            attempt=0,
            attempts_used=1,
            dataset_root=KeyedRng(333),
            device="cuda",
            n_train=8,
            n_test=4,
            requested_device="cuda",
            resolved_device="cuda",
            device_fallback_reason=None,
            x=torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4),
            y=torch.linspace(0.0, 1.0, 12, dtype=torch.float32),
            aux_meta={"filter": {"enabled": False}},
            shift_params=resolve_shift_runtime_params(cfg),
            noise_runtime_selection=SimpleNamespace(),
            dtype=torch.float32,
        )

    assert captured["missingness_device_arg"] == "cuda"
    assert captured["missingness_x_train_device"] == "cpu"
    assert captured["missingness_x_test_device"] == "cpu"


def test_finalize_generated_tensors_skips_missingness_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    layout = _layout_stub(
        feature_types=["num", "num", "num", "num"],
        graph_nodes=4,
        adjacency=torch.zeros((4, 4), dtype=torch.bool),
        feature_node_assignment=[0, 1, 2, 3],
        target_node_assignment=3,
    )

    monkeypatch.setattr(
        "dagzoo.core.generation_runtime.inject_missingness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missingness helper should be skipped when disabled")
        ),
    )

    bundle = _finalize_generated_tensors(
        cfg,
        layout,
        dataset_seed=334,
        attempt=0,
        attempts_used=1,
        dataset_root=KeyedRng(334),
        device="cpu",
        n_train=8,
        n_test=4,
        requested_device="cpu",
        resolved_device="cpu",
        device_fallback_reason=None,
        x=torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4),
        y=torch.linspace(0.0, 1.0, 12, dtype=torch.float32),
        aux_meta={"filter": {"enabled": False}},
        shift_params=resolve_shift_runtime_params(cfg),
        noise_runtime_selection=NoiseRuntimeSelection(
            family_requested="gaussian",
            family_sampled="gaussian",
            sampling_strategy="global",
            base_scale=1.0,
            student_t_df=5.0,
            mixture_weights=None,
        ),
        dtype=torch.float32,
    )

    assert "missingness" not in bundle.metadata


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_finalize_generated_chunk_preserve_schema_matches_scalar_helper(task: str) -> None:
    cfg = _tiny_regression_config() if task == "regression" else _tiny_config()
    cfg.dataset.task = task
    cfg.filter.enabled = False
    cfg.dataset.n_train = 8
    cfg.dataset.n_test = 4

    layout = _layout_stub(
        feature_types=["num", "num", "num", "num"],
        graph_nodes=4,
        adjacency=torch.zeros((4, 4), dtype=torch.bool),
        feature_node_assignment=[0, 1, 2, 3],
        target_node_assignment=3,
    )
    shift_params = resolve_shift_runtime_params(cfg)
    selection = NoiseRuntimeSelection(
        family_requested="gaussian",
        family_sampled="gaussian",
        sampling_strategy="global",
        base_scale=1.0,
        student_t_df=5.0,
        mixture_weights=None,
    )
    seeds = [111, 222]
    x = torch.arange(len(seeds) * 12 * 4, dtype=torch.float32).reshape(len(seeds), 12, 4) / 10.0
    if task == "classification":
        y = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2],
            ],
            dtype=torch.int64,
        )
    else:
        y = torch.linspace(0.0, 1.0, len(seeds) * 12, dtype=torch.float32).reshape(len(seeds), 12)

    context = _build_fixed_schema_finalization_context(
        cfg,
        layout,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        shift_params=shift_params,
    )
    batch_bundles = _finalize_generated_chunk_preserve_schema(
        cfg,
        layout,
        context=context,
        dataset_roots=[KeyedRng(seed) for seed in seeds],
        attempt=0,
        attempts_used=1,
        device="cpu",
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        requested_device="cpu",
        resolved_device="cpu",
        device_fallback_reason=None,
        x=x,
        y=y,
        aux_meta_batch=[{"filter": {"mode": "deferred", "status": "not_run"}} for _ in seeds],
        noise_runtime_selection=selection,
        dtype=torch.float32,
    )

    scalar_bundles = [
        _finalize_generated_tensors(
            cfg,
            layout,
            dataset_seed=KeyedRng(seed).child_seed(),
            attempt=0,
            attempts_used=1,
            dataset_root=KeyedRng(seed),
            device="cpu",
            n_train=cfg.dataset.n_train,
            n_test=cfg.dataset.n_test,
            requested_device="cpu",
            resolved_device="cpu",
            device_fallback_reason=None,
            x=x[index],
            y=y[index],
            aux_meta={"filter": {"mode": "deferred", "status": "not_run"}},
            shift_params=shift_params,
            noise_runtime_selection=selection,
            dtype=torch.float32,
            preserve_feature_schema=True,
        )
        for index, seed in enumerate(seeds)
    ]

    for batched, scalar in zip(batch_bundles, scalar_bundles, strict=True):
        assert batched is not None
        torch.testing.assert_close(batched.X_train, scalar.X_train)
        torch.testing.assert_close(batched.y_train, scalar.y_train)
        torch.testing.assert_close(batched.X_test, scalar.X_test)
        torch.testing.assert_close(batched.y_test, scalar.y_test)
        assert batched.feature_types == scalar.feature_types
        assert batched.metadata == scalar.metadata


def test_finalize_generated_chunk_preserve_schema_reuses_provided_split_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.dataset.n_train = 8
    cfg.dataset.n_test = 4
    layout = _layout_stub(
        feature_types=["num", "num", "num", "num"],
        graph_nodes=4,
        adjacency=torch.zeros((4, 4), dtype=torch.bool),
        feature_node_assignment=[0, 1, 2, 3],
        target_node_assignment=3,
    )
    shift_params = resolve_shift_runtime_params(cfg)
    context = _build_fixed_schema_finalization_context(
        cfg,
        layout,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        shift_params=shift_params,
    )
    y = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2],
        ],
        dtype=torch.int64,
    )
    resolved_split_indices = [
        (torch.tensor([0, 1, 4, 5, 8, 9, 2, 6]), torch.tensor([3, 7, 10, 11])),
        (torch.tensor([0, 1, 2, 4, 5, 7, 8, 10]), torch.tensor([3, 6, 9, 11])),
    ]

    monkeypatch.setattr(
        "dagzoo.core.generation_runtime._resolve_split_indices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("split indices should be reused from the grouped runtime pass")
        ),
    )

    bundles = _finalize_generated_chunk_preserve_schema(
        cfg,
        layout,
        context=context,
        dataset_roots=[KeyedRng(111), KeyedRng(222)],
        attempt=0,
        attempts_used=1,
        device="cpu",
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        requested_device="cpu",
        resolved_device="cpu",
        device_fallback_reason=None,
        x=torch.arange(2 * 12 * 4, dtype=torch.float32).reshape(2, 12, 4) / 10.0,
        y=y,
        aux_meta_batch=[{"filter": {"mode": "deferred", "status": "not_run"}} for _ in range(2)],
        noise_runtime_selection=NoiseRuntimeSelection(
            family_requested="gaussian",
            family_sampled="gaussian",
            sampling_strategy="global",
            base_scale=1.0,
            student_t_df=5.0,
            mixture_weights=None,
        ),
        dtype=torch.float32,
        resolved_split_indices=resolved_split_indices,
    )

    assert all(bundle is not None for bundle in bundles)


def test_finalize_generated_chunk_preserve_schema_copies_metadata_templates() -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    layout = _layout_stub(
        feature_types=["num", "num"],
        graph_nodes=2,
        adjacency=torch.zeros((2, 2), dtype=torch.bool),
        feature_node_assignment=[0, 1],
        target_node_assignment=1,
    )
    shift_params = resolve_shift_runtime_params(cfg)
    selection = NoiseRuntimeSelection(
        family_requested="gaussian",
        family_sampled="gaussian",
        sampling_strategy="global",
        base_scale=1.0,
        student_t_df=5.0,
        mixture_weights=None,
    )
    context = _build_fixed_schema_finalization_context(
        cfg,
        layout,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        shift_params=shift_params,
    )
    bundles = _finalize_generated_chunk_preserve_schema(
        cfg,
        layout,
        context=context,
        dataset_roots=[KeyedRng(13), KeyedRng(17)],
        attempt=0,
        attempts_used=1,
        device="cpu",
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        requested_device="cpu",
        resolved_device="cpu",
        device_fallback_reason=None,
        x=torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2),
        y=torch.linspace(0.0, 1.0, 12, dtype=torch.float32).reshape(2, 6),
        aux_meta_batch=[{"filter": {"mode": "deferred", "status": "not_run"}} for _ in range(2)],
        noise_runtime_selection=selection,
        dtype=torch.float32,
    )

    assert bundles[0] is not None and bundles[1] is not None
    assert bundles[0].metadata is not bundles[1].metadata
    assert bundles[0].metadata["config"] is not bundles[1].metadata["config"]
    bundles[0].metadata["config"]["dataset"]["n_train"] = -1
    assert bundles[1].metadata["config"]["dataset"]["n_train"] == cfg.dataset.n_train


def test_finalize_generated_chunk_preserve_schema_supports_per_dataset_contexts() -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    varied_cfg = deepcopy(cfg)
    varied_cfg.dataset.missing_rate = 0.25
    varied_cfg.dataset.missing_mechanism = "mcar"
    varied_cfg.validate_generation_constraints()
    layout = _layout_stub(
        feature_types=["num", "num"],
        graph_nodes=2,
        adjacency=torch.zeros((2, 2), dtype=torch.bool),
        feature_node_assignment=[0, 1],
        target_node_assignment=1,
    )
    selection = NoiseRuntimeSelection(
        family_requested="gaussian",
        family_sampled="gaussian",
        sampling_strategy="global",
        base_scale=1.0,
        student_t_df=5.0,
        mixture_weights=None,
    )
    base_context = _build_fixed_schema_finalization_context(
        cfg,
        layout,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        shift_params=resolve_shift_runtime_params(cfg),
    )
    varied_context = _build_fixed_schema_finalization_context(
        varied_cfg,
        layout,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        shift_params=resolve_shift_runtime_params(varied_cfg),
    )

    bundles = _finalize_generated_chunk_preserve_schema(
        cfg,
        layout,
        context=base_context,
        contexts_by_batch=[base_context, varied_context],
        configs_by_batch=[cfg, varied_cfg],
        dataset_roots=[KeyedRng(13), KeyedRng(17)],
        attempt=0,
        attempts_used=1,
        device="cpu",
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        requested_device="cpu",
        resolved_device="cpu",
        device_fallback_reason=None,
        x=torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2),
        y=torch.linspace(0.0, 1.0, 12, dtype=torch.float32).reshape(2, 6),
        aux_meta_batch=[{"filter": {"mode": "deferred", "status": "not_run"}} for _ in range(2)],
        noise_runtime_selection=selection,
        dtype=torch.float32,
    )

    assert bundles[0] is not None and bundles[1] is not None
    assert bundles[0].metadata["config"]["dataset"]["missing_rate"] == pytest.approx(0.0)
    assert bundles[1].metadata["config"]["dataset"]["missing_rate"] == pytest.approx(0.25)
    assert "missingness" not in bundles[0].metadata
    assert "missingness" in bundles[1].metadata
    assert bundles[0].metadata["noise_distribution"]["family_requested"] == "gaussian"
    assert bundles[1].metadata["noise_distribution"]["family_requested"] == "gaussian"


def test_finalize_generated_chunk_preserve_schema_rejects_misaligned_contexts() -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    layout = _layout_stub(
        feature_types=["num", "num"],
        graph_nodes=2,
        adjacency=torch.zeros((2, 2), dtype=torch.bool),
        feature_node_assignment=[0, 1],
        target_node_assignment=1,
    )
    selection = NoiseRuntimeSelection(
        family_requested="gaussian",
        family_sampled="gaussian",
        sampling_strategy="global",
        base_scale=1.0,
        student_t_df=5.0,
        mixture_weights=None,
    )
    context = _build_fixed_schema_finalization_context(
        cfg,
        layout,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        shift_params=resolve_shift_runtime_params(cfg),
    )

    with pytest.raises(
        ValueError, match="contexts_by_batch must align with provided dataset roots"
    ):
        _finalize_generated_chunk_preserve_schema(
            cfg,
            layout,
            context=context,
            contexts_by_batch=[context],
            configs_by_batch=None,
            dataset_roots=[KeyedRng(13), KeyedRng(17)],
            attempt=0,
            attempts_used=1,
            device="cpu",
            n_train=cfg.dataset.n_train,
            n_test=cfg.dataset.n_test,
            requested_device="cpu",
            resolved_device="cpu",
            device_fallback_reason=None,
            x=torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2),
            y=torch.linspace(0.0, 1.0, 12, dtype=torch.float32).reshape(2, 6),
            aux_meta_batch=[
                {"filter": {"mode": "deferred", "status": "not_run"}} for _ in range(2)
            ],
            noise_runtime_selection=selection,
            dtype=torch.float32,
        )


def test_finalize_generated_chunk_preserve_schema_rejects_misaligned_configs() -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    layout = _layout_stub(
        feature_types=["num", "num"],
        graph_nodes=2,
        adjacency=torch.zeros((2, 2), dtype=torch.bool),
        feature_node_assignment=[0, 1],
        target_node_assignment=1,
    )
    selection = NoiseRuntimeSelection(
        family_requested="gaussian",
        family_sampled="gaussian",
        sampling_strategy="global",
        base_scale=1.0,
        student_t_df=5.0,
        mixture_weights=None,
    )
    context = _build_fixed_schema_finalization_context(
        cfg,
        layout,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        shift_params=resolve_shift_runtime_params(cfg),
    )

    with pytest.raises(ValueError, match="configs_by_batch must align with provided dataset roots"):
        _finalize_generated_chunk_preserve_schema(
            cfg,
            layout,
            context=context,
            contexts_by_batch=None,
            configs_by_batch=[cfg],
            dataset_roots=[KeyedRng(13), KeyedRng(17)],
            attempt=0,
            attempts_used=1,
            device="cpu",
            n_train=cfg.dataset.n_train,
            n_test=cfg.dataset.n_test,
            requested_device="cpu",
            resolved_device="cpu",
            device_fallback_reason=None,
            x=torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2),
            y=torch.linspace(0.0, 1.0, 12, dtype=torch.float32).reshape(2, 6),
            aux_meta_batch=[
                {"filter": {"mode": "deferred", "status": "not_run"}} for _ in range(2)
            ],
            noise_runtime_selection=selection,
            dtype=torch.float32,
        )


def test_finalize_generated_chunk_preserve_schema_reuses_cached_lineage_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    layout = _layout_stub(
        feature_types=["num", "cat"],
        graph_nodes=2,
        adjacency=torch.tensor([[0, 1], [0, 0]], dtype=torch.bool),
        feature_node_assignment=[0, 1],
        target_node_assignment=1,
    )
    shift_params = resolve_shift_runtime_params(cfg)
    selection = NoiseRuntimeSelection(
        family_requested="gaussian",
        family_sampled="gaussian",
        sampling_strategy="global",
        base_scale=1.0,
        student_t_df=5.0,
        mixture_weights=None,
    )
    actual_build_lineage = dagzoo.core.generation_runtime._build_lineage_metadata
    lineage_calls = 0

    def _counting_build_lineage(*args, **kwargs):
        nonlocal lineage_calls
        lineage_calls += 1
        return actual_build_lineage(*args, **kwargs)

    monkeypatch.setattr(
        "dagzoo.core.generation_runtime._build_lineage_metadata",
        _counting_build_lineage,
    )

    context = _build_fixed_schema_finalization_context(
        cfg,
        layout,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        shift_params=shift_params,
    )

    assert lineage_calls == 1

    bundles = _finalize_generated_chunk_preserve_schema(
        cfg,
        layout,
        context=context,
        dataset_roots=[KeyedRng(13), KeyedRng(17)],
        attempt=0,
        attempts_used=1,
        device="cpu",
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        requested_device="cpu",
        resolved_device="cpu",
        device_fallback_reason=None,
        x=torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2),
        y=torch.linspace(0.0, 1.0, 12, dtype=torch.float32).reshape(2, 6),
        aux_meta_batch=[{"filter": {"mode": "deferred", "status": "not_run"}} for _ in range(2)],
        noise_runtime_selection=selection,
        dtype=torch.float32,
    )

    assert lineage_calls == 1
    assert all(bundle is not None for bundle in bundles)


@pytest.mark.parametrize(
    ("y_train", "y_test", "expected"),
    [
        (
            torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
            True,
        ),
        (
            torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            torch.tensor([0, 2], dtype=torch.int64),
            False,
        ),
        (
            torch.tensor([1, 1, 1, 1], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
            False,
        ),
    ],
)
def test_classification_split_valid_tensor_only(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    expected: bool,
) -> None:
    assert _classification_split_valid(y_train, y_test) is expected


def test_finalize_generated_chunk_preserve_schema_omits_unset_fixed_layout_target_cells() -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    assert cfg.runtime.fixed_layout_target_cells is None
    layout = _layout_stub(
        feature_types=["num", "num"],
        graph_nodes=2,
        adjacency=torch.zeros((2, 2), dtype=torch.bool),
        feature_node_assignment=[0, 1],
        target_node_assignment=1,
    )
    shift_params = resolve_shift_runtime_params(cfg)
    selection = NoiseRuntimeSelection(
        family_requested="gaussian",
        family_sampled="gaussian",
        sampling_strategy="global",
        base_scale=1.0,
        student_t_df=5.0,
        mixture_weights=None,
    )
    context = _build_fixed_schema_finalization_context(
        cfg,
        layout,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        shift_params=shift_params,
    )
    bundles = _finalize_generated_chunk_preserve_schema(
        cfg,
        layout,
        context=context,
        dataset_roots=[KeyedRng(23)],
        attempt=0,
        attempts_used=1,
        device="cpu",
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        requested_device="cpu",
        resolved_device="cpu",
        device_fallback_reason=None,
        x=torch.arange(1 * 6 * 2, dtype=torch.float32).reshape(1, 6, 2),
        y=torch.linspace(0.0, 1.0, 6, dtype=torch.float32).reshape(1, 6),
        aux_meta_batch=[{"filter": {"mode": "deferred", "status": "not_run"}}],
        noise_runtime_selection=selection,
        dtype=torch.float32,
    )

    assert bundles[0] is not None
    runtime_config = bundles[0].metadata["config"]["runtime"]
    assert "fixed_layout_target_cells" not in runtime_config


def test_generate_batch_iter_matches_batch_ordering() -> None:
    cfg = _tiny_config()
    batch = generate_batch(cfg, num_datasets=2, seed=321, device="cpu")
    streamed = list(generate_batch_iter(cfg, num_datasets=2, seed=321, device="cpu"))

    assert len(streamed) == len(batch)
    for a, b in zip(batch, streamed, strict=True):
        np.testing.assert_allclose(np.asarray(a.X_train), np.asarray(b.X_train), atol=1e-6)
        assert a.metadata["seed"] == b.metadata["seed"]
        assert a.metadata["dataset_seed"] == b.metadata["dataset_seed"]
        assert a.metadata["dataset_index"] == b.metadata["dataset_index"]
        assert a.metadata["dataset_id"] == b.metadata["dataset_id"]
        assert a.metadata["run_num_datasets"] == b.metadata["run_num_datasets"]
        assert a.metadata["layout_plan_signature"] == b.metadata["layout_plan_signature"]
        assert a.metadata["split_groups"] == b.metadata["split_groups"]


def test_generate_one_matches_first_dataset_of_generate_batch() -> None:
    cfg = _tiny_regression_config()

    single = generate_one(cfg, seed=4321, device="cpu")
    batch = generate_batch(cfg, num_datasets=1, seed=4321, device="cpu")

    assert len(batch) == 1
    np.testing.assert_allclose(np.asarray(single.X_train), np.asarray(batch[0].X_train), atol=1e-6)
    np.testing.assert_allclose(np.asarray(single.X_test), np.asarray(batch[0].X_test), atol=1e-6)
    np.testing.assert_allclose(np.asarray(single.y_train), np.asarray(batch[0].y_train), atol=1e-6)
    np.testing.assert_allclose(np.asarray(single.y_test), np.asarray(batch[0].y_test), atol=1e-6)
    assert int(single.metadata["seed"]) == 4321
    assert int(batch[0].metadata["seed"]) == 4321
    assert int(single.metadata["dataset_seed"]) == int(batch[0].metadata["dataset_seed"])
    assert int(single.metadata["dataset_index"]) == 0
    assert int(batch[0].metadata["dataset_index"]) == 0
    assert str(single.metadata["dataset_id"]) == str(batch[0].metadata["dataset_id"])
    assert int(single.metadata["run_num_datasets"]) == 1
    assert int(batch[0].metadata["run_num_datasets"]) == 1
    assert single.metadata["layout_plan_signature"] == batch[0].metadata["layout_plan_signature"]
    assert single.metadata["split_groups"] == batch[0].metadata["split_groups"]


def test_generate_batch_metadata_preserves_run_seed_and_dataset_indices() -> None:
    cfg = _tiny_regression_config()

    batch = generate_batch(cfg, num_datasets=3, seed=4321, device="cpu")

    assert [int(bundle.metadata["seed"]) for bundle in batch] == [4321, 4321, 4321]
    assert [int(bundle.metadata["dataset_index"]) for bundle in batch] == [0, 1, 2]
    assert [int(bundle.metadata["run_num_datasets"]) for bundle in batch] == [3, 3, 3]
    dataset_seeds = [int(bundle.metadata["dataset_seed"]) for bundle in batch]
    dataset_ids = [str(bundle.metadata["dataset_id"]) for bundle in batch]
    request_run_groups = [bundle.metadata["split_groups"]["request_run"] for bundle in batch]
    layout_plan_groups = [bundle.metadata["split_groups"]["layout_plan"] for bundle in batch]
    assert len(set(dataset_seeds)) == 3
    assert len(set(dataset_ids)) == 3
    assert len(set(request_run_groups)) == 1
    assert len(set(layout_plan_groups)) == 1


def test_generate_one_request_run_identity_changes_with_noise_contract() -> None:
    baseline = _tiny_regression_config()
    drifted = deepcopy(baseline)
    baseline.noise.family = NOISE_FAMILY_GAUSSIAN
    drifted.noise.family = NOISE_FAMILY_LAPLACE

    bundle_base = generate_one(baseline, seed=1234, device="cpu")
    bundle_drifted = generate_one(drifted, seed=1234, device="cpu")

    assert bundle_base.metadata["noise_distribution"]["family_sampled"] == NOISE_FAMILY_GAUSSIAN
    assert bundle_drifted.metadata["noise_distribution"]["family_sampled"] == NOISE_FAMILY_LAPLACE
    assert (
        bundle_base.metadata["layout_plan_signature"]
        == bundle_drifted.metadata["layout_plan_signature"]
    )
    assert (
        bundle_base.metadata["split_groups"]["layout_plan"]
        == bundle_drifted.metadata["split_groups"]["layout_plan"]
    )
    assert (
        bundle_base.metadata["split_groups"]["request_run"]
        != bundle_drifted.metadata["split_groups"]["request_run"]
    )
    assert bundle_base.metadata["dataset_id"] != bundle_drifted.metadata["dataset_id"]


def test_generate_one_request_run_identity_changes_with_realized_row_shape() -> None:
    baseline = _tiny_regression_config()
    drifted = deepcopy(baseline)
    baseline.dataset.n_train = 16
    baseline.dataset.n_test = 8
    drifted.dataset.n_train = 32
    drifted.dataset.n_test = 8

    bundle_base = generate_one(baseline, seed=1234, device="cpu")
    bundle_drifted = generate_one(drifted, seed=1234, device="cpu")

    assert bundle_base.X_train.shape != bundle_drifted.X_train.shape
    assert (
        bundle_base.metadata["layout_plan_signature"]
        == bundle_drifted.metadata["layout_plan_signature"]
    )
    assert (
        bundle_base.metadata["split_groups"]["layout_plan"]
        == bundle_drifted.metadata["split_groups"]["layout_plan"]
    )
    assert (
        bundle_base.metadata["split_groups"]["request_run"]
        != bundle_drifted.metadata["split_groups"]["request_run"]
    )
    assert bundle_base.metadata["dataset_id"] != bundle_drifted.metadata["dataset_id"]


def test_generate_one_request_run_identity_ignores_output_path_changes() -> None:
    baseline = _tiny_regression_config()
    relocated = deepcopy(baseline)
    relocated.output.out_dir = "data/run_relocated"

    bundle_base = generate_one(baseline, seed=1234, device="cpu")
    bundle_relocated = generate_one(relocated, seed=1234, device="cpu")

    assert (
        bundle_base.metadata["config"]["output"]["out_dir"]
        != bundle_relocated.metadata["config"]["output"]["out_dir"]
    )
    assert bundle_base.metadata["split_groups"] == bundle_relocated.metadata["split_groups"]
    assert bundle_base.metadata["dataset_id"] == bundle_relocated.metadata["dataset_id"]


def test_generate_batch_request_run_identity_is_run_stable_for_mixture_noise() -> None:
    cfg = _tiny_regression_config()
    cfg.noise.family = NOISE_FAMILY_MIXTURE
    cfg.noise.mixture_weights = {"gaussian": 0.7, "laplace": 0.2, "student_t": 0.1}

    batch = generate_batch(cfg, num_datasets=5, seed=1, device="cpu")

    sampled_families = [bundle.metadata["noise_distribution"]["family_sampled"] for bundle in batch]
    request_run_groups = [bundle.metadata["split_groups"]["request_run"] for bundle in batch]
    assert len(set(sampled_families)) > 1
    assert len(set(request_run_groups)) == 1


def test_generate_batch_request_run_identity_changes_with_fixed_layout_target_cells() -> None:
    baseline = _tiny_regression_config()
    drifted = deepcopy(baseline)
    baseline.runtime.fixed_layout_target_cells = 1_000_000
    drifted.runtime.fixed_layout_target_cells = 16

    batch_base = generate_batch(baseline, num_datasets=5, seed=1234, device="cpu")
    batch_drifted = generate_batch(drifted, num_datasets=5, seed=1234, device="cpu")

    assert (
        batch_base[0].metadata["layout_plan_signature"]
        == batch_drifted[0].metadata["layout_plan_signature"]
    )
    assert (
        batch_base[0].metadata["split_groups"]["layout_plan"]
        == batch_drifted[0].metadata["split_groups"]["layout_plan"]
    )
    assert not np.array_equal(
        np.asarray(batch_base[0].X_train), np.asarray(batch_drifted[0].X_train)
    )
    assert not np.array_equal(
        np.asarray(batch_base[0].y_train), np.asarray(batch_drifted[0].y_train)
    )
    assert (
        batch_base[0].metadata["split_groups"]["request_run"]
        != batch_drifted[0].metadata["split_groups"]["request_run"]
    )
    assert batch_base[0].metadata["dataset_id"] != batch_drifted[0].metadata["dataset_id"]


def test_generate_batch_request_run_identity_changes_with_target_parent_prior() -> None:
    baseline = _tiny_regression_config()
    drifted = deepcopy(baseline)
    drifted.dataset.target_parent_prior = "all_features"

    batch_base = generate_batch(baseline, num_datasets=3, seed=4242, device="cpu")
    batch_drifted = generate_batch(drifted, num_datasets=3, seed=4242, device="cpu")

    assert (
        batch_base[0].metadata["split_groups"]["request_run"]
        != batch_drifted[0].metadata["split_groups"]["request_run"]
    )
    assert batch_base[0].metadata["dataset_id"] != batch_drifted[0].metadata["dataset_id"]


def test_generate_batch_request_run_identity_normalizes_default_fixed_layout_target_cells() -> None:
    baseline = _tiny_regression_config()
    explicit_default = deepcopy(baseline)
    explicit_default.runtime.fixed_layout_target_cells = 4_000_000

    batch_base = generate_batch(baseline, num_datasets=5, seed=1234, device="cpu")
    batch_explicit = generate_batch(explicit_default, num_datasets=5, seed=1234, device="cpu")

    for bundle_base, bundle_explicit in zip(batch_base, batch_explicit, strict=True):
        np.testing.assert_allclose(
            np.asarray(bundle_base.X_train), np.asarray(bundle_explicit.X_train), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(bundle_base.X_test), np.asarray(bundle_explicit.X_test), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(bundle_base.y_train), np.asarray(bundle_explicit.y_train), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(bundle_base.y_test), np.asarray(bundle_explicit.y_test), atol=1e-6
        )
        assert bundle_base.metadata["split_groups"] == bundle_explicit.metadata["split_groups"]
        assert bundle_base.metadata["dataset_id"] == bundle_explicit.metadata["dataset_id"]


def test_generate_batch_bundle_replays_from_run_metadata() -> None:
    cfg = _tiny_regression_config()

    batch = generate_batch(cfg, num_datasets=3, seed=4321, device="cpu")

    for bundle in (batch[0], batch[2]):
        dataset_index = int(bundle.metadata["dataset_index"])
        replayed = generate_batch(
            cfg,
            num_datasets=int(bundle.metadata["run_num_datasets"]),
            seed=int(bundle.metadata["seed"]),
            device="cpu",
        )[dataset_index]

        np.testing.assert_allclose(
            np.asarray(bundle.X_train), np.asarray(replayed.X_train), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(bundle.X_test), np.asarray(replayed.X_test), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(bundle.y_train), np.asarray(replayed.y_train), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(bundle.y_test), np.asarray(replayed.y_test), atol=1e-6
        )
        assert int(bundle.metadata["seed"]) == int(replayed.metadata["seed"])
        assert int(bundle.metadata["dataset_seed"]) == int(replayed.metadata["dataset_seed"])
        assert int(bundle.metadata["dataset_index"]) == int(replayed.metadata["dataset_index"])
        assert str(bundle.metadata["dataset_id"]) == str(replayed.metadata["dataset_id"])
        assert int(bundle.metadata["run_num_datasets"]) == int(
            replayed.metadata["run_num_datasets"]
        )
        assert (
            bundle.metadata["layout_plan_signature"] == replayed.metadata["layout_plan_signature"]
        )
        assert bundle.metadata["split_groups"] == replayed.metadata["split_groups"]


def test_sample_fixed_layout_rejects_variable_rows_spec() -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.rows = "400..60000"  # type: ignore[assignment]

    with pytest.raises(ValueError, match=r"variable dataset\.rows"):
        _sample_fixed_layout(cfg, seed=90209, device="cpu")


def test_sample_fixed_layout_accepts_fixed_rows_spec() -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.rows = 1024  # type: ignore[assignment]
    cfg.dataset.n_test = 256
    cfg.dataset.n_train = 64

    plan = _sample_fixed_layout(cfg, seed=90208, device="cpu")
    assert plan.n_train == 768
    assert plan.n_test == 256


def test_sample_fixed_layout_is_deterministic_for_seed() -> None:
    cfg = _tiny_regression_config()
    plan_a = _sample_fixed_layout(cfg, seed=90210, device="cpu")
    plan_b = _sample_fixed_layout(cfg, seed=90210, device="cpu")

    assert plan_a.layout_signature == plan_b.layout_signature
    assert plan_a.plan_seed == plan_b.plan_seed
    assert plan_a.plan_signature == plan_b.plan_signature
    assert plan_a.execution_plan == plan_b.execution_plan
    assert int(plan_a.layout.n_features) == int(plan_b.layout.n_features)
    assert list(plan_a.layout.feature_types) == list(plan_b.layout.feature_types)


def test_sample_fixed_layout_propagates_mechanism_drift_tilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.shift.enabled = True
    cfg.shift.mode = "mechanism_drift"
    observed_tilts: list[float] = []

    def _stub_sample_function_family(
        _generator=None,
        *,
        keyed_rng=None,
        mechanism_logit_tilt: float,
        function_family_mix=None,
        device=None,
    ) -> str:
        _ = keyed_rng
        _ = function_family_mix
        _ = device
        observed_tilts.append(float(mechanism_logit_tilt))
        return "linear"

    monkeypatch.setattr(
        "dagzoo.core.execution_semantics.sample_function_family",
        _stub_sample_function_family,
    )

    runtime = resolve_shift_runtime_params(cfg)
    _ = _sample_fixed_layout(cfg, seed=90210, device="cpu")

    assert observed_tilts
    assert all(tilt == pytest.approx(runtime.mechanism_logit_tilt) for tilt in observed_tilts)


def test_generate_batch_with_plan_iter_groups_mixed_noise_runtime_subbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    run_seed = 321
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=11,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )
    run_root = KeyedRng(run_seed)
    dataset_roots = [run_root.keyed("dataset", idx) for idx in range(4)]
    dataset_seeds = [dataset_root.child_seed() for dataset_root in dataset_roots]
    family_pattern = ["gaussian", "gaussian", "laplace", "gaussian"]
    family_by_noise_root_seed = {
        dataset_root.keyed("noise_runtime").child_seed(): family
        for dataset_root, family in zip(dataset_roots, family_pattern, strict=True)
    }
    grouped_call_sizes: list[int] = []

    def _stub_resolve_noise_runtime_selection(
        _config,
        *,
        keyed_rng: KeyedRng,
        device: str = "cpu",
    ) -> NoiseRuntimeSelection:
        _ = device
        family = family_by_noise_root_seed[int(keyed_rng.child_seed())]
        return NoiseRuntimeSelection(
            family_requested="mixture",
            family_sampled=family,
            sampling_strategy="dataset_level",
            base_scale=1.0,
            student_t_df=5.0,
            mixture_weights={"gaussian": 0.5, "laplace": 0.5},
        )

    def _stub_generate_fixed_layout_graph_batch(
        _config,
        _layout,
        *,
        execution_plan,
        dataset_seeds,
        device,
        noise_sigma_multiplier,
        noise_spec,
        runtime_metrics_out=None,
    ):
        _ = execution_plan
        _ = device
        _ = noise_sigma_multiplier
        _ = noise_spec
        _ = runtime_metrics_out
        grouped_call_sizes.append(len(dataset_seeds))
        batch_size = len(dataset_seeds)
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return (
            torch.zeros((batch_size, n_rows, 2), dtype=torch.float32),
            torch.zeros((batch_size, n_rows), dtype=torch.float32),
            [{} for _ in dataset_seeds],
        )

    def _stub_finalize_generated_chunk_preserve_schema(
        _config,
        _layout,
        *,
        context,
        dataset_roots,
        attempt,
        attempts_used,
        device,
        n_train,
        n_test,
        requested_device,
        resolved_device,
        device_fallback_reason,
        x,
        y,
        aux_meta_batch,
        noise_runtime_selection,
        dtype,
        resolved_split_indices=None,
    ) -> list[DatasetBundle | None]:
        _ = context
        _ = attempt
        _ = attempts_used
        _ = device
        _ = n_train
        _ = n_test
        _ = requested_device
        _ = resolved_device
        _ = device_fallback_reason
        _ = x
        _ = y
        _ = aux_meta_batch
        _ = dtype
        _ = resolved_split_indices
        return [
            DatasetBundle(
                X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
                y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.float32),
                X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
                y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.float32),
                feature_types=["num", "num"],
                metadata={
                    "backend": "torch",
                    "n_features": 2,
                    "dataset_seed_seen": int(dataset_root.child_seed()),
                    "family_sampled": str(noise_runtime_selection.family_sampled),
                    "lineage": {
                        "assignments": {
                            "feature_to_node": [0, 1],
                            "target_to_node": 1,
                            "target_parent_features": [0, 1],
                            "target_parent_count": 2,
                            "target_parent_fraction": 1.0,
                            "target_parent_prior": "all_features",
                            "target_parent_regime": "all_features",
                            "target_parent_sqrt_threshold": 1,
                        }
                    },
                },
            )
            for dataset_root in dataset_roots
        ]

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._resolve_noise_runtime_selection",
        _stub_resolve_noise_runtime_selection,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime.generate_fixed_layout_graph_batch",
        _stub_generate_fixed_layout_graph_batch,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_chunk_preserve_schema",
        _stub_finalize_generated_chunk_preserve_schema,
    )

    bundles = list(
        _generate_batch_with_plan_iter(
            cfg,
            plan=plan,
            num_datasets=4,
            seed=run_seed,
            batch_size=4,
        )
    )

    assert grouped_call_sizes == [3, 1]
    assert [int(bundle.metadata["dataset_seed_seen"]) for bundle in bundles] == dataset_seeds
    assert [str(bundle.metadata["family_sampled"]) for bundle in bundles] == family_pattern


def test_fixed_layout_plan_supports_classification_run_groups_mixed_noise_runtime_subbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.filter.enabled = False
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    run_seed = 654
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=17,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )
    run_root = KeyedRng(run_seed)
    dataset_roots = [run_root.keyed("dataset", idx) for idx in range(3)]
    family_pattern = ["gaussian", "laplace", "gaussian"]
    family_by_noise_root_seed = {
        dataset_root.keyed("noise_runtime").child_seed(): family
        for dataset_root, family in zip(dataset_roots, family_pattern, strict=True)
    }
    grouped_call_sizes: list[int] = []

    def _stub_resolve_noise_runtime_selection(
        _config,
        *,
        keyed_rng: KeyedRng,
        device: str = "cpu",
    ) -> NoiseRuntimeSelection:
        _ = device
        family = family_by_noise_root_seed[int(keyed_rng.child_seed())]
        return NoiseRuntimeSelection(
            family_requested="mixture",
            family_sampled=family,
            sampling_strategy="dataset_level",
            base_scale=1.0,
            student_t_df=5.0,
            mixture_weights={"gaussian": 0.5, "laplace": 0.5},
        )

    def _stub_generate_fixed_layout_label_batch(
        _config,
        _layout,
        *,
        execution_plan,
        dataset_seeds,
        device,
        noise_sigma_multiplier,
        noise_spec,
    ):
        _ = execution_plan
        _ = device
        _ = noise_sigma_multiplier
        _ = noise_spec
        grouped_call_sizes.append(len(dataset_seeds))
        batch_size = len(dataset_seeds)
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return torch.zeros((batch_size, n_rows), dtype=torch.int64), [{} for _ in dataset_seeds]

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._resolve_noise_runtime_selection",
        _stub_resolve_noise_runtime_selection,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime.generate_fixed_layout_label_batch",
        _stub_generate_fixed_layout_label_batch,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._raw_classification_labels_support_split",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._first_valid_classification_attempt_for_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("per-dataset replay fallback should not run when raw splits validate")
        ),
    )

    supported = _fixed_layout_plan_supports_classification_run(
        cfg,
        plan=plan,
        requested_device="cpu",
        resolved_device="cpu",
        run_root=run_root,
        num_datasets=3,
        batch_size=3,
    )

    assert supported is True
    assert grouped_call_sizes == [2, 1]


def test_fixed_layout_plan_classification_attempt_plan_does_not_scalarize_other_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.filter.enabled = False
    cfg.filter.max_attempts = 3
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=17,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )
    run_root = KeyedRng(654)
    dataset_roots = [run_root.keyed("dataset", idx) for idx in range(4)]
    retry_calls: list[int] = []

    def _stub_group_noise_runtime_chunk(
        _config: GeneratorConfig,
        *,
        dataset_roots: list[KeyedRng],
        attempts: list[int] | None = None,
    ):
        local_attempts = list(attempts or [0] * len(dataset_roots))
        assert all(int(attempt) == 0 for attempt in local_attempts)
        return [
            SimpleNamespace(
                chunk_offsets=list(range(len(dataset_roots))),
                generation_seeds=[
                    dataset_root.keyed("attempt", 0, "raw_generation").child_seed()
                    for dataset_root in dataset_roots
                ],
                selection=NoiseRuntimeSelection(
                    family_requested="gaussian",
                    family_sampled="gaussian",
                    sampling_strategy="dataset_level",
                    base_scale=1.0,
                    student_t_df=5.0,
                    mixture_weights=None,
                ),
                attempt=0,
            )
        ]

    def _stub_generate_fixed_layout_label_batch(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan,
        dataset_seeds: list[int],
        device: str,
        noise_sigma_multiplier: float,
        noise_spec,
    ) -> tuple[torch.Tensor, list[dict[str, object]]]:
        _ = execution_plan
        _ = device
        _ = noise_sigma_multiplier
        _ = noise_spec
        y_batch = torch.zeros(
            (len(dataset_seeds), cfg.dataset.n_train + cfg.dataset.n_test), dtype=torch.int64
        )
        return y_batch, [{} for _ in dataset_seeds]

    def _stub_raw_classification_labels_support_split(
        _y: torch.Tensor,
        *,
        dataset_root: KeyedRng,
        attempt: int,
        n_train: int,
    ) -> bool:
        _ = n_train
        dataset_seed = dataset_root.child_seed()
        if dataset_seed == dataset_roots[0].child_seed():
            return False
        return attempt == 0

    def _stub_first_valid_classification_attempt_for_dataset(
        _config: GeneratorConfig,
        *,
        plan: _FixedLayoutPlan,
        dataset_root: KeyedRng,
        requested_device: str,
        resolved_device: str,
        start_attempt: int = 0,
    ) -> int | None:
        _ = plan
        _ = requested_device
        _ = resolved_device
        retry_calls.append(dataset_root.child_seed())
        if dataset_root.child_seed() == dataset_roots[0].child_seed():
            assert start_attempt == 1
            return 1
        raise AssertionError("only the invalid attempt-0 dataset should use scalar replay lookup")

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._group_noise_runtime_chunk",
        _stub_group_noise_runtime_chunk,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime.generate_fixed_layout_label_batch",
        _stub_generate_fixed_layout_label_batch,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._raw_classification_labels_support_split",
        _stub_raw_classification_labels_support_split,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._first_valid_classification_attempt_for_dataset",
        _stub_first_valid_classification_attempt_for_dataset,
    )

    attempt_plan = _fixed_layout_plan_classification_attempt_plan(
        cfg,
        plan=plan,
        requested_device="cpu",
        resolved_device="cpu",
        run_root=run_root,
        num_datasets=4,
        batch_size=2,
    )

    assert attempt_plan == (1, 0, 0, 0)
    assert retry_calls == [dataset_roots[0].child_seed()]


def test_generate_one_returns_torch_tensors_on_cpu() -> None:
    cfg = _tiny_config()
    bundle = generate_one(cfg, seed=1234, device="cpu")
    assert isinstance(bundle.X_train, torch.Tensor)
    assert isinstance(bundle.y_train, torch.Tensor)
    assert isinstance(bundle.X_test, torch.Tensor)
    assert isinstance(bundle.y_test, torch.Tensor)
    assert bundle.metadata["backend"] == "torch"


def test_torch_path_sets_deferred_filter_not_run_metadata() -> None:
    cfg = _tiny_config()

    bundle = generate_one(cfg, seed=77, device="cpu")
    assert bundle.metadata["backend"] == "torch"
    assert bundle.metadata["filter"]["mode"] == "deferred"
    assert bundle.metadata["filter"]["status"] == "not_run"
    attempts = bundle.metadata["generation_attempts"]
    assert attempts["total_attempts"] >= 1
    assert attempts["retry_count"] == int(attempts["total_attempts"]) - 1
    assert attempts["filter_attempts"] == 0
    assert attempts["filter_rejections"] == 0
    assert attempts["filter_rejection_rate"] is None


def test_generate_rejects_inline_filter_enabled() -> None:
    cfg = _tiny_config()
    cfg.filter.enabled = True
    with pytest.raises(ValueError, match="Inline filtering has been removed from generate"):
        _ = generate_one(cfg, seed=1122, device="cpu")


def test_auto_surfaces_mps_runtime_failure_in_generate_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _stub_generate_fixed_layout_graph_batch(
        _config,
        _layout,
        *,
        execution_plan,
        dataset_seeds,
        device,
        noise_sigma_multiplier,
        noise_spec,
        runtime_metrics_out=None,
    ):
        _ = execution_plan
        _ = dataset_seeds
        _ = noise_sigma_multiplier
        _ = noise_spec
        _ = runtime_metrics_out
        calls.append(str(device))
        if device == "mps":
            raise RuntimeError("simulated mps failure")
        n_rows = 3
        n_features = int(_layout.n_features)
        return (
            torch.zeros((1, n_rows, n_features), dtype=torch.float32),
            torch.zeros((1, n_rows), dtype=torch.float32),
            [{}],
        )

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.prepare._resolve_device",
        lambda *_args, **_kwargs: "mps",
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime.generate_fixed_layout_graph_batch",
        _stub_generate_fixed_layout_graph_batch,
    )
    cfg = _tiny_regression_config()

    with pytest.raises(RuntimeError, match="simulated mps failure"):
        generate_one(cfg, seed=123, device="auto")
    assert calls == ["mps"]


def test_generate_batch_iter_auto_surfaces_mps_batch_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    calls: list[str] = []

    def _stub_generate_fixed_layout_graph_batch(
        _config,
        _layout,
        *,
        execution_plan,
        dataset_seeds,
        device,
        noise_sigma_multiplier,
        noise_spec,
        runtime_metrics_out=None,
    ):
        _ = execution_plan
        _ = dataset_seeds
        _ = noise_sigma_multiplier
        _ = noise_spec
        _ = runtime_metrics_out
        calls.append(str(device))
        if device == "mps":
            raise RuntimeError("simulated mps failure")
        n_rows = int(cfg.dataset.n_train + cfg.dataset.n_test)
        n_features = int(_layout.n_features)
        x = torch.zeros((1, n_rows, n_features), dtype=torch.float32)
        y = torch.zeros((1, n_rows), dtype=torch.float32)
        return x, y, [{}]

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.prepare._resolve_device",
        lambda *_args, **_kwargs: "mps",
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime.generate_fixed_layout_graph_batch",
        _stub_generate_fixed_layout_graph_batch,
    )

    with pytest.raises(RuntimeError, match="simulated mps failure"):
        next(generate_batch_iter(cfg, num_datasets=1, seed=230, device="auto"))
    assert calls == ["mps"]


def test_auto_does_not_fallback_to_numpy_if_torch_runtime_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_runtime(*_args, **_kwargs):
        raise RuntimeError("simulated torch runtime failure")

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime.generate_fixed_layout_graph_batch",
        _raise_runtime,
    )
    cfg = _tiny_config()

    with pytest.raises(RuntimeError, match="simulated torch runtime failure"):
        generate_one(cfg, seed=123, device="auto")


def test_explicit_cuda_request_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dagzoo.core.generation_context.torch.cuda.is_available",
        lambda: False,
    )
    cfg = _tiny_config()
    with pytest.raises(RuntimeError, match="Requested device 'cuda'"):
        generate_one(cfg, seed=123, device="cuda")


def test_invalid_device_raises() -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match="Unsupported device"):
        generate_one(cfg, seed=123, device="cud")


def test_generate_one_rejects_inverted_graph_node_bounds() -> None:
    cfg = _tiny_config()
    cfg.graph.n_nodes_min = 10
    cfg.graph.n_nodes_max = 5

    with pytest.raises(ValueError, match=r"graph\.n_nodes_min must be <= n_nodes_max"):
        generate_one(cfg, seed=123, device="cpu")


def test_generate_one_rejects_inverted_feature_bounds() -> None:
    cfg = _tiny_config()
    cfg.dataset.n_features_min = 10
    cfg.dataset.n_features_max = 5

    with pytest.raises(ValueError, match=r"dataset\.n_features_min must be <= n_features_max"):
        generate_one(cfg, seed=123, device="cpu")


def test_negative_num_datasets_raises() -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match="num_datasets must be >= 0"):
        list(generate_batch_iter(cfg, num_datasets=-1, seed=123, device="cpu"))


@pytest.mark.parametrize("bad_seed", [-1, 4294967296])
def test_generate_one_rejects_out_of_range_seed_override(bad_seed: int) -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        generate_one(cfg, seed=bad_seed, device="cpu")


@pytest.mark.parametrize("bad_seed", [-1, 4294967296])
def test_generate_batch_iter_rejects_out_of_range_seed_override(bad_seed: int) -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        list(generate_batch_iter(cfg, num_datasets=1, seed=bad_seed, device="cpu"))


@pytest.mark.parametrize("bad_seed", [-1, 4294967296])
def test_sample_fixed_layout_rejects_out_of_range_seed_override(bad_seed: int) -> None:
    cfg = _tiny_config()
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        _sample_fixed_layout(cfg, seed=bad_seed, device="cpu")


def test_sample_fixed_layout_accepts_32bit_seed_boundaries() -> None:
    cfg = _tiny_config()
    plan_min = _sample_fixed_layout(cfg, seed=0, device="cpu")
    plan_max = _sample_fixed_layout(cfg, seed=4294967295, device="cpu")
    assert plan_min.plan_seed == KeyedRng(0).child_seed("plan_candidate", 0, "layout")
    assert plan_max.plan_seed == KeyedRng(4294967295).child_seed("plan_candidate", 0, "layout")


def test_sample_fixed_layout_candidate_stores_layout_root_seed() -> None:
    cfg = _tiny_regression_config()
    run_root = KeyedRng(4321)
    candidate_root = run_root.keyed("plan_candidate", 0)

    plan = _sample_fixed_layout_candidate(
        cfg,
        keyed_rng=candidate_root,
        rows_seed=run_root.child_seed("rows"),
        requested_device="cpu",
        resolved_device="cpu",
    )

    assert plan.plan_seed == run_root.child_seed("plan_candidate", 0, "layout")


def test_prepare_canonical_fixed_layout_run_uses_candidate_root_for_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    plan = _sample_fixed_layout(_tiny_regression_config(), seed=701, device="cpu")
    sample_calls: list[tuple[tuple[str | int, ...], int, str, str]] = []

    def _stub_sample_fixed_layout_candidate(
        _config: GeneratorConfig,
        *,
        keyed_rng: KeyedRng,
        rows_seed: int,
        requested_device: str,
        resolved_device: str,
    ) -> _FixedLayoutPlan:
        sample_calls.append(
            (
                tuple(keyed_rng.path),
                int(rows_seed),
                str(requested_device),
                str(resolved_device),
            )
        )
        return plan

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._sample_fixed_layout_candidate",
        _stub_sample_fixed_layout_candidate,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._sample_fixed_layout",
        lambda *_args, **_kwargs: pytest.fail(
            "regression preparation should sample directly from the candidate root"
        ),
    )

    prepared = prepare_canonical_fixed_layout_run(cfg, num_datasets=2, seed=19, device="cpu")

    assert sample_calls == [(("plan_candidate", 0), KeyedRng(19).child_seed("rows"), "cpu", "cpu")]
    assert int(prepared.plan.plan_seed) == int(plan.plan_seed)


def test_zero_num_datasets_does_not_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _tiny_config()

    def _raise_if_called(*_args, **_kwargs):
        raise RuntimeError("device resolution should not run for empty batches")

    monkeypatch.setattr("dagzoo.core.generation_context._resolve_device", _raise_if_called)
    assert list(generate_batch_iter(cfg, num_datasets=0, seed=5, device="cuda")) == []


def test_resolve_split_indices_prefers_cuda_before_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _stub_resolve_split_indices_for_device(
        _y,
        *,
        task: str,
        n_train: int,
        keyed_rng: KeyedRng,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = task
        _ = n_train
        _ = keyed_rng
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("forced cuda split-device fallback")
        return torch.tensor([0]), torch.tensor([1])

    monkeypatch.setattr(
        "dagzoo.core.generation_runtime._resolve_split_indices_for_device",
        _stub_resolve_split_indices_for_device,
    )

    fake_y = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    train_idx, test_idx = _resolve_split_indices(
        fake_y,  # type: ignore[arg-type]
        task="classification",
        n_train=1,
        keyed_rng=KeyedRng(7),
    )

    assert calls == ["cuda", "cpu"]
    torch.testing.assert_close(train_idx, torch.tensor([0]))
    torch.testing.assert_close(test_idx, torch.tensor([1]))


def test_stratified_split_ensures_valid_class_split_with_many_classes() -> None:
    """High n_classes with low n_test should not fail with stratified splitting."""
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.dataset.n_classes_min = 10
    cfg.dataset.n_classes_max = 10
    cfg.dataset.n_train = 100
    cfg.dataset.n_test = 28
    cfg.filter.max_attempts = 3

    bundle = generate_one(cfg, seed=42, device="cpu")
    train_classes = set(torch.unique(bundle.y_train).tolist())
    test_classes = set(torch.unique(bundle.y_test).tolist())
    all_classes = torch.unique(torch.cat([bundle.y_train, bundle.y_test], dim=0), sorted=True)
    expected = torch.arange(all_classes.numel(), dtype=all_classes.dtype)
    assert len(train_classes) >= 2
    assert train_classes == test_classes
    assert torch.equal(all_classes, expected)

    class_structure = bundle.metadata["class_structure"]
    assert bundle.metadata["n_classes"] == int(class_structure["n_classes_realized"])
    assert int(class_structure["n_classes_sampled"]) == 10
    assert bool(class_structure["labels_contiguous"]) is True
    assert bool(class_structure["train_test_class_match"]) is True
    assert int(class_structure["min_label"]) == 0
    assert int(class_structure["max_label"]) == int(all_classes.numel() - 1)


def test_metadata_n_classes_uses_realized_class_count_for_classification() -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.dataset.n_classes_min = 32
    cfg.dataset.n_classes_max = 32
    cfg.dataset.n_train = 256
    cfg.dataset.n_test = 256
    cfg.filter.max_attempts = 3

    bundle = generate_one(cfg, seed=52, device="cpu")
    all_classes = torch.unique(torch.cat([bundle.y_train, bundle.y_test], dim=0), sorted=True)
    assert bundle.metadata["n_classes"] == int(all_classes.numel())
    assert int(bundle.metadata["class_structure"]["n_classes_sampled"]) == 32
    assert int(bundle.metadata["class_structure"]["n_classes_realized"]) == int(all_classes.numel())


def test_stratified_split_indices_returns_exact_requested_sizes() -> None:
    y = torch.tensor([0] * 8 + [1] * 5 + [2] * 3 + [3], dtype=torch.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    train_idx, test_idx = _stratified_split_indices(y, 10, generator, "cpu")

    assert int(train_idx.shape[0]) == 10
    assert int(test_idx.shape[0]) == 7

    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    assert train_set.isdisjoint(test_set)
    assert train_set | test_set == set(range(int(y.shape[0])))


def test_stratified_split_indices_raises_for_infeasible_constraints() -> None:
    y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    with pytest.raises(ValueError, match="infeasible_stratified_split"):
        _stratified_split_indices(y, 2, generator, "cpu")


def test_generate_retries_when_stratified_split_is_infeasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.filter.max_attempts = 2

    def _raise_infeasible_split(
        *_args: object, **_kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise InfeasibleStratifiedSplitError("infeasible_stratified_split: forced for test")

    monkeypatch.setattr(
        "dagzoo.core.generation_runtime._stratified_split_indices",
        _raise_infeasible_split,
    )

    with pytest.raises(ValueError, match=r"Last reason: invalid_class_split"):
        generate_one(cfg, seed=99, device="cpu")


def test_prepare_canonical_fixed_layout_run_precomputes_run_wide_classification_attempt_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.filter.max_attempts = 2

    plan = _sample_fixed_layout(_tiny_regression_config(), seed=701, device="cpu")
    sample_calls: list[tuple[tuple[str | int, ...], int]] = []
    attempt_plan_calls: list[tuple[int, int]] = []

    def _stub_sample_fixed_layout_candidate(
        _config: GeneratorConfig,
        *,
        keyed_rng: KeyedRng,
        rows_seed: int,
        requested_device: str,
        resolved_device: str,
    ) -> _FixedLayoutPlan:
        assert requested_device == "cpu"
        assert resolved_device == "cpu"
        sample_calls.append((tuple(keyed_rng.path), int(rows_seed)))
        return plan

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._sample_fixed_layout_candidate",
        _stub_sample_fixed_layout_candidate,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._fixed_layout_plan_classification_attempt_plan",
        lambda _config, **kwargs: (
            attempt_plan_calls.append((int(kwargs["num_datasets"]), int(kwargs["batch_size"])))
            or tuple(0 for _ in range(int(kwargs["num_datasets"])))
        ),
    )

    prepared = prepare_canonical_fixed_layout_run(cfg, num_datasets=10, seed=16, device="cpu")
    expected_rows_seed = KeyedRng(16).child_seed("rows")
    expected_batch_size = _resolve_fixed_layout_batch_size(
        plan,
        num_datasets=10,
        batch_size=None,
    )

    assert sample_calls == [(("plan_candidate", 0), expected_rows_seed)]
    assert attempt_plan_calls == [(10, expected_batch_size)]
    assert int(prepared.plan.plan_seed) == int(plan.plan_seed)
    assert int(prepared.batch_size) == expected_batch_size
    assert prepared.classification_attempt_plan == tuple(0 for _ in range(10))


def test_prepare_canonical_fixed_layout_run_leaves_classification_attempt_plan_unset_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.filter.max_attempts = 2
    plan = _sample_fixed_layout(_tiny_regression_config(), seed=801, device="cpu")

    def _stub_sample_fixed_layout_candidate(
        _config: GeneratorConfig,
        *,
        keyed_rng: KeyedRng,
        rows_seed: int,
        requested_device: str,
        resolved_device: str,
    ) -> _FixedLayoutPlan:
        assert tuple(keyed_rng.path) == ("plan_candidate", 0)
        assert int(rows_seed) == KeyedRng(17).child_seed("rows")
        assert requested_device == "cpu"
        assert resolved_device == "cpu"
        return plan

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._sample_fixed_layout_candidate",
        _stub_sample_fixed_layout_candidate,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._sample_fixed_layout",
        lambda *_args, **_kwargs: pytest.fail(
            "classification prep should skip single-dataset replay validation sampling"
        ),
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._fixed_layout_plan_classification_attempt_plan",
        lambda *_args, **_kwargs: pytest.fail(
            "canonical preparation should defer classification retry discovery to generation"
        ),
    )

    prepared = prepare_canonical_fixed_layout_run(
        cfg,
        num_datasets=4,
        seed=17,
        device="cpu",
        precompute_classification_attempt_plan=False,
    )

    assert int(prepared.plan.plan_seed) == int(plan.plan_seed)
    assert int(prepared.batch_size) == int(
        _resolve_fixed_layout_batch_size(plan, num_datasets=4, batch_size=None)
    )
    assert prepared.classification_attempt_plan is None


def test_resolve_fixed_layout_batch_size_uses_configured_target_cells() -> None:
    plan = _sample_fixed_layout(_tiny_regression_config(), seed=911, device="cpu")

    default_batch = _resolve_fixed_layout_batch_size(
        plan,
        num_datasets=10,
        batch_size=None,
    )
    larger_target_batch = _resolve_fixed_layout_batch_size(
        plan,
        num_datasets=10,
        batch_size=None,
        target_cells=8_000_000,
    )

    assert default_batch >= 1
    assert larger_target_batch >= default_batch


def test_h100_large_shape_reduces_auto_batch_size_relative_to_standard_h100() -> None:
    standard_cfg = GeneratorConfig.from_yaml("configs/benchmark_cuda_h100.yaml")
    large_cfg = GeneratorConfig.from_yaml("configs/benchmark_cuda_h100_large_shape.yaml")

    standard_plan = SimpleNamespace(
        n_train=standard_cfg.dataset.n_train,
        n_test=standard_cfg.dataset.n_test,
        layout=SimpleNamespace(n_features=standard_cfg.dataset.n_features_max),
    )
    large_plan = SimpleNamespace(
        n_train=large_cfg.dataset.n_train,
        n_test=large_cfg.dataset.n_test,
        layout=SimpleNamespace(n_features=large_cfg.dataset.n_features_max),
    )

    standard_batch = _resolve_fixed_layout_batch_size(
        standard_plan,
        num_datasets=32,
        batch_size=None,
        target_cells=standard_cfg.runtime.fixed_layout_target_cells,
    )
    large_batch = _resolve_fixed_layout_batch_size(
        large_plan,
        num_datasets=32,
        batch_size=None,
        target_cells=large_cfg.runtime.fixed_layout_target_cells,
    )

    assert standard_batch >= 1
    assert large_batch >= 1
    assert large_batch <= standard_batch


def test_fixed_layout_batch_size_cap_limits_auto_batch_size() -> None:
    cfg = _tiny_config()
    cfg.runtime.fixed_layout_target_cells = 64_000_000
    cfg.runtime.fixed_layout_batch_size_cap = 128

    plan = SimpleNamespace(
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout=SimpleNamespace(n_features=cfg.dataset.n_features_max),
    )

    capped_batch = _resolve_fixed_layout_batch_size(
        plan,
        num_datasets=512,
        batch_size=None,
        target_cells=cfg.runtime.fixed_layout_target_cells,
        batch_size_cap=cfg.runtime.fixed_layout_batch_size_cap,
    )

    assert capped_batch == 128


def test_generate_batch_with_plan_iter_allows_late_classification_failure_after_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.filter.max_attempts = 2
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=902,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )

    def _make_bundle(seed: int) -> DatasetBundle:
        return DatasetBundle(
            X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
            y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.int64),
            X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
            y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.int64),
            feature_types=["num", "num"],
            metadata={
                "seed": int(seed),
                "n_features": 2,
                "lineage": {
                    "assignments": {
                        "feature_to_node": [0, 1],
                        "target_to_node": 1,
                        "target_parent_features": [0, 1],
                        "target_parent_count": 2,
                        "target_parent_fraction": 1.0,
                        "target_parent_prior": "all_features",
                        "target_parent_regime": "all_features",
                        "target_parent_sqrt_threshold": 1,
                    }
                },
                "filter": {"mode": "deferred", "status": "not_run"},
                "generation_attempts": {
                    "total_attempts": 1,
                    "retry_count": 0,
                    "filter_attempts": 0,
                    "filter_rejections": 0,
                    "filter_rejection_rate": None,
                },
            },
        )

    def _stub_group_noise_runtime_chunk(
        _config: GeneratorConfig,
        *,
        dataset_roots: list[KeyedRng],
        attempts: list[int] | None = None,
    ):
        _ = attempts
        return [
            SimpleNamespace(
                chunk_offsets=[0, 1],
                generation_seeds=[
                    dataset_root.keyed("attempt", 0, "raw_generation").child_seed()
                    for dataset_root in dataset_roots
                ],
                selection=NoiseRuntimeSelection(
                    family_requested="gaussian",
                    family_sampled="gaussian",
                    sampling_strategy="dataset_level",
                    base_scale=1.0,
                    student_t_df=5.0,
                    mixture_weights=None,
                ),
                attempt=0,
            )
        ]

    def _stub_generate_grouped_raw_batches(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan: FixedLayoutExecutionPlan,
        grouped_noise_runtime,
        requested_device: str,
        resolved_device: str,
        noise_sigma_multiplier: float,
    ) -> list[SimpleNamespace]:
        _ = execution_plan
        _ = requested_device
        _ = resolved_device
        _ = noise_sigma_multiplier
        group = grouped_noise_runtime[0]
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return [
            SimpleNamespace(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=group.attempt,
                x_batch=torch.zeros((2, n_rows, 2), dtype=torch.float32),
                y_batch=torch.zeros((2, n_rows), dtype=torch.int64),
                aux_meta_batch=[
                    {"filter": {"mode": "deferred", "status": "not_run"}},
                    {"filter": {"mode": "deferred", "status": "not_run"}},
                ],
                effective_resolved_device="cpu",
                device_fallback_reason=None,
            )
        ]

    def _stub_finalize_generated_chunk_preserve_schema(
        _config: GeneratorConfig,
        _layout,
        *,
        context,
        dataset_roots: list[KeyedRng],
        attempt: int,
        attempts_used: int,
        device: str,
        n_train: int,
        n_test: int,
        requested_device: str,
        resolved_device: str,
        device_fallback_reason: str | None,
        x: torch.Tensor,
        y: torch.Tensor,
        aux_meta_batch: list[dict[str, object]],
        noise_runtime_selection: NoiseRuntimeSelection,
        dtype: torch.dtype,
        resolved_split_indices=None,
    ) -> list[DatasetBundle | None]:
        _ = context
        _ = attempt
        _ = attempts_used
        _ = device
        _ = n_train
        _ = n_test
        _ = requested_device
        _ = resolved_device
        _ = device_fallback_reason
        _ = x
        _ = y
        _ = aux_meta_batch
        _ = noise_runtime_selection
        _ = dtype
        _ = resolved_split_indices
        return [_make_bundle(dataset_roots[0].child_seed()), None]

    def _stub_generate_fixed_layout_bundle_with_retries(
        _config: GeneratorConfig,
        *,
        plan: _FixedLayoutPlan,
        dataset_root: KeyedRng,
        requested_device: str,
        resolved_device: str,
        preserve_feature_schema: bool,
        start_attempt: int = 0,
        finalization_context=None,
        on_raw_batch_metrics=None,
    ) -> DatasetBundle:
        _ = plan
        _ = requested_device
        _ = resolved_device
        _ = preserve_feature_schema
        _ = on_raw_batch_metrics
        assert finalization_context is not None
        assert int(start_attempt) == 1
        raise ValueError("Failed to generate a valid fixed-layout dataset after 2 attempts.")

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._group_noise_runtime_chunk",
        _stub_group_noise_runtime_chunk,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_grouped_raw_batches",
        _stub_generate_grouped_raw_batches,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_chunk_preserve_schema",
        _stub_finalize_generated_chunk_preserve_schema,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_fixed_layout_bundle_with_retries",
        _stub_generate_fixed_layout_bundle_with_retries,
    )

    bundles = _generate_batch_with_plan_iter(
        cfg,
        plan=plan,
        num_datasets=2,
        seed=44,
        batch_size=2,
    )

    first_bundle = next(bundles)
    assert int(first_bundle.metadata["seed"]) == KeyedRng(44).child_seed("dataset", 0)
    with pytest.raises(ValueError, match="Failed to generate a valid fixed-layout dataset"):
        next(bundles)


def test_generate_batch_with_plan_iter_uses_cached_classification_attempt_plan_for_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    cfg.dataset.task = "classification"
    cfg.filter.max_attempts = 3
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=901,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )
    attempt_plan = (2, 0, 1)
    grouped_attempts_seen: list[list[int]] = []
    retry_starts: list[int] = []

    def _make_bundle(seed: int) -> DatasetBundle:
        return DatasetBundle(
            X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
            y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.int64),
            X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
            y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.int64),
            feature_types=["num", "num"],
            metadata={
                "seed": int(seed),
                "n_features": 2,
                "lineage": {
                    "assignments": {
                        "feature_to_node": [0, 1],
                        "target_to_node": 1,
                        "target_parent_features": [0, 1],
                        "target_parent_count": 2,
                        "target_parent_fraction": 1.0,
                        "target_parent_prior": "all_features",
                        "target_parent_regime": "all_features",
                        "target_parent_sqrt_threshold": 1,
                    }
                },
                "filter": {"mode": "deferred", "status": "not_run"},
                "generation_attempts": {
                    "total_attempts": 1,
                    "retry_count": 0,
                    "filter_attempts": 0,
                    "filter_rejections": 0,
                    "filter_rejection_rate": None,
                },
            },
        )

    def _stub_group_noise_runtime_chunk(
        _config: GeneratorConfig,
        *,
        dataset_roots: list[KeyedRng],
        attempts: list[int] | None = None,
    ):
        local_attempts = list(attempts or [0] * len(dataset_roots))
        grouped_attempts_seen.append(local_attempts)
        return [
            SimpleNamespace(
                chunk_offsets=[index],
                generation_seeds=[
                    dataset_root.keyed(
                        "attempt", local_attempts[index], "raw_generation"
                    ).child_seed()
                ],
                selection=NoiseRuntimeSelection(
                    family_requested="gaussian",
                    family_sampled="gaussian",
                    sampling_strategy="dataset_level",
                    base_scale=1.0,
                    student_t_df=5.0,
                    mixture_weights=None,
                ),
                attempt=local_attempts[index],
            )
            for index, dataset_root in enumerate(dataset_roots)
        ]

    def _stub_generate_grouped_raw_batches(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan: FixedLayoutExecutionPlan,
        grouped_noise_runtime,
        requested_device: str,
        resolved_device: str,
        noise_sigma_multiplier: float,
    ) -> list[SimpleNamespace]:
        _ = execution_plan
        _ = requested_device
        _ = resolved_device
        _ = noise_sigma_multiplier
        assert all(group.attempt == 0 for group in grouped_noise_runtime)
        n_rows = cfg.dataset.n_train + cfg.dataset.n_test
        return [
            SimpleNamespace(
                chunk_offsets=list(group.chunk_offsets),
                selection=group.selection,
                attempt=group.attempt,
                x_batch=torch.zeros((1, n_rows, 2), dtype=torch.float32),
                y_batch=torch.zeros((1, n_rows), dtype=torch.int64),
                aux_meta_batch=[{"filter": {"mode": "deferred", "status": "not_run"}}],
                effective_resolved_device="cpu",
                device_fallback_reason=None,
            )
            for group in grouped_noise_runtime
        ]

    def _stub_finalize_generated_chunk_preserve_schema(
        _config: GeneratorConfig,
        _layout,
        *,
        context,
        dataset_roots: list[KeyedRng],
        attempt: int,
        attempts_used: int,
        device: str,
        n_train: int,
        n_test: int,
        requested_device: str,
        resolved_device: str,
        device_fallback_reason: str | None,
        x: torch.Tensor,
        y: torch.Tensor,
        aux_meta_batch: list[dict[str, object]],
        noise_runtime_selection: NoiseRuntimeSelection,
        dtype: torch.dtype,
        resolved_split_indices=None,
    ) -> list[DatasetBundle | None]:
        _ = context
        _ = device
        _ = n_train
        _ = n_test
        _ = requested_device
        _ = resolved_device
        _ = device_fallback_reason
        _ = x
        _ = y
        _ = aux_meta_batch
        _ = noise_runtime_selection
        _ = dtype
        _ = resolved_split_indices
        assert len(dataset_roots) == 1
        dataset_seed = dataset_roots[0].child_seed()
        assert attempts_used == attempt + 1
        return [_make_bundle(dataset_seed)]

    def _stub_generate_fixed_layout_bundle_with_retries(
        _config: GeneratorConfig,
        *,
        plan: _FixedLayoutPlan,
        dataset_root: KeyedRng,
        requested_device: str,
        resolved_device: str,
        preserve_feature_schema: bool,
        start_attempt: int = 0,
        finalization_context=None,
        on_raw_batch_metrics=None,
    ) -> DatasetBundle:
        _ = plan
        _ = requested_device
        _ = resolved_device
        _ = preserve_feature_schema
        _ = on_raw_batch_metrics
        assert finalization_context is not None
        retry_starts.append(int(start_attempt))
        return _make_bundle(dataset_root.child_seed())

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._group_noise_runtime_chunk",
        _stub_group_noise_runtime_chunk,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_grouped_raw_batches",
        _stub_generate_grouped_raw_batches,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_chunk_preserve_schema",
        _stub_finalize_generated_chunk_preserve_schema,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._generate_fixed_layout_bundle_with_retries",
        _stub_generate_fixed_layout_bundle_with_retries,
    )

    bundles = list(
        _generate_batch_with_plan_iter(
            cfg,
            plan=plan,
            num_datasets=3,
            seed=33,
            batch_size=3,
            classification_attempt_plan=attempt_plan,
        )
    )

    assert len(bundles) == 3
    assert grouped_attempts_seen == [[0]]
    assert retry_starts == [2, 1]
    assert [int(bundle.metadata["seed"]) for bundle in bundles] == [
        KeyedRng(33).child_seed("dataset", 0),
        KeyedRng(33).child_seed("dataset", 1),
        KeyedRng(33).child_seed("dataset", 2),
    ]


def test_generate_fixed_layout_bundle_with_retries_reuses_cached_finalization_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_regression_config()
    cfg.dataset.n_train = 4
    cfg.dataset.n_test = 2
    plan = _FixedLayoutPlan(
        layout=_layout_stub(
            feature_types=["num", "num"],
            graph_nodes=2,
            adjacency=torch.zeros((2, 2), dtype=torch.bool),
            feature_node_assignment=[0, 1],
            target_node_assignment=1,
        ),
        requested_device="cpu",
        resolved_device="cpu",
        plan_seed=901,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        layout_signature="layout_sig",
        execution_plan=FixedLayoutExecutionPlan(),
        plan_signature="plan_sig",
    )
    finalization_context = _build_fixed_schema_finalization_context(
        cfg,
        plan.layout,
        n_train=cfg.dataset.n_train,
        n_test=cfg.dataset.n_test,
        shift_params=resolve_shift_runtime_params(cfg),
    )
    seen_contexts: list[object | None] = []

    def _stub_generate_fixed_layout_graph_batch(
        _config: GeneratorConfig,
        _layout,
        *,
        execution_plan: FixedLayoutExecutionPlan,
        dataset_seeds: list[int],
        device: str,
        noise_sigma_multiplier: float,
        noise_spec,
        runtime_metrics_out=None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
        _ = execution_plan
        _ = dataset_seeds
        _ = device
        _ = noise_sigma_multiplier
        _ = noise_spec
        _ = runtime_metrics_out
        return (
            torch.zeros((1, 6, 2), dtype=torch.float32),
            torch.zeros((1, 6), dtype=torch.float32),
            [{"filter": {"mode": "deferred", "status": "not_run"}}],
        )

    def _stub_finalize_generated_tensors(
        _config: GeneratorConfig,
        _layout,
        *,
        dataset_seed: int,
        attempt: int,
        attempts_used: int,
        dataset_root: KeyedRng,
        device: str,
        n_train: int,
        n_test: int,
        requested_device: str,
        resolved_device: str,
        device_fallback_reason: str | None,
        x: torch.Tensor,
        y: torch.Tensor,
        aux_meta: dict[str, object],
        shift_params,
        noise_runtime_selection,
        dtype: torch.dtype,
        preserve_feature_schema: bool = False,
        finalization_context=None,
    ) -> DatasetBundle:
        _ = attempt
        _ = attempts_used
        _ = dataset_root
        _ = device
        _ = n_train
        _ = n_test
        _ = requested_device
        _ = resolved_device
        _ = device_fallback_reason
        _ = x
        _ = y
        _ = aux_meta
        _ = shift_params
        _ = noise_runtime_selection
        _ = dtype
        _ = preserve_feature_schema
        seen_contexts.append(finalization_context)
        return DatasetBundle(
            X_train=torch.zeros((cfg.dataset.n_train, 2), dtype=torch.float32),
            y_train=torch.zeros(cfg.dataset.n_train, dtype=torch.float32),
            X_test=torch.zeros((cfg.dataset.n_test, 2), dtype=torch.float32),
            y_test=torch.zeros(cfg.dataset.n_test, dtype=torch.float32),
            feature_types=["num", "num"],
            metadata={
                "seed": int(dataset_seed),
                "n_features": 2,
                "lineage": {
                    "assignments": {
                        "feature_to_node": [0, 1],
                        "target_to_node": 1,
                        "target_parent_features": [0, 1],
                        "target_parent_count": 2,
                        "target_parent_fraction": 1.0,
                        "target_parent_prior": "all_features",
                        "target_parent_regime": "all_features",
                        "target_parent_sqrt_threshold": 1,
                    }
                },
                "filter": {"mode": "deferred", "status": "not_run"},
                "generation_attempts": {
                    "total_attempts": 1,
                    "retry_count": 0,
                    "filter_attempts": 0,
                    "filter_rejections": 0,
                    "filter_rejection_rate": None,
                },
            },
        )

    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime.generate_fixed_layout_graph_batch",
        _stub_generate_fixed_layout_graph_batch,
    )
    monkeypatch.setattr(
        "dagzoo.core.fixed_layout.runtime._finalize_generated_tensors",
        _stub_finalize_generated_tensors,
    )

    _generate_fixed_layout_bundle_with_retries(
        cfg,
        plan=plan,
        dataset_root=KeyedRng(123),
        requested_device="cpu",
        resolved_device="cpu",
        preserve_feature_schema=True,
        finalization_context=finalization_context,
    )

    assert seen_contexts == [finalization_context]


def test_generate_one_replays_from_emitted_metadata_seed() -> None:
    cfg = _tiny_regression_config()

    bundle = generate_one(cfg, seed=4321, device="cpu")
    replayed = generate_one(cfg, seed=int(bundle.metadata["seed"]), device="cpu")

    assert int(bundle.metadata["seed"]) == 4321
    assert int(bundle.metadata["layout_plan_seed"]) == KeyedRng(4321).child_seed(
        "plan_candidate",
        0,
        "layout",
    )
    assert int(replayed.metadata["seed"]) == 4321
    np.testing.assert_allclose(np.asarray(bundle.X_train), np.asarray(replayed.X_train), atol=1e-6)
    np.testing.assert_allclose(np.asarray(bundle.X_test), np.asarray(replayed.X_test), atol=1e-6)
    np.testing.assert_allclose(np.asarray(bundle.y_train), np.asarray(replayed.y_train), atol=1e-6)
    np.testing.assert_allclose(np.asarray(bundle.y_test), np.asarray(replayed.y_test), atol=1e-6)
    assert bundle.metadata["layout_plan_signature"] == replayed.metadata["layout_plan_signature"]


def test_generate_one_keyed_replay_layout_root_path_replays_layout_signature() -> None:
    cfg = _tiny_regression_config()

    bundle = generate_one(cfg, seed=4321, device="cpu")
    keyed_replay = bundle.metadata["keyed_replay"]
    replayed_layout = _sample_layout(
        cfg,
        KeyedRng(int(bundle.metadata["seed"])).keyed(*keyed_replay["layout_root_path"]),
        "cpu",
    )

    assert keyed_replay["layout_root_path"] == ["plan_candidate", 0, "layout"]
    assert _layout_signature(replayed_layout) == str(bundle.metadata["layout_signature"])


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda bundle: bundle.metadata.__setitem__("seed", True),
            r"metadata\.seed must be an integer",
        ),
        (
            lambda bundle: bundle.metadata.__setitem__("keyed_replay", []),
            r"metadata\.keyed_replay must be a mapping",
        ),
        (
            lambda bundle: bundle.metadata["keyed_replay"].__setitem__("layout_root_path", []),
            r"metadata\.keyed_replay\.layout_root_path must be a non-empty list",
        ),
        (
            lambda bundle: bundle.metadata["keyed_replay"].__setitem__(
                "layout_root_path",
                ["plan_candidate", {}, "layout"],
            ),
            r"metadata\.keyed_replay\.layout_root_path\[1\] must be an int or string path component",
        ),
        (
            lambda bundle: bundle.metadata.__setitem__("config", []),
            r"metadata\.config must be a mapping",
        ),
        (
            lambda bundle: bundle.metadata.__setitem__("requested_device", ""),
            r"metadata\.requested_device must be a non-empty string",
        ),
        (
            lambda bundle: bundle.metadata.__setitem__("resolved_device", ""),
            r"metadata\.resolved_device must be a non-empty string",
        ),
        (
            lambda bundle: bundle.metadata.__setitem__(
                "layout_signature", "wrong-layout-signature"
            ),
            r"Replayed fixed-layout plan does not match metadata\.layout_signature",
        ),
        (
            lambda bundle: bundle.metadata.__setitem__(
                "layout_plan_signature", "wrong-plan-signature"
            ),
            r"Replayed fixed-layout plan does not match metadata\.layout_plan_signature",
        ),
    ],
)
def test_replay_emitted_fixed_layout_plan_rejects_invalid_metadata(
    mutator,
    match: str,
) -> None:
    cfg = _tiny_regression_config()
    bundle = deepcopy(generate_one(cfg, seed=4321, device="cpu"))

    mutator(bundle)

    with pytest.raises(ValueError, match=match):
        _replay_emitted_fixed_layout_plan(cfg, bundle)


def test_replay_emitted_fixed_layout_plan_recomputes_missing_layout_plan_seed() -> None:
    cfg = _tiny_regression_config()
    bundle = deepcopy(generate_one(cfg, seed=4321, device="cpu"))
    keyed_replay = deepcopy(bundle.metadata["keyed_replay"])

    bundle.metadata.pop("layout_plan_seed")
    replayed_plan = _replay_emitted_fixed_layout_plan(cfg, bundle)

    assert (
        int(replayed_plan.plan_seed)
        == KeyedRng(int(bundle.metadata["seed"]))
        .keyed(*keyed_replay["layout_root_path"])
        .child_seed()
    )


def test_generate_batch_graph_steering_preserves_base_replay_roots_and_replays_plan() -> None:
    cfg = _tiny_regression_config()
    cfg.steering.enabled = True
    cfg.steering.preset = "anti_memorization_piecewise_v1"
    cfg.validate_generation_constraints()

    batch = generate_batch(cfg, num_datasets=5, seed=4321, device="cpu")
    base_keyed_replay = batch[0].metadata["keyed_replay"]
    steered_bundle = batch[2]
    steered_keyed_replay = steered_bundle.metadata["keyed_replay"]
    replayed_plan = _replay_emitted_fixed_layout_plan(cfg, steered_bundle)

    assert "steering_layout_root_path" not in base_keyed_replay
    assert "steering_execution_plan_root_path" not in base_keyed_replay
    assert "steering_layout_root_path" not in batch[4].metadata["keyed_replay"]
    assert "steering_execution_plan_root_path" not in batch[4].metadata["keyed_replay"]
    assert steered_keyed_replay["layout_root_path"] == ["plan_candidate", 0, "layout"]
    assert steered_keyed_replay["execution_plan_root_path"] == [
        "plan_candidate",
        0,
        "execution_plan",
    ]
    assert steered_keyed_replay["steering_layout_root_path"] == [
        "dataset",
        2,
        "steering",
        "layout",
    ]
    assert steered_keyed_replay["steering_execution_plan_root_path"] == [
        "dataset",
        2,
        "steering",
        "execution_plan",
    ]
    assert int(steered_bundle.metadata["layout_plan_schema_version"]) == 10
    assert str(steered_bundle.metadata["layout_execution_contract"]) == "chunk_batched_v3"
    assert str(replayed_plan.layout_signature) == str(steered_bundle.metadata["layout_signature"])
    assert str(replayed_plan.plan_signature) == str(
        steered_bundle.metadata["layout_plan_signature"]
    )


def test_generate_one_keyed_replay_dataset_root_path_replays_noise_runtime_metadata() -> None:
    cfg = _tiny_regression_config()
    cfg.noise.family = NOISE_FAMILY_MIXTURE
    cfg.noise.mixture_weights = {
        str(NOISE_FAMILY_GAUSSIAN): 0.2,
        str(NOISE_FAMILY_LAPLACE): 0.8,
    }
    cfg.noise.base_scale = 0.35
    cfg.noise.student_t_df = 7.0

    bundle = generate_one(cfg, seed=4321, device="cpu")
    keyed_replay = bundle.metadata["keyed_replay"]
    dataset_root = KeyedRng(int(bundle.metadata["seed"])).keyed(*keyed_replay["dataset_root_path"])
    replayed_selection = _resolve_noise_runtime_selection(
        cfg,
        keyed_rng=dataset_root.keyed("noise_runtime"),
    )

    assert keyed_replay["dataset_root_path"] == ["dataset", 0]
    assert int(bundle.metadata["dataset_seed"]) == int(dataset_root.child_seed())
    assert bundle.metadata["noise_distribution"] == _build_noise_distribution_metadata(
        replayed_selection
    )


def _tiny_missingness_config(
    *,
    task: str,
    mechanism: str,
    missing_rate: float = 0.25,
) -> GeneratorConfig:
    cfg = _tiny_config()
    cfg.dataset.task = task
    cfg.dataset.n_train = 320
    cfg.dataset.n_test = 160
    cfg.dataset.missing_rate = missing_rate
    cfg.dataset.missing_mechanism = mechanism  # type: ignore[assignment]
    return cfg


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_generate_one_applies_missingness_and_emits_summary(task: str, mechanism: str) -> None:
    cfg = _tiny_missingness_config(task=task, mechanism=mechanism, missing_rate=0.25)
    bundle = generate_one(cfg, seed=2718, device="cpu")

    assert bundle.X_train.shape[0] == cfg.dataset.n_train
    assert bundle.X_test.shape[0] == cfg.dataset.n_test
    assert bundle.X_train.shape[1] == bundle.X_test.shape[1]
    assert len(bundle.feature_types) == bundle.X_train.shape[1]

    assert torch.isnan(bundle.X_train).any()
    assert torch.isnan(bundle.X_test).any()

    payload = bundle.metadata["missingness"]
    assert payload["enabled"] is True
    assert payload["mechanism"] == mechanism
    assert payload["target_rate"] == pytest.approx(0.25)
    assert payload["missing_count_overall"] == (
        payload["missing_count_train"] + payload["missing_count_test"]
    )
    assert 0.0 <= float(payload["realized_rate_train"]) <= 1.0
    assert 0.0 <= float(payload["realized_rate_test"]) <= 1.0
    assert 0.0 <= float(payload["realized_rate_overall"]) <= 1.0
    assert abs(float(payload["realized_rate_overall"]) - 0.25) <= 0.05

    if task == "classification":
        assert bundle.y_train.dtype == torch.int64
        assert bundle.y_test.dtype == torch.int64
    else:
        assert torch.isfinite(bundle.y_train).all()
        assert torch.isfinite(bundle.y_test).all()


def test_generate_one_missingness_disabled_preserves_default_behavior() -> None:
    cfg = _tiny_missingness_config(task="classification", mechanism="none", missing_rate=0.0)
    bundle = generate_one(cfg, seed=31415, device="cpu")
    assert "missingness" not in bundle.metadata
    assert not torch.isnan(bundle.X_train).any()
    assert not torch.isnan(bundle.X_test).any()


def test_generate_one_missingness_mask_is_reproducible_for_fixed_seed() -> None:
    cfg = _tiny_missingness_config(task="classification", mechanism="mar", missing_rate=0.3)
    a = generate_one(cfg, seed=12345, device="cpu")
    b = generate_one(cfg, seed=12345, device="cpu")

    assert torch.equal(torch.isnan(a.X_train), torch.isnan(b.X_train))
    assert torch.equal(torch.isnan(a.X_test), torch.isnan(b.X_test))
    assert a.metadata["missingness"] == b.metadata["missingness"]


def test_generate_one_missingness_does_not_change_targets_for_same_seed() -> None:
    base_cfg = _tiny_missingness_config(task="regression", mechanism="none", missing_rate=0.0)
    masked_cfg = _tiny_missingness_config(task="regression", mechanism="mnar", missing_rate=0.3)

    base = generate_one(base_cfg, seed=24680, device="cpu")
    masked = generate_one(masked_cfg, seed=24680, device="cpu")

    torch.testing.assert_close(base.y_train, masked.y_train)
    torch.testing.assert_close(base.y_test, masked.y_test)
    assert torch.isnan(masked.X_train).any() or torch.isnan(masked.X_test).any()
