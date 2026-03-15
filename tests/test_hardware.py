import math

import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.hardware import HardwareInfo, detect_hardware, get_peak_flops
from dagzoo.hardware_policy import apply_hardware_policy, list_hardware_policies


def test_peak_flops_lookup() -> None:
    assert get_peak_flops("NVIDIA H100 PCIe") == 756e12
    assert get_peak_flops("NVIDIA H100 SXM") == 989e12
    assert get_peak_flops("NVIDIA RTX 4090") == 165.2e12


def test_peak_flops_fallback() -> None:
    val = get_peak_flops("Some Unknown GPU XYZ")
    assert math.isinf(val)


def test_hardware_policy_none_is_immutable_and_noop() -> None:
    cfg = GeneratorConfig()
    original = cfg.to_dict()

    hw = HardwareInfo(
        backend="cpu",
        requested_device="auto",
        device_name="cpu",
        total_memory_gb=None,
        peak_flops=float("inf"),
        tier="cpu",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="none")

    assert cfg.to_dict() == original
    assert effective.to_dict() == original
    assert effective is not cfg


def test_hardware_policy_cuda_tiered_applied_when_h100_profile() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"
    original_train = cfg.dataset.n_train

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA H100 SXM",
        total_memory_gb=80.0,
        peak_flops=989e12,
        tier="cuda_h100",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="cuda_tiered_v1")

    assert cfg.dataset.n_train == original_train
    assert effective.dataset.n_train >= 4096
    assert effective.benchmark.preset_name == "cuda_h100_auto"
    assert effective.runtime.fixed_layout_target_cells == 240_000_000


@pytest.mark.parametrize(
    (
        "tier",
        "expected_preset_name",
        "expected_n_train",
        "expected_n_test",
        "expected_n_features_max",
        "expected_n_nodes_max",
        "expected_shard_size",
        "expected_num_datasets",
        "expected_warmup_datasets",
        "expected_target_cells",
    ),
    [
        (
            "cuda_desktop",
            "cuda_desktop_auto",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            48_000_000,
        ),
        (
            "cuda_datacenter",
            "cuda_datacenter_auto",
            1536,
            512,
            96,
            48,
            256,
            2500,
            30,
            160_000_000,
        ),
        (
            "cuda_h100",
            "cuda_h100_auto",
            4096,
            1024,
            192,
            64,
            512,
            5000,
            50,
            240_000_000,
        ),
    ],
)
def test_hardware_policy_cuda_tiered_matrix_by_tier(
    hardware_info_factory,
    tier: str,
    expected_preset_name: str,
    expected_n_train: int | None,
    expected_n_test: int | None,
    expected_n_features_max: int | None,
    expected_n_nodes_max: int | None,
    expected_shard_size: int | None,
    expected_num_datasets: int | None,
    expected_warmup_datasets: int | None,
    expected_target_cells: int,
) -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"
    baseline = GeneratorConfig()

    effective = apply_hardware_policy(
        cfg,
        hardware_info_factory(tier),
        policy_name="cuda_tiered_v1",
    )

    assert effective.benchmark.preset_name == expected_preset_name
    assert effective.dataset.n_train == (
        baseline.dataset.n_train if expected_n_train is None else expected_n_train
    )
    assert effective.dataset.n_test == (
        baseline.dataset.n_test if expected_n_test is None else expected_n_test
    )
    assert effective.dataset.n_features_max == (
        baseline.dataset.n_features_max
        if expected_n_features_max is None
        else expected_n_features_max
    )
    assert effective.graph.n_nodes_max == (
        baseline.graph.n_nodes_max if expected_n_nodes_max is None else expected_n_nodes_max
    )
    assert effective.output.shard_size == (
        baseline.output.shard_size if expected_shard_size is None else expected_shard_size
    )
    assert effective.benchmark.num_datasets == (
        baseline.benchmark.num_datasets if expected_num_datasets is None else expected_num_datasets
    )
    assert effective.benchmark.warmup_datasets == (
        baseline.benchmark.warmup_datasets
        if expected_warmup_datasets is None
        else expected_warmup_datasets
    )
    assert effective.runtime.fixed_layout_target_cells == expected_target_cells


def test_hardware_policy_raises_low_fixed_layout_target_cells_to_memory_scaled_floor() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"
    cfg.runtime.fixed_layout_target_cells = 16_000_000

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA RTX 4090",
        total_memory_gb=24.0,
        peak_flops=165.2e12,
        tier="cuda_desktop",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="cuda_tiered_v1")

    assert effective.runtime.fixed_layout_target_cells == 48_000_000


def test_hardware_policy_preserves_explicit_high_fixed_layout_target_cells_override() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"
    cfg.runtime.fixed_layout_target_cells = 96_000_000

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA RTX 4090",
        total_memory_gb=24.0,
        peak_flops=165.2e12,
        tier="cuda_desktop",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="cuda_tiered_v1")

    assert effective.runtime.fixed_layout_target_cells == 96_000_000


def test_hardware_policy_cuda_datacenter_scales_fixed_layout_target_cells_with_memory() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="NVIDIA A100 80GB",
        total_memory_gb=80.0,
        peak_flops=312e12,
        tier="cuda_datacenter",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="cuda_tiered_v1")

    assert effective.runtime.fixed_layout_target_cells == 160_000_000


def test_hardware_policy_cuda_unknown_fallback_applies_fixed_layout_target_floor() -> None:
    cfg = GeneratorConfig()
    cfg.runtime.device = "cuda"

    hw = HardwareInfo(
        backend="cuda",
        requested_device="cuda",
        device_name="Unknown GPU XYZ",
        total_memory_gb=None,
        peak_flops=float("inf"),
        tier="cuda_unknown_fallback",
    )
    effective = apply_hardware_policy(cfg, hw, policy_name="cuda_tiered_v1")

    assert effective.benchmark.preset_name == "cuda_unknown_fallback"
    assert effective.runtime.fixed_layout_target_cells == 48_000_000


def test_hardware_policy_unknown_name_raises() -> None:
    cfg = GeneratorConfig()
    hw = HardwareInfo(
        backend="cpu",
        requested_device="cpu",
        device_name="cpu",
        total_memory_gb=None,
        peak_flops=float("inf"),
        tier="cpu",
    )
    with pytest.raises(ValueError, match="Unknown hardware policy"):
        _ = apply_hardware_policy(cfg, hw, policy_name="missing")


def test_hardware_policy_registry_includes_builtins() -> None:
    names = list_hardware_policies()
    assert "none" in names
    assert "cuda_tiered_v1" in names


def test_detect_hardware_cpu_backend_label() -> None:
    hw = detect_hardware("cpu")
    assert hw.backend == "cpu"
