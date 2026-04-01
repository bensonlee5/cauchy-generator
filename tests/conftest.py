from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import torch
import yaml

from dagzoo.config import GeneratorConfig, clone_generator_config
from dagzoo.hardware import HardwareInfo
from dagzoo.rng import KeyedRng, keyed_rng_from_generator

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_generator(seed: int = 42) -> torch.Generator:
    """Create a seeded torch Generator on CPU for deterministic tests."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def make_keyed_rng(generator: torch.Generator, *components: str | int) -> KeyedRng:
    """Consume one ambient draw and derive the same keyed root used by helpers."""

    return keyed_rng_from_generator(generator, *components)


@lru_cache(maxsize=None)
def _cached_repo_config(resource_name: str) -> GeneratorConfig:
    return GeneratorConfig.from_yaml(REPO_ROOT / "configs" / resource_name)


def load_repo_config(resource_name: str = "default.yaml") -> GeneratorConfig:
    """Load one repo config and return a fresh mutable copy for the caller.

    Most public-surface unit tests exercise the supported stratified path unless
    they opt into heterogeneous mode or an internal fixed-layout helper
    directly.
    """

    config = clone_generator_config(_cached_repo_config(resource_name), revalidate=False)
    config.runtime.layout_mode = "stratified"
    return config


def write_yaml(tmp_path: Path, name: str, payload: object) -> Path:
    """Write one YAML payload under ``tmp_path`` and return the created path."""

    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def write_config(tmp_path: Path, config: GeneratorConfig, name: str = "config.yaml") -> Path:
    """Serialize one ``GeneratorConfig`` under ``tmp_path`` and return the path."""

    return write_yaml(tmp_path, name, config.to_dict())


def load_script_module(module_name: str, rel_path: str):
    """Load one repo script module from a repo-relative path."""

    script_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_mock_hardware(tier: str) -> HardwareInfo:
    normalized_tier = str(tier).strip().lower()
    if normalized_tier == "cpu":
        return HardwareInfo(
            backend="cpu",
            requested_device="cpu",
            device_name="cpu",
            total_memory_gb=None,
            peak_flops=float("inf"),
            tier="cpu",
        )
    if normalized_tier == "cuda_desktop":
        return HardwareInfo(
            backend="cuda",
            requested_device="cuda",
            device_name="NVIDIA RTX 4090",
            total_memory_gb=24.0,
            peak_flops=165.2e12,
            tier="cuda_desktop",
        )
    if normalized_tier == "cuda_datacenter":
        return HardwareInfo(
            backend="cuda",
            requested_device="cuda",
            device_name="NVIDIA A100 80GB",
            total_memory_gb=80.0,
            peak_flops=312e12,
            tier="cuda_datacenter",
        )
    if normalized_tier == "cuda_h100":
        return HardwareInfo(
            backend="cuda",
            requested_device="cuda",
            device_name="NVIDIA H100 SXM",
            total_memory_gb=80.0,
            peak_flops=989e12,
            tier="cuda_h100",
        )
    raise ValueError(f"Unsupported mock hardware tier {tier!r}.")


@pytest.fixture
def hardware_info_factory():
    """Return one helper that builds stable mock hardware profiles by tier."""

    return _build_mock_hardware
