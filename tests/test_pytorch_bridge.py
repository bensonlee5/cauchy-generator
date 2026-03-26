from __future__ import annotations

import pytest
import torch
from conftest import load_repo_config, write_config

from dagzoo.core.dataset import generate_batch_iter
from dagzoo.pytorch import DagzooDataset, build_dataloader


def _tiny_bridge_config():
    cfg = load_repo_config()
    cfg.filter.enabled = False
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    return cfg


def test_dagzoo_dataset_matches_generate_batch_iter_outputs() -> None:
    cfg = _tiny_bridge_config()

    samples = list(DagzooDataset(cfg, num_datasets=3, seed=123, device="cpu"))
    bundles = list(generate_batch_iter(cfg, num_datasets=3, seed=123, device="cpu"))

    assert len(samples) == len(bundles) == 3
    for sample, bundle in zip(samples, bundles, strict=True):
        assert sample["metadata"]["dataset_id"] == bundle.metadata["dataset_id"]
        assert sample["feature_types"] == bundle.feature_types
        assert sample["metadata"] == bundle.metadata
        assert torch.equal(sample["X_train"], bundle.X_train)
        assert torch.equal(sample["y_train"], bundle.y_train)
        assert torch.equal(sample["X_test"], bundle.X_test)
        assert torch.equal(sample["y_test"], bundle.y_test)


def test_dagzoo_dataset_accepts_config_path(tmp_path) -> None:
    cfg = _tiny_bridge_config()
    config_path = write_config(tmp_path, cfg)

    from_path = list(DagzooDataset(config_path, num_datasets=1, seed=321, device="cpu"))
    from_config = list(DagzooDataset(cfg, num_datasets=1, seed=321, device="cpu"))

    assert len(from_path) == len(from_config) == 1
    assert from_path[0]["metadata"]["dataset_id"] == from_config[0]["metadata"]["dataset_id"]
    assert torch.equal(from_path[0]["X_train"], from_config[0]["X_train"])
    assert torch.equal(from_path[0]["y_train"], from_config[0]["y_train"])
    assert torch.equal(from_path[0]["X_test"], from_config[0]["X_test"])
    assert torch.equal(from_path[0]["y_test"], from_config[0]["y_test"])


def test_build_dataloader_returns_task_sized_samples() -> None:
    cfg = _tiny_bridge_config()

    direct_sample = next(iter(DagzooDataset(cfg, num_datasets=1, seed=55, device="cpu")))
    loader = build_dataloader(cfg, num_datasets=1, seed=55, device="cpu")
    batch = next(iter(loader))

    assert loader.batch_size is None
    assert isinstance(batch["metadata"], dict)
    assert batch["feature_types"] == direct_sample["feature_types"]
    assert batch["X_train"].ndim == 2
    assert batch["y_train"].ndim == 1
    assert batch["X_test"].ndim == 2
    assert batch["y_test"].ndim == 1
    assert torch.equal(batch["X_train"], direct_sample["X_train"])
    assert torch.equal(batch["y_train"], direct_sample["y_train"])
    assert torch.equal(batch["X_test"], direct_sample["X_test"])
    assert torch.equal(batch["y_test"], direct_sample["y_test"])


def test_dagzoo_dataset_handles_zero_num_datasets() -> None:
    cfg = _tiny_bridge_config()
    dataset = DagzooDataset(cfg, num_datasets=0, seed=7, device="cpu")

    assert len(dataset) == 0
    assert list(dataset) == []


def test_dagzoo_dataset_rejects_inline_filter_enabled() -> None:
    cfg = _tiny_bridge_config()
    cfg.filter.enabled = True

    with pytest.raises(ValueError, match="Inline filtering has been removed from generate"):
        _ = DagzooDataset(cfg, num_datasets=1, seed=7, device="cpu")

    with pytest.raises(ValueError, match="Inline filtering has been removed from generate"):
        _ = build_dataloader(cfg, num_datasets=1, seed=7, device="cpu")


def test_dagzoo_dataset_rejects_multi_worker_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _tiny_bridge_config()

    with pytest.raises(ValueError, match="num_workers=0"):
        _ = build_dataloader(cfg, num_datasets=1, seed=7, device="cpu", num_workers=1)

    monkeypatch.setattr("dagzoo.pytorch.get_worker_info", lambda: object())
    with pytest.raises(RuntimeError, match="num_workers=0"):
        _ = list(DagzooDataset(cfg, num_datasets=1, seed=7, device="cpu"))
