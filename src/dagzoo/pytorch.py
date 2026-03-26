"""PyTorch-native bridge for consuming dagzoo in process."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dagzoo.config import GeneratorConfig, clone_generator_config
from dagzoo.core.dataset import (
    _iter_prepared_canonical_batch_iter,
    _validate_public_generation_config,
)
from dagzoo.core.fixed_layout.runtime import prepare_canonical_fixed_layout_run
from dagzoo.types import DatasetBundle


class DagzooSample(TypedDict):
    """One PyTorch-friendly dagzoo task sample."""

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    feature_types: list[str]
    metadata: dict[str, Any]


def _resolve_config(config: GeneratorConfig | str | Path) -> GeneratorConfig:
    if isinstance(config, GeneratorConfig):
        return clone_generator_config(config, revalidate=False)
    if isinstance(config, str | Path):
        return GeneratorConfig.from_yaml(config)
    raise TypeError(
        f"config must be a GeneratorConfig, string path, or Path, got {type(config)!r}."
    )


def _validate_optional_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer or None, got {type(seed)!r}.")
    return int(seed)


def _validate_num_datasets(num_datasets: int) -> int:
    if isinstance(num_datasets, bool) or not isinstance(num_datasets, int):
        raise TypeError(f"num_datasets must be an integer, got {type(num_datasets)!r}.")
    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    return int(num_datasets)


def _validate_num_workers(num_workers: int) -> int:
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError(f"num_workers must be an integer, got {type(num_workers)!r}.")
    if num_workers != 0:
        raise ValueError(
            "dagzoo.pytorch v1 only supports num_workers=0. "
            "Multi-worker loading would duplicate generation without explicit worker sharding."
        )
    return 0


def _bundle_to_sample(bundle: DatasetBundle) -> DagzooSample:
    return DagzooSample(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        feature_types=list(bundle.feature_types),
        metadata=dict(bundle.metadata),
    )


class DagzooDataset(IterableDataset[DagzooSample]):
    """Yield canonical dagzoo datasets as PyTorch-friendly samples.

    One ``DagzooDataset`` instance corresponds to one canonical dagzoo run, so
    all yielded datasets preserve the existing shared fixed-layout semantics for
    that run.
    """

    def __init__(
        self,
        config: GeneratorConfig | str | Path,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self._config = _resolve_config(config)
        _validate_public_generation_config(self._config)
        self._num_datasets = _validate_num_datasets(num_datasets)
        self._seed = _validate_optional_seed(seed)
        self._device = None if device is None else str(device)

    def __len__(self) -> int:
        return int(self._num_datasets)

    def __iter__(self) -> Iterator[DagzooSample]:
        worker_info = get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "DagzooDataset does not support DataLoader workers yet. Use num_workers=0."
            )
        if self._num_datasets == 0:
            return
        prepared = prepare_canonical_fixed_layout_run(
            self._config,
            num_datasets=self._num_datasets,
            seed=self._seed,
            device=self._device,
        )
        for bundle in _iter_prepared_canonical_batch_iter(
            prepared,
            num_datasets=self._num_datasets,
        ):
            yield _bundle_to_sample(bundle)


def build_dataloader(
    config: GeneratorConfig | str | Path,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
    num_workers: int = 0,
) -> DataLoader[DagzooSample]:
    """Return a task-sized ``DataLoader`` for one canonical dagzoo run."""

    _validate_num_workers(num_workers)
    return DataLoader(
        DagzooDataset(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        ),
        batch_size=None,
        num_workers=0,
    )


__all__ = [
    "DagzooDataset",
    "DagzooSample",
    "build_dataloader",
]
