"""Seeded synthetic dataset generation public entrypoints."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from dagzoo.config import GeneratorConfig
from dagzoo.core.fixed_layout.runtime import (
    CanonicalFixedLayoutRun,
    _generate_batch_with_plan_iter,
    prepare_canonical_fixed_layout_run,
)
from dagzoo.core.identity import (
    canonical_dataset_id,
    canonical_layout_plan_split_group,
    canonical_request_run_provenance,
    canonical_request_run_split_group,
)
from dagzoo.rng import KeyedRng
from dagzoo.types import DatasetBundle

__all__ = [
    "generate_batch",
    "generate_batch_iter",
    "generate_one",
]


def _validate_public_generation_config(config: GeneratorConfig) -> None:
    """Reject public generation configs that are intentionally unsupported."""

    if bool(config.filter.enabled):
        raise ValueError(
            "Inline filtering has been removed from generate. Set filter.enabled=false and run "
            "`dagzoo filter --in <shard_dir> --out <out_dir>` after generation."
        )


def _require_metadata_string(metadata: Mapping[str, Any], *, key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Canonical generation metadata is missing required string field {key!r}.")
    return value


def _annotate_canonical_batch_metadata(
    bundle: DatasetBundle,
    *,
    run_seed: int,
    dataset_index: int,
    run_num_datasets: int,
) -> DatasetBundle:
    """Rewrite canonical bundle metadata to preserve run-level replay information."""

    dataset_seed = bundle.metadata.get("seed")
    if not isinstance(dataset_seed, int) or isinstance(dataset_seed, bool):
        dataset_seed = KeyedRng(int(run_seed)).child_seed("dataset", int(dataset_index))
    keyed_replay = bundle.metadata.get("keyed_replay")
    if not isinstance(keyed_replay, dict):
        keyed_replay = {}
    keyed_replay["dataset_root_path"] = ["dataset", int(dataset_index)]
    layout_signature = _require_metadata_string(bundle.metadata, key="layout_signature")
    layout_plan_signature = _require_metadata_string(bundle.metadata, key="layout_plan_signature")
    layout_execution_contract = _require_metadata_string(
        bundle.metadata,
        key="layout_execution_contract",
    )
    request_run_provenance = canonical_request_run_provenance(bundle.metadata)
    split_groups = {
        "request_run": canonical_request_run_split_group(
            seed=int(run_seed),
            run_num_datasets=int(run_num_datasets),
            layout_signature=layout_signature,
            layout_plan_signature=layout_plan_signature,
            request_run_provenance=request_run_provenance,
        ),
        "layout_plan": canonical_layout_plan_split_group(
            layout_signature=layout_signature,
            layout_plan_signature=layout_plan_signature,
            layout_execution_contract=layout_execution_contract,
        ),
    }
    bundle.metadata["seed"] = int(run_seed)
    bundle.metadata["dataset_seed"] = int(dataset_seed)
    bundle.metadata["dataset_index"] = int(dataset_index)
    bundle.metadata["dataset_id"] = canonical_dataset_id(
        request_run_split_group=split_groups["request_run"],
        layout_plan_split_group=split_groups["layout_plan"],
        dataset_index=int(dataset_index),
        dataset_seed=int(dataset_seed),
    )
    bundle.metadata["run_num_datasets"] = int(run_num_datasets)
    bundle.metadata["split_groups"] = split_groups
    bundle.metadata["keyed_replay"] = keyed_replay
    return bundle


def generate_one(
    config: GeneratorConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> DatasetBundle:
    """Generate one dataset bundle using the canonical fixed-layout run model."""

    return next(generate_batch_iter(config, num_datasets=1, seed=seed, device=device))


def generate_batch(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> list[DatasetBundle]:
    """Generate a batch of datasets using deterministic per-dataset child seeds."""

    return list(
        generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        )
    )


def generate_batch_iter(
    config: GeneratorConfig,
    *,
    num_datasets: int,
    seed: int | None = None,
    device: str | None = None,
) -> Iterator[DatasetBundle]:
    """Yield datasets from one canonical fixed-layout run.

    Classification runs emit bundles as soon as fixed-layout generation succeeds
    for each dataset. A later dataset may still exhaust the retry budget after
    earlier bundles have already been yielded.
    """

    if num_datasets < 0:
        raise ValueError(f"num_datasets must be >= 0, got {num_datasets}")
    if num_datasets == 0:
        return
    _validate_public_generation_config(config)

    prepared = prepare_canonical_fixed_layout_run(
        config,
        num_datasets=num_datasets,
        seed=seed,
        device=device,
    )
    yield from _iter_prepared_canonical_batch_iter(prepared, num_datasets=num_datasets)


def _iter_prepared_canonical_batch_iter(
    prepared: CanonicalFixedLayoutRun,
    *,
    num_datasets: int,
) -> Iterator[DatasetBundle]:
    """Yield annotated bundles from one already-prepared canonical fixed-layout run."""

    for dataset_index, bundle in enumerate(
        _generate_batch_with_plan_iter(
            prepared.config,
            plan=prepared.plan,
            num_datasets=num_datasets,
            seed=prepared.run_seed,
            batch_size=prepared.batch_size,
        )
    ):
        yield _annotate_canonical_batch_metadata(
            bundle,
            run_seed=prepared.run_seed,
            dataset_index=dataset_index,
            run_num_datasets=num_datasets,
        )
