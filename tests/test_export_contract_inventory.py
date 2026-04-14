from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from conftest import load_repo_config, load_script_module

from dagzoo.core.dataset import generate_batch
from dagzoo.core.generate_handoff import build_generate_handoff_manifest
from dagzoo.diagnostics.coverage import CoverageAggregationConfig, CoverageAggregator
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.io.shard_contract import DATASET_CATALOG_FILENAME, iter_ndjson_records

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "reference" / "export_contract_inventory.yaml"
CATALOG_PATH = REPO_ROOT / "docs" / "export-contract-fields.md"


@dataclass(frozen=True, slots=True)
class _InventoryEntry:
    artifact: str
    path: str
    path_tokens: tuple[str, ...]
    type_spec: str
    presence: str


def _normalize_inventory_path(path: str) -> str:
    if len(path) >= 2 and path[0] == "'" and path[-1] == "'":
        return path[1:-1]
    return path


def _parse_path_tokens(path: str) -> tuple[str, ...]:
    normalized = _normalize_inventory_path(path).replace("[]", ".[]")
    return tuple(segment for segment in normalized.split(".") if segment)


def _format_tokens(tokens: tuple[str, ...]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token == "[]":
            if parts:
                parts[-1] = f"{parts[-1]}[]"
            else:
                parts.append("[]")
            continue
        parts.append(token)
    return ".".join(parts)


def _flatten_tokens(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    flattened: list[tuple[tuple[str, ...], Any]] = []
    if prefix:
        flattened.append((prefix, value))
    if isinstance(value, dict):
        for key, child in value.items():
            flattened.extend(_flatten_tokens(child, prefix + (str(key),)))
    elif isinstance(value, list):
        for child in value:
            flattened.extend(_flatten_tokens(child, prefix + ("[]",)))
    return flattened


def _path_matches(pattern_tokens: tuple[str, ...], actual_tokens: tuple[str, ...]) -> bool:
    if len(pattern_tokens) != len(actual_tokens):
        return False
    return all(
        pattern == "*" or pattern == actual
        for pattern, actual in zip(pattern_tokens, actual_tokens)
    )


def _split_top_level_union(type_spec: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in type_spec:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif ch == "|" and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _is_int_like(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number_like(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _value_matches_atomic_type(value: Any, type_spec: str) -> bool:
    if type_spec == "mixed":
        return True
    if type_spec == "null":
        return value is None
    if type_spec == "object":
        return isinstance(value, dict)
    if type_spec == "list":
        return isinstance(value, list)
    if type_spec.startswith("list["):
        return isinstance(value, list)
    if type_spec.startswith("torch.Tensor["):
        return isinstance(value, torch.Tensor)
    if type_spec == "string":
        return isinstance(value, str)
    if type_spec in {"int", "int64"}:
        return _is_int_like(value)
    if type_spec in {"float", "float32", "float64", "number"}:
        return _is_number_like(value)
    if type_spec == "bool":
        return isinstance(value, bool)
    return False


def _value_matches_type(value: Any, type_spec: str) -> bool:
    return any(
        _value_matches_atomic_type(value, option) for option in _split_top_level_union(type_spec)
    )


def _best_inventory_match(
    actual_tokens: tuple[str, ...],
    entries: list[_InventoryEntry],
) -> _InventoryEntry | None:
    matches = [entry for entry in entries if _path_matches(entry.path_tokens, actual_tokens)]
    if not matches:
        return None
    return max(
        matches,
        key=lambda entry: (
            sum(1 for token in entry.path_tokens if token != "*"),
            len(entry.path_tokens),
        ),
    )


def _load_inventory() -> tuple[dict[str, object], dict[str, list[_InventoryEntry]]]:
    payload = yaml.safe_load(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    entries_raw = payload.get("entries")
    assert isinstance(entries_raw, list)

    by_artifact: dict[str, list[_InventoryEntry]] = {}
    for raw_entry in entries_raw:
        assert isinstance(raw_entry, Mapping)
        artifact = raw_entry.get("artifact")
        path = raw_entry.get("path")
        type_spec = raw_entry.get("type")
        presence = raw_entry.get("presence")
        assert isinstance(artifact, str)
        assert isinstance(path, str)
        assert isinstance(type_spec, str)
        assert isinstance(presence, str)
        by_artifact.setdefault(artifact, []).append(
            _InventoryEntry(
                artifact=artifact,
                path=_normalize_inventory_path(path),
                path_tokens=_parse_path_tokens(path),
                type_spec=type_spec,
                presence=presence,
            )
        )
    return payload, by_artifact


def _contract_sample_artifacts(tmp_path: Path) -> dict[str, Any]:
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    cfg = load_repo_config()
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.dataset.task = "classification"
    cfg.dataset.n_train = 16
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6
    cfg.dataset.missing_rate = 0.1
    cfg.dataset.missing_mechanism = "mcar"

    batch = generate_batch(cfg, num_datasets=1, seed=123, device="cpu")
    bundle = batch[0]

    handoff_root = tmp_path / "handoff_root"
    generated_dir = handoff_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    write_packed_parquet_shards_stream(batch, generated_dir, shard_size=8, compression="zstd")

    effective_config_path = generated_dir / "effective_config.yaml"
    effective_trace_path = generated_dir / "effective_config_trace.yaml"
    effective_config_path.write_text("seed: 123\n", encoding="utf-8")
    effective_trace_path.write_text("- source: unit-test\n", encoding="utf-8")

    catalog_record = next(
        iter_ndjson_records(generated_dir / "shard_00000" / DATASET_CATALOG_FILENAME)
    )
    train_row = (
        pyarrow_parquet.read_table(generated_dir / "shard_00000" / "train.parquet")
        .slice(0, 1)
        .to_pylist()[0]
    )
    manifest = build_generate_handoff_manifest(
        config_path="recipe:default-baseline",
        generate_invocation_overrides={
            "num_datasets": 1,
            "seed": 123,
            "rows": None,
            "device": "cpu",
            "hardware_policy": "none",
            "missing_rate": 0.1,
            "missing_mechanism": "mcar",
            "missing_mar_observed_fraction": None,
            "missing_mar_logit_scale": None,
            "missing_mnar_logit_scale": None,
            "diagnostics": True,
            "diagnostics_out_dir": None,
            "handoff_root": str(handoff_root),
            "set_overrides": [["dataset.n_train", 16]],
        },
        run_root=handoff_root,
        generated_dir=generated_dir,
        effective_config_path=effective_config_path,
        effective_config_trace_path=effective_trace_path,
        generated_datasets=1,
        generation_elapsed_seconds=1.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="cpu",
        hardware_tier="cpu",
        hardware_policy="none",
    )

    aggregator = CoverageAggregator(
        CoverageAggregationConfig(
            histogram_bins=4,
            quantiles=(0.25, 0.5, 0.75),
        )
    )
    aggregator.update_bundle(bundle)
    coverage_summary = aggregator.build_summary()
    coverage_surface = {
        "generated_at": coverage_summary["generated_at"],
        "num_datasets": coverage_summary["num_datasets"],
        "task_counts": coverage_summary["task_counts"],
        "histogram_bins": coverage_summary["histogram_bins"],
        "quantiles": coverage_summary["quantiles"],
    }

    return {
        "dataset_catalog_record": catalog_record,
        "parquet_split_row": train_row,
        "generate_handoff_manifest": manifest,
        "coverage_summary_json": coverage_surface,
    }


def test_export_contract_inventory_is_well_formed() -> None:
    payload, by_artifact = _load_inventory()

    assert payload["schema"] == "dagzoo-export-contract-inventory-v1"
    artifacts = payload.get("artifacts")
    assert isinstance(artifacts, Mapping)
    assert set(by_artifact) == set(artifacts)

    seen: set[tuple[str, str]] = set()
    for artifact, entries in by_artifact.items():
        assert entries, f"{artifact} should have at least one inventory entry."
        for entry in entries:
            key = (artifact, entry.path)
            assert key not in seen, f"Duplicate inventory entry for {artifact}:{entry.path}"
            seen.add(key)


def test_export_contract_catalog_is_in_sync() -> None:
    module = load_script_module(
        "render_export_contract_catalog",
        "scripts/contracts/render_export_contract_catalog.py",
    )

    assert CATALOG_PATH.exists()
    assert module.main(["--check", "--output", str(CATALOG_PATH)]) == 0


def test_export_contract_inventory_covers_real_generated_artifacts(tmp_path: Path) -> None:
    _, by_artifact = _load_inventory()
    actual_artifacts = _contract_sample_artifacts(tmp_path)

    uncovered_paths: list[str] = []
    missing_always_present_paths: list[str] = []
    type_mismatches: list[str] = []

    for artifact, actual_payload in actual_artifacts.items():
        entries = by_artifact[artifact]
        actual_items = _flatten_tokens(actual_payload)
        actual_token_set = {tokens for tokens, _ in actual_items}

        for tokens, value in actual_items:
            match = _best_inventory_match(tokens, entries)
            if match is None:
                uncovered_paths.append(f"{artifact}:{_format_tokens(tokens)}")
                continue
            if not _value_matches_type(value, match.type_spec):
                type_mismatches.append(
                    f"{artifact}:{_format_tokens(tokens)} expected {match.type_spec!r}, got {type(value).__name__}"
                )

        for entry in entries:
            if entry.presence != "always":
                continue
            if not any(_path_matches(entry.path_tokens, tokens) for tokens in actual_token_set):
                missing_always_present_paths.append(f"{artifact}:{entry.path}")

    assert uncovered_paths == []
    assert missing_always_present_paths == []
    assert type_mismatches == []
