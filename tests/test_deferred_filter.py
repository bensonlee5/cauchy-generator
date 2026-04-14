import json
from pathlib import Path

import numpy as np
import pytest

from dagzoo.filtering.deferred_filter import (
    _iter_packed_split_datasets,
    run_deferred_filter,
)
from dagzoo.io.lineage_artifact import pack_upper_triangle_adjacency, sha256_hex
from dagzoo.io.lineage_schema import (
    LINEAGE_ADJACENCY_ENCODING,
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION_COMPACT,
    LINEAGE_SCHEMA_VERSION_DENSE,
)
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.io.shard_contract import (
    DATASET_CATALOG_FILENAME,
    REPLAY_CATALOG_FILENAME,
    iter_parquet_json_records,
    write_dataset_catalog_records,
    write_json_record_parquet_records,
)
from dagzoo.types import DatasetBundle


def _allow_deferred_filter_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = monkeypatch


def _bundle_with_embedded_config(
    seed: int,
    *,
    dataset_seed: int | None = None,
    dataset_index: int | None = None,
    dataset_id: str | None = None,
    split_groups: dict[str, str] | None = None,
    filter_overrides: dict[str, object] | None = None,
    lineage: dict[str, object] | None = None,
) -> DatasetBundle:
    embedded_filter = {"enabled": True}
    if filter_overrides is not None:
        embedded_filter.update(filter_overrides)
    metadata = {
        "seed": seed,
        "filter": {"mode": "deferred", "status": "not_run"},
        "config": {
            "dataset": {"task": "classification"},
            "filter": embedded_filter,
        },
    }
    if dataset_seed is not None:
        metadata["dataset_seed"] = int(dataset_seed)
    if dataset_index is not None:
        metadata["dataset_index"] = int(dataset_index)
    if dataset_id is not None:
        metadata["dataset_id"] = str(dataset_id)
    if split_groups is not None:
        metadata["split_groups"] = dict(split_groups)
    if lineage is not None:
        metadata["lineage"] = dict(lineage)

    return DatasetBundle(
        X_train=np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        ),
        y_train=np.array([0, 1, 0], dtype=np.int64),
        X_test=np.array([[1.5, 0.5], [0.5, 1.5]], dtype=np.float32),
        y_test=np.array([1, 0], dtype=np.int64),
        feature_types=["num", "num"],
        metadata=metadata,
    )


def _bundle_without_config(seed: int) -> DatasetBundle:
    bundle = _bundle_with_embedded_config(seed)
    bundle.metadata["config"] = {"dataset": {"task": "classification"}}
    return bundle


def _load_ndjson(path) -> list[dict[str, object]]:
    return [dict(record) for record in iter_parquet_json_records(path)]


def _write_ndjson_records(path, records: list[dict[str, object]]) -> None:
    resolved_path = Path(path)
    if resolved_path.name == DATASET_CATALOG_FILENAME:
        write_dataset_catalog_records(path, records)
        return
    write_json_record_parquet_records(resolved_path, records)


def _rewrite_replay_lineage_to_compact(metadata_path: Path) -> list[dict[str, object]]:
    records = _load_ndjson(metadata_path)
    record = records[0]
    metadata = record["metadata"]
    assert isinstance(metadata, dict)
    lineage = metadata["lineage"]
    assert isinstance(lineage, dict)
    graph = lineage["graph"]
    assert isinstance(graph, dict)
    if "adjacency_ref" in graph:
        return records
    adjacency = graph["adjacency"]
    n_nodes, edge_count, payload = pack_upper_triangle_adjacency(adjacency)
    lineage_dir = metadata_path.parent / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    blob_path = lineage_dir / "adjacency.bitpack.bin"
    index_path = lineage_dir / "adjacency.index.json"
    blob_path.write_bytes(payload)
    index_path.write_text("{}", encoding="utf-8")
    lineage["schema_name"] = LINEAGE_SCHEMA_NAME
    lineage["schema_version"] = LINEAGE_SCHEMA_VERSION_COMPACT
    lineage["graph"] = {
        "n_nodes": n_nodes,
        "edge_count": edge_count,
        "adjacency_ref": {
            "encoding": LINEAGE_ADJACENCY_ENCODING,
            "blob_path": "lineage/adjacency.bitpack.bin",
            "index_path": "lineage/adjacency.index.json",
            "dataset_index": 0,
            "bit_offset": 0,
            "bit_length": (n_nodes * (n_nodes - 1)) // 2,
            "sha256": sha256_hex(payload),
        },
    }
    _write_ndjson_records(metadata_path, records)
    return records


def _dense_lineage_payload() -> dict[str, object]:
    return {
        "schema_name": "dagzoo.dag_lineage",
        "schema_version": LINEAGE_SCHEMA_VERSION_DENSE,
        "graph": {
            "n_nodes": 4,
            "adjacency": [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
        },
        "assignments": {
            "feature_to_node": [0, 0],
            "target_to_node": 3,
            "target_relevant_features": [],
            "target_relevant_feature_count": 0,
            "target_relevant_feature_fraction": 0.0,
        },
    }


def _write_split_table(
    path,
    *,
    dataset_indices: list[int],
    row_indices: list[int],
    x_rows: list[list[float]],
    y_rows: list[int],
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    table = pyarrow.table(
        {
            "dataset_index": pyarrow.array(dataset_indices, type=pyarrow.int64()),
            "row_index": pyarrow.array(row_indices, type=pyarrow.int64()),
            "x": pyarrow.array(x_rows, type=pyarrow.list_(pyarrow.float32())),
            "y": pyarrow.array(y_rows, type=pyarrow.int64()),
        }
    )
    pyarrow_parquet.write_table(table, path, compression="zstd")


def test_iter_packed_split_datasets_handles_dataset_split_across_record_batches(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")

    table = pyarrow.table(
        {
            "dataset_index": pyarrow.array([0, 0, 0, 1, 1], type=pyarrow.int64()),
            "row_index": pyarrow.array([0, 1, 2, 0, 1], type=pyarrow.int64()),
            "x": pyarrow.array(
                [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]],
                type=pyarrow.list_(pyarrow.float32()),
            ),
            "y": pyarrow.array([0, 1, 0, 1, 0], type=pyarrow.int64()),
        }
    )
    batches = [
        table.slice(0, 2).to_batches()[0],
        table.slice(2, 1).to_batches()[0],
        table.slice(3, 2).to_batches()[0],
    ]

    class _FakeParquetFile:
        def __init__(self, _path) -> None:
            self.schema_arrow = table.schema

        def iter_batches(self, *, columns):
            assert columns == ["dataset_index", "row_index", "x", "y"]
            return iter(batches)

    monkeypatch.setattr("dagzoo.filtering.deferred_filter.pq.ParquetFile", _FakeParquetFile)

    split_path = tmp_path / "train.parquet"
    datasets = list(_iter_packed_split_datasets(split_path))

    assert [dataset.dataset_index for dataset in datasets] == [0, 1]
    assert datasets[0].x.shape == (3, 2)
    assert datasets[0].y.tolist() == [0, 1, 0]
    assert np.allclose(datasets[0].x[:, 0], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert datasets[1].x.shape == (2, 2)
    assert datasets[1].y.tolist() == [1, 0]


def test_iter_packed_split_datasets_handles_mixed_feature_widths_within_record_batch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")

    table = pyarrow.table(
        {
            "dataset_index": pyarrow.array([0, 0, 1, 1], type=pyarrow.int64()),
            "row_index": pyarrow.array([0, 1, 0, 1], type=pyarrow.int64()),
            "x": pyarrow.array(
                [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5, 2.75], [3.0, 3.5, 3.75]],
                type=pyarrow.list_(pyarrow.float32()),
            ),
            "y": pyarrow.array([0, 1, 1, 0], type=pyarrow.int64()),
        }
    )
    batches = table.to_batches(max_chunksize=4)

    class _FakeParquetFile:
        def __init__(self, _path) -> None:
            self.schema_arrow = table.schema

        def iter_batches(self, *, columns):
            assert columns == ["dataset_index", "row_index", "x", "y"]
            return iter(batches)

    monkeypatch.setattr("dagzoo.filtering.deferred_filter.pq.ParquetFile", _FakeParquetFile)

    split_path = tmp_path / "train.parquet"
    datasets = list(_iter_packed_split_datasets(split_path))

    assert [dataset.dataset_index for dataset in datasets] == [0, 1]
    assert datasets[0].x.shape == (2, 2)
    assert datasets[0].y.tolist() == [0, 1]
    assert datasets[1].x.shape == (2, 3)
    assert datasets[1].y.tolist() == [1, 0]


def test_run_deferred_filter_writes_manifest_and_summary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_with_embedded_config(101), _bundle_with_embedded_config(102)]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=2, compression="zstd")

    filter_calls = {"count": 0}

    def _stub_filter(*_args, **_kwargs):
        filter_calls["count"] += 1
        accepted = filter_calls["count"] == 1
        details = {"filter_mode": "structural_v1", "target_indegree": 2}
        if not accepted:
            details["reason"] = "target_root"
        return accepted, details

    monkeypatch.setattr("dagzoo.filtering.deferred_filter.apply_structural_filter", _stub_filter)

    result = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)
    assert result.total_datasets == 2
    assert result.accepted_datasets == 1
    assert result.rejected_datasets == 1

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["accepted_datasets"] == 1
    assert summary["rejected_datasets"] == 1
    assert summary["filter_mode"] == "structural_v1"
    assert summary["rejected_reason_counts"]["target_root"] == 1

    manifest_records = _load_ndjson(result.manifest_path)
    assert len(manifest_records) == 2
    statuses = {str(record["status"]) for record in manifest_records}
    assert statuses == {"accepted", "rejected"}
    for record in manifest_records:
        filter_payload = record["filter"]
        assert isinstance(filter_payload, dict)
        assert filter_payload["mode"] == "deferred"
        assert filter_payload["status"] in {"accepted", "rejected"}


def test_run_deferred_filter_writes_curated_output_for_accepted_only(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    curated_out = tmp_path / "curated"
    bundles = [
        _bundle_with_embedded_config(
            201,
            dataset_id="dataset-201",
            split_groups={"request_run": "run-group-a", "layout_plan": "layout-group-x"},
        ),
        _bundle_with_embedded_config(
            202,
            dataset_id="dataset-202",
            split_groups={"request_run": "run-group-a", "layout_plan": "layout-group-x"},
        ),
        _bundle_with_embedded_config(
            203,
            dataset_id="dataset-203",
            split_groups={"request_run": "run-group-a", "layout_plan": "layout-group-x"},
        ),
    ]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=3, compression="zstd")

    filter_calls = {"count": 0}

    def _stub_filter(*_args, **_kwargs):
        filter_calls["count"] += 1
        accepted = filter_calls["count"] != 2
        return accepted, {"filter_mode": "structural_v1"}

    monkeypatch.setattr("dagzoo.filtering.deferred_filter.apply_structural_filter", _stub_filter)

    result = run_deferred_filter(in_dir=in_dir, out_dir=out_dir, curated_out_dir=curated_out)
    assert result.curated_accepted_datasets == 2

    shard_dir = curated_out / "shard_00000"
    assert shard_dir.exists()
    input_metadata_records = _load_ndjson(in_dir / "shard_00000" / DATASET_CATALOG_FILENAME)
    metadata_records = _load_ndjson(shard_dir / DATASET_CATALOG_FILENAME)
    assert [int(record["dataset_index"]) for record in metadata_records] == [0, 2]
    input_metadata_by_index = {
        int(record["dataset_index"]): record for record in input_metadata_records
    }
    curated_metadata_by_index = {
        int(record["dataset_index"]): record for record in metadata_records
    }
    for dataset_index in (0, 2):
        assert (
            curated_metadata_by_index[dataset_index]["dataset_id"]
            == input_metadata_by_index[dataset_index]["dataset_id"]
        )
        assert (
            curated_metadata_by_index[dataset_index]["group_ids"]
            == input_metadata_by_index[dataset_index]["group_ids"]
        )

    train_table = pyarrow_parquet.read_table(shard_dir / "train.parquet")
    dataset_indices = {int(value) for value in train_table.column("dataset_index").to_pylist()}
    assert dataset_indices == {0, 2}


def test_run_deferred_filter_requires_embedded_filter_config(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_without_config(7)]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=1, compression="zstd")

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter.apply_structural_filter",
        lambda *_args, **_kwargs: (True, {"filter_mode": "structural_v1"}),
    )

    with pytest.raises(ValueError, match="requires embedded metadata\\.config\\.filter"):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)


def test_run_deferred_filter_prefers_dataset_seed_when_present(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [
        _bundle_with_embedded_config(101, dataset_seed=501, dataset_index=0),
        _bundle_with_embedded_config(102, dataset_seed=502, dataset_index=1),
    ]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=2, compression="zstd")

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter.apply_structural_filter",
        lambda *_args, **_kwargs: (True, {"filter_mode": "structural_v1"}),
    )

    result = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)

    assert result.total_datasets == 2
    assert result.accepted_datasets == 2
    manifest_records = _load_ndjson(result.manifest_path)
    assert [int(record["seed"]) for record in manifest_records] == [501, 502]


def test_run_deferred_filter_resolves_internal_sidecars_from_direct_handoff_shard_input(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    handoff_root = tmp_path / "handoff"
    generated_dir = handoff_root / "generated"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_with_embedded_config(601, filter_overrides={"enabled": False})]
    _ = write_packed_parquet_shards_stream(
        bundles,
        generated_dir,
        shard_size=1,
        compression="zstd",
        internal_root=handoff_root / "internal",
    )

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter.apply_structural_filter",
        lambda *_args, **_kwargs: (True, {"filter_mode": "structural_v1"}),
    )

    result = run_deferred_filter(in_dir=generated_dir / "shard_00000", out_dir=out_dir)

    assert result.total_datasets == 1
    assert result.accepted_datasets == 1
    assert (handoff_root / "internal" / "shard_00000" / REPLAY_CATALOG_FILENAME).exists()
    manifest_records = _load_ndjson(result.manifest_path)
    assert [str(record["status"]) for record in manifest_records] == ["accepted"]


def test_run_deferred_filter_applies_structural_overrides_and_records_summary_provenance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_with_embedded_config(301), _bundle_with_embedded_config(302)]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=2, compression="zstd")

    seen_min_target_indegrees: list[int] = []
    seen_min_target_feature_counts: list[int] = []

    def _stub_filter(*_args, **_kwargs):
        seen_min_target_indegrees.append(int(_kwargs["min_target_indegree"]))
        seen_min_target_feature_counts.append(int(_kwargs["min_target_relevant_feature_count"]))
        return True, {
            "filter_mode": "structural_v1",
            "min_target_indegree": int(_kwargs["min_target_indegree"]),
            "min_target_relevant_feature_count": int(_kwargs["min_target_relevant_feature_count"]),
        }

    monkeypatch.setattr("dagzoo.filtering.deferred_filter.apply_structural_filter", _stub_filter)

    result = run_deferred_filter(
        in_dir=in_dir,
        out_dir=out_dir,
        path_overrides=(
            ("filter.min_target_indegree", 0),
            ("filter.min_target_relevant_feature_count", 1),
        ),
    )

    assert result.accepted_datasets == 2
    assert seen_min_target_indegrees == [0, 0]
    assert seen_min_target_feature_counts == [1, 1]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["filter_mode"] == "structural_v1"
    assert summary["path_overrides"] == [
        {"path": "filter.min_target_indegree", "value": 0},
        {"path": "filter.min_target_relevant_feature_count", "value": 1},
    ]
    manifest_records = _load_ndjson(result.manifest_path)
    assert manifest_records[0]["filter"]["min_target_indegree"] == 0
    assert manifest_records[0]["filter"]["min_target_relevant_feature_count"] == 1


def test_run_deferred_filter_uses_embedded_structural_threshold_when_override_is_omitted(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_with_embedded_config(401, filter_overrides={"min_target_indegree": 0})]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=1, compression="zstd")

    seen_thresholds: list[int] = []

    def _stub_filter(*_args, **_kwargs):
        seen_thresholds.append(int(_kwargs["min_target_indegree"]))
        return True, {"filter_mode": "structural_v1"}

    monkeypatch.setattr("dagzoo.filtering.deferred_filter.apply_structural_filter", _stub_filter)

    _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)

    assert seen_thresholds == [0]


def test_run_deferred_filter_rejects_threshold_era_embedded_filter_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_with_embedded_config(451, filter_overrides={"threshold": 0.95})]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=1, compression="zstd")

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter.apply_structural_filter",
        lambda *_args, **_kwargs: pytest.fail("filter replay should fail before model scoring"),
    )

    with pytest.raises(
        ValueError,
        match=r"Removed filter fields are not supported",
    ):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)


def test_run_deferred_filter_requires_lineage_metadata_for_structural_replay(
    tmp_path,
) -> None:
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_with_embedded_config(481)]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=1, compression="zstd")

    with pytest.raises(ValueError, match=r"Structural filtering requires metadata\.lineage"):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)


def test_run_deferred_filter_decodes_compact_lineage_and_applies_no_path_veto(
    tmp_path,
) -> None:
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_with_embedded_config(501, lineage=_dense_lineage_payload())]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=1, compression="zstd")

    result = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)

    assert result.accepted_datasets == 0
    assert result.rejected_datasets == 1
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["rejected_reason_counts"] == {"no_feature_target_path": 1}
    manifest_records = _load_ndjson(result.manifest_path)
    assert manifest_records[0]["reason"] == "no_feature_target_path"
    assert manifest_records[0]["filter"]["structural_filter_applied"] is True


def test_run_deferred_filter_rejects_compact_lineage_checksum_mismatch(
    tmp_path,
) -> None:
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_with_embedded_config(541, lineage=_dense_lineage_payload())]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=1, compression="zstd")

    metadata_path = in_dir / "internal" / "shard_00000" / REPLAY_CATALOG_FILENAME
    records = _rewrite_replay_lineage_to_compact(metadata_path)
    record = records[0]
    metadata = record["metadata"]
    assert isinstance(metadata, dict)
    lineage = metadata["lineage"]
    assert isinstance(lineage, dict)
    graph = lineage["graph"]
    assert isinstance(graph, dict)
    adjacency_ref = graph["adjacency_ref"]
    assert isinstance(adjacency_ref, dict)
    adjacency_ref["sha256"] = "f" * 64
    _write_ndjson_records(metadata_path, records)

    with pytest.raises(
        ValueError,
        match=(
            "metadata.lineage.graph.adjacency_ref.sha256 must match the resolved adjacency "
            "blob slice."
        ),
    ):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)


@pytest.mark.parametrize("blob_path_mode", ["relative_escape", "absolute"])
def test_run_deferred_filter_rejects_compact_lineage_blob_paths_outside_shard_tree(
    tmp_path,
    blob_path_mode: str,
) -> None:
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    bundles = [_bundle_with_embedded_config(551, lineage=_dense_lineage_payload())]
    _ = write_packed_parquet_shards_stream(bundles, in_dir, shard_size=1, compression="zstd")

    metadata_path = in_dir / "internal" / "shard_00000" / REPLAY_CATALOG_FILENAME
    records = _rewrite_replay_lineage_to_compact(metadata_path)
    record = records[0]
    metadata = record["metadata"]
    assert isinstance(metadata, dict)
    lineage = metadata["lineage"]
    assert isinstance(lineage, dict)
    graph = lineage["graph"]
    assert isinstance(graph, dict)
    adjacency_ref = graph["adjacency_ref"]
    assert isinstance(adjacency_ref, dict)
    adjacency_ref["blob_path"] = (
        "../outside.bin" if blob_path_mode == "relative_escape" else str(tmp_path / "outside.bin")
    )
    _write_ndjson_records(metadata_path, records)

    with pytest.raises(
        ValueError,
        match=(
            "metadata.lineage.graph.adjacency_ref.blob_path must be a relative path "
            "that resolves inside the shard lineage directory."
        ),
    ):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)


def test_run_deferred_filter_rejects_stale_filter_output_dir(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    _ = write_packed_parquet_shards_stream(
        [_bundle_with_embedded_config(301)],
        in_dir,
        shard_size=1,
        compression="zstd",
    )

    out_dir.mkdir()
    (out_dir / "filter_manifest.parquet").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already contains prior artifacts"):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)


def test_run_deferred_filter_rejects_extra_split_rows_beyond_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    _ = write_packed_parquet_shards_stream(
        [_bundle_with_embedded_config(401), _bundle_with_embedded_config(402)],
        in_dir,
        shard_size=2,
        compression="zstd",
    )

    metadata_path = in_dir / "shard_00000" / DATASET_CATALOG_FILENAME
    records = _load_ndjson(metadata_path)
    _write_ndjson_records(metadata_path, [records[0]])

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter.apply_structural_filter",
        lambda *_args, **_kwargs: (True, {"filter_mode": "structural_v1"}),
    )

    with pytest.raises(ValueError, match="extra dataset rows beyond metadata coverage"):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)
    assert list(out_dir.iterdir()) == []

    with pytest.raises(ValueError, match="extra dataset rows beyond metadata coverage"):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir)
    assert list(out_dir.iterdir()) == []


def test_run_deferred_filter_rejects_non_monotonic_split_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    shard_dir = tmp_path / "input" / "shard_00000"
    replay_dir = tmp_path / "input" / "internal" / "shard_00000"
    shard_dir.mkdir(parents=True)
    replay_dir.mkdir(parents=True)
    catalog_path = shard_dir / DATASET_CATALOG_FILENAME
    replay_path = replay_dir / REPLAY_CATALOG_FILENAME
    train_path = shard_dir / "train.parquet"
    test_path = shard_dir / "test.parquet"

    catalog_records = [
        {
            "dataset_index": 0,
            "dataset_id": "2" * 32,
            "task": "classification",
            "n_train": 2,
            "n_test": 1,
            "n_features": 2,
            "feature_types": ["num", "num"],
            "n_classes": 2,
            "group_ids": {
                "request_run": "1" * 32,
                "layout_plan": "4" * 32,
            },
        },
        {
            "dataset_index": 1,
            "dataset_id": "3" * 32,
            "task": "classification",
            "n_train": 1,
            "n_test": 1,
            "n_features": 2,
            "feature_types": ["num", "num"],
            "n_classes": 2,
            "group_ids": {
                "request_run": "1" * 32,
                "layout_plan": "4" * 32,
            },
        },
    ]
    replay_records = [
        {
            "dataset_index": 0,
            "n_train": 2,
            "n_test": 1,
            "n_features": 2,
            "feature_types": ["num", "num"],
            "metadata": {
                "seed": 11,
                "filter": {"mode": "deferred", "status": "not_run"},
                "config": {
                    "dataset": {"task": "classification"},
                    "filter": {"enabled": True},
                },
            },
        },
        {
            "dataset_index": 1,
            "n_train": 1,
            "n_test": 1,
            "n_features": 2,
            "feature_types": ["num", "num"],
            "metadata": {
                "seed": 12,
                "filter": {"mode": "deferred", "status": "not_run"},
                "config": {
                    "dataset": {"task": "classification"},
                    "filter": {"enabled": True},
                },
            },
        },
    ]
    _write_ndjson_records(catalog_path, catalog_records)
    _write_ndjson_records(replay_path, replay_records)
    _write_split_table(
        train_path,
        dataset_indices=[0, 1, 0],
        row_indices=[0, 0, 1],
        x_rows=[[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]],
        y_rows=[0, 1, 0],
    )
    _write_split_table(
        test_path,
        dataset_indices=[0, 1],
        row_indices=[0, 0],
        x_rows=[[0.25, 0.25], [1.25, 1.25]],
        y_rows=[0, 1],
    )

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter.apply_structural_filter",
        lambda *_args, **_kwargs: (True, {"filter_mode": "structural_v1"}),
    )

    with pytest.raises(ValueError, match="monotonically increasing dataset_index"):
        _ = run_deferred_filter(in_dir=shard_dir, out_dir=tmp_path / "filter_out")


def test_run_deferred_filter_ignores_public_lineage_tree_when_curating_minimal_outputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    curated_out = tmp_path / "curated_out"
    _ = write_packed_parquet_shards_stream(
        [_bundle_with_embedded_config(501)],
        in_dir,
        shard_size=1,
        compression="zstd",
    )

    lineage_dir = in_dir / "shard_00000" / "lineage"
    lineage_dir.mkdir()
    outside_path = tmp_path / "outside.txt"
    outside_path.write_text("sentinel\n", encoding="utf-8")
    try:
        (lineage_dir / "escape.txt").symlink_to(outside_path)
    except OSError:
        pytest.skip("symlinks unavailable in this environment")

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter.apply_structural_filter",
        lambda *_args, **_kwargs: (True, {"filter_mode": "structural_v1"}),
    )

    result = run_deferred_filter(in_dir=in_dir, out_dir=out_dir, curated_out_dir=curated_out)

    assert result.accepted_datasets == 1
    assert (curated_out / "shard_00000" / DATASET_CATALOG_FILENAME).exists()
    assert not (curated_out / "shard_00000" / "lineage").exists()


def test_run_deferred_filter_cleans_up_curated_output_after_split_exhaustion_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_deferred_filter_impl(monkeypatch)
    pytest.importorskip("pyarrow.parquet")

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "filter_out"
    curated_out = tmp_path / "curated_out"
    _ = write_packed_parquet_shards_stream(
        [_bundle_with_embedded_config(601), _bundle_with_embedded_config(602)],
        in_dir,
        shard_size=2,
        compression="zstd",
    )

    metadata_path = in_dir / "shard_00000" / DATASET_CATALOG_FILENAME
    records = _load_ndjson(metadata_path)
    _write_ndjson_records(metadata_path, [records[0]])

    monkeypatch.setattr(
        "dagzoo.filtering.deferred_filter.apply_structural_filter",
        lambda *_args, **_kwargs: (True, {"filter_mode": "structural_v1"}),
    )

    with pytest.raises(ValueError, match="extra dataset rows beyond metadata coverage"):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir, curated_out_dir=curated_out)
    assert list(out_dir.iterdir()) == []
    assert list(curated_out.iterdir()) == []

    with pytest.raises(ValueError, match="extra dataset rows beyond metadata coverage"):
        _ = run_deferred_filter(in_dir=in_dir, out_dir=out_dir, curated_out_dir=curated_out)
    assert list(out_dir.iterdir()) == []
    assert list(curated_out.iterdir()) == []
