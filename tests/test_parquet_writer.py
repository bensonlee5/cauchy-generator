import io
import json
from pathlib import Path

import numpy as np
import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.core.dataset import generate_batch, generate_one
from dagzoo.io import resolve_lineage_path
from dagzoo.io.lineage_artifact import unpack_upper_triangle_adjacency
from dagzoo.io.lineage_schema import (
    LINEAGE_ADJACENCY_ENCODING,
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION,
    LINEAGE_SCHEMA_VERSION_COMPACT,
    validate_lineage_payload,
)
from dagzoo.io.parquet_writer import _sanitize_json, write_packed_parquet_shards_stream
from dagzoo.io.shard_contract import DATASET_CATALOG_FILENAME, REPLAY_CATALOG_FILENAME
from dagzoo.types import DatasetBundle


def test_sanitize_json_replaces_non_finite_floats() -> None:
    payload = {
        "ok": 1.0,
        "inf": float("inf"),
        "neg_inf": float("-inf"),
        "nan": float("nan"),
        "nested": {"vals": [0.0, float("inf")]},
    }
    sanitized = _sanitize_json(payload)
    assert sanitized["ok"] == 1.0
    assert sanitized["inf"] is None
    assert sanitized["neg_inf"] is None
    assert sanitized["nan"] is None
    assert sanitized["nested"]["vals"][1] is None
    json.dumps(sanitized, allow_nan=False)


def _bundle(seed: int) -> DatasetBundle:
    x_train = np.full((2, 2), float(seed), dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.int64)
    x_test = np.full((1, 2), float(seed), dtype=np.float32)
    y_test = np.array([1], dtype=np.int64)
    return DatasetBundle(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        feature_types=["num", "num"],
        metadata={"seed": seed, "peak_flops": float("inf")},
    )


def _bundle_with_dense_lineage(seed: int) -> DatasetBundle:
    bundle = _bundle(seed)
    bundle.metadata["lineage"] = {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "graph": {
            "n_nodes": 2,
            "adjacency": [[0, 1], [0, 0]],
        },
        "assignments": {
            "feature_to_node": [0, 1],
            "target_to_node": 1,
            "target_relevant_features": [0, 1],
            "target_relevant_feature_count": 2,
            "target_relevant_feature_fraction": 1.0,
        },
    }
    return bundle


def _generate_one_with_retries(
    config: GeneratorConfig,
    *,
    seed: int,
    device: str,
    max_attempts: int = 12,
) -> DatasetBundle:
    """Generate one bundle with bounded retry on stochastic invalid splits."""

    last_exc: Exception | None = None
    for offset in range(max_attempts):
        try:
            return generate_one(config, seed=seed + offset, device=device)
        except ValueError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise AssertionError("unreachable")


def _stub_write_packed_split(*, state, split, dataset_index, x, y, compression) -> None:
    _ = x
    _ = y
    _ = compression
    split_path = state.train_path if split == "train" else state.test_path
    with split_path.open("a", encoding="utf-8") as f:
        f.write(f"{dataset_index}\n")


def _load_ndjson_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        records.append(json.loads(line))
    return records


def test_io_exports_resolve_lineage_path(tmp_path) -> None:
    resolved = resolve_lineage_path(tmp_path, "lineage/adjacency.bitpack.bin")
    assert resolved == tmp_path / "lineage/adjacency.bitpack.bin"


def test_write_packed_parquet_shards_stream_writes_iterable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "dagzoo.io.parquet_writer._write_packed_split",
        _stub_write_packed_split,
    )
    bundles = (_bundle(i) for i in range(3))
    written = write_packed_parquet_shards_stream(
        bundles,
        tmp_path,
        shard_size=2,
        compression="zstd",
    )

    assert written == 3
    assert (tmp_path / "shard_00000" / "train.parquet").exists()
    assert (tmp_path / "shard_00000" / "test.parquet").exists()
    assert (tmp_path / "shard_00001" / DATASET_CATALOG_FILENAME).exists()
    public_records = _load_ndjson_records(tmp_path / "shard_00001" / DATASET_CATALOG_FILENAME)
    replay_records = _load_ndjson_records(
        tmp_path / "internal" / "shard_00001" / REPLAY_CATALOG_FILENAME
    )
    assert public_records[0]["dataset_index"] == 2
    assert "metadata" not in public_records[0]
    assert replay_records[0]["metadata"]["peak_flops"] is None


def test_write_packed_parquet_shards_stream_writes_real_parquet_tables(tmp_path) -> None:
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    written = write_packed_parquet_shards_stream(
        [_bundle(1), _bundle(2)],
        tmp_path,
        shard_size=8,
        compression="zstd",
    )
    assert written == 2

    train_table = pyarrow_parquet.read_table(tmp_path / "shard_00000" / "train.parquet")
    test_table = pyarrow_parquet.read_table(tmp_path / "shard_00000" / "test.parquet")
    assert train_table.num_rows == 4
    assert test_table.num_rows == 2
    assert train_table.column("dataset_index").to_pylist() == [0, 0, 1, 1]
    train_x = train_table.column("x").to_pylist()
    assert len(train_x[0]) == 2
    assert train_x[0] == [1.0, 1.0]


def test_write_packed_parquet_shards_stream_preserves_canonical_replay_metadata(tmp_path) -> None:
    pytest.importorskip("pyarrow.parquet")

    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6

    bundles = generate_batch(cfg, num_datasets=2, seed=4321, device="cpu")
    written = write_packed_parquet_shards_stream(
        bundles, tmp_path, shard_size=2, compression="zstd"
    )

    assert written == 2
    records = _load_ndjson_records(tmp_path / "internal" / "shard_00000" / REPLAY_CATALOG_FILENAME)
    assert [int(record["dataset_index"]) for record in records] == [0, 1]
    metadata = [record["metadata"] for record in records]
    assert [int(payload["seed"]) for payload in metadata] == [4321, 4321]
    assert [int(payload["dataset_index"]) for payload in metadata] == [0, 1]
    assert [int(payload["run_num_datasets"]) for payload in metadata] == [2, 2]
    dataset_seeds = [int(payload["dataset_seed"]) for payload in metadata]
    dataset_ids = [str(payload["dataset_id"]) for payload in metadata]
    request_run_groups = [str(payload["split_groups"]["request_run"]) for payload in metadata]
    layout_plan_groups = [str(payload["split_groups"]["layout_plan"]) for payload in metadata]
    assert len(set(dataset_seeds)) == 2
    assert len(set(dataset_ids)) == 2
    assert len(set(request_run_groups)) == 1
    assert len(set(layout_plan_groups)) == 1


def test_write_packed_parquet_shards_stream_preserves_float_targets(tmp_path) -> None:
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    bundle = DatasetBundle(
        X_train=np.full((2, 2), 2.0, dtype=np.float32),
        y_train=np.array([0.5, 1.5], dtype=np.float32),
        X_test=np.full((1, 2), 2.0, dtype=np.float32),
        y_test=np.array([2.5], dtype=np.float32),
        feature_types=["num", "num"],
        metadata={"seed": 7},
    )

    written = write_packed_parquet_shards_stream(
        [bundle], tmp_path, shard_size=8, compression="zstd"
    )
    assert written == 1

    train_table = pyarrow_parquet.read_table(tmp_path / "shard_00000" / "train.parquet")
    test_table = pyarrow_parquet.read_table(tmp_path / "shard_00000" / "test.parquet")

    assert str(train_table.schema.field("y").type) == "float"
    assert train_table.column("y").to_pylist() == pytest.approx([0.5, 1.5])
    assert test_table.column("y").to_pylist() == pytest.approx([2.5])


def test_write_packed_parquet_shards_stream_rejects_incompatible_split_schema(tmp_path) -> None:
    pytest.importorskip("pyarrow.parquet")

    int_bundle = _bundle(1)
    float_bundle = DatasetBundle(
        X_train=np.full((2, 2), 2.0, dtype=np.float32),
        y_train=np.array([0.5, 1.5], dtype=np.float32),
        X_test=np.full((1, 2), 2.0, dtype=np.float32),
        y_test=np.array([1.5], dtype=np.float32),
        feature_types=["num", "num"],
        metadata={"seed": 2},
    )

    with pytest.raises(ValueError, match="Incompatible packed train schema"):
        write_packed_parquet_shards_stream(
            [int_bundle, float_bundle],
            tmp_path,
            shard_size=8,
            compression="zstd",
        )


def test_write_packed_parquet_shards_stream_rejects_stale_output(tmp_path) -> None:
    stale_dir = tmp_path / "shard_00000"
    stale_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="already contains shard data"):
        write_packed_parquet_shards_stream([_bundle(1)], tmp_path, shard_size=1)


def test_write_packed_parquet_shards_stream_writes_lineage_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "dagzoo.io.parquet_writer._write_packed_split",
        _stub_write_packed_split,
    )
    bundle = _bundle_with_dense_lineage(7)

    written = write_packed_parquet_shards_stream(
        [bundle], tmp_path, shard_size=1, compression="zstd"
    )
    assert written == 1

    records = _load_ndjson_records(tmp_path / "internal" / "shard_00000" / REPLAY_CATALOG_FILENAME)
    metadata = records[0]["metadata"]
    lineage = metadata["lineage"]
    assert lineage["schema_name"] == LINEAGE_SCHEMA_NAME
    assert lineage["schema_version"] == LINEAGE_SCHEMA_VERSION_COMPACT
    graph = lineage["graph"]
    assert graph["n_nodes"] == 2
    assert graph["edge_count"] == 1
    adjacency_ref = graph["adjacency_ref"]
    assert adjacency_ref["encoding"] == LINEAGE_ADJACENCY_ENCODING
    assert adjacency_ref["dataset_index"] == 0
    assert adjacency_ref["bit_offset"] == 0
    assert adjacency_ref["bit_length"] == 1
    assert isinstance(adjacency_ref["sha256"], str)
    assert len(adjacency_ref["sha256"]) == 64

    internal_shard_dir = tmp_path / "internal" / "shard_00000"
    blob_path = resolve_lineage_path(internal_shard_dir, adjacency_ref["blob_path"])
    index_path = resolve_lineage_path(internal_shard_dir, adjacency_ref["index_path"])
    assert blob_path.exists()
    assert index_path.exists()

    byte_offset = int(adjacency_ref["bit_offset"]) // 8
    byte_length = (int(adjacency_ref["bit_length"]) + 7) // 8
    with blob_path.open("rb") as f:
        f.seek(byte_offset)
        payload = f.read(byte_length)
    dense = unpack_upper_triangle_adjacency(
        payload,
        n_nodes=int(graph["n_nodes"]),
        bit_length=int(adjacency_ref["bit_length"]),
    )
    assert dense.tolist() == [[0, 1], [0, 0]]

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["schema_name"] == LINEAGE_SCHEMA_NAME
    assert index_payload["schema_version"] == LINEAGE_SCHEMA_VERSION_COMPACT
    assert index_payload["encoding"] == LINEAGE_ADJACENCY_ENCODING
    assert isinstance(index_payload["records"], list)
    assert index_payload["records"][0]["dataset_index"] == 0


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_generate_and_persist_compact_lineage_for_task(
    task: str,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = task
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 6

    bundle = _generate_one_with_retries(cfg, seed=919, device="cpu")

    monkeypatch.setattr(
        "dagzoo.io.parquet_writer._write_packed_split",
        _stub_write_packed_split,
    )
    written = write_packed_parquet_shards_stream(
        [bundle], tmp_path, shard_size=1, compression="zstd"
    )
    assert written == 1

    records = _load_ndjson_records(tmp_path / "internal" / "shard_00000" / REPLAY_CATALOG_FILENAME)
    metadata = records[0]["metadata"]
    lineage = metadata["lineage"]
    assert lineage["schema_version"] == LINEAGE_SCHEMA_VERSION_COMPACT
    validate_lineage_payload(lineage)

    graph = lineage["graph"]
    adjacency_ref = graph["adjacency_ref"]
    internal_shard_dir = tmp_path / "internal" / "shard_00000"
    blob_path = resolve_lineage_path(internal_shard_dir, adjacency_ref["blob_path"])
    index_path = resolve_lineage_path(internal_shard_dir, adjacency_ref["index_path"])

    n_nodes = int(graph["n_nodes"])
    with blob_path.open("rb") as f:
        f.seek(int(adjacency_ref["bit_offset"]) // 8)
        payload = f.read((int(adjacency_ref["bit_length"]) + 7) // 8)
    decoded = unpack_upper_triangle_adjacency(
        payload,
        n_nodes=n_nodes,
        bit_length=int(adjacency_ref["bit_length"]),
    )
    assert decoded.shape == (n_nodes, n_nodes)

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["records"][0]["dataset_index"] == 0


def test_write_packed_parquet_shards_stream_opens_lineage_blob_once_per_shard(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "dagzoo.io.parquet_writer._write_packed_split",
        _stub_write_packed_split,
    )

    original_open = Path.open
    counters = {"open": 0, "close": 0}

    class _CountingBlob(io.BytesIO):
        def close(self) -> None:
            counters["close"] += 1
            super().close()

    open_blobs: dict[str, _CountingBlob] = {}

    def _patched_open(self: Path, mode: str = "r", *args, **kwargs):
        if self.name == "adjacency.bitpack.bin" and mode == "ab":
            counters["open"] += 1
            key = str(self)
            blob = open_blobs.get(key)
            if blob is None or blob.closed:
                blob = _CountingBlob()
                open_blobs[key] = blob
            blob.seek(0, io.SEEK_END)
            return blob
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _patched_open)

    bundles = [_bundle_with_dense_lineage(1), _bundle_with_dense_lineage(2)]
    written = write_packed_parquet_shards_stream(
        bundles, tmp_path, shard_size=8, compression="zstd"
    )

    assert written == 2
    assert counters["open"] == 1
    assert counters["close"] == 1


def test_write_packed_parquet_shards_stream_limits_open_lineage_blob_descriptors(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "dagzoo.io.parquet_writer._write_packed_split",
        _stub_write_packed_split,
    )

    original_open = Path.open
    counters = {"max_open": 0}
    open_count = {"value": 0}

    class _CountingBlob(io.BytesIO):
        def __init__(self) -> None:
            super().__init__()
            open_count["value"] += 1
            counters["max_open"] = max(counters["max_open"], open_count["value"])

        def close(self) -> None:
            if not self.closed:
                open_count["value"] -= 1
            super().close()

    open_blobs: dict[str, _CountingBlob] = {}

    def _patched_open(self: Path, mode: str = "r", *args, **kwargs):
        if self.name == "adjacency.bitpack.bin" and mode == "ab":
            key = str(self)
            blob = open_blobs.get(key)
            if blob is None or blob.closed:
                blob = _CountingBlob()
                open_blobs[key] = blob
            blob.seek(0, io.SEEK_END)
            return blob
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _patched_open)

    bundles = [_bundle_with_dense_lineage(i) for i in range(12)]
    written = write_packed_parquet_shards_stream(
        bundles, tmp_path, shard_size=1, compression="zstd"
    )

    assert written == 12
    assert counters["max_open"] <= 1
    assert open_count["value"] == 0


def test_write_packed_parquet_shards_stream_closes_lineage_blob_on_failure(
    tmp_path, monkeypatch
) -> None:
    split_calls = {"count": 0}

    def _failing_write_packed_split(*, state, split, dataset_index, x, y, compression):
        _ = state
        _ = split
        _ = dataset_index
        _ = x
        _ = y
        _ = compression
        split_calls["count"] += 1
        if split_calls["count"] >= 3:
            raise RuntimeError("forced split failure")

    monkeypatch.setattr(
        "dagzoo.io.parquet_writer._write_packed_split",
        _failing_write_packed_split,
    )

    original_open = Path.open
    counters = {"open": 0, "close": 0}

    class _CountingBlob(io.BytesIO):
        def close(self) -> None:
            counters["close"] += 1
            super().close()

    open_blobs: dict[str, _CountingBlob] = {}

    def _patched_open(self: Path, mode: str = "r", *args, **kwargs):
        if self.name == "adjacency.bitpack.bin" and mode == "ab":
            counters["open"] += 1
            key = str(self)
            blob = open_blobs.get(key)
            if blob is None or blob.closed:
                blob = _CountingBlob()
                open_blobs[key] = blob
            blob.seek(0, io.SEEK_END)
            return blob
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _patched_open)

    bundles = [_bundle_with_dense_lineage(1), _bundle_with_dense_lineage(2)]
    with pytest.raises(RuntimeError, match="forced split failure"):
        write_packed_parquet_shards_stream(bundles, tmp_path, shard_size=8, compression="zstd")

    assert counters["open"] == 1
    assert counters["close"] == 1


def test_write_packed_parquet_shards_stream_writes_lineage_index_on_failure(
    tmp_path, monkeypatch
) -> None:
    split_calls = {"count": 0}

    def _failing_write_packed_split(*, state, split, dataset_index, x, y, compression):
        _ = state
        _ = split
        _ = dataset_index
        _ = x
        _ = y
        _ = compression
        split_calls["count"] += 1
        if split_calls["count"] >= 3:
            raise RuntimeError("forced split failure")

    monkeypatch.setattr(
        "dagzoo.io.parquet_writer._write_packed_split",
        _failing_write_packed_split,
    )

    bundles = [_bundle_with_dense_lineage(1), _bundle_with_dense_lineage(2)]
    with pytest.raises(RuntimeError, match="forced split failure"):
        write_packed_parquet_shards_stream(bundles, tmp_path, shard_size=8, compression="zstd")

    internal_shard_dir = tmp_path / "internal" / "shard_00000"
    records = _load_ndjson_records(internal_shard_dir / REPLAY_CATALOG_FILENAME)
    adjacency_ref = records[0]["metadata"]["lineage"]["graph"]["adjacency_ref"]
    index_path = resolve_lineage_path(internal_shard_dir, adjacency_ref["index_path"])
    assert index_path.exists()

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = index_payload["records"]
    assert any(int(record["dataset_index"]) == 0 for record in records)
