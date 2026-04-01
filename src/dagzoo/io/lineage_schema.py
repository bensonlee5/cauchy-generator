"""Validation helpers for DAG lineage metadata schema."""

from __future__ import annotations

import math
import string
from collections.abc import Mapping
from typing import Any, NoReturn, TypeGuard, cast

LINEAGE_SCHEMA_NAME = "dagzoo.dag_lineage"
LINEAGE_SCHEMA_VERSION_DENSE = "1.4.0"
LINEAGE_SCHEMA_VERSION_COMPACT = "1.5.0"
LINEAGE_SCHEMA_VERSION = LINEAGE_SCHEMA_VERSION_DENSE
LINEAGE_ADJACENCY_ENCODING = "upper_triangle_bitpack_v1"

_SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {
        LINEAGE_SCHEMA_VERSION_DENSE,
        LINEAGE_SCHEMA_VERSION_COMPACT,
    }
)

_LINEAGE_REQUIRED_KEYS = frozenset({"schema_name", "schema_version", "graph", "assignments"})
_GRAPH_DENSE_REQUIRED_KEYS = frozenset({"n_nodes", "adjacency"})
_GRAPH_COMPACT_REQUIRED_KEYS = frozenset({"n_nodes", "edge_count", "adjacency_ref"})
_ADJACENCY_REF_REQUIRED_KEYS = frozenset(
    {
        "encoding",
        "blob_path",
        "index_path",
        "dataset_index",
        "bit_offset",
        "bit_length",
        "sha256",
    }
)
_ASSIGNMENTS_REQUIRED_KEYS = frozenset(
    {
        "feature_to_node",
        "target_to_node",
        "target_relevant_features",
        "target_relevant_feature_count",
        "target_relevant_feature_fraction",
    }
)
_ASSIGNMENTS_ALLOWED_KEYS = frozenset(
    {
        "feature_to_node",
        "target_to_node",
        "target_relevant_features",
        "target_relevant_feature_count",
        "target_relevant_feature_fraction",
    }
)


class LineageValidationError(ValueError):
    """Raised when a lineage payload fails schema validation."""


def _is_int(value: object) -> TypeGuard[int]:
    """Return True for int values excluding bool."""

    return isinstance(value, int) and not isinstance(value, bool)


def _raise(path: str, message: str) -> NoReturn:
    """Raise a schema validation error with a field path."""

    raise LineageValidationError(f"{path}: {message}")


def _as_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    """Validate and return a mapping payload."""

    if not isinstance(value, Mapping):
        _raise(path, "must be an object")
    return cast(Mapping[str, Any], value)


def _validate_required_and_unknown_keys(
    payload: Mapping[str, Any],
    *,
    path: str,
    required_keys: frozenset[str],
) -> None:
    """Validate required keys are present and reject unknown keys."""

    raw_keys = list(payload.keys())
    non_string_keys = [key for key in raw_keys if not isinstance(key, str)]
    if non_string_keys:
        formatted = ", ".join(repr(key) for key in non_string_keys)
        _raise(path, f"contains non-string key(s): {formatted}")

    keys = {str(key) for key in raw_keys}
    missing = sorted(required_keys - keys)
    if missing:
        _raise(path, f"missing required key(s): {', '.join(missing)}")

    unknown = sorted(keys - required_keys)
    if unknown:
        _raise(path, f"unknown key(s): {', '.join(unknown)}")


def _validate_assignment_node_index(value: object, *, n_nodes: int, path: str) -> None:
    """Validate one assignment node index with bound checks."""

    if not _is_int(value):
        _raise(path, "must be an integer node index")
    if value < 0 or value >= n_nodes:
        _raise(path, f"must be in range [0, {n_nodes - 1}]")


def _validate_n_nodes(value: object, *, path: str) -> int:
    """Validate graph node count and return normalized integer."""

    if not _is_int(value):
        _raise(path, "must be an integer >= 2")
    if value < 2:
        _raise(path, "must be >= 2")
    return value


def _upper_triangle_bit_length(n_nodes: int) -> int:
    """Return the number of bits required for strict upper-triangle DAG encoding."""

    return (n_nodes * (n_nodes - 1)) // 2


def _validate_dense_graph(graph: Mapping[str, Any], *, path: str) -> int:
    """Validate dense adjacency graph payload and return `n_nodes`."""

    _validate_required_and_unknown_keys(
        graph,
        path=path,
        required_keys=_GRAPH_DENSE_REQUIRED_KEYS,
    )

    n_nodes_path = f"{path}.n_nodes"
    n_nodes = _validate_n_nodes(graph["n_nodes"], path=n_nodes_path)

    adjacency = graph["adjacency"]
    adjacency_path = f"{path}.adjacency"
    if not isinstance(adjacency, list):
        _raise(adjacency_path, "must be a list of rows")
    if len(adjacency) != n_nodes:
        _raise(adjacency_path, f"must have {n_nodes} rows (got {len(adjacency)})")

    for row_idx, row in enumerate(adjacency):
        row_path = f"{adjacency_path}[{row_idx}]"
        if not isinstance(row, list):
            _raise(row_path, "must be a list")
        if len(row) != n_nodes:
            _raise(row_path, f"must have {n_nodes} columns (got {len(row)})")
        for col_idx, value in enumerate(row):
            value_path = f"{row_path}[{col_idx}]"
            if not _is_int(value):
                _raise(value_path, "must be integer 0 or 1")
            if value not in (0, 1):
                _raise(value_path, "must be 0 or 1")
            if row_idx == col_idx and value != 0:
                _raise(value_path, "must be 0 on the diagonal")
            if row_idx > col_idx and value != 0:
                _raise(value_path, "must be 0 for upper-triangular DAG encoding")
    return n_nodes


def _validate_compact_graph(graph: Mapping[str, Any], *, path: str) -> int:
    """Validate compact adjacency reference graph payload and return `n_nodes`."""

    _validate_required_and_unknown_keys(
        graph,
        path=path,
        required_keys=_GRAPH_COMPACT_REQUIRED_KEYS,
    )

    n_nodes_path = f"{path}.n_nodes"
    n_nodes = _validate_n_nodes(graph["n_nodes"], path=n_nodes_path)
    max_edges = _upper_triangle_bit_length(n_nodes)

    edge_count = graph["edge_count"]
    edge_count_path = f"{path}.edge_count"
    if not _is_int(edge_count):
        _raise(edge_count_path, "must be an integer in valid edge-count range")
    if edge_count < 0 or edge_count > max_edges:
        _raise(edge_count_path, f"must be in range [0, {max_edges}]")

    adjacency_ref_path = f"{path}.adjacency_ref"
    adjacency_ref = _as_mapping(graph["adjacency_ref"], path=adjacency_ref_path)
    _validate_required_and_unknown_keys(
        adjacency_ref,
        path=adjacency_ref_path,
        required_keys=_ADJACENCY_REF_REQUIRED_KEYS,
    )

    encoding = adjacency_ref["encoding"]
    encoding_path = f"{adjacency_ref_path}.encoding"
    if not isinstance(encoding, str):
        _raise(encoding_path, "must be a string")
    if encoding != LINEAGE_ADJACENCY_ENCODING:
        _raise(
            encoding_path,
            f"must equal '{LINEAGE_ADJACENCY_ENCODING}' (got '{encoding}')",
        )

    for field_name in ("blob_path", "index_path"):
        value = adjacency_ref[field_name]
        value_path = f"{adjacency_ref_path}.{field_name}"
        if not isinstance(value, str):
            _raise(value_path, "must be a string")
        if not value.strip():
            _raise(value_path, "must be a non-empty path")

    dataset_index = adjacency_ref["dataset_index"]
    dataset_index_path = f"{adjacency_ref_path}.dataset_index"
    if not _is_int(dataset_index):
        _raise(dataset_index_path, "must be a non-negative integer")
    if dataset_index < 0:
        _raise(dataset_index_path, "must be >= 0")

    bit_offset = adjacency_ref["bit_offset"]
    bit_offset_path = f"{adjacency_ref_path}.bit_offset"
    if not _is_int(bit_offset):
        _raise(bit_offset_path, "must be a non-negative integer")
    if bit_offset < 0:
        _raise(bit_offset_path, "must be >= 0")
    if bit_offset % 8 != 0:
        _raise(bit_offset_path, "must be a multiple of 8")

    bit_length = adjacency_ref["bit_length"]
    bit_length_path = f"{adjacency_ref_path}.bit_length"
    if not _is_int(bit_length):
        _raise(bit_length_path, "must be an integer")
    if bit_length != max_edges:
        _raise(bit_length_path, f"must equal {max_edges} for n_nodes={n_nodes}")

    sha256 = adjacency_ref["sha256"]
    sha256_path = f"{adjacency_ref_path}.sha256"
    if not isinstance(sha256, str):
        _raise(sha256_path, "must be a string")
    hexdigits = set(string.hexdigits)
    if len(sha256) != 64 or any(ch not in hexdigits for ch in sha256):
        _raise(sha256_path, "must be a 64-character hexadecimal string")
    return n_nodes


def _validate_lineage_payload(payload: object, *, root_path: str) -> None:
    """Validate the DAG lineage payload rooted at a given path."""

    lineage = _as_mapping(payload, path=root_path)
    _validate_required_and_unknown_keys(
        lineage,
        path=root_path,
        required_keys=_LINEAGE_REQUIRED_KEYS,
    )

    schema_name = lineage["schema_name"]
    schema_name_path = f"{root_path}.schema_name"
    if not isinstance(schema_name, str):
        _raise(schema_name_path, "must be a string")
    if schema_name != LINEAGE_SCHEMA_NAME:
        _raise(
            schema_name_path,
            f"must equal '{LINEAGE_SCHEMA_NAME}' (got '{schema_name}')",
        )

    schema_version = lineage["schema_version"]
    schema_version_path = f"{root_path}.schema_version"
    if not isinstance(schema_version, str):
        _raise(schema_version_path, "must be a string")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(_SUPPORTED_SCHEMA_VERSIONS))
        _raise(
            schema_version_path,
            f"unsupported version '{schema_version}' (supported: {supported})",
        )

    graph_path = f"{root_path}.graph"
    graph = _as_mapping(lineage["graph"], path=graph_path)
    if schema_version == LINEAGE_SCHEMA_VERSION_DENSE:
        n_nodes = _validate_dense_graph(graph, path=graph_path)
    else:
        n_nodes = _validate_compact_graph(graph, path=graph_path)

    assignments_path = f"{root_path}.assignments"
    assignments = _as_mapping(lineage["assignments"], path=assignments_path)
    raw_assignment_keys = list(assignments.keys())
    non_string_keys = [key for key in raw_assignment_keys if not isinstance(key, str)]
    if non_string_keys:
        formatted = ", ".join(repr(key) for key in non_string_keys)
        _raise(assignments_path, f"contains non-string key(s): {formatted}")
    assignment_keys = {str(key) for key in raw_assignment_keys}
    missing = sorted(_ASSIGNMENTS_REQUIRED_KEYS - assignment_keys)
    if missing:
        _raise(assignments_path, f"missing required key(s): {', '.join(missing)}")
    unknown = sorted(assignment_keys - _ASSIGNMENTS_ALLOWED_KEYS)
    if unknown:
        _raise(assignments_path, f"unknown key(s): {', '.join(unknown)}")

    feature_to_node = assignments["feature_to_node"]
    feature_to_node_path = f"{assignments_path}.feature_to_node"
    if not isinstance(feature_to_node, list):
        _raise(feature_to_node_path, "must be a list of node indices")
    for idx, node_index in enumerate(feature_to_node):
        _validate_assignment_node_index(
            node_index,
            n_nodes=n_nodes,
            path=f"{feature_to_node_path}[{idx}]",
        )

    _validate_assignment_node_index(
        assignments["target_to_node"],
        n_nodes=n_nodes,
        path=f"{assignments_path}.target_to_node",
    )

    feature_count = len(feature_to_node)
    target_relevant_features = assignments.get("target_relevant_features")
    target_relevant_feature_count = assignments.get("target_relevant_feature_count")
    target_relevant_feature_fraction = assignments.get("target_relevant_feature_fraction")
    target_relevant_features_path = f"{assignments_path}.target_relevant_features"
    if not isinstance(target_relevant_features, list):
        _raise(target_relevant_features_path, "must be a list of feature indices")
    last_index = -1
    for index, feature_index in enumerate(target_relevant_features):
        feature_index_path = f"{target_relevant_features_path}[{index}]"
        if not _is_int(feature_index):
            _raise(feature_index_path, "must be an integer feature index")
        if feature_index < 0 or feature_index >= feature_count:
            _raise(feature_index_path, f"must be in range [0, {feature_count - 1}]")
        if feature_index <= last_index:
            _raise(feature_index_path, "must be strictly increasing and unique")
        last_index = feature_index

    target_relevant_feature_count_path = f"{assignments_path}.target_relevant_feature_count"
    if not _is_int(target_relevant_feature_count):
        _raise(target_relevant_feature_count_path, "must be an integer")
    if target_relevant_feature_count != len(target_relevant_features):
        _raise(
            target_relevant_feature_count_path,
            "must equal len(target_relevant_features)",
        )

    target_relevant_feature_fraction_path = f"{assignments_path}.target_relevant_feature_fraction"
    if isinstance(target_relevant_feature_fraction, bool) or not isinstance(
        target_relevant_feature_fraction,
        (int, float),
    ):
        _raise(target_relevant_feature_fraction_path, "must be a finite ratio in [0, 1]")
    target_relevant_feature_fraction_value = float(target_relevant_feature_fraction)
    if not math.isfinite(target_relevant_feature_fraction_value) or not (
        0.0 <= target_relevant_feature_fraction_value <= 1.0
    ):
        _raise(target_relevant_feature_fraction_path, "must be a finite ratio in [0, 1]")
    expected_fraction = (
        float(len(target_relevant_features)) / float(feature_count) if feature_count > 0 else 0.0
    )
    if not math.isclose(
        target_relevant_feature_fraction_value,
        expected_fraction,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        _raise(
            target_relevant_feature_fraction_path,
            "must equal target_relevant_feature_count / len(feature_to_node)",
        )


def validate_lineage_payload(payload: Mapping[str, Any]) -> None:
    """Validate a lineage payload object rooted at `lineage`."""

    _validate_lineage_payload(payload, root_path="lineage")


def validate_metadata_lineage(metadata: Mapping[str, Any], *, required: bool = False) -> None:
    """Validate `metadata.lineage` when present.

    When `required` is False (default), missing lineage is accepted.
    """

    metadata_payload = _as_mapping(metadata, path="metadata")
    if "lineage" not in metadata_payload:
        if required:
            _raise("metadata.lineage", "is required")
        return
    _validate_lineage_payload(metadata_payload["lineage"], root_path="metadata.lineage")


__all__ = [
    "LINEAGE_ADJACENCY_ENCODING",
    "LINEAGE_SCHEMA_NAME",
    "LINEAGE_SCHEMA_VERSION",
    "LINEAGE_SCHEMA_VERSION_COMPACT",
    "LINEAGE_SCHEMA_VERSION_DENSE",
    "LineageValidationError",
    "validate_lineage_payload",
    "validate_metadata_lineage",
]
