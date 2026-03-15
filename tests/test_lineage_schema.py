from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dagzoo.io.lineage_schema import (
    LINEAGE_ADJACENCY_ENCODING,
    LINEAGE_SCHEMA_NAME,
    LINEAGE_SCHEMA_VERSION,
    LINEAGE_SCHEMA_VERSION_COMPACT,
    LineageValidationError,
    validate_lineage_payload,
    validate_metadata_lineage,
)

_N_NODES_STRATEGY = st.integers(min_value=2, max_value=12)
_FEATURE_COUNT_STRATEGY = st.integers(min_value=0, max_value=32)
_PATH_STRATEGY = st.text(
    alphabet=tuple("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/-"),
    min_size=1,
    max_size=64,
)
_HEX_SHA256_STRATEGY = st.text(
    alphabet=tuple("0123456789abcdef"),
    min_size=64,
    max_size=64,
)


def _upper_triangle_capacity(n_nodes: int) -> int:
    return (n_nodes * (n_nodes - 1)) // 2


def _dense_adjacency_from_bits(n_nodes: int, upper_bits: list[int]) -> list[list[int]]:
    adjacency = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
    idx = 0
    for row in range(n_nodes):
        for col in range(row + 1, n_nodes):
            adjacency[row][col] = int(upper_bits[idx])
            idx += 1
    return adjacency


@st.composite
def _dense_lineage_payload_strategy(draw: st.DrawFn) -> dict[str, Any]:
    n_nodes = draw(_N_NODES_STRATEGY)
    adjacency = _dense_adjacency_from_bits(
        n_nodes,
        draw(
            st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=_upper_triangle_capacity(n_nodes),
                max_size=_upper_triangle_capacity(n_nodes),
            )
        ),
    )
    feature_count = draw(_FEATURE_COUNT_STRATEGY)
    feature_to_node = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_nodes - 1),
            min_size=feature_count,
            max_size=feature_count,
        )
    )
    return {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "graph": {
            "n_nodes": n_nodes,
            "adjacency": adjacency,
        },
        "assignments": {
            "feature_to_node": feature_to_node,
            "target_to_node": draw(st.integers(min_value=0, max_value=n_nodes - 1)),
        },
    }


@st.composite
def _compact_lineage_payload_strategy(draw: st.DrawFn) -> dict[str, Any]:
    n_nodes = draw(_N_NODES_STRATEGY)
    capacity = _upper_triangle_capacity(n_nodes)
    feature_count = draw(_FEATURE_COUNT_STRATEGY)
    feature_to_node = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_nodes - 1),
            min_size=feature_count,
            max_size=feature_count,
        )
    )
    return {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION_COMPACT,
        "graph": {
            "n_nodes": n_nodes,
            "edge_count": draw(st.integers(min_value=0, max_value=capacity)),
            "adjacency_ref": {
                "encoding": LINEAGE_ADJACENCY_ENCODING,
                "blob_path": draw(_PATH_STRATEGY),
                "index_path": draw(_PATH_STRATEGY),
                "dataset_index": draw(st.integers(min_value=0, max_value=64)),
                "bit_offset": draw(st.integers(min_value=0, max_value=64)).__mul__(8),
                "bit_length": capacity,
                "sha256": draw(_HEX_SHA256_STRATEGY),
            },
        },
        "assignments": {
            "feature_to_node": feature_to_node,
            "target_to_node": draw(st.integers(min_value=0, max_value=n_nodes - 1)),
        },
    }


def _valid_lineage_payload() -> dict[str, Any]:
    return {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "graph": {
            "n_nodes": 4,
            "adjacency": [
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
        },
        "assignments": {
            "feature_to_node": [0, 1, 1, 2, 3],
            "target_to_node": 2,
        },
    }


def _valid_compact_lineage_payload() -> dict[str, Any]:
    return {
        "schema_name": LINEAGE_SCHEMA_NAME,
        "schema_version": LINEAGE_SCHEMA_VERSION_COMPACT,
        "graph": {
            "n_nodes": 4,
            "edge_count": 4,
            "adjacency_ref": {
                "encoding": LINEAGE_ADJACENCY_ENCODING,
                "blob_path": "../lineage/adjacency.bitpack.bin",
                "index_path": "../lineage/adjacency.index.json",
                "dataset_index": 0,
                "bit_offset": 0,
                "bit_length": 6,
                "sha256": "0" * 64,
            },
        },
        "assignments": {
            "feature_to_node": [0, 1, 1, 2, 3],
            "target_to_node": 2,
        },
    }


def test_validate_lineage_payload_accepts_valid_payload() -> None:
    validate_lineage_payload(_valid_lineage_payload())


def test_validate_lineage_payload_accepts_valid_compact_payload() -> None:
    validate_lineage_payload(_valid_compact_lineage_payload())


def test_validate_metadata_lineage_accepts_absent_payload_when_optional() -> None:
    validate_metadata_lineage({"seed": 1}, required=False)


def test_validate_metadata_lineage_rejects_missing_payload_when_required() -> None:
    with pytest.raises(LineageValidationError, match=r"metadata\.lineage: is required"):
        validate_metadata_lineage({"seed": 1}, required=True)


def test_validate_lineage_payload_rejects_wrong_schema_name() -> None:
    payload = _valid_lineage_payload()
    payload["schema_name"] = "other.schema"
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.schema_name: must equal 'dagzoo\.dag_lineage'",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_unknown_root_keys() -> None:
    payload = _valid_lineage_payload()
    payload["extra"] = True
    with pytest.raises(LineageValidationError, match=r"lineage: unknown key\(s\): extra"):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_non_square_adjacency() -> None:
    payload = _valid_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    graph["adjacency"] = [[0, 1], [0, 0]]
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.graph\.adjacency: must have 4 rows \(got 2\)",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_non_binary_adjacency_values() -> None:
    payload = _valid_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    adjacency = deepcopy(graph["adjacency"])
    assert isinstance(adjacency, list)
    adjacency[0][1] = 2
    graph["adjacency"] = adjacency
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError, match=r"lineage\.graph\.adjacency\[0\]\[1\]: must be 0 or 1"
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_bool_adjacency_values() -> None:
    payload = _valid_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    adjacency = deepcopy(graph["adjacency"])
    assert isinstance(adjacency, list)
    adjacency[0][1] = True
    graph["adjacency"] = adjacency
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.graph\.adjacency\[0\]\[1\]: must be integer 0 or 1",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_diagonal_edges() -> None:
    payload = _valid_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    adjacency = deepcopy(graph["adjacency"])
    assert isinstance(adjacency, list)
    adjacency[2][2] = 1
    graph["adjacency"] = adjacency
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.graph\.adjacency\[2\]\[2\]: must be 0 on the diagonal",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_lower_triangle_edges() -> None:
    payload = _valid_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    adjacency = deepcopy(graph["adjacency"])
    assert isinstance(adjacency, list)
    adjacency[3][1] = 1
    graph["adjacency"] = adjacency
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.graph\.adjacency\[3\]\[1\]: must be 0 for upper-triangular DAG encoding",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_feature_assignment_out_of_range() -> None:
    payload = _valid_lineage_payload()
    assignments = deepcopy(payload["assignments"])
    assert isinstance(assignments, dict)
    assignments["feature_to_node"] = [0, 1, 4]
    payload["assignments"] = assignments
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.assignments\.feature_to_node\[2\]: must be in range \[0, 3\]",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_target_assignment_out_of_range() -> None:
    payload = _valid_lineage_payload()
    assignments = deepcopy(payload["assignments"])
    assert isinstance(assignments, dict)
    assignments["target_to_node"] = -1
    payload["assignments"] = assignments
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.assignments\.target_to_node: must be in range \[0, 3\]",
    ):
        validate_lineage_payload(payload)


def test_validate_metadata_lineage_rejects_non_object_payload() -> None:
    with pytest.raises(LineageValidationError, match=r"metadata\.lineage: must be an object"):
        validate_metadata_lineage({"lineage": 42}, required=False)


def test_validate_lineage_payload_rejects_unknown_compact_encoding() -> None:
    payload = _valid_compact_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    adjacency_ref = deepcopy(graph["adjacency_ref"])
    assert isinstance(adjacency_ref, dict)
    adjacency_ref["encoding"] = "other"
    graph["adjacency_ref"] = adjacency_ref
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.graph\.adjacency_ref\.encoding: must equal 'upper_triangle_bitpack_v1'",
    ):
        validate_lineage_payload(payload)


def test_validate_lineage_payload_rejects_compact_bit_length_mismatch() -> None:
    payload = _valid_compact_lineage_payload()
    graph = deepcopy(payload["graph"])
    assert isinstance(graph, dict)
    adjacency_ref = deepcopy(graph["adjacency_ref"])
    assert isinstance(adjacency_ref, dict)
    adjacency_ref["bit_length"] = 5
    graph["adjacency_ref"] = adjacency_ref
    payload["graph"] = graph
    with pytest.raises(
        LineageValidationError,
        match=r"lineage\.graph\.adjacency_ref\.bit_length: must equal 6 for n_nodes=4",
    ):
        validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_dense_lineage_payload_strategy())
def test_validate_lineage_payload_accepts_generated_dense_payloads_hypothesis(
    payload: dict[str, Any],
) -> None:
    validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_compact_lineage_payload_strategy())
def test_validate_lineage_payload_accepts_generated_compact_payloads_hypothesis(
    payload: dict[str, Any],
) -> None:
    validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_dense_lineage_payload_strategy())
def test_validate_lineage_payload_rejects_generated_diagonal_edge_mutations_hypothesis(
    payload: dict[str, Any],
) -> None:
    payload["graph"]["adjacency"][0][0] = 1
    with pytest.raises(LineageValidationError):
        validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_dense_lineage_payload_strategy())
def test_validate_lineage_payload_rejects_generated_lower_triangle_mutations_hypothesis(
    payload: dict[str, Any],
) -> None:
    payload["graph"]["adjacency"][1][0] = 1
    with pytest.raises(LineageValidationError):
        validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_dense_lineage_payload_strategy())
def test_validate_lineage_payload_rejects_generated_nonbinary_adjacency_mutations_hypothesis(
    payload: dict[str, Any],
) -> None:
    payload["graph"]["adjacency"][0][1] = 2
    with pytest.raises(LineageValidationError):
        validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_dense_lineage_payload_strategy())
def test_validate_lineage_payload_rejects_generated_feature_assignment_mutations_hypothesis(
    payload: dict[str, Any],
) -> None:
    n_nodes = int(payload["graph"]["n_nodes"])
    payload["assignments"]["feature_to_node"] = [n_nodes]
    with pytest.raises(LineageValidationError):
        validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_dense_lineage_payload_strategy())
def test_validate_lineage_payload_rejects_generated_target_assignment_mutations_hypothesis(
    payload: dict[str, Any],
) -> None:
    payload["assignments"]["target_to_node"] = -1
    with pytest.raises(LineageValidationError):
        validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_compact_lineage_payload_strategy())
def test_validate_lineage_payload_rejects_generated_compact_encoding_mutations_hypothesis(
    payload: dict[str, Any],
) -> None:
    payload["graph"]["adjacency_ref"]["encoding"] = "other"
    with pytest.raises(LineageValidationError):
        validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_compact_lineage_payload_strategy())
def test_validate_lineage_payload_rejects_generated_compact_bit_length_mutations_hypothesis(
    payload: dict[str, Any],
) -> None:
    payload["graph"]["adjacency_ref"]["bit_length"] += 1
    with pytest.raises(LineageValidationError):
        validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_compact_lineage_payload_strategy())
def test_validate_lineage_payload_rejects_generated_compact_bit_offset_mutations_hypothesis(
    payload: dict[str, Any],
) -> None:
    payload["graph"]["adjacency_ref"]["bit_offset"] += 4
    with pytest.raises(LineageValidationError):
        validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_compact_lineage_payload_strategy())
def test_validate_lineage_payload_rejects_generated_compact_sha256_mutations_hypothesis(
    payload: dict[str, Any],
) -> None:
    payload["graph"]["adjacency_ref"]["sha256"] = "g" * 64
    with pytest.raises(LineageValidationError):
        validate_lineage_payload(payload)


@settings(max_examples=100, deadline=None)
@given(payload=_compact_lineage_payload_strategy())
def test_validate_lineage_payload_rejects_generated_compact_edge_count_mutations_hypothesis(
    payload: dict[str, Any],
) -> None:
    n_nodes = int(payload["graph"]["n_nodes"])
    payload["graph"]["edge_count"] = _upper_triangle_capacity(n_nodes) + 1
    with pytest.raises(LineageValidationError):
        validate_lineage_payload(payload)
