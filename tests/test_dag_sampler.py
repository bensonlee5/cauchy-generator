"""Tests for graph/dag_sampler.py."""

import pytest
import torch
from conftest import make_generator as _make_generator
from hypothesis import given, settings
from hypothesis import strategies as st

from dagzoo.graph.dag_sampler import (
    dag_edge_density,
    dag_longest_path_nodes,
    sample_dag,
)
from dagzoo.rng import SEED32_MAX

_SEED32_STRATEGY = st.integers(min_value=0, max_value=SEED32_MAX)
_EDGE_LOGIT_BIAS_STRATEGY = st.integers(min_value=-6, max_value=6).map(lambda value: value / 2.0)


@st.composite
def _strict_upper_triangular_adjacency_strategy(draw: st.DrawFn) -> torch.Tensor:
    n_nodes = draw(st.integers(min_value=0, max_value=8))
    capacity = n_nodes * (n_nodes - 1) // 2
    edges = draw(st.lists(st.booleans(), min_size=capacity, max_size=capacity))

    adjacency = torch.zeros((n_nodes, n_nodes), dtype=torch.bool)
    offset = 0
    for src in range(n_nodes):
        for dst in range(src + 1, n_nodes):
            adjacency[src, dst] = edges[offset]
            offset += 1
    return adjacency


def _oracle_longest_path_nodes(adjacency: torch.Tensor) -> int:
    rows = adjacency.to(dtype=torch.bool).tolist()
    n_nodes = len(rows)
    if n_nodes == 0:
        return 0

    memo: dict[int, int] = {}

    def _visit(node: int) -> int:
        if node in memo:
            return memo[node]
        children = [dst for dst in range(node + 1, n_nodes) if rows[node][dst]]
        memo[node] = 1 + max((_visit(child) for child in children), default=0)
        return memo[node]

    return max(_visit(node) for node in range(n_nodes))


def test_dag_shape() -> None:
    g = _make_generator(42)
    adj = sample_dag(5, g)
    assert adj.shape == (5, 5)
    assert adj.dtype == torch.bool


def test_dag_upper_triangular() -> None:
    g = _make_generator(7)
    adj = sample_dag(8, g)
    for i in range(8):
        for j in range(i + 1):
            assert adj[i, j] == False  # noqa: E712


def test_dag_deterministic() -> None:
    a = sample_dag(6, _make_generator(99))
    b = sample_dag(6, _make_generator(99))
    torch.testing.assert_close(a, b)


def test_dag_deterministic_with_edge_logit_bias() -> None:
    a = sample_dag(6, _make_generator(99), edge_logit_bias=0.75)
    b = sample_dag(6, _make_generator(99), edge_logit_bias=0.75)
    torch.testing.assert_close(a, b)


def test_dag_single_node() -> None:
    g = _make_generator(0)
    adj = sample_dag(1, g)
    assert adj.shape == (1, 1)
    assert not adj[0, 0]


def test_edge_logit_bias_increases_edge_count_with_same_rng_stream() -> None:
    low_bias = sample_dag(16, _make_generator(123), edge_logit_bias=-1.0)
    high_bias = sample_dag(16, _make_generator(123), edge_logit_bias=1.0)
    assert int(high_bias.sum().item()) >= int(low_bias.sum().item())


def test_dag_longest_path_nodes_on_known_graph() -> None:
    adjacency = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    assert dag_longest_path_nodes(adjacency) == 4


def test_dag_longest_path_nodes_rejects_non_upper_triangular_input() -> None:
    adjacency = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.bool,
    )
    with pytest.raises(ValueError, match="upper-triangular"):
        dag_longest_path_nodes(adjacency)


def test_dag_longest_path_nodes_rejects_singleton_self_loop() -> None:
    adjacency = torch.tensor([[1]], dtype=torch.bool)
    with pytest.raises(ValueError, match="upper-triangular"):
        dag_longest_path_nodes(adjacency)


def test_dag_edge_density_on_known_graph() -> None:
    adjacency = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    assert dag_edge_density(adjacency) == 0.5


def test_dag_edge_density_single_node_is_zero() -> None:
    adjacency = torch.zeros((1, 1), dtype=torch.bool)
    assert dag_edge_density(adjacency) == 0.0


def test_dag_edge_density_rejects_singleton_self_loop() -> None:
    adjacency = torch.tensor([[1]], dtype=torch.bool)
    with pytest.raises(ValueError, match="upper-triangular"):
        dag_edge_density(adjacency)


@settings(max_examples=100, deadline=None)
@given(adjacency=_strict_upper_triangular_adjacency_strategy())
def test_dag_edge_density_matches_capacity_ratio_hypothesis(adjacency: torch.Tensor) -> None:
    n_nodes = int(adjacency.shape[0])
    capacity = n_nodes * (n_nodes - 1) // 2
    expected = 0.0 if capacity == 0 else int(adjacency.sum().item()) / capacity
    observed = dag_edge_density(adjacency)

    assert 0.0 <= observed <= 1.0
    assert observed == pytest.approx(expected)


@settings(max_examples=100, deadline=None)
@given(adjacency=_strict_upper_triangular_adjacency_strategy())
def test_dag_longest_path_nodes_matches_oracle_hypothesis(adjacency: torch.Tensor) -> None:
    assert dag_longest_path_nodes(adjacency) == _oracle_longest_path_nodes(adjacency)


@settings(max_examples=100, deadline=None)
@given(
    n_nodes=st.integers(min_value=0, max_value=12),
    seed=_SEED32_STRATEGY,
    edge_logit_bias=_EDGE_LOGIT_BIAS_STRATEGY,
)
def test_sample_dag_returns_strict_upper_triangular_hypothesis(
    n_nodes: int,
    seed: int,
    edge_logit_bias: float,
) -> None:
    adjacency = sample_dag(n_nodes, _make_generator(seed), edge_logit_bias=edge_logit_bias)

    assert adjacency.shape == (n_nodes, n_nodes)
    assert adjacency.dtype == torch.bool
    assert not bool(torch.tril(adjacency, diagonal=0).any().item())


@settings(max_examples=100, deadline=None)
@given(
    n_nodes=st.integers(min_value=0, max_value=12),
    seed=_SEED32_STRATEGY,
    edge_logit_bias=_EDGE_LOGIT_BIAS_STRATEGY,
)
def test_sample_dag_is_deterministic_hypothesis(
    n_nodes: int,
    seed: int,
    edge_logit_bias: float,
) -> None:
    first = sample_dag(n_nodes, _make_generator(seed), edge_logit_bias=edge_logit_bias)
    second = sample_dag(n_nodes, _make_generator(seed), edge_logit_bias=edge_logit_bias)

    torch.testing.assert_close(first, second)


@settings(max_examples=100, deadline=None)
@given(
    n_nodes=st.integers(min_value=0, max_value=12),
    seed=_SEED32_STRATEGY,
    bias_pair=st.tuples(_EDGE_LOGIT_BIAS_STRATEGY, _EDGE_LOGIT_BIAS_STRATEGY).map(
        lambda pair: tuple(sorted(pair))
    ),
)
def test_sample_dag_higher_bias_does_not_reduce_edges_hypothesis(
    n_nodes: int,
    seed: int,
    bias_pair: tuple[float, float],
) -> None:
    low_bias, high_bias = bias_pair
    lower = sample_dag(n_nodes, _make_generator(seed), edge_logit_bias=low_bias)
    higher = sample_dag(n_nodes, _make_generator(seed), edge_logit_bias=high_bias)

    assert int(higher.sum().item()) >= int(lower.sum().item())
