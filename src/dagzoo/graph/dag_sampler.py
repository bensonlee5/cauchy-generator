"""Random DAG sampler."""

from __future__ import annotations

import math

import torch


def sample_dag(
    n_nodes: int,
    generator: torch.Generator,
    *,
    edge_logit_bias: float = 0.0,
) -> torch.Tensor:
    """
    Sample a DAG adjacency matrix using latent variable edge sampling on the CPU.

    For i < j:
      p_ij = sigmoid(A + B_i + C_j + edge_logit_bias)
    where A, B_i, C_j are latent variables drawn from a standard Cauchy distribution.
    """

    if n_nodes < 2:
        return torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device="cpu")

    # Latent variables from standard Cauchy: tan(pi * (U - 0.5)) where U ~ Uniform(0, 1)
    u_a = torch.empty(1, device="cpu").uniform_(0, 1, generator=generator)
    a = torch.tan(math.pi * (u_a - 0.5)).item()

    u_b = torch.empty(n_nodes, device="cpu").uniform_(0, 1, generator=generator)
    b = torch.tan(math.pi * (u_b - 0.5))

    u_c = torch.empty(n_nodes, device="cpu").uniform_(0, 1, generator=generator)
    c = torch.tan(math.pi * (u_c - 0.5))

    logits = a + b[:, None] + c[None, :] + float(edge_logit_bias)
    probs = torch.sigmoid(logits)

    upper_mask = torch.triu(
        torch.ones(n_nodes, n_nodes, dtype=torch.bool, device="cpu"), diagonal=1
    )
    edges = torch.empty(n_nodes, n_nodes, device="cpu").uniform_(0, 1, generator=generator) < probs
    adjacency = edges & upper_mask
    return adjacency


def dag_longest_path_nodes(adjacency: torch.Tensor) -> int:
    """Return DAG longest path length measured in number of nodes."""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got shape={tuple(adjacency.shape)!r}")
    adj_bool = adjacency.to(dtype=torch.bool)
    if bool(torch.tril(adj_bool, diagonal=0).any().item()):
        raise ValueError("adjacency must be strict upper-triangular for DAG depth computation.")
    n_nodes = int(adjacency.shape[0])
    if n_nodes == 0:
        return 0
    if n_nodes == 1:
        return 1

    longest_from = [1] * n_nodes
    for src in range(n_nodes - 1, -1, -1):
        children = torch.where(adj_bool[src])[0]
        if int(children.numel()) == 0:
            longest_from[src] = 1
        else:
            longest_from[src] = 1 + max(longest_from[int(child.item())] for child in children)
    return max(longest_from)


def dag_longest_path_to_target_nodes(adjacency: torch.Tensor, target_node: int) -> int:
    """Return the longest root-to-target path length measured in number of nodes."""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got shape={tuple(adjacency.shape)!r}")
    adj_bool = adjacency.to(dtype=torch.bool)
    if bool(torch.tril(adj_bool, diagonal=0).any().item()):
        raise ValueError("adjacency must be strict upper-triangular for DAG depth computation.")
    n_nodes = int(adjacency.shape[0])
    if target_node < 0 or target_node >= n_nodes:
        raise ValueError(f"target_node must be in range [0, {n_nodes - 1}], got {target_node}.")
    if n_nodes == 0:
        return 0

    longest_to = [1] * n_nodes
    for dst in range(n_nodes):
        parents = torch.where(adj_bool[:, dst])[0]
        if int(parents.numel()) == 0:
            longest_to[dst] = 1
        else:
            longest_to[dst] = 1 + max(longest_to[int(parent.item())] for parent in parents)
    return int(longest_to[int(target_node)])


def dag_edge_density(adjacency: torch.Tensor) -> float:
    """Return realized DAG edge density over strict upper-triangular capacity."""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got shape={tuple(adjacency.shape)!r}")
    adj_bool = adjacency.to(dtype=torch.bool)
    if bool(torch.tril(adj_bool, diagonal=0).any().item()):
        raise ValueError("adjacency must be strict upper-triangular for density computation.")
    n_nodes = int(adjacency.shape[0])
    if n_nodes < 2:
        return 0.0

    capacity = n_nodes * (n_nodes - 1) // 2
    edges = int(adj_bool.sum().item())
    return float(edges / capacity)
