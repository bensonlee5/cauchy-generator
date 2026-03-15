"""Dataset split validation helpers."""

from __future__ import annotations

import math

import torch


class InfeasibleStratifiedSplitError(ValueError):
    """Raised when a classification split cannot satisfy stratification constraints."""


class InvalidClassSplitError(ValueError):
    """Raised when a classification split violates emitted bundle invariants."""


def _stratified_split_indices(
    y: torch.Tensor,
    n_train: int,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (train_indices, test_indices) with proportional class representation.

    For classification tasks this keeps class balance close to proportional and
    ensures classes with at least two members appear in both splits. For
    infeasible combinations, this raises ``InfeasibleStratifiedSplitError``
    with an ``infeasible_stratified_split`` prefix.
    """
    n_total = int(y.shape[0])
    n_test = n_total - n_train
    if n_total <= 0 or n_train <= 0 or n_test <= 0:
        raise InfeasibleStratifiedSplitError(
            f"infeasible_stratified_split: expected 0 < n_train < n_total, got n_train={n_train}, n_total={n_total}."
        )

    classes = torch.unique(y, sorted=True)
    train_frac = n_train / n_total

    cls_indices: list[torch.Tensor] = []
    cls_values: list[int] = []
    cls_train_counts: list[int] = []
    cls_train_min: list[int] = []
    cls_train_max: list[int] = []
    cls_remainders: list[float] = []

    for cls in classes:
        idx = torch.where(y == cls)[0]
        perm = torch.randperm(idx.shape[0], generator=generator, device=device)
        cls_indices.append(idx[perm])
        cls_values.append(int(cls.item()))

        n_cls = int(idx.shape[0])
        proportional = float(n_cls * train_frac)
        base_alloc = int(math.floor(proportional))
        remainder = proportional - base_alloc
        if n_cls >= 2:
            train_min = 1
            train_max = n_cls - 1
        else:
            train_min = 0
            train_max = n_cls

        n_cls_train = max(train_min, min(base_alloc, train_max))
        cls_train_counts.append(n_cls_train)
        cls_train_min.append(train_min)
        cls_train_max.append(train_max)
        cls_remainders.append(remainder)

    deficit = n_train - sum(cls_train_counts)
    if deficit > 0:
        order = sorted(
            range(len(cls_train_counts)), key=lambda i: (-cls_remainders[i], cls_values[i])
        )
        while deficit > 0:
            progressed = False
            for i in order:
                if cls_train_counts[i] < cls_train_max[i]:
                    cls_train_counts[i] += 1
                    deficit -= 1
                    progressed = True
                    if deficit == 0:
                        break
            if not progressed:
                break
        if deficit > 0:
            raise InfeasibleStratifiedSplitError(
                "infeasible_stratified_split: unable to allocate requested train rows while "
                f"preserving class constraints (remaining={deficit})."
            )
    elif deficit < 0:
        surplus = -deficit
        order = sorted(
            range(len(cls_train_counts)), key=lambda i: (cls_remainders[i], cls_values[i])
        )
        while surplus > 0:
            progressed = False
            for i in order:
                if cls_train_counts[i] > cls_train_min[i]:
                    cls_train_counts[i] -= 1
                    surplus -= 1
                    progressed = True
                    if surplus == 0:
                        break
            if not progressed:
                break
        if surplus > 0:
            raise InfeasibleStratifiedSplitError(
                "infeasible_stratified_split: unable to allocate requested test rows while "
                f"preserving class constraints (remaining={surplus})."
            )

    if sum(cls_train_counts) != n_train:
        raise InfeasibleStratifiedSplitError(
            "infeasible_stratified_split: train allocation mismatch after rebalance "
            f"(expected={n_train}, actual={sum(cls_train_counts)})."
        )

    train_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    for idx, n_cls_train in zip(cls_indices, cls_train_counts, strict=True):
        train_parts.append(idx[:n_cls_train])
        test_parts.append(idx[n_cls_train:])

    train_idx = torch.cat(train_parts)
    test_idx = torch.cat(test_parts)
    if int(train_idx.shape[0]) != n_train or int(test_idx.shape[0]) != n_test:
        raise InfeasibleStratifiedSplitError(
            "infeasible_stratified_split: index cardinality mismatch "
            f"(expected_train={n_train}, actual_train={int(train_idx.shape[0])}, "
            f"expected_test={n_test}, actual_test={int(test_idx.shape[0])})."
        )

    # Shuffle within each split
    train_idx = train_idx[torch.randperm(train_idx.shape[0], generator=generator, device=device)]
    test_idx = test_idx[torch.randperm(test_idx.shape[0], generator=generator, device=device)]

    return train_idx, test_idx


def _classification_split_valid(y_train: torch.Tensor, y_test: torch.Tensor) -> bool:
    """Validate classification split constraints."""

    train_classes = torch.unique(y_train.to(torch.int64), sorted=True)
    test_classes = torch.unique(y_test.to(torch.int64), sorted=True)
    return bool(train_classes.numel() >= 2 and torch.equal(train_classes, test_classes))
