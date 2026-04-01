"""Dataset split validation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from dagzoo.core.fixed_layout.plan_types import (
    CategoricalConverterPlan,
    ConcatNodeSource,
    FixedLayoutFunctionPlan,
    FixedLayoutNodePlan,
    PiecewiseFunctionPlan,
    ProductFunctionPlan,
    RandomPointsNodeSource,
    StackedNodeSource,
    TreeFunctionPlan,
)

_VARIANCE_EPSILON = 1e-12


class InfeasibleStratifiedSplitError(ValueError):
    """Raised when a classification split cannot satisfy stratification constraints."""


class InvalidClassSplitError(ValueError):
    """Raised when a classification split violates emitted bundle invariants."""


class RetryableDegeneracyError(ValueError):
    """Raised when a sampled plan or realized tensor is retryably degenerate."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = str(reason)
        super().__init__(self.reason if message is None else str(message))


class InvalidFeatureMatrixError(RetryableDegeneracyError):
    """Raised when emitted features violate non-classification bundle invariants."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


RecoverableRetryScope = Literal[
    "same_plan_attempt",
    "next_plan_candidate",
]
RECOVERABLE_RETRY_SCOPE_SAME_PLAN_ATTEMPT: RecoverableRetryScope = "same_plan_attempt"
RECOVERABLE_RETRY_SCOPE_NEXT_PLAN_CANDIDATE: RecoverableRetryScope = "next_plan_candidate"


@dataclass(slots=True, frozen=True)
class RecoverableGenerationFailure:
    """Recoverable generation failure classification for runtime retry routing."""

    reason: str
    retry_scope: RecoverableRetryScope


def classify_recoverable_generation_failure(
    exc: Exception,
    *,
    degeneracy_retry_scope: RecoverableRetryScope = RECOVERABLE_RETRY_SCOPE_SAME_PLAN_ATTEMPT,
) -> RecoverableGenerationFailure | None:
    """Classify retryable runtime failures into their retry scope."""

    if isinstance(exc, InvalidClassSplitError):
        return RecoverableGenerationFailure(
            reason="invalid_class_split",
            retry_scope=RECOVERABLE_RETRY_SCOPE_SAME_PLAN_ATTEMPT,
        )
    if isinstance(exc, RetryableDegeneracyError):
        if exc.reason in {
            "all_constant_features",
            "degenerate_tree_plan",
            "width_incompatible_activation",
            "nonfinite_pathway_output",
            "constant_pathway_output",
        }:
            return RecoverableGenerationFailure(
                reason=exc.reason,
                retry_scope=degeneracy_retry_scope,
            )
    return None


def _collapsed_pathway_mask(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.to(torch.float32)
    if values.dim() == 0:
        return torch.ones((), dtype=torch.bool, device=values.device)
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    if values.dim() < 2:
        return torch.ones(values.shape[:-1], dtype=torch.bool, device=values.device)
    if int(values.shape[-2]) <= 1:
        return torch.ones(values.shape[:-2], dtype=torch.bool, device=values.device)
    channel_variance = torch.var(values, dim=-2, correction=0)
    if channel_variance.dim() == 0:
        return channel_variance <= _VARIANCE_EPSILON
    return torch.all(channel_variance <= _VARIANCE_EPSILON, dim=-1)


def _collapsed_matrix_mask(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.to(torch.float32)
    if values.dim() == 0:
        return torch.zeros((), dtype=torch.bool, device=values.device)
    if values.dim() >= 2 and min(int(values.shape[-2]), int(values.shape[-1])) <= 1:
        return torch.zeros(values.shape[:-2], dtype=torch.bool, device=values.device)
    if values.dim() == 1:
        flat = values.unsqueeze(0)
    else:
        flat = values.reshape(*values.shape[:-2], -1)
    if int(flat.shape[-1]) <= 1:
        return torch.zeros(flat.shape[:-1], dtype=torch.bool, device=values.device)
    matrix_variance = torch.var(flat, dim=-1, correction=0)
    return matrix_variance <= _VARIANCE_EPSILON


def validate_matrix_output(
    tensor: torch.Tensor,
    *,
    context: str,
) -> None:
    """Reject non-finite or fully constant realized matrix tensors."""

    values = tensor.to(torch.float32)
    if not bool(torch.all(torch.isfinite(values))):
        raise RetryableDegeneracyError(
            "nonfinite_pathway_output",
            message=f"{context} produced non-finite matrix values.",
        )
    if bool(torch.any(_collapsed_matrix_mask(values))):
        raise RetryableDegeneracyError(
            "constant_pathway_output",
            message=f"{context} collapsed to a constant matrix.",
        )


def validate_pathway_output(
    tensor: torch.Tensor,
    *,
    context: str,
    input_tensor: torch.Tensor | None = None,
) -> None:
    """Reject non-finite or fully collapsed pathway outputs.

    When an ``input_tensor`` is supplied, collapse is only rejected for batch
    items whose corresponding input still contained row-wise variation.
    """

    values = tensor.to(torch.float32)
    if not bool(torch.all(torch.isfinite(values))):
        raise RetryableDegeneracyError(
            "nonfinite_pathway_output",
            message=f"{context} produced non-finite output values.",
        )
    collapsed = _collapsed_pathway_mask(values)
    if input_tensor is not None:
        collapsed = torch.logical_and(
            collapsed,
            torch.logical_not(_collapsed_pathway_mask(input_tensor)),
        )
    if bool(torch.any(collapsed)):
        raise RetryableDegeneracyError(
            "constant_pathway_output",
            message=f"{context} collapsed every emitted channel.",
        )


def validate_function_plan_nondegeneracy(plan: FixedLayoutFunctionPlan) -> None:
    """Reject structurally trivial fixed-layout function plans."""

    if isinstance(plan, TreeFunctionPlan):
        if int(plan.n_trees) == 1 and tuple(int(depth) for depth in plan.depths) == (1,):
            raise RetryableDegeneracyError(
                "degenerate_tree_plan",
                message="TreeFunctionPlan(n_trees=1, depths=(1,)) is structurally degenerate.",
            )
        return
    if isinstance(plan, ProductFunctionPlan):
        validate_function_plan_nondegeneracy(plan.lhs)
        validate_function_plan_nondegeneracy(plan.rhs)
        return
    if isinstance(plan, PiecewiseFunctionPlan):
        validate_function_plan_nondegeneracy(plan.lhs)
        validate_function_plan_nondegeneracy(plan.rhs)


def validate_converter_plan_nondegeneracy(plan: object) -> None:
    """Reject converter plans that embed structurally degenerate functions."""

    if isinstance(plan, CategoricalConverterPlan) and plan.function is not None:
        validate_function_plan_nondegeneracy(plan.function)


def validate_node_plan_nondegeneracy(plan: FixedLayoutNodePlan) -> None:
    """Reject node plans that embed structurally degenerate nested plans."""

    for converter_plan in plan.converter_plans:
        validate_converter_plan_nondegeneracy(converter_plan)
    if isinstance(plan.source, RandomPointsNodeSource | ConcatNodeSource):
        validate_function_plan_nondegeneracy(plan.source.function)
        return
    if isinstance(plan.source, StackedNodeSource):
        for function_plan in plan.source.parent_functions:
            validate_function_plan_nondegeneracy(function_plan)


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
