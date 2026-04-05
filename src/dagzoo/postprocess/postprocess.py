"""Postprocessing hooks."""

from __future__ import annotations

from typing import Any, Literal, overload

import torch

from dagzoo.config import (
    MISSINGNESS_MECHANISM_NONE,
    DatasetConfig,
    normalize_missing_mechanism,
)
from dagzoo.core.validation import InvalidFeatureMatrixError
from dagzoo.rng import KeyedRng
from dagzoo.sampling import sample_missingness_mask


def _standardization_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Return the working dtype for numeric postprocess statistics."""

    if tensor.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _remove_constant_columns(
    x: torch.Tensor, feature_types: list[str]
) -> tuple[torch.Tensor, list[str], list[int]]:
    """Drop columns with near-zero variance and align feature type metadata."""

    keep = torch.std(x, dim=0, correction=0) > 1e-12
    if not torch.any(keep):
        raise InvalidFeatureMatrixError("all_constant_features")
    keep_indices = [int(i) for i, keep_col in enumerate(keep.tolist()) if keep_col]
    kept_types = [feature_types[i] for i in keep_indices]
    return x[:, keep], kept_types, keep_indices


def _fit_numeric_standardization(
    x: torch.Tensor, feature_types: list[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fit numeric clipping and standardization parameters from one feature matrix."""

    numeric_indices = [i for i, t in enumerate(feature_types) if t != "cat"]
    if not numeric_indices:
        return None

    numeric_index = torch.tensor(numeric_indices, device=x.device, dtype=torch.long)
    feature_dim = x.ndim - 1
    row_dim = x.ndim - 2
    numeric = x.index_select(dim=feature_dim, index=numeric_index)
    stats_dtype = _standardization_dtype(numeric)
    numeric_stats = numeric.to(dtype=stats_dtype)
    quantiles = torch.quantile(
        numeric_stats,
        torch.tensor([0.01, 0.99], device=numeric.device, dtype=stats_dtype),
        dim=row_dim,
    )
    lo = quantiles[0].unsqueeze(row_dim)
    hi = quantiles[1].unsqueeze(row_dim)
    numeric_stats = torch.clamp(numeric_stats, lo, hi)
    mu = torch.mean(numeric_stats, dim=row_dim, keepdim=True)
    sd = torch.std(numeric_stats, dim=row_dim, correction=0, keepdim=True).clamp_min(1e-6)
    return numeric_index, lo, hi, mu, sd


def _apply_numeric_standardization(
    x: torch.Tensor,
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    """Apply fitted numeric clipping and standardization parameters to one feature matrix."""

    out = x.clone()
    if params is None:
        return out

    numeric_index, lo, hi, mu, sd = params
    if numeric_index.device != out.device:
        numeric_index = numeric_index.to(device=out.device)
    if lo.device != out.device:
        lo = lo.to(device=out.device)
        hi = hi.to(device=out.device)
        mu = mu.to(device=out.device)
        sd = sd.to(device=out.device)
    feature_dim = out.ndim - 1
    numeric = out.index_select(dim=feature_dim, index=numeric_index)
    numeric = torch.clamp(numeric, lo, hi)
    out.index_copy_(feature_dim, numeric_index, (numeric - mu) / sd)
    return out


def _clip_and_standardize_rows(x: torch.Tensor, feature_types: list[str]) -> torch.Tensor:
    """Clip numeric outliers and standardize numeric columns along the row axis."""

    return _apply_numeric_standardization(x, _fit_numeric_standardization(x, feature_types))


def _fit_target_standardization(
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit clipping and standardization parameters from one target tensor."""

    row_dim = y.ndim - 1
    stats_dtype = _standardization_dtype(y)
    y_stats = y.to(dtype=stats_dtype)
    quantiles = torch.quantile(
        y_stats,
        torch.tensor([0.01, 0.99], device=y.device, dtype=stats_dtype),
        dim=row_dim,
    )
    lo = quantiles[0].unsqueeze(row_dim)
    hi = quantiles[1].unsqueeze(row_dim)
    y_stats = torch.clamp(y_stats, lo, hi)
    mu = torch.mean(y_stats, dim=row_dim, keepdim=True)
    sd = torch.std(y_stats, dim=row_dim, correction=0, keepdim=True).clamp_min(1e-6)
    return lo, hi, mu, sd


def _apply_target_standardization(
    y: torch.Tensor,
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply fitted clipping and standardization parameters to one target tensor."""

    lo, hi, mu, sd = params
    if lo.device != y.device:
        lo = lo.to(device=y.device)
        hi = hi.to(device=y.device)
        mu = mu.to(device=y.device)
        sd = sd.to(device=y.device)
    return (torch.clamp(y, lo, hi) - mu) / sd


def _postprocess_feature_splits(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    feature_types: list[str],
    *,
    keyed_rng: KeyedRng | None,
    preserve_feature_schema: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    """Postprocess feature tensors for both scalar and fixed-schema batched flows."""

    x_train_p = x_train
    x_test_p = x_test
    if preserve_feature_schema:
        feature_types_out = list(feature_types)
        feature_index_map = [int(i) for i in range(int(x_train_p.shape[-1]))]
    else:
        if x_train_p.ndim != 2 or x_test_p.ndim != 2:
            raise ValueError("Constant-column removal is only supported for unbatched features.")
        x_train_p, feature_types_out, feature_index_map = _remove_constant_columns(
            x_train_p, feature_types
        )
        keep_index = torch.tensor(feature_index_map, device=x_test_p.device, dtype=torch.long)
        x_test_p = x_test_p.index_select(dim=x_test_p.ndim - 1, index=keep_index)

    params = _fit_numeric_standardization(x_train_p, feature_types_out)
    x_train_p = _apply_numeric_standardization(x_train_p, params)
    x_test_p = _apply_numeric_standardization(x_test_p, params)

    if not preserve_feature_schema:
        if keyed_rng is None:
            raise ValueError("keyed_rng is required when preserve_feature_schema is False.")
        perm_cpu = torch.randperm(
            x_train_p.shape[-1],
            generator=keyed_rng.keyed("feature_permutation").torch_rng(device="cpu"),
            device="cpu",
        )
        perm_list = [int(i) for i in perm_cpu.tolist()]
        x_train_p = x_train_p.index_select(
            dim=x_train_p.ndim - 1, index=perm_cpu.to(x_train_p.device)
        )
        x_test_p = x_test_p.index_select(dim=x_test_p.ndim - 1, index=perm_cpu.to(x_test_p.device))
        feature_types_out = [feature_types_out[i] for i in perm_list]
        feature_index_map = [feature_index_map[i] for i in perm_list]
    return x_train_p, x_test_p, feature_types_out, feature_index_map


def postprocess_feature_matrix(
    x: torch.Tensor,
    feature_types: list[str],
    *,
    keyed_rng: KeyedRng | None,
    preserve_feature_schema: bool,
) -> tuple[torch.Tensor, list[str], list[int]]:
    """Postprocess one full complete-data feature matrix before target generation."""

    if preserve_feature_schema:
        feature_types_out = list(feature_types)
        feature_index_map = [int(i) for i in range(int(x.shape[-1]))]
    else:
        if x.ndim != 2:
            raise ValueError("Constant-column removal is only supported for unbatched features.")
        x, feature_types_out, feature_index_map = _remove_constant_columns(x, feature_types)

    x = _clip_and_standardize_rows(x, feature_types_out)

    if not preserve_feature_schema:
        if keyed_rng is None:
            raise ValueError("keyed_rng is required when preserve_feature_schema is False.")
        perm_cpu = torch.randperm(
            x.shape[-1],
            generator=keyed_rng.keyed("feature_permutation").torch_rng(device="cpu"),
            device="cpu",
        )
        perm_list = [int(i) for i in perm_cpu.tolist()]
        perm = perm_cpu.to(device=x.device)
        x = x.index_select(dim=x.ndim - 1, index=perm)
        feature_types_out = [feature_types_out[i] for i in perm_list]
        feature_index_map = [feature_index_map[i] for i in perm_list]
    return x, feature_types_out, feature_index_map


def _postprocess_regression_targets(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip and standardize regression targets for scalar or batched inputs."""

    y_train_p = y_train
    y_test_p = y_test
    params = _fit_target_standardization(y_train_p)
    return _apply_target_standardization(y_train_p, params), _apply_target_standardization(
        y_test_p, params
    )


def _has_at_least_two_classes(y: torch.Tensor) -> bool:
    """Return whether a non-empty label tensor contains at least two classes."""

    y_i64 = y.to(torch.int64)
    return bool(torch.min(y_i64) != torch.max(y_i64))


def _sample_label_permutation(
    *,
    num_classes: int,
    keyed_rng: KeyedRng,
    device: str,
) -> torch.Tensor:
    """Sample one deterministic class-label permutation with device-local preference."""

    preferred_device = str(device).strip().lower()
    if preferred_device == "cuda":
        try:
            return torch.randperm(
                num_classes,
                generator=keyed_rng.keyed("label_permutation").torch_rng(device="cuda"),
                device="cuda",
            )
        except Exception:
            pass
    return torch.randperm(
        num_classes,
        generator=keyed_rng.keyed("label_permutation").torch_rng(device="cpu"),
        device="cpu",
    )


def _postprocess_classification_labels_with_permutation_device(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    keyed_rng: KeyedRng,
    *,
    permutation_device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap classification labels into dense space with deterministic permutation."""

    n_train = int(y_train.shape[0])
    y_all_original = torch.cat([y_train, y_test], dim=0).to(torch.int64)
    classes, inverse = torch.unique(y_all_original, sorted=True, return_inverse=True)
    y_all_original_dense = inverse.to(torch.int64)

    perm_dense = _sample_label_permutation(
        num_classes=classes.numel(),
        keyed_rng=keyed_rng,
        device=permutation_device,
    )
    if perm_dense.device != inverse.device:
        perm_dense = perm_dense.to(device=inverse.device)
    y_all_permuted_dense = perm_dense[inverse].to(torch.int64)
    y_train_candidate = y_all_permuted_dense[:n_train]
    y_test_candidate = y_all_permuted_dense[n_train:]

    if not _has_at_least_two_classes(y_train_candidate) or not _has_at_least_two_classes(
        y_test_candidate
    ):
        y_all_dense = y_all_original_dense
    else:
        y_all_dense = y_all_permuted_dense
    return y_all_dense[:n_train], y_all_dense[n_train:]


def _postprocess_classification_labels(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    keyed_rng: KeyedRng,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap scalar classification labels with the historical CPU permutation path."""

    return _postprocess_classification_labels_with_permutation_device(
        y_train,
        y_test,
        keyed_rng,
        permutation_device="cpu",
    )


def _postprocess_classification_label_batch(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    *,
    postprocess_roots: list[KeyedRng],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap one fixed-schema classification batch while preferring device-local permutations."""

    permutation_device = "cuda" if str(y_train.device.type) == "cuda" else "cpu"
    y_train_batches: list[torch.Tensor] = []
    y_test_batches: list[torch.Tensor] = []
    for batch_index, keyed_rng in enumerate(postprocess_roots):
        y_train_p, y_test_p = _postprocess_classification_labels_with_permutation_device(
            y_train[batch_index],
            y_test[batch_index],
            keyed_rng,
            permutation_device=permutation_device,
        )
        y_train_batches.append(y_train_p)
        y_test_batches.append(y_test_p)
    return torch.stack(y_train_batches), torch.stack(y_test_batches)


def postprocess_targets(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    task: str,
    *,
    keyed_rng: KeyedRng,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Postprocess targets after complete-data target generation and splitting."""

    if task == "regression":
        return _postprocess_regression_targets(y_train, y_test)
    return _postprocess_classification_labels(y_train, y_test, keyed_rng)


def postprocess_fixed_schema_target_batch(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    task: str,
    *,
    postprocess_roots: list[KeyedRng],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Postprocess one fixed-schema target batch after splitting."""

    if task == "regression":
        return _postprocess_regression_targets(y_train, y_test)
    return _postprocess_classification_label_batch(
        y_train,
        y_test,
        postprocess_roots=postprocess_roots,
    )


@overload
def postprocess_dataset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feature_types: list[str],
    task: str,
    keyed_rng: KeyedRng,
    device: str,
    *,
    return_feature_index_map: Literal[False] = False,
    preserve_feature_schema: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]: ...


@overload
def postprocess_dataset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feature_types: list[str],
    task: str,
    keyed_rng: KeyedRng,
    device: str,
    *,
    return_feature_index_map: Literal[True],
    preserve_feature_schema: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]: ...


def postprocess_dataset(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feature_types: list[str],
    task: str,
    keyed_rng: KeyedRng,
    device: str,
    *,
    return_feature_index_map: bool = False,
    preserve_feature_schema: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]
):
    """
    Apply postprocessing to train/test splits.

    - Remove train-constant columns
    - Standardize non-categorical columns using train-fit statistics
    - Permute column order
    - Standardize regression targets using train-fit statistics
    - Permute class labels for classification
    """
    _ = device

    x_train_p, x_test_p, feature_types, feature_index_map = _postprocess_feature_splits(
        x_train,
        x_test,
        feature_types,
        keyed_rng=keyed_rng,
        preserve_feature_schema=preserve_feature_schema,
    )

    y_train_p, y_test_p = postprocess_targets(
        y_train,
        y_test,
        task,
        keyed_rng=keyed_rng,
    )

    if return_feature_index_map:
        return x_train_p, y_train_p, x_test_p, y_test_p, feature_types, feature_index_map
    return x_train_p, y_train_p, x_test_p, y_test_p, feature_types


def postprocess_fixed_schema_batch(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    feature_types: list[str],
    task: str,
    *,
    postprocess_roots: list[KeyedRng],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Postprocess one batch of fixed-schema train/test splits."""

    if x_train.ndim != 3 or x_test.ndim != 3:
        raise ValueError("Expected batched feature tensors with shape [batch, rows, features].")
    if y_train.ndim != 2 or y_test.ndim != 2:
        raise ValueError("Expected batched target tensors with shape [batch, rows].")
    if int(x_train.shape[0]) != len(postprocess_roots):
        raise ValueError("postprocess_roots must align with the leading batch dimension.")

    x_train_p, x_test_p, _feature_types, _feature_index_map = _postprocess_feature_splits(
        x_train,
        x_test,
        feature_types,
        keyed_rng=None,
        preserve_feature_schema=True,
    )

    y_train_p, y_test_p = postprocess_fixed_schema_target_batch(
        y_train,
        y_test,
        task,
        postprocess_roots=postprocess_roots,
    )
    return x_train_p, y_train_p, x_test_p, y_test_p


def inject_missingness(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    dataset_cfg: DatasetConfig,
    keyed_rng: KeyedRng,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    """
    Inject one full-matrix missingness process into train/test feature tensors.

    Missing values are encoded as NaN and summary stats are returned for metadata.
    """

    missing_rate = float(dataset_cfg.missing_rate)
    mechanism = normalize_missing_mechanism(dataset_cfg.missing_mechanism)
    enabled = missing_rate > 0.0 and mechanism != MISSINGNESS_MECHANISM_NONE
    if not enabled:
        return x_train, x_test, None

    row_dim = x_train.ndim - 2
    x_all = torch.cat([x_train, x_test], dim=row_dim)
    full_mask = sample_missingness_mask(
        x_all,
        dataset_cfg=dataset_cfg,
        keyed_rng=keyed_rng.keyed("full_matrix"),
        device=device,
    )
    x_all_missing = x_all.masked_fill(full_mask, float("nan"))
    n_train = int(x_train.shape[row_dim])
    n_test = int(x_test.shape[row_dim])
    x_train_missing = x_all_missing.narrow(row_dim, 0, n_train)
    x_test_missing = x_all_missing.narrow(row_dim, n_train, n_test)
    train_mask = full_mask.narrow(row_dim, 0, n_train)
    test_mask = full_mask.narrow(row_dim, n_train, n_test)

    missing_count_train = int(train_mask.sum().item())
    missing_count_test = int(test_mask.sum().item())
    train_total = max(1, int(train_mask.numel()))
    test_total = max(1, int(test_mask.numel()))
    total = train_total + test_total
    missing_count_overall = missing_count_train + missing_count_test

    summary: dict[str, Any] = {
        "enabled": True,
        "mechanism": mechanism,
        "target_rate": float(missing_rate),
        "realized_rate_train": float(missing_count_train / train_total),
        "realized_rate_test": float(missing_count_test / test_total),
        "realized_rate_overall": float(missing_count_overall / total),
        "missing_count_train": missing_count_train,
        "missing_count_test": missing_count_test,
        "missing_count_overall": missing_count_overall,
    }
    return x_train_missing, x_test_missing, summary
