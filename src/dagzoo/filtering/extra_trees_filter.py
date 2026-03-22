"""CPU ExtraTrees filtering for learnability checks."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from dagzoo.filter_thresholds import (
    ZERO_FILTER_THRESHOLD_BACKEND,
    ZERO_FILTER_THRESHOLD_POLICY,
    is_filter_threshold_bypass,
    validate_filter_threshold,
)
from dagzoo.rng import validate_seed32

_CLASS_AWARE_THRESHOLD_POLICY = "class_aware_piecewise_v1"
_CLASS_AWARE_THRESHOLD_FLOOR = 0.80


def _resolve_max_features(max_features: str | int | float, n_features: int, task: str) -> int:
    if isinstance(max_features, str):
        key = max_features.lower()
        if key == "auto":
            value = (
                int(math.sqrt(n_features)) if task == "classification" else max(1, n_features // 3)
            )
        elif key == "sqrt":
            value = int(math.sqrt(n_features))
        elif key == "log2":
            value = int(math.log2(max(2, n_features)))
        elif key in {"all", "none"}:
            value = n_features
        else:
            raise ValueError(
                f"Unsupported max_features='{max_features}'. "
                "Expected one of: auto, sqrt, log2, all, none."
            )
    elif isinstance(max_features, int):
        value = int(max_features)
    elif isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError(f"Float max_features must be in (0, 1], got {max_features}")
        value = int(round(max_features * n_features))
    else:
        raise TypeError(
            f"max_features must be str, int, or float, got {type(max_features).__name__}"
        )
    return max(1, min(n_features, value))


def _resolve_threshold_diagnostics(
    *,
    task: str,
    requested_threshold: float,
    class_count: int | None,
) -> dict[str, Any]:
    """Return class-aware threshold diagnostics and effective threshold."""

    if task != "classification" or class_count is None:
        effective_threshold = float(requested_threshold)
        return {
            "threshold_requested": float(requested_threshold),
            "threshold_effective": effective_threshold,
            "threshold_policy": _CLASS_AWARE_THRESHOLD_POLICY,
            "class_count": None,
            "class_bucket": "not_applicable",
            "threshold_delta": float(requested_threshold) - effective_threshold,
        }

    if class_count <= 8:
        class_bucket = "<=8"
        threshold_delta = 0.00
    elif class_count <= 16:
        class_bucket = "9-16"
        threshold_delta = 0.05
    elif class_count <= 24:
        class_bucket = "17-24"
        threshold_delta = 0.10
    else:
        class_bucket = "25-32"
        threshold_delta = 0.15

    relaxed_threshold = max(
        _CLASS_AWARE_THRESHOLD_FLOOR, float(requested_threshold) - threshold_delta
    )
    effective_threshold = min(float(requested_threshold), relaxed_threshold)
    return {
        "threshold_requested": float(requested_threshold),
        "threshold_effective": float(effective_threshold),
        "threshold_policy": _CLASS_AWARE_THRESHOLD_POLICY,
        "class_count": int(class_count),
        "class_bucket": class_bucket,
        "threshold_delta": float(requested_threshold) - float(effective_threshold),
    }


def _bootstrap_wins_ratio(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> float:
    """Compute bootstrap wins ratio P[MSE(pred) < MSE(baseline)]."""

    n_valid = int(target.shape[0])
    wins = 0
    chunk_size = 16
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    for start in range(0, n_bootstrap, chunk_size):
        bs = min(chunk_size, n_bootstrap - start)
        idx = torch.randint(0, n_valid, (bs, n_valid), generator=gen, device="cpu").numpy()
        sampled_target = target[idx]
        sampled_pred = pred[idx]
        mse_pred = ((sampled_pred - sampled_target) ** 2).mean(axis=(1, 2))
        mse_base = ((baseline - sampled_target) ** 2).mean(axis=(1, 2))
        wins += int((mse_pred < mse_base).sum())
    return float(wins / float(n_bootstrap))


def _oob_valid_mask_from_samples(estimators_samples: Any, *, n_rows: int) -> np.ndarray:
    """Return rows that received at least one OOB vote across fitted trees."""

    if not isinstance(estimators_samples, list) or len(estimators_samples) == 0:
        return np.zeros(n_rows, dtype=bool)

    oob_vote_counts = np.zeros(n_rows, dtype=np.int32)
    for sampled in estimators_samples:
        sampled_idx = np.asarray(sampled, dtype=np.int64).reshape(-1)
        inbag = np.zeros(n_rows, dtype=bool)
        if sampled_idx.size > 0:
            inbag[sampled_idx] = True
        oob_vote_counts += (~inbag).astype(np.int32)
    return oob_vote_counts > 0


def _apply_extra_trees_filter_numpy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    seed: int,
    n_estimators: int = 25,
    max_depth: int = 6,
    min_samples_leaf: int = 1,
    max_leaf_nodes: int | None = None,
    max_features: str | int | float = "auto",
    n_bootstrap: int = 200,
    threshold: float = 0.95,
    n_jobs: int = -1,
) -> tuple[bool, dict[str, Any]]:
    """Apply CPU ExtraTrees filtering from NumPy arrays."""

    seed = validate_seed32(seed, field_name="seed")
    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    if min_samples_leaf < 1:
        raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
    if max_leaf_nodes is not None and max_leaf_nodes < 1:
        raise ValueError(f"max_leaf_nodes must be >= 1 when set, got {max_leaf_nodes}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        raise ValueError(f"n_jobs must be -1 or an integer >= 1, got {n_jobs!r}")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be -1 or an integer >= 1, got {n_jobs}")

    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"x must be rank-2 [n_rows, n_features], got shape {tuple(x_np.shape)}")
    x_np = np.ascontiguousarray(x_np)
    n_rows = int(x_np.shape[0])
    n_features = int(x_np.shape[1])
    m_try = _resolve_max_features(max_features, n_features, task)

    if task == "classification":
        y_raw = np.asarray(y, dtype=np.int64).reshape(-1)
        unique_labels, y_dense = np.unique(y_raw, return_inverse=True)
        n_classes = int(unique_labels.size)
        class_count: int | None = n_classes
        y_target = np.eye(n_classes, dtype=np.float32)[y_dense]
    elif task == "regression":
        class_count = None
        y_target = np.asarray(y, dtype=np.float32)
        if y_target.ndim == 1:
            y_target = y_target.reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported task '{task}'.")

    y_np = np.ascontiguousarray(y_target, dtype=np.float32)
    if int(y_np.shape[0]) != n_rows:
        raise ValueError(
            "x/y row-count mismatch in filter: "
            f"x has {n_rows} rows, y has {int(y_np.shape[0])} rows."
        )

    requested_threshold = validate_filter_threshold(
        threshold,
        field_name="threshold",
    )
    if is_filter_threshold_bypass(requested_threshold):
        return True, {
            "backend": ZERO_FILTER_THRESHOLD_BACKEND,
            "bypass": True,
            "threshold_requested": float(requested_threshold),
            "threshold_effective": float(requested_threshold),
            "threshold_policy": ZERO_FILTER_THRESHOLD_POLICY,
            "threshold_delta": 0.0,
            "n_jobs": int(n_jobs),
        }

    threshold_details = _resolve_threshold_diagnostics(
        task=task,
        requested_threshold=float(requested_threshold),
        class_count=class_count,
    )
    effective_threshold = float(threshold_details["threshold_effective"])
    max_leaf_nodes_model = int(max_leaf_nodes) if max_leaf_nodes is not None else None
    min_samples_split_override: int | None = None
    if max_leaf_nodes_model == 1:
        # sklearn requires max_leaf_nodes >= 2; preserve old compat behavior by
        # forcing a no-split tree when user requests 1.
        max_leaf_nodes_model = None
        min_samples_split_override = max(2, n_rows + 1)

    model_kwargs: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "bootstrap": True,
        "max_depth": int(max_depth) if max_depth > 0 else None,
        "min_samples_leaf": int(min_samples_leaf),
        "max_leaf_nodes": max_leaf_nodes_model,
        "max_features": int(m_try),
        "oob_score": True,
        "random_state": seed,
        "n_jobs": int(n_jobs),
    }
    if max_depth == 0:
        # Preserve prior semantics where depth=0 forbids splits.
        min_samples_split_override = max(int(min_samples_split_override or 2), n_rows + 1)
    if min_samples_split_override is not None:
        model_kwargs["min_samples_split"] = int(min_samples_split_override)

    model = ExtraTreesRegressor(**model_kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Some inputs do not have OOB scores",
        )
        if task == "regression" and y_np.shape[1] == 1:
            model.fit(x_np, y_np[:, 0])
        else:
            model.fit(x_np, y_np)

    oob_pred = np.asarray(model.oob_prediction_, dtype=np.float32)
    if oob_pred.ndim == 1:
        oob_pred = oob_pred.reshape(-1, 1)
    valid_oob = _oob_valid_mask_from_samples(
        getattr(model, "estimators_samples_", None),
        n_rows=n_rows,
    )
    valid_oob &= np.isfinite(oob_pred).all(axis=1)
    n_valid = int(valid_oob.sum())
    if n_valid < max(32, int(0.25 * n_rows)):
        return False, {
            "reason": "insufficient_oob_predictions",
            "n_valid_oob": n_valid,
            "backend": "extra_trees_cpu",
            "n_jobs": int(n_jobs),
            **threshold_details,
        }

    pred = oob_pred[valid_oob]
    target = y_np[valid_oob]
    baseline = np.mean(target, axis=0, keepdims=True)
    wins_ratio = _bootstrap_wins_ratio(
        pred=pred,
        target=target,
        baseline=baseline,
        seed=seed,
        n_bootstrap=int(n_bootstrap),
    )
    return bool(wins_ratio >= effective_threshold), {
        "wins_ratio": float(wins_ratio),
        "n_valid_oob": n_valid,
        "backend": "extra_trees_cpu",
        "n_jobs": int(n_jobs),
        **threshold_details,
    }


def apply_extra_trees_filter(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    task: str,
    seed: int,
    n_estimators: int = 25,
    max_depth: int = 6,
    min_samples_leaf: int = 1,
    max_leaf_nodes: int | None = None,
    max_features: str | int | float = "auto",
    n_bootstrap: int = 200,
    threshold: float = 0.95,
    n_jobs: int = -1,
) -> tuple[bool, dict[str, Any]]:
    """Apply CPU ExtraTrees filtering with OOB bootstrap win-ratio scoring."""

    x_np = np.asarray(x.detach().to(device="cpu", dtype=torch.float32).numpy(), dtype=np.float32)
    if task == "classification":
        y_np = np.asarray(y.detach().to(device="cpu", dtype=torch.int64).view(-1).numpy())
    else:
        y_np = np.asarray(y.detach().to(device="cpu", dtype=torch.float32).numpy())
    return _apply_extra_trees_filter_numpy(
        x_np,
        y_np,
        task=task,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        max_features=max_features,
        n_bootstrap=n_bootstrap,
        threshold=threshold,
        n_jobs=n_jobs,
    )
