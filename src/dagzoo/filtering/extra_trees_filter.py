"""CPU ExtraTrees filtering for small-shot ease checks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

from dagzoo.io.lineage_artifact import (
    resolve_lineage_path,
    sha256_hex,
    unpack_upper_triangle_adjacency,
)
from dagzoo.io.lineage_schema import validate_lineage_payload
from dagzoo.rng import validate_seed32

_BACKEND = "extra_trees_cpu"
_FILTER_MODE = "small_shot_ease_v1"
_SKILL_EPS = 1e-12
_LINEAGE_BLOB_PATH_ERROR = (
    "metadata.lineage.graph.adjacency_ref.blob_path must be a relative path "
    "that resolves inside the shard lineage directory."
)
_LINEAGE_BLOB_SHA256_ERROR = (
    "metadata.lineage.graph.adjacency_ref.sha256 must match the resolved adjacency blob slice."
)


def _validate_unit_interval(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite value in [0.0, 1.0].")
    as_float = float(value)
    if not (0.0 <= as_float <= 1.0):
        raise ValueError(f"{field_name} must be a finite value in [0.0, 1.0].")
    return as_float


def _validate_filter_n_jobs(n_jobs: object) -> int:
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        raise ValueError(f"n_jobs must be -1 or an integer >= 1, got {n_jobs!r}")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be -1 or an integer >= 1, got {n_jobs}")
    return int(n_jobs)


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


def _prepare_targets(
    *,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    if task == "classification":
        y_train_raw = np.asarray(y_train, dtype=np.int64).reshape(-1)
        y_test_raw = np.asarray(y_test, dtype=np.int64).reshape(-1)
        labels = np.unique(np.concatenate([y_train_raw, y_test_raw], axis=0))
        dense_train = np.searchsorted(labels, y_train_raw)
        dense_test = np.searchsorted(labels, y_test_raw)
        eye = np.eye(int(labels.size), dtype=np.float32)
        return eye[dense_train], eye[dense_test]
    if task == "regression":
        y_train_target = np.asarray(y_train, dtype=np.float32)
        y_test_target = np.asarray(y_test, dtype=np.float32)
        if y_train_target.ndim == 1:
            y_train_target = y_train_target.reshape(-1, 1)
        if y_test_target.ndim == 1:
            y_test_target = y_test_target.reshape(-1, 1)
        return y_train_target.astype(np.float32, copy=False), y_test_target.astype(
            np.float32, copy=False
        )
    raise ValueError(f"Unsupported task '{task}'.")


def _clip_skill(
    loss_probe: np.ndarray | float, loss_const: np.ndarray | float
) -> np.ndarray | float:
    skill = 1.0 - (np.asarray(loss_probe) / np.maximum(np.asarray(loss_const), _SKILL_EPS))
    return np.clip(skill, -1.0, 1.0)


def _skill_from_predictions(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    baseline_mean: np.ndarray,
) -> float:
    loss_probe = float(np.mean((pred - target) ** 2))
    loss_const = float(np.mean((baseline_mean - target) ** 2))
    return float(_clip_skill(loss_probe, loss_const))


def _shared_baseline_skill_gain(
    *,
    pred_small: np.ndarray,
    pred_full: np.ndarray,
    target: np.ndarray,
    baseline_mean: np.ndarray,
) -> float:
    skill_small = _skill_from_predictions(
        pred=pred_small,
        target=target,
        baseline_mean=baseline_mean,
    )
    skill_full = _skill_from_predictions(
        pred=pred_full,
        target=target,
        baseline_mean=baseline_mean,
    )
    return float(skill_full - skill_small)


def _bootstrap_skill_summary(
    *,
    pred_small: np.ndarray,
    pred_full: np.ndarray,
    target: np.ndarray,
    baseline_small_mean: np.ndarray,
    baseline_full_mean: np.ndarray,
    baseline_gain_mean: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> dict[str, float]:
    n_rows = int(target.shape[0])
    if n_rows <= 0:
        raise ValueError("Held-out skill bootstrap requires at least one test row.")

    rng = np.random.default_rng(int(seed))
    chunk_size = 16
    small_samples = np.empty(n_bootstrap, dtype=np.float32)
    full_samples = np.empty(n_bootstrap, dtype=np.float32)
    gain_samples = np.empty(n_bootstrap, dtype=np.float32)

    for start in range(0, n_bootstrap, chunk_size):
        bs = min(chunk_size, n_bootstrap - start)
        sample_idx = rng.integers(0, n_rows, size=(bs, n_rows), endpoint=False)
        sampled_target = target[sample_idx]
        sampled_small = pred_small[sample_idx]
        sampled_full = pred_full[sample_idx]
        loss_small = np.mean((sampled_small - sampled_target) ** 2, axis=(1, 2))
        loss_full = np.mean((sampled_full - sampled_target) ** 2, axis=(1, 2))
        loss_const_small = np.mean((baseline_small_mean - sampled_target) ** 2, axis=(1, 2))
        loss_const_full = np.mean((baseline_full_mean - sampled_target) ** 2, axis=(1, 2))
        loss_const_gain = np.mean((baseline_gain_mean - sampled_target) ** 2, axis=(1, 2))
        chunk_small = np.asarray(_clip_skill(loss_small, loss_const_small), dtype=np.float32)
        chunk_full = np.asarray(_clip_skill(loss_full, loss_const_full), dtype=np.float32)
        small_samples[start : start + bs] = chunk_small
        full_samples[start : start + bs] = chunk_full
        gain_samples[start : start + bs] = np.asarray(
            _clip_skill(loss_full, loss_const_gain) - _clip_skill(loss_small, loss_const_gain),
            dtype=np.float32,
        )

    return {
        "skill_small_lb95": float(np.quantile(small_samples, 0.05)),
        "skill_gain_ub95": float(np.quantile(gain_samples, 0.95)),
        "skill_full_ub95": float(np.quantile(full_samples, 0.95)),
    }


def _fit_extra_trees_predictions(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task: str,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    max_leaf_nodes: int | None,
    max_features: str | int | float,
    n_jobs: int,
) -> np.ndarray:
    n_rows = int(x_train.shape[0])
    n_features = int(x_train.shape[1])
    m_try = _resolve_max_features(max_features, n_features, task)
    max_leaf_nodes_model = int(max_leaf_nodes) if max_leaf_nodes is not None else None
    min_samples_split_override: int | None = None
    if max_leaf_nodes_model == 1:
        max_leaf_nodes_model = None
        min_samples_split_override = max(2, n_rows + 1)

    model_kwargs: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "bootstrap": True,
        "max_depth": int(max_depth) if max_depth > 0 else None,
        "min_samples_leaf": int(min_samples_leaf),
        "max_leaf_nodes": max_leaf_nodes_model,
        "max_features": int(m_try),
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
    }
    if max_depth == 0:
        min_samples_split_override = max(int(min_samples_split_override or 2), n_rows + 1)
    if min_samples_split_override is not None:
        model_kwargs["min_samples_split"] = int(min_samples_split_override)

    model = ExtraTreesRegressor(**model_kwargs)
    if y_train.shape[1] == 1:
        model.fit(x_train, y_train[:, 0])
    else:
        model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_test), dtype=np.float32)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return pred


def _fit_best_stump_skill(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    baseline_small_mean: np.ndarray,
    seed: int,
) -> float:
    best_skill = float("-inf")
    for feature_index in range(int(x_train.shape[1])):
        model = DecisionTreeRegressor(max_depth=1, random_state=int(seed))
        model.fit(x_train[:, [feature_index]], y_train[:, 0] if y_train.shape[1] == 1 else y_train)
        pred = np.asarray(model.predict(x_test[:, [feature_index]]), dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        skill = _skill_from_predictions(pred=pred, target=y_test, baseline_mean=baseline_small_mean)
        if skill > best_skill:
            best_skill = float(skill)
    return float(best_skill)


def _resolve_safe_lineage_blob_path(*, lineage_base_dir: Path, blob_path_hint: str) -> Path:
    hinted = Path(blob_path_hint)
    if hinted.is_absolute():
        raise ValueError(_LINEAGE_BLOB_PATH_ERROR)

    lineage_root = (lineage_base_dir / "lineage").resolve()
    resolved = resolve_lineage_path(lineage_base_dir, blob_path_hint).resolve()
    try:
        resolved.relative_to(lineage_root)
    except ValueError as exc:
        raise ValueError(_LINEAGE_BLOB_PATH_ERROR) from exc
    return resolved


def _resolve_lineage_adjacency(
    *,
    lineage_payload: Mapping[str, Any],
    lineage_base_dir: Path | None,
) -> np.ndarray:
    validate_lineage_payload(dict(lineage_payload))
    graph = lineage_payload["graph"]
    assert isinstance(graph, Mapping)
    n_nodes = int(graph["n_nodes"])

    dense_adjacency = graph.get("adjacency")
    if isinstance(dense_adjacency, list):
        adjacency = np.asarray(dense_adjacency, dtype=np.uint8)
        return adjacency

    adjacency_ref = graph.get("adjacency_ref")
    if not isinstance(adjacency_ref, Mapping):
        raise ValueError("metadata.lineage.graph must contain adjacency or adjacency_ref.")
    if lineage_base_dir is None:
        raise ValueError(
            "Compact lineage payload requires lineage_base_dir to resolve adjacency artifacts."
        )
    blob_path = _resolve_safe_lineage_blob_path(
        lineage_base_dir=lineage_base_dir,
        blob_path_hint=str(adjacency_ref["blob_path"]),
    )
    bit_offset = int(adjacency_ref["bit_offset"])
    bit_length = int(adjacency_ref["bit_length"])
    byte_offset = bit_offset // 8
    byte_length = (bit_length + 7) // 8
    with blob_path.open("rb") as handle:
        handle.seek(byte_offset)
        payload = handle.read(byte_length)
    if len(payload) == byte_length and sha256_hex(payload) != str(adjacency_ref["sha256"]):
        raise ValueError(_LINEAGE_BLOB_SHA256_ERROR)
    return unpack_upper_triangle_adjacency(payload, n_nodes=n_nodes, bit_length=bit_length)


def _lineage_has_feature_target_path(
    *,
    lineage_payload: Mapping[str, Any] | None,
    lineage_base_dir: Path | None,
) -> tuple[bool, bool | None]:
    if not isinstance(lineage_payload, Mapping):
        return False, None

    assignments = lineage_payload.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("metadata.lineage.assignments must be an object.")

    feature_nodes = [int(value) for value in list(assignments.get("feature_to_node", []))]
    target_node = int(assignments["target_to_node"])
    if target_node in feature_nodes:
        return True, True

    adjacency = _resolve_lineage_adjacency(
        lineage_payload=lineage_payload,
        lineage_base_dir=lineage_base_dir,
    )
    visited = set(feature_nodes)
    frontier = list(feature_nodes)
    while frontier:
        node = int(frontier.pop())
        children = np.flatnonzero(adjacency[node] != 0)
        for child_raw in children:
            child = int(child_raw)
            if child == target_node:
                return True, True
            if child not in visited:
                visited.add(child)
                frontier.append(child)
    return True, False


def _validate_filter_params(
    *,
    seed: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    max_leaf_nodes: int | None,
    n_bootstrap: int,
    ease_k_small: int,
    easy_skill_threshold: float,
    easy_gain_threshold: float,
    hard_skill_threshold: float,
    stump_skill_threshold: float | None,
    n_jobs: int,
) -> tuple[int, int, float, float, float, float | None]:
    validated_seed = validate_seed32(seed, field_name="seed")
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
    if ease_k_small < 1:
        raise ValueError(f"ease_k_small must be >= 1, got {ease_k_small}")
    validated_n_jobs = _validate_filter_n_jobs(n_jobs)
    validated_easy_skill = _validate_unit_interval(
        easy_skill_threshold,
        field_name="easy_skill_threshold",
    )
    validated_easy_gain = _validate_unit_interval(
        easy_gain_threshold,
        field_name="easy_gain_threshold",
    )
    validated_hard_skill = _validate_unit_interval(
        hard_skill_threshold,
        field_name="hard_skill_threshold",
    )
    validated_stump_skill = None
    if stump_skill_threshold is not None:
        validated_stump_skill = _validate_unit_interval(
            stump_skill_threshold,
            field_name="stump_skill_threshold",
        )
    return (
        validated_seed,
        validated_n_jobs,
        validated_easy_skill,
        validated_easy_gain,
        validated_hard_skill,
        validated_stump_skill,
    )


def _apply_extra_trees_filter_numpy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    task: str,
    seed: int,
    lineage_payload: Mapping[str, Any] | None = None,
    lineage_base_dir: Path | None = None,
    n_estimators: int = 25,
    max_depth: int = 6,
    min_samples_leaf: int = 1,
    max_leaf_nodes: int | None = None,
    max_features: str | int | float = "auto",
    n_bootstrap: int = 200,
    ease_k_small: int = 16,
    easy_skill_threshold: float = 0.8,
    easy_gain_threshold: float = 0.1,
    hard_skill_threshold: float = 0.0,
    stump_skill_threshold: float | None = None,
    use_lineage_veto: bool = True,
    n_jobs: int = -1,
) -> tuple[bool, dict[str, Any]]:
    """Apply the small-shot ease filter from NumPy train/test arrays."""

    (
        seed,
        validated_n_jobs,
        easy_skill_threshold,
        easy_gain_threshold,
        hard_skill_threshold,
        stump_skill_threshold,
    ) = _validate_filter_params(
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        n_bootstrap=n_bootstrap,
        ease_k_small=ease_k_small,
        easy_skill_threshold=easy_skill_threshold,
        easy_gain_threshold=easy_gain_threshold,
        hard_skill_threshold=hard_skill_threshold,
        stump_skill_threshold=stump_skill_threshold,
        n_jobs=n_jobs,
    )

    x_train_np = np.asarray(x_train, dtype=np.float32)
    x_test_np = np.asarray(x_test, dtype=np.float32)
    if x_train_np.ndim != 2:
        raise ValueError(
            f"x_train must be rank-2 [n_rows, n_features], got shape {tuple(x_train_np.shape)}"
        )
    if x_test_np.ndim != 2:
        raise ValueError(
            f"x_test must be rank-2 [n_rows, n_features], got shape {tuple(x_test_np.shape)}"
        )
    if int(x_train_np.shape[1]) != int(x_test_np.shape[1]):
        raise ValueError(
            "x_train/x_test feature-count mismatch in filter: "
            f"x_train has {int(x_train_np.shape[1])} columns, "
            f"x_test has {int(x_test_np.shape[1])} columns."
        )
    x_train_np = np.ascontiguousarray(x_train_np)
    x_test_np = np.ascontiguousarray(x_test_np)

    n_train = int(x_train_np.shape[0])
    n_test = int(x_test_np.shape[0])
    if n_train <= 0:
        raise ValueError("x_train must contain at least one row for filter replay.")
    if n_test <= 0:
        raise ValueError("x_test must contain at least one row for filter replay.")

    y_train_target, y_test_target = _prepare_targets(
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        task=task,
    )
    if int(y_train_target.shape[0]) != n_train:
        raise ValueError(
            "x_train/y_train row-count mismatch in filter: "
            f"x_train has {n_train} rows, y_train has {int(y_train_target.shape[0])} rows."
        )
    if int(y_test_target.shape[0]) != n_test:
        raise ValueError(
            "x_test/y_test row-count mismatch in filter: "
            f"x_test has {n_test} rows, y_test has {int(y_test_target.shape[0])} rows."
        )

    ease_k_small_effective = min(int(ease_k_small), n_train)
    details: dict[str, Any] = {
        "backend": _BACKEND,
        "filter_mode": _FILTER_MODE,
        "n_jobs": int(validated_n_jobs),
        "ease_k_small_requested": int(ease_k_small),
        "ease_k_small_effective": int(ease_k_small_effective),
        "easy_skill_threshold": float(easy_skill_threshold),
        "easy_gain_threshold": float(easy_gain_threshold),
        "hard_skill_threshold": float(hard_skill_threshold),
        "stump_skill_threshold": (
            None if stump_skill_threshold is None else float(stump_skill_threshold)
        ),
        "lineage_veto_applied": False,
        "stump_skill": None,
        "skill_small": None,
        "skill_full": None,
        "skill_gain": None,
        "skill_small_lb95": None,
        "skill_gain_ub95": None,
        "skill_full_ub95": None,
    }

    if bool(use_lineage_veto):
        lineage_veto_applied, has_path = _lineage_has_feature_target_path(
            lineage_payload=lineage_payload,
            lineage_base_dir=lineage_base_dir,
        )
        details["lineage_veto_applied"] = bool(lineage_veto_applied)
        if lineage_veto_applied and has_path is False:
            details["reason"] = "no_feature_target_path"
            return False, details

    rng = np.random.default_rng(int(seed))
    if ease_k_small_effective >= n_train:
        small_idx = np.arange(n_train, dtype=np.int64)
    else:
        small_idx = np.sort(rng.choice(n_train, size=ease_k_small_effective, replace=False))

    x_small = np.ascontiguousarray(x_train_np[small_idx], dtype=np.float32)
    y_small = np.ascontiguousarray(y_train_target[small_idx], dtype=np.float32)
    y_train_target = np.ascontiguousarray(y_train_target, dtype=np.float32)
    y_test_target = np.ascontiguousarray(y_test_target, dtype=np.float32)

    baseline_small_mean = cast(
        np.ndarray,
        np.asarray(np.mean(y_small, axis=0, keepdims=True, dtype=np.float32), dtype=np.float32),
    )
    baseline_full_mean = cast(
        np.ndarray,
        np.asarray(
            np.mean(y_train_target, axis=0, keepdims=True, dtype=np.float32),
            dtype=np.float32,
        ),
    )

    if stump_skill_threshold is not None:
        stump_skill = _fit_best_stump_skill(
            x_train=x_small,
            y_train=y_small,
            x_test=x_test_np,
            y_test=y_test_target,
            baseline_small_mean=baseline_small_mean,
            seed=seed,
        )
        details["stump_skill"] = float(stump_skill)
        if float(stump_skill) > float(stump_skill_threshold):
            details["reason"] = "too_easy_stump"
            return False, details

    pred_small = _fit_extra_trees_predictions(
        x_train=x_small,
        y_train=y_small,
        x_test=x_test_np,
        task=task,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        max_features=max_features,
        n_jobs=validated_n_jobs,
    )
    pred_full = _fit_extra_trees_predictions(
        x_train=x_train_np,
        y_train=y_train_target,
        x_test=x_test_np,
        task=task,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        max_features=max_features,
        n_jobs=validated_n_jobs,
    )

    skill_small = _skill_from_predictions(
        pred=pred_small,
        target=y_test_target,
        baseline_mean=baseline_small_mean,
    )
    skill_full = _skill_from_predictions(
        pred=pred_full,
        target=y_test_target,
        baseline_mean=baseline_full_mean,
    )
    skill_gain = _shared_baseline_skill_gain(
        pred_small=pred_small,
        pred_full=pred_full,
        target=y_test_target,
        baseline_mean=baseline_full_mean,
    )
    details["skill_small"] = float(skill_small)
    details["skill_full"] = float(skill_full)
    details["skill_gain"] = float(skill_gain)
    details.update(
        _bootstrap_skill_summary(
            pred_small=pred_small,
            pred_full=pred_full,
            target=y_test_target,
            baseline_small_mean=baseline_small_mean,
            baseline_full_mean=baseline_full_mean,
            baseline_gain_mean=baseline_full_mean,
            seed=seed,
            n_bootstrap=int(n_bootstrap),
        )
    )

    if float(details["skill_small_lb95"]) > float(easy_skill_threshold) and float(
        details["skill_gain_ub95"]
    ) < float(easy_gain_threshold):
        details["reason"] = "too_easy_small_shot"
        return False, details

    if float(details["skill_full_ub95"]) < float(hard_skill_threshold):
        details["reason"] = "too_hard_garbage"
        return False, details

    return True, details


def apply_extra_trees_filter(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    task: str,
    seed: int,
    lineage_payload: Mapping[str, Any] | None = None,
    lineage_base_dir: str | Path | None = None,
    n_estimators: int = 25,
    max_depth: int = 6,
    min_samples_leaf: int = 1,
    max_leaf_nodes: int | None = None,
    max_features: str | int | float = "auto",
    n_bootstrap: int = 200,
    ease_k_small: int = 16,
    easy_skill_threshold: float = 0.8,
    easy_gain_threshold: float = 0.1,
    hard_skill_threshold: float = 0.0,
    stump_skill_threshold: float | None = None,
    use_lineage_veto: bool = True,
    n_jobs: int = -1,
) -> tuple[bool, dict[str, Any]]:
    """Apply the small-shot ease filter from torch train/test tensors."""

    x_train_np = np.asarray(
        x_train.detach().to(device="cpu", dtype=torch.float32).numpy(),
        dtype=np.float32,
    )
    x_test_np = np.asarray(
        x_test.detach().to(device="cpu", dtype=torch.float32).numpy(),
        dtype=np.float32,
    )
    if task == "classification":
        y_train_np = np.asarray(
            y_train.detach().to(device="cpu", dtype=torch.int64).view(-1).numpy()
        )
        y_test_np = np.asarray(y_test.detach().to(device="cpu", dtype=torch.int64).view(-1).numpy())
    else:
        y_train_np = np.asarray(y_train.detach().to(device="cpu", dtype=torch.float32).numpy())
        y_test_np = np.asarray(y_test.detach().to(device="cpu", dtype=torch.float32).numpy())

    lineage_base_dir_path = None if lineage_base_dir is None else Path(lineage_base_dir)
    return _apply_extra_trees_filter_numpy(
        x_train_np,
        y_train_np,
        x_test_np,
        y_test_np,
        task=task,
        seed=seed,
        lineage_payload=lineage_payload,
        lineage_base_dir=lineage_base_dir_path,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        max_features=max_features,
        n_bootstrap=n_bootstrap,
        ease_k_small=ease_k_small,
        easy_skill_threshold=easy_skill_threshold,
        easy_gain_threshold=easy_gain_threshold,
        hard_skill_threshold=hard_skill_threshold,
        stump_skill_threshold=stump_skill_threshold,
        use_lineage_veto=use_lineage_veto,
        n_jobs=n_jobs,
    )
