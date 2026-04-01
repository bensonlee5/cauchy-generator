"""Torch-native metric extraction for diagnostics and generation telemetry."""

from __future__ import annotations

import math
from typing import Any, cast

import torch

from dagzoo.core.layout_types import MechanismFamily
from dagzoo.core.metric_constants import (
    EPS as _EPS,
)
from dagzoo.core.metric_constants import (
    NUM_BOOTSTRAP as _NUM_BOOTSTRAP,
)
from dagzoo.core.metric_constants import (
    RIDGE_LAMBDA as _RIDGE_LAMBDA,
)
from dagzoo.core.metric_constants import (
    TASK_CLASSIFICATION as _TASK_CLASSIFICATION,
)
from dagzoo.core.metric_constants import (
    TASK_REGRESSION as _TASK_REGRESSION,
)
from dagzoo.core.metric_constants import (
    finite_or_none as _finite_or_none,
)
from dagzoo.core.metric_constants import (
    resolve_task_from_metadata,
    validate_metric_shapes,
)
from dagzoo.core.shift import mechanism_nonlinear_mass
from dagzoo.math import (
    coerce_optional_finite_float as _coerce_optional_finite_float,
)
from dagzoo.types import DatasetBundle


def extract_torch_metrics(
    bundle: DatasetBundle,
    *,
    target_metric_names: set[str],
    include_spearman: bool = False,
) -> dict[str, Any]:
    """Extract a metric subset directly from torch tensors without NumPy conversion."""

    x_train = _as_2d_float_tensor(bundle.X_train, name="X_train")
    x_test = _as_2d_float_tensor(bundle.X_test, name="X_test")
    y_train = _as_target_tensor(bundle.y_train, name="y_train")
    y_test = _as_target_tensor(bundle.y_test, name="y_test")

    validate_metric_shapes(
        feature_types=bundle.feature_types,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
    x_all = torch.cat([x_train, x_test], dim=0)
    y_all = _concat_targets(y_train, y_test)
    task = resolve_task(bundle.metadata, y_all)

    required = set(target_metric_names)
    metrics: dict[str, Any] = {}
    metrics["task"] = task
    metrics["n_rows"] = float(x_all.shape[0])
    metrics["n_features"] = float(x_all.shape[1])

    (
        n_categorical_features,
        categorical_ratio,
        cat_cardinality_min,
        cat_cardinality_mean,
        cat_cardinality_max,
    ) = _categorical_cardinality_stats(x_all, bundle.feature_types)
    metrics["n_categorical_features"] = float(n_categorical_features)
    metrics["categorical_ratio"] = float(categorical_ratio)
    metrics["cat_cardinality_min"] = (
        None if cat_cardinality_min is None else float(cat_cardinality_min)
    )
    metrics["cat_cardinality_mean"] = (
        None if cat_cardinality_mean is None else float(cat_cardinality_mean)
    )
    metrics["cat_cardinality_max"] = (
        None if cat_cardinality_max is None else float(cat_cardinality_max)
    )

    n_classes: float | None = None
    if task == _TASK_CLASSIFICATION:
        y_labels = _classification_labels(y_all)
        n_classes = float(torch.unique(y_labels).numel())
        class_entropy, majority_minority_ratio = _class_imbalance_stats(y_labels)
        metrics["class_entropy"] = class_entropy
        metrics["majority_minority_ratio"] = majority_minority_ratio
    else:
        metrics["class_entropy"] = None
        metrics["majority_minority_ratio"] = None
    metrics["n_classes"] = n_classes
    metrics.update(_extract_shift_observability(bundle.metadata))

    needs_pearson = (
        bool(
            {"pearson_abs_mean", "pearson_abs_max", "spearman_abs_mean", "spearman_abs_max"}
            & required
        )
        or include_spearman
    )
    pearson_abs_mean: float | None = None
    pearson_abs_max: float | None = None
    if needs_pearson:
        pearson_abs_mean, pearson_abs_max = _pearson_abs_stats(x_all)
    metrics["pearson_abs_mean"] = pearson_abs_mean
    metrics["pearson_abs_max"] = pearson_abs_max

    spearman_abs_mean: float | None = None
    spearman_abs_max: float | None = None
    if include_spearman or {"spearman_abs_mean", "spearman_abs_max"} & required:
        spearman_abs_mean, spearman_abs_max = _spearman_abs_stats(x_all)
    metrics["spearman_abs_mean"] = spearman_abs_mean
    metrics["spearman_abs_max"] = spearman_abs_max

    needs_linearity = bool({"linearity_proxy", "nonlinearity_proxy"} & required)
    linearity_proxy: float | None = None
    if needs_linearity:
        linearity_proxy = _compute_linearity_proxy(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            task=task,
        )
    metrics["linearity_proxy"] = linearity_proxy

    needs_wins = bool({"wins_ratio_proxy", "nonlinearity_proxy"} & required)
    wins_ratio_proxy: float | None = None
    if needs_wins:
        wins_ratio_proxy = _compute_wins_ratio_proxy(x=x_all, y=y_all, task=task)
    metrics["wins_ratio_proxy"] = wins_ratio_proxy

    nonlinearity_proxy: float | None = None
    if linearity_proxy is not None and wins_ratio_proxy is not None:
        nonlinearity_proxy = max(0.0, wins_ratio_proxy - linearity_proxy)
    metrics["nonlinearity_proxy"] = nonlinearity_proxy

    snr_proxy_db: float | None = None
    if "snr_proxy_db" in required:
        snr_proxy_db = _compute_snr_proxy_db(
            x_train=x_train,
            x_all=x_all,
            y_train=y_train,
            y_all=y_all,
            task=task,
        )
    metrics["snr_proxy_db"] = snr_proxy_db

    # Return only targets and task, keeping deterministic key ordering for metrics.
    results: dict[str, Any] = {
        metric_name: metrics.get(metric_name) for metric_name in sorted(required)
    }
    results["task"] = task
    return results


def _as_2d_float_tensor(value: Any, *, name: str) -> torch.Tensor:
    arr = torch.as_tensor(value, dtype=torch.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array, got shape={tuple(arr.shape)!r}")
    return arr


def _as_target_tensor(value: Any, *, name: str) -> torch.Tensor:
    arr = torch.as_tensor(value)
    if arr.ndim not in {1, 2}:
        raise ValueError(f"{name} must be rank-1 or rank-2, got shape={tuple(arr.shape)!r}")
    return arr


def _concat_targets(y_train: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
    if y_train.ndim == y_test.ndim:
        return torch.cat([y_train, y_test], dim=0)

    if y_train.ndim == 1 and y_test.ndim == 2 and y_test.shape[1] == 1:
        return torch.cat([y_train.unsqueeze(1), y_test], dim=0)

    if y_train.ndim == 2 and y_test.ndim == 1 and y_train.shape[1] == 1:
        return torch.cat([y_train, y_test.unsqueeze(1)], dim=0)

    raise ValueError(
        "y_train/y_test rank mismatch is unsupported: "
        f"{tuple(y_train.shape)!r} vs {tuple(y_test.shape)!r}"
    )


def _extract_shift_observability(metadata: dict[str, Any]) -> dict[str, float | None]:
    shift_raw = metadata.get("shift")
    shift_payload = shift_raw if isinstance(shift_raw, dict) else {}
    graph_edge_density = _coerce_optional_finite_float(metadata.get("graph_edge_density"))
    shift_enabled = _coerce_bool(shift_payload.get("enabled"), default=False)
    shift_graph_scale = _coerce_finite_float(shift_payload.get("graph_scale"), default=0.0)
    shift_mechanism_scale = _coerce_finite_float(
        shift_payload.get("mechanism_scale"),
        default=0.0,
    )
    shift_variance_scale = _coerce_finite_float(shift_payload.get("variance_scale"), default=0.0)
    shift_edge_odds_multiplier = _coerce_finite_float(
        shift_payload.get("edge_odds_multiplier"),
        default=1.0,
    )
    shift_noise_variance_multiplier = _coerce_finite_float(
        shift_payload.get("noise_variance_multiplier"),
        default=1.0,
    )
    if shift_edge_odds_multiplier <= 0.0:
        shift_edge_odds_multiplier = 1.0
    if shift_noise_variance_multiplier <= 0.0:
        shift_noise_variance_multiplier = 1.0

    shift_mechanism_nonlinear_mass = _coerce_optional_finite_float(
        shift_payload.get("mechanism_nonlinear_mass")
    )
    if shift_mechanism_nonlinear_mass is None:
        mechanism_logit_tilt = _coerce_finite_float(
            shift_payload.get("mechanism_logit_tilt"),
            default=shift_mechanism_scale,
        )
        family_weights: dict[str, float] | None = None
        config_payload = metadata.get("config")
        if isinstance(config_payload, dict):
            mechanism_payload = config_payload.get("mechanism")
            if isinstance(mechanism_payload, dict):
                raw_mix = mechanism_payload.get("function_family_mix")
                if isinstance(raw_mix, dict):
                    parsed_mix: dict[str, float] = {}
                    for raw_key, raw_weight in raw_mix.items():
                        if isinstance(raw_key, bool) or not isinstance(raw_key, str):
                            continue
                        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                            continue
                        weight = float(raw_weight)
                        if not math.isfinite(weight) or weight <= 0.0:
                            continue
                        parsed_mix[raw_key.strip().lower()] = weight
                    if parsed_mix:
                        family_weights = parsed_mix
        shift_mechanism_nonlinear_mass = mechanism_nonlinear_mass(
            mechanism_logit_tilt=mechanism_logit_tilt,
            family_weights=cast(dict[MechanismFamily, float] | None, family_weights),
        )
    shift_mechanism_nonlinear_mass = min(1.0, max(0.0, shift_mechanism_nonlinear_mass))

    return {
        "graph_edge_density": graph_edge_density,
        "shift_enabled": 1.0 if shift_enabled else 0.0,
        "shift_graph_scale": float(shift_graph_scale),
        "shift_mechanism_scale": float(shift_mechanism_scale),
        "shift_variance_scale": float(shift_variance_scale),
        "shift_edge_odds_multiplier": float(shift_edge_odds_multiplier),
        "shift_mechanism_nonlinear_mass": float(shift_mechanism_nonlinear_mass),
        "shift_noise_variance_multiplier": float(shift_noise_variance_multiplier),
    }


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _coerce_finite_float(value: Any, *, default: float) -> float:
    as_float = _coerce_optional_finite_float(value)
    if as_float is None:
        return default
    return as_float


def resolve_task(metadata: dict[str, Any], y_all: torch.Tensor) -> str:
    task = resolve_task_from_metadata(metadata)
    if task is not None:
        return task

    labels = y_all.reshape(-1)
    if labels.numel() == 0:
        raise ValueError("Cannot infer task from empty target array.")
    if labels.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        return _TASK_CLASSIFICATION

    if labels.dtype in {torch.float16, torch.float32, torch.float64, torch.bfloat16}:
        finite_mask = torch.isfinite(labels)
        if bool(finite_mask.any()):
            finite_vals = labels[finite_mask].to(torch.float64)
            rounded = torch.round(finite_vals)
            if bool(torch.all(torch.abs(finite_vals - rounded) <= 1e-8)):
                n_unique = int(torch.unique(rounded.to(torch.int64)).numel())
                max_classes = max(10, int(math.sqrt(float(finite_vals.numel()))) + 1)
                if 2 <= n_unique <= max_classes:
                    return _TASK_CLASSIFICATION

    return _TASK_REGRESSION


def _classification_labels(y: torch.Tensor) -> torch.Tensor:
    arr = y
    if arr.ndim == 2:
        if arr.shape[1] != 1:
            raise ValueError(
                "Classification targets must be rank-1 or single-column rank-2 arrays, "
                f"got shape={tuple(arr.shape)!r}."
            )
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"Classification targets must be rank-1, got shape={tuple(arr.shape)!r}.")
    return torch.round(arr.to(torch.float64)).to(torch.int64)


def _regression_target_matrix(y: torch.Tensor) -> torch.Tensor:
    arr = y.to(torch.float64)
    if arr.ndim == 1:
        return arr.unsqueeze(1)
    if arr.ndim == 2:
        return arr
    raise ValueError(
        f"Regression targets must be rank-1 or rank-2, got shape={tuple(arr.shape)!r}."
    )


def _classification_targets_for_splits(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_train_labels = _classification_labels(y_train)
    y_test_labels = _classification_labels(y_test)
    all_labels = torch.cat([y_train_labels, y_test_labels], dim=0)
    classes, inverse = torch.unique(all_labels, sorted=True, return_inverse=True)
    n_classes = int(classes.numel())
    if n_classes <= 0:
        raise ValueError("Classification targets must include at least one class.")
    y_train_targets = torch.nn.functional.one_hot(
        inverse[: y_train_labels.shape[0]],
        num_classes=n_classes,
    ).to(torch.float64)
    y_test_targets = torch.nn.functional.one_hot(
        inverse[y_train_labels.shape[0] :],
        num_classes=n_classes,
    ).to(torch.float64)
    return y_train_targets, y_test_targets, all_labels


def _ridge_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_pred: torch.Tensor,
    *,
    ridge_lambda: float = _RIDGE_LAMBDA,
) -> torch.Tensor:
    train_design = torch.cat(
        [torch.ones((x_train.shape[0], 1), dtype=torch.float64, device=x_train.device), x_train],
        dim=1,
    )
    pred_design = torch.cat(
        [torch.ones((x_pred.shape[0], 1), dtype=torch.float64, device=x_pred.device), x_pred],
        dim=1,
    )
    gram = train_design.t() @ train_design
    reg = torch.eye(gram.shape[0], dtype=torch.float64, device=gram.device) * float(ridge_lambda)
    reg[0, 0] = 0.0
    rhs = train_design.t() @ y_train
    try:
        weights = torch.linalg.solve(gram + reg, rhs)
    except RuntimeError:
        weights = torch.linalg.pinv(gram + reg) @ rhs
    return cast(torch.Tensor, pred_design @ weights)


def _bootstrap_wins_ratio(
    pred: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
) -> float:
    n_rows = int(target.shape[0])
    if n_rows <= 0:
        raise ValueError("Bootstrap wins ratio requires at least one row.")
    gen = torch.Generator(device=target.device)
    gen.manual_seed(0)
    wins = 0
    chunk_size = 16
    for start in range(0, _NUM_BOOTSTRAP, chunk_size):
        bs = min(chunk_size, _NUM_BOOTSTRAP - start)
        idx = torch.randint(0, n_rows, (bs, n_rows), generator=gen, device=target.device)
        sampled_target = target[idx]
        sampled_pred = pred[idx]
        mse_pred = ((sampled_pred - sampled_target) ** 2).mean(dim=(1, 2))
        mse_base = ((baseline - sampled_target) ** 2).mean(dim=(1, 2))
        wins += int((mse_pred < mse_base).sum().item())
    return float(wins / _NUM_BOOTSTRAP)


def _compute_linearity_proxy(
    *,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    task: str,
) -> float | None:
    if x_train.shape[0] <= 0 or x_test.shape[0] <= 0:
        return None

    if task == _TASK_CLASSIFICATION:
        y_train_targets, y_test_targets, _ = _classification_targets_for_splits(y_train, y_test)
    elif task == _TASK_REGRESSION:
        y_train_targets = _regression_target_matrix(y_train)
        y_test_targets = _regression_target_matrix(y_test)
    else:
        raise ValueError(f"Unsupported task '{task}'.")

    pred_test = _ridge_predict(x_train, y_train_targets, x_test)
    baseline = torch.mean(y_train_targets, dim=0, keepdim=True)
    wins_ratio = _bootstrap_wins_ratio(pred=pred_test, target=y_test_targets, baseline=baseline)
    return _finite_or_none(wins_ratio)


def _compute_wins_ratio_proxy(*, x: torch.Tensor, y: torch.Tensor, task: str) -> float | None:
    try:
        x_tensor = x.to(torch.float32)
        n_rows = int(x_tensor.shape[0])
        if n_rows < 4:
            return None
        n_test = max(1, n_rows // 4)
        n_train = n_rows - n_test
        if n_train <= 0:
            return None
        gen = torch.Generator(device=x_tensor.device)
        gen.manual_seed(0)
        perm = torch.randperm(n_rows, generator=gen, device=x_tensor.device)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        if task == _TASK_CLASSIFICATION:
            y_train_targets, y_test_targets, _ = _classification_targets_for_splits(
                y[train_idx],
                y[test_idx],
            )
        elif task == _TASK_REGRESSION:
            y_train_targets = _regression_target_matrix(y[train_idx])
            y_test_targets = _regression_target_matrix(y[test_idx])
        else:
            raise ValueError(f"Unsupported task '{task}'.")
        pred_test = _ridge_predict(
            x_tensor[train_idx],
            y_train_targets,
            x_tensor[test_idx],
        )
        baseline = torch.mean(y_train_targets, dim=0, keepdim=True)
        wins_ratio = _bootstrap_wins_ratio(
            pred=pred_test,
            target=y_test_targets,
            baseline=baseline,
        )
    except Exception:
        return None

    return _finite_or_none(wins_ratio)


def _compute_snr_proxy_db(
    *,
    x_train: torch.Tensor,
    x_all: torch.Tensor,
    y_train: torch.Tensor,
    y_all: torch.Tensor,
    task: str,
) -> float | None:
    if x_train.shape[0] <= 0:
        return None

    if task == _TASK_CLASSIFICATION:
        y_train_labels = _classification_labels(y_train)
        y_all_labels = _classification_labels(y_all)
        classes, y_all_inverse = torch.unique(y_all_labels, sorted=True, return_inverse=True)
        n_classes = int(classes.numel())
        if n_classes <= 0:
            return None
        y_train_inverse = torch.searchsorted(classes, y_train_labels)
        y_train_target = torch.nn.functional.one_hot(y_train_inverse, num_classes=n_classes).to(
            torch.float64
        )
        y_all_target = torch.nn.functional.one_hot(y_all_inverse, num_classes=n_classes).to(
            torch.float64
        )
    elif task == _TASK_REGRESSION:
        y_train_target = _regression_target_matrix(y_train)
        y_all_target = _regression_target_matrix(y_all)
    else:
        raise ValueError(f"Unsupported task '{task}'.")

    pred_all = _ridge_predict(x_train, y_train_target, x_all)
    residual = y_all_target - pred_all
    signal_var = float(torch.var(pred_all, correction=0).item())
    noise_var = float(torch.var(residual, correction=0).item())
    if not math.isfinite(signal_var) or not math.isfinite(noise_var):
        return None
    if signal_var <= 0.0:
        return None
    noise_var = max(noise_var, _EPS)
    snr = signal_var / noise_var
    if not math.isfinite(snr) or snr <= 0.0:
        return None
    return float(10.0 * math.log10(snr))


def _pearson_abs_stats(x: torch.Tensor) -> tuple[float | None, float | None]:
    if x.shape[1] < 2 or x.shape[0] < 2:
        return None, None
    centered = x - torch.mean(x, dim=0, keepdim=True)
    cov = centered.t() @ centered
    cov = cov / max(1, x.shape[0] - 1)
    var = torch.diagonal(cov, 0)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    denom = std[:, None] * std[None, :]
    corr = cov / denom
    upper = torch.triu_indices(corr.shape[0], corr.shape[1], offset=1, device=corr.device)
    values = torch.abs(corr[upper[0], upper[1]])
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None, None
    return float(torch.mean(values).item()), float(torch.max(values).item())


def _rankdata_average(values: torch.Tensor) -> torch.Tensor:
    arr = values.to(torch.float64)
    ranks = torch.full((arr.shape[0],), float("nan"), dtype=torch.float64, device=arr.device)
    finite_mask = torch.isfinite(arr)
    if not bool(finite_mask.any()):
        return ranks

    finite_values = arr[finite_mask]
    order = torch.argsort(finite_values, stable=True)
    sorted_values = finite_values[order]
    finite_ranks = torch.empty_like(finite_values, dtype=torch.float64)

    i = 0
    n = int(sorted_values.shape[0])
    while i < n:
        j = i + 1
        while j < n and bool(sorted_values[j] == sorted_values[i]):
            j += 1
        mean_rank = (i + j - 1) * 0.5 + 1.0
        finite_ranks[order[i:j]] = mean_rank
        i = j

    finite_indices = torch.where(finite_mask)[0]
    ranks[finite_indices] = finite_ranks
    return ranks


def _spearman_abs_stats(x: torch.Tensor) -> tuple[float | None, float | None]:
    ranked = torch.empty_like(x, dtype=torch.float64)
    for i in range(x.shape[1]):
        ranked[:, i] = _rankdata_average(x[:, i])
    return _pearson_abs_stats(ranked)


def _class_imbalance_stats(y_labels: torch.Tensor) -> tuple[float | None, float | None]:
    if y_labels.numel() == 0:
        return None, None
    _, counts = torch.unique(y_labels, return_counts=True)
    if counts.numel() == 0:
        return None, None
    probs = counts.to(torch.float64) / float(torch.sum(counts).item())
    entropy = float(-(probs * torch.log(torch.clamp(probs, min=_EPS))).sum().item())
    positive = counts[counts > 0]
    if positive.numel() == 0:
        return entropy, None
    ratio = float(torch.max(positive).item() / torch.min(positive).item())
    return entropy, ratio


def _categorical_cardinality_stats(
    x_all: torch.Tensor,
    feature_types: list[str],
) -> tuple[int, float, int | None, float | None, int | None]:
    n_features = int(x_all.shape[1])
    if len(feature_types) != n_features:
        raise ValueError(
            f"feature_types length must match feature count: {len(feature_types)} != {n_features}"
        )
    cat_indices = [i for i, kind in enumerate(feature_types) if kind == "cat"]
    n_categorical = int(len(cat_indices))
    cat_ratio = float(n_categorical / max(1, n_features))
    if n_categorical == 0:
        return n_categorical, cat_ratio, None, None, None

    cardinalities: list[int] = []
    for idx in cat_indices:
        col = x_all[:, idx]
        finite = col[torch.isfinite(col)]
        cardinalities.append(int(torch.unique(finite).numel()))

    if not cardinalities:
        return n_categorical, cat_ratio, None, None, None
    return (
        n_categorical,
        cat_ratio,
        int(min(cardinalities)),
        float(sum(cardinalities) / len(cardinalities)),
        int(max(cardinalities)),
    )
