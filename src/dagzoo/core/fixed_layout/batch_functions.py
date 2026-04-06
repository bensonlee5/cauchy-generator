"""Function-family execution helpers for fixed-layout batched generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch

from dagzoo.core.trees import compute_odt_leaf_indices_batch, sample_odt_splits_batch
from dagzoo.core.validation import validate_matrix_output
from dagzoo.functions import activations as activations_module
from dagzoo.sampling.noise import NoiseSamplingSpec, sample_noise_from_spec

from .batch_common import (
    FixedLayoutBatchRng,
    _batch_standardize,
    _lp_distances_to_centers,
    _nearest_lp_center_indices,
    _row_normalize_batch,
    _sample_random_weights_batch,
)
from .plan_types import (
    ActivationMatrixPlan,
    DiscretizationFunctionPlan,
    EmFunctionPlan,
    FixedLayoutActivationPlan,
    FixedLayoutMatrixBaseKind,
    FixedLayoutMatrixPlan,
    FixedLayoutRootBaseKind,
    GaussianMatrixPlan,
    GpFunctionPlan,
    KernelMatrixPlan,
    LinearFunctionPlan,
    NeuralNetFunctionPlan,
    ParametricActivationPlan,
    PiecewiseFunctionPlan,
    QuadraticFunctionPlan,
    SingularValuesMatrixPlan,
    TreeFunctionPlan,
    WeightsMatrixPlan,
)


@dataclass(frozen=True, slots=True)
class _SampledActivationPlanParams:
    """One sampled activation payload reused across deterministic application."""

    standardize_scale: torch.Tensor | None = None
    standardize_row_index: torch.Tensor | None = None
    parametric_q: torch.Tensor | None = None
    uniform_noise: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _SampledLinearFunctionParams:
    matrix: torch.Tensor


@dataclass(frozen=True, slots=True)
class _SampledQuadraticFunctionParams:
    feature_subset: torch.Tensor | None
    matrix: torch.Tensor


@dataclass(frozen=True, slots=True)
class _SampledNeuralNetFunctionParams:
    input_activation: _SampledActivationPlanParams | None
    layer_matrices: tuple[torch.Tensor, ...]
    hidden_activations: tuple[_SampledActivationPlanParams, ...]
    output_activation: _SampledActivationPlanParams | None


@dataclass(frozen=True, slots=True)
class _SampledTreeLevelParams:
    split_feats: torch.Tensor
    thresholds: torch.Tensor
    leaf_values: torch.Tensor


@dataclass(frozen=True, slots=True)
class _SampledTreeFunctionParams:
    levels: tuple[_SampledTreeLevelParams, ...]


@dataclass(frozen=True, slots=True)
class _SampledDiscretizationFunctionParams:
    center_index: torch.Tensor
    lp_norm: torch.Tensor
    linear: _SampledLinearFunctionParams


@dataclass(frozen=True, slots=True)
class _SampledGpFunctionParams:
    input_projection: torch.Tensor | None
    omega: torch.Tensor
    phase_bias: torch.Tensor
    harmonics: torch.Tensor | None
    output_matrix: torch.Tensor


@dataclass(frozen=True, slots=True)
class _SampledEmFunctionParams:
    base_index: torch.Tensor
    center_noise: torch.Tensor
    sigma: torch.Tensor
    p_val: torch.Tensor
    q_val: torch.Tensor
    linear: _SampledLinearFunctionParams


@dataclass(frozen=True, slots=True)
class _SampledPiecewiseGateParams:
    gate_matrix: torch.Tensor


SampledLeafFunctionParams = (
    _SampledLinearFunctionParams
    | _SampledQuadraticFunctionParams
    | _SampledNeuralNetFunctionParams
    | _SampledTreeFunctionParams
    | _SampledDiscretizationFunctionParams
    | _SampledGpFunctionParams
    | _SampledEmFunctionParams
)


def _concat_optional_tensor(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        return None
    return torch.cat(present, dim=0)


def _concat_activation_plan_params(
    params_list: list[_SampledActivationPlanParams],
) -> _SampledActivationPlanParams:
    return _SampledActivationPlanParams(
        standardize_scale=_concat_optional_tensor(
            [params.standardize_scale for params in params_list]
        ),
        standardize_row_index=_concat_optional_tensor(
            [params.standardize_row_index for params in params_list]
        ),
        parametric_q=_concat_optional_tensor([params.parametric_q for params in params_list]),
        uniform_noise=_concat_optional_tensor([params.uniform_noise for params in params_list]),
    )


def _concat_piecewise_gate_params(
    params_list: list[_SampledPiecewiseGateParams],
) -> _SampledPiecewiseGateParams:
    if not params_list:
        raise ValueError("params_list must be non-empty.")
    return _SampledPiecewiseGateParams(
        gate_matrix=torch.cat([params.gate_matrix for params in params_list], dim=0)
    )


def _sample_activation_plan_params(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: FixedLayoutActivationPlan,
    *,
    with_standardize: bool,
) -> _SampledActivationPlanParams:
    y = x.to(torch.float32)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    leading_shape = tuple(int(dim) for dim in y.shape[:-2])
    params = _SampledActivationPlanParams()
    if with_standardize:
        params = _SampledActivationPlanParams(
            standardize_scale=rng.keyed("standardize_scale").log_uniform(
                (y.shape[0],), low=1.0, high=10.0
            ),
            standardize_row_index=rng.keyed("standardize_row_index").randint(
                0, y.shape[1], (y.shape[0],)
            ),
        )

    if isinstance(plan, ParametricActivationPlan):
        kind = str(plan.kind)
        if kind in {"relu_pow", "signed_pow", "inv_pow"}:
            q = rng.keyed(kind).log_uniform(leading_shape, low=0.1, high=10.0)
            return _SampledActivationPlanParams(
                standardize_scale=params.standardize_scale,
                standardize_row_index=params.standardize_row_index,
                parametric_q=q,
            )
        if kind == "gumbel_softmax":
            return _SampledActivationPlanParams(
                standardize_scale=params.standardize_scale,
                standardize_row_index=params.standardize_row_index,
                uniform_noise=rng.keyed("gumbel_softmax").uniform(
                    y.shape,
                    low=1e-6,
                    high=1.0 - 1e-6,
                ),
            )
    return params


def _apply_activation_plan_with_params(
    x: torch.Tensor,
    plan: FixedLayoutActivationPlan,
    params: _SampledActivationPlanParams,
    *,
    with_standardize: bool,
) -> torch.Tensor:
    y = x.to(torch.float32)
    squeezed = False
    if y.dim() == 2:
        y = y.unsqueeze(0)
        squeezed = True
    leading_shape = tuple(int(dim) for dim in y.shape[:-2])
    if with_standardize:
        if params.standardize_scale is None or params.standardize_row_index is None:
            raise ValueError("Activation params are missing standardization draws.")
        y = _batch_standardize(y)
        offsets = y[
            torch.arange(y.shape[0], device=y.device),
            params.standardize_row_index.to(device=y.device, dtype=torch.long),
        ].unsqueeze(1)
        y = params.standardize_scale.to(device=y.device, dtype=y.dtype).view(-1, 1, 1) * (
            y - offsets
        )

    if isinstance(plan, ParametricActivationPlan):
        kind = str(plan.kind)
        if kind == "relu_pow":
            if params.parametric_q is None:
                raise ValueError("relu_pow activation params are missing q.")
            y = torch.pow(
                torch.clamp(y, min=0.0),
                params.parametric_q.to(device=y.device, dtype=y.dtype).reshape(
                    *leading_shape, 1, 1
                ),
            )
        elif kind == "signed_pow":
            if params.parametric_q is None:
                raise ValueError("signed_pow activation params are missing q.")
            y = torch.sign(y) * torch.pow(
                torch.abs(y),
                params.parametric_q.to(device=y.device, dtype=y.dtype).reshape(
                    *leading_shape, 1, 1
                ),
            )
        elif kind == "inv_pow":
            if params.parametric_q is None:
                raise ValueError("inv_pow activation params are missing q.")
            y = torch.pow(
                torch.abs(y) + 1e-3,
                -params.parametric_q.to(device=y.device, dtype=y.dtype).reshape(
                    *leading_shape, 1, 1
                ),
            )
        elif kind == "poly":
            if plan.poly_power is None:
                raise ValueError("poly activation plan requires poly_power.")
            y = torch.pow(y, float(int(plan.poly_power)))
        elif kind == "gumbel_softmax":
            if plan.temperature is None:
                raise ValueError("gumbel_softmax activation plan requires temperature.")
            if params.uniform_noise is None:
                raise ValueError("gumbel_softmax activation params are missing noise.")
            y = activations_module._gumbel_softmax_activation(
                y,
                temperature=float(plan.temperature),
                uniform_noise=params.uniform_noise.to(device=y.device, dtype=y.dtype),
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown activation plan kind: {kind!r}")
    else:
        y = activations_module._fixed_activation(
            y.reshape(-1, y.shape[-1]),
            str(plan.name),
        ).reshape_as(y)

    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    y = torch.clamp(y, -1e6, 1e6)
    if with_standardize:
        y = _batch_standardize(y)
    if squeezed:
        y = y.squeeze(0)
    return y.to(torch.float32)


def _apply_activation_plan(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: FixedLayoutActivationPlan,
    *,
    with_standardize: bool,
) -> torch.Tensor:
    return _apply_activation_plan_with_params(
        x,
        plan,
        _sample_activation_plan_params(
            x,
            rng,
            plan,
            with_standardize=with_standardize,
        ),
        with_standardize=with_standardize,
    )


def _base_matrix_plan(kind: FixedLayoutMatrixBaseKind) -> FixedLayoutMatrixPlan:
    if kind == "gaussian":
        return GaussianMatrixPlan()
    if kind == "weights":
        return WeightsMatrixPlan()
    if kind == "singular_values":
        return SingularValuesMatrixPlan()
    if kind == "kernel":
        return KernelMatrixPlan()
    raise ValueError(f"Unsupported fixed-layout matrix base kind: {kind!r}")


def _sample_random_matrix_from_plan_batch(
    plan: FixedLayoutMatrixPlan,
    *,
    out_dim: int,
    in_dim: int,
    rng: FixedLayoutBatchRng,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
    matrix_count: int | None = None,
) -> torch.Tensor:
    leading_shape = () if matrix_count is None else (int(matrix_count),)
    shape = (rng.batch_size, *leading_shape, int(out_dim), int(in_dim))
    if isinstance(plan, GaussianMatrixPlan):
        matrix = sample_noise_from_spec(
            shape,
            generator=rng.keyed("gaussian").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
    elif isinstance(plan, WeightsMatrixPlan):
        g = sample_noise_from_spec(
            shape,
            generator=rng.keyed("gaussian").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        rows = _sample_random_weights_batch(
            rng.keyed("row_weights"),
            dim=int(in_dim),
            leading_shape=(*leading_shape, int(out_dim)),
            parameter_shape=leading_shape,
            correlation_name="weights_matrix_decay",
            sigma_multiplier=float(noise_sigma_multiplier),
            noise_spec=noise_spec,
        )
        matrix = g * rows
    elif isinstance(plan, SingularValuesMatrixPlan):
        d = min(int(out_dim), int(in_dim))
        u_shape = (rng.batch_size, *leading_shape, int(out_dim), d)
        v_shape = (rng.batch_size, *leading_shape, d, int(in_dim))
        u = sample_noise_from_spec(
            u_shape,
            generator=rng.keyed("u").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        v = sample_noise_from_spec(
            v_shape,
            generator=rng.keyed("v").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        weights = _sample_random_weights_batch(
            rng.keyed("singular_values"),
            dim=d,
            leading_shape=leading_shape,
            correlation_name="singular_values_decay",
            sigma_multiplier=float(noise_sigma_multiplier),
            noise_spec=noise_spec,
        )
        matrix = torch.matmul(u * weights.unsqueeze(-2), v)
    elif isinstance(plan, KernelMatrixPlan):
        pts = sample_noise_from_spec(
            (rng.batch_size, *leading_shape, int(out_dim) + int(in_dim), 3),
            generator=rng.keyed("points").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        left = pts[..., : int(out_dim), :].unsqueeze(-2)
        right = pts[..., int(out_dim) :, :].unsqueeze(-3)
        dist = torch.norm(left - right, dim=-1)
        gamma = torch.full(
            (rng.batch_size, *leading_shape, 1, 1),
            float(plan.gamma),
            dtype=torch.float32,
            device=rng.device,
        )
        kernel = torch.exp(-gamma * dist)
        if bool(plan.signed):
            sign = torch.where(
                rng.keyed("sign").uniform(shape, low=0.0, high=1.0) < 0.5,
                -1.0,
                1.0,
            )
            matrix = kernel * sign
        else:
            matrix = kernel
    elif isinstance(plan, ActivationMatrixPlan):
        matrix = _sample_random_matrix_from_plan_batch(
            _base_matrix_plan(plan.base_kind),
            out_dim=int(out_dim),
            in_dim=int(in_dim),
            rng=rng.keyed("base"),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            matrix_count=matrix_count,
        )
        matrix = _apply_activation_plan(
            matrix,
            rng.keyed("activation"),
            plan.activation,
            with_standardize=False,
        )
        matrix = matrix + 1e-3 * sample_noise_from_spec(
            matrix.shape,
            generator=rng.keyed("activation_noise").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
    else:
        raise ValueError(f"Unknown matrix plan: {plan!r}")

    matrix = matrix + 1e-6 * sample_noise_from_spec(
        matrix.shape,
        generator=rng.keyed("jitter").torch_generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    matrix = _row_normalize_batch(matrix)
    validate_matrix_output(matrix, context="_sample_random_matrix_from_plan_batch")
    return matrix


def _sample_piecewise_gate_params(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: PiecewiseFunctionPlan,
    *,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> _SampledPiecewiseGateParams:
    return _SampledPiecewiseGateParams(
        gate_matrix=_sample_random_matrix_from_plan_batch(
            plan.gate_matrix,
            out_dim=1,
            in_dim=int(x.shape[2]),
            rng=rng,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    )


def _apply_piecewise_gate_with_params(
    x: torch.Tensor,
    plan: PiecewiseFunctionPlan,
    params: _SampledPiecewiseGateParams,
) -> torch.Tensor:
    gate_projection = torch.einsum("bni,boi->bno", x, params.gate_matrix)
    return torch.sigmoid((gate_projection + float(plan.gate_bias)) * float(plan.gate_temperature))


def _sample_linear_function_params(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: LinearFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> _SampledLinearFunctionParams:
    return _SampledLinearFunctionParams(
        matrix=_sample_random_matrix_from_plan_batch(
            plan.matrix,
            out_dim=int(out_dim),
            in_dim=int(x.shape[2]),
            rng=rng.keyed("matrix"),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    )


def _apply_sampled_linear_batch(
    x: torch.Tensor,
    params: _SampledLinearFunctionParams,
) -> torch.Tensor:
    return torch.einsum("bni,boi->bno", x, params.matrix)


def _apply_linear_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: LinearFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    return _apply_sampled_linear_batch(
        x,
        _sample_linear_function_params(
            x,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        ),
    )


def _sample_quadratic_function_params(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: QuadraticFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> _SampledQuadraticFunctionParams:
    feature_cap = min(int(x.shape[2]), 20)
    feature_subset: torch.Tensor | None = None
    x_sub = x
    if int(x.shape[2]) > feature_cap:
        feature_subset = rng.keyed("feature_subset").randperm_indices(
            length=int(x.shape[2]),
            sample_size=feature_cap,
        )
        x_sub = torch.gather(
            x,
            2,
            feature_subset.unsqueeze(1).expand(-1, x.shape[1], -1),
        )
    ones = torch.ones((x_sub.shape[0], x_sub.shape[1], 1), device=x.device, dtype=x_sub.dtype)
    x_aug = torch.cat([x_sub, ones], dim=2)
    return _SampledQuadraticFunctionParams(
        feature_subset=feature_subset,
        matrix=_sample_random_matrix_from_plan_batch(
            plan.matrix,
            out_dim=int(x_aug.shape[2]),
            in_dim=int(x_aug.shape[2]),
            rng=rng.keyed("matrix"),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
            matrix_count=int(out_dim),
        ),
    )


def _apply_sampled_quadratic_batch(
    x: torch.Tensor,
    params: _SampledQuadraticFunctionParams,
) -> torch.Tensor:
    x_sub = x
    if params.feature_subset is not None:
        x_sub = torch.gather(
            x,
            2,
            params.feature_subset.to(device=x.device, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, x.shape[1], -1),
        )
    ones = torch.ones((x_sub.shape[0], x_sub.shape[1], 1), device=x.device, dtype=x_sub.dtype)
    x_aug = torch.cat([x_sub, ones], dim=2)
    return torch.einsum("bni,boij,bnj->bno", x_aug, params.matrix, x_aug)


def _apply_quadratic_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: QuadraticFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    return _apply_sampled_quadratic_batch(
        x,
        _sample_quadratic_function_params(
            x,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        ),
    )


def _sample_unit_ball_batch(
    rng: FixedLayoutBatchRng,
    *,
    n_rows: int,
    dim: int,
) -> torch.Tensor:
    vectors = rng.keyed("vectors").normal((rng.batch_size, n_rows, dim))
    vectors = vectors / torch.clamp(torch.norm(vectors, dim=2, keepdim=True), min=1e-6)
    radii = rng.keyed("radii").uniform((rng.batch_size, n_rows, 1), low=0.0, high=1.0)
    return vectors * torch.pow(radii, 1.0 / max(1, dim))


def _sample_random_points_batch(
    rng: FixedLayoutBatchRng,
    *,
    n_rows: int,
    dim: int,
    base_kind: FixedLayoutRootBaseKind,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    if base_kind == "normal":
        return sample_noise_from_spec(
            (rng.batch_size, n_rows, dim),
            generator=rng.keyed("normal").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
    if base_kind == "uniform":
        return rng.keyed("uniform").uniform((rng.batch_size, n_rows, dim), low=-1.0, high=1.0)
    if base_kind == "unit_ball":
        return _sample_unit_ball_batch(rng.keyed("unit_ball"), n_rows=n_rows, dim=dim)

    points = sample_noise_from_spec(
        (rng.batch_size, n_rows, dim),
        generator=rng.keyed("points").torch_generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    weights = _sample_random_weights_batch(
        rng.keyed("weights"),
        dim=dim,
        correlation_name="normal_cov_weights_decay",
        sigma_multiplier=float(noise_sigma_multiplier),
        noise_spec=noise_spec,
    )
    matrices = sample_noise_from_spec(
        (rng.batch_size, dim, dim),
        generator=rng.keyed("matrix").torch_generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    return torch.einsum("bni,bi,bij->bnj", points, weights, matrices.transpose(1, 2))


def _sample_nn_function_params(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: NeuralNetFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> _SampledNeuralNetFunctionParams:
    hidden_width = max(1, int(plan.hidden_width))
    n_layers = max(1, int(plan.n_layers))
    layer_dims = [int(x.shape[2])]
    for _ in range(max(0, n_layers - 1)):
        layer_dims.append(hidden_width)
    layer_dims.append(int(out_dim))
    return _SampledNeuralNetFunctionParams(
        input_activation=(
            None
            if plan.input_activation is None
            else _sample_activation_plan_params(
                x,
                rng.keyed("input_activation"),
                plan.input_activation,
                with_standardize=True,
            )
        ),
        layer_matrices=tuple(
            _sample_random_matrix_from_plan_batch(
                plan.layer_matrices[layer_index],
                out_dim=int(dout),
                in_dim=int(din),
                rng=rng.keyed("layer_matrix", layer_index),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            for layer_index, (din, dout) in enumerate(
                zip(layer_dims[:-1], layer_dims[1:], strict=True)
            )
        ),
        hidden_activations=tuple(
            _sample_activation_plan_params(
                torch.empty(
                    (x.shape[0], x.shape[1], int(layer_dims[layer_index + 1])),
                    device=x.device,
                    dtype=x.dtype,
                ),
                rng.keyed("hidden_activation", layer_index),
                plan.hidden_activations[layer_index],
                with_standardize=True,
            )
            for layer_index in range(len(layer_dims) - 2)
        ),
        output_activation=(
            None
            if plan.output_activation is None
            else _sample_activation_plan_params(
                torch.empty(
                    (x.shape[0], x.shape[1], int(out_dim)),
                    device=x.device,
                    dtype=x.dtype,
                ),
                rng.keyed("output_activation"),
                plan.output_activation,
                with_standardize=True,
            )
        ),
    )


def _apply_sampled_nn_batch(
    x: torch.Tensor,
    plan: NeuralNetFunctionPlan,
    params: _SampledNeuralNetFunctionParams,
) -> torch.Tensor:
    y = x
    if plan.input_activation is not None:
        if params.input_activation is None:
            raise ValueError("NN params are missing input activation draws.")
        y = _apply_activation_plan_with_params(
            y,
            plan.input_activation,
            params.input_activation,
            with_standardize=True,
        )
    for layer_index, matrix in enumerate(params.layer_matrices):
        y = torch.einsum("bni,boi->bno", y, matrix)
        if layer_index < len(params.hidden_activations):
            y = _apply_activation_plan_with_params(
                y,
                plan.hidden_activations[layer_index],
                params.hidden_activations[layer_index],
                with_standardize=True,
            )
    if plan.output_activation is not None:
        if params.output_activation is None:
            raise ValueError("NN params are missing output activation draws.")
        y = _apply_activation_plan_with_params(
            y,
            plan.output_activation,
            params.output_activation,
            with_standardize=True,
        )
    return y


def _apply_nn_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: NeuralNetFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    return _apply_sampled_nn_batch(
        x,
        plan,
        _sample_nn_function_params(
            x,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        ),
    )


def _sample_tree_function_params(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: TreeFunctionPlan,
    *,
    out_dim: int,
    noise_spec: NoiseSamplingSpec | None,
) -> _SampledTreeFunctionParams:
    correction = 1 if int(x.shape[1]) > 1 else 0
    var, _mean = torch.var_mean(x, dim=1, correction=correction)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    probs = torch.clamp(std, min=0.0)
    totals = torch.sum(probs, dim=1, keepdim=True)
    uniform = torch.full_like(probs, 1.0 / max(1, int(probs.shape[1])))
    valid = (
        torch.isfinite(probs).all(dim=1, keepdim=True) & torch.isfinite(totals) & (totals > 1e-12)
    )
    probs = torch.where(valid, probs / torch.clamp(totals, min=1e-12), uniform)
    levels = []
    tree_root = rng.keyed("tree")
    for tree_index, depth in enumerate(plan.depths):
        tree_rng = tree_root.keyed(tree_index)
        split_feats, thresholds = sample_odt_splits_batch(
            x,
            int(depth),
            tree_rng.keyed("splits").torch_generator,
            feature_probs=probs,
        )
        levels.append(
            _SampledTreeLevelParams(
                split_feats=split_feats,
                thresholds=thresholds,
                leaf_values=sample_noise_from_spec(
                    (int(x.shape[0]), 2 ** int(depth), int(out_dim)),
                    generator=tree_rng.keyed("leaf_values").torch_generator,
                    device=str(x.device),
                    noise_spec=noise_spec,
                ),
            )
        )
    return _SampledTreeFunctionParams(levels=tuple(levels))


def _apply_sampled_tree_batch(
    x: torch.Tensor,
    params: _SampledTreeFunctionParams,
) -> torch.Tensor:
    out_dim = int(params.levels[0].leaf_values.shape[2]) if params.levels else 0
    outputs = torch.zeros((int(x.shape[0]), int(x.shape[1]), out_dim), device=x.device)
    for level in params.levels:
        leaf_idx = compute_odt_leaf_indices_batch(x, level.split_feats, level.thresholds)
        outputs += torch.gather(
            level.leaf_values.to(device=x.device, dtype=x.dtype),
            1,
            leaf_idx.unsqueeze(-1).expand(-1, -1, out_dim),
        )
    return outputs / float(max(1, len(params.levels)))


def _apply_tree_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: TreeFunctionPlan,
    *,
    out_dim: int,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    return _apply_sampled_tree_batch(
        x,
        _sample_tree_function_params(
            x,
            rng,
            plan,
            out_dim=out_dim,
            noise_spec=noise_spec,
        ),
    )


def _sample_discretization_function_params(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: DiscretizationFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> _SampledDiscretizationFunctionParams:
    n_centers = min(int(plan.n_centers), int(x.shape[1]))
    center_index = rng.keyed("center_index").randperm_indices(
        length=int(x.shape[1]),
        sample_size=n_centers,
    )
    gathered = torch.gather(
        x,
        1,
        center_index.unsqueeze(-1).expand(-1, -1, int(x.shape[2])),
    )
    return _SampledDiscretizationFunctionParams(
        center_index=center_index,
        lp_norm=rng.keyed("lp_norm").log_uniform((rng.batch_size,), low=0.5, high=4.0),
        linear=_sample_linear_function_params(
            gathered,
            rng.keyed("linear"),
            LinearFunctionPlan(matrix=plan.linear_matrix),
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        ),
    )


def _apply_sampled_discretization_batch(
    x: torch.Tensor,
    params: _SampledDiscretizationFunctionParams,
) -> torch.Tensor:
    centers = torch.gather(
        x,
        1,
        params.center_index.to(device=x.device, dtype=torch.long)
        .unsqueeze(-1)
        .expand(-1, -1, int(x.shape[2])),
    )
    nearest = _nearest_lp_center_indices(
        x,
        centers,
        p=params.lp_norm.to(device=x.device, dtype=x.dtype),
    )
    gathered = torch.gather(
        centers,
        1,
        nearest.unsqueeze(-1).expand(-1, -1, int(x.shape[2])),
    )
    return _apply_sampled_linear_batch(gathered, params.linear)


def _apply_discretization_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: DiscretizationFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    return _apply_sampled_discretization_batch(
        x,
        _sample_discretization_function_params(
            x,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        ),
    )


def _sample_radial_ha_batch(
    rng: FixedLayoutBatchRng,
    *,
    n: int,
    a: torch.Tensor,
) -> torch.Tensor:
    u = rng.keyed("u").uniform((rng.batch_size, n), low=0.0, high=1.0)
    return torch.pow(1.0 - u, 1.0 / (1.0 - a.view(-1, 1))) - 1.0


def _apply_gp_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: GpFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    batch_size, _, din = x.shape
    p = 256
    a = rng.keyed("a").log_uniform((rng.batch_size,), low=2.0, high=20.0)
    input_projection: torch.Tensor | None = None

    if str(plan.branch_kind) == "ha":
        r = _sample_radial_ha_batch(rng.keyed("ha_radius"), n=p * din, a=a).view(batch_size, p, din)
        signs = torch.where(
            rng.keyed("ha_sign").uniform((batch_size, p, din), low=0.0, high=1.0) < 0.5,
            -1.0,
            1.0,
        )
        omega = r * signs
    else:
        z = sample_noise_from_spec(
            (batch_size, p, din),
            generator=rng.keyed("projected_direction").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        z = z / torch.clamp(torch.norm(z, dim=2, keepdim=True), min=1e-6)
        r = _sample_radial_ha_batch(rng.keyed("projected_radius"), n=p, a=a)
        omega = z * r.unsqueeze(2)
        weights = _sample_random_weights_batch(
            rng.keyed("weights"),
            dim=din,
            correlation_name="gp_projected_weights_decay",
            sigma_multiplier=float(noise_sigma_multiplier),
            noise_spec=noise_spec,
        )
        alpha = rng.keyed("alpha").log_uniform((batch_size,), low=0.5, high=10.0)
        a_mat = sample_noise_from_spec(
            (batch_size, din, din),
            generator=rng.keyed("matrix").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        input_projection = alpha.view(-1, 1, 1) * (weights.unsqueeze(2) * a_mat)

    variant = str(plan.variant)
    if variant == "multiscale":
        low_scale = rng.keyed("multiscale_low").log_uniform((batch_size,), low=0.35, high=1.0)
        high_scale = rng.keyed("multiscale_high").log_uniform((batch_size,), low=1.5, high=6.0)
        split = p // 2
        feature_scale = torch.cat(
            [
                low_scale.view(-1, 1).expand(-1, split),
                high_scale.view(-1, 1).expand(-1, p - split),
            ],
            dim=1,
        ).to(device=rng.device, dtype=x.dtype)
        omega = omega * feature_scale.unsqueeze(2)
    elif variant not in {"standard", "periodic"}:
        raise ValueError(f"Unknown GP variant: {plan.variant!r}")

    return _apply_sampled_gp_batch(
        x,
        plan,
        _SampledGpFunctionParams(
            input_projection=input_projection,
            omega=omega,
            phase_bias=rng.keyed("phase").uniform((batch_size, p), low=0.0, high=2.0 * math.pi),
            harmonics=(
                None
                if variant != "periodic"
                else rng.keyed("periodic_harmonics")
                .randint(1, 6, (batch_size, p))
                .to(
                    device=rng.device,
                    dtype=x.dtype,
                )
            ),
            output_matrix=sample_noise_from_spec(
                (batch_size, out_dim, p),
                generator=rng.keyed("output_matrix").torch_generator,
                device=rng.device,
                noise_spec=noise_spec,
            ),
        ),
    )


def _apply_sampled_gp_batch(
    x: torch.Tensor,
    plan: GpFunctionPlan,
    params: _SampledGpFunctionParams,
) -> torch.Tensor:
    x_proj = x
    if params.input_projection is not None:
        x_proj = torch.einsum("bni,bij->bnj", x, params.input_projection.transpose(1, 2))
    phase_logits = torch.einsum(
        "bnd,bpd->bnp",
        x_proj,
        params.omega.to(device=x.device, dtype=x.dtype),
    )
    phase_bias = params.phase_bias.to(device=x.device, dtype=x.dtype)
    if str(plan.variant) == "periodic":
        if params.harmonics is None:
            raise ValueError("Periodic GP params are missing harmonics.")
        phase_logits = (
            params.harmonics.to(device=x.device, dtype=x.dtype).unsqueeze(1) * phase_logits
        )
        phi = (
            torch.cos(phase_logits + phase_bias.unsqueeze(1))
            + torch.sin(phase_logits - phase_bias.unsqueeze(1))
        ) / math.sqrt(2.0)
    else:
        phi = torch.cos(phase_logits + phase_bias.unsqueeze(1))
    return torch.einsum(
        "bnp,bop->bno",
        phi,
        params.output_matrix.to(device=x.device, dtype=x.dtype),
    ) / math.sqrt(float(params.output_matrix.shape[-1]))


def _apply_em_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: EmFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    m_val = max(2, int(plan.m_val))
    return _apply_sampled_em_batch(
        x,
        _SampledEmFunctionParams(
            base_index=rng.keyed("base_index").randint(0, x.shape[1], (rng.batch_size, m_val)),
            center_noise=sample_noise_from_spec(
                (rng.batch_size, m_val, x.shape[2]),
                generator=rng.keyed("center_noise").torch_generator,
                device=rng.device,
                noise_spec=noise_spec,
            ),
            sigma=torch.exp(
                sample_noise_from_spec(
                    (rng.batch_size, m_val),
                    generator=rng.keyed("sigma").torch_generator,
                    device=rng.device,
                    noise_spec=noise_spec,
                    scale_multiplier=0.1,
                )
            ),
            p_val=rng.keyed("p_val").log_uniform((rng.batch_size,), low=1.0, high=4.0),
            q_val=rng.keyed("q_val").log_uniform((rng.batch_size,), low=1.0, high=2.0),
            linear=_sample_linear_function_params(
                torch.empty((x.shape[0], x.shape[1], m_val), device=x.device, dtype=x.dtype),
                rng.keyed("linear"),
                LinearFunctionPlan(matrix=plan.linear_matrix),
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            ),
        ),
    )


def _apply_sampled_em_batch(
    x: torch.Tensor,
    params: _SampledEmFunctionParams,
) -> torch.Tensor:
    centers = torch.gather(
        x,
        1,
        params.base_index.to(device=x.device, dtype=torch.long)
        .unsqueeze(-1)
        .expand(-1, -1, x.shape[2]),
    )
    centers = centers + params.center_noise.to(device=x.device, dtype=x.dtype)
    sigma = params.sigma.to(device=x.device, dtype=x.dtype)
    dist_p = _lp_distances_to_centers(
        x,
        centers,
        p=params.p_val.to(device=x.device, dtype=x.dtype),
        take_root=True,
    )
    logits = -0.5 * torch.log(2.0 * math.pi * sigma**2).unsqueeze(1) - torch.pow(
        dist_p / torch.clamp(sigma.unsqueeze(1), min=1e-6),
        params.q_val.to(device=x.device, dtype=x.dtype).view(-1, 1, 1),
    )
    probs = torch.softmax(logits, dim=2)
    return _apply_sampled_linear_batch(probs, params.linear)


def _concat_sampled_leaf_function_params(
    params_list: list[SampledLeafFunctionParams],
) -> SampledLeafFunctionParams:
    if not params_list:
        raise ValueError("params_list must be non-empty.")
    first = params_list[0]
    if isinstance(first, _SampledLinearFunctionParams):
        linear_params = cast(list[_SampledLinearFunctionParams], params_list)
        return _SampledLinearFunctionParams(
            matrix=torch.cat([params.matrix for params in linear_params], dim=0)
        )
    if isinstance(first, _SampledQuadraticFunctionParams):
        quadratic_params = cast(list[_SampledQuadraticFunctionParams], params_list)
        return _SampledQuadraticFunctionParams(
            feature_subset=_concat_optional_tensor(
                [params.feature_subset for params in quadratic_params]
            ),
            matrix=torch.cat([params.matrix for params in quadratic_params], dim=0),
        )
    if isinstance(first, _SampledNeuralNetFunctionParams):
        neural_params = cast(list[_SampledNeuralNetFunctionParams], params_list)
        return _SampledNeuralNetFunctionParams(
            input_activation=(
                None
                if first.input_activation is None
                else _concat_activation_plan_params(
                    [
                        params.input_activation
                        for params in neural_params
                        if params.input_activation is not None
                    ]
                )
            ),
            layer_matrices=tuple(
                torch.cat(
                    [params.layer_matrices[layer_index] for params in neural_params],
                    dim=0,
                )
                for layer_index in range(len(first.layer_matrices))
            ),
            hidden_activations=tuple(
                _concat_activation_plan_params(
                    [params.hidden_activations[layer_index] for params in neural_params]
                )
                for layer_index in range(len(first.hidden_activations))
            ),
            output_activation=(
                None
                if first.output_activation is None
                else _concat_activation_plan_params(
                    [
                        params.output_activation
                        for params in neural_params
                        if params.output_activation is not None
                    ]
                )
            ),
        )
    if isinstance(first, _SampledTreeFunctionParams):
        tree_params = cast(list[_SampledTreeFunctionParams], params_list)
        return _SampledTreeFunctionParams(
            levels=tuple(
                _SampledTreeLevelParams(
                    split_feats=torch.cat(
                        [params.levels[level_index].split_feats for params in tree_params],
                        dim=0,
                    ),
                    thresholds=torch.cat(
                        [params.levels[level_index].thresholds for params in tree_params],
                        dim=0,
                    ),
                    leaf_values=torch.cat(
                        [params.levels[level_index].leaf_values for params in tree_params],
                        dim=0,
                    ),
                )
                for level_index in range(len(first.levels))
            )
        )
    if isinstance(first, _SampledDiscretizationFunctionParams):
        discretization_params = cast(list[_SampledDiscretizationFunctionParams], params_list)
        return _SampledDiscretizationFunctionParams(
            center_index=torch.cat(
                [params.center_index for params in discretization_params], dim=0
            ),
            lp_norm=torch.cat([params.lp_norm for params in discretization_params], dim=0),
            linear=_SampledLinearFunctionParams(
                matrix=torch.cat(
                    [params.linear.matrix for params in discretization_params],
                    dim=0,
                )
            ),
        )
    if isinstance(first, _SampledGpFunctionParams):
        gp_params = cast(list[_SampledGpFunctionParams], params_list)
        return _SampledGpFunctionParams(
            input_projection=_concat_optional_tensor(
                [params.input_projection for params in gp_params]
            ),
            omega=torch.cat([params.omega for params in gp_params], dim=0),
            phase_bias=torch.cat([params.phase_bias for params in gp_params], dim=0),
            harmonics=_concat_optional_tensor([params.harmonics for params in gp_params]),
            output_matrix=torch.cat([params.output_matrix for params in gp_params], dim=0),
        )
    if isinstance(first, _SampledEmFunctionParams):
        em_params = cast(list[_SampledEmFunctionParams], params_list)
        return _SampledEmFunctionParams(
            base_index=torch.cat([params.base_index for params in em_params], dim=0),
            center_noise=torch.cat([params.center_noise for params in em_params], dim=0),
            sigma=torch.cat([params.sigma for params in em_params], dim=0),
            p_val=torch.cat([params.p_val for params in em_params], dim=0),
            q_val=torch.cat([params.q_val for params in em_params], dim=0),
            linear=_SampledLinearFunctionParams(
                matrix=torch.cat([params.linear.matrix for params in em_params], dim=0)
            ),
        )
    raise ValueError(f"Unsupported sampled leaf-function params type: {type(first)!r}")


def _sample_leaf_function_params_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: LinearFunctionPlan
    | QuadraticFunctionPlan
    | NeuralNetFunctionPlan
    | TreeFunctionPlan
    | DiscretizationFunctionPlan
    | GpFunctionPlan
    | EmFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> SampledLeafFunctionParams:
    if isinstance(plan, LinearFunctionPlan):
        return _sample_linear_function_params(
            x,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, QuadraticFunctionPlan):
        return _sample_quadratic_function_params(
            x,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, NeuralNetFunctionPlan):
        return _sample_nn_function_params(
            x,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, TreeFunctionPlan):
        return _sample_tree_function_params(
            x,
            rng,
            plan,
            out_dim=out_dim,
            noise_spec=noise_spec,
        )
    if isinstance(plan, DiscretizationFunctionPlan):
        return _sample_discretization_function_params(
            x,
            rng,
            plan,
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(plan, GpFunctionPlan):
        batch_size, _, din = x.shape
        p = 256
        a = rng.keyed("a").log_uniform((rng.batch_size,), low=2.0, high=20.0)
        input_projection: torch.Tensor | None = None
        if str(plan.branch_kind) == "ha":
            r = _sample_radial_ha_batch(rng.keyed("ha_radius"), n=p * din, a=a).view(
                batch_size, p, din
            )
            signs = torch.where(
                rng.keyed("ha_sign").uniform((batch_size, p, din), low=0.0, high=1.0) < 0.5,
                -1.0,
                1.0,
            )
            omega = r * signs
        else:
            z = sample_noise_from_spec(
                (batch_size, p, din),
                generator=rng.keyed("projected_direction").torch_generator,
                device=rng.device,
                noise_spec=noise_spec,
            )
            z = z / torch.clamp(torch.norm(z, dim=2, keepdim=True), min=1e-6)
            r = _sample_radial_ha_batch(rng.keyed("projected_radius"), n=p, a=a)
            omega = z * r.unsqueeze(2)
            weights = _sample_random_weights_batch(
                rng.keyed("weights"),
                dim=din,
                correlation_name="gp_projected_weights_decay",
                sigma_multiplier=float(noise_sigma_multiplier),
                noise_spec=noise_spec,
            )
            alpha = rng.keyed("alpha").log_uniform((batch_size,), low=0.5, high=10.0)
            a_mat = sample_noise_from_spec(
                (batch_size, din, din),
                generator=rng.keyed("matrix").torch_generator,
                device=rng.device,
                noise_spec=noise_spec,
            )
            input_projection = alpha.view(-1, 1, 1) * (weights.unsqueeze(2) * a_mat)
        variant = str(plan.variant)
        if variant == "multiscale":
            low_scale = rng.keyed("multiscale_low").log_uniform((batch_size,), low=0.35, high=1.0)
            high_scale = rng.keyed("multiscale_high").log_uniform((batch_size,), low=1.5, high=6.0)
            split = p // 2
            feature_scale = torch.cat(
                [
                    low_scale.view(-1, 1).expand(-1, split),
                    high_scale.view(-1, 1).expand(-1, p - split),
                ],
                dim=1,
            ).to(device=rng.device, dtype=x.dtype)
            omega = omega * feature_scale.unsqueeze(2)
        elif variant not in {"standard", "periodic"}:
            raise ValueError(f"Unknown GP variant: {plan.variant!r}")
        return _SampledGpFunctionParams(
            input_projection=input_projection,
            omega=omega,
            phase_bias=rng.keyed("phase").uniform((batch_size, p), low=0.0, high=2.0 * math.pi),
            harmonics=(
                None
                if variant != "periodic"
                else rng.keyed("periodic_harmonics")
                .randint(1, 6, (batch_size, p))
                .to(
                    device=rng.device,
                    dtype=x.dtype,
                )
            ),
            output_matrix=sample_noise_from_spec(
                (batch_size, out_dim, p),
                generator=rng.keyed("output_matrix").torch_generator,
                device=rng.device,
                noise_spec=noise_spec,
            ),
        )
    if isinstance(plan, EmFunctionPlan):
        m_val = max(2, int(plan.m_val))
        return _SampledEmFunctionParams(
            base_index=rng.keyed("base_index").randint(0, x.shape[1], (rng.batch_size, m_val)),
            center_noise=sample_noise_from_spec(
                (rng.batch_size, m_val, x.shape[2]),
                generator=rng.keyed("center_noise").torch_generator,
                device=rng.device,
                noise_spec=noise_spec,
            ),
            sigma=torch.exp(
                sample_noise_from_spec(
                    (rng.batch_size, m_val),
                    generator=rng.keyed("sigma").torch_generator,
                    device=rng.device,
                    noise_spec=noise_spec,
                    scale_multiplier=0.1,
                )
            ),
            p_val=rng.keyed("p_val").log_uniform((rng.batch_size,), low=1.0, high=4.0),
            q_val=rng.keyed("q_val").log_uniform((rng.batch_size,), low=1.0, high=2.0),
            linear=_sample_linear_function_params(
                torch.empty((x.shape[0], x.shape[1], m_val), device=x.device, dtype=x.dtype),
                rng.keyed("linear"),
                LinearFunctionPlan(matrix=plan.linear_matrix),
                out_dim=out_dim,
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            ),
        )
    raise ValueError(f"Unsupported leaf function plan: {plan!r}")


def _apply_sampled_leaf_function_batch(
    x: torch.Tensor,
    plan: LinearFunctionPlan
    | QuadraticFunctionPlan
    | NeuralNetFunctionPlan
    | TreeFunctionPlan
    | DiscretizationFunctionPlan
    | GpFunctionPlan
    | EmFunctionPlan,
    params: SampledLeafFunctionParams,
) -> torch.Tensor:
    if isinstance(plan, LinearFunctionPlan):
        if not isinstance(params, _SampledLinearFunctionParams):
            raise ValueError("Linear function params type mismatch.")
        return _apply_sampled_linear_batch(x, params)
    if isinstance(plan, QuadraticFunctionPlan):
        if not isinstance(params, _SampledQuadraticFunctionParams):
            raise ValueError("Quadratic function params type mismatch.")
        return _apply_sampled_quadratic_batch(x, params)
    if isinstance(plan, NeuralNetFunctionPlan):
        if not isinstance(params, _SampledNeuralNetFunctionParams):
            raise ValueError("NN function params type mismatch.")
        return _apply_sampled_nn_batch(x, plan, params)
    if isinstance(plan, TreeFunctionPlan):
        if not isinstance(params, _SampledTreeFunctionParams):
            raise ValueError("Tree function params type mismatch.")
        return _apply_sampled_tree_batch(x, params)
    if isinstance(plan, DiscretizationFunctionPlan):
        if not isinstance(params, _SampledDiscretizationFunctionParams):
            raise ValueError("Discretization function params type mismatch.")
        return _apply_sampled_discretization_batch(x, params)
    if isinstance(plan, GpFunctionPlan):
        if not isinstance(params, _SampledGpFunctionParams):
            raise ValueError("GP function params type mismatch.")
        return _apply_sampled_gp_batch(x, plan, params)
    if not isinstance(params, _SampledEmFunctionParams):
        raise ValueError("EM function params type mismatch.")
    return _apply_sampled_em_batch(x, params)


def _apply_linear_pair_batch_from_branch_inputs(
    branch_inputs: torch.Tensor,
    *,
    rngs: tuple[FixedLayoutBatchRng, FixedLayoutBatchRng],
    matrix_plans: tuple[FixedLayoutMatrixPlan, FixedLayoutMatrixPlan],
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    """Apply two matrix-backed linear branches with one fused output projection."""

    matrices = torch.stack(
        [
            _sample_random_matrix_from_plan_batch(
                matrix_plan,
                out_dim=int(out_dim),
                in_dim=int(branch_inputs.shape[3]),
                rng=rng.keyed("matrix"),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
            )
            for rng, matrix_plan in zip(rngs, matrix_plans, strict=True)
        ],
        dim=1,
    )
    return torch.einsum("bkni,bkoi->bkno", branch_inputs, matrices)


def _apply_linear_pair_batch(
    x: torch.Tensor,
    *,
    rngs: tuple[FixedLayoutBatchRng, FixedLayoutBatchRng],
    plans: tuple[LinearFunctionPlan, LinearFunctionPlan],
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    branch_inputs = torch.stack((x, x), dim=1)
    return _apply_linear_pair_batch_from_branch_inputs(
        branch_inputs,
        rngs=rngs,
        matrix_plans=(plans[0].matrix, plans[1].matrix),
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def _apply_quadratic_pair_batch(
    x: torch.Tensor,
    *,
    rngs: tuple[FixedLayoutBatchRng, FixedLayoutBatchRng],
    plans: tuple[QuadraticFunctionPlan, QuadraticFunctionPlan],
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    feature_cap = min(int(x.shape[2]), 20)
    augmented_inputs: list[torch.Tensor] = []
    matrices: list[torch.Tensor] = []
    for rng, plan in zip(rngs, plans, strict=True):
        if int(x.shape[2]) > feature_cap:
            indices = rng.keyed("feature_subset").randperm_indices(
                length=int(x.shape[2]),
                sample_size=feature_cap,
            )
            x_sub = torch.gather(
                x,
                2,
                indices.unsqueeze(1).expand(-1, x.shape[1], -1),
            )
        else:
            x_sub = x
        ones = torch.ones((x_sub.shape[0], x_sub.shape[1], 1), device=x.device, dtype=x_sub.dtype)
        x_aug = torch.cat([x_sub, ones], dim=2)
        augmented_inputs.append(x_aug)
        matrices.append(
            _sample_random_matrix_from_plan_batch(
                plan.matrix,
                out_dim=int(x_aug.shape[2]),
                in_dim=int(x_aug.shape[2]),
                rng=rng.keyed("matrix"),
                noise_sigma_multiplier=noise_sigma_multiplier,
                noise_spec=noise_spec,
                matrix_count=int(out_dim),
            )
        )
    augmented_inputs_stacked = torch.stack(augmented_inputs, dim=1)
    matrices_stacked = torch.stack(matrices, dim=1)
    return torch.einsum(
        "bkni,bkoij,bknj->bkno",
        augmented_inputs_stacked,
        matrices_stacked,
        augmented_inputs_stacked,
    )


def _apply_discretization_pair_batch(
    x: torch.Tensor,
    *,
    rngs: tuple[FixedLayoutBatchRng, FixedLayoutBatchRng],
    plans: tuple[DiscretizationFunctionPlan, DiscretizationFunctionPlan],
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    gathered_inputs: list[torch.Tensor] = []
    linear_rngs: list[FixedLayoutBatchRng] = []
    linear_matrix_plans: list[FixedLayoutMatrixPlan] = []
    for rng, plan in zip(rngs, plans, strict=True):
        n_centers = min(int(plan.n_centers), int(x.shape[1]))
        center_idx = rng.keyed("center_index").randperm_indices(
            length=int(x.shape[1]),
            sample_size=n_centers,
        )
        centers = torch.gather(
            x,
            1,
            center_idx.unsqueeze(-1).expand(-1, -1, x.shape[2]),
        )
        p = rng.keyed("lp_norm").log_uniform((rng.batch_size,), low=0.5, high=4.0)
        nearest = _nearest_lp_center_indices(x, centers, p=p)
        gathered_inputs.append(
            torch.gather(
                centers,
                1,
                nearest.unsqueeze(-1).expand(-1, -1, x.shape[2]),
            )
        )
        linear_rngs.append(rng.keyed("linear"))
        linear_matrix_plans.append(plan.linear_matrix)
    return _apply_linear_pair_batch_from_branch_inputs(
        torch.stack(gathered_inputs, dim=1),
        rngs=(linear_rngs[0], linear_rngs[1]),
        matrix_plans=(linear_matrix_plans[0], linear_matrix_plans[1]),
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def _apply_gp_pair_batch(
    x: torch.Tensor,
    *,
    rngs: tuple[FixedLayoutBatchRng, FixedLayoutBatchRng],
    plans: tuple[GpFunctionPlan, GpFunctionPlan],
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    batch_size, _, din = x.shape
    p = 256
    phis: list[torch.Tensor] = []
    output_matrices: list[torch.Tensor] = []
    for rng, plan in zip(rngs, plans, strict=True):
        a = rng.keyed("a").log_uniform((batch_size,), low=2.0, high=20.0)
        if str(plan.branch_kind) == "ha":
            r = _sample_radial_ha_batch(rng.keyed("ha_radius"), n=p * din, a=a).view(
                batch_size, p, din
            )
            signs = torch.where(
                rng.keyed("ha_sign").uniform((batch_size, p, din), low=0.0, high=1.0) < 0.5,
                -1.0,
                1.0,
            )
            omega = r * signs
            x_proj = x
        else:
            z = sample_noise_from_spec(
                (batch_size, p, din),
                generator=rng.keyed("projected_direction").torch_generator,
                device=rng.device,
                noise_spec=noise_spec,
            )
            z = z / torch.clamp(torch.norm(z, dim=2, keepdim=True), min=1e-6)
            r = _sample_radial_ha_batch(rng.keyed("projected_radius"), n=p, a=a)
            omega = z * r.unsqueeze(2)
            weights = _sample_random_weights_batch(
                rng.keyed("weights"),
                dim=din,
                correlation_name="gp_projected_weights_decay",
                sigma_multiplier=float(noise_sigma_multiplier),
                noise_spec=noise_spec,
            )
            alpha = rng.keyed("alpha").log_uniform((batch_size,), low=0.5, high=10.0)
            a_mat = sample_noise_from_spec(
                (batch_size, din, din),
                generator=rng.keyed("matrix").torch_generator,
                device=rng.device,
                noise_spec=noise_spec,
            )
            matrices = alpha.view(-1, 1, 1) * (weights.unsqueeze(2) * a_mat)
            x_proj = torch.einsum("bni,bij->bnj", x, matrices.transpose(1, 2))

        variant = str(plan.variant)
        if variant == "multiscale":
            low_scale = rng.keyed("multiscale_low").log_uniform((batch_size,), low=0.35, high=1.0)
            high_scale = rng.keyed("multiscale_high").log_uniform((batch_size,), low=1.5, high=6.0)
            split = p // 2
            feature_scale = torch.cat(
                [
                    low_scale.view(-1, 1).expand(-1, split),
                    high_scale.view(-1, 1).expand(-1, p - split),
                ],
                dim=1,
            ).to(device=rng.device, dtype=x.dtype)
            omega = omega * feature_scale.unsqueeze(2)
        elif variant not in {"standard", "periodic"}:
            raise ValueError(f"Unknown GP variant: {plan.variant!r}")

        b = rng.keyed("phase").uniform((batch_size, p), low=0.0, high=2.0 * math.pi)
        phase_logits = torch.einsum("bnd,bpd->bnp", x_proj, omega)
        if variant == "periodic":
            harmonics = (
                rng.keyed("periodic_harmonics")
                .randint(1, 6, (batch_size, p))
                .to(
                    device=rng.device,
                    dtype=x.dtype,
                )
            )
            phase_logits = harmonics.unsqueeze(1) * phase_logits
            phi = (
                torch.cos(phase_logits + b.unsqueeze(1)) + torch.sin(phase_logits - b.unsqueeze(1))
            ) / math.sqrt(2.0)
        else:
            phi = torch.cos(phase_logits + b.unsqueeze(1))
        phis.append(phi)
        output_matrices.append(
            sample_noise_from_spec(
                (batch_size, out_dim, p),
                generator=rng.keyed("output_matrix").torch_generator,
                device=rng.device,
                noise_spec=noise_spec,
            )
        )
    return torch.einsum(
        "bknp,bkop->bkno", torch.stack(phis, dim=1), torch.stack(output_matrices, dim=1)
    ) / math.sqrt(float(p))


def _apply_em_pair_batch(
    x: torch.Tensor,
    *,
    rngs: tuple[FixedLayoutBatchRng, FixedLayoutBatchRng],
    plans: tuple[EmFunctionPlan, EmFunctionPlan],
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    branch_probs: list[torch.Tensor] = []
    linear_rngs: list[FixedLayoutBatchRng] = []
    linear_matrix_plans: list[FixedLayoutMatrixPlan] = []
    for rng, plan in zip(rngs, plans, strict=True):
        m_val = max(2, int(plan.m_val))
        base_idx = rng.keyed("base_index").randint(0, x.shape[1], (rng.batch_size, m_val))
        centers = torch.gather(
            x,
            1,
            base_idx.unsqueeze(-1).expand(-1, -1, x.shape[2]),
        )
        centers = centers + sample_noise_from_spec(
            (rng.batch_size, m_val, x.shape[2]),
            generator=rng.keyed("center_noise").torch_generator,
            device=rng.device,
            noise_spec=noise_spec,
        )
        sigma = torch.exp(
            sample_noise_from_spec(
                (rng.batch_size, m_val),
                generator=rng.keyed("sigma").torch_generator,
                device=rng.device,
                noise_spec=noise_spec,
                scale_multiplier=0.1,
            )
        )
        p_val = rng.keyed("p_val").log_uniform((rng.batch_size,), low=1.0, high=4.0)
        q_val = rng.keyed("q_val").log_uniform((rng.batch_size,), low=1.0, high=2.0)
        dist_p = _lp_distances_to_centers(
            x,
            centers,
            p=p_val,
            take_root=True,
        )
        logits = -0.5 * torch.log(2.0 * math.pi * sigma**2).unsqueeze(1) - torch.pow(
            dist_p / torch.clamp(sigma.unsqueeze(1), min=1e-6),
            q_val.view(-1, 1, 1),
        )
        branch_probs.append(torch.softmax(logits, dim=2))
        linear_rngs.append(rng.keyed("linear"))
        linear_matrix_plans.append(plan.linear_matrix)
    return _apply_linear_pair_batch_from_branch_inputs(
        torch.stack(branch_probs, dim=1),
        rngs=(linear_rngs[0], linear_rngs[1]),
        matrix_plans=(linear_matrix_plans[0], linear_matrix_plans[1]),
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


def _apply_leaf_pair_batch(
    x: torch.Tensor,
    *,
    rngs: tuple[FixedLayoutBatchRng, FixedLayoutBatchRng],
    plans: tuple[
        LinearFunctionPlan
        | QuadraticFunctionPlan
        | DiscretizationFunctionPlan
        | GpFunctionPlan
        | EmFunctionPlan,
        LinearFunctionPlan
        | QuadraticFunctionPlan
        | DiscretizationFunctionPlan
        | GpFunctionPlan
        | EmFunctionPlan,
    ],
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    """Apply one exact paired leaf-family batch and return `[B, 2, rows, out_dim]`."""

    lhs_plan, rhs_plan = plans
    if isinstance(lhs_plan, LinearFunctionPlan) and isinstance(rhs_plan, LinearFunctionPlan):
        return _apply_linear_pair_batch(
            x,
            rngs=rngs,
            plans=(lhs_plan, rhs_plan),
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(lhs_plan, QuadraticFunctionPlan) and isinstance(rhs_plan, QuadraticFunctionPlan):
        return _apply_quadratic_pair_batch(
            x,
            rngs=rngs,
            plans=(lhs_plan, rhs_plan),
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(lhs_plan, DiscretizationFunctionPlan) and isinstance(
        rhs_plan, DiscretizationFunctionPlan
    ):
        return _apply_discretization_pair_batch(
            x,
            rngs=rngs,
            plans=(lhs_plan, rhs_plan),
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(lhs_plan, GpFunctionPlan) and isinstance(rhs_plan, GpFunctionPlan):
        return _apply_gp_pair_batch(
            x,
            rngs=rngs,
            plans=(lhs_plan, rhs_plan),
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    if isinstance(lhs_plan, EmFunctionPlan) and isinstance(rhs_plan, EmFunctionPlan):
        return _apply_em_pair_batch(
            x,
            rngs=rngs,
            plans=(lhs_plan, rhs_plan),
            out_dim=out_dim,
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
    raise ValueError("Unsupported paired fixed-layout leaf family combination.")


__all__ = [
    "_apply_activation_plan",
    "_apply_discretization_batch",
    "_apply_discretization_pair_batch",
    "_apply_em_batch",
    "_apply_em_pair_batch",
    "_apply_gp_batch",
    "_apply_gp_pair_batch",
    "_apply_leaf_pair_batch",
    "_apply_linear_batch",
    "_apply_linear_pair_batch",
    "_apply_nn_batch",
    "_apply_piecewise_gate_with_params",
    "_apply_quadratic_batch",
    "_apply_quadratic_pair_batch",
    "_apply_tree_batch",
    "_base_matrix_plan",
    "_concat_piecewise_gate_params",
    "_sample_piecewise_gate_params",
    "_sample_radial_ha_batch",
    "_sample_random_matrix_from_plan_batch",
    "_sample_random_points_batch",
    "_sample_unit_ball_batch",
]
