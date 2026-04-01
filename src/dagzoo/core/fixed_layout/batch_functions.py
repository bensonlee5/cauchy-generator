"""Function-family execution helpers for fixed-layout batched generation."""

from __future__ import annotations

import math

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
    QuadraticFunctionPlan,
    SingularValuesMatrixPlan,
    TreeFunctionPlan,
    WeightsMatrixPlan,
)


def _apply_activation_plan(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: FixedLayoutActivationPlan,
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
        y = _batch_standardize(y)
        a = rng.keyed("standardize_scale").log_uniform((y.shape[0],), low=1.0, high=10.0)
        row_idx = rng.keyed("standardize_row_index").randint(0, y.shape[1], (y.shape[0],))
        offsets = y[torch.arange(y.shape[0], device=y.device), row_idx].unsqueeze(1)
        y = a.view(-1, 1, 1) * (y - offsets)

    if isinstance(plan, ParametricActivationPlan):
        kind = str(plan.kind)
        if kind == "relu_pow":
            q = rng.keyed("relu_pow").log_uniform(leading_shape, low=0.1, high=10.0)
            y = torch.pow(torch.clamp(y, min=0.0), q.reshape(*leading_shape, 1, 1))
        elif kind == "signed_pow":
            q = rng.keyed("signed_pow").log_uniform(leading_shape, low=0.1, high=10.0)
            y = torch.sign(y) * torch.pow(torch.abs(y), q.reshape(*leading_shape, 1, 1))
        elif kind == "inv_pow":
            q = rng.keyed("inv_pow").log_uniform(leading_shape, low=0.1, high=10.0)
            y = torch.pow(torch.abs(y) + 1e-3, -q.reshape(*leading_shape, 1, 1))
        elif kind == "poly":
            if plan.poly_power is None:
                raise ValueError("poly activation plan requires poly_power.")
            y = torch.pow(y, float(int(plan.poly_power)))
        elif kind == "gumbel_softmax":
            if plan.temperature is None:
                raise ValueError("gumbel_softmax activation plan requires temperature.")
            uniform_noise = rng.keyed("gumbel_softmax").uniform(
                y.shape,
                low=1e-6,
                high=1.0 - 1e-6,
            )
            y = activations_module._gumbel_softmax_activation(
                y,
                temperature=float(plan.temperature),
                uniform_noise=uniform_noise,
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
        q_low = 0.1 / math.log(int(in_dim) + 1.0)
        shared_q = rng.keyed("shared_q").log_uniform(
            (rng.batch_size, *leading_shape), low=q_low, high=6.0
        )
        shared_sigma = rng.keyed("shared_sigma").log_uniform(
            (rng.batch_size, *leading_shape), low=1e-4, high=10.0
        )
        rows = _sample_random_weights_batch(
            rng.keyed("row_weights"),
            dim=int(in_dim),
            leading_shape=(*leading_shape, int(out_dim)),
            parameter_shape=leading_shape,
            sigma_multiplier=float(noise_sigma_multiplier),
            noise_spec=noise_spec,
            q=shared_q,
            sigma=shared_sigma,
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
        gamma = rng.keyed("gamma").log_uniform((rng.batch_size, *leading_shape), low=0.1, high=10.0)
        left = pts[..., : int(out_dim), :].unsqueeze(-2)
        right = pts[..., int(out_dim) :, :].unsqueeze(-3)
        dist = torch.norm(left - right, dim=-1)
        kernel = torch.exp(-gamma.unsqueeze(-1).unsqueeze(-1) * dist)
        sign = torch.where(
            rng.keyed("sign").uniform(shape, low=0.0, high=1.0) < 0.5,
            -1.0,
            1.0,
        )
        matrix = kernel * sign
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


def _apply_linear_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: LinearFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    matrices = _sample_random_matrix_from_plan_batch(
        plan.matrix,
        out_dim=int(out_dim),
        in_dim=int(x.shape[2]),
        rng=rng.keyed("matrix"),
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    return torch.einsum("bni,boi->bno", x, matrices)


def _apply_quadratic_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: QuadraticFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    feature_cap = min(int(x.shape[2]), 20)
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
    matrices = _sample_random_matrix_from_plan_batch(
        plan.matrix,
        out_dim=int(x_aug.shape[2]),
        in_dim=int(x_aug.shape[2]),
        rng=rng.keyed("matrix"),
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        matrix_count=int(out_dim),
    )
    return torch.einsum("bni,boij,bnj->bno", x_aug, matrices, x_aug)


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


def _apply_nn_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: NeuralNetFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    y = x
    if plan.input_activation is not None:
        y = _apply_activation_plan(
            y,
            rng.keyed("input_activation"),
            plan.input_activation,
            with_standardize=True,
        )

    hidden_width = max(1, int(plan.hidden_width))
    n_layers = max(1, int(plan.n_layers))
    layer_dims = [int(x.shape[2])]
    for _ in range(max(0, n_layers - 1)):
        layer_dims.append(hidden_width)
    layer_dims.append(int(out_dim))

    for layer_index, (din, dout) in enumerate(zip(layer_dims[:-1], layer_dims[1:], strict=True)):
        matrices = _sample_random_matrix_from_plan_batch(
            plan.layer_matrices[layer_index],
            out_dim=int(dout),
            in_dim=int(din),
            rng=rng.keyed("layer_matrix", layer_index),
            noise_sigma_multiplier=noise_sigma_multiplier,
            noise_spec=noise_spec,
        )
        y = torch.einsum("bni,boi->bno", y, matrices)
        if layer_index < len(layer_dims) - 2:
            y = _apply_activation_plan(
                y,
                rng.keyed("hidden_activation", layer_index),
                plan.hidden_activations[layer_index],
                with_standardize=True,
            )

    if plan.output_activation is not None:
        y = _apply_activation_plan(
            y,
            rng.keyed("output_activation"),
            plan.output_activation,
            with_standardize=True,
        )
    return y


def _apply_tree_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: TreeFunctionPlan,
    *,
    out_dim: int,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
    batch_size = int(x.shape[0])
    outputs = torch.zeros((batch_size, x.shape[1], out_dim), device=x.device, dtype=torch.float32)
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
    for tree_index, depth in enumerate(plan.depths):
        tree_rng = rng.keyed("tree", tree_index)
        split_dims, thresholds = sample_odt_splits_batch(
            x,
            int(depth),
            tree_rng.keyed("splits").torch_generator,
            feature_probs=probs,
        )
        leaf_idx = compute_odt_leaf_indices_batch(x, split_dims, thresholds)
        n_leaves = 2 ** int(depth)
        leaf_vals = sample_noise_from_spec(
            (batch_size, n_leaves, out_dim),
            generator=tree_rng.keyed("leaf_values").torch_generator,
            device=str(x.device),
            noise_spec=noise_spec,
        )
        outputs += torch.gather(
            leaf_vals,
            1,
            leaf_idx.unsqueeze(-1).expand(-1, -1, out_dim),
        )
    return outputs / float(max(1, int(plan.n_trees)))


def _apply_discretization_batch(
    x: torch.Tensor,
    rng: FixedLayoutBatchRng,
    plan: DiscretizationFunctionPlan,
    *,
    out_dim: int,
    noise_sigma_multiplier: float,
    noise_spec: NoiseSamplingSpec | None,
) -> torch.Tensor:
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
    gathered = torch.gather(
        centers,
        1,
        nearest.unsqueeze(-1).expand(-1, -1, x.shape[2]),
    )
    return _apply_linear_batch(
        gathered,
        rng.keyed("linear"),
        LinearFunctionPlan(matrix=plan.linear_matrix),
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
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

    if str(plan.branch_kind) == "ha":
        r = _sample_radial_ha_batch(rng.keyed("ha_radius"), n=p * din, a=a).view(batch_size, p, din)
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
    z_out = sample_noise_from_spec(
        (batch_size, out_dim, p),
        generator=rng.keyed("output_matrix").torch_generator,
        device=rng.device,
        noise_spec=noise_spec,
    )
    return torch.einsum("bnp,bop->bno", phi, z_out) / math.sqrt(float(p))


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
    probs = torch.softmax(logits, dim=2)
    return _apply_linear_batch(
        probs,
        rng.keyed("linear"),
        LinearFunctionPlan(matrix=plan.linear_matrix),
        out_dim=out_dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )


__all__ = [
    "_apply_activation_plan",
    "_apply_discretization_batch",
    "_apply_em_batch",
    "_apply_gp_batch",
    "_apply_linear_batch",
    "_apply_nn_batch",
    "_apply_quadratic_batch",
    "_apply_tree_batch",
    "_base_matrix_plan",
    "_sample_radial_ha_batch",
    "_sample_random_matrix_from_plan_batch",
    "_sample_random_points_batch",
    "_sample_unit_ball_batch",
]
