"""Random function families."""

from __future__ import annotations

import torch

from dagzoo.core.execution_semantics import (
    sample_function_family,
    sample_function_plan,
    sample_function_plan_for_family,
)
from dagzoo.core.fixed_layout.batched import FixedLayoutBatchRng, apply_function_plan_batch
from dagzoo.core.layout_types import MechanismFamily
from dagzoo.math import sanitize_and_standardize
from dagzoo.rng import KeyedRng, keyed_rng_from_generator
from dagzoo.sampling.noise import NoiseSamplingSpec

# Retained as a stable monkeypatch point for diagnostics tests and audit tools.
_sample_function_family = sample_function_family


def apply_random_function(
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    out_dim: int | None = None,
    function_type: MechanismFamily | None = None,
    mechanism_logit_tilt: float = 0.0,
    function_family_mix: dict[MechanismFamily, float] | None = None,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
    _keyed_root: KeyedRng | None = None,
) -> torch.Tensor:
    """Apply one sampled typed function plan to `x` in torch."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    y = sanitize_and_standardize(y)

    dout = out_dim if out_dim is not None else y.shape[1]
    root = _keyed_root or keyed_rng_from_generator(generator, "apply_random_function")
    sampling_device = str(generator.device)

    if function_type is None:
        plan = sample_function_plan(
            keyed_rng=root.keyed("plan"),
            input_dim=int(y.shape[1]),
            out_dim=int(dout),
            mechanism_logit_tilt=mechanism_logit_tilt,
            function_family_mix=function_family_mix,
            device=sampling_device,
        )
    else:
        if function_family_mix is not None and function_type not in function_family_mix:
            raise ValueError(
                f"Mechanism family '{function_type}' is not enabled by mechanism.function_family_mix."
            )
        try:
            plan = sample_function_plan_for_family(
                keyed_rng=root.keyed("plan"),
                family=function_type,
                input_dim=int(y.shape[1]),
                out_dim=int(dout),
                mechanism_logit_tilt=mechanism_logit_tilt,
                function_family_mix=function_family_mix,
                device=sampling_device,
            )
        except ValueError as exc:
            if "Unsupported mechanism family" in str(exc):
                raise ValueError(f"Unknown random function family: {function_type}") from exc
            raise

    rng = FixedLayoutBatchRng.from_keyed_rng(
        root.keyed("execution"),
        batch_size=1,
        device=str(y.device),
    )
    out = apply_function_plan_batch(
        y.unsqueeze(0),
        rng,
        plan,
        out_dim=int(dout),
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
        standardize_input=False,
    )
    return out.squeeze(0)
