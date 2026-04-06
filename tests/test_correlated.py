"""Tests for sampling/correlated.py."""

import torch

import dagzoo.sampling.correlated as correlated_mod
from dagzoo.rng import KeyedRng
from dagzoo.sampling.correlated import CorrelatedSampler, correlated_sampling_scope


def test_sample_num_within_bounds() -> None:
    cs = CorrelatedSampler(KeyedRng(0), "cpu")
    for _ in range(50):
        v = cs.sample_num("x", 2.0, 10.0)
        assert 2.0 <= v <= 10.0


def test_sample_num_log_scale() -> None:
    cs = CorrelatedSampler(KeyedRng(1), "cpu")
    for _ in range(50):
        v = cs.sample_num("ls", 0.01, 100.0, log_scale=True)
        assert 0.01 <= v <= 100.0


def test_sample_num_as_int() -> None:
    cs = CorrelatedSampler(KeyedRng(2), "cpu")
    for _ in range(50):
        v = cs.sample_num("i", 1.0, 20.0, as_int=True)
        assert isinstance(v, int)
        assert 1 <= v <= 20


def test_sample_num_deterministic() -> None:
    vals_a = []
    vals_b = []
    for seed_vals, vals in [(0, vals_a), (0, vals_b)]:
        cs = CorrelatedSampler(KeyedRng(seed_vals), "cpu")
        for _ in range(10):
            vals.append(cs.sample_num("d", 0.0, 1.0))
    assert vals_a == vals_b


def test_sample_num_does_not_reseed_global_rng(monkeypatch) -> None:
    cs = CorrelatedSampler(KeyedRng(0), "cpu")

    def _manual_seed_forbidden(seed: int) -> torch.Generator:
        raise AssertionError(f"unexpected global reseed: {seed}")

    monkeypatch.setattr(torch, "manual_seed", _manual_seed_forbidden)
    _ = cs.sample_num("d", 0.0, 1.0)


def test_sample_category_deterministic() -> None:
    vals_a = []
    vals_b = []
    for seed_vals, vals in [(0, vals_a), (0, vals_b)]:
        cs = CorrelatedSampler(KeyedRng(seed_vals), "cpu")
        for _ in range(10):
            vals.append(cs.sample_category("cat", 5))
    assert vals_a == vals_b


def test_sample_category_valid_index() -> None:
    cs = CorrelatedSampler(KeyedRng(3), "cpu")
    n = 7
    for _ in range(50):
        idx = cs.sample_category("cat", n)
        assert isinstance(idx, int)
        assert 0 <= idx < n


def test_correlated_sampler_keeps_named_sequences_independent() -> None:
    sampler_a = CorrelatedSampler(KeyedRng(4), "cpu")
    sampler_b = CorrelatedSampler(KeyedRng(4), "cpu")

    first_b = sampler_a.sample_num("beta", 0.0, 1.0)
    _ = [sampler_a.sample_num("alpha", 0.0, 1.0) for _ in range(5)]
    second_b = sampler_a.sample_num("beta", 0.0, 1.0)

    ref_first_b = sampler_b.sample_num("beta", 0.0, 1.0)
    ref_second_b = sampler_b.sample_num("beta", 0.0, 1.0)

    assert first_b == ref_first_b
    assert second_b == ref_second_b


def test_correlated_sampling_scope_reuses_numeric_params_for_compatible_paths(
    monkeypatch,
) -> None:
    draw_calls: list[str] = []
    original = correlated_mod._draw_numeric_params

    def _tracking_draw_numeric_params(
        generator: torch.Generator,
        *,
        device: str,
    ) -> correlated_mod._NumericParams:
        draw_calls.append(device)
        return original(generator, device=device)

    monkeypatch.setattr(correlated_mod, "_draw_numeric_params", _tracking_draw_numeric_params)
    dataset_root = KeyedRng(17).keyed("dataset", 0)

    with correlated_sampling_scope(dataset_root):
        first = correlated_mod.numeric_params_for_name(
            dataset_root.keyed("lhs"),
            "shared_numeric",
            device="cpu",
        )
        second = correlated_mod.numeric_params_for_name(
            dataset_root.keyed("rhs"),
            "shared_numeric",
            device="cpu",
        )

    assert draw_calls == ["cpu"]
    assert first == second


def test_correlated_sampling_scope_reuses_categorical_weight_tensor_for_compatible_paths(
    monkeypatch,
) -> None:
    categorical_weight_calls: list[tuple[str, str, str]] = []
    original = correlated_mod.CorrelatedSamplingContext.categorical_weight

    def _tracking_categorical_weight(
        self: correlated_mod.CorrelatedSamplingContext,
        *,
        name: str,
        label: str,
        device: str,
    ) -> float:
        categorical_weight_calls.append((name, label, device))
        return original(self, name=name, label=label, device=device)

    monkeypatch.setattr(
        correlated_mod.CorrelatedSamplingContext,
        "categorical_weight",
        _tracking_categorical_weight,
    )
    dataset_root = KeyedRng(23).keyed("dataset", 0)

    with correlated_sampling_scope(dataset_root):
        _ = correlated_mod.sample_correlated_choice(
            dataset_root.keyed("lhs"),
            name="shared_category",
            values=("alpha", "beta", "gamma"),
            device="cpu",
        )
        _ = correlated_mod.sample_correlated_choice(
            dataset_root.keyed("rhs"),
            name="shared_category",
            values=("alpha", "beta", "gamma"),
            device="cpu",
        )

    assert categorical_weight_calls == [
        ("shared_category", "alpha", "cpu"),
        ("shared_category", "beta", "cpu"),
        ("shared_category", "gamma", "cpu"),
    ]
