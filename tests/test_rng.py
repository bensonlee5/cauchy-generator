"""Tests for rng.py."""

import string

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import dagzoo.rng as rng_mod
from dagzoo.rng import (
    SEED32_MAX,
    KeyedRng,
    derive_seed,
    keyed_rng_from_generator,
    validate_seed32,
)

_COMPONENT_CHARS = string.ascii_letters + string.digits + "_-"
_SEED32_STRATEGY = st.integers(min_value=0, max_value=SEED32_MAX)
_COMPONENT_STRATEGY = st.one_of(
    st.text(alphabet=_COMPONENT_CHARS, min_size=1, max_size=8),
    st.integers(min_value=-32, max_value=32),
)
_COMPONENT_TUPLE_STRATEGY = st.lists(_COMPONENT_STRATEGY, min_size=0, max_size=4).map(tuple)
_AMBIENT_NONCE_STRATEGY = st.lists(_SEED32_STRATEGY, min_size=0, max_size=3).map(tuple)


def test_derive_seed_deterministic() -> None:
    assert derive_seed(42, "a", 1) == derive_seed(42, "a", 1)


def test_derive_seed_varies_with_base() -> None:
    assert derive_seed(1, "x") != derive_seed(2, "x")


def test_derive_seed_varies_with_component() -> None:
    assert derive_seed(42, "a") != derive_seed(42, "b")


def test_derive_seed_returns_valid_32bit() -> None:
    seed = derive_seed(99, "comp", 7)
    assert 0 <= seed < 2**32


def test_keyed_rng_child_seed_matches_derive() -> None:
    rng = KeyedRng(seed=42, path=("node", 3))
    assert rng.child_seed("weights") == derive_seed(42, "node", 3, "weights")


def test_keyed_rng_normalizes_list_path_to_tuple() -> None:
    rng = KeyedRng(seed=42, path=["node", 3])  # type: ignore[arg-type]
    assert rng.path == ("node", 3)
    assert rng.keyed("weights").child_seed() == derive_seed(42, "node", 3, "weights")


def test_keyed_rng_treats_scalar_string_path_as_one_component() -> None:
    rng = KeyedRng(seed=42, path="node")  # type: ignore[arg-type]
    assert rng.path == ("node",)
    assert rng.child_seed() == KeyedRng(seed=42).keyed("node").child_seed()


def test_keyed_rng_treats_scalar_int_path_as_one_component() -> None:
    rng = KeyedRng(seed=42, path=3)  # type: ignore[arg-type]
    assert rng.path == (3,)
    assert rng.child_seed() == KeyedRng(seed=42).keyed(3).child_seed()


def test_keyed_rng_path_does_not_track_source_list_mutation() -> None:
    source_path = ["node", 3]
    rng = KeyedRng(seed=42, path=source_path)  # type: ignore[arg-type]
    before = rng.child_seed("weights")
    source_path.append("mutated")
    assert rng.path == ("node", 3)
    assert rng.child_seed("weights") == before


def test_keyed_rng_keyed_chaining_matches_flat_derivation() -> None:
    root = KeyedRng(seed=17)
    assert root.keyed("dataset", 4).child_seed("plan") == root.child_seed("dataset", 4, "plan")


def test_keyed_rng_from_generator_uses_ambient_nonce_in_child_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = iter(
        [
            torch.tensor([17, 101, 102, 103], dtype=torch.int64),
            torch.tensor([17, 201, 202, 203], dtype=torch.int64),
        ]
    )
    monkeypatch.setattr(rng_mod.torch, "randint", lambda *args, **kwargs: next(words))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(99)
    first = keyed_rng_from_generator(generator, "root")
    second = keyed_rng_from_generator(generator, "root")

    assert first.seed == second.seed == 17
    assert first.child_seed("leaf") != second.child_seed("leaf")


def test_generator_derived_keyed_rng_preserves_nonce_across_keyed_children() -> None:
    root = KeyedRng(seed=17, _ambient_nonce=(101, 102, 103))
    assert root.keyed("dataset", 4).child_seed("plan") == root.child_seed("dataset", 4, "plan")


def test_keyed_rng_sibling_order_is_independent() -> None:
    root = KeyedRng(seed=99)
    first_a = root.keyed("a").child_seed("leaf")
    first_b = root.keyed("b").child_seed("leaf")
    second_b = root.keyed("b").child_seed("leaf")
    second_a = root.keyed("a").child_seed("leaf")
    assert first_a == second_a
    assert first_b == second_b


def test_keyed_rng_torch_rng_is_deterministic_for_same_key() -> None:
    root = KeyedRng(seed=123)
    draws_a = torch.rand(8, generator=root.torch_rng("node", 2), device="cpu")
    draws_b = torch.rand(8, generator=root.torch_rng("node", 2), device="cpu")
    torch.testing.assert_close(draws_a, draws_b)


@settings(max_examples=100, deadline=None)
@given(
    seed=_SEED32_STRATEGY,
    path=_COMPONENT_TUPLE_STRATEGY,
    ambient_nonce=_AMBIENT_NONCE_STRATEGY,
    components=_COMPONENT_TUPLE_STRATEGY,
)
def test_keyed_rng_child_seed_matches_derive_contract_hypothesis(
    seed: int,
    path: tuple[str | int, ...],
    ambient_nonce: tuple[int, ...],
    components: tuple[str | int, ...],
) -> None:
    rng = KeyedRng(seed=seed, path=path, _ambient_nonce=ambient_nonce)
    ambient_components = (rng_mod._AMBIENT_NONCE_MARKER, *ambient_nonce) if ambient_nonce else ()

    assert rng.child_seed(*components) == derive_seed(
        seed,
        *ambient_components,
        *path,
        *components,
    )


@settings(max_examples=100, deadline=None)
@given(
    seed=_SEED32_STRATEGY,
    path=_COMPONENT_TUPLE_STRATEGY,
    ambient_nonce=_AMBIENT_NONCE_STRATEGY,
    first=_COMPONENT_TUPLE_STRATEGY,
    second=_COMPONENT_TUPLE_STRATEGY,
)
def test_keyed_rng_keyed_chaining_matches_flat_derivation_hypothesis(
    seed: int,
    path: tuple[str | int, ...],
    ambient_nonce: tuple[int, ...],
    first: tuple[str | int, ...],
    second: tuple[str | int, ...],
) -> None:
    root = KeyedRng(seed=seed, path=path, _ambient_nonce=ambient_nonce)

    assert root.keyed(*first).keyed(*second).child_seed() == root.child_seed(*first, *second)


@settings(max_examples=100, deadline=None)
@given(
    seed=_SEED32_STRATEGY,
    path=_COMPONENT_TUPLE_STRATEGY,
    ambient_nonce=_AMBIENT_NONCE_STRATEGY,
    components=_COMPONENT_TUPLE_STRATEGY,
)
def test_keyed_rng_torch_rng_is_deterministic_for_same_key_hypothesis(
    seed: int,
    path: tuple[str | int, ...],
    ambient_nonce: tuple[int, ...],
    components: tuple[str | int, ...],
) -> None:
    root = KeyedRng(seed=seed, path=path, _ambient_nonce=ambient_nonce)
    draws_a = torch.rand(6, generator=root.torch_rng(*components), device="cpu")
    draws_b = torch.rand(6, generator=root.torch_rng(*components), device="cpu")

    torch.testing.assert_close(draws_a, draws_b)


def test_validate_seed32_accepts_boundaries() -> None:
    assert validate_seed32(0) == 0
    assert validate_seed32(SEED32_MAX) == SEED32_MAX


@pytest.mark.parametrize("bad_seed", [-1, SEED32_MAX + 1, True])  # type: ignore[list-item]
def test_validate_seed32_rejects_out_of_range_values(bad_seed: int | bool) -> None:
    with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
        _ = validate_seed32(bad_seed)
