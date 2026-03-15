"""Property tests for dataset row-spec normalization and resolution."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dagzoo.config import (
    DATASET_ROWS_MAX_TOTAL,
    DATASET_ROWS_MIN_TOTAL,
    DatasetRowsSpec,
    dataset_rows_bounds,
    dataset_rows_is_variable,
    normalize_dataset_rows,
    resolve_dataset_total_rows,
)
from dagzoo.rng import SEED32_MAX

_ROW_TOTAL_STRATEGY = st.integers(
    min_value=DATASET_ROWS_MIN_TOTAL,
    max_value=DATASET_ROWS_MAX_TOTAL,
)
_SEED32_STRATEGY = st.integers(min_value=0, max_value=SEED32_MAX)


@st.composite
def _fixed_rows_spec_strategy(draw: st.DrawFn) -> DatasetRowsSpec:
    value = draw(_ROW_TOTAL_STRATEGY)
    return DatasetRowsSpec(mode="fixed", value=value)


@st.composite
def _range_rows_spec_strategy(draw: st.DrawFn) -> DatasetRowsSpec:
    start = draw(
        st.integers(
            min_value=DATASET_ROWS_MIN_TOTAL,
            max_value=DATASET_ROWS_MAX_TOTAL - 1,
        )
    )
    stop = draw(st.integers(min_value=start + 1, max_value=DATASET_ROWS_MAX_TOTAL))
    return DatasetRowsSpec(mode="range", start=start, stop=stop)


@st.composite
def _choices_rows_spec_strategy(draw: st.DrawFn) -> DatasetRowsSpec:
    choices = draw(st.lists(_ROW_TOTAL_STRATEGY, min_size=2, max_size=5, unique=True))
    return DatasetRowsSpec(mode="choices", choices=choices)


_CANONICAL_ROWS_SPEC_STRATEGY = st.one_of(
    _fixed_rows_spec_strategy(),
    _range_rows_spec_strategy(),
    _choices_rows_spec_strategy(),
)
_VARIABLE_ROWS_SPEC_STRATEGY = st.one_of(
    _range_rows_spec_strategy(),
    _choices_rows_spec_strategy(),
)


def _representations(spec: DatasetRowsSpec) -> list[object]:
    if spec.mode == "fixed":
        assert spec.value is not None
        value = int(spec.value)
        return [
            value,
            str(value),
            {"mode": "fixed", "value": value},
            DatasetRowsSpec(mode="fixed", value=value),
        ]
    if spec.mode == "range":
        assert spec.start is not None and spec.stop is not None
        start = int(spec.start)
        stop = int(spec.stop)
        return [
            f"{start}..{stop}",
            {"mode": "range", "start": start, "stop": stop},
            DatasetRowsSpec(mode="range", start=start, stop=stop),
        ]

    assert spec.mode == "choices"
    choices = [int(choice) for choice in spec.choices]
    return [
        list(choices),
        ",".join(str(choice) for choice in choices),
        {"mode": "choices", "choices": list(choices)},
        {"mode": "choices", "values": list(choices)},
        DatasetRowsSpec(mode="choices", choices=list(choices)),
    ]


@settings(max_examples=100, deadline=None)
@given(spec=_CANONICAL_ROWS_SPEC_STRATEGY)
def test_normalize_dataset_rows_canonicalizes_supported_representations_hypothesis(
    spec: DatasetRowsSpec,
) -> None:
    for representation in _representations(spec):
        assert normalize_dataset_rows(representation) == spec


@settings(max_examples=100, deadline=None)
@given(spec=_CANONICAL_ROWS_SPEC_STRATEGY)
def test_dataset_rows_bounds_match_canonical_envelope_hypothesis(spec: DatasetRowsSpec) -> None:
    bounds = dataset_rows_bounds(spec)

    if spec.mode == "fixed":
        assert spec.value is not None
        assert bounds == (int(spec.value), int(spec.value))
        return
    if spec.mode == "range":
        assert spec.start is not None and spec.stop is not None
        assert bounds == (int(spec.start), int(spec.stop))
        return

    assert spec.mode == "choices"
    assert bounds == (min(spec.choices), max(spec.choices))


@settings(max_examples=100, deadline=None)
@given(spec=_CANONICAL_ROWS_SPEC_STRATEGY)
def test_dataset_rows_is_variable_matches_mode_hypothesis(spec: DatasetRowsSpec) -> None:
    assert dataset_rows_is_variable(spec) is (spec.mode in {"range", "choices"})


@settings(max_examples=100, deadline=None)
@given(spec=_fixed_rows_spec_strategy())
def test_resolve_dataset_total_rows_returns_fixed_value_hypothesis(spec: DatasetRowsSpec) -> None:
    assert spec.value is not None
    assert resolve_dataset_total_rows(spec, dataset_seed=None) == int(spec.value)


@settings(max_examples=100, deadline=None)
@given(spec=_CANONICAL_ROWS_SPEC_STRATEGY, dataset_seed=_SEED32_STRATEGY)
def test_resolve_dataset_total_rows_is_deterministic_for_same_seed_hypothesis(
    spec: DatasetRowsSpec,
    dataset_seed: int,
) -> None:
    first = resolve_dataset_total_rows(spec, dataset_seed=dataset_seed)
    second = resolve_dataset_total_rows(spec, dataset_seed=dataset_seed)

    assert first == second


@settings(max_examples=100, deadline=None)
@given(spec=_VARIABLE_ROWS_SPEC_STRATEGY, dataset_seed=_SEED32_STRATEGY)
def test_resolve_dataset_total_rows_stays_within_valid_support_hypothesis(
    spec: DatasetRowsSpec,
    dataset_seed: int,
) -> None:
    resolved = resolve_dataset_total_rows(spec, dataset_seed=dataset_seed)

    if spec.mode == "range":
        assert spec.start is not None and spec.stop is not None
        assert resolved is not None
        assert int(spec.start) <= resolved <= int(spec.stop)
        return

    assert spec.mode == "choices"
    assert resolved in spec.choices


@settings(max_examples=100, deadline=None)
@given(spec=_VARIABLE_ROWS_SPEC_STRATEGY)
def test_resolve_dataset_total_rows_requires_seed_for_variable_modes_hypothesis(
    spec: DatasetRowsSpec,
) -> None:
    with pytest.raises(ValueError, match="require dataset seed context"):
        resolve_dataset_total_rows(spec, dataset_seed=None)
