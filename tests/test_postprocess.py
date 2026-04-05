"""Tests for postprocess/postprocess.py."""

import pytest
import torch
from conftest import make_generator as _make_generator

from dagzoo.config import DatasetConfig
from dagzoo.core.validation import InvalidFeatureMatrixError
from dagzoo.postprocess.postprocess import (
    _clip_and_standardize_rows,
    inject_missingness,
    postprocess_dataset,
    postprocess_fixed_schema_batch,
)
from dagzoo.rng import KeyedRng
from dagzoo.sampling import sample_missingness_mask


def _make_data(
    generator: torch.Generator,
    n_train: int = 100,
    n_test: int = 50,
    n_feat: int = 5,
    n_classes: int = 3,
    task: str = "classification",
    add_constant_col: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    x_train = torch.randn(n_train, n_feat, generator=generator)
    x_test = torch.randn(n_test, n_feat, generator=generator)
    if add_constant_col:
        x_train[:, -1] = 1.0
        x_test[:, -1] = 1.0
    ftypes = ["num"] * n_feat

    if task == "classification":
        y_train = torch.randint(0, n_classes, (n_train,), generator=generator).to(torch.int64)
        y_test = torch.randint(0, n_classes, (n_test,), generator=generator).to(torch.int64)
    else:
        y_train = torch.randn(n_train, generator=generator)
        y_test = torch.randn(n_test, generator=generator)
    return x_train, y_train, x_test, y_test, ftypes, task


def _clip_and_standardize_reference_loop(x: torch.Tensor, feature_types: list[str]) -> torch.Tensor:
    out = x.clone()
    for i, t in enumerate(feature_types):
        if t == "cat":
            continue
        col = out[:, i]
        q = torch.quantile(col.float(), torch.tensor([0.01, 0.99], device=col.device))
        lo, hi = q[0], q[1]
        col = torch.clamp(col, lo, hi)
        mu = torch.mean(col)
        sd = torch.std(col, correction=0).clamp_min(1e-6)
        out[:, i] = (col - mu) / sd
    return out


def test_removes_constant_columns() -> None:
    g = _make_generator(0)
    xt, yt, xte, yte, ft, task = _make_data(g, n_feat=4, add_constant_col=True)
    xtp, _, xtep, _, ft_out = postprocess_dataset(xt, yt, xte, yte, ft, task, KeyedRng(0), "cpu")
    assert xtp.shape[1] < 4
    assert len(ft_out) == xtp.shape[1]


def test_postprocess_raises_specific_reason_when_all_columns_are_constant() -> None:
    xt = torch.ones((16, 3), dtype=torch.float32)
    xte = torch.ones((8, 3), dtype=torch.float32)
    yt = torch.randint(0, 2, (16,), generator=_make_generator(123), dtype=torch.int64)
    yte = torch.randint(0, 2, (8,), generator=_make_generator(124), dtype=torch.int64)

    with pytest.raises(InvalidFeatureMatrixError, match="all_constant_features"):
        postprocess_dataset(
            xt,
            yt,
            xte,
            yte,
            ["num", "num", "num"],
            "classification",
            KeyedRng(9),
            "cpu",
        )


def test_standardizes_numeric() -> None:
    g = _make_generator(1)
    xt, yt, xte, yte, ft, task = _make_data(g)
    xtp, _, xtep, _, _ = postprocess_dataset(xt, yt, xte, yte, ft, task, KeyedRng(1), "cpu")
    assert xtep.shape[1] == xtp.shape[1]
    for i in range(xtp.shape[1]):
        col = xtp[:, i]
        assert abs(float(torch.mean(col))) < 1e-5
        assert abs(float(torch.std(col, correction=0)) - 1.0) < 1e-5


def test_feature_postprocess_is_invariant_to_test_rows() -> None:
    x_train = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
        ],
        dtype=torch.float32,
    )
    x_test_a = torch.tensor([[4.0, 14.0], [5.0, 15.0]], dtype=torch.float32)
    x_test_b = torch.tensor([[4000.0, 4014.0], [5000.0, 5015.0]], dtype=torch.float32)
    y_train = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    y_test = torch.tensor([0, 1], dtype=torch.int64)
    feature_types = ["num", "num"]

    out_a = postprocess_dataset(
        x_train,
        y_train,
        x_test_a,
        y_test,
        list(feature_types),
        "classification",
        KeyedRng(123),
        "cpu",
    )
    out_b = postprocess_dataset(
        x_train,
        y_train,
        x_test_b,
        y_test,
        list(feature_types),
        "classification",
        KeyedRng(123),
        "cpu",
    )

    torch.testing.assert_close(out_a[0], out_b[0])


def test_clip_and_standardize_all_categorical_is_noop() -> None:
    g = _make_generator(16)
    x = torch.randn(48, 6, generator=g)
    feature_types = ["cat"] * x.shape[1]

    out = _clip_and_standardize_rows(x, feature_types)

    torch.testing.assert_close(out, x)


def test_clip_and_standardize_preserves_categorical_columns() -> None:
    g = _make_generator(17)
    x = torch.randn(64, 5, generator=g)
    feature_types = ["cat", "num", "cat", "num", "num"]

    out = _clip_and_standardize_rows(x, feature_types)

    torch.testing.assert_close(out[:, 0], x[:, 0])
    torch.testing.assert_close(out[:, 2], x[:, 2])


def test_clip_and_standardize_matches_loop_reference_for_mixed_types() -> None:
    g = _make_generator(18)
    x = torch.randn(192, 7, generator=g)
    feature_types = ["num", "cat", "num", "num", "cat", "num", "cat"]

    out = _clip_and_standardize_rows(x, feature_types)
    ref = _clip_and_standardize_reference_loop(x, feature_types)

    torch.testing.assert_close(out, ref)


def test_clip_and_standardize_batch_matches_scalar_helper() -> None:
    g = _make_generator(19)
    x = torch.randn(3, 96, 6, generator=g)
    feature_types = ["num", "cat", "num", "num", "cat", "num"]

    out = _clip_and_standardize_rows(x, feature_types)
    ref = torch.stack([_clip_and_standardize_rows(batch, feature_types) for batch in x], dim=0)

    torch.testing.assert_close(out, ref)


def test_preserves_class_counts() -> None:
    g = _make_generator(2)
    xt, yt, xte, yte, ft, task = _make_data(g, n_classes=4)
    _, ytp, _, ytep, _ = postprocess_dataset(xt, yt, xte, yte, ft, task, KeyedRng(2), "cpu")
    y_all_before = torch.cat([yt, yte])
    y_all_after = torch.cat([ytp, ytep])
    before_counts = sorted(torch.bincount(y_all_before).tolist())
    after_counts = sorted(torch.bincount(y_all_after).tolist())
    assert before_counts == after_counts


def test_classification_labels_are_remapped_to_contiguous_indices() -> None:
    g = _make_generator(12)
    x_train = torch.randn(40, 4, generator=g)
    x_test = torch.randn(20, 4, generator=g)
    y_train = torch.tensor([0, 2, 5, 9] * 10, dtype=torch.int64)
    y_test = torch.tensor([0, 2, 5, 9] * 5, dtype=torch.int64)
    feature_types = ["num", "num", "num", "num"]

    _, y_train_p, _, y_test_p, _ = postprocess_dataset(
        x_train,
        y_train,
        x_test,
        y_test,
        feature_types,
        "classification",
        KeyedRng(13),
        "cpu",
    )

    y_all_after = torch.cat([y_train_p, y_test_p], dim=0)
    unique_after = torch.unique(y_all_after, sorted=True)
    expected = torch.arange(unique_after.numel(), dtype=unique_after.dtype)
    assert torch.equal(unique_after, expected)

    _, before_counts_raw = torch.unique(torch.cat([y_train, y_test], dim=0), return_counts=True)
    _, after_counts_raw = torch.unique(y_all_after, return_counts=True)
    before_counts = sorted(before_counts_raw.tolist())
    after_counts = sorted(after_counts_raw.tolist())
    assert before_counts == after_counts


def test_many_class_postprocess_outputs_contiguous_labels() -> None:
    g = _make_generator(14)
    xt, yt, xte, yte, ft, task = _make_data(g, n_classes=32, n_train=256, n_test=256)
    _, ytp, _, ytep, _ = postprocess_dataset(xt, yt, xte, yte, ft, task, KeyedRng(15), "cpu")

    y_all = torch.cat([ytp, ytep], dim=0)
    unique_after = torch.unique(y_all, sorted=True)
    expected = torch.arange(unique_after.numel(), dtype=unique_after.dtype)
    assert torch.equal(unique_after, expected)
    assert torch.all(y_all >= 0)
    assert int(unique_after[-1].item()) == int(unique_after.numel() - 1)


def test_regression_clips_targets() -> None:
    g = _make_generator(3)
    xt, yt, xte, yte, ft, task = _make_data(g, task="regression")
    yt[0] = 1e6
    _, ytp, _, ytep, _ = postprocess_dataset(xt, yt, xte, yte, ft, task, KeyedRng(3), "cpu")
    y_all = torch.cat([ytp, ytep])
    assert torch.all(torch.isfinite(y_all))


def test_regression_target_postprocess_is_invariant_to_test_targets() -> None:
    x_train = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
        ],
        dtype=torch.float32,
    )
    x_test = torch.tensor([[4.0, 14.0], [5.0, 15.0]], dtype=torch.float32)
    y_train = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float32)
    y_test_a = torch.tensor([2.0, 2.5], dtype=torch.float32)
    y_test_b = torch.tensor([2000.0, 2500.0], dtype=torch.float32)
    feature_types = ["num", "num"]

    out_a = postprocess_dataset(
        x_train,
        y_train,
        x_test,
        y_test_a,
        list(feature_types),
        "regression",
        KeyedRng(456),
        "cpu",
    )
    out_b = postprocess_dataset(
        x_train,
        y_train,
        x_test,
        y_test_b,
        list(feature_types),
        "regression",
        KeyedRng(456),
        "cpu",
    )

    torch.testing.assert_close(out_a[1], out_b[1])


def test_postprocess_preserves_float64_feature_precision() -> None:
    x_train = torch.tensor(
        [
            [1.0, 0.0],
            [1.0 + 1e-12, 1.0],
            [1.0 + 2e-12, 2.0],
            [1.0 + 3e-12, 3.0],
        ],
        dtype=torch.float64,
    )
    x_test = torch.tensor(
        [
            [1.0 + 4e-12, 4.0],
            [1.0 + 5e-12, 5.0],
        ],
        dtype=torch.float64,
    )
    y_train = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    y_test = torch.tensor([0, 1], dtype=torch.int64)

    x_train_p, _, x_test_p, _, _, feature_index_map = postprocess_dataset(
        x_train,
        y_train,
        x_test,
        y_test,
        ["num", "num"],
        "classification",
        KeyedRng(910),
        "cpu",
        return_feature_index_map=True,
    )

    assert x_train_p.dtype == torch.float64
    assert x_test_p.dtype == torch.float64
    assert set(feature_index_map) == {0, 1}


def test_postprocess_preserves_float64_regression_target_precision() -> None:
    x_train = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
        ],
        dtype=torch.float64,
    )
    x_test = torch.tensor([[4.0, 14.0], [5.0, 15.0]], dtype=torch.float64)
    y_train = torch.tensor([1.0, 1.0 + 1e-12, 1.0 + 2e-12, 1.0 + 3e-12], dtype=torch.float64)
    y_test = torch.tensor([1.0 + 4e-12, 1.0 + 5e-12], dtype=torch.float64)

    _, y_train_p, _, y_test_p, _ = postprocess_dataset(
        x_train,
        y_train,
        x_test,
        y_test,
        ["num", "num"],
        "regression",
        KeyedRng(911),
        "cpu",
    )

    y_all = torch.cat([y_train_p, y_test_p])
    assert y_train_p.dtype == torch.float64
    assert y_test_p.dtype == torch.float64
    assert torch.unique(y_all).numel() > 1


def test_deterministic() -> None:
    g_data = _make_generator(99)
    xt, yt, xte, yte, ft, task = _make_data(g_data)
    out1 = postprocess_dataset(
        xt.clone(), yt.clone(), xte.clone(), yte.clone(), list(ft), task, KeyedRng(0), "cpu"
    )
    out2 = postprocess_dataset(
        xt.clone(), yt.clone(), xte.clone(), yte.clone(), list(ft), task, KeyedRng(0), "cpu"
    )
    torch.testing.assert_close(out1[0], out2[0])
    torch.testing.assert_close(out1[1], out2[1])


def test_feature_index_map_tracks_dropped_and_permuted_columns() -> None:
    g = _make_generator(10)
    xt, yt, xte, yte, _, task = _make_data(g, n_feat=5, add_constant_col=True)
    feature_types = ["num", "cat", "num", "cat", "num"]
    xtp, _, xtep, _, ft_out, feature_index_map = postprocess_dataset(
        xt,
        yt,
        xte,
        yte,
        feature_types,
        task,
        KeyedRng(11),
        "cpu",
        return_feature_index_map=True,
    )

    assert xtp.shape[1] == xtep.shape[1]
    assert len(feature_index_map) == xtp.shape[1]
    assert len(feature_index_map) == len(ft_out)
    assert len(set(feature_index_map)) == len(feature_index_map)
    assert all(0 <= int(i) < len(feature_types) for i in feature_index_map)
    assert 4 not in feature_index_map
    assert [feature_types[i] for i in feature_index_map] == ft_out


def test_postprocess_drops_columns_constant_in_train_even_if_test_varies() -> None:
    x_train = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    x_test = torch.tensor([[4.0, 7.0], [5.0, 8.0]], dtype=torch.float32)
    y_train = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    y_test = torch.tensor([0, 1], dtype=torch.int64)

    x_train_p, _, x_test_p, _, _, feature_index_map = postprocess_dataset(
        x_train,
        y_train,
        x_test,
        y_test,
        ["num", "num"],
        "classification",
        KeyedRng(789),
        "cpu",
        return_feature_index_map=True,
    )

    assert x_train_p.shape[1] == 1
    assert x_test_p.shape[1] == 1
    assert feature_index_map == [0]


def test_postprocess_keeps_columns_variable_in_train_even_if_test_is_constant() -> None:
    x_train = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
        ],
        dtype=torch.float32,
    )
    x_test = torch.tensor([[4.0, 99.0], [5.0, 99.0]], dtype=torch.float32)
    y_train = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    y_test = torch.tensor([0, 1], dtype=torch.int64)

    x_train_p, _, x_test_p, _, _, feature_index_map = postprocess_dataset(
        x_train,
        y_train,
        x_test,
        y_test,
        ["num", "num"],
        "classification",
        KeyedRng(790),
        "cpu",
        return_feature_index_map=True,
    )

    assert x_train_p.shape[1] == 2
    assert x_test_p.shape[1] == 2
    assert set(feature_index_map) == {0, 1}


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_postprocess_fixed_schema_batch_matches_scalar_preserve_schema(task: str) -> None:
    g = _make_generator(20)
    x_train = torch.randn(2, 32, 5, generator=g)
    x_test = torch.randn(2, 16, 5, generator=g)
    feature_types = ["num", "num", "cat", "num", "num"]
    if task == "classification":
        y_train = torch.tensor(
            [
                [0, 0, 1, 1] * 8,
                [0, 1, 2, 0] * 8,
            ],
            dtype=torch.int64,
        )
        y_test = torch.tensor(
            [
                [0, 0, 1, 1] * 4,
                [0, 1, 2, 0] * 4,
            ],
            dtype=torch.int64,
        )
    else:
        y_train = torch.randn(2, 32, generator=g)
        y_test = torch.randn(2, 16, generator=g)

    seeds = [301, 302]
    batched = postprocess_fixed_schema_batch(
        x_train,
        y_train,
        x_test,
        y_test,
        feature_types,
        task,
        postprocess_roots=[KeyedRng(seed) for seed in seeds],
    )
    scalar = [
        postprocess_dataset(
            x_train[index],
            y_train[index],
            x_test[index],
            y_test[index],
            list(feature_types),
            task,
            KeyedRng(seeds[index]),
            "cpu",
            preserve_feature_schema=True,
        )
        for index in range(2)
    ]

    for index, scalar_out in enumerate(scalar):
        torch.testing.assert_close(batched[0][index], scalar_out[0])
        torch.testing.assert_close(batched[1][index], scalar_out[1])
        torch.testing.assert_close(batched[2][index], scalar_out[2])
        torch.testing.assert_close(batched[3][index], scalar_out[3])


def test_inject_missingness_disabled_noop() -> None:
    g = _make_generator(5)
    x_train = torch.randn(16, 4, generator=g)
    x_test = torch.randn(8, 4, generator=g)
    cfg = DatasetConfig(missing_rate=0.0, missing_mechanism="none")

    out_train, out_test, summary = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, keyed_rng=KeyedRng(77), device="cpu"
    )

    torch.testing.assert_close(out_train, x_train)
    torch.testing.assert_close(out_test, x_test)
    assert summary is None


def test_inject_missingness_adds_nans_and_preserves_shape() -> None:
    g = _make_generator(6)
    x_train = torch.randn(64, 6, generator=g)
    x_test = torch.randn(32, 6, generator=g)
    cfg = DatasetConfig(missing_rate=0.3, missing_mechanism="mcar")

    out_train, out_test, summary = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, keyed_rng=KeyedRng(88), device="cpu"
    )

    assert out_train.shape == x_train.shape
    assert out_test.shape == x_test.shape
    assert torch.isnan(out_train).any()
    assert torch.isnan(out_test).any()
    assert summary is not None
    assert summary["mechanism"] == "mcar"
    assert summary["target_rate"] == 0.3
    assert 0.0 <= float(summary["realized_rate_overall"]) <= 1.0


def test_inject_missingness_deterministic_for_fixed_seed_and_attempt() -> None:
    g = _make_generator(7)
    x_train = torch.randn(96, 5, generator=g)
    x_test = torch.randn(48, 5, generator=g)
    cfg = DatasetConfig(missing_rate=0.35, missing_mechanism="mar")

    a_train, a_test, _ = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, keyed_rng=KeyedRng(101), device="cpu"
    )
    b_train, b_test, _ = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, keyed_rng=KeyedRng(101), device="cpu"
    )

    assert torch.equal(torch.isnan(a_train), torch.isnan(b_train))
    assert torch.equal(torch.isnan(a_test), torch.isnan(b_test))


def test_inject_missingness_changes_for_different_seed() -> None:
    g = _make_generator(8)
    x_train = torch.randn(96, 5, generator=g)
    x_test = torch.randn(48, 5, generator=g)
    cfg = DatasetConfig(missing_rate=0.35, missing_mechanism="mnar")

    a_train, a_test, _ = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, keyed_rng=KeyedRng(202), device="cpu"
    )
    b_train, b_test, _ = inject_missingness(
        x_train, x_test, dataset_cfg=cfg, keyed_rng=KeyedRng(203), device="cpu"
    )

    assert not torch.equal(torch.isnan(a_train), torch.isnan(b_train))
    assert not torch.equal(torch.isnan(a_test), torch.isnan(b_test))


def test_inject_missingness_samples_one_full_matrix_mask_before_resplitting() -> None:
    g = _make_generator(21)
    x_train = torch.randn(24, 4, generator=g)
    x_test = torch.randn(12, 4, generator=g)
    cfg = DatasetConfig(missing_rate=0.25, missing_mechanism="mar")
    keyed_rng = KeyedRng(909)

    out_train, out_test, summary = inject_missingness(
        x_train,
        x_test,
        dataset_cfg=cfg,
        keyed_rng=keyed_rng,
        device="cpu",
    )

    x_all = torch.cat([x_train, x_test], dim=0)
    full_mask = sample_missingness_mask(
        x_all,
        keyed_rng=KeyedRng(909).keyed("full_matrix"),
        dataset_cfg=cfg,
        device="cpu",
    )
    expected_all = x_all.masked_fill(full_mask, float("nan"))
    expected_train = expected_all[: x_train.shape[0]]
    expected_test = expected_all[x_train.shape[0] :]

    torch.testing.assert_close(out_train, expected_train, equal_nan=True)
    torch.testing.assert_close(out_test, expected_test, equal_nan=True)
    assert summary is not None
    assert summary["missing_count_overall"] == int(full_mask.sum().item())
