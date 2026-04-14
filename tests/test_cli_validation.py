from pathlib import Path

import pytest
import yaml
from conftest import load_repo_config, write_config

from dagzoo.cli.entrypoint import main
from dagzoo.config import GeneratorConfig


def test_generate_cli_rejects_invalid_device() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--device",
                "cud",
                "--num-datasets",
                "1",
                "--no-dataset-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_rejects_negative_num_datasets() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "-1",
                "--no-dataset-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_rejects_negative_seed() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                "--seed",
                "-1",
                "--no-dataset-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_rejects_oversized_seed() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                "--seed",
                "4294967296",
                "--no-dataset-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_rejects_inline_filter_enabled(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_repo_config()
    cfg.filter.enabled = True
    config_path = write_config(tmp_path, cfg, "inline_filter.yaml")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                str(config_path),
                "--num-datasets",
                "1",
                "--device",
                "cpu",
                "--hardware-policy",
                "none",
                "--no-dataset-write",
            ]
        )
    assert int(exc.value.code) == 2
    captured = capsys.readouterr()
    assert "Inline filtering has been removed from generate" in captured.err


def test_generate_cli_rejects_removed_parallel_generation_runtime_keys(tmp_path) -> None:
    config_path = tmp_path / "removed_parallel_runtime.yaml"
    config_path.write_text(
        yaml.safe_dump({"runtime": {"worker_count": 2, "worker_index": 1}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                str(config_path),
                "--num-datasets",
                "3",
                "--device",
                "cpu",
                "--hardware-policy",
                "none",
                "--out",
                str(tmp_path / "out"),
            ]
        )
    assert int(exc.value.code) == 2
    assert not (tmp_path / "out" / "effective_config.yaml").exists()
    assert not (tmp_path / "out" / "effective_config_trace.yaml").exists()


def test_generate_cli_rejects_unknown_recipe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "recipe:not-a-real-recipe",
                "--num-datasets",
                "1",
                "--no-dataset-write",
            ]
        )

    assert int(exc.value.code) == 2
    captured = capsys.readouterr()
    assert "dagzoo recipe list" in captured.err


def test_publish_hub_cli_surfaces_auth_guidance(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "dagzoo.cli.commands.publish.publish_handoff_to_hub",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "Hugging Face authentication is required. Run `hf auth login` or set `HF_TOKEN`, then retry `dagzoo publish hub`."
            )
        ),
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "publish",
                "hub",
                "--handoff-root",
                "handoffs/default-baseline",
                "--repo-id",
                "bensonlee/default-baseline-corpus",
            ]
        )

    assert int(exc.value.code) == 2
    captured = capsys.readouterr()
    assert "hf auth login" in captured.err
    assert "HF_TOKEN" in captured.err


def test_generate_cli_rejects_rows_override_for_stress_profile(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_repo_config()
    cfg.stress.profile = "anti_memorization_piecewise_classification_slice_v1"
    config_path = write_config(tmp_path, cfg, "stress_profile.yaml")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                str(config_path),
                "--num-datasets",
                "1",
                "--rows",
                "400..60000",
                "--no-dataset-write",
            ]
        )

    assert int(exc.value.code) == 2
    assert "locks dataset.rows to unset" in capsys.readouterr().err


def test_generate_cli_rejects_locked_path_override_for_stress_profile(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_repo_config()
    cfg.stress.profile = "anti_memorization_piecewise_classification_slice_v1"
    config_path = write_config(tmp_path, cfg, "stress_profile.yaml")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                str(config_path),
                "--num-datasets",
                "1",
                "--set",
                "dataset.n_train=2048",
                "--no-dataset-write",
            ]
        )

    assert int(exc.value.code) == 2
    assert "locks dataset.n_train to 768" in capsys.readouterr().err


def test_generate_cli_stress_profile_effective_config_is_concrete(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_repo_config()
    cfg.stress.profile = "anti_memorization_piecewise_classification_slice_v1"
    config_path = write_config(tmp_path, cfg, "stress_profile.yaml")
    out_dir = tmp_path / "stress_run"

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        assert config.stress.profile is None
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--out",
            str(out_dir),
            "--no-dataset-write",
        ]
    )

    assert code == 0
    effective_config = yaml.safe_load(
        (out_dir / "effective_config.yaml").read_text(encoding="utf-8")
    )
    assert "stress" not in effective_config
    assert effective_config["steering"]["preset"] == "anti_memorization_piecewise_v1"

    trace_payload = yaml.safe_load(
        (out_dir / "effective_config_trace.yaml").read_text(encoding="utf-8")
    )
    assert any(
        isinstance(item, dict)
        and item.get("source") == "stress.profile_materialization"
        and item.get("path") == "stress.profile"
        and item.get("new_value") is None
        for item in trace_payload
    )


def test_generate_cli_hard_intervention_effective_config_is_canonical(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_repo_config()
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 5
    cfg.intervention.mode = "hard_interventional"
    cfg.intervention.targets = [
        {"target_kind": "target", "value": 1.0},  # type: ignore[list-item]
        {"target_kind": "feature_node", "index": 1, "value": 2.0},  # type: ignore[list-item]
    ]
    cfg.validate_generation_constraints()
    config_path = write_config(tmp_path, cfg, "hard_intervention.yaml")
    out_dir = tmp_path / "hard_intervention_run"

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        assert config.intervention.signature == cfg.intervention.signature
        assert [str(target.target_kind) for target in config.intervention.targets] == [
            "feature_node",
            "target",
        ]
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--out",
            str(out_dir),
            "--no-dataset-write",
        ]
    )

    assert code == 0
    effective_config = yaml.safe_load(
        (out_dir / "effective_config.yaml").read_text(encoding="utf-8")
    )
    assert effective_config["intervention"] == {
        "mode": "hard_interventional",
        "targets": [
            {"target_kind": "feature_node", "index": 1, "value": 2.0},
            {"target_kind": "target", "index": None, "value": 1.0},
        ],
        "signature": cfg.intervention.signature,
    }

    roundtripped = GeneratorConfig.from_dict(effective_config)
    assert roundtripped.intervention.signature == cfg.intervention.signature


def test_fixed_layout_subcommand_is_removed() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["fixed-layout", "sample", "--config", "configs/default.yaml"])
    assert int(exc.value.code) == 2


def test_benchmark_cli_rejects_removed_parallel_generation_runtime_keys(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "benchmark_removed_parallel_runtime.yaml"
    config_path.write_text(
        yaml.safe_dump({"runtime": {"worker_count": 2, "worker_index": 0}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "benchmark",
                "--config",
                str(config_path),
                "--preset",
                "custom",
                "--suite",
                "smoke",
                "--no-memory",
            ]
        )

    assert int(exc.value.code) == 2
    assert (
        "runtime.worker_count, runtime.worker_index is no longer supported"
        in capsys.readouterr().err
    )


def test_benchmark_cli_rejects_device_override_with_multiple_presets(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "dagzoo.cli.commands.benchmark.run_benchmark_suite",
        lambda *args, **kwargs: pytest.fail("run_benchmark_suite should not be called"),
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "benchmark",
                "--config",
                "configs/default.yaml",
                "--preset",
                "cpu",
                "--preset",
                "custom",
                "--device",
                "mps",
                "--suite",
                "smoke",
                "--no-memory",
            ]
        )

    assert int(exc.value.code) == 2
    captured = capsys.readouterr()
    assert "--device" in captured.err
    assert "multiple --preset values" in captured.err


def test_diversity_audit_cli_rejects_removed_parallel_generation_runtime_keys(tmp_path) -> None:
    config_path = tmp_path / "diversity_removed_parallel_runtime.yaml"
    config_path.write_text(
        yaml.safe_dump({"runtime": {"worker_count": 2, "worker_index": 1}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "diversity-audit",
                "--baseline-config",
                str(config_path),
                "--variant-config",
                "configs/default.yaml",
            ]
        )
    assert int(exc.value.code) == 2


@pytest.mark.parametrize(
    "flag,value", [("--warn-threshold-pct", "nan"), ("--fail-threshold-pct", "inf")]
)
def test_diversity_audit_cli_rejects_non_finite_regression_thresholds(
    flag: str, value: str
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "diversity-audit",
                "--baseline-config",
                "configs/default.yaml",
                "--variant-config",
                "configs/preset_shift_benchmark_smoke.yaml",
                flag,
                value,
            ]
        )

    assert int(exc.value.code) == 2


def test_diversity_audit_cli_rejects_swapped_warn_and_fail_thresholds() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "diversity-audit",
                "--baseline-config",
                "configs/default.yaml",
                "--variant-config",
                "configs/preset_shift_benchmark_smoke.yaml",
                "--warn-threshold-pct",
                "10",
                "--fail-threshold-pct",
                "5",
            ]
        )

    assert int(exc.value.code) == 2


def test_filter_cli_rejects_removed_n_jobs_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "filter",
                "--in",
                "input",
                "--out",
                "out",
                "--n-jobs",
                "0",
            ]
        )
    assert int(exc.value.code) == 2


def test_filter_cli_rejects_removed_easy_skill_threshold_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "filter",
                "--in",
                "input",
                "--out",
                "out",
                "--easy-skill-threshold",
                "0.7",
            ]
        )
    assert int(exc.value.code) == 2


def test_filter_cli_rejects_removed_threshold_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "filter",
                "--in",
                "input",
                "--out",
                "out",
                "--threshold",
                "0.4",
            ]
        )
    assert int(exc.value.code) == 2


def test_filter_cli_passes_structural_overrides_to_runner(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _Result:
        def __init__(self) -> None:
            self.manifest_path = tmp_path / "filter_out" / "filter_manifest.parquet"
            self.summary_path = tmp_path / "filter_out" / "filter_summary.json"
            self.total_datasets = 2
            self.accepted_datasets = 2
            self.rejected_datasets = 0
            self.elapsed_seconds = 1.0
            self.datasets_per_minute = 120.0
            self.curated_out_dir = None
            self.curated_accepted_datasets = 0

    def _stub_run_deferred_filter(**kwargs):
        captured["kwargs"] = kwargs
        return _Result()

    monkeypatch.setattr(
        "dagzoo.cli.commands.filter.run_deferred_filter",
        _stub_run_deferred_filter,
    )

    code = main(
        [
            "filter",
            "--in",
            "input_shards",
            "--out",
            str(tmp_path / "filter_out"),
            "--set",
            "filter.min_target_indegree=0",
            "--set",
            "filter.min_target_relevant_feature_count=1",
            "--set",
            "filter.min_target_relevant_feature_fraction=0.2",
            "--set",
            "filter.max_attempts=5",
        ]
    )

    assert code == 0
    kwargs = captured["kwargs"]
    assert kwargs["in_dir"] == "input_shards"
    assert kwargs["out_dir"] == str(tmp_path / "filter_out")
    assert kwargs["curated_out_dir"] is None
    assert kwargs["path_overrides"] == (
        ("filter.min_target_indegree", 0),
        ("filter.min_target_relevant_feature_count", 1),
        ("filter.min_target_relevant_feature_fraction", 0.2),
        ("filter.max_attempts", 5),
    )


def test_filter_cli_passes_structural_override_to_runner(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _Result:
        def __init__(self) -> None:
            self.manifest_path = tmp_path / "filter_out" / "filter_manifest.parquet"
            self.summary_path = tmp_path / "filter_out" / "filter_summary.json"
            self.total_datasets = 1
            self.accepted_datasets = 1
            self.rejected_datasets = 0
            self.elapsed_seconds = 1.0
            self.datasets_per_minute = 60.0
            self.curated_out_dir = None
            self.curated_accepted_datasets = 0

    def _stub_run_deferred_filter(**kwargs):
        captured["kwargs"] = kwargs
        return _Result()

    monkeypatch.setattr(
        "dagzoo.cli.commands.filter.run_deferred_filter",
        _stub_run_deferred_filter,
    )

    code = main(
        [
            "filter",
            "--in",
            "input_shards",
            "--out",
            str(tmp_path / "filter_out"),
            "--set",
            "filter.min_target_indegree=2",
        ]
    )

    assert code == 0
    assert captured["kwargs"]["path_overrides"] == (("filter.min_target_indegree", 2),)


def test_filter_cli_leaves_structural_overrides_unset_by_default(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _Result:
        def __init__(self) -> None:
            self.manifest_path = tmp_path / "filter_out" / "filter_manifest.parquet"
            self.summary_path = tmp_path / "filter_out" / "filter_summary.json"
            self.total_datasets = 1
            self.accepted_datasets = 1
            self.rejected_datasets = 0
            self.elapsed_seconds = 1.0
            self.datasets_per_minute = 60.0
            self.curated_out_dir = None
            self.curated_accepted_datasets = 0

    def _stub_run_deferred_filter(**kwargs):
        captured["kwargs"] = kwargs
        return _Result()

    monkeypatch.setattr(
        "dagzoo.cli.commands.filter.run_deferred_filter",
        _stub_run_deferred_filter,
    )

    code = main(
        [
            "filter",
            "--in",
            "input_shards",
            "--out",
            str(tmp_path / "filter_out"),
        ]
    )

    assert code == 0
    assert captured["kwargs"]["path_overrides"] == ()


def test_filter_cli_rejects_removed_config_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "filter",
                "--in",
                "input",
                "--out",
                "out",
                "--config",
                "configs/default.yaml",
            ]
        )
    assert int(exc.value.code) == 2


def test_benchmark_cli_accepts_filter_enabled_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_repo_config()
    cfg.filter.enabled = True
    config_path = write_config(tmp_path, cfg, "filter_enabled_benchmark.yaml")
    captured: dict[str, object] = {}

    def _stub_run_benchmark_suite(preset_specs, **kwargs):
        captured["filter_enabled"] = bool(preset_specs[0].config.filter.enabled)
        captured["suite"] = kwargs["suite"]
        return {"preset_results": [], "regression": {"status": "pass", "issues": []}}

    monkeypatch.setattr(
        "dagzoo.cli.commands.benchmark.run_benchmark_suite",
        _stub_run_benchmark_suite,
    )
    monkeypatch.setattr(
        "dagzoo.cli.commands.benchmark.write_suite_json",
        lambda _summary, path: Path(path),
    )

    code = main(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--preset",
            "custom",
            "--suite",
            "smoke",
            "--json-out",
            str(tmp_path / "summary.json"),
            "--no-memory",
        ]
    )

    assert code == 0
    assert captured == {"filter_enabled": True, "suite": "smoke"}


def test_diversity_audit_cli_accepts_filter_enabled_configs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = load_repo_config()
    baseline.filter.enabled = True
    baseline_path = write_config(tmp_path, baseline, "baseline_filter_enabled.yaml")
    variant = load_repo_config()
    variant.filter.enabled = True
    variant_path = write_config(tmp_path, variant, "variant_filter_enabled.yaml")
    captured: dict[str, object] = {}

    def _stub_run_effective_diversity_audit(**kwargs):
        captured["baseline_filter_enabled"] = bool(kwargs["baseline_config"].filter.enabled)
        captured["variant_filters_enabled"] = [
            bool(config.filter.enabled) for config in kwargs["variant_configs"]
        ]
        return {"summary": {"overall_status": "pass", "num_variants": 1}}

    monkeypatch.setattr(
        "dagzoo.cli.commands.diagnostics.run_effective_diversity_audit",
        _stub_run_effective_diversity_audit,
    )
    monkeypatch.setattr(
        "dagzoo.cli.commands.diagnostics.write_effective_diversity_artifacts",
        lambda _report, out_dir: {"summary_json": Path(out_dir) / "summary.json"},
    )

    code = main(
        [
            "diversity-audit",
            "--baseline-config",
            str(baseline_path),
            "--variant-config",
            str(variant_path),
            "--out-dir",
            str(tmp_path / "diversity"),
        ]
    )

    assert code == 0
    assert captured == {
        "baseline_filter_enabled": True,
        "variant_filters_enabled": [True],
    }


def test_request_subcommand_is_removed() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "request",
                "--request",
                "request.yaml",
            ]
        )

    assert int(exc.value.code) == 2


def test_filter_calibration_subcommand_is_removed() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "filter-calibration",
                "--config",
                "configs/preset_filter_benchmark_smoke.yaml",
            ]
        )

    assert int(exc.value.code) == 2


def test_generate_cli_rejects_handoff_root_with_out(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--handoff-root",
                str(tmp_path / "handoff"),
                "--out",
                str(tmp_path / "out"),
            ]
        )

    assert int(exc.value.code) == 2
    captured = capsys.readouterr()
    assert "`--handoff-root` cannot be combined with `--out`." in captured.err


def test_generate_cli_rejects_handoff_root_with_no_dataset_write(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--handoff-root",
                str(tmp_path / "handoff"),
                "--no-dataset-write",
            ]
        )

    assert int(exc.value.code) == 2
    captured = capsys.readouterr()
    assert "`--handoff-root` cannot be combined with `--no-dataset-write`." in captured.err


def test_generate_cli_rejects_stale_handoff_root(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale_generated_dir = tmp_path / "handoff" / "generated" / "shard_00000"
    stale_generated_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--handoff-root",
                str(tmp_path / "handoff"),
            ]
        )

    assert int(exc.value.code) == 2
    captured = capsys.readouterr()
    assert "already contains shard data" in captured.err


def test_generate_cli_rejects_stale_handoff_root_manifest(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    handoff_root = tmp_path / "handoff"
    handoff_root.mkdir(parents=True, exist_ok=True)
    (handoff_root / "handoff_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--handoff-root",
                str(handoff_root),
            ]
        )

    assert int(exc.value.code) == 2
    captured = capsys.readouterr()
    assert "already contains a prior handoff manifest" in captured.err


def test_generate_cli_rejects_stale_handoff_root_filter_artifacts(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    filter_dir = tmp_path / "handoff" / "filter"
    filter_dir.mkdir(parents=True, exist_ok=True)
    (filter_dir / "filter_manifest.parquet").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--handoff-root",
                str(tmp_path / "handoff"),
            ]
        )

    assert int(exc.value.code) == 2
    captured = capsys.readouterr()
    assert "already contains prior filter artifacts" in captured.err


def test_generate_cli_rejects_stale_handoff_root_curated_shards(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    curated_dir = tmp_path / "handoff" / "curated" / "shard_00000"
    curated_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--handoff-root",
                str(tmp_path / "handoff"),
            ]
        )

    assert int(exc.value.code) == 2
    captured = capsys.readouterr()
    assert "already contains curated shard data" in captured.err


def test_generate_cli_uses_default_config_without_noise_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, bool] = {"called": False}

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = config
        _ = seed
        _ = device
        _ = num_datasets
        captured["called"] = True
        yield object()

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )
    assert code == 0
    assert captured["called"] is True


def test_generate_cli_applies_rows_override_no_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    out_dir = tmp_path / "rows_override_run"

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        captured["rows_spec"] = config.dataset.rows
        captured["n_train"] = int(config.dataset.n_train)
        captured["n_test"] = int(config.dataset.n_test)
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--rows",
            "400..60000",
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--out",
            str(out_dir),
            "--no-dataset-write",
        ]
    )
    assert code == 0
    rows_spec = captured["rows_spec"]
    assert rows_spec is not None
    assert rows_spec.mode == "fixed"
    assert int(rows_spec.value) == captured["n_train"] + captured["n_test"]
    effective_config = yaml.safe_load(
        (out_dir / "effective_config.yaml").read_text(encoding="utf-8")
    )
    assert effective_config["dataset"]["rows"]["mode"] == "fixed"
    assert int(effective_config["dataset"]["rows"]["value"]) == int(rows_spec.value)
    trace_payload = yaml.safe_load(
        (out_dir / "effective_config_trace.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(trace_payload, list)
    realization_events = [
        item
        for item in trace_payload
        if isinstance(item, dict) and item.get("source") == "generate.run_realization"
    ]
    assert realization_events
    assert any(
        isinstance(item, dict)
        and item.get("path") == "dataset.n_train"
        and int(item["new_value"]) == int(effective_config["dataset"]["n_train"])
        for item in realization_events
    )
    assert any(
        isinstance(item, dict)
        and item.get("path") == "dataset.rows.value"
        and int(item["new_value"]) == int(effective_config["dataset"]["rows"]["value"])
        for item in realization_events
    )


def test_generate_cli_writes_resolution_trace_artifact_no_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "run"

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = config
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--out",
            str(out_dir),
            "--no-dataset-write",
        ]
    )
    assert code == 0
    trace_path = out_dir / "effective_config_trace.yaml"
    assert trace_path.exists()
    trace_payload = yaml.safe_load(trace_path.read_text(encoding="utf-8"))
    assert isinstance(trace_payload, list)
    assert any(
        isinstance(item, dict) and item.get("path") == "runtime.device" for item in trace_payload
    )


def test_generate_cli_many_class_preset_end_to_end_no_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dagzoo.core import dataset as dataset_mod

    captured_metadata: list[dict[str, object]] = []
    original_generate_batch_iter = dataset_mod.generate_batch_iter

    def _capture_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        for bundle in original_generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        ):
            captured_metadata.append(bundle.metadata)
            yield bundle

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _capture_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/preset_many_class_generate_smoke.yaml",
            "--num-datasets",
            "2",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )

    assert code == 0
    assert len(captured_metadata) == 2
    for metadata in captured_metadata:
        class_structure = metadata["class_structure"]
        assert isinstance(class_structure, dict)
        assert 2 <= int(class_structure["n_classes_realized"]) <= 32
        filter_metadata = metadata["filter"]
        assert isinstance(filter_metadata, dict)
        assert filter_metadata["mode"] == "deferred"
        assert filter_metadata["status"] == "not_run"


@pytest.mark.parametrize(
    ("config_path", "expected_profile"),
    [
        ("configs/preset_shift_graph_drift_generate_smoke.yaml", "graph_drift"),
        ("configs/preset_shift_mechanism_drift_generate_smoke.yaml", "mechanism_drift"),
        ("configs/preset_shift_noise_drift_generate_smoke.yaml", "noise_drift"),
        ("configs/preset_shift_mixed_generate_smoke.yaml", "mixed"),
    ],
)
def test_generate_cli_shift_presets_emit_shift_metadata_no_write(
    monkeypatch: pytest.MonkeyPatch,
    config_path: str,
    expected_profile: str,
) -> None:
    from dagzoo.core import dataset as dataset_mod

    captured_shift: list[dict[str, object]] = []
    original_generate_batch_iter = dataset_mod.generate_batch_iter

    def _capture_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        for bundle in original_generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        ):
            payload = bundle.metadata["shift"]
            assert isinstance(payload, dict)
            captured_shift.append(payload)
            yield bundle

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _capture_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            config_path,
            "--num-datasets",
            "2",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )

    assert code == 0
    assert len(captured_shift) == 2
    for payload in captured_shift:
        assert payload["enabled"] is True
        assert payload["mode"] == expected_profile


@pytest.mark.parametrize(
    ("preset_path", "output_dir_name"),
    [
        (
            "configs/preset_intervention_target_generate_smoke.yaml",
            "intervention_target",
        ),
        (
            "configs/preset_intervention_feature_node_generate_smoke.yaml",
            "intervention_feature_node",
        ),
        (
            "configs/preset_intervention_latent_node_generate_smoke.yaml",
            "intervention_latent_node",
        ),
    ],
)
def test_generate_cli_intervention_presets_emit_summary_metadata_no_write(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    preset_path: str,
    output_dir_name: str,
) -> None:
    from dagzoo.core import dataset as dataset_mod

    cfg = GeneratorConfig.from_yaml(preset_path)
    cfg.runtime.device = "cpu"
    cfg.output.out_dir = str(tmp_path / output_dir_name)
    config_path = write_config(tmp_path, cfg, f"{output_dir_name}.yaml")

    captured_metadata: list[dict[str, object]] = []
    original_generate_batch_iter = dataset_mod.generate_batch_iter

    def _capture_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        for bundle in original_generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        ):
            captured_metadata.append(bundle.metadata)
            yield bundle

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _capture_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "2",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )

    assert code == 0
    assert len(captured_metadata) == 2
    expected_summary = {
        "mode": "hard_interventional",
        "signature": str(cfg.intervention.signature),
    }
    for metadata in captured_metadata:
        assert metadata["intervention"] == expected_summary
        assert "intervention" not in metadata["config"]


@pytest.mark.parametrize(
    ("config_path", "expected_family"),
    [
        ("configs/preset_noise_gaussian_generate_smoke.yaml", "gaussian"),
        ("configs/preset_noise_laplace_generate_smoke.yaml", "laplace"),
        ("configs/preset_noise_student_t_generate_smoke.yaml", "student_t"),
        ("configs/preset_noise_mixture_generate_smoke.yaml", "mixture"),
    ],
)
def test_generate_cli_noise_presets_emit_noise_metadata_no_write(
    monkeypatch: pytest.MonkeyPatch,
    config_path: str,
    expected_family: str,
) -> None:
    from dagzoo.core import dataset as dataset_mod

    captured_noise: list[dict[str, object]] = []
    original_generate_batch_iter = dataset_mod.generate_batch_iter

    def _capture_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        for bundle in original_generate_batch_iter(
            config,
            num_datasets=num_datasets,
            seed=seed,
            device=device,
        ):
            payload = bundle.metadata["noise_distribution"]
            assert isinstance(payload, dict)
            captured_noise.append(payload)
            yield bundle

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _capture_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            config_path,
            "--num-datasets",
            "2",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )

    assert code == 0
    assert len(captured_noise) == 2
    for payload in captured_noise:
        assert payload["family_requested"] == expected_family
        assert payload["sampling_strategy"] == "dataset_level"
        if expected_family == "mixture":
            assert payload["family_sampled"] in {"gaussian", "laplace", "student_t"}
            weights = payload["mixture_weights"]
            assert isinstance(weights, dict)
            assert sum(float(v) for v in weights.values()) == pytest.approx(1.0)
        else:
            assert payload["family_sampled"] == expected_family
            assert payload["mixture_weights"] is None


def test_benchmark_cli_rejects_negative_warmup() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "benchmark",
                "--config",
                "configs/default.yaml",
                "--preset",
                "custom",
                "--suite",
                "smoke",
                "--warmup",
                "-1",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_coverage_tolerates_null_quantiles_and_targets(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_repo_config()
    cfg.runtime.device = "cpu"
    cfg.output.out_dir = str(tmp_path / "run")
    cfg.diagnostics.enabled = True
    cfg.diagnostics.quantiles = None  # type: ignore[assignment]
    cfg.diagnostics.meta_feature_targets = None  # type: ignore[assignment]
    cfg.diagnostics.max_values_per_metric = None
    config_path = write_config(tmp_path, cfg, "null_diagnostics.yaml")

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.CoverageAggregator.update_bundle",
        lambda _self, _bundle: None,
    )

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )
    assert code == 0
    assert (tmp_path / "run" / "coverage_summary.json").exists()
    assert (tmp_path / "run" / "coverage_summary.md").exists()


@pytest.mark.parametrize(
    ("preset_path", "output_dir_name"),
    [
        (
            "configs/preset_stress_classification_slice_generate_smoke.yaml",
            "stress_classification_slice",
        ),
        (
            "configs/preset_stress_graph_breadth_generate_smoke.yaml",
            "stress_graph_breadth",
        ),
        (
            "configs/preset_stress_compositional_generate_smoke.yaml",
            "stress_compositional",
        ),
    ],
)
def test_generate_cli_stress_profile_generate_presets_resolve_with_no_write(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    preset_path: str,
    output_dir_name: str,
) -> None:
    cfg = GeneratorConfig.from_yaml(preset_path)
    cfg.runtime.device = "cpu"
    cfg.output.out_dir = str(tmp_path / output_dir_name)
    config_path = write_config(tmp_path, cfg, f"{output_dir_name}.yaml")

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.CoverageAggregator.update_bundle",
        lambda _self, _bundle: None,
    )

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )

    assert code == 0
    assert (tmp_path / output_dir_name / "coverage_summary.json").exists()
    assert (tmp_path / output_dir_name / "coverage_summary.md").exists()


def test_generate_cli_no_write_allows_null_output_dir_when_coverage_disabled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_repo_config()
    cfg.runtime.device = "cpu"
    cfg.output.out_dir = None  # type: ignore[assignment]
    cfg.diagnostics.enabled = False
    config_path = write_config(tmp_path, cfg, "null_output.yaml")

    def _stub_generate_batch_iter(
        _config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )
    assert code == 0


def test_generate_cli_enables_diagnostics_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        captured["diagnostics_enabled"] = config.diagnostics.enabled
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.CoverageAggregator.update_bundle",
        lambda _self, _bundle: None,
    )
    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--diagnostics",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )
    assert code == 0
    assert captured["diagnostics_enabled"] is True


def test_generate_cli_applies_missingness_overrides_no_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _stub_generate_batch_iter(
        config,
        *,
        num_datasets: int,
        seed: int | None = None,
        device: str | None = None,
    ):
        captured["missing_rate"] = config.dataset.missing_rate
        captured["missing_mechanism"] = config.dataset.missing_mechanism
        captured["missing_mar_observed_fraction"] = config.dataset.missing_mar_observed_fraction
        captured["missing_mar_logit_scale"] = config.dataset.missing_mar_logit_scale
        captured["missing_mnar_logit_scale"] = config.dataset.missing_mnar_logit_scale
        _ = seed
        _ = device
        for _ in range(num_datasets):
            yield object()

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--set",
            "dataset.missing_rate=0.3",
            "--set",
            "dataset.missing_mechanism=mar",
            "--set",
            "dataset.missing_mar_observed_fraction=0.7",
            "--set",
            "dataset.missing_mar_logit_scale=1.8",
            "--set",
            "dataset.missing_mnar_logit_scale=2.2",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )
    assert code == 0
    assert captured["missing_rate"] == pytest.approx(0.3)
    assert captured["missing_mechanism"] == "mar"
    assert captured["missing_mar_observed_fraction"] == pytest.approx(0.7)
    assert captured["missing_mar_logit_scale"] == pytest.approx(1.8)
    assert captured["missing_mnar_logit_scale"] == pytest.approx(2.2)


def test_generate_cli_rejects_invalid_missingness_combination() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                "--device",
                "cpu",
                "--set",
                "dataset.missing_rate=0.2",
                "--set",
                "dataset.missing_mechanism=none",
                "--hardware-policy",
                "none",
                "--no-dataset-write",
            ]
        )
    assert int(exc.value.code) == 2


@pytest.mark.parametrize("rows_value", ["399", "60001", "2000..300", "1024,1024", "abc"])
def test_generate_cli_rejects_invalid_rows_spec(rows_value: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--rows",
                rows_value,
                "--num-datasets",
                "1",
                "--device",
                "cpu",
                "--hardware-policy",
                "none",
                "--no-dataset-write",
            ]
        )
    assert int(exc.value.code) == 2


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("dataset.missing_rate", "1.1"),
        ("dataset.missing_rate", "-0.1"),
        ("dataset.missing_mar_observed_fraction", "0"),
        ("dataset.missing_mar_observed_fraction", "1.1"),
        ("dataset.missing_mar_logit_scale", "0"),
        ("dataset.missing_mnar_logit_scale", "-1"),
    ],
)
def test_generate_cli_rejects_invalid_missingness_scalar(path: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                "--set",
                f"{path}={value}",
                "--no-dataset-write",
            ]
        )
    assert int(exc.value.code) == 2


@pytest.mark.parametrize(
    "flag_args", [["--steer-meta"], ["--meta-target", "linearity_proxy=0.2:0.8"]]
)
def test_generate_cli_rejects_removed_steering_flags(flag_args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate",
                "--config",
                "configs/default.yaml",
                "--num-datasets",
                "1",
                *flag_args,
                "--no-dataset-write",
            ]
        )
    assert int(exc.value.code) == 2


def test_generate_cli_missingness_no_write_end_to_end(tmp_path) -> None:
    cfg = load_repo_config()
    cfg.runtime.device = "cpu"
    cfg.dataset.task = "classification"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 8
    cfg.dataset.n_classes_min = 2
    cfg.dataset.n_classes_max = 8
    cfg.dataset.n_features_min = 8
    cfg.dataset.n_features_max = 8
    cfg.graph.n_nodes_min = 2
    cfg.graph.n_nodes_max = 4
    cfg.output.out_dir = str(tmp_path / "run")
    cfg.diagnostics.enabled = False
    config_path = write_config(tmp_path, cfg, "missingness_e2e.yaml")

    code = main(
        [
            "generate",
            "--config",
            str(config_path),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--set",
            "dataset.missing_rate=0.2",
            "--set",
            "dataset.missing_mechanism=mnar",
            "--set",
            "dataset.missing_mnar_logit_scale=1.5",
            "--hardware-policy",
            "none",
            "--no-dataset-write",
        ]
    )
    assert code == 0
