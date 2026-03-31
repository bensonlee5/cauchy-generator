from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from dagzoo.cli.entrypoint import main
from dagzoo.core.generate_handoff import (
    GENERATE_HANDOFF_SCHEMA_NAME,
    GENERATE_HANDOFF_SCHEMA_VERSION,
    _load_generated_provenance,
    build_generate_handoff_manifest,
    validate_generate_handoff_manifest,
    write_generate_handoff_manifest,
)
from dagzoo.core.identity import stable_blake2s_hex

_UNIT_REQUEST_RUN_ID = "1" * 32
_UNIT_DATASET_IDS = ("2" * 32, "3" * 32)
_UNIT_FACTORIZATION = "independent_p_x_complete_and_p_y_given_x_complete"
_UNIT_METRIC_DEFINITION = "label-target log loss per test cell"


def _generate_overrides(handoff_root: str) -> dict[str, object]:
    return {
        "num_datasets": 2,
        "seed": 7,
        "rows": "1024..4096",
        "device": "cpu",
        "hardware_policy": "none",
        "missing_rate": None,
        "missing_mechanism": None,
        "missing_mar_observed_fraction": None,
        "missing_mar_logit_scale": None,
        "missing_mnar_logit_scale": None,
        "diagnostics": False,
        "diagnostics_out_dir": None,
        "handoff_root": handoff_root,
    }


def _write_generate_run_artifacts(run_root: Path) -> None:
    generated_dir = run_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    (generated_dir / "effective_config.yaml").write_text(
        f"seed: 7\noutput:\n  out_dir: {generated_dir.resolve()}\n",
        encoding="utf-8",
    )
    (generated_dir / "effective_config_trace.yaml").write_text(
        "- source: generate.handoff_root\n"
        "  path: output.out_dir\n"
        f"  new_value: {generated_dir.resolve()}\n",
        encoding="utf-8",
    )
    shard_dir = generated_dir / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_records = [
        {
            "dataset_index": dataset_index,
            "metadata": {
                "dataset_id": dataset_id,
                "config": {
                    "dataset": {
                        "target_parent_prior": "near_max_mixture",
                        "target_parent_near_max_band_min_fraction": 0.75,
                        "target_parent_below_sqrt_prob": 0.05,
                        "target_parent_midrange_prob": 0.20,
                    }
                },
                "lineage": {
                    "assignments": {
                        "target_parent_count": 6 + dataset_index,
                        "target_parent_fraction": 0.75 + (0.05 * dataset_index),
                        "target_parent_regime": "near_max" if dataset_index == 0 else "midrange",
                    }
                },
                "posterior_predictive": {
                    "factorization": _UNIT_FACTORIZATION,
                    "metric_definition": _UNIT_METRIC_DEFINITION,
                    "teacher_conditional_export_enabled": False,
                    "teacher_conditionals_available": False,
                },
                "prior": {
                    "factorization": _UNIT_FACTORIZATION,
                },
                "split_groups": {"request_run": _UNIT_REQUEST_RUN_ID},
            },
        }
        for dataset_index, dataset_id in enumerate(_UNIT_DATASET_IDS)
    ]
    (shard_dir / "metadata.ndjson").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in metadata_records) + "\n",
        encoding="utf-8",
    )


def _write_stub_generated_metadata(out_dir: Path, *, num_datasets: int) -> None:
    shard_dir = out_dir / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset_index in range(num_datasets):
        dataset_id = stable_blake2s_hex(
            {
                "request_run": _UNIT_REQUEST_RUN_ID,
                "dataset_index": dataset_index,
            }
        )
        records.append(
            {
                "dataset_index": dataset_index,
                "metadata": {
                    "dataset_id": dataset_id,
                    "config": {
                        "dataset": {
                            "target_parent_prior": "near_max_mixture",
                            "target_parent_near_max_band_min_fraction": 0.75,
                            "target_parent_below_sqrt_prob": 0.05,
                            "target_parent_midrange_prob": 0.20,
                        }
                    },
                    "lineage": {
                        "assignments": {
                            "target_parent_count": 6 + dataset_index,
                            "target_parent_fraction": 0.75 + (0.05 * dataset_index),
                            "target_parent_regime": "near_max"
                            if dataset_index == 0
                            else "midrange",
                        }
                    },
                    "posterior_predictive": {
                        "factorization": _UNIT_FACTORIZATION,
                        "metric_definition": _UNIT_METRIC_DEFINITION,
                        "teacher_conditional_export_enabled": False,
                        "teacher_conditionals_available": False,
                    },
                    "prior": {
                        "factorization": _UNIT_FACTORIZATION,
                    },
                    "split_groups": {"request_run": _UNIT_REQUEST_RUN_ID},
                },
            }
        )
    (shard_dir / "metadata.ndjson").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _load_ndjson(path: Path) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload.append(json.loads(line))
    return payload


def _unit_generated_metadata_record(
    *,
    dataset_index: int = 0,
    dataset_id: str | None = None,
    target_parent_prior: str = "near_max_mixture",
    target_parent_near_max_band_min_fraction: float = 0.75,
    target_parent_below_sqrt_prob: float = 0.05,
    target_parent_midrange_prob: float = 0.20,
    target_parent_count: int | None = None,
    target_parent_fraction: float | None = None,
    target_parent_regime: str = "near_max",
) -> dict[str, object]:
    resolved_dataset_id = dataset_id or stable_blake2s_hex(
        {
            "request_run": _UNIT_REQUEST_RUN_ID,
            "dataset_index": dataset_index,
        }
    )
    resolved_target_parent_count = target_parent_count if target_parent_count is not None else 6
    resolved_target_parent_fraction = (
        target_parent_fraction if target_parent_fraction is not None else 0.75
    )
    return {
        "dataset_index": dataset_index,
        "metadata": {
            "dataset_id": resolved_dataset_id,
            "config": {
                "dataset": {
                    "target_parent_prior": target_parent_prior,
                    "target_parent_near_max_band_min_fraction": (
                        target_parent_near_max_band_min_fraction
                    ),
                    "target_parent_below_sqrt_prob": target_parent_below_sqrt_prob,
                    "target_parent_midrange_prob": target_parent_midrange_prob,
                }
            },
            "lineage": {
                "assignments": {
                    "target_parent_count": resolved_target_parent_count,
                    "target_parent_fraction": resolved_target_parent_fraction,
                    "target_parent_regime": target_parent_regime,
                }
            },
            "posterior_predictive": {
                "factorization": _UNIT_FACTORIZATION,
                "metric_definition": _UNIT_METRIC_DEFINITION,
                "teacher_conditional_export_enabled": False,
                "teacher_conditionals_available": False,
            },
            "prior": {
                "factorization": _UNIT_FACTORIZATION,
            },
            "split_groups": {"request_run": _UNIT_REQUEST_RUN_ID},
        },
    }


def _write_generated_metadata_records(
    generated_dir: Path, records: list[dict[str, object]]
) -> None:
    shard_dir = generated_dir / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "metadata.ndjson").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def test_build_generate_handoff_manifest_is_versioned_and_valid(tmp_path) -> None:
    run_root = tmp_path / "run"
    _write_generate_run_artifacts(run_root)
    payload = build_generate_handoff_manifest(
        config_path="configs/default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root)),
        run_root=run_root,
        generated_dir=run_root / "generated",
        effective_config_path=run_root / "generated" / "effective_config.yaml",
        effective_config_trace_path=run_root / "generated" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
    )

    validate_generate_handoff_manifest(payload)

    assert payload["schema_name"] == GENERATE_HANDOFF_SCHEMA_NAME
    assert payload["schema_version"] == GENERATE_HANDOFF_SCHEMA_VERSION
    assert "request" not in payload
    assert payload["identity"]["source_family"] == "dagzoo.fixed_layout_scm"
    assert payload["identity"]["generate_run_id"] == _UNIT_REQUEST_RUN_ID
    assert payload["identity"]["generated_corpus_id"] == stable_blake2s_hex(
        {
            "generate_run_id": _UNIT_REQUEST_RUN_ID,
            "dataset_ids": list(_UNIT_DATASET_IDS),
        }
    )
    assert payload["generate_invocation"]["config_path"] == str(
        Path("configs/default.yaml").resolve()
    )
    assert set(payload["generate_invocation"]["overrides"]) == {
        "num_datasets",
        "seed",
        "rows",
        "device",
        "hardware_policy",
        "missing_rate",
        "missing_mechanism",
        "missing_mar_observed_fraction",
        "missing_mar_logit_scale",
        "missing_mnar_logit_scale",
        "diagnostics",
        "diagnostics_out_dir",
        "handoff_root",
    }
    assert payload["generate_invocation"]["overrides"]["handoff_root"] == str(run_root.resolve())
    assert payload["artifacts"]["generated_dir"] == str((run_root / "generated").resolve())
    assert payload["artifacts_relative"] == {
        "run_root": ".",
        "generated_dir": "generated",
        "effective_config_path": "generated/effective_config.yaml",
        "effective_config_trace_path": "generated/effective_config_trace.yaml",
    }
    assert len(payload["checksums"]["effective_config_sha256"]) == 64
    assert payload["defaults"] == {
        "recommended_training_corpus": "generated",
        "recommended_training_artifact_key": "generated_dir",
        "curation_policy": "none",
    }
    assert payload["summary"]["generated_datasets"] == 2
    assert payload["provenance"] == {
        "posterior_predictive_factorization": _UNIT_FACTORIZATION,
        "teacher_conditional_export": False,
        "teacher_conditional_metric_definition": _UNIT_METRIC_DEFINITION,
        "target_parent_prior": "near_max_mixture",
        "target_parent_count_range": {"min": 6, "max": 7},
        "target_parent_fraction_range": {"min": 0.75, "max": 0.8},
        "target_parent_regimes_present": ["midrange", "near_max"],
        "target_parent_near_max_band_min_fraction": 0.75,
        "target_parent_below_sqrt_prob": 0.05,
        "target_parent_midrange_prob": 0.20,
    }
    assert payload["throughput"]["generation_stage"]["datasets_per_minute"] == pytest.approx(10.0)
    assert payload["diversity_artifacts"] == {
        "summary_json_path": None,
        "summary_md_path": None,
    }


def test_build_generate_handoff_manifest_identity_ignores_root_specific_paths(tmp_path) -> None:
    run_root_a = tmp_path / "run_a"
    run_root_b = tmp_path / "run_b"
    _write_generate_run_artifacts(run_root_a)
    _write_generate_run_artifacts(run_root_b)

    payload_a = build_generate_handoff_manifest(
        config_path="configs/default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root_a)),
        run_root=run_root_a,
        generated_dir=run_root_a / "generated",
        effective_config_path=run_root_a / "generated" / "effective_config.yaml",
        effective_config_trace_path=run_root_a / "generated" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
    )
    payload_b = build_generate_handoff_manifest(
        config_path="configs/copied_default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root_b)),
        run_root=run_root_b,
        generated_dir=run_root_b / "generated",
        effective_config_path=run_root_b / "generated" / "effective_config.yaml",
        effective_config_trace_path=run_root_b / "generated" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
    )

    assert payload_a["generate_invocation"] != payload_b["generate_invocation"]
    assert payload_a["artifacts"] != payload_b["artifacts"]
    assert payload_a["checksums"] != payload_b["checksums"]
    assert payload_a["identity"] == payload_b["identity"]


def test_load_generated_provenance_requires_metadata_files(tmp_path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(
        ValueError,
        match=r"generated_dir: must contain shard metadata under shard_\*/metadata\.ndjson",
    ):
        _load_generated_provenance(generated_dir=generated_dir)


def test_load_generated_provenance_rejects_blank_metadata_stream(tmp_path) -> None:
    generated_dir = tmp_path / "generated"
    shard_dir = generated_dir / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "metadata.ndjson").write_text("\n\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"generated_dir: must contain posterior predictive provenance metadata",
    ):
        _load_generated_provenance(generated_dir=generated_dir)


def test_load_generated_provenance_rejects_invalid_json_after_blank_line(tmp_path) -> None:
    generated_dir = tmp_path / "generated"
    shard_dir = generated_dir / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "metadata.ndjson").write_text("\n{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"contains invalid JSON"):
        _load_generated_provenance(generated_dir=generated_dir)


@pytest.mark.parametrize(
    ("field_name", "bad_value", "pattern"),
    [
        ("target_parent_prior", "all_features", r"contains mixed target-parent priors"),
        (
            "target_parent_near_max_band_min_fraction",
            0.8,
            r"contains mixed target-parent near-max band minimum fractions",
        ),
        (
            "target_parent_below_sqrt_prob",
            0.15,
            r"contains mixed target-parent below-sqrt probabilities",
        ),
        (
            "target_parent_midrange_prob",
            0.3,
            r"contains mixed target-parent midrange probabilities",
        ),
    ],
)
def test_load_generated_provenance_rejects_mixed_target_parent_values(
    tmp_path,
    field_name: str,
    bad_value: object,
    pattern: str,
) -> None:
    generated_dir = tmp_path / "generated"
    records = [
        _unit_generated_metadata_record(dataset_index=0),
        _unit_generated_metadata_record(dataset_index=1, target_parent_regime="midrange"),
    ]
    dataset_config = records[1]["metadata"]["config"]["dataset"]
    dataset_config[field_name] = bad_value
    _write_generated_metadata_records(generated_dir, records)

    with pytest.raises(ValueError, match=pattern):
        _load_generated_provenance(generated_dir=generated_dir)


def test_write_generate_handoff_manifest_writes_json_and_rejects_invalid_payload(
    tmp_path,
) -> None:
    run_root = tmp_path / "run"
    _write_generate_run_artifacts(run_root)
    manifest_path = write_generate_handoff_manifest(
        config_path="configs/default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root)),
        run_root=run_root,
        generated_dir=run_root / "generated",
        effective_config_path=run_root / "generated" / "effective_config.yaml",
        effective_config_trace_path=run_root / "generated" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
        out_path=run_root / "handoff_manifest.json",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(payload)
    assert payload["summary"]["generated_datasets"] == 2
    assert payload["defaults"]["recommended_training_corpus"] == "generated"
    assert list(run_root.glob(".handoff_manifest.json.*.tmp")) == []

    payload["generate_invocation"]["overrides"]["num_datasets"] = "two"
    with pytest.raises(
        ValueError,
        match=r"handoff_manifest.generate_invocation.overrides.num_datasets: must be an integer",
    ):
        validate_generate_handoff_manifest(payload)


def test_build_generate_handoff_manifest_preserves_recipe_reference(tmp_path) -> None:
    run_root = tmp_path / "run"
    _write_generate_run_artifacts(run_root)

    payload = build_generate_handoff_manifest(
        config_path="recipe:default-baseline",
        generate_invocation_overrides=_generate_overrides(str(run_root)),
        run_root=run_root,
        generated_dir=run_root / "generated",
        effective_config_path=run_root / "generated" / "effective_config.yaml",
        effective_config_trace_path=run_root / "generated" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
    )

    validate_generate_handoff_manifest(payload)
    assert payload["generate_invocation"]["config_path"] == "recipe:default-baseline"


def test_validate_generate_handoff_manifest_rejects_non_list_target_parent_regimes_present(
    tmp_path,
) -> None:
    run_root = tmp_path / "run"
    _write_generate_run_artifacts(run_root)
    payload = build_generate_handoff_manifest(
        config_path="configs/default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root)),
        run_root=run_root,
        generated_dir=run_root / "generated",
        effective_config_path=run_root / "generated" / "effective_config.yaml",
        effective_config_trace_path=run_root / "generated" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
    )
    payload["provenance"]["target_parent_regimes_present"] = "near_max"

    with pytest.raises(
        ValueError,
        match=(
            r"handoff_manifest\.provenance\.target_parent_regimes_present: "
            r"must be a list of strings"
        ),
    ):
        validate_generate_handoff_manifest(payload)


def test_write_generate_handoff_manifest_does_not_overwrite_existing_file(tmp_path) -> None:
    run_root = tmp_path / "run"
    manifest_path = run_root / "handoff_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"sentinel": true}\n', encoding="utf-8")
    _write_generate_run_artifacts(run_root)

    with pytest.raises(RuntimeError, match="promotion target already exists"):
        _ = write_generate_handoff_manifest(
            config_path="configs/default.yaml",
            generate_invocation_overrides=_generate_overrides(str(run_root)),
            run_root=run_root,
            generated_dir=run_root / "generated",
            effective_config_path=run_root / "generated" / "effective_config.yaml",
            effective_config_trace_path=run_root / "generated" / "effective_config_trace.yaml",
            generated_datasets=2,
            generation_elapsed_seconds=12.0,
            requested_device="cpu",
            resolved_device="cpu",
            hardware_backend="cpu",
            hardware_device_name="CPU",
            hardware_tier="cpu",
            hardware_policy="none",
            out_path=manifest_path,
        )

    assert manifest_path.read_text(encoding="utf-8") == '{"sentinel": true}\n'
    assert list(manifest_path.parent.glob(".handoff_manifest.json.*.tmp")) == []


def test_generate_cli_handoff_root_writes_generated_outputs_and_manifest(tmp_path) -> None:
    handoff_root = tmp_path / "handoff_run"

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--handoff-root",
            str(handoff_root),
            "--num-datasets",
            "1",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
        ]
    )

    assert code == 0
    effective_config = yaml.safe_load(
        (handoff_root / "generated" / "effective_config.yaml").read_text(encoding="utf-8")
    )
    trace_payload = yaml.safe_load(
        (handoff_root / "generated" / "effective_config_trace.yaml").read_text(encoding="utf-8")
    )
    assert effective_config["output"]["out_dir"] == str((handoff_root / "generated").resolve())
    assert any(
        isinstance(item, dict)
        and item.get("source") == "generate.handoff_root"
        and item.get("path") == "output.out_dir"
        for item in trace_payload
    )

    generated_metadata_path = handoff_root / "generated" / "shard_00000" / "metadata.ndjson"
    assert generated_metadata_path.exists()
    generated_metadata = _load_ndjson(generated_metadata_path)
    assert len(generated_metadata) == 1

    handoff = json.loads((handoff_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(handoff)
    assert handoff["schema_name"] == GENERATE_HANDOFF_SCHEMA_NAME
    assert "request" not in handoff
    assert handoff["artifacts"]["generated_dir"] == str((handoff_root / "generated").resolve())
    assert handoff["artifacts_relative"]["generated_dir"] == "generated"
    assert handoff["defaults"]["recommended_training_corpus"] == "generated"
    assert handoff["defaults"]["recommended_training_artifact_key"] == "generated_dir"
    assert handoff["defaults"]["curation_policy"] == "none"
    assert handoff["identity"]["source_family"] == "dagzoo.fixed_layout_scm"
    assert handoff["generate_invocation"]["config_path"] == str(
        Path("configs/default.yaml").resolve()
    )
    assert set(handoff["generate_invocation"]["overrides"]) == {
        "num_datasets",
        "seed",
        "rows",
        "device",
        "hardware_policy",
        "missing_rate",
        "missing_mechanism",
        "missing_mar_observed_fraction",
        "missing_mar_logit_scale",
        "missing_mnar_logit_scale",
        "diagnostics",
        "diagnostics_out_dir",
        "handoff_root",
    }
    assert handoff["generate_invocation"]["overrides"]["num_datasets"] == 1
    assert handoff["generate_invocation"]["overrides"]["handoff_root"] == str(
        handoff_root.resolve()
    )
    assert len(handoff["checksums"]["effective_config_sha256"]) == 64
    assert handoff["summary"]["generated_datasets"] == 1
    assert handoff["diversity_artifacts"]["summary_json_path"] is None


def test_generate_handoff_manifest_uses_wall_clock_generation_timing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = tmp_path / "handoff_run"
    perf_counter_values = iter((10.0, 16.0))
    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.perf_counter", lambda: next(perf_counter_values)
    )

    def _stub_generate_batch_iter(*_args, **_kwargs):
        return iter(())

    def _stub_write_packed_parquet_shards_stream(
        _stream,
        *,
        out_dir: Path,
        shard_size: int,
        compression: str,
    ) -> int:
        assert shard_size > 0
        assert compression
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_stub_generated_metadata(out_dir, num_datasets=2)
        return 2

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.write_packed_parquet_shards_stream",
        _stub_write_packed_parquet_shards_stream,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--handoff-root",
            str(handoff_root),
            "--num-datasets",
            "2",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
        ]
    )

    assert code == 0
    handoff = json.loads((handoff_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(handoff)
    assert handoff["throughput"]["generation_stage"]["elapsed_seconds"] == pytest.approx(6.0)
    assert handoff["throughput"]["generation_stage"]["datasets_per_minute"] == pytest.approx(20.0)


def test_generate_cli_handoff_root_preserves_rows_under_cuda_policy(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    patch_detect_hardware,
) -> None:
    handoff_root = tmp_path / "handoff_cuda"
    patch_detect_hardware("cuda_h100", "dagzoo.core.config_resolution.detect_hardware")
    monkeypatch.setattr("dagzoo.core.generation_context.torch.cuda.is_available", lambda: True)

    def _stub_generate_batch_iter(*_args, **_kwargs):
        return iter(())

    def _stub_write_packed_parquet_shards_stream(
        _stream,
        *,
        out_dir: Path,
        shard_size: int,
        compression: str,
    ) -> int:
        assert shard_size > 0
        assert compression
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_stub_generated_metadata(out_dir, num_datasets=1)
        return 1

    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.generate_batch_iter",
        _stub_generate_batch_iter,
    )
    monkeypatch.setattr(
        "dagzoo.cli.commands.generate.write_packed_parquet_shards_stream",
        _stub_write_packed_parquet_shards_stream,
    )

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--handoff-root",
            str(handoff_root),
            "--num-datasets",
            "1",
            "--rows",
            "2000..60000",
            "--device",
            "cuda",
            "--hardware-policy",
            "cuda_tiered_v1",
        ]
    )

    assert code == 0
    effective_config = yaml.safe_load(
        (handoff_root / "generated" / "effective_config.yaml").read_text(encoding="utf-8")
    )
    assert effective_config["dataset"]["rows"]["mode"] == "fixed"
    assert int(effective_config["runtime"]["fixed_layout_target_cells"]) == 240_000_000

    handoff = json.loads((handoff_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(handoff)
    assert handoff["hardware"]["tier"] == "cuda_h100"
    assert handoff["hardware"]["resolved_device"] == "cuda"
    assert handoff["generate_invocation"]["overrides"]["rows"] == "2000..60000"


def test_generate_handoff_identity_is_stable_after_handoff_root_move(tmp_path) -> None:
    handoff_root = tmp_path / "handoff_run"

    code = main(
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--handoff-root",
            str(handoff_root),
            "--num-datasets",
            "1",
            "--rows",
            "1024",
            "--seed",
            "7",
            "--device",
            "cpu",
            "--hardware-policy",
            "none",
        ]
    )

    assert code == 0
    original_manifest_path = handoff_root / "handoff_manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(original_manifest)

    moved_root = tmp_path / "handoff_run_copy"
    shutil.copytree(handoff_root, moved_root)
    moved_manifest_path = moved_root / "handoff_manifest.json"
    moved_manifest = json.loads(moved_manifest_path.read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(moved_manifest)
    original_generated_records = _load_ndjson(
        handoff_root / "generated" / "shard_00000" / "metadata.ndjson"
    )
    moved_generated_records = _load_ndjson(
        moved_root / "generated" / "shard_00000" / "metadata.ndjson"
    )

    assert moved_manifest["identity"] == original_manifest["identity"]
    assert moved_manifest["checksums"] == original_manifest["checksums"]
    assert moved_generated_records == original_generated_records
    for key, relative_path in moved_manifest["artifacts_relative"].items():
        resolved = (moved_root / relative_path).resolve()
        if key == "run_root":
            assert resolved == moved_root.resolve()
        else:
            assert resolved.exists()


def test_generate_handoff_identity_is_stable_across_equivalent_roots(tmp_path) -> None:
    handoff_root_a = tmp_path / "handoff_a"
    handoff_root_b = tmp_path / "handoff_b"
    cli_args = [
        "generate",
        "--config",
        "configs/default.yaml",
        "--num-datasets",
        "1",
        "--rows",
        "1024",
        "--seed",
        "7",
        "--device",
        "cpu",
        "--hardware-policy",
        "none",
    ]

    assert main([*cli_args, "--handoff-root", str(handoff_root_a)]) == 0
    assert main([*cli_args, "--handoff-root", str(handoff_root_b)]) == 0

    manifest_a = json.loads((handoff_root_a / "handoff_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((handoff_root_b / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(manifest_a)
    validate_generate_handoff_manifest(manifest_b)

    metadata_a = _load_ndjson(handoff_root_a / "generated" / "shard_00000" / "metadata.ndjson")[0]
    metadata_b = _load_ndjson(handoff_root_b / "generated" / "shard_00000" / "metadata.ndjson")[0]

    assert (
        manifest_a["generate_invocation"]["overrides"]["handoff_root"]
        != (manifest_b["generate_invocation"]["overrides"]["handoff_root"])
    )
    assert manifest_a["artifacts"]["generated_dir"] != manifest_b["artifacts"]["generated_dir"]
    assert manifest_a["checksums"] != manifest_b["checksums"]
    assert metadata_a["metadata"]["dataset_id"] == metadata_b["metadata"]["dataset_id"]
    assert metadata_a["metadata"]["split_groups"] == metadata_b["metadata"]["split_groups"]
    assert manifest_a["identity"] == manifest_b["identity"]
    assert (
        manifest_a["identity"]["generate_run_id"]
        == metadata_a["metadata"]["split_groups"]["request_run"]
    )
