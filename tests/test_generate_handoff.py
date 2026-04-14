from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from dagzoo.cli.entrypoint import main
from dagzoo.config import GeneratorConfig
from dagzoo.core.generate_handoff import (
    GENERATE_HANDOFF_SCHEMA_NAME,
    GENERATE_HANDOFF_SCHEMA_VERSION,
    build_generate_handoff_manifest,
    validate_generate_handoff_manifest,
    write_generate_handoff_manifest,
)
from dagzoo.core.identity import stable_blake2s_hex
from dagzoo.io.shard_contract import (
    DATASET_CATALOG_FILENAME,
    REPLAY_CATALOG_FILENAME,
    iter_ndjson_records,
    write_dataset_catalog_records,
)

_UNIT_REQUEST_RUN_ID = "1" * 32
_UNIT_LAYOUT_PLAN_ID = "4" * 32
_UNIT_DATASET_IDS = ("2" * 32, "3" * 32)
_UNIT_INTERVENTION = {
    "mode": "hard_interventional",
    "signature": "a" * 32,
}


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


def _catalog_record(
    *,
    dataset_index: int,
    dataset_id: str,
    request_run: str = _UNIT_REQUEST_RUN_ID,
    layout_plan: str = _UNIT_LAYOUT_PLAN_ID,
    target_relevant_feature_count: int = 5,
    target_relevant_feature_fraction: float = 0.625,
    intervention: dict[str, str] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "dataset_index": dataset_index,
        "dataset_id": dataset_id,
        "task": "classification",
        "n_train": 16,
        "n_test": 8,
        "n_features": 8,
        "feature_types": ["num"] * 8,
        "n_classes": 3,
        "group_ids": {
            "request_run": request_run,
            "layout_plan": layout_plan,
        },
        "target_derivation": "tabiclv2_latent_node",
        "target_relevance": {
            "feature_count": target_relevant_feature_count,
            "feature_fraction": target_relevant_feature_fraction,
        },
    }
    if intervention is not None:
        record["intervention"] = dict(intervention)
    return record


def _write_ndjson(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        write_dataset_catalog_records(path, records)
        return
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _load_ndjson(path: Path) -> list[dict[str, object]]:
    return [dict(record) for record in iter_ndjson_records(path)]


def _write_generate_run_artifacts(
    run_root: Path,
    *,
    include_curated: bool = False,
    intervention: dict[str, str] | None = None,
) -> None:
    generated_dir = run_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = generated_dir / "shard_00000"
    records = [
        _catalog_record(
            dataset_index=index,
            dataset_id=dataset_id,
            target_relevant_feature_count=5 + index,
            target_relevant_feature_fraction=0.625 + (0.125 * index),
            intervention=intervention,
        )
        for index, dataset_id in enumerate(_UNIT_DATASET_IDS)
    ]
    _write_ndjson(shard_dir / DATASET_CATALOG_FILENAME, records)
    if include_curated:
        curated_shard_dir = run_root / "curated" / "shard_00000"
        _write_ndjson(curated_shard_dir / DATASET_CATALOG_FILENAME, records)


def test_build_generate_handoff_manifest_is_versioned_and_valid(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_generate_run_artifacts(run_root)

    payload = build_generate_handoff_manifest(
        config_path="configs/default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root)),
        run_root=run_root,
        generated_dir=run_root / "generated",
        effective_config_path=run_root / "internal" / "effective_config.yaml",
        effective_config_trace_path=run_root / "internal" / "effective_config_trace.yaml",
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

    assert payload == {
        "schema_name": GENERATE_HANDOFF_SCHEMA_NAME,
        "schema_version": GENERATE_HANDOFF_SCHEMA_VERSION,
        "identity": {
            "source_family": "dagzoo.fixed_layout_scm",
            "generate_run_id": _UNIT_REQUEST_RUN_ID,
            "generated_corpus_id": stable_blake2s_hex(
                {
                    "generate_run_id": _UNIT_REQUEST_RUN_ID,
                    "dataset_ids": list(_UNIT_DATASET_IDS),
                }
            ),
        },
        "artifacts_relative": {
            "generated_dir": "generated",
        },
        "summary": {
            "generated_datasets": 2,
        },
        "provenance": {
            "target_derivation": "tabiclv2_latent_node",
            "target_relevant_feature_count_range": {"min": 5, "max": 6},
            "target_relevant_feature_fraction_range": {"min": 0.625, "max": 0.75},
        },
    }


def test_build_generate_handoff_manifest_includes_curated_dir_and_provenance(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    _write_generate_run_artifacts(run_root, include_curated=True)

    payload = build_generate_handoff_manifest(
        config_path="configs/default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root)),
        run_root=run_root,
        generated_dir=run_root / "generated",
        effective_config_path=run_root / "internal" / "effective_config.yaml",
        effective_config_trace_path=run_root / "internal" / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
    )

    assert payload["artifacts_relative"] == {
        "generated_dir": "generated",
        "curated_dir": "curated",
    }
    assert payload["provenance"] == {
        "target_derivation": "tabiclv2_latent_node",
        "target_relevant_feature_count_range": {"min": 5, "max": 6},
        "target_relevant_feature_fraction_range": {"min": 0.625, "max": 0.75},
    }


def test_build_generate_handoff_manifest_includes_intervention_provenance(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_generate_run_artifacts(run_root, intervention=_UNIT_INTERVENTION)

    payload = build_generate_handoff_manifest(
        config_path="configs/default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root)),
        run_root=run_root,
        generated_dir=run_root / "generated",
        effective_config_path=run_root / "internal" / "effective_config.yaml",
        effective_config_trace_path=run_root / "internal" / "effective_config_trace.yaml",
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
    assert payload["provenance"] == {
        "intervention": _UNIT_INTERVENTION,
        "target_derivation": "tabiclv2_latent_node",
        "target_relevant_feature_count_range": {"min": 5, "max": 6},
        "target_relevant_feature_fraction_range": {"min": 0.625, "max": 0.75},
    }


def test_build_generate_handoff_manifest_rejects_mixed_intervention_provenance(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    generated_dir = run_root / "generated"
    shard_dir = generated_dir / "shard_00000"
    _write_ndjson(
        shard_dir / DATASET_CATALOG_FILENAME,
        [
            _catalog_record(
                dataset_index=0, dataset_id=_UNIT_DATASET_IDS[0], intervention=_UNIT_INTERVENTION
            ),
            _catalog_record(
                dataset_index=1,
                dataset_id=_UNIT_DATASET_IDS[1],
                intervention={
                    "mode": "hard_interventional",
                    "signature": "b" * 32,
                },
            ),
        ],
    )

    with pytest.raises(ValueError, match="contains mixed intervention summaries; expected one"):
        build_generate_handoff_manifest(
            config_path="configs/default.yaml",
            generate_invocation_overrides=_generate_overrides(str(run_root)),
            run_root=run_root,
            generated_dir=generated_dir,
            effective_config_path=run_root / "internal" / "effective_config.yaml",
            effective_config_trace_path=run_root / "internal" / "effective_config_trace.yaml",
            generated_datasets=2,
            generation_elapsed_seconds=12.0,
            requested_device="cpu",
            resolved_device="cpu",
            hardware_backend="cpu",
            hardware_device_name="CPU",
            hardware_tier="cpu",
            hardware_policy="none",
        )


def test_write_generate_handoff_manifest_writes_json_and_validates_payload(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_generate_run_artifacts(run_root)

    manifest_path = write_generate_handoff_manifest(
        config_path="configs/default.yaml",
        generate_invocation_overrides=_generate_overrides(str(run_root)),
        run_root=run_root,
        generated_dir=run_root / "generated",
        effective_config_path=run_root / "internal" / "effective_config.yaml",
        effective_config_trace_path=run_root / "internal" / "effective_config_trace.yaml",
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
    assert list(run_root.glob(".handoff_manifest.json.*.tmp")) == []

    payload["summary"]["generated_datasets"] = "two"
    with pytest.raises(
        ValueError,
        match=r"handoff_manifest.summary.generated_datasets: must be an integer",
    ):
        validate_generate_handoff_manifest(payload)


def test_write_generate_handoff_manifest_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    manifest_path = run_root / "handoff_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"sentinel": true}\n', encoding="utf-8")
    _write_generate_run_artifacts(run_root)

    with pytest.raises(RuntimeError, match="promotion target already exists"):
        write_generate_handoff_manifest(
            config_path="configs/default.yaml",
            generate_invocation_overrides=_generate_overrides(str(run_root)),
            run_root=run_root,
            generated_dir=run_root / "generated",
            effective_config_path=run_root / "internal" / "effective_config.yaml",
            effective_config_trace_path=run_root / "internal" / "effective_config_trace.yaml",
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


def test_generate_cli_handoff_root_writes_minimal_public_outputs_and_internal_sidecars(
    tmp_path: Path,
) -> None:
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
    assert not (handoff_root / "generated" / "effective_config.yaml").exists()
    assert not (handoff_root / "generated" / "effective_config_trace.yaml").exists()
    assert (handoff_root / "internal" / "effective_config.yaml").exists()
    assert (handoff_root / "internal" / "effective_config_trace.yaml").exists()
    assert (handoff_root / "internal" / "run_context.json").exists()

    generated_catalog_path = handoff_root / "generated" / "shard_00000" / DATASET_CATALOG_FILENAME
    replay_catalog_path = handoff_root / "internal" / "shard_00000" / REPLAY_CATALOG_FILENAME
    assert generated_catalog_path.exists()
    assert replay_catalog_path.exists()

    generated_catalog = _load_ndjson(generated_catalog_path)
    replay_catalog = _load_ndjson(replay_catalog_path)
    assert len(generated_catalog) == 1
    assert len(replay_catalog) == 1
    assert "metadata" not in generated_catalog[0]
    assert "intervention" not in generated_catalog[0]
    assert replay_catalog[0]["metadata"]["dataset_id"] == generated_catalog[0]["dataset_id"]
    assert "intervention" not in replay_catalog[0]["metadata"]

    effective_config = yaml.safe_load(
        (handoff_root / "internal" / "effective_config.yaml").read_text(encoding="utf-8")
    )
    assert effective_config["output"]["out_dir"] == str((handoff_root / "generated").resolve())

    handoff = json.loads((handoff_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(handoff)
    assert handoff["schema_name"] == GENERATE_HANDOFF_SCHEMA_NAME
    assert handoff["artifacts_relative"] == {"generated_dir": "generated"}
    assert "intervention" not in handoff["provenance"]
    assert set(handoff) == {
        "schema_name",
        "schema_version",
        "identity",
        "artifacts_relative",
        "summary",
        "provenance",
    }


def test_generate_cli_handoff_root_projects_intervention_summary_across_artifacts(
    tmp_path: Path,
) -> None:
    handoff_root = tmp_path / "handoff_run"

    code = main(
        [
            "generate",
            "--config",
            "configs/preset_intervention_target_generate_smoke.yaml",
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

    generated_catalog_path = handoff_root / "generated" / "shard_00000" / DATASET_CATALOG_FILENAME
    replay_catalog_path = handoff_root / "internal" / "shard_00000" / REPLAY_CATALOG_FILENAME
    generated_catalog = _load_ndjson(generated_catalog_path)
    replay_catalog = _load_ndjson(replay_catalog_path)
    assert len(generated_catalog) == 1
    assert len(replay_catalog) == 1

    signature = GeneratorConfig.from_yaml(
        "configs/preset_intervention_target_generate_smoke.yaml"
    ).intervention.signature
    assert signature is not None
    expected_summary = {
        "mode": "hard_interventional",
        "signature": signature,
    }

    assert generated_catalog[0]["intervention"] == expected_summary
    assert replay_catalog[0]["metadata"]["intervention"] == expected_summary

    handoff = json.loads((handoff_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(handoff)
    assert handoff["provenance"]["intervention"] == expected_summary


def test_generate_handoff_identity_is_stable_after_handoff_root_move(tmp_path: Path) -> None:
    handoff_root = tmp_path / "handoff_run"

    assert (
        main(
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
        == 0
    )

    original_manifest = json.loads(
        (handoff_root / "handoff_manifest.json").read_text(encoding="utf-8")
    )
    validate_generate_handoff_manifest(original_manifest)

    moved_root = tmp_path / "handoff_run_copy"
    shutil.copytree(handoff_root, moved_root)
    moved_manifest = json.loads((moved_root / "handoff_manifest.json").read_text(encoding="utf-8"))
    validate_generate_handoff_manifest(moved_manifest)

    original_catalog = _load_ndjson(
        handoff_root / "generated" / "shard_00000" / DATASET_CATALOG_FILENAME
    )
    moved_catalog = _load_ndjson(
        moved_root / "generated" / "shard_00000" / DATASET_CATALOG_FILENAME
    )

    assert moved_manifest["identity"] == original_manifest["identity"]
    assert moved_catalog == original_catalog
    resolved_generated_dir = (
        moved_root / moved_manifest["artifacts_relative"]["generated_dir"]
    ).resolve()
    assert resolved_generated_dir.exists()


def test_generate_handoff_identity_is_stable_across_equivalent_roots(tmp_path: Path) -> None:
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

    catalog_a = _load_ndjson(
        handoff_root_a / "generated" / "shard_00000" / DATASET_CATALOG_FILENAME
    )[0]
    catalog_b = _load_ndjson(
        handoff_root_b / "generated" / "shard_00000" / DATASET_CATALOG_FILENAME
    )[0]

    assert catalog_a["dataset_id"] == catalog_b["dataset_id"]
    assert catalog_a["group_ids"] == catalog_b["group_ids"]
    assert manifest_a["identity"] == manifest_b["identity"]
    assert manifest_a["identity"]["generate_run_id"] == catalog_a["group_ids"]["request_run"]
