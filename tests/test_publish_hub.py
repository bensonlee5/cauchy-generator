from __future__ import annotations

import json
from pathlib import Path

import pytest
from huggingface_hub.errors import LocalTokenNotFoundError

from dagzoo.core.generate_handoff import build_generate_handoff_manifest
from dagzoo.io.shard_contract import DATASET_CATALOG_FILENAME, write_dataset_catalog_records
from dagzoo.publish.hub import build_hub_dataset_card, publish_handoff_to_hub

_REQUEST_RUN_ID = "1" * 32
_LAYOUT_PLAN_ID = "2" * 32
_DATASET_IDS = ("3" * 32, "4" * 32)
_INTERVENTION = {"mode": "hard_interventional", "signature": "a" * 32}


def _write_ndjson(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        write_dataset_catalog_records(path, records)
        return
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _catalog_record(
    *,
    dataset_index: int,
    dataset_id: str,
    intervention: dict[str, str] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "dataset_index": dataset_index,
        "dataset_id": dataset_id,
        "task": "classification",
        "n_train": 16 + dataset_index,
        "n_test": 8,
        "n_features": 6 + dataset_index,
        "feature_types": ["num", "cat"],
        "n_classes": 3,
        "group_ids": {
            "request_run": _REQUEST_RUN_ID,
            "layout_plan": _LAYOUT_PLAN_ID,
        },
        "target_derivation": "tabiclv2_latent_node",
        "target_relevance": {
            "feature_count": 4 + dataset_index,
            "feature_fraction": 0.5 + (0.1 * dataset_index),
        },
    }
    if intervention is not None:
        record["intervention"] = dict(intervention)
    return record


def _write_corpus(
    root: Path, *, include_curated: bool, intervention: dict[str, str] | None
) -> None:
    generated_records = [
        _catalog_record(dataset_index=index, dataset_id=dataset_id, intervention=intervention)
        for index, dataset_id in enumerate(_DATASET_IDS)
    ]
    generated_shard = root / "generated" / "shard_00000"
    _write_ndjson(generated_shard / DATASET_CATALOG_FILENAME, generated_records)
    (generated_shard / "train.parquet").write_bytes(b"train")
    (generated_shard / "test.parquet").write_bytes(b"test")
    if include_curated:
        curated_shard = root / "curated" / "shard_00000"
        _write_ndjson(curated_shard / DATASET_CATALOG_FILENAME, generated_records[:1])
        (curated_shard / "train.parquet").write_bytes(b"curated-train")
        (curated_shard / "test.parquet").write_bytes(b"curated-test")


def _write_handoff_root(
    tmp_path: Path,
    *,
    config_path: str = "recipe:default-baseline",
    include_curated: bool = False,
    include_run_context: bool = True,
    intervention: dict[str, str] | None = None,
) -> Path:
    handoff_root = tmp_path / "handoff"
    _write_corpus(handoff_root, include_curated=include_curated, intervention=intervention)

    internal_dir = handoff_root / "internal"
    internal_dir.mkdir(parents=True, exist_ok=True)
    (internal_dir / "effective_config.yaml").write_text(
        "dataset:\n  missing_rate: 0.25\n", encoding="utf-8"
    )
    (internal_dir / "effective_config_trace.yaml").write_text(
        "- path: dataset.missing_rate\n", encoding="utf-8"
    )
    if include_run_context:
        (internal_dir / "run_context.json").write_text(
            json.dumps(
                {
                    "config_path": config_path,
                    "effective_config": {
                        "dataset": {
                            "missing_rate": 0.25,
                            "missing_mechanism": "mnar",
                        }
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = build_generate_handoff_manifest(
        config_path=config_path,
        generate_invocation_overrides={"handoff_root": str(handoff_root), "num_datasets": 2},
        run_root=handoff_root,
        generated_dir=handoff_root / "generated",
        effective_config_path=internal_dir / "effective_config.yaml",
        effective_config_trace_path=internal_dir / "effective_config_trace.yaml",
        generated_datasets=2,
        generation_elapsed_seconds=1.5,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
        source_family="dagzoo.heterogeneous_scm",
    )
    (handoff_root / "handoff_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return handoff_root


def test_build_hub_dataset_card_includes_recipe_and_curated_summary(tmp_path: Path) -> None:
    handoff_root = _write_handoff_root(
        tmp_path,
        include_curated=True,
        intervention=_INTERVENTION,
    )
    handoff_manifest = json.loads(
        (handoff_root / "handoff_manifest.json").read_text(encoding="utf-8")
    )
    run_context = json.loads(
        (handoff_root / "internal" / "run_context.json").read_text(encoding="utf-8")
    )

    card = build_hub_dataset_card(
        handoff_manifest=handoff_manifest,
        generated_summary={
            "dataset_count": 2,
            "task_counts": {"classification": 2},
            "feature_types": ("cat", "num"),
            "n_train_range": (16, 17),
            "n_test_range": (8, 8),
            "n_features_range": (6, 7),
            "n_classes_range": (3, 3),
        },
        curated_summary={
            "dataset_count": 1,
            "task_counts": {"classification": 1},
            "feature_types": ("cat", "num"),
            "n_train_range": (16, 16),
            "n_test_range": (8, 8),
            "n_features_range": (6, 6),
            "n_classes_range": (3, 3),
        },
        run_context=run_context,
        repo_id="bensonlee/default-baseline-corpus",
        license_id="apache-2.0",
    )

    assert "pretty_name: Default Baseline Synthetic Tabular Corpus" in card
    assert "license: apache-2.0" in card
    assert "- config reference: `recipe:default-baseline`" in card
    assert "- curated accepted datasets: 1" in card
    assert "- intervention mode: `hard_interventional`" in card
    assert "- missingness: `mnar (0.25)`" in card


def test_build_hub_dataset_card_falls_back_without_run_context(tmp_path: Path) -> None:
    handoff_root = _write_handoff_root(
        tmp_path,
        config_path="/private/tmp/custom.yaml",
        include_run_context=False,
    )
    handoff_manifest = json.loads(
        (handoff_root / "handoff_manifest.json").read_text(encoding="utf-8")
    )

    card = build_hub_dataset_card(
        handoff_manifest=handoff_manifest,
        generated_summary={
            "dataset_count": 2,
            "task_counts": {"classification": 2},
            "feature_types": ("num",),
            "n_train_range": (16, 17),
            "n_test_range": (8, 8),
            "n_features_range": (6, 7),
            "n_classes_range": (3, 3),
        },
        curated_summary=None,
        run_context=None,
        repo_id="bensonlee/custom-corpus",
        license_id=None,
    )

    assert "config reference" not in card
    assert "license:" not in card
    assert "# Custom Corpus Synthetic Tabular Corpus" in card


def test_publish_handoff_to_hub_uploads_only_public_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = _write_handoff_root(tmp_path, include_curated=True, intervention=_INTERVENTION)
    calls: dict[str, object] = {}

    class FakeApi:
        def create_repo(
            self, *, repo_id: str, repo_type: str, private: bool, exist_ok: bool
        ) -> None:
            calls["create_repo"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "exist_ok": exist_ok,
            }

        def upload_folder(
            self,
            *,
            repo_id: str,
            repo_type: str,
            folder_path: str,
            delete_patterns: list[str],
            commit_message: str,
        ) -> None:
            staging_dir = Path(folder_path)
            calls["upload"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "delete_patterns": tuple(delete_patterns),
                "commit_message": commit_message,
                "files": sorted(
                    path.relative_to(staging_dir).as_posix()
                    for path in staging_dir.rglob("*")
                    if path.is_file()
                ),
                "has_internal": (staging_dir / "internal").exists(),
                "readme": (staging_dir / "README.md").read_text(encoding="utf-8"),
            }

    monkeypatch.setattr("dagzoo.publish.hub.HfApi", FakeApi)

    result = publish_handoff_to_hub(
        handoff_root=handoff_root,
        repo_id="bensonlee/default-baseline-corpus",
        private=True,
        license_id="apache-2.0",
    )

    assert result.repo_url == "https://huggingface.co/datasets/bensonlee/default-baseline-corpus"
    assert result.generated_datasets == 2
    assert result.curated_datasets == 1
    assert calls["create_repo"] == {
        "repo_id": "bensonlee/default-baseline-corpus",
        "repo_type": "dataset",
        "private": True,
        "exist_ok": True,
    }
    upload = calls["upload"]
    assert upload["repo_id"] == "bensonlee/default-baseline-corpus"
    assert upload["repo_type"] == "dataset"
    assert upload["commit_message"] == "Publish dagzoo corpus"
    assert upload["has_internal"] is False
    assert upload["delete_patterns"] == (
        "generated",
        "generated/**",
        "curated",
        "curated/**",
        "handoff_manifest.json",
        "README.md",
    )
    assert "generated/shard_00000/train.parquet" in upload["files"]
    assert "curated/shard_00000/train.parquet" in upload["files"]
    assert "internal/run_context.json" not in upload["files"]
    assert "README.md" in upload["files"]
    assert "handoff_manifest.json" in upload["files"]
    assert "recipe:default-baseline" in upload["readme"]


def test_publish_handoff_to_hub_surfaces_auth_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = _write_handoff_root(tmp_path)

    class FakeApi:
        def create_repo(self, **_kwargs) -> None:
            raise LocalTokenNotFoundError("missing token")

    monkeypatch.setattr("dagzoo.publish.hub.HfApi", FakeApi)

    with pytest.raises(RuntimeError, match="hf auth login"):
        publish_handoff_to_hub(
            handoff_root=handoff_root,
            repo_id="bensonlee/default-baseline-corpus",
        )


def test_publish_handoff_to_hub_rejects_missing_manifest(tmp_path: Path) -> None:
    handoff_root = tmp_path / "handoff"
    handoff_root.mkdir()

    with pytest.raises(FileNotFoundError, match="handoff_manifest.json"):
        publish_handoff_to_hub(
            handoff_root=handoff_root,
            repo_id="bensonlee/default-baseline-corpus",
        )
