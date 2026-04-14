"""Publish dagzoo handoff roots to Hugging Face Hub dataset repositories."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

from dagzoo.core.generate_handoff import (
    HANDOFF_MANIFEST_FILENAME,
    validate_generate_handoff_manifest,
)
from dagzoo.io.shard_contract import (
    DATASET_CATALOG_FILENAME,
    INTERNAL_DIRNAME,
    RUN_CONTEXT_FILENAME,
    iter_ndjson_records,
)
from dagzoo.recipes import get_recipe_spec, parse_recipe_reference

HF_DATASET_BASE_URL = "https://huggingface.co/datasets"
_MANAGED_DELETE_PATTERNS = (
    "generated",
    "generated/**",
    "curated",
    "curated/**",
    HANDOFF_MANIFEST_FILENAME,
    "README.md",
)
_CARD_TAGS = ("tabular", "synthetic", "dagzoo")


@dataclass(frozen=True, slots=True)
class HubPublishResult:
    """Summary returned after publishing one handoff root to the Hub."""

    repo_id: str
    repo_url: str
    generated_datasets: int
    curated_datasets: int | None


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} at {path} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} at {path} must be a JSON object.")
    return payload


def _resolve_relative_dir(
    handoff_root: Path,
    *,
    relative_path: object,
    field_name: str,
    required: bool,
) -> Path | None:
    if relative_path is None:
        if required:
            raise ValueError(f"{field_name} is required in {HANDOFF_MANIFEST_FILENAME}.")
        return None
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{field_name} in {HANDOFF_MANIFEST_FILENAME} must be a non-empty string.")
    resolved = (handoff_root / relative_path).resolve()
    if not resolved.is_relative_to(handoff_root.resolve()):
        raise ValueError(f"{field_name} must stay within the handoff root.")
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} path does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"{field_name} path must be a directory: {resolved}")
    return resolved


def _iter_catalog_paths(corpus_dir: Path) -> list[Path]:
    return sorted(corpus_dir.glob(f"shard_*/{DATASET_CATALOG_FILENAME}"))


def _summarize_corpus(corpus_dir: Path) -> dict[str, Any]:
    catalog_paths = _iter_catalog_paths(corpus_dir)
    if not catalog_paths:
        raise ValueError(
            f"{corpus_dir} must contain shard catalogs under shard_*/{DATASET_CATALOG_FILENAME}."
        )

    dataset_count = 0
    task_counts: Counter[str] = Counter()
    feature_type_values: set[str] = set()
    n_train_values: list[int] = []
    n_test_values: list[int] = []
    n_feature_values: list[int] = []
    n_class_values: list[int] = []

    for catalog_path in catalog_paths:
        for record in iter_ndjson_records(catalog_path):
            dataset_count += 1
            task = record.get("task")
            if isinstance(task, str) and task.strip():
                task_counts[task.strip()] += 1
            feature_types = record.get("feature_types")
            if isinstance(feature_types, list):
                for value in feature_types:
                    if isinstance(value, str) and value.strip():
                        feature_type_values.add(value.strip())
            for key, values in (
                ("n_train", n_train_values),
                ("n_test", n_test_values),
                ("n_features", n_feature_values),
            ):
                value = record.get(key)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{catalog_path} record field `{key}` must be an integer.")
                values.append(int(value))
            n_classes = record.get("n_classes")
            if n_classes is not None:
                if isinstance(n_classes, bool) or not isinstance(n_classes, int):
                    raise ValueError(f"{catalog_path} record field `n_classes` must be an integer.")
                n_class_values.append(int(n_classes))

    if dataset_count == 0:
        raise ValueError(f"{corpus_dir} must contain at least one dataset catalog record.")

    return {
        "dataset_count": dataset_count,
        "task_counts": dict(sorted(task_counts.items())),
        "feature_types": tuple(sorted(feature_type_values)),
        "n_train_range": (min(n_train_values), max(n_train_values)),
        "n_test_range": (min(n_test_values), max(n_test_values)),
        "n_features_range": (min(n_feature_values), max(n_feature_values)),
        "n_classes_range": None
        if not n_class_values
        else (min(n_class_values), max(n_class_values)),
    }


def _sanitize_config_reference(run_context: Mapping[str, Any] | None) -> str | None:
    if run_context is None:
        return None
    config_path = run_context.get("config_path")
    if not isinstance(config_path, str) or not config_path.strip():
        return None
    stripped = config_path.strip()
    if parse_recipe_reference(stripped) is not None:
        return stripped
    if Path(stripped).is_absolute():
        return "custom YAML config"
    return stripped


def _extract_missingness_summary(run_context: Mapping[str, Any] | None) -> str | None:
    if run_context is None:
        return None
    effective_config = run_context.get("effective_config")
    if not isinstance(effective_config, Mapping):
        return None
    dataset = effective_config.get("dataset")
    if not isinstance(dataset, Mapping):
        return None
    mechanism = dataset.get("missing_mechanism")
    rate = dataset.get("missing_rate")
    if not isinstance(mechanism, str) or not mechanism.strip():
        return None
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        return mechanism.strip()
    normalized_rate = float(rate)
    if mechanism.strip() == "none" or normalized_rate <= 0.0:
        return "none"
    return f"{mechanism.strip()} ({normalized_rate:g})"


def _pretty_name(repo_id: str, *, recipe_title: str | None = None) -> str:
    if recipe_title is not None and recipe_title.strip():
        return f"{recipe_title.strip()} Synthetic Tabular Corpus"
    repo_leaf = repo_id.split("/", 1)[-1]
    return f"{repo_leaf.replace('-', ' ').replace('_', ' ').title()} Synthetic Tabular Corpus"


def _format_range(label: str, value_range: tuple[int, int] | None) -> str | None:
    if value_range is None:
        return None
    low, high = value_range
    if low == high:
        return f"- {label}: {low}"
    return f"- {label}: {low} to {high}"


def _format_task_counts(task_counts: Mapping[str, int]) -> str:
    return ", ".join(f"{task} ({count})" for task, count in task_counts.items())


def build_hub_dataset_card(
    *,
    handoff_manifest: Mapping[str, Any],
    generated_summary: Mapping[str, Any],
    curated_summary: Mapping[str, Any] | None,
    run_context: Mapping[str, Any] | None,
    repo_id: str,
    license_id: str | None = None,
) -> str:
    """Build the Hub dataset card README for one dagzoo handoff root."""

    identity = handoff_manifest["identity"]
    provenance = handoff_manifest.get("provenance")
    config_reference = _sanitize_config_reference(run_context)
    recipe_title: str | None = None
    recipe_summary: str | None = None
    recipe_citations: tuple[str, ...] = ()
    if config_reference is not None:
        recipe_name = parse_recipe_reference(config_reference)
        if recipe_name is not None:
            spec = get_recipe_spec(recipe_name)
            recipe_title = spec.title
            recipe_summary = spec.summary
            recipe_citations = spec.citations

    pretty_name = _pretty_name(repo_id, recipe_title=recipe_title)
    metadata: dict[str, Any] = {
        "pretty_name": pretty_name,
        "tags": list(_CARD_TAGS),
    }
    if license_id is not None:
        metadata["license"] = license_id
    front_matter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False).strip()

    lines = [
        "---",
        front_matter,
        "---",
        "",
        f"# {pretty_name}",
        "",
        "This dataset repository contains synthetic tabular corpora generated with `dagzoo`.",
    ]
    if recipe_summary is not None:
        lines.append(recipe_summary)
    else:
        lines.append(
            "The corpus is generated from latent DAG structure and exported through the public handoff contract."
        )

    lines.extend(
        [
            "",
            "## What is included",
            "",
            "- `generated/`: public parquet shards plus per-shard `dataset_catalog.parquet` catalogs",
            "- `curated/`: optional accepted-only shards written later by `dagzoo filter`",
            "- `handoff_manifest.json`: portable corpus identity and provenance metadata",
            "",
            "## Corpus summary",
            "",
            f"- generated datasets: {generated_summary['dataset_count']}",
            f"- task mix: {_format_task_counts(generated_summary['task_counts'])}",
        ]
    )
    for line in (
        _format_range("train rows per dataset", generated_summary["n_train_range"]),
        _format_range("test rows per dataset", generated_summary["n_test_range"]),
        _format_range("features per dataset", generated_summary["n_features_range"]),
        _format_range("classes per classification dataset", generated_summary["n_classes_range"]),
    ):
        if line is not None:
            lines.append(line)
    feature_types = generated_summary["feature_types"]
    if feature_types:
        lines.append(f"- observed feature types: {', '.join(feature_types)}")
    if curated_summary is not None:
        lines.append(f"- curated accepted datasets: {curated_summary['dataset_count']}")

    lines.extend(
        [
            "",
            "## Generation provenance",
            "",
            f"- source family: `{identity['source_family']}`",
            f"- generate run id: `{identity['generate_run_id']}`",
            f"- generated corpus id: `{identity['generated_corpus_id']}`",
        ]
    )
    if config_reference is not None:
        lines.append(f"- config reference: `{config_reference}`")
    if provenance is not None and isinstance(provenance, Mapping):
        target_derivation = provenance.get("target_derivation")
        if isinstance(target_derivation, str) and target_derivation.strip():
            lines.append(f"- target derivation: `{target_derivation.strip()}`")
        intervention = provenance.get("intervention")
        if isinstance(intervention, Mapping):
            mode = intervention.get("mode")
            if isinstance(mode, str) and mode.strip():
                lines.append(f"- intervention mode: `{mode.strip()}`")
    missingness_summary = _extract_missingness_summary(run_context)
    if missingness_summary is not None:
        lines.append(f"- missingness: `{missingness_summary}`")

    lines.extend(
        [
            "",
            "## Publishing workflow",
            "",
            "This repo is produced from a local handoff root with:",
            "",
            "```bash",
            "dagzoo generate --config recipe:<name> --num-datasets <n> --handoff-root handoffs/<run_name>",
            "dagzoo publish hub --handoff-root handoffs/<run_name> --repo-id <namespace/name>",
            "```",
            "",
            "Only the public handoff artifacts are uploaded. Local `internal/` sidecars stay on disk for dagzoo tooling and are not published to the Hub.",
            "",
            "## Links",
            "",
            "- dagzoo repo: [github.com/bensonlee5/dagzoo](https://github.com/bensonlee5/dagzoo)",
            "- dagzoo docs: [bensonlee5.github.io/dagzoo/docs/](https://bensonlee5.github.io/dagzoo/docs/)",
        ]
    )
    if recipe_citations:
        lines.extend(["", "## References", ""])
        lines.extend(f"- {citation}" for citation in recipe_citations)

    return "\n".join(lines).rstrip() + "\n"


def _copy_tree(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest, dirs_exist_ok=True)


def _stage_public_tree(
    *,
    staging_dir: Path,
    handoff_root: Path,
    generated_dir: Path,
    curated_dir: Path | None,
    card_text: str,
) -> None:
    _copy_tree(generated_dir, staging_dir / "generated")
    if curated_dir is not None:
        _copy_tree(curated_dir, staging_dir / "curated")
    shutil.copy2(handoff_root / HANDOFF_MANIFEST_FILENAME, staging_dir / HANDOFF_MANIFEST_FILENAME)
    (staging_dir / "README.md").write_text(card_text, encoding="utf-8")


def _hub_auth_error_message() -> str:
    return (
        "Hugging Face authentication is required. Run `hf auth login` or set `HF_TOKEN`, "
        "then retry `dagzoo publish hub`."
    )


def _normalize_hub_error(exc: Exception) -> RuntimeError:
    if isinstance(exc, LocalTokenNotFoundError):
        return RuntimeError(_hub_auth_error_message())
    if isinstance(exc, HfHubHTTPError):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {401, 403}:
            return RuntimeError(_hub_auth_error_message())
        return RuntimeError(f"Hugging Face Hub request failed: {exc}")
    return RuntimeError(str(exc))


def publish_handoff_to_hub(
    *,
    handoff_root: str | Path,
    repo_id: str,
    private: bool = False,
    license_id: str | None = None,
) -> HubPublishResult:
    """Publish one dagzoo handoff root to a Hugging Face Hub dataset repository."""

    resolved_handoff_root = Path(handoff_root).resolve()
    if not resolved_handoff_root.exists():
        raise FileNotFoundError(f"Handoff root does not exist: {resolved_handoff_root}")
    if not resolved_handoff_root.is_dir():
        raise ValueError(f"Handoff root must be a directory: {resolved_handoff_root}")

    manifest_path = resolved_handoff_root / HANDOFF_MANIFEST_FILENAME
    handoff_manifest = _read_json_mapping(manifest_path, label="handoff manifest")
    validate_generate_handoff_manifest(handoff_manifest)

    artifacts_relative = handoff_manifest.get("artifacts_relative")
    if not isinstance(artifacts_relative, Mapping):
        raise ValueError(f"{HANDOFF_MANIFEST_FILENAME} must contain `artifacts_relative`.")
    generated_dir = _resolve_relative_dir(
        resolved_handoff_root,
        relative_path=artifacts_relative.get("generated_dir"),
        field_name="artifacts_relative.generated_dir",
        required=True,
    )
    assert generated_dir is not None
    curated_dir = _resolve_relative_dir(
        resolved_handoff_root,
        relative_path=artifacts_relative.get("curated_dir"),
        field_name="artifacts_relative.curated_dir",
        required=False,
    )

    run_context_path = resolved_handoff_root / INTERNAL_DIRNAME / RUN_CONTEXT_FILENAME
    run_context = (
        _read_json_mapping(run_context_path, label="run context")
        if run_context_path.exists()
        else None
    )
    generated_summary = _summarize_corpus(generated_dir)
    curated_summary = _summarize_corpus(curated_dir) if curated_dir is not None else None
    card_text = build_hub_dataset_card(
        handoff_manifest=handoff_manifest,
        generated_summary=generated_summary,
        curated_summary=curated_summary,
        run_context=run_context,
        repo_id=repo_id,
        license_id=license_id,
    )

    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        with TemporaryDirectory(prefix="dagzoo_hub_publish_") as staging_tmp:
            staging_dir = Path(staging_tmp)
            _stage_public_tree(
                staging_dir=staging_dir,
                handoff_root=resolved_handoff_root,
                generated_dir=generated_dir,
                curated_dir=curated_dir,
                card_text=card_text,
            )
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(staging_dir),
                delete_patterns=list(_MANAGED_DELETE_PATTERNS),
                commit_message="Publish dagzoo corpus",
            )
    except (LocalTokenNotFoundError, HfHubHTTPError, OSError, ValueError) as exc:
        raise _normalize_hub_error(exc) from exc

    return HubPublishResult(
        repo_id=repo_id,
        repo_url=f"{HF_DATASET_BASE_URL}/{repo_id}",
        generated_datasets=int(generated_summary["dataset_count"]),
        curated_datasets=None if curated_summary is None else int(curated_summary["dataset_count"]),
    )


__all__ = [
    "HubPublishResult",
    "build_hub_dataset_card",
    "publish_handoff_to_hub",
]
