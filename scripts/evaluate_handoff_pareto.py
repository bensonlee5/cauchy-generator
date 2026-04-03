#!/usr/bin/env python
"""Run matched handoff-root evaluations for RD-005-style generator variants."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import click
import numpy as np
import yaml

try:
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover - optional dependency at import time
    raise RuntimeError("pyarrow is required for handoff evaluation.") from exc

from dagzoo.cli.entrypoint import main as dagzoo_main
from dagzoo.config import GeneratorConfig
from dagzoo.diagnostics.effective_diversity.compare import compare_coverage_summaries
from dagzoo.io.shard_contract import DATASET_CATALOG_FILENAME, iter_ndjson_records

_RIDGE_LAMBDA = 1e-2
_EASY_TASK_CEILING_MARGIN = 0.10
_SUPPORTING_METRICS = (
    "graph_edge_density",
    "graph_depth_ratio",
    "graph_reachability_ratio",
    "graph_ancestor_overlap_mean",
    "graph_target_ancestor_fraction",
    "mechanism_family_cooccurrence_ratio",
)
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@dataclass(slots=True)
class _VariantSpec:
    label: str
    config_path: Path
    regime_id: str


@dataclass(slots=True)
class _CliArgs:
    baseline_config: str
    out_root: str
    variant_config: tuple[str, ...]
    stress_profile: tuple[str, ...]
    num_datasets: int
    seed: int
    device: str
    hardware_policy: str
    rows: str | None
    warn_threshold_pct: float
    fail_threshold_pct: float
    reuse_existing: bool


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.strip())
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed.lower() or "variant"


def _build_variant_specs(args: _CliArgs, *, temp_dir: Path) -> list[_VariantSpec]:
    baseline_config_path = Path(args.baseline_config).resolve()
    specs: list[_VariantSpec] = [
        _VariantSpec(
            label="baseline",
            config_path=baseline_config_path,
            regime_id="baseline",
        )
    ]
    for config_path_text in args.variant_config:
        config_path = Path(config_path_text).resolve()
        specs.append(
            _VariantSpec(
                label=f"config:{config_path.stem}",
                config_path=config_path,
                regime_id=f"config:{config_path.stem}",
            )
        )
    if args.stress_profile:
        baseline_cfg = GeneratorConfig.from_yaml(str(baseline_config_path))
        for profile in args.stress_profile:
            cfg = GeneratorConfig.from_dict(baseline_cfg.to_dict())
            cfg.stress.profile = str(profile)
            temp_path = temp_dir / f"{_sanitize_label(f'stress_{profile}')}.yaml"
            temp_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
            specs.append(
                _VariantSpec(
                    label=f"stress:{profile}",
                    config_path=temp_path,
                    regime_id=f"stress:{profile}",
                )
            )
    if len(specs) <= 1:
        raise ValueError("Provide at least one --variant-config or --stress-profile.")
    return specs


def _run_generate(
    spec: _VariantSpec,
    *,
    args: _CliArgs,
    out_root: Path,
) -> tuple[Path, float]:
    run_root = out_root / _sanitize_label(spec.label)
    manifest_path = run_root / "handoff_manifest.json"
    if manifest_path.exists():
        if not args.reuse_existing:
            raise RuntimeError(
                f"Handoff root already exists for {spec.label!r}: {run_root}. "
                "Pass --reuse-existing to reuse prior artifacts."
            )
        return run_root, math.nan

    cli_args = [
        "generate",
        "--config",
        str(spec.config_path),
        "--num-datasets",
        str(int(args.num_datasets)),
        "--device",
        str(args.device),
        "--hardware-policy",
        str(args.hardware_policy),
        "--handoff-root",
        str(run_root),
        "--seed",
        str(int(args.seed)),
        "--diagnostics",
    ]
    if args.rows is not None:
        cli_args.extend(["--rows", str(args.rows)])

    started_at = perf_counter()
    exit_code = dagzoo_main(cli_args)
    elapsed_seconds = perf_counter() - started_at
    if exit_code != 0:
        raise RuntimeError(
            f"`dagzoo generate` failed for {spec.label!r} with exit code {exit_code}."
        )
    return run_root, float(elapsed_seconds)


def _catalog_task(run_root: Path) -> str:
    catalog_paths = sorted((run_root / "generated").glob(f"shard_*/{DATASET_CATALOG_FILENAME}"))
    for catalog_path in catalog_paths:
        for record in iter_ndjson_records(catalog_path):
            task = record.get("task")
            if isinstance(task, str) and task.strip():
                return task.strip().lower()
    raise RuntimeError(f"Unable to infer task from shard catalogs under {run_root}.")


def _load_split_arrays(run_root: Path, *, split: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    split_paths = sorted((run_root / "generated").glob(f"shard_*/{split}.parquet"))
    if not split_paths:
        raise RuntimeError(f"No {split}.parquet files found under {run_root / 'generated'}.")
    grouped: dict[int, dict[str, list[Any]]] = {}
    for split_path in split_paths:
        table = pq.read_table(split_path, columns=["dataset_index", "x", "y"])
        dataset_indices = table.column("dataset_index").to_pylist()
        features = table.column("x").to_pylist()
        targets = table.column("y").to_pylist()
        for dataset_index, x_row, y_value in zip(
            dataset_indices,
            features,
            targets,
            strict=True,
        ):
            idx = int(dataset_index)
            entry = grouped.setdefault(idx, {"x": [], "y": []})
            entry["x"].append([float(value) for value in x_row])
            entry["y"].append(float(y_value))
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for dataset_index, payload in grouped.items():
        arrays[int(dataset_index)] = (
            np.asarray(payload["x"], dtype=np.float32),
            np.asarray(payload["y"], dtype=np.float32),
        )
    return arrays


def _standardize_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean_vec = np.mean(x_train, axis=0, keepdims=True)
    std_vec = np.std(x_train, axis=0, keepdims=True)
    std_vec = np.where(std_vec > 1e-6, std_vec, 1.0)
    return (
        (x_train - mean_vec) / std_vec,
        (x_test - mean_vec) / std_vec,
    )


def _ridge_weights(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate(
        [x_train.astype(np.float64), np.ones((x_train.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    y = y_train.astype(np.float64)
    gram = x_aug.T @ x_aug
    ridge = _RIDGE_LAMBDA * np.eye(gram.shape[0], dtype=np.float64)
    ridge[-1, -1] = 0.0
    return np.linalg.solve(gram + ridge, x_aug.T @ y)


def _score_regression_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    x_train_std, x_test_std = _standardize_features(x_train, x_test)
    weights = _ridge_weights(x_train_std, y_train)
    x_test_aug = np.concatenate(
        [x_test_std.astype(np.float64), np.ones((x_test_std.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    predictions = x_test_aug @ weights
    centered = y_test.astype(np.float64) - np.mean(y_test.astype(np.float64))
    denom = float(np.sum(centered**2))
    if denom <= 1e-12:
        return 0.0
    residual = y_test.astype(np.float64) - predictions
    return float(1.0 - (np.sum(residual**2) / denom))


def _score_classification_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    x_train_std, x_test_std = _standardize_features(x_train, x_test)
    labels = np.unique(np.concatenate([y_train, y_test]).astype(np.int64))
    label_to_index = {int(label): idx for idx, label in enumerate(labels.tolist())}
    train_targets = np.zeros((x_train_std.shape[0], len(labels)), dtype=np.float64)
    for row_index, raw_label in enumerate(y_train.astype(np.int64).tolist()):
        train_targets[row_index, label_to_index[int(raw_label)]] = 1.0
    weights = _ridge_weights(x_train_std, train_targets)
    x_test_aug = np.concatenate(
        [x_test_std.astype(np.float64), np.ones((x_test_std.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    logits = x_test_aug @ weights
    predicted = labels[np.argmax(logits, axis=1)]
    return float(np.mean(predicted.astype(np.int64) == y_test.astype(np.int64)))


def _score_run(run_root: Path) -> dict[str, Any]:
    task = _catalog_task(run_root)
    train_arrays = _load_split_arrays(run_root, split="train")
    test_arrays = _load_split_arrays(run_root, split="test")
    dataset_indices = sorted(set(train_arrays) & set(test_arrays))
    if not dataset_indices:
        raise RuntimeError(f"No aligned train/test datasets found under {run_root}.")
    scores: list[float] = []
    for dataset_index in dataset_indices:
        x_train, y_train = train_arrays[dataset_index]
        x_test, y_test = test_arrays[dataset_index]
        if task == "classification":
            score = _score_classification_dataset(x_train, y_train, x_test, y_test)
        elif task == "regression":
            score = _score_regression_dataset(x_train, y_train, x_test, y_test)
        else:
            raise RuntimeError(f"Unsupported task {task!r} for downstream evaluation.")
        if math.isfinite(score):
            scores.append(float(score))
    if not scores:
        raise RuntimeError(f"No finite downstream scores produced for {run_root}.")
    return {
        "task": task,
        "count": len(scores),
        "mean": float(mean(scores)),
        "median": float(median(scores)),
        "min": float(min(scores)),
        "max": float(max(scores)),
        "std": float(np.std(np.asarray(scores, dtype=np.float64))),
        "scores": [float(score) for score in scores],
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage_summary(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "internal" / "diagnostics_artifacts" / "coverage_summary.json"
    if not path.exists():
        return None
    return _load_json(path)


def _supporting_metric_snapshot(summary: dict[str, Any] | None) -> dict[str, float | None]:
    if summary is None:
        return {metric: None for metric in _SUPPORTING_METRICS}
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return {metric: None for metric in _SUPPORTING_METRICS}
    snapshot: dict[str, float | None] = {}
    for metric in _SUPPORTING_METRICS:
        payload = metrics.get(metric)
        if isinstance(payload, dict) and isinstance(payload.get("mean"), (int, float)):
            snapshot[metric] = float(payload["mean"])
        else:
            snapshot[metric] = None
    return snapshot


def _datasets_per_minute(num_datasets: int, elapsed_seconds: float) -> float | None:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        return None
    return float((float(num_datasets) / elapsed_seconds) * 60.0)


def _entry_payload(
    spec: _VariantSpec,
    *,
    run_root: Path,
    num_datasets: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    handoff_manifest = _load_json(run_root / "handoff_manifest.json")
    coverage_summary = _coverage_summary(run_root)
    downstream = _score_run(run_root)
    return {
        "label": spec.label,
        "regime_id": spec.regime_id,
        "config_path": str(spec.config_path),
        "run_root": str(run_root),
        "generated_corpus_id": handoff_manifest["identity"]["generated_corpus_id"],
        "generate_run_id": handoff_manifest["identity"]["generate_run_id"],
        "generated_datasets": int(num_datasets),
        "generation_elapsed_seconds": (
            None if not math.isfinite(elapsed_seconds) else float(elapsed_seconds)
        ),
        "datasets_per_minute": _datasets_per_minute(num_datasets, elapsed_seconds),
        "downstream": downstream,
        "supporting_metrics": _supporting_metric_snapshot(coverage_summary),
        "coverage_summary": coverage_summary,
    }


def _downstream_mean(entry: dict[str, Any]) -> float:
    return float(entry["downstream"]["mean"])


def _datasets_per_minute_value(entry: dict[str, Any]) -> float:
    value = entry.get("datasets_per_minute")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _structural_diversity_value(entry: dict[str, Any]) -> float:
    value = entry.get("structural_diversity_composite_shift_pct")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _easy_task_ceiling_threshold(baseline_entry: dict[str, Any]) -> float:
    return float(_downstream_mean(baseline_entry) + _EASY_TASK_CEILING_MARGIN)


def _passes_easy_task_ceiling(entry: dict[str, Any], *, baseline_entry: dict[str, Any]) -> bool:
    return bool(_downstream_mean(entry) <= _easy_task_ceiling_threshold(baseline_entry))


def _augment_variant_priority_fields(
    entry: dict[str, Any],
    *,
    baseline_entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        **entry,
        "easy_task_ceiling_pass": _passes_easy_task_ceiling(entry, baseline_entry=baseline_entry),
        "easy_task_ceiling_margin": float(_EASY_TASK_CEILING_MARGIN),
        "easy_task_ceiling_downstream_mean": _easy_task_ceiling_threshold(baseline_entry),
    }


def _rd005_priority_sort_key(entry: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -_structural_diversity_value(entry),
        -_datasets_per_minute_value(entry),
        _downstream_mean(entry),
    )


def _rank_variants_for_rd005(variants: list[dict[str, Any]]) -> list[str]:
    ranked = sorted(variants, key=_rd005_priority_sort_key)
    return [str(entry["label"]) for entry in ranked]


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_diversity = _structural_diversity_value(left)
    right_diversity = _structural_diversity_value(right)
    left_throughput = _datasets_per_minute_value(left)
    right_throughput = _datasets_per_minute_value(right)
    no_worse = left_diversity >= right_diversity and left_throughput >= right_throughput
    strictly_better = left_diversity > right_diversity or left_throughput > right_throughput
    return bool(no_worse and strictly_better)


def _pareto_frontier(entries: list[dict[str, Any]]) -> list[str]:
    frontier: list[str] = []
    for index, candidate in enumerate(entries):
        dominated = False
        for other_index, other in enumerate(entries):
            if index == other_index:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate["label"]))
    return frontier


def _write_markdown_report(report: dict[str, Any], *, out_path: Path) -> None:
    lines = [
        "# RD-005 Handoff Pareto Evaluation",
        "",
        f"- Baseline: `{report['baseline']['label']}`",
        f"- Variants: {len(report['variants'])}",
        f"- Datasets per run: {report['summary']['num_datasets']}",
        f"- Shared seed: {report['summary']['seed']}",
        f"- Device: `{report['summary']['device']}`",
        "- Objective order: structural diversity, throughput, then lower downstream mean",
        "- Easy-task ceiling downstream mean: "
        f"{report['summary']['easy_task_ceiling_downstream_mean']:.4f}",
        "",
        "## RD-005 Priority Order",
        "",
    ]
    for label in report["summary"]["priority_variant_labels"]:
        lines.append(f"- `{label}`")
    lines.extend(["", "## Structural Frontier", ""])
    for label in report["summary"]["pareto_frontier_labels"]:
        lines.append(f"- `{label}`")
    lines.extend(["", "## Runs", ""])
    for entry in [report["baseline"], *report["variants"]]:
        lines.extend(
            [
                f"### {entry['label']}",
                "",
                f"- Regime id: `{entry['regime_id']}`",
                f"- Generated corpus id: `{entry['generated_corpus_id']}`",
                f"- Downstream mean: {entry['downstream']['mean']:.4f}",
                f"- Downstream median: {entry['downstream']['median']:.4f}",
                "- Datasets/minute: "
                + (
                    f"{entry['datasets_per_minute']:.2f}"
                    if entry.get("datasets_per_minute") is not None
                    else "reused"
                ),
            ]
        )
        if "diversity_status" in entry:
            lines.append(f"- Diversity status: `{entry['diversity_status']}`")
            lines.append(
                "- Diversity composite shift pct: "
                f"{float(entry.get('diversity_composite_shift_pct') or 0.0):.2f}"
            )
            lines.append(
                "- Structural diversity composite shift pct: "
                f"{float(entry.get('structural_diversity_composite_shift_pct') or 0.0):.2f}"
            )
            lines.append(
                f"- Easy-task ceiling pass: `{bool(entry.get('easy_task_ceiling_pass', False))}`"
            )
        lines.append("- Supporting metrics:")
        for metric, value in entry["supporting_metrics"].items():
            rendered = "-" if value is None else f"{float(value):.4f}"
            lines.append(f"  - `{metric}`: {rendered}")
        lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Generate matched handoff corpora and compare downstream score, diversity shift, "
        "and throughput for baseline vs RD-005 variants."
    ),
)
@click.option("--baseline-config", required=True, help="Baseline config YAML path.")
@click.option("--out-root", required=True, help="Output root for generated runs.")
@click.option(
    "--variant-config",
    multiple=True,
    default=(),
    help="Additional full config YAML path to evaluate as one variant. Repeatable.",
)
@click.option(
    "--stress-profile",
    multiple=True,
    default=(),
    help="Stress profile name to evaluate by applying it to the baseline config. Repeatable.",
)
@click.option("--num-datasets", type=int, default=8, show_default=True, help="Datasets per run.")
@click.option("--seed", type=int, default=0, show_default=True, help="Shared generation seed.")
@click.option("--device", default="cpu", show_default=True, help="Generation device.")
@click.option(
    "--hardware-policy",
    default="none",
    show_default=True,
    help="Hardware policy passed through to `dagzoo generate`.",
)
@click.option("--rows", default=None, help="Optional rows override.")
@click.option(
    "--warn-threshold-pct",
    type=float,
    default=2.5,
    show_default=True,
    help="Warn threshold passed to diversity-summary comparison.",
)
@click.option(
    "--fail-threshold-pct",
    type=float,
    default=5.0,
    show_default=True,
    help="Fail threshold passed to diversity-summary comparison.",
)
@click.option(
    "--reuse-existing",
    is_flag=True,
    help="Reuse existing handoff runs under out-root when a manifest already exists.",
)
def cli(
    *,
    baseline_config: str,
    out_root: str,
    variant_config: tuple[str, ...],
    stress_profile: tuple[str, ...],
    num_datasets: int,
    seed: int,
    device: str,
    hardware_policy: str,
    rows: str | None,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    reuse_existing: bool,
) -> int:
    args = _CliArgs(
        baseline_config=baseline_config,
        out_root=out_root,
        variant_config=variant_config,
        stress_profile=stress_profile,
        num_datasets=num_datasets,
        seed=seed,
        device=device,
        hardware_policy=hardware_policy,
        rows=rows,
        warn_threshold_pct=warn_threshold_pct,
        fail_threshold_pct=fail_threshold_pct,
        reuse_existing=reuse_existing,
    )
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="rd005_pareto_", dir=str(out_root)) as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        specs = _build_variant_specs(args, temp_dir=temp_dir)
        results: list[dict[str, Any]] = []
        for spec in specs:
            run_root, elapsed_seconds = _run_generate(spec, args=args, out_root=out_root)
            results.append(
                _entry_payload(
                    spec,
                    run_root=run_root,
                    num_datasets=int(args.num_datasets),
                    elapsed_seconds=elapsed_seconds,
                )
            )

    baseline = results[0]
    variants: list[dict[str, Any]] = []
    for entry in results[1:]:
        if str(entry["downstream"]["task"]) != str(baseline["downstream"]["task"]):
            raise RuntimeError(
                "All variants must resolve to the same task as the baseline. "
                f"Baseline task={baseline['downstream']['task']!r}, "
                f"variant {entry['label']!r} task={entry['downstream']['task']!r}."
            )
        comparison = None
        if baseline["coverage_summary"] is not None and entry["coverage_summary"] is not None:
            comparison = compare_coverage_summaries(
                baseline_summary=baseline["coverage_summary"],
                variant_summary=entry["coverage_summary"],
                warn_threshold_pct=float(args.warn_threshold_pct),
                fail_threshold_pct=float(args.fail_threshold_pct),
                include_structural_summary=True,
            )
        if comparison is not None:
            entry = {
                **entry,
                "diversity_status": comparison["diversity_status"],
                "diversity_composite_shift_pct": comparison["diversity_composite_shift_pct"],
                "diversity_metric_shift_pct": comparison["diversity_metric_shift_pct"],
                "structural_diversity_composite_shift_pct": comparison[
                    "structural_diversity_composite_shift_pct"
                ],
                "structural_diversity_metric_shift_pct": comparison[
                    "structural_diversity_metric_shift_pct"
                ],
            }
        variants.append(_augment_variant_priority_fields(entry, baseline_entry=baseline))

    eligible_frontier_entries = [
        baseline,
        *[v for v in variants if v.get("easy_task_ceiling_pass")],
    ]
    frontier_labels = _pareto_frontier(eligible_frontier_entries)
    priority_variant_labels = _rank_variants_for_rd005(variants)
    report = {
        "schema_name": "dagzoo_rd005_handoff_pareto_report",
        "schema_version": 1,
        "baseline": baseline,
        "variants": variants,
        "summary": {
            "num_datasets": int(args.num_datasets),
            "seed": int(args.seed),
            "device": str(args.device),
            "hardware_policy": str(args.hardware_policy),
            "warn_threshold_pct": float(args.warn_threshold_pct),
            "fail_threshold_pct": float(args.fail_threshold_pct),
            "easy_task_ceiling_margin": float(_EASY_TASK_CEILING_MARGIN),
            "easy_task_ceiling_downstream_mean": _easy_task_ceiling_threshold(baseline),
            "priority_variant_labels": priority_variant_labels,
            "pareto_frontier_labels": frontier_labels,
        },
    }
    json_path = out_root / "pareto_summary.json"
    md_path = out_root / "pareto_summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown_report(report, out_path=md_path)
    print(f"Wrote Pareto summary: {json_path}")
    print(f"Wrote Pareto markdown: {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        result = cli.main(args=argv, prog_name="evaluate_handoff_pareto.py", standalone_mode=False)
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return int(exc.exit_code)
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.Abort:
        return 1
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
