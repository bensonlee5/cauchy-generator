#!/usr/bin/env python
"""Run the full RD-005 follow-on promotion suite."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import yaml

from dagzoo.config import GeneratorConfig, clone_generator_config
from dagzoo.diagnostics.effective_diversity import (
    run_effective_diversity_audit,
    write_effective_diversity_artifacts,
)
from dagzoo.diagnostics.rd005_follow_on import (
    DEFAULT_THROUGHPUT_FLOOR_RATIO,
    RD005_FOLLOW_ON_LANES,
    build_rd005_follow_on_report,
    write_rd005_follow_on_artifacts,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class _VariantInput:
    label: str
    stress_profile: str
    config_path: Path
    config: GeneratorConfig


def _load_repo_script_module(module_name: str, rel_path: str):
    """Load one repo script module from a repo-relative path."""

    script_path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_variant_inputs(
    *,
    baseline_config: GeneratorConfig,
    out_dir: Path,
) -> list[_VariantInput]:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants: list[_VariantInput] = []
    for lane in RD005_FOLLOW_ON_LANES:
        config = clone_generator_config(baseline_config, revalidate=False)
        config.stress.profile = lane.stress_profile
        config_path = out_dir / f"{lane.label}.yaml"
        config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
        variants.append(
            _VariantInput(
                label=lane.label,
                stress_profile=lane.stress_profile,
                config_path=config_path,
                config=config,
            )
        )
    return variants


def _run_diversity_audit_suite(
    *,
    baseline_config: GeneratorConfig,
    baseline_config_path: Path,
    variants: list[_VariantInput],
    suite: str,
    num_datasets: int,
    warmup: int | None,
    device: str,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    out_dir: Path,
    reuse_existing: bool,
) -> tuple[dict[str, Path], dict[str, Any]]:
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    if reuse_existing and summary_json.exists() and summary_md.exists():
        return {"summary_json": summary_json, "summary_md": summary_md}, _load_json(summary_json)

    report = run_effective_diversity_audit(
        baseline_config=baseline_config,
        baseline_config_path=str(baseline_config_path),
        variant_configs=[variant.config for variant in variants],
        variant_config_paths=[str(variant.config_path) for variant in variants],
        variant_labels=[variant.label for variant in variants],
        suite=suite,
        num_datasets=num_datasets,
        warmup=warmup,
        device=device,
        warn_threshold_pct=warn_threshold_pct,
        fail_threshold_pct=fail_threshold_pct,
    )
    artifact_paths = write_effective_diversity_artifacts(report, out_dir=out_dir)
    return artifact_paths, _load_json(artifact_paths["summary_json"])


def _run_parity_report(
    *,
    summary_json: Path,
    out_dir: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    module = _load_repo_script_module(
        "evaluate_rd005_follow_on_parity_report",
        "scripts/render_tabiclv2_parity_report.py",
    )
    exit_code = module.main(["--summary-json", str(summary_json), "--out-dir", str(out_dir)])
    if exit_code != 0:
        raise RuntimeError(f"render_tabiclv2_parity_report.py failed with exit code {exit_code}.")
    json_path = out_dir / "parity_report.json"
    md_path = out_dir / "parity_report.md"
    return {"summary_json": json_path, "summary_md": md_path}, _load_json(json_path)


def _run_pareto_suite(
    *,
    baseline_config_path: Path,
    variants: list[_VariantInput],
    out_dir: Path,
    num_datasets: int,
    seed: int,
    device: str,
    hardware_policy: str,
    rows: str | None,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    reuse_existing: bool,
) -> tuple[dict[str, Path], dict[str, Any]]:
    module = _load_repo_script_module(
        "evaluate_rd005_follow_on_pareto",
        "scripts/evaluate_handoff_pareto.py",
    )
    argv = [
        "--baseline-config",
        str(baseline_config_path),
        "--out-root",
        str(out_dir),
        "--num-datasets",
        str(int(num_datasets)),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--hardware-policy",
        str(hardware_policy),
        "--warn-threshold-pct",
        str(float(warn_threshold_pct)),
        "--fail-threshold-pct",
        str(float(fail_threshold_pct)),
    ]
    if rows is not None:
        argv.extend(["--rows", str(rows)])
    if reuse_existing:
        argv.append("--reuse-existing")
    for variant in variants:
        argv.extend(["--variant-config", str(variant.config_path)])
        argv.extend(["--variant-label", variant.label])
    exit_code = module.main(argv)
    if exit_code != 0:
        raise RuntimeError(f"evaluate_handoff_pareto.py failed with exit code {exit_code}.")
    json_path = out_dir / "pareto_summary.json"
    md_path = out_dir / "pareto_summary.md"
    return {"summary_json": json_path, "summary_md": md_path}, _load_json(json_path)


@click.command(
    context_settings=CONTEXT_SETTINGS,
    help="Run the full RD-005 follow-on suite and write a promotion decision summary.",
)
@click.option(
    "--baseline-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Baseline config YAML path.",
)
@click.option(
    "--out-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output root for the full follow-on suite.",
)
@click.option(
    "--suite",
    type=click.Choice(["smoke", "standard"]),
    default="smoke",
    show_default=True,
    help="Diversity-audit probe suite.",
)
@click.option(
    "--num-datasets",
    type=int,
    default=8,
    show_default=True,
    help="Datasets per run for diversity and handoff evaluation.",
)
@click.option(
    "--warmup",
    type=int,
    default=None,
    help="Optional diversity-audit warmup override.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Shared generation seed.")
@click.option("--device", default="cpu", show_default=True, help="Generation device.")
@click.option(
    "--hardware-policy",
    default="none",
    show_default=True,
    help="Hardware policy passed through to `dagzoo generate`.",
)
@click.option("--rows", default=None, help="Optional rows override for handoff generation.")
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
    "--throughput-floor-ratio",
    type=float,
    default=DEFAULT_THROUGHPUT_FLOOR_RATIO,
    show_default=True,
    help="Minimum datasets/minute ratio vs baseline required for promotion.",
)
@click.option(
    "--reuse-existing",
    is_flag=True,
    help="Reuse existing diversity-audit and handoff artifacts when present.",
)
def cli(
    *,
    baseline_config: Path,
    out_root: Path,
    suite: str,
    num_datasets: int,
    warmup: int | None,
    seed: int,
    device: str,
    hardware_policy: str,
    rows: str | None,
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    throughput_floor_ratio: float,
    reuse_existing: bool,
) -> int:
    if throughput_floor_ratio <= 0.0:
        raise click.ClickException("--throughput-floor-ratio must be > 0.")

    baseline_config_path = baseline_config.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_cfg = GeneratorConfig.from_yaml(str(baseline_config_path))
    variant_inputs = _write_variant_inputs(
        baseline_config=baseline_cfg,
        out_dir=out_root / "variant_inputs",
    )

    diversity_paths, diversity_report = _run_diversity_audit_suite(
        baseline_config=baseline_cfg,
        baseline_config_path=baseline_config_path,
        variants=variant_inputs,
        suite=suite,
        num_datasets=int(num_datasets),
        warmup=warmup,
        device=str(device),
        warn_threshold_pct=float(warn_threshold_pct),
        fail_threshold_pct=float(fail_threshold_pct),
        out_dir=out_root / "diversity_audit",
        reuse_existing=bool(reuse_existing),
    )
    diversity_report.setdefault("summary", {})
    diversity_report["summary"]["source_summary_json"] = str(diversity_paths["summary_json"])

    parity_paths, parity_report = _run_parity_report(
        summary_json=diversity_paths["summary_json"],
        out_dir=out_root / "diversity_audit" / "parity_report",
    )
    parity_report.setdefault("summary", {})
    parity_report["summary"]["source_parity_report_json"] = str(parity_paths["summary_json"])

    pareto_paths, pareto_report = _run_pareto_suite(
        baseline_config_path=baseline_config_path,
        variants=variant_inputs,
        out_dir=out_root / "handoff_pareto",
        num_datasets=int(num_datasets),
        seed=int(seed),
        device=str(device),
        hardware_policy=str(hardware_policy),
        rows=rows,
        warn_threshold_pct=float(warn_threshold_pct),
        fail_threshold_pct=float(fail_threshold_pct),
        reuse_existing=bool(reuse_existing),
    )
    pareto_report.setdefault("summary", {})
    pareto_report["summary"]["source_pareto_summary_json"] = str(pareto_paths["summary_json"])

    report = build_rd005_follow_on_report(
        baseline_config_path=str(baseline_config_path),
        diversity_report=diversity_report,
        parity_report=parity_report,
        pareto_report=pareto_report,
        throughput_floor_ratio=float(throughput_floor_ratio),
    )
    report.setdefault("artifacts", {})
    report["artifacts"].update(
        {
            "diversity_summary_json": str(diversity_paths["summary_json"]),
            "diversity_summary_md": str(diversity_paths["summary_md"]),
            "parity_report_json": str(parity_paths["summary_json"]),
            "parity_report_md": str(parity_paths["summary_md"]),
            "pareto_summary_json": str(pareto_paths["summary_json"]),
            "pareto_summary_md": str(pareto_paths["summary_md"]),
        }
    )
    artifact_paths = write_rd005_follow_on_artifacts(report, out_dir=out_root)
    print(f"Wrote follow-on promotion summary: {artifact_paths['summary_json']}")
    print(f"Wrote follow-on promotion markdown: {artifact_paths['summary_md']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        result = cli.main(
            args=argv,
            prog_name="evaluate_rd005_follow_on_suite.py",
            standalone_mode=False,
        )
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
