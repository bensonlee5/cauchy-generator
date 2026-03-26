"""Generate command handler."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

from dagzoo.config import clone_generator_config
from dagzoo.core.config_resolution import (
    append_config_diff_events,
    resolve_generate_config,
    serialize_resolution_events,
)
from dagzoo.core.dataset import generate_batch_iter
from dagzoo.core.fixed_layout.runtime import realize_generation_config_for_run
from dagzoo.core.generate_handoff import (
    HANDOFF_MANIFEST_FILENAME,
    write_generate_handoff_manifest,
)
from dagzoo.diagnostics import (
    CoverageAggregator,
    write_coverage_summary_json,
    write_coverage_summary_markdown,
)
from dagzoo.diagnostics_targets import build_diagnostics_aggregation_config
from dagzoo.filtering.deferred_filter import MANIFEST_FILENAME, SUMMARY_FILENAME
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream

from ..common import load_config_or_usage_error, raise_usage_error
from ..effective_config import (
    print_effective_config,
    print_resolution_trace,
    write_effective_config,
    write_effective_config_trace,
)


def _write_generate_diagnostics_artifacts(
    diagnostics_aggregator: CoverageAggregator,
    *,
    diagnostics_out_dir: Path,
) -> None:
    """Write generation diagnostics coverage artifacts and print output paths."""

    summary = diagnostics_aggregator.build_summary()
    json_path = write_coverage_summary_json(summary, diagnostics_out_dir / "coverage_summary.json")
    md_path = write_coverage_summary_markdown(summary, diagnostics_out_dir / "coverage_summary.md")
    print(f"Wrote diagnostics artifacts: {json_path} and {md_path}")


def _generate_handoff_dir(run_root: Path) -> Path:
    """Return the fixed handoff generated artifact directory."""

    return run_root / "generated"


def _ensure_generate_handoff_output_safe(run_root: Path) -> Path:
    """Fail fast when one generate handoff root already holds prior artifacts."""

    if run_root.exists() and not run_root.is_dir():
        raise RuntimeError(f"Generate handoff root must be a directory path: {run_root}")

    generated_dir = _generate_handoff_dir(run_root)
    filter_dir = run_root / "filter"
    curated_dir = run_root / "curated"
    for path in (generated_dir, filter_dir, curated_dir):
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"Generate handoff artifact path must be a directory: {path}")

    stale_generated = next(generated_dir.glob("shard_*"), None) if generated_dir.exists() else None
    if stale_generated is not None:
        raise RuntimeError(
            "Generate handoff output already contains shard data: "
            f"{generated_dir}. Remove existing shard_* folders or choose a new handoff root."
        )

    if filter_dir.exists() and (
        (filter_dir / MANIFEST_FILENAME).exists() or (filter_dir / SUMMARY_FILENAME).exists()
    ):
        raise RuntimeError(
            "Generate handoff root already contains prior filter artifacts: "
            f"{filter_dir}. Remove prior filter artifacts or choose a new handoff root."
        )

    stale_curated = next(curated_dir.glob("shard_*"), None) if curated_dir.exists() else None
    if stale_curated is not None:
        raise RuntimeError(
            "Generate handoff root already contains curated shard data: "
            f"{curated_dir}. Remove existing shard_* folders or choose a new handoff root."
        )

    handoff_manifest_path = run_root / HANDOFF_MANIFEST_FILENAME
    if handoff_manifest_path.exists():
        raise RuntimeError(
            "Generate handoff root already contains a prior handoff manifest: "
            f"{handoff_manifest_path}. Remove the existing manifest or choose a new handoff root."
        )

    return generated_dir


def _build_generate_invocation_overrides(
    args: argparse.Namespace,
    *,
    handoff_root: Path,
) -> dict[str, Any]:
    """Serialize user-supplied CLI overrides for the handoff manifest."""

    diagnostics_out_dir = (
        str(Path(args.diagnostics_out_dir).resolve())
        if args.diagnostics_out_dir is not None
        else None
    )
    return {
        "num_datasets": int(args.num_datasets),
        "seed": int(args.seed) if args.seed is not None else None,
        "rows": args.rows,
        "device": args.device,
        "hardware_policy": str(args.hardware_policy),
        "missing_rate": args.missing_rate,
        "missing_mechanism": args.missing_mechanism,
        "missing_mar_observed_fraction": args.missing_mar_observed_fraction,
        "missing_mar_logit_scale": args.missing_mar_logit_scale,
        "missing_mnar_logit_scale": args.missing_mnar_logit_scale,
        "diagnostics": bool(args.diagnostics),
        "diagnostics_out_dir": diagnostics_out_dir,
        "handoff_root": str(handoff_root.resolve()),
    }


def run_generate_command(args: argparse.Namespace) -> int:
    """Execute the ``generate`` command."""

    handoff_root: Path | None = None
    generated_dir: Path | None = None
    if args.handoff_root is not None:
        if args.out is not None:
            raise_usage_error("`--handoff-root` cannot be combined with `--out`.")
        if args.no_dataset_write:
            raise_usage_error("`--handoff-root` cannot be combined with `--no-dataset-write`.")
        handoff_root = Path(args.handoff_root).resolve()
        try:
            generated_dir = _ensure_generate_handoff_output_safe(handoff_root)
        except RuntimeError as exc:
            raise_usage_error(str(exc))

    config = load_config_or_usage_error(args.config)
    try:
        resolved = resolve_generate_config(
            config,
            device_override=args.device,
            rows=args.rows,
            hardware_policy=str(args.hardware_policy),
            missing_rate=args.missing_rate,
            missing_mechanism=args.missing_mechanism,
            missing_mar_observed_fraction=args.missing_mar_observed_fraction,
            missing_mar_logit_scale=args.missing_mar_logit_scale,
            missing_mnar_logit_scale=args.missing_mnar_logit_scale,
            diagnostics_enabled=bool(args.diagnostics),
        )
    except ValueError as exc:
        raise_usage_error(str(exc))

    seed = args.seed if args.seed is not None else resolved.config.seed
    config, run_seed, requested_device, resolved_device = realize_generation_config_for_run(
        resolved.config,
        seed=seed,
        device=resolved.requested_device,
    )
    if bool(config.filter.enabled):
        raise_usage_error(
            "Inline filtering has been removed from generate. Set filter.enabled=false and run "
            "`dagzoo filter --in <shard_dir> --out <out_dir>` after generation."
        )
    hw = resolved.hardware
    trace_events = list(resolved.trace_events)
    append_config_diff_events(
        resolved.config,
        config,
        source="generate.run_realization",
        events=trace_events,
    )
    out_dir: str | Path | None = args.out or config.output.out_dir
    effective_config_root: str | Path | None = out_dir
    if handoff_root is not None:
        assert generated_dir is not None
        pre_handoff_config = clone_generator_config(config, revalidate=False)
        config.output.out_dir = str(generated_dir)
        append_config_diff_events(
            pre_handoff_config,
            config,
            source="generate.handoff_root",
            events=trace_events,
        )
        out_dir = generated_dir
        effective_config_root = generated_dir
    elif effective_config_root is None:
        effective_config_root = (
            args.diagnostics_out_dir or config.diagnostics.out_dir or "effective_config_artifacts"
        )

    trace_payload = serialize_resolution_events(trace_events)
    if args.print_effective_config:
        print_effective_config(config, header="Effective config:")

    effective_config_path = write_effective_config(
        config,
        Path(effective_config_root) / "effective_config.yaml",
    )
    trace_path = write_effective_config_trace(
        trace_payload,
        Path(effective_config_root) / "effective_config_trace.yaml",
    )
    print(f"Wrote effective config: {effective_config_path}")
    print(f"Wrote effective config trace: {trace_path}")
    if args.print_resolution_trace:
        print_resolution_trace(trace_payload, header="Resolution trace:")

    diagnostics_enabled = bool(config.diagnostics.enabled)
    diagnostics_out_dir: Path | None = None
    diagnostics_aggregator: CoverageAggregator | None = None
    if diagnostics_enabled:
        diagnostics_root = args.diagnostics_out_dir or config.diagnostics.out_dir or out_dir
        if diagnostics_root is None:
            diagnostics_root = "diagnostics_artifacts"
        diagnostics_out_dir = Path(diagnostics_root)
        diagnostics_aggregator = CoverageAggregator(build_diagnostics_aggregation_config(config))
    print(
        f"Hardware backend={hw.backend} device='{hw.device_name}' "
        f"memory_gb={hw.total_memory_gb} peak_flops={hw.peak_flops:.3e} tier={hw.tier} "
        f"hardware_policy={args.hardware_policy}"
    )

    stream: Iterator[Any] = generate_batch_iter(
        config,
        num_datasets=args.num_datasets,
        seed=run_seed,
        device=requested_device,
    )
    if diagnostics_aggregator is not None:
        base_stream = stream

        def _stream_with_diagnostics() -> Iterator[Any]:
            for bundle in base_stream:
                diagnostics_aggregator.update_bundle(bundle)
                yield bundle

        stream = _stream_with_diagnostics()

    if args.no_dataset_write:
        generated = sum(1 for _ in stream)
        if diagnostics_aggregator is not None:
            assert diagnostics_out_dir is not None
            _write_generate_diagnostics_artifacts(
                diagnostics_aggregator, diagnostics_out_dir=diagnostics_out_dir
            )
        print(f"Generated {generated} datasets (no-dataset-write mode).")
        return 0

    if out_dir is None:
        raise_usage_error(
            "No output directory resolved for generation. Set output.out_dir in the config or "
            "pass `--out` or `--handoff-root`."
        )
    resolved_out_dir: str | Path = out_dir
    generation_started_at = perf_counter()
    written = write_packed_parquet_shards_stream(
        stream,
        out_dir=resolved_out_dir,
        shard_size=config.output.shard_size,
        compression=config.output.compression,
    )
    generation_elapsed_seconds = perf_counter() - generation_started_at
    if diagnostics_aggregator is not None:
        assert diagnostics_out_dir is not None
        _write_generate_diagnostics_artifacts(
            diagnostics_aggregator, diagnostics_out_dir=diagnostics_out_dir
        )
    if handoff_root is not None:
        assert generated_dir is not None
        handoff_manifest_path = write_generate_handoff_manifest(
            config_path=args.config,
            generate_invocation_overrides=_build_generate_invocation_overrides(
                args,
                handoff_root=handoff_root,
            ),
            run_root=handoff_root,
            generated_dir=generated_dir,
            effective_config_path=effective_config_path,
            effective_config_trace_path=trace_path,
            generated_datasets=int(written),
            generation_elapsed_seconds=float(generation_elapsed_seconds),
            requested_device=str(requested_device),
            resolved_device=str(resolved_device),
            hardware_backend=str(hw.backend),
            hardware_device_name=str(hw.device_name),
            hardware_tier=str(hw.tier),
            hardware_policy=str(args.hardware_policy),
        )
        print(f"Wrote handoff manifest: {handoff_manifest_path}")
    print(f"Wrote {written} datasets to: {Path(resolved_out_dir)}")
    return 0
