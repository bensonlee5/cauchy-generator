"""Generate command handler."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict
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
    HANDOFF_SOURCE_FAMILY_FIXED,
    HANDOFF_SOURCE_FAMILY_HETEROGENEOUS,
    write_generate_handoff_manifest,
)
from dagzoo.diagnostics import (
    CoverageAggregator,
    write_coverage_summary_json,
    write_coverage_summary_markdown,
)
from dagzoo.diagnostics_targets import build_diagnostics_aggregation_config
from dagzoo.filtering.deferred_filter import MANIFEST_FILENAME, SUMMARY_FILENAME
from dagzoo.hardware import detect_hardware
from dagzoo.io.parquet_writer import write_packed_parquet_shards_stream
from dagzoo.io.shard_contract import RUN_CONTEXT_FILENAME

from ..common import load_config_or_usage_error, raise_usage_error
from ..effective_config import (
    print_effective_config as emit_effective_config,
)
from ..effective_config import (
    print_resolution_trace as emit_resolution_trace,
)
from ..effective_config import (
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
    internal_dir = run_root / "internal"
    for path in (generated_dir, filter_dir, curated_dir, internal_dir):
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
    stale_internal = next(internal_dir.glob("shard_*"), None) if internal_dir.exists() else None
    if stale_internal is not None or (internal_dir / RUN_CONTEXT_FILENAME).exists():
        raise RuntimeError(
            "Generate handoff root already contains prior internal sidecars: "
            f"{internal_dir}. Remove existing internal artifacts or choose a new handoff root."
        )

    handoff_manifest_path = run_root / HANDOFF_MANIFEST_FILENAME
    if handoff_manifest_path.exists():
        raise RuntimeError(
            "Generate handoff root already contains a prior handoff manifest: "
            f"{handoff_manifest_path}. Remove the existing manifest or choose a new handoff root."
        )

    return generated_dir


def _build_generate_invocation_overrides(
    *,
    num_datasets: int,
    seed: int | None,
    rows: str | None,
    device: str | None,
    hardware_policy: str,
    diagnostics: bool,
    diagnostics_out_dir: Path | None,
    set_overrides: Sequence[tuple[str, Any]] | None,
    handoff_root: Path,
) -> dict[str, Any]:
    """Serialize user-supplied CLI overrides for the handoff manifest."""

    diagnostics_out_dir_str = (
        str(diagnostics_out_dir.resolve()) if diagnostics_out_dir is not None else None
    )
    overrides: dict[str, Any] = {
        "num_datasets": int(num_datasets),
        "seed": int(seed) if seed is not None else None,
        "rows": rows,
        "device": device,
        "hardware_policy": str(hardware_policy),
        "missing_rate": None,
        "missing_mechanism": None,
        "missing_mar_observed_fraction": None,
        "missing_mar_logit_scale": None,
        "missing_mnar_logit_scale": None,
        "diagnostics": bool(diagnostics),
        "diagnostics_out_dir": diagnostics_out_dir_str,
        "handoff_root": str(handoff_root.resolve()),
    }
    if set_overrides:
        overrides["set_overrides"] = list(set_overrides)
    return overrides


def run_generate_command(
    *,
    config: str,
    out: str | Path | None = None,
    handoff_root: str | Path | None = None,
    num_datasets: int = 10,
    seed: int | None = None,
    rows: str | None = None,
    device: str | None = None,
    hardware_policy: str = "none",
    no_dataset_write: bool = False,
    diagnostics: bool = False,
    diagnostics_out_dir: str | Path | None = None,
    set_overrides: Sequence[tuple[str, Any]] | None = None,
    print_effective_config: bool = False,
    print_resolution_trace: bool = False,
) -> int:
    """Execute the ``generate`` command."""

    diagnostics_out_dir_path = (
        Path(diagnostics_out_dir) if diagnostics_out_dir is not None else None
    )
    handoff_root_path: Path | None = None
    generated_dir: Path | None = None
    if handoff_root is not None:
        if out is not None:
            raise_usage_error("`--handoff-root` cannot be combined with `--out`.")
        if no_dataset_write:
            raise_usage_error("`--handoff-root` cannot be combined with `--no-dataset-write`.")
        handoff_root_path = Path(handoff_root).resolve()
        try:
            generated_dir = _ensure_generate_handoff_output_safe(handoff_root_path)
        except RuntimeError as exc:
            raise_usage_error(str(exc))

    loaded_config = load_config_or_usage_error(config)
    try:
        resolved = resolve_generate_config(
            loaded_config,
            device_override=device,
            rows=rows,
            hardware_policy=str(hardware_policy),
            diagnostics_enabled=bool(diagnostics),
            path_overrides=list(set_overrides or ()),
        )
    except ValueError as exc:
        raise_usage_error(str(exc))

    resolved_config = resolved["config"]
    effective_seed = seed if seed is not None else resolved_config.seed
    prefer_cpu_for_mps_auto = str(resolved_config.runtime.layout_mode) != "fixed"
    run_config, run_seed, requested_device, resolved_device, _carried_stress_profile = (
        realize_generation_config_for_run(
            resolved_config,
            seed=effective_seed,
            device=str(resolved["requested_device"]),
            prefer_cpu_for_mps_auto=prefer_cpu_for_mps_auto,
            carried_stress_profile=resolved.get("carried_stress_profile"),
        )
    )
    if bool(run_config.filter.enabled):
        raise_usage_error(
            "Inline filtering has been removed from generate. Set filter.enabled=false and run "
            "`dagzoo filter --in <shard_dir> --out <out_dir>` after generation. "
            "Generation still uses filter.min_target_* and filter.max_attempts while "
            "resampling structurally valid layouts."
        )
    hw = detect_hardware(resolved_device)
    trace_events = list(resolved["trace_events"])
    append_config_diff_events(
        resolved_config,
        run_config,
        source="generate.run_realization",
        events=trace_events,
    )
    out_dir: str | Path | None = out or run_config.output.out_dir
    effective_config_root: str | Path | None = out_dir
    internal_root: Path | None = None
    if handoff_root_path is not None:
        assert generated_dir is not None
        pre_handoff_config = clone_generator_config(run_config, revalidate=False)
        run_config.output.out_dir = str(generated_dir)
        append_config_diff_events(
            pre_handoff_config,
            run_config,
            source="generate.handoff_root",
            events=trace_events,
        )
        out_dir = generated_dir
        internal_root = handoff_root_path / "internal"
        effective_config_root = internal_root
    elif effective_config_root is None:
        effective_config_root = (
            diagnostics_out_dir_path
            or run_config.diagnostics.out_dir
            or "effective_config_artifacts"
        )

    trace_payload = serialize_resolution_events(trace_events)
    if print_effective_config:
        emit_effective_config(run_config, header="Effective config:")

    effective_config_path = write_effective_config(
        run_config,
        Path(effective_config_root) / "effective_config.yaml",
    )
    trace_path = write_effective_config_trace(
        trace_payload,
        Path(effective_config_root) / "effective_config_trace.yaml",
    )
    print(f"Wrote effective config: {effective_config_path}")
    print(f"Wrote effective config trace: {trace_path}")
    if print_resolution_trace:
        emit_resolution_trace(trace_payload, header="Resolution trace:")

    diagnostics_enabled = bool(run_config.diagnostics.enabled)
    diagnostics_output_dir: Path | None = None
    diagnostics_aggregator: CoverageAggregator | None = None
    if diagnostics_enabled:
        diagnostics_root = diagnostics_out_dir_path or run_config.diagnostics.out_dir
        if diagnostics_root is None:
            diagnostics_root = (
                (internal_root / "diagnostics_artifacts") if internal_root is not None else out_dir
            )
        if diagnostics_root is None:
            diagnostics_root = "diagnostics_artifacts"
        diagnostics_output_dir = Path(diagnostics_root)
        diagnostics_aggregator = CoverageAggregator(
            build_diagnostics_aggregation_config(run_config)
        )
    print(
        f"Hardware backend={hw.backend} device='{hw.device_name}' "
        f"memory_gb={hw.total_memory_gb} peak_flops={hw.peak_flops:.3e} tier={hw.tier} "
        f"hardware_policy={hardware_policy}"
    )

    stream: Iterator[Any] = generate_batch_iter(
        run_config,
        num_datasets=num_datasets,
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

    if no_dataset_write:
        generated = sum(1 for _ in stream)
        if diagnostics_aggregator is not None:
            assert diagnostics_output_dir is not None
            _write_generate_diagnostics_artifacts(
                diagnostics_aggregator, diagnostics_out_dir=diagnostics_output_dir
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
        shard_size=run_config.output.shard_size,
        compression=run_config.output.compression,
        internal_root=internal_root,
    )
    generation_elapsed_seconds = perf_counter() - generation_started_at
    if diagnostics_aggregator is not None:
        assert diagnostics_output_dir is not None
        _write_generate_diagnostics_artifacts(
            diagnostics_aggregator, diagnostics_out_dir=diagnostics_output_dir
        )
    if handoff_root_path is not None:
        assert generated_dir is not None
        assert internal_root is not None
        internal_root.mkdir(parents=True, exist_ok=True)
        (internal_root / RUN_CONTEXT_FILENAME).write_text(
            json.dumps(
                {
                    "config_path": str(config),
                    "resolved_config_path": str(effective_config_path.resolve()),
                    "resolved_config_trace_path": str(trace_path.resolve()),
                    "generate_invocation_overrides": _build_generate_invocation_overrides(
                        num_datasets=num_datasets,
                        seed=seed,
                        rows=rows,
                        device=device,
                        hardware_policy=hardware_policy,
                        diagnostics=diagnostics,
                        diagnostics_out_dir=diagnostics_out_dir_path,
                        set_overrides=set_overrides,
                        handoff_root=handoff_root_path,
                    ),
                    "effective_config": asdict(run_config),
                    "resolution_trace": trace_payload,
                    "hardware": {
                        "requested_device": str(requested_device),
                        "resolved_device": str(resolved_device),
                        "backend": str(hw.backend),
                        "device_name": str(hw.device_name),
                        "tier": str(hw.tier),
                        "hardware_policy": str(hardware_policy),
                    },
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        handoff_manifest_path = write_generate_handoff_manifest(
            config_path=config,
            generate_invocation_overrides=_build_generate_invocation_overrides(
                num_datasets=num_datasets,
                seed=seed,
                rows=rows,
                device=device,
                hardware_policy=hardware_policy,
                diagnostics=diagnostics,
                diagnostics_out_dir=diagnostics_out_dir_path,
                set_overrides=set_overrides,
                handoff_root=handoff_root_path,
            ),
            run_root=handoff_root_path,
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
            hardware_policy=str(hardware_policy),
            source_family=(
                HANDOFF_SOURCE_FAMILY_FIXED
                if str(run_config.runtime.layout_mode) == "fixed"
                else HANDOFF_SOURCE_FAMILY_HETEROGENEOUS
            ),
        )
        print(f"Wrote handoff manifest: {handoff_manifest_path}")
    print(f"Wrote {written} datasets to: {Path(resolved_out_dir)}")
    return 0
