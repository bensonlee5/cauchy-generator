"""Click CLI construction."""

from __future__ import annotations

from pathlib import Path

import click

from dagzoo.rng import SEED32_MAX, SEED32_MIN

from .commands.benchmark import run_benchmark_command
from .commands.diagnostics import run_diversity_audit_command
from .commands.filter import run_filter_command
from .commands.generate import run_generate_command
from .commands.hardware import run_hardware_command
from .commands.publish import run_publish_hub_command
from .commands.recipe import run_recipe_list_command
from .parsing import (
    DEVICE_CHOICES,
    NON_NEGATIVE_INT,
    POSITIVE_INT,
    SEED32_INT,
    device_choice,
    fail_threshold_pct_callback,
    hardware_policy_choice,
    set_overrides_callback,
    warn_threshold_pct_callback,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
BENCHMARK_PRESET_CHOICES = ("all", "cpu", "cuda_desktop", "cuda_h100", "custom")
BENCHMARK_SUITE_CHOICES = ("smoke", "standard", "full")
DIVERSITY_AUDIT_SUITE_CHOICES = ("smoke", "standard")


@click.group(context_settings=CONTEXT_SETTINGS)
def cli() -> None:
    """dagzoo command-line interface."""


@click.command("generate", context_settings=CONTEXT_SETTINGS, help="Generate synthetic datasets.")
@click.option(
    "--config",
    required=True,
    help="YAML config path or curated recipe reference (`recipe:<name>`).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for parquet shards.",
)
@click.option(
    "--handoff-root",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Optional handoff root; writes generated artifacts under "
        "<handoff_root>/generated. Cannot be combined with --out or "
        "--no-dataset-write."
    ),
)
@click.option(
    "--num-datasets",
    type=POSITIVE_INT,
    default=10,
    show_default=True,
    help="Number of datasets to generate.",
)
@click.option(
    "--seed",
    type=SEED32_INT,
    default=None,
    help=f"Optional override for run seed in [{SEED32_MIN}, {SEED32_MAX}].",
)
@click.option(
    "--rows",
    default=None,
    help=(
        "Optional total-row spec override for generation. "
        "Supports fixed int (e.g. 1024) or range (e.g. 400..60000)."
    ),
)
@click.option(
    "--device",
    default=None,
    type=device_choice(),
    help=f"Device override ({'/'.join(DEVICE_CHOICES)}).",
)
@click.option(
    "--hardware-policy",
    default="none",
    type=hardware_policy_choice(),
    help="Explicit hardware policy to apply to config (default: none).",
)
@click.option(
    "--no-dataset-write",
    is_flag=True,
    help="Generate in memory only and do not write parquet files.",
)
@click.option(
    "--diagnostics",
    is_flag=True,
    help="Enable diagnostics coverage aggregation artifacts for this run.",
)
@click.option(
    "--diagnostics-out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional directory for diagnostics artifacts (defaults to output directory).",
)
@click.option(
    "--set",
    "set_overrides",
    type=str,
    multiple=True,
    callback=set_overrides_callback,
    help="Repeatable advanced override in dotted.path=value form.",
)
@click.option(
    "--print-effective-config",
    is_flag=True,
    help="Print resolved effective config YAML before generation.",
)
@click.option(
    "--print-resolution-trace",
    is_flag=True,
    help="Print field-level override trace for resolved config before generation.",
)
def generate_command(
    *,
    config: str,
    out: Path | None,
    handoff_root: Path | None,
    num_datasets: int,
    seed: int | None,
    rows: str | None,
    device: str | None,
    hardware_policy: str,
    no_dataset_write: bool,
    diagnostics: bool,
    diagnostics_out_dir: Path | None,
    set_overrides: tuple[tuple[str, object], ...],
    print_effective_config: bool,
    print_resolution_trace: bool,
) -> int:
    """Execute the generate command."""

    return run_generate_command(
        config=config,
        out=out,
        handoff_root=handoff_root,
        num_datasets=num_datasets,
        seed=seed,
        rows=rows,
        device=device,
        hardware_policy=hardware_policy,
        no_dataset_write=no_dataset_write,
        diagnostics=diagnostics,
        diagnostics_out_dir=diagnostics_out_dir,
        set_overrides=set_overrides,
        print_effective_config=print_effective_config,
        print_resolution_trace=print_resolution_trace,
    )


@click.command(
    "filter",
    context_settings=CONTEXT_SETTINGS,
    help="Replay deferred filtering over generated shard outputs.",
)
@click.option(
    "--in",
    "in_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Input directory containing shard_* outputs (or a single shard directory).",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for deferred filter manifest/summary artifacts.",
)
@click.option(
    "--curated-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional output directory for accepted-only curated shards.",
)
@click.option(
    "--set",
    "set_overrides",
    type=str,
    multiple=True,
    callback=set_overrides_callback,
    help="Repeatable advanced override in filter.<field>=value form.",
)
def filter_command(
    *,
    in_dir: Path,
    out: Path,
    curated_out: Path | None,
    set_overrides: tuple[tuple[str, object], ...],
) -> int:
    """Execute the filter command."""

    return run_filter_command(
        in_dir=str(in_dir),
        out=str(out),
        curated_out=str(curated_out) if curated_out is not None else None,
        set_overrides=set_overrides,
    )


@click.command(
    "benchmark",
    context_settings=CONTEXT_SETTINGS,
    help="Run benchmark suite across one or more presets.",
)
@click.option(
    "--config",
    default=None,
    help="Optional YAML config path or `recipe:<name>` for preset 'custom'.",
)
@click.option(
    "--device",
    default=None,
    type=device_choice(),
    help=f"Device override for preset 'custom' or a single resolved preset ({'/'.join(DEVICE_CHOICES)}).",
)
@click.option(
    "--num-datasets",
    type=POSITIVE_INT,
    default=None,
    help="Override benchmark dataset count.",
)
@click.option(
    "--warmup",
    type=NON_NEGATIVE_INT,
    default=None,
    help="Override benchmark warmup count.",
)
@click.option(
    "--hardware-policy",
    default="none",
    type=hardware_policy_choice(),
    help="Explicit hardware policy to apply to configs (default: none).",
)
@click.option(
    "--json-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to write suite summary JSON.",
)
@click.option(
    "--suite",
    default=None,
    type=click.Choice(BENCHMARK_SUITE_CHOICES),
    help="Benchmark suite level. Defaults to config benchmark.suite.",
)
@click.option(
    "--preset",
    multiple=True,
    default=(),
    type=click.Choice(BENCHMARK_PRESET_CHOICES),
    help="Benchmark preset key. Repeat to run multiple presets.",
)
@click.option(
    "--baseline",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional baseline JSON path for regression checks.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional directory for summary artifacts.",
)
@click.option(
    "--fail-on-regression",
    is_flag=True,
    help="Return non-zero exit code if regression status is fail.",
)
@click.option(
    "--warn-threshold-pct",
    type=float,
    default=None,
    callback=warn_threshold_pct_callback,
    help="Warning degradation threshold percentage.",
)
@click.option(
    "--fail-threshold-pct",
    type=float,
    default=None,
    callback=fail_threshold_pct_callback,
    help="Failure degradation threshold percentage.",
)
@click.option(
    "--no-memory",
    is_flag=True,
    help="Disable memory collection for benchmark presets.",
)
@click.option(
    "--collect-reproducibility",
    is_flag=True,
    help="Force reproducibility checks even outside full suite mode.",
)
@click.option(
    "--save-baseline",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to write a baseline JSON derived from this run.",
)
@click.option(
    "--diagnostics",
    is_flag=True,
    help="Enable diagnostics coverage aggregation artifacts for each benchmark preset run.",
)
@click.option(
    "--diagnostics-out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional root directory for benchmark diagnostics artifacts.",
)
@click.option(
    "--print-effective-config",
    is_flag=True,
    help="Print each preset's resolved effective config YAML before execution.",
)
@click.option(
    "--print-resolution-trace",
    is_flag=True,
    help="Print each preset's field-level override trace before execution.",
)
def benchmark_command(
    *,
    config: str | None,
    device: str | None,
    num_datasets: int | None,
    warmup: int | None,
    hardware_policy: str,
    json_out: Path | None,
    suite: str | None,
    preset: tuple[str, ...],
    baseline: Path | None,
    out_dir: Path | None,
    fail_on_regression: bool,
    warn_threshold_pct: float | None,
    fail_threshold_pct: float | None,
    no_memory: bool,
    collect_reproducibility: bool,
    save_baseline: Path | None,
    diagnostics: bool,
    diagnostics_out_dir: Path | None,
    print_effective_config: bool,
    print_resolution_trace: bool,
) -> int:
    """Execute the benchmark command."""

    return run_benchmark_command(
        config=config,
        device=device,
        num_datasets=num_datasets,
        warmup=warmup,
        hardware_policy=hardware_policy,
        json_out=json_out,
        suite=suite,
        preset=preset,
        baseline=baseline,
        out_dir=out_dir,
        fail_on_regression=fail_on_regression,
        warn_threshold_pct=warn_threshold_pct,
        fail_threshold_pct=fail_threshold_pct,
        no_memory=no_memory,
        collect_reproducibility=collect_reproducibility,
        save_baseline=save_baseline,
        diagnostics=diagnostics,
        diagnostics_out_dir=diagnostics_out_dir,
        print_effective_config=print_effective_config,
        print_resolution_trace=print_resolution_trace,
    )


@click.command(
    "diversity-audit",
    context_settings=CONTEXT_SETTINGS,
    help="Compare accepted-corpus diversity for a baseline config and one or more variants.",
)
@click.option(
    "--baseline-config",
    required=True,
    help="Baseline generator config YAML path or `recipe:<name>`.",
)
@click.option(
    "--variant-config",
    multiple=True,
    required=True,
    help="Variant generator config YAML path or `recipe:<name>`. Repeat for multiple variants.",
)
@click.option(
    "--warn-threshold-pct",
    type=float,
    default=2.5,
    show_default=True,
    callback=warn_threshold_pct_callback,
    help="Warn threshold for composite diversity shift vs baseline.",
)
@click.option(
    "--fail-threshold-pct",
    type=float,
    default=5.0,
    show_default=True,
    callback=fail_threshold_pct_callback,
    help="Fail threshold for composite diversity shift vs baseline.",
)
@click.option(
    "--fail-on-regression",
    is_flag=True,
    help="Return non-zero exit code if any variant reaches fail severity.",
)
@click.option(
    "--suite",
    type=click.Choice(DIVERSITY_AUDIT_SUITE_CHOICES),
    default="standard",
    show_default=True,
    help="Probe size to use for the baseline and each variant.",
)
@click.option(
    "--num-datasets",
    type=POSITIVE_INT,
    default=None,
    help="Optional override for per-config dataset count.",
)
@click.option(
    "--warmup",
    type=NON_NEGATIVE_INT,
    default=None,
    help="Optional override for per-config warmup count.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("effective_config_artifacts") / "diversity_audit",
    show_default=True,
    help="Output directory for diversity audit artifacts.",
)
@click.option(
    "--device",
    default=None,
    type=device_choice(),
    help="Optional device override for scale-phase generation.",
)
def diversity_audit_command(
    *,
    baseline_config: str,
    variant_config: tuple[str, ...],
    warn_threshold_pct: float,
    fail_threshold_pct: float,
    fail_on_regression: bool,
    suite: str,
    num_datasets: int | None,
    warmup: int | None,
    out_dir: Path,
    device: str | None,
) -> int:
    """Execute the diversity-audit command."""

    return run_diversity_audit_command(
        baseline_config=baseline_config,
        variant_config=variant_config,
        warn_threshold_pct=warn_threshold_pct,
        fail_threshold_pct=fail_threshold_pct,
        fail_on_regression=fail_on_regression,
        suite=suite,
        num_datasets=num_datasets,
        warmup=warmup,
        out_dir=out_dir,
        device=device,
    )


@click.command(
    "hardware",
    context_settings=CONTEXT_SETTINGS,
    help="Inspect detected hardware and tier mapping.",
)
@click.option(
    "--device",
    default=None,
    type=device_choice(),
    help=f"Requested device ({'/'.join(DEVICE_CHOICES)}).",
)
def hardware_command(*, device: str | None) -> int:
    """Execute the hardware command."""

    return run_hardware_command(device=device)


@click.group(
    "publish",
    context_settings=CONTEXT_SETTINGS,
    help="Publish generated corpora to supported dataset repositories.",
)
def publish_group() -> None:
    """Publish subcommands."""


@click.command(
    "hub",
    context_settings=CONTEXT_SETTINGS,
    help="Publish a handoff-root corpus to a Hugging Face Hub dataset repo.",
)
@click.option(
    "--handoff-root",
    required=True,
    type=click.Path(path_type=Path),
    help="Handoff root produced by `dagzoo generate --handoff-root ...`.",
)
@click.option(
    "--repo-id",
    required=True,
    help="Target Hugging Face dataset repo id in `namespace/name` form.",
)
@click.option(
    "--private",
    is_flag=True,
    help="Create the dataset repo as private if it does not already exist.",
)
@click.option(
    "--license",
    "license_id",
    default=None,
    help="Optional Hugging Face dataset-card license identifier.",
)
def publish_hub_command(
    *,
    handoff_root: Path,
    repo_id: str,
    private: bool,
    license_id: str | None,
) -> int:
    """Execute the publish hub command."""

    return run_publish_hub_command(
        handoff_root=str(handoff_root),
        repo_id=repo_id,
        private=private,
        license_id=license_id,
    )


@click.group(
    "recipe",
    context_settings=CONTEXT_SETTINGS,
    help="Inspect the curated public recipe catalog.",
)
def recipe_group() -> None:
    """Recipe subcommands."""


@click.command(
    "list",
    context_settings=CONTEXT_SETTINGS,
    help="List curated recipe references.",
)
def recipe_list_command() -> int:
    """Execute the recipe list command."""

    return run_recipe_list_command()


cli.add_command(generate_command)
cli.add_command(filter_command)
cli.add_command(benchmark_command)
cli.add_command(diversity_audit_command)
cli.add_command(hardware_command)
publish_group.add_command(publish_hub_command)
cli.add_command(publish_group)
recipe_group.add_command(recipe_list_command)
cli.add_command(recipe_group)


def build_cli() -> click.Group:
    """Return the root Click command for the public CLI."""

    return cli
