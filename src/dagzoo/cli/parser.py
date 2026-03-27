"""CLI parser construction."""

from __future__ import annotations

import argparse
from pathlib import Path

from dagzoo.rng import SEED32_MAX, SEED32_MIN

from .commands.benchmark import run_benchmark_command
from .commands.diagnostics import run_diversity_audit_command
from .commands.filter import run_filter_command
from .commands.generate import run_generate_command
from .commands.hardware import run_hardware_command
from .commands.recipe import run_recipe_list_command
from .parsing import (
    DEVICE_CHOICES,
    HARDWARE_POLICY_CHOICES,
    non_negative_int,
    parse_fail_threshold_pct_arg,
    parse_set_override,
    parse_warn_threshold_pct_arg,
    positive_int,
    seed_32bit_int,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser and register all subcommands/options."""

    parser = argparse.ArgumentParser(prog="dagzoo")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate synthetic datasets.")
    g.set_defaults(handler=run_generate_command)
    g.add_argument(
        "--config",
        required=True,
        help="YAML config path or curated recipe reference (`recipe:<name>`).",
    )
    g.add_argument("--out", default=None, help="Output directory for parquet shards.")
    g.add_argument(
        "--handoff-root",
        default=None,
        help=(
            "Optional handoff root; writes generated artifacts under "
            "<handoff_root>/generated. Cannot be combined with --out or "
            "--no-dataset-write."
        ),
    )
    g.add_argument(
        "--num-datasets",
        type=positive_int,
        default=10,
        help="Number of datasets to generate.",
    )
    g.add_argument(
        "--seed",
        type=seed_32bit_int,
        default=None,
        help=f"Optional override for run seed in [{SEED32_MIN}, {SEED32_MAX}].",
    )
    g.add_argument(
        "--rows",
        default=None,
        help=(
            "Optional total-row spec override for generation. "
            "Supports fixed int (e.g. 1024) or range (e.g. 400..60000)."
        ),
    )
    g.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Device override (auto/cpu/cuda/mps).",
    )
    g.add_argument(
        "--hardware-policy",
        default="none",
        choices=HARDWARE_POLICY_CHOICES,
        help="Explicit hardware policy to apply to config (default: none).",
    )
    g.add_argument(
        "--no-dataset-write",
        action="store_true",
        help="Generate in memory only and do not write parquet files.",
    )
    g.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable diagnostics coverage aggregation artifacts for this run.",
    )
    g.add_argument(
        "--diagnostics-out-dir",
        default=None,
        help="Optional directory for diagnostics artifacts (defaults to output directory).",
    )
    g.add_argument(
        "--set",
        dest="set_overrides",
        type=parse_set_override,
        action="append",
        default=None,
        help="Repeatable advanced override in dotted.path=value form.",
    )
    g.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print resolved effective config YAML before generation.",
    )
    g.add_argument(
        "--print-resolution-trace",
        action="store_true",
        help="Print field-level override trace for resolved config before generation.",
    )

    f = sub.add_parser(
        "filter",
        help="Replay deferred filtering over generated shard outputs.",
    )
    f.set_defaults(handler=run_filter_command)
    f.add_argument(
        "--in",
        dest="in_dir",
        required=True,
        help="Input directory containing shard_* outputs (or a single shard directory).",
    )
    f.add_argument(
        "--out",
        required=True,
        help="Directory for deferred filter manifest/summary artifacts.",
    )
    f.add_argument(
        "--curated-out",
        default=None,
        help="Optional output directory for accepted-only curated shards.",
    )
    f.add_argument(
        "--set",
        dest="set_overrides",
        type=parse_set_override,
        action="append",
        default=None,
        help="Repeatable advanced override in filter.<field>=value form.",
    )

    b = sub.add_parser("benchmark", help="Run benchmark suite across one or more presets.")
    b.set_defaults(handler=run_benchmark_command)
    b.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path or `recipe:<name>` for preset 'custom'.",
    )
    b.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Device override for preset 'custom' or a single resolved preset.",
    )
    b.add_argument(
        "--num-datasets",
        type=positive_int,
        default=None,
        help="Override benchmark dataset count.",
    )
    b.add_argument(
        "--warmup",
        type=non_negative_int,
        default=None,
        help="Override benchmark warmup count.",
    )
    b.add_argument(
        "--hardware-policy",
        default="none",
        choices=HARDWARE_POLICY_CHOICES,
        help="Explicit hardware policy to apply to configs (default: none).",
    )
    b.add_argument("--json-out", default=None, help="Optional path to write suite summary JSON.")
    b.add_argument(
        "--suite",
        default=None,
        choices=["smoke", "standard", "full"],
        help="Benchmark suite level. Defaults to config benchmark.suite.",
    )
    b.add_argument(
        "--preset",
        action="append",
        default=None,
        choices=["all", "cpu", "cuda_desktop", "cuda_h100", "custom"],
        help="Benchmark preset key. Repeat to run multiple presets.",
    )
    b.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline JSON path for regression checks.",
    )
    b.add_argument("--out-dir", default=None, help="Optional directory for summary artifacts.")
    b.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return non-zero exit code if regression status is fail.",
    )
    b.add_argument(
        "--warn-threshold-pct",
        type=float,
        default=None,
        help="Warning degradation threshold percentage.",
    )
    b.add_argument(
        "--fail-threshold-pct",
        type=float,
        default=None,
        help="Failure degradation threshold percentage.",
    )
    b.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory collection for benchmark presets.",
    )
    b.add_argument(
        "--collect-reproducibility",
        action="store_true",
        help="Force reproducibility checks even outside full suite mode.",
    )
    b.add_argument(
        "--save-baseline",
        default=None,
        help="Optional path to write a baseline JSON derived from this run.",
    )
    b.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable diagnostics coverage aggregation artifacts for each benchmark preset run.",
    )
    b.add_argument(
        "--diagnostics-out-dir",
        default=None,
        help="Optional root directory for benchmark diagnostics artifacts.",
    )
    b.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print each preset's resolved effective config YAML before execution.",
    )
    b.add_argument(
        "--print-resolution-trace",
        action="store_true",
        help="Print each preset's field-level override trace before execution.",
    )

    d = sub.add_parser(
        "diversity-audit",
        help="Compare accepted-corpus diversity for a baseline config and one or more variants.",
    )
    d.set_defaults(handler=run_diversity_audit_command)
    d.add_argument(
        "--baseline-config",
        required=True,
        help="Baseline generator config YAML path or `recipe:<name>`.",
    )
    d.add_argument(
        "--variant-config",
        action="append",
        required=True,
        help="Variant generator config YAML path or `recipe:<name>`. Repeat for multiple variants.",
    )
    d.add_argument(
        "--warn-threshold-pct",
        type=parse_warn_threshold_pct_arg,
        default=2.5,
        help="Warn threshold for composite diversity shift vs baseline.",
    )
    d.add_argument(
        "--fail-threshold-pct",
        type=parse_fail_threshold_pct_arg,
        default=5.0,
        help="Fail threshold for composite diversity shift vs baseline.",
    )
    d.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return non-zero exit code if any variant reaches fail severity.",
    )
    d.add_argument(
        "--suite",
        choices=["smoke", "standard"],
        default="standard",
        help="Probe size to use for the baseline and each variant.",
    )
    d.add_argument(
        "--num-datasets",
        type=positive_int,
        default=None,
        help="Optional override for per-config dataset count.",
    )
    d.add_argument(
        "--warmup",
        type=non_negative_int,
        default=None,
        help="Optional override for per-config warmup count.",
    )
    d.add_argument(
        "--out-dir",
        default=str(Path("effective_config_artifacts") / "diversity_audit"),
        help="Output directory for diversity audit artifacts.",
    )
    d.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Optional device override for scale-phase generation.",
    )

    h = sub.add_parser("hardware", help="Inspect detected hardware and tier mapping.")
    h.set_defaults(handler=run_hardware_command)
    h.add_argument(
        "--device",
        default=None,
        choices=DEVICE_CHOICES,
        help="Requested device (auto/cpu/cuda/mps).",
    )

    r = sub.add_parser("recipe", help="Inspect the curated public recipe catalog.")
    r_sub = r.add_subparsers(dest="recipe_command", required=True)
    r_list = r_sub.add_parser("list", help="List curated recipe references.")
    r_list.set_defaults(handler=run_recipe_list_command)
    return parser
