"""Filter command handler."""

from __future__ import annotations

import argparse

from dagzoo.filtering import run_deferred_filter

from ..common import raise_usage_error


def run_filter_command(args: argparse.Namespace) -> int:
    """Execute the ``filter`` command."""

    try:
        result = run_deferred_filter(
            in_dir=args.in_dir,
            out_dir=args.out,
            curated_out_dir=args.curated_out,
            path_overrides=tuple(args.set_overrides or ()),
        )
    except NotImplementedError as exc:
        raise_usage_error(str(exc))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise_usage_error(str(exc))
    print(f"Wrote filter manifest: {result.manifest_path}")
    print(f"Wrote filter summary: {result.summary_path}")
    print(
        "Deferred filter summary: "
        f"total={result.total_datasets} accepted={result.accepted_datasets} "
        f"rejected={result.rejected_datasets} dpm={result.datasets_per_minute:.2f}"
    )
    if result.curated_out_dir is not None:
        print(
            f"Wrote curated accepted-only shards: {result.curated_out_dir} "
            f"(datasets={result.curated_accepted_datasets})"
        )
    return 0
