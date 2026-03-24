"""Filter command handler."""

from __future__ import annotations

import argparse

from ..common import get_cli_public_api, raise_usage_error


def run_filter_command(args: argparse.Namespace) -> int:
    """Execute the ``filter`` command."""

    cli_api = get_cli_public_api()
    try:
        result = cli_api.run_deferred_filter(
            in_dir=args.in_dir,
            out_dir=args.out,
            curated_out_dir=args.curated_out,
            ease_k_small_override=args.ease_k_small,
            easy_skill_threshold_override=args.easy_skill_threshold,
            easy_gain_threshold_override=args.easy_gain_threshold,
            hard_skill_threshold_override=args.hard_skill_threshold,
            stump_skill_threshold_override=args.stump_skill_threshold,
            use_lineage_veto_override=(False if bool(args.no_lineage_veto) else None),
            n_jobs_override=args.n_jobs,
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
