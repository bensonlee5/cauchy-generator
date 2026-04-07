"""Publish command handlers."""

from __future__ import annotations

from dagzoo.publish import publish_handoff_to_hub

from ..common import raise_usage_error


def run_publish_hub_command(
    *,
    handoff_root: str,
    repo_id: str,
    private: bool = False,
    license_id: str | None = None,
) -> int:
    """Publish one handoff root to a Hugging Face Hub dataset repository."""

    try:
        result = publish_handoff_to_hub(
            handoff_root=handoff_root,
            repo_id=repo_id,
            private=private,
            license_id=license_id,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise_usage_error(str(exc))

    print(f"Published dataset repo: {result.repo_url}")
    print(f"Published generated datasets: {result.generated_datasets}")
    if result.curated_datasets is not None:
        print(f"Published curated accepted datasets: {result.curated_datasets}")
    return 0
