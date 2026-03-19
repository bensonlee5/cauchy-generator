from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script_module


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_and_render_benchmark_pr_report(tmp_path) -> None:
    module = load_script_module("benchmark_pr_report", "scripts/ci/benchmark_pr_report.py")
    summary_path = tmp_path / "summary.json"
    baseline_path = tmp_path / "baseline.json"

    summary_payload = {
        "suite": "smoke",
        "preset_results": [
            {
                "preset_key": "cpu",
                "datasets_per_minute": 85.0,
                "generation_datasets_per_minute": 100.0,
                "write_datasets_per_minute": 100.0,
                "filter_datasets_per_minute": None,
                "filter_accepted_datasets_per_minute": None,
            }
        ],
    }
    baseline_payload = {
        "version": 1,
        "suite": "smoke",
        "metrics": [
            "datasets_per_minute",
            "generation_datasets_per_minute",
            "write_datasets_per_minute",
        ],
        "presets": {
            "cpu": {
                "datasets_per_minute": 100.0,
                "generation_datasets_per_minute": 100.0,
                "write_datasets_per_minute": 100.0,
            }
        },
    }
    _write_json(summary_path, summary_payload)
    _write_json(baseline_path, baseline_payload)

    report = module.build_benchmark_pr_report(
        summary_payload,
        baseline_payload,
        title="Benchmark Smoke (CPU)",
        summary_path=summary_path,
        baseline_path=baseline_path,
        warn_threshold_pct=10.0,
        fail_threshold_pct=20.0,
    )
    comment = module.render_comment_markdown(report)
    step_summary = module.render_step_summary_markdown(
        {
            **report,
            "comment_path": "benchmarks/results/ci_smoke/benchmark_pr_comment.md",
            "delta_path": "benchmarks/results/ci_smoke/benchmark_delta.json",
        }
    )
    comment_path, step_summary_path, delta_path = module.write_report_artifacts(
        {
            **report,
            "comment_path": "benchmarks/results/ci_smoke/benchmark_pr_comment.md",
            "delta_path": "benchmarks/results/ci_smoke/benchmark_delta.json",
        },
        comment_out=tmp_path / "benchmark_pr_comment.md",
        step_summary_out=tmp_path / "benchmark_pr_step_summary.md",
        delta_out=tmp_path / "benchmark_delta.json",
    )

    assert report["status"] == "warn"
    assert report["issue_count"] == 1
    assert "| Metric | Current | Baseline | Delta % | Severity |" in comment
    assert "datasets_per_minute" in comment
    assert "warn" in comment
    assert "## Artifacts" in step_summary
    assert comment_path.exists()
    assert step_summary_path.exists()
    assert delta_path.exists()
    delta_payload = json.loads(delta_path.read_text(encoding="utf-8"))
    assert delta_payload["status"] == "warn"
    assert delta_payload["preset_reports"][0]["rows"][0]["severity"] == "warn"


def test_main_handles_missing_summary_file(tmp_path) -> None:
    module = load_script_module("benchmark_pr_report_missing", "scripts/ci/benchmark_pr_report.py")
    baseline_path = tmp_path / "baseline.json"
    _write_json(
        baseline_path,
        {
            "version": 1,
            "suite": "smoke",
            "metrics": ["datasets_per_minute"],
            "presets": {"cpu": {"datasets_per_minute": 100.0}},
        },
    )

    exit_code = module.main(
        [
            "--summary-json",
            str(tmp_path / "missing_summary.json"),
            "--baseline-json",
            str(baseline_path),
            "--title",
            "Benchmark Smoke (CPU)",
            "--comment-out",
            str(tmp_path / "comment.md"),
            "--step-summary-out",
            str(tmp_path / "step_summary.md"),
            "--delta-out",
            str(tmp_path / "delta.json"),
        ]
    )

    assert exit_code == 0
    assert "Benchmark summary file was not generated." in (tmp_path / "comment.md").read_text(
        encoding="utf-8"
    )
    assert '"missing-summary"' in (tmp_path / "delta.json").read_text(encoding="utf-8")
