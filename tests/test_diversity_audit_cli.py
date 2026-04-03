import copy
import json

import yaml

from dagzoo.bench.corpus_probe import CorpusProbeResult
from dagzoo.cli.entrypoint import main
from dagzoo.config import GeneratorConfig


def test_diversity_audit_cli_writes_summary_artifacts(tmp_path) -> None:
    baseline = GeneratorConfig.from_yaml("configs/default.yaml")
    baseline.dataset.task = "regression"
    baseline.runtime.device = "cpu"
    baseline.filter.enabled = False
    baseline.dataset.n_train = 24
    baseline.dataset.n_test = 12
    baseline.dataset.n_features_min = 8
    baseline.dataset.n_features_max = 8

    variant = copy.deepcopy(baseline)
    variant.graph.n_nodes_min = 6
    variant.graph.n_nodes_max = 7

    baseline_path = tmp_path / "baseline.yaml"
    variant_path = tmp_path / "variant.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline.to_dict()), encoding="utf-8")
    variant_path.write_text(yaml.safe_dump(variant.to_dict()), encoding="utf-8")

    out_dir = tmp_path / "diversity_audit"
    code = main(
        [
            "diversity-audit",
            "--baseline-config",
            str(baseline_path),
            "--variant-config",
            str(variant_path),
            "--suite",
            "smoke",
            "--num-datasets",
            "2",
            "--warmup",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    summary_path = out_dir / "summary.json"
    markdown_path = out_dir / "summary.md"
    assert summary_path.exists()
    assert markdown_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["schema_name"] == "dagzoo_diversity_audit_report"
    assert payload["baseline"]["config_path"] == str(baseline_path)
    assert payload["variants"][0]["config_path"] == str(variant_path)


def test_diversity_audit_cli_uses_shared_smoke_probe_counts_for_mismatched_configs(
    tmp_path,
) -> None:
    out_dir = tmp_path / "diversity_audit_default_vs_shift"
    code = main(
        [
            "diversity-audit",
            "--baseline-config",
            "configs/default.yaml",
            "--variant-config",
            "configs/preset_shift_benchmark_smoke.yaml",
            "--suite",
            "smoke",
            "--device",
            "cpu",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["summary"]["probe_num_datasets"] == 25
    assert payload["summary"]["probe_warmup_datasets"] == 5
    assert payload["baseline"]["num_datasets"] == 25
    assert payload["variants"][0]["num_datasets"] == 25
    assert payload["baseline"]["warmup_datasets"] == 5
    assert payload["variants"][0]["warmup_datasets"] == 5


def test_diversity_audit_cli_writes_summary_artifacts_for_stress_profile_preset(tmp_path) -> None:
    probe_results = [
        CorpusProbeResult(
            label="baseline",
            config_path="configs/default.yaml",
            suite="smoke",
            num_datasets=2,
            warmup_datasets=0,
            requested_device="cpu",
            resolved_device="cpu",
            resolved_config={"runtime": {"device": "cpu"}},
            datasets_per_minute=120.0,
            filter_datasets_per_minute=None,
            filter_accepted_datasets_per_minute=None,
            filter_accepted_datasets_measured=0,
            filter_rejected_datasets_measured=0,
            filter_acceptance_rate_dataset_level=None,
            filter_rejection_rate_dataset_level=None,
            coverage_summary={
                "num_datasets": 2,
                "mechanism_family_summary": {
                    "metadata_coverage_rate": 1.0,
                    "bundles_with_metadata": 2,
                    "sampled_family_counts": {"linear": 2},
                    "dataset_presence_rate_by_family": {"linear": 1.0},
                    "sampled_variant_counts": {},
                    "dataset_presence_rate_by_variant": {},
                    "mean_total_function_plans": 2.0,
                },
                "metrics": {},
            },
            filter_summary=None,
        ),
        CorpusProbeResult(
            label="variant_1",
            config_path="configs/preset_stress_graph_breadth_benchmark_smoke.yaml",
            suite="smoke",
            num_datasets=2,
            warmup_datasets=0,
            requested_device="cpu",
            resolved_device="cpu",
            resolved_config={"runtime": {"device": "cpu"}},
            datasets_per_minute=110.0,
            filter_datasets_per_minute=None,
            filter_accepted_datasets_per_minute=None,
            filter_accepted_datasets_measured=0,
            filter_rejected_datasets_measured=0,
            filter_acceptance_rate_dataset_level=None,
            filter_rejection_rate_dataset_level=None,
            coverage_summary={
                "num_datasets": 2,
                "mechanism_family_summary": {
                    "metadata_coverage_rate": 1.0,
                    "bundles_with_metadata": 2,
                    "sampled_family_counts": {"linear": 2},
                    "dataset_presence_rate_by_family": {"linear": 1.0},
                    "sampled_variant_counts": {},
                    "dataset_presence_rate_by_variant": {},
                    "mean_total_function_plans": 2.0,
                },
                "metrics": {},
            },
            filter_summary=None,
        ),
    ]

    def _stub_run_corpus_probe(*_args, **_kwargs):
        return probe_results.pop(0)

    import dagzoo.diagnostics.effective_diversity.runner as runner

    original_run_corpus_probe = runner.run_corpus_probe
    runner.run_corpus_probe = _stub_run_corpus_probe
    try:
        out_dir = tmp_path / "diversity_audit_default_vs_stress_graph_breadth"
        code = main(
            [
                "diversity-audit",
                "--baseline-config",
                "configs/default.yaml",
                "--variant-config",
                "configs/preset_stress_graph_breadth_benchmark_smoke.yaml",
                "--suite",
                "smoke",
                "--num-datasets",
                "2",
                "--warmup",
                "0",
                "--device",
                "cpu",
                "--out-dir",
                str(out_dir),
            ]
        )
    finally:
        runner.run_corpus_probe = original_run_corpus_probe

    assert code == 0
    summary_path = out_dir / "summary.json"
    markdown_path = out_dir / "summary.md"
    assert summary_path.exists()
    assert markdown_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["baseline"]["config_path"] == "configs/default.yaml"
    assert (
        payload["variants"][0]["config_path"]
        == "configs/preset_stress_graph_breadth_benchmark_smoke.yaml"
    )
