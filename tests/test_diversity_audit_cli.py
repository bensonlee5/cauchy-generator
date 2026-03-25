import copy
import json

import yaml

from dagzoo.cli.entrypoint import main
from dagzoo.config import GeneratorConfig


def test_diversity_audit_cli_writes_summary_artifacts(tmp_path) -> None:
    baseline = GeneratorConfig.from_yaml("configs/default.yaml")
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
