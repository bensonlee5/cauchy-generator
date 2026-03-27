from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def _import_dev_module(module_name: str):
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    return importlib.import_module(module_name)


def _load_dev_cli():
    module_name = "repo_dev_cli"
    script_path = SCRIPTS_ROOT / "dev.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_import_graph_hotspot_captures_execution_semantics_cascade() -> None:
    graph_module = _import_dev_module("devlib.import_graph")

    graph = graph_module.build_import_graph()
    summary = graph.module_summary("dagzoo.core.execution_semantics")

    assert "dagzoo.core.fixed_layout.batched" in summary.direct_importers
    assert "dagzoo.core.node_pipeline" in summary.direct_importers
    assert "dagzoo.core.fixed_layout.runtime" in summary.transitive_importers
    assert "dagzoo.bench" in summary.impacted_packages
    assert "dagzoo.cli" in summary.impacted_packages


def test_impact_report_flags_execution_semantics_as_architecture_and_bench() -> None:
    impact_module = _import_dev_module("devlib.impact")

    report = impact_module.build_impact_report(("src/dagzoo/core/execution_semantics.py",))

    assert report.tags == ("architecture", "code")
    assert report.recommended_modes == ("quick", "code", "bench")
    assert report.module_summaries[0].module == "dagzoo.core.execution_semantics"
    assert "dagzoo.bench" in report.module_summaries[0].downstream_packages


def test_impact_report_includes_pytest_selection_payload() -> None:
    impact_module = _import_dev_module("devlib.impact")

    report = impact_module.build_impact_report(("src/dagzoo/cli/entrypoint.py",))
    payload = json.loads(impact_module.render_json(report))

    assert report.pytest_selection.mode == "targeted"
    assert "tests/test_cli_validation.py" in report.pytest_selection.targets
    assert payload["pytest_selection"] == {
        "mode": "targeted",
        "targets": list(report.pytest_selection.targets),
        "reason": report.pytest_selection.reason,
    }


def test_release_contract_requires_version_and_changelog_for_release_risk() -> None:
    impact_module = _import_dev_module("devlib.impact")
    contract_module = _import_dev_module("devlib.contract")

    report = impact_module.build_impact_report(("src/dagzoo/cli/entrypoint.py",))
    result = contract_module.evaluate_release_contract(report)

    assert result.ok is False
    assert "pyproject.toml" in result.errors[0]
    assert "CHANGELOG.md" in result.errors[0]


def test_release_contract_warns_for_internal_code_change() -> None:
    impact_module = _import_dev_module("devlib.impact")
    contract_module = _import_dev_module("devlib.contract")

    report = impact_module.build_impact_report(("src/dagzoo/functions/activations.py",))
    result = contract_module.evaluate_release_contract(report)

    assert result.ok is True
    assert result.warnings
    assert "Internal-only refactors" in result.warnings[0]


def test_dev_cli_prunes_unsupported_wrapper_commands() -> None:
    module = _load_dev_cli()
    parser = module.build_parser()
    subparsers_action = next(
        action for action in parser._actions if getattr(action, "choices", None) is not None
    )

    assert "impact" in subparsers_action.choices
    assert "contract" in subparsers_action.choices
    assert "bootstrap" in subparsers_action.choices
    assert "review-base" in subparsers_action.choices
    assert "verify" not in subparsers_action.choices
    assert "deps" not in subparsers_action.choices
    assert "doctor" not in subparsers_action.choices
    assert "ready" not in subparsers_action.choices


def test_fast_pr_workflow_uses_ci_runner_not_dev_verify() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")

    assert "scripts/ci/run_affected_verification.py" in workflow
    assert "./scripts/dev verify affected" not in workflow


def test_dev_cli_impact_command_renders_report(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    module = _load_dev_cli()
    monkeypatch.setattr(module, "detect_changed_files", lambda **_kwargs: ("README.md",))
    monkeypatch.setattr(
        module,
        "build_impact_report",
        lambda _changed_files: type("Report", (), {"changed_files": ("README.md",)})(),
    )
    monkeypatch.setattr(module, "render_impact_text", lambda _report: "impact report\n")

    exit_code = module.main(["impact"])

    assert exit_code == 0
    assert capsys.readouterr().out == "impact report\n"
