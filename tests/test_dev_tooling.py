from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_dependency_graph_hotspot_captures_execution_semantics_cascade() -> None:
    deps_module = _import_dev_module("devlib.deps")

    graph = deps_module.build_import_graph()
    summary = graph.module_summary("dagzoo.core.execution_semantics")

    assert "dagzoo.core.fixed_layout.batched" in summary.direct_importers
    assert "dagzoo.core.node_pipeline" in summary.direct_importers
    assert "dagzoo.functions.random_functions" in summary.direct_importers
    assert "dagzoo.functions.multi" in summary.direct_importers
    assert "dagzoo.converters.numeric" in summary.direct_importers
    assert "dagzoo.converters.categorical" in summary.direct_importers
    assert "dagzoo.sampling.random_points" in summary.direct_importers
    assert "dagzoo.core.fixed_layout.runtime" in summary.transitive_importers
    assert "dagzoo.bench" in summary.impacted_packages
    assert "dagzoo.cli" in summary.impacted_packages


def test_render_dependency_map_includes_hotspot_example() -> None:
    deps_module = _import_dev_module("devlib.deps")

    content = deps_module.render_dependency_map_markdown(deps_module.build_import_graph())

    assert "## Change-Impact Hotspots" in content
    assert "### `dagzoo.core.execution_semantics`" in content
    assert "dagzoo.core.fixed_layout.batched" in content
    assert "dagzoo.core.fixed_layout.runtime" in content


def test_write_dependency_docs_and_check_current(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    deps_module = _import_dev_module("devlib.deps")
    target = tmp_path / "module-dependency-map.md"
    monkeypatch.setattr(deps_module, "DOCS_DEP_MAP_PATH", target)

    graph = deps_module.build_import_graph()
    deps_module.write_dependency_docs(graph)

    assert target.exists()
    assert deps_module.dependency_docs_are_current(graph) is True

    target.write_text(target.read_text() + "\nextra\n")

    assert deps_module.dependency_docs_are_current(graph) is False


def test_impact_report_flags_execution_semantics_as_architecture_and_bench() -> None:
    impact_module = _import_dev_module("devlib.impact")

    report = impact_module.build_impact_report(("src/dagzoo/core/execution_semantics.py",))

    assert report.tags == ("architecture", "code")
    assert report.recommended_modes == ("quick", "code", "bench")
    assert report.module_summaries[0].module == "dagzoo.core.execution_semantics"
    assert "dagzoo.bench" in report.module_summaries[0].downstream_packages


def test_detect_changed_files_staged_uses_cached_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impact_module = _import_dev_module("devlib.impact")
    calls: list[tuple[str, ...]] = []

    def _stub_run_git_lines(*args: str) -> tuple[str, ...]:
        calls.append(args)
        return ("src/dagzoo/cli/entrypoint.py", "CHANGELOG.md")

    monkeypatch.setattr(impact_module, "run_git_lines", _stub_run_git_lines)

    changed_files = impact_module.detect_changed_files(source="staged")

    assert changed_files == ("CHANGELOG.md", "src/dagzoo/cli/entrypoint.py")
    assert calls == [("diff", "--cached", "--name-only")]


def test_release_contract_requires_version_and_changelog_for_release_risk() -> None:
    impact_module = _import_dev_module("devlib.impact")
    contract_module = _import_dev_module("devlib.contract")

    report = impact_module.build_impact_report(("src/dagzoo/cli/entrypoint.py",))
    result = contract_module.evaluate_release_contract(report)

    assert result.ok is False
    assert "pyproject.toml" in result.errors[0]
    assert "CHANGELOG.md" in result.errors[0]


def test_release_contract_passes_with_release_risk_and_version_updates() -> None:
    impact_module = _import_dev_module("devlib.impact")
    contract_module = _import_dev_module("devlib.contract")

    report = impact_module.build_impact_report(
        ("src/dagzoo/cli/entrypoint.py", "pyproject.toml", "CHANGELOG.md")
    )
    result = contract_module.evaluate_release_contract(report)

    assert result.ok is True
    assert result.errors == ()
    assert result.warnings == ()


def test_release_contract_warns_for_internal_code_change() -> None:
    impact_module = _import_dev_module("devlib.impact")
    contract_module = _import_dev_module("devlib.contract")

    report = impact_module.build_impact_report(("src/dagzoo/functions/activations.py",))
    result = contract_module.evaluate_release_contract(report)

    assert result.ok is True
    assert result.warnings
    assert "Internal-only refactors" in result.warnings[0]


def test_pyproject_change_alone_does_not_trigger_release_risk_contract() -> None:
    impact_module = _import_dev_module("devlib.impact")
    contract_module = _import_dev_module("devlib.contract")

    report = impact_module.build_impact_report(("pyproject.toml",))
    result = contract_module.evaluate_release_contract(report)

    assert "release-risk" not in report.tags
    assert result.ok is True
    assert result.warnings == ()


def test_verify_plan_docs_only_uses_docs_commands() -> None:
    verify_module = _import_dev_module("devlib.verify")

    plan = verify_module.build_verify_plan(
        mode="quick",
        source="working-tree",
        base=None,
        files=["scripts/docs/check_links.py"],
        incremental=False,
        parallel=False,
    )

    assert plan.headline == "verify quick (docs-only change set)"
    assert all(command.label.startswith("docs") for command in plan.commands)


def test_impact_report_suggests_pytest_targets_for_cli_paths() -> None:
    impact_module = _import_dev_module("devlib.impact")

    report = impact_module.build_impact_report(("src/dagzoo/cli/entrypoint.py",))

    assert report.suggested_pytest_targets == (
        "tests/test_cli_validation.py",
        "tests/test_cli_outputs.py",
        "tests/test_benchmark_cli.py",
        "tests/test_generate_handoff.py",
    )


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


def test_verify_plan_code_includes_incremental_parallel_pytest_and_architecture_checks() -> None:
    verify_module = _import_dev_module("devlib.verify")

    plan = verify_module.build_verify_plan(
        mode="code",
        source="working-tree",
        base=None,
        files=["src/dagzoo/core/layout.py"],
        incremental=True,
        parallel=True,
    )

    labels = [command.label for command in plan.commands]
    assert "ruff check" in labels
    assert "mypy" in labels
    assert "deptry" in labels
    assert "import-linter" in labels
    pytest_command = next(command for command in plan.commands if command.label == "pytest")
    assert "--testmon" in pytest_command.argv
    assert "-n" in pytest_command.argv
    assert "auto" in pytest_command.argv


def test_verify_plan_affected_uses_targeted_pytest_when_safe() -> None:
    verify_module = _import_dev_module("devlib.verify")

    plan = verify_module.build_verify_plan(
        mode="affected",
        source="working-tree",
        base=None,
        files=["src/dagzoo/cli/entrypoint.py"],
        incremental=True,
        parallel=True,
    )

    pytest_command = next(command for command in plan.commands if command.label == "pytest")

    assert plan.mode == "affected"
    assert plan.report.pytest_selection.mode == "targeted"
    assert "tests/test_cli_validation.py" in plan.report.pytest_selection.targets
    assert pytest_command.argv[1:5] == ("-q", "--testmon", "-n", "auto")
    assert "tests/test_cli_validation.py" in pytest_command.argv


def test_verify_plan_affected_falls_back_to_full_for_tooling_changes() -> None:
    verify_module = _import_dev_module("devlib.verify")

    plan = verify_module.build_verify_plan(
        mode="affected",
        source="working-tree",
        base=None,
        files=["scripts/devlib/impact.py"],
        incremental=True,
        parallel=True,
    )

    pytest_command = next(command for command in plan.commands if command.label == "pytest")

    assert plan.report.pytest_selection.mode == "full"
    assert pytest_command.argv[1:5] == ("-q", "--testmon", "-n", "auto")
    assert all(not arg.startswith("tests/") for arg in pytest_command.argv)


def test_verify_plan_affected_docs_only_uses_docs_commands() -> None:
    verify_module = _import_dev_module("devlib.verify")

    plan = verify_module.build_verify_plan(
        mode="affected",
        source="working-tree",
        base=None,
        files=["scripts/docs/check_links.py"],
        incremental=False,
        parallel=False,
    )

    assert plan.headline == "verify affected (docs-only change set)"
    assert all(command.label.startswith("docs") for command in plan.commands)
    assert all(command.label != "pytest" for command in plan.commands)


def test_impact_report_treats_scripts_docs_as_docs_only() -> None:
    impact_module = _import_dev_module("devlib.impact")

    report = impact_module.build_impact_report(("scripts/docs/check_links.py",))
    payload = json.loads(impact_module.render_json(report))

    assert report.tags == ("docs", "tooling")
    assert report.recommended_modes == ("docs",)
    assert report.pytest_selection.mode == "skip"
    assert report.pytest_selection.reason == "docs-only change set"
    assert payload["recommended_modes"] == ["docs"]
    assert payload["pytest_selection"] == {
        "mode": "skip",
        "targets": [],
        "reason": "docs-only change set",
    }


def test_verify_execute_dry_run_lists_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    verify_module = _import_dev_module("devlib.verify")
    contract_module = _import_dev_module("devlib.contract")

    plan = verify_module.build_verify_plan(
        mode="quick",
        source="working-tree",
        base=None,
        files=["src/dagzoo/core/layout.py"],
        incremental=False,
        parallel=False,
    )

    monkeypatch.setattr(verify_module, "run_doctor", lambda mode: ())
    monkeypatch.setattr(verify_module, "doctor_passed", lambda results: True)
    monkeypatch.setattr(
        verify_module,
        "evaluate_release_contract",
        lambda report: contract_module.ContractResult(ok=True, warnings=(), errors=()),
    )
    monkeypatch.setattr(verify_module, "dependency_docs_are_current", lambda graph: True)

    output = verify_module.execute_verify_plan(plan, dry_run=True)

    assert "dry-run:" in output
    assert "ruff check" in output
    assert "deptry" in output


def test_verify_execute_dry_run_lists_suggested_pytest_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_module = _import_dev_module("devlib.verify")
    contract_module = _import_dev_module("devlib.contract")

    plan = verify_module.build_verify_plan(
        mode="quick",
        source="working-tree",
        base=None,
        files=["scripts/devlib/impact.py"],
        incremental=False,
        parallel=False,
    )

    monkeypatch.setattr(verify_module, "run_doctor", lambda mode: ())
    monkeypatch.setattr(verify_module, "doctor_passed", lambda results: True)
    monkeypatch.setattr(
        verify_module,
        "evaluate_release_contract",
        lambda report: contract_module.ContractResult(ok=True, warnings=(), errors=()),
    )
    monkeypatch.setattr(verify_module, "dependency_docs_are_current", lambda graph: True)

    output = verify_module.execute_verify_plan(plan, dry_run=True)

    assert "suggested pytest targets:" in output
    assert "tests/test_dev_tooling.py" in output


def test_bootstrap_environment_runs_uv_then_pre_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_module = _import_dev_module("devlib.bootstrap")
    python_path = tmp_path / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def _stub_run(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        _ = cwd
        _ = check
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(bootstrap_module, "tool_exists", lambda name: name == "uv")
    monkeypatch.setattr(bootstrap_module, "venv_python", lambda: python_path)
    monkeypatch.setattr(bootstrap_module.subprocess, "run", _stub_run)

    output = bootstrap_module.bootstrap_environment()

    assert calls == [
        ("uv", "sync", "--group", "dev"),
        (str(python_path), "-m", "pre_commit", "install"),
    ]
    assert "bootstrap complete" in output
    assert ".venv/bin/python -m pre_commit install" in output


def test_collect_review_scope_unions_review_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    review_module = _import_dev_module("devlib.review")
    merge_base_calls: list[tuple[str, ...]] = []
    line_calls: list[tuple[str, ...]] = []
    outputs = {
        ("diff", "--name-only", "abc123..HEAD"): (
            "src/dagzoo/cli/entrypoint.py",
            "CHANGELOG.md",
        ),
        ("diff", "--name-only", "--cached"): ("src/dagzoo/cli/entrypoint.py",),
        ("diff", "--name-only"): ("README.md",),
        ("ls-files", "--others", "--exclude-standard"): ("docs/new.md",),
    }

    def _stub_run_git_capture(*args: str) -> str:
        merge_base_calls.append(args)
        return "abc123\n"

    def _stub_run_git_lines(*args: str) -> tuple[str, ...]:
        line_calls.append(args)
        return outputs[args]

    monkeypatch.setattr(review_module, "run_git_capture", _stub_run_git_capture)
    monkeypatch.setattr(review_module, "run_git_lines", _stub_run_git_lines)

    scope = review_module.collect_review_scope()

    assert scope.base_ref == "origin/main"
    assert scope.merge_base == "abc123"
    assert scope.changed_files == (
        "CHANGELOG.md",
        "README.md",
        "docs/new.md",
        "src/dagzoo/cli/entrypoint.py",
    )
    assert merge_base_calls == [("merge-base", "origin/main", "HEAD")]
    assert line_calls == [
        ("diff", "--name-only", "abc123..HEAD"),
        ("diff", "--name-only", "--cached"),
        ("diff", "--name-only"),
        ("ls-files", "--others", "--exclude-standard"),
    ]


def test_render_review_base_report_includes_contract_findings() -> None:
    contract_module = _import_dev_module("devlib.contract")
    impact_module = _import_dev_module("devlib.impact")
    review_module = _import_dev_module("devlib.review")
    changed_files = ("src/dagzoo/cli/entrypoint.py",)
    report = impact_module.build_impact_report(changed_files)
    result = review_module.ReviewBaseResult(
        scope=review_module.ReviewScope(
            base_ref="origin/main",
            merge_base="abc123",
            changed_files=changed_files,
        ),
        report=report,
        contract=contract_module.evaluate_release_contract(report),
    )

    output = review_module.render_review_base_report(result)

    assert "merge base: abc123" in output
    assert "recommended verify modes:" in output
    assert "suggested pytest targets:" in output
    assert "pytest selection:" in output
    assert "release contract:" in output
    assert "error: release-risk changes require updates" in output


def test_doctor_reports_pre_commit_package_and_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doctor_module = _import_dev_module("devlib.doctor")
    python_path = tmp_path / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    hook_path = tmp_path / ".git" / "hooks" / "pre-commit"
    hook_path.parent.mkdir(parents=True)
    hook_path.write_text("#!/bin/sh\n", encoding="utf-8")

    def _stub_run(
        argv: tuple[str, ...],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        _ = cwd
        _ = check
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(doctor_module, "venv_python", lambda: python_path)
    monkeypatch.setattr(doctor_module, "git_hook_path", lambda hook_name: hook_path)
    monkeypatch.setattr(
        doctor_module.shutil,
        "which",
        lambda tool_name: f"/usr/bin/{tool_name}" if tool_name == "uv" else None,
    )
    monkeypatch.setattr(doctor_module.subprocess, "run", _stub_run)

    results = doctor_module.run_doctor("code")
    by_name = {result.name: result for result in results}

    assert by_name["pre-commit package"].ok is True
    assert by_name["pre-commit hook"].ok is True


def test_dev_cli_help_exposes_new_commands(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_dev_cli()

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "bootstrap" in captured.out
    assert "review-base" in captured.out
    assert "ready" in captured.out


def test_dev_cli_review_base_uses_default_base_ref(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_dev_cli()
    captured: dict[str, str] = {}
    fake_result = SimpleNamespace(contract=SimpleNamespace(ok=True))

    def _stub_build_review_base_result(base_ref: str = "origin/main") -> SimpleNamespace:
        captured["base_ref"] = base_ref
        return fake_result

    monkeypatch.setattr(module, "build_review_base_result", _stub_build_review_base_result)
    monkeypatch.setattr(module, "render_review_base_report", lambda result: "review report\n")

    exit_code = module.main(["review-base"])

    assert exit_code == 0
    assert captured == {"base_ref": "origin/main"}
    assert "review report" in capsys.readouterr().out


def test_dev_cli_review_base_returns_nonzero_on_contract_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_dev_cli()
    fake_result = SimpleNamespace(contract=SimpleNamespace(ok=False))

    monkeypatch.setattr(
        module, "build_review_base_result", lambda base_ref="origin/main": fake_result
    )
    monkeypatch.setattr(module, "render_review_base_report", lambda result: "review report\n")

    exit_code = module.main(["review-base"])

    assert exit_code == 1
    assert "review report" in capsys.readouterr().out


def test_dev_cli_ready_reuses_review_scope_for_affected_verify(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_dev_cli()
    captured: dict[str, object] = {}
    fake_result = SimpleNamespace(
        scope=SimpleNamespace(changed_files=("CHANGELOG.md", "src/dagzoo/cli/entrypoint.py")),
        contract=SimpleNamespace(ok=True),
    )

    monkeypatch.setattr(
        module, "build_review_base_result", lambda base_ref="origin/main": fake_result
    )
    monkeypatch.setattr(module, "render_review_base_report", lambda result: "review report\n")

    def _stub_build_verify_plan(**kwargs) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(module, "build_verify_plan", _stub_build_verify_plan)
    monkeypatch.setattr(module, "execute_verify_plan", lambda plan, dry_run: "verify output\n")

    exit_code = module.main(["ready"])

    assert exit_code == 0
    assert captured == {
        "mode": "affected",
        "source": "working-tree",
        "base": None,
        "files": ["CHANGELOG.md", "src/dagzoo/cli/entrypoint.py"],
        "incremental": True,
        "parallel": True,
    }
    output = capsys.readouterr().out
    assert "review report" in output
    assert "verify output" in output


def test_dev_cli_ready_fails_fast_when_review_base_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_dev_cli()
    fake_result = SimpleNamespace(
        scope=SimpleNamespace(changed_files=("src/dagzoo/cli/entrypoint.py",)),
        contract=SimpleNamespace(ok=False),
    )

    monkeypatch.setattr(
        module, "build_review_base_result", lambda base_ref="origin/main": fake_result
    )
    monkeypatch.setattr(module, "render_review_base_report", lambda result: "review report\n")
    monkeypatch.setattr(
        module,
        "build_verify_plan",
        lambda **kwargs: pytest.fail("ready should not build verify plan when review-base fails"),
    )

    exit_code = module.main(["ready"])

    assert exit_code == 1
    assert "review report" in capsys.readouterr().out


def test_dev_cli_contract_accepts_staged_source(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_dev_cli()
    contract_module = _import_dev_module("devlib.contract")
    captured: dict[str, object] = {}

    def _stub_detect_changed_files(
        *,
        source: str = "working-tree",
        base: str | None = None,
        files: list[str] | None = None,
    ) -> tuple[str, ...]:
        captured["source"] = source
        captured["base"] = base
        captured["files"] = files
        return ()

    monkeypatch.setattr(module, "detect_changed_files", _stub_detect_changed_files)
    monkeypatch.setattr(
        module,
        "build_impact_report",
        lambda changed_files: SimpleNamespace(changed_files=changed_files, tags=()),
    )
    monkeypatch.setattr(
        module,
        "evaluate_release_contract",
        lambda report, **_kw: contract_module.ContractResult(ok=True, warnings=(), errors=()),
    )

    exit_code = module.main(["contract", "--source", "staged"])

    assert exit_code == 0
    assert captured == {"source": "staged", "base": None, "files": None}
