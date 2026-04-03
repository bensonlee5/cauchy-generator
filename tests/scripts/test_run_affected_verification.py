from __future__ import annotations

from types import SimpleNamespace

from conftest import load_script_module


def _load_module():
    return load_script_module(
        "affected_verification_script",
        "scripts/ci/run_affected_verification.py",
    )


def _report(module, *, changed_files, tags=(), selection=None):
    if selection is None:
        selection = module.PytestSelection(
            mode="skip",
            targets=(),
            reason="skip requested",
        )
    return SimpleNamespace(
        changed_files=tuple(changed_files),
        tags=tuple(tags),
        pytest_selection=selection,
        recommended_modes=("quick",),
    )


def test_docs_only_change_set_selects_docs_checks_and_skips_pytest() -> None:
    module = _load_module()
    report = _report(
        module,
        changed_files=("docs/features/noise.md",),
        tags=("docs",),
        selection=module.PytestSelection(
            mode="skip",
            targets=(),
            reason="docs-only change set",
        ),
    )

    commands = module.build_affected_verification_commands(
        report,
        incremental=True,
        parallel=True,
    )

    assert [command.label for command in commands] == [
        "docs repo paths",
        "docs sync",
        "docs sync check",
        "docs links",
        "docs build",
        "docs built links",
        "docs adoption surface",
    ]


def test_targeted_selection_runs_pytest_with_explicit_targets() -> None:
    module = _load_module()
    report = _report(
        module,
        changed_files=("src/dagzoo/cli/parser.py",),
        tags=("code",),
        selection=module.PytestSelection(
            mode="targeted",
            targets=("tests/test_cli_validation.py", "tests/test_cli_outputs.py"),
            reason="targeted selection",
        ),
    )

    commands = module.build_affected_verification_commands(
        report,
        incremental=True,
        parallel=True,
    )

    pytest_command = next(command for command in commands if command.label == "pytest")
    assert pytest_command.argv[:4] == (module.python_tool("pytest"), "-q", "--testmon", "-n")
    assert pytest_command.argv[4] == "auto"
    assert pytest_command.argv[-2:] == (
        "tests/test_cli_validation.py",
        "tests/test_cli_outputs.py",
    )


def test_full_selection_runs_full_pytest() -> None:
    module = _load_module()
    report = _report(
        module,
        changed_files=("tests/test_dev_tooling.py",),
        tags=("code", "tooling"),
        selection=module.PytestSelection(
            mode="full",
            targets=(),
            reason="full pytest required",
        ),
    )

    commands = module.build_affected_verification_commands(
        report,
        incremental=True,
        parallel=True,
    )

    pytest_command = next(command for command in commands if command.label == "pytest")
    assert pytest_command.argv == (
        module.python_tool("pytest"),
        "-q",
        "-n",
        "auto",
    )


def test_skip_selection_omits_pytest_for_non_docs_change_set() -> None:
    module = _load_module()
    report = _report(
        module,
        changed_files=("src/dagzoo/diagnostics/coverage.py",),
        tags=("code",),
        selection=module.PytestSelection(
            mode="skip",
            targets=(),
            reason="skip selection",
        ),
    )

    commands = module.build_affected_verification_commands(
        report,
        incremental=True,
        parallel=True,
    )

    assert "pytest" not in [command.label for command in commands]


def test_architecture_tag_includes_import_linter() -> None:
    module = _load_module()
    report = _report(
        module,
        changed_files=("src/dagzoo/core/execution_semantics.py",),
        tags=("architecture", "code"),
        selection=module.PytestSelection(
            mode="skip",
            targets=(),
            reason="skip selection",
        ),
    )

    commands = module.build_affected_verification_commands(
        report,
        incremental=False,
        parallel=False,
    )

    assert "import-linter" in [command.label for command in commands]


def test_run_affected_verification_executes_commands_and_renders_summary(monkeypatch) -> None:
    module = _load_module()
    report = _report(
        module,
        changed_files=("src/dagzoo/cli/parser.py",),
        tags=("code",),
        selection=module.PytestSelection(
            mode="targeted",
            targets=("tests/test_cli_validation.py",),
            reason="targeted selection",
        ),
    )
    executed: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr(module, "detect_changed_files", lambda **_kwargs: report.changed_files)
    monkeypatch.setattr(module, "build_impact_report", lambda _changed_files: report)
    monkeypatch.setattr(
        module,
        "run_command",
        lambda command: executed.append((command.label, command.argv)),
    )

    summary = module.run_affected_verification(
        base="origin/main",
        incremental=True,
        parallel=True,
    )

    assert summary.startswith("affected verification\n")
    assert "pytest selection: targeted" in summary
    assert "tests/test_cli_validation.py" in summary
    assert executed
