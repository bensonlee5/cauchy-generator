from pathlib import Path

import pytest
from click.testing import CliRunner

import dagzoo.cli.parser as cli_parser
from dagzoo.cli.parser import build_cli
from dagzoo.cli.parsing import DEVICE_CHOICES, HARDWARE_POLICY_CHOICES

_COMMAND_BASE_ARGS = {
    "generate": ["generate", "--config", "configs/default.yaml"],
    "benchmark": ["benchmark"],
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _stub_command(monkeypatch: pytest.MonkeyPatch, command: str, seen: dict[str, object]) -> None:
    def _stub(**kwargs: object) -> int:
        seen.update(kwargs)
        return 0

    target = "run_generate_command" if command == "generate" else "run_benchmark_command"
    monkeypatch.setattr(cli_parser, target, _stub)


@pytest.mark.parametrize("command", sorted(_COMMAND_BASE_ARGS))
@pytest.mark.parametrize("device", DEVICE_CHOICES)
def test_cli_accepts_supported_device_choices(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    device: str,
) -> None:
    seen: dict[str, object] = {}
    _stub_command(monkeypatch, command, seen)

    args = [*_COMMAND_BASE_ARGS[command], "--device", device]
    if command == "generate":
        args.extend(["--num-datasets", "1", "--no-dataset-write"])
    result = runner.invoke(build_cli(), args)

    assert result.exit_code == 0
    assert seen["device"] == device


@pytest.mark.parametrize("command", sorted(_COMMAND_BASE_ARGS))
def test_cli_rejects_invalid_device_choice(runner: CliRunner, command: str) -> None:
    args = [*_COMMAND_BASE_ARGS[command], "--device", "invalid-device"]
    if command == "generate":
        args.extend(["--num-datasets", "1", "--no-dataset-write"])

    result = runner.invoke(build_cli(), args)

    assert result.exit_code == 2


@pytest.mark.parametrize("command", sorted(_COMMAND_BASE_ARGS))
@pytest.mark.parametrize("hardware_policy", HARDWARE_POLICY_CHOICES)
def test_cli_accepts_supported_hardware_policies(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    hardware_policy: str,
) -> None:
    seen: dict[str, object] = {}
    _stub_command(monkeypatch, command, seen)

    args = [*_COMMAND_BASE_ARGS[command], "--hardware-policy", hardware_policy]
    if command == "generate":
        args.extend(["--num-datasets", "1", "--no-dataset-write"])
    result = runner.invoke(build_cli(), args)

    assert result.exit_code == 0
    assert seen["hardware_policy"] == hardware_policy


@pytest.mark.parametrize("command", sorted(_COMMAND_BASE_ARGS))
def test_cli_rejects_invalid_hardware_policy_choice(runner: CliRunner, command: str) -> None:
    args = [*_COMMAND_BASE_ARGS[command], "--hardware-policy", "missing-policy"]
    if command == "generate":
        args.extend(["--num-datasets", "1", "--no-dataset-write"])

    result = runner.invoke(build_cli(), args)

    assert result.exit_code == 2


def test_cli_accepts_generate_handoff_root(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    _stub_command(monkeypatch, "generate", seen)

    result = runner.invoke(
        build_cli(),
        [
            "generate",
            "--config",
            "configs/default.yaml",
            "--handoff-root",
            "handoffs/smoke",
        ],
    )

    assert result.exit_code == 0
    assert seen["handoff_root"] == Path("handoffs/smoke")


def test_cli_exposes_only_supported_top_level_commands() -> None:
    assert set(build_cli().commands) == {
        "benchmark",
        "diversity-audit",
        "filter",
        "generate",
        "hardware",
        "publish",
        "recipe",
    }


def test_top_level_help_omits_removed_filter_calibration_subcommand(runner: CliRunner) -> None:
    result = runner.invoke(build_cli(), ["--help"])

    assert result.exit_code == 0
    assert "filter-calibration" not in result.output


def test_generate_help_mentions_handoff_incompatible_flags(runner: CliRunner) -> None:
    result = runner.invoke(build_cli(), ["generate", "--help"])

    assert result.exit_code == 0
    assert "Cannot be combined" in result.output
    assert "--out" in result.output
    assert "--no-dataset-write" in result.output


def test_benchmark_help_mentions_device_single_preset_constraint(runner: CliRunner) -> None:
    result = runner.invoke(build_cli(), ["benchmark", "--help"])

    assert result.exit_code == 0
    assert "preset 'custom'" in result.output
    assert "resolved preset" in result.output


def test_recipe_help_exposes_list_subcommand(runner: CliRunner) -> None:
    result = runner.invoke(build_cli(), ["recipe", "--help"])

    assert result.exit_code == 0
    assert "list" in result.output


def test_publish_help_exposes_hub_subcommand(runner: CliRunner) -> None:
    result = runner.invoke(build_cli(), ["publish", "--help"])

    assert result.exit_code == 0
    assert "hub" in result.output


def test_publish_hub_help_mentions_handoff_and_repo_id(runner: CliRunner) -> None:
    result = runner.invoke(build_cli(), ["publish", "hub", "--help"])

    assert result.exit_code == 0
    assert "--handoff-root" in result.output
    assert "--repo-id" in result.output
