from __future__ import annotations

import os
from pathlib import Path

import nox

nox.options.sessions = ["quick"]

_REPO_ROOT = Path(__file__).resolve().parent
_VENV_BIN = _REPO_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
_PYTHON = _VENV_BIN / ("python.exe" if os.name == "nt" else "python")
_DAGZOO = _VENV_BIN / ("dagzoo.exe" if os.name == "nt" else "dagzoo")
_LINT_IMPORTS = _VENV_BIN / ("lint-imports.exe" if os.name == "nt" else "lint-imports")

_BENCH_SMOKE_ARGS = (
    "dagzoo",
    "benchmark",
    "--suite",
    "smoke",
    "--preset",
    "cpu",
    "--baseline",
    "benchmarks/baselines/cpu_smoke.json",
    "--warn-threshold-pct",
    "10",
    "--fail-threshold-pct",
    "20",
    "--fail-on-regression",
    "--hardware-policy",
    "none",
    "--no-memory",
    "--out-dir",
    "benchmarks/results/dev_smoke",
)

_BENCH_PUBLIC_SMOKE_ARGS = (
    "-m",
    "dagzoo.bench.public_throughput_smoke",
    "--config",
    "configs/benchmark_cpu.yaml",
    "--baseline",
    "benchmarks/baselines/cpu_public_smoke.json",
    "--warn-threshold-pct",
    "10",
    "--fail-threshold-pct",
    "20",
    "--fail-on-regression",
    "--out-dir",
    "benchmarks/results/dev_public_smoke",
)


def _run_quick_checks(session: nox.Session) -> None:
    session.run(str(_PYTHON), "-m", "ruff", "check", "src", "tests", "scripts")
    session.run(str(_PYTHON), "-m", "ruff", "format", "--check", "src", "tests", "scripts")
    session.run(str(_PYTHON), "-m", "mypy", "src")
    session.run(str(_PYTHON), "-m", "deptry", ".")
    session.run(str(_LINT_IMPORTS))
    session.run(str(_PYTHON), "scripts/ci/check_adoption_surface.py")


def _run_docs_checks(session: nox.Session) -> None:
    session.run(str(_PYTHON), "scripts/docs/check_repo_paths.py")
    session.run(str(_PYTHON), "scripts/docs/sync_hugo_content.py")
    session.run(str(_PYTHON), "scripts/docs/sync_hugo_content.py", "--check")
    session.run(str(_PYTHON), "scripts/docs/check_links.py")
    session.run(
        "hugo",
        "--source",
        "site",
        "--minify",
        "--gc",
        "--destination",
        "public",
        external=True,
    )
    session.run(str(_PYTHON), "scripts/docs/check_built_output_links.py", "site/public")


@nox.session(venv_backend="none")
def quick(session: nox.Session) -> None:
    _run_quick_checks(session)


@nox.session(venv_backend="none")
def docs(session: nox.Session) -> None:
    _run_docs_checks(session)


@nox.session(venv_backend="none", name="bench_smoke")
def bench_smoke(session: nox.Session) -> None:
    session.run(str(_DAGZOO), *_BENCH_SMOKE_ARGS[1:])


@nox.session(venv_backend="none", name="bench_public_smoke")
def bench_public_smoke(session: nox.Session) -> None:
    session.run(str(_PYTHON), *_BENCH_PUBLIC_SMOKE_ARGS)


@nox.session(venv_backend="none")
def full(session: nox.Session) -> None:
    _run_quick_checks(session)
    _run_docs_checks(session)
    session.run(str(_PYTHON), "-m", "pytest", "-q", "-n", "auto")
