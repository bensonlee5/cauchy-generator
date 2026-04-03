#!/usr/bin/env python3
"""Install one pinned Hugo release into a local bin directory."""

from __future__ import annotations

import json
import platform
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import click

_RELEASE_API_TEMPLATE = "https://api.github.com/repos/gohugoio/hugo/releases/tags/v{version}"
_RELEASE_DOWNLOAD_TEMPLATE = (
    "https://github.com/gohugoio/hugo/releases/download/v{version}/{asset_name}"
)
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _normalized_system(value: str | None = None) -> str:
    system = (value or platform.system()).strip().lower()
    if system == "darwin":
        return "darwin"
    if system == "linux":
        return "linux"
    if system.startswith("win"):
        return "windows"
    raise ValueError(f"Unsupported operating system for Hugo install: {value!r}.")


def _normalized_machine(value: str | None = None) -> str:
    machine = (value or platform.machine()).strip().lower()
    aliases = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    try:
        return aliases[machine]
    except KeyError as exc:
        raise ValueError(f"Unsupported architecture for Hugo install: {value!r}.") from exc


def _candidate_asset_names(
    *,
    version: str,
    system: str | None = None,
    machine: str | None = None,
    extended: bool,
) -> tuple[str, ...]:
    normalized_system = _normalized_system(system)
    normalized_machine = _normalized_machine(machine)
    prefix = "hugo_extended" if extended else "hugo"
    if normalized_system == "windows":
        extension = ".zip"
    else:
        extension = ".tar.gz"

    platform_tokens: list[str]
    if normalized_system == "linux" and normalized_machine == "amd64":
        platform_tokens = ["linux-amd64", "Linux-64bit"]
    elif normalized_system == "linux" and normalized_machine == "arm64":
        platform_tokens = ["linux-arm64"]
    elif normalized_system == "darwin":
        platform_tokens = ["darwin-universal", f"darwin-{normalized_machine}"]
    elif normalized_system == "windows":
        platform_tokens = [f"windows-{normalized_machine}"]
    else:
        platform_tokens = [f"{normalized_system}-{normalized_machine}"]

    return tuple(f"{prefix}_{version}_{token}{extension}" for token in platform_tokens)


def _fetch_release(version: str) -> dict[str, Any]:
    url = _RELEASE_API_TEMPLATE.format(version=str(version).strip())
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "dagzoo-ci-install-hugo",
        },
    )
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected release payload shape for Hugo {version!r}.")
    return payload


def _direct_asset_download_url(*, version: str, asset_name: str) -> str:
    return _RELEASE_DOWNLOAD_TEMPLATE.format(
        version=str(version).strip().removeprefix("v"),
        asset_name=str(asset_name),
    )


def _select_asset_download_url(
    release_payload: dict[str, Any],
    *,
    version: str,
    system: str | None = None,
    machine: str | None = None,
    extended: bool,
) -> tuple[str, str]:
    assets = release_payload.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError(f"Hugo release payload for {version!r} did not include assets.")
    by_name = {
        str(asset.get("name")): str(asset.get("browser_download_url"))
        for asset in assets
        if isinstance(asset, dict)
        and isinstance(asset.get("name"), str)
        and isinstance(asset.get("browser_download_url"), str)
    }
    for candidate in _candidate_asset_names(
        version=version,
        system=system,
        machine=machine,
        extended=extended,
    ):
        url = by_name.get(candidate)
        if url:
            return candidate, url
    expected = ", ".join(
        _candidate_asset_names(
            version=version,
            system=system,
            machine=machine,
            extended=extended,
        )
    )
    available = ", ".join(sorted(by_name))
    raise RuntimeError(
        f"Unable to find a matching Hugo asset for version {version!r}. "
        f"Expected one of: {expected}. Available assets: {available}."
    )


def _download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "dagzoo-ci-install-hugo"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download_direct_release_asset(
    *,
    version: str,
    destination_dir: Path,
    system: str | None = None,
    machine: str | None = None,
    extended: bool,
) -> Path | None:
    for asset_name in _candidate_asset_names(
        version=version,
        system=system,
        machine=machine,
        extended=extended,
    ):
        archive_path = destination_dir / asset_name
        try:
            _download_file(
                _direct_asset_download_url(version=version, asset_name=asset_name), archive_path
            )
        except urllib.error.HTTPError as exc:
            if exc.code not in {403, 404}:
                raise
            continue
        except urllib.error.URLError:
            continue
        return archive_path
    return None


def _extract_hugo_binary(archive_path: Path, *, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_name = "hugo.exe" if archive_path.suffix == ".zip" else "hugo"
    destination = out_dir / binary_name

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if Path(member.filename).name != binary_name:
                    continue
                with archive.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                break
            else:
                raise RuntimeError(f"No {binary_name!r} binary found in {archive_path.name}.")
    else:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive.getmembers():
                if Path(member.name).name != binary_name:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(
                        f"Unable to extract {binary_name!r} from {archive_path.name}."
                    )
                with extracted, destination.open("wb") as dst:
                    shutil.copyfileobj(extracted, dst)
                break
            else:
                raise RuntimeError(f"No {binary_name!r} binary found in {archive_path.name}.")

    current_mode = destination.stat().st_mode
    destination.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return destination


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--version", required=True, help="Hugo version without the leading v.")
@click.option("--extended", is_flag=True, help="Install the extended Hugo build.")
@click.option(
    "--bin-dir",
    required=True,
    help="Directory where the extracted Hugo binary should be written.",
)
@click.option(
    "--github-path-file",
    default=None,
    help="Optional path to the GitHub Actions PATH file.",
)
def cli(
    *,
    version: str,
    extended: bool,
    bin_dir: str,
    github_path_file: str | None,
) -> int:
    """Install one pinned Hugo release into a local bin directory."""

    version = str(version).strip().removeprefix("v")
    bin_dir = Path(bin_dir).resolve()

    with tempfile.TemporaryDirectory(prefix="dagzoo_hugo_") as temp_dir:
        temp_root = Path(temp_dir)
        archive_path = _download_direct_release_asset(
            version=version,
            destination_dir=temp_root,
            extended=bool(extended),
        )
        if archive_path is None:
            release_payload = _fetch_release(version)
            asset_name, download_url = _select_asset_download_url(
                release_payload,
                version=version,
                extended=bool(extended),
            )
            archive_path = temp_root / asset_name
            _download_file(download_url, archive_path)
        binary_path = _extract_hugo_binary(archive_path, out_dir=bin_dir)

    if github_path_file is not None:
        with Path(github_path_file).open("a", encoding="utf-8") as handle:
            handle.write(f"{bin_dir}\n")
    print(f"Installed Hugo {version} to {binary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        result = cli.main(args=argv, prog_name="install_hugo.py", standalone_mode=False)
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return int(exc.exit_code)
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.Abort:
        return 1
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
