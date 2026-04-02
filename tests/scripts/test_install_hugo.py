from __future__ import annotations

import io
import tarfile
from pathlib import Path

from conftest import load_script_module


def _load_module():
    return load_script_module("install_hugo_script", "scripts/ci/install_hugo.py")


def test_candidate_asset_names_prefers_linux_amd64_release_aliases() -> None:
    module = _load_module()

    candidates = module._candidate_asset_names(
        version="0.152.2",
        system="Linux",
        machine="x86_64",
        extended=True,
    )

    assert candidates == (
        "hugo_extended_0.152.2_linux-amd64.tar.gz",
        "hugo_extended_0.152.2_Linux-64bit.tar.gz",
    )


def test_select_asset_download_url_accepts_linux_64bit_fallback() -> None:
    module = _load_module()
    release_payload = {
        "assets": [
            {
                "name": "hugo_extended_0.152.2_Linux-64bit.tar.gz",
                "browser_download_url": "https://example.invalid/hugo.tar.gz",
            }
        ]
    }

    asset_name, download_url = module._select_asset_download_url(
        release_payload,
        version="0.152.2",
        system="Linux",
        machine="x86_64",
        extended=True,
    )

    assert asset_name == "hugo_extended_0.152.2_Linux-64bit.tar.gz"
    assert download_url == "https://example.invalid/hugo.tar.gz"


def test_extract_hugo_binary_writes_executable(tmp_path: Path) -> None:
    module = _load_module()
    archive_path = tmp_path / "hugo_extended_0.152.2_linux-amd64.tar.gz"
    payload = b"#!/bin/sh\nexit 0\n"
    info = tarfile.TarInfo(name="hugo")
    info.size = len(payload)
    info.mode = 0o755
    with tarfile.open(archive_path, mode="w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))

    binary_path = module._extract_hugo_binary(archive_path, out_dir=tmp_path / "bin")

    assert binary_path.exists()
    assert binary_path.read_bytes() == payload
    assert binary_path.stat().st_mode & 0o111
