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


def test_direct_asset_download_url_uses_release_download_path() -> None:
    module = _load_module()

    url = module._direct_asset_download_url(
        version="0.152.2",
        asset_name="hugo_extended_0.152.2_linux-amd64.tar.gz",
    )

    assert url == (
        "https://github.com/gohugoio/hugo/releases/download/"
        "v0.152.2/hugo_extended_0.152.2_linux-amd64.tar.gz"
    )


def test_download_direct_release_asset_falls_back_across_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    requested_urls: list[str] = []

    def _fake_download(url: str, destination: Path) -> None:
        requested_urls.append(url)
        if url.endswith("linux-amd64.tar.gz"):
            raise module.urllib.error.HTTPError(url, 404, "missing", hdrs=None, fp=None)
        destination.write_bytes(b"ok")

    monkeypatch.setattr(module, "_download_file", _fake_download)

    archive_path = module._download_direct_release_asset(
        version="0.152.2",
        destination_dir=tmp_path,
        system="Linux",
        machine="x86_64",
        extended=True,
    )

    assert archive_path == tmp_path / "hugo_extended_0.152.2_Linux-64bit.tar.gz"
    assert requested_urls == [
        "https://github.com/gohugoio/hugo/releases/download/"
        "v0.152.2/hugo_extended_0.152.2_linux-amd64.tar.gz",
        "https://github.com/gohugoio/hugo/releases/download/"
        "v0.152.2/hugo_extended_0.152.2_Linux-64bit.tar.gz",
    ]


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
