from __future__ import annotations

from pathlib import Path

from conftest import load_script_module


def test_sync_docs_helpers_handle_route_aliases_and_heading_attrs() -> None:
    module = load_script_module("sync_hugo_content", "scripts/docs/sync_hugo_content.py")

    stripped = module._strip_matching_h1(
        "# How dagzoo Works {#how-dagzoo-works}\n\nBody\n",
        title="How dagzoo Works",
    )
    route_map = module._build_route_map("/dagzoo")

    assert stripped == "Body\n"
    assert route_map["how-it-works.md"] == "/dagzoo/docs/how-it-works/"
    assert route_map["transforms.md"] == "/dagzoo/docs/transforms/"
    assert route_map["how-it-works.html"] == "/dagzoo/docs/how-it-works/"
    assert route_map["transforms.html"] == "/dagzoo/docs/transforms/"


def test_sync_docs_rewrites_usage_guide_links_to_steering_page() -> None:
    module = load_script_module("sync_hugo_content", "scripts/docs/sync_hugo_content.py")

    route_map = module._build_route_map("/dagzoo")
    rewritten = module._rewrite_markdown_links(
        "Detailed guide: [Meta-Feature Coverage Steering](features/steering.md)\n",
        "usage-guide.md",
        route_map,
    )

    assert route_map["features/steering.md"] == "/dagzoo/docs/features/steering/"
    assert (
        rewritten
        == "Detailed guide: [Meta-Feature Coverage Steering](/dagzoo/docs/features/steering/)\n"
    )
    assert module.PAGE_METADATA["features/steering.md"].weight == 64


def test_sync_docs_front_matter_includes_aliases_and_page_flags() -> None:
    module = load_script_module("sync_hugo_content", "scripts/docs/sync_hugo_content.py")
    meta = module.PAGE_METADATA["how-it-works.md"]

    front_matter = module._front_matter(
        "How It Works",
        meta,
        extra_params={"canonical_repo_path": "docs/how-it-works.md"},
    )

    assert "title: How It Works" in front_matter
    assert "aliases:" in front_matter
    assert "/canonical/how-it-works.html" in front_matter
    assert "mermaid: true" in front_matter
    assert "canonical_repo_path: docs/how-it-works.md" in front_matter


def test_sync_docs_removes_legacy_generated_paths(tmp_path) -> None:
    module = load_script_module("sync_hugo_content", "scripts/docs/sync_hugo_content.py")
    legacy_dir = tmp_path / "static" / "canonical"
    legacy_dir.mkdir(parents=True)

    changed: list[Path] = []
    module._remove_stale_path(legacy_dir, check=False, changed=changed)

    assert changed == [legacy_dir]
    assert not legacy_dir.exists()


def test_check_repo_paths_passes_for_existing_repo_paths(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    module = load_script_module("check_repo_paths", "scripts/docs/check_repo_paths.py")
    (tmp_path / "src" / "dagzoo").mkdir(parents=True)
    (tmp_path / "src" / "dagzoo" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tool.py").write_text("", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text("", encoding="utf-8")
    (tmp_path / "docs" / "features").mkdir(parents=True)
    (tmp_path / "docs" / "features" / "diagnostics.md").write_text(
        "# Diagnostics\n", encoding="utf-8"
    )
    usage_guide = tmp_path / "docs" / "usage-guide.md"
    usage_guide.write_text(
        "\n".join(
            [
                "Use `src/dagzoo/__init__.py` or `scripts/tool.py [--check]`, and allow `configs/*.yaml`.",
                "See [Diagnostics](features/diagnostics.md) and [Tool](../scripts/tool.py#usage).",
                'Config link: [default](../configs/default.yaml "default config").',
                "Ignore [external](https://example.com), [anchor](#overview), and [site](/dagzoo/docs/how-it-works/).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    code = module.main(["docs"])

    assert code == 0
    assert "Repo path check passed." in capsys.readouterr().out


def test_check_repo_paths_rejects_missing_and_legacy_markdown_links(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    module = load_script_module("check_repo_paths", "scripts/docs/check_repo_paths.py")
    readme = tmp_path / "README.md"
    readme.write_text(
        "Legacy [CLI](src/dagzoo/cli.py) and missing [Doc](docs/does_not_exist.md).\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    code = module.main(["README.md"])
    output = capsys.readouterr().out

    assert code == 1
    assert "src/dagzoo/cli.py (legacy path)" in output
    assert "docs/does_not_exist.md (missing path)" in output


def test_check_repo_paths_rejects_missing_scan_root(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    module = load_script_module("check_repo_paths", "scripts/docs/check_repo_paths.py")
    (tmp_path / "README.md").write_text("# Repo\n", encoding="utf-8")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    code = module.main(["README.md", "docs"])
    output = capsys.readouterr().out

    assert code == 1
    assert "Missing Markdown scan roots:" in output
    assert "- docs" in output


def test_check_links_accepts_repo_site_urls_in_readme(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    module = load_script_module("check_links", "scripts/docs/check_links.py")
    (tmp_path / "site" / "content" / "docs").mkdir(parents=True)
    (tmp_path / "site" / "content" / "docs" / "_index.md").write_text(
        "---\ntitle: Docs\n---\n",
        encoding="utf-8",
    )
    (tmp_path / "site" / "hugo.yaml").write_text(
        "baseURL: https://example.com/dagzoo/\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "[Docs](https://example.com/dagzoo/docs/)\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "SITE_ROOT", tmp_path / "site")
    monkeypatch.setattr(module, "SITE_CONTENT_ROOT", tmp_path / "site" / "content")
    monkeypatch.setattr(module, "BASE_PATH", "/dagzoo")
    monkeypatch.setattr(module, "BASE_URL", "https://example.com/dagzoo")
    monkeypatch.setattr(module, "BASE_URL_PARSED", module.urlparse("https://example.com/dagzoo"))

    code = module.main(["README.md"])

    assert code == 0
    assert "Link check passed." in capsys.readouterr().out


def test_check_built_output_links_rejects_generated_source_links(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    module = load_script_module(
        "check_built_output_links",
        "scripts/docs/check_built_output_links.py",
    )
    (tmp_path / "site" / "public" / "docs").mkdir(parents=True)
    (tmp_path / "site" / "hugo.yaml").write_text(
        "baseURL: https://example.com/dagzoo/\n",
        encoding="utf-8",
    )
    (tmp_path / "site" / "public" / "docs" / "index.html").write_text(
        (
            "<html><body>"
            '<a href="https://github.com/example/repo/tree/main/.generated/content/docs/page.md">'
            "View page source"
            "</a>"
            "</body></html>"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "SITE_ROOT", tmp_path / "site")
    monkeypatch.setattr(module, "BASE_PATH", "/dagzoo")

    code = module.main(["site/public"])
    output = capsys.readouterr().out

    assert code == 1
    assert "Generated-source GitHub link violations:" in output
    assert ".generated/content/docs/page.md" in output
