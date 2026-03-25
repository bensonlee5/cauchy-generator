# Third-Party Notices

This repository's own source code is licensed under the Apache License 2.0. See
[LICENSE](LICENSE).

This notice file documents the third-party components that are relevant to
repo-level disclosure and packaged distribution review as of March 25, 2026.
It is intentionally scoped to:

- Declared runtime dependencies of the published `dagzoo` Python package.
- Tracked docs-site build dependencies declared in this repository.
- Redistribution guidance for cases where downstream artifacts bundle third-party
  package contents.

It is not a complete SBOM for every possible virtualenv, npm install tree, or
container image built from this repo.

## Current Repo Status

- No vendored third-party source trees or binary assets are tracked in the Git
  repository.
- `site/node_modules/` and `site/resources/` are ignored build artifacts rather
  than committed source.
- The published Python package metadata declares
  `License-Expression: Apache-2.0` and is configured to ship `LICENSE`,
  `NOTICE`, and this notice file in built distributions.

## Python Runtime Dependencies

These packages are declared as runtime dependencies in `pyproject.toml`. They
are installed separately by package managers; they are not vendored into this
repository.

| Package        | License metadata observed locally                    | Notes                                                                   |
| -------------- | ---------------------------------------------------- | ----------------------------------------------------------------------- |
| `numpy`        | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | NumPy metadata lists multiple bundled upstream license files.           |
| `torch`        | `BSD-3-Clause`                                       | Package metadata also includes upstream `LICENSE` and `NOTICE` files.   |
| `PyYAML`       | `MIT`                                                | Standard permissive attribution notice.                                 |
| `pyarrow`      | `Apache-2.0`                                         | Package metadata also includes upstream `LICENSE.txt` and `NOTICE.txt`. |
| `scikit-learn` | `BSD-3-Clause`                                       | Standard BSD-3-Clause notice obligations apply on redistribution.       |

## Docs Site Build Dependencies

These components are declared in tracked site manifests:

| Component                 | Source in repo                                 | License    |
| ------------------------- | ---------------------------------------------- | ---------- |
| `github.com/google/docsy` | `site/go.mod`                                  | Apache-2.0 |
| `autoprefixer`            | `site/package.json` / `site/package-lock.json` | MIT        |
| `postcss`                 | `site/package.json` / `site/package-lock.json` | MIT        |
| `postcss-cli`             | `site/package.json` / `site/package-lock.json` | MIT        |

The tracked npm lockfile also includes transitive packages with additional
attribution obligations, including `caniuse-lite` under `CC-BY-4.0`. Because
`site/node_modules/` is not committed, those packages are not redistributed by
the source repository itself. If you redistribute the installed npm dependency
tree or a container/image that bundles it, preserve the upstream attribution and
license text required by those packages.

## Redistribution Guidance

- Source repo distribution: keep `LICENSE` at the repo root and do not commit
  vendored third-party code without its original license/notice files.
- Python package distribution: keep `LICENSE`, `NOTICE`, and this file in
  sdists and wheels so downstream consumers can see both your project license
  and the third-party dependency landscape.
- Binary/container distribution: if you publish an environment that bundles
  Python, Go, or npm package contents, generate an image-level notice bundle
  from the exact installed dependency set rather than relying only on this file.
- Apache-2.0 components can require preservation of upstream `NOTICE` content on
  redistribution.
- CC-BY-4.0 materials require attribution, a license link, and indication of
  changes when redistributed in covered form.

## Maintenance

Update this file when any of the following change:

- `pyproject.toml` runtime dependencies
- `site/go.mod`
- `site/package.json` / `site/package-lock.json`
- Any newly vendored third-party source, assets, fonts, or generated bundles

This document is operational guidance, not legal advice. For a release that
ships bundled third-party package contents, review the exact dependency tree and
its license files before publication.
