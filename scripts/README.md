# Repository Scripts

Run these helper scripts from the repo root. The packaged `dagzoo` CLI is the
only supported CLI surface; this directory now holds repo workflow tooling,
docs helpers, and maintenance utilities rather than convenience wrappers.

## Developer CLI

- `.venv/bin/nox -s quick`
  - Canonical local code-quality verification on the shared repo `.venv/`.
- `.venv/bin/nox -s full`
  - Runs the quick checks, docs checks, and the full pytest suite.
- `.venv/bin/nox -s docs`
  - Runs the docs sync, docs link, and built-site verification workflow.
- `.venv/bin/nox -s bench_smoke`
  - Runs the supported smoke benchmark regression workflow.
- `./scripts/dev bootstrap`
  - Creates or syncs `.venv/` via `uv sync --group dev`, installs the Hugo site Node deps with `npm ci --prefix site`, and installs the repo pre-commit hook.
- `./scripts/dev impact [--source working-tree|staged|base] [--base <git-ref>] [--files ...] [--format text|json]`
  - Classifies changed files and shows dependency-aware downstream impact.
- `./scripts/dev contract [--source working-tree|staged|base] [--base <git-ref>] [--files ...] [--strict]`
  - Enforces version/changelog expectations for likely user-facing changes.
- `./scripts/dev review-base [--base-ref <git-ref>]`
  - Summarizes the current review scope against the selected base ref, including contract warnings/errors.

## Scripts

- `scripts/fetch-additional-references.sh`
  - Downloads a curated hardcoded list of additional arXiv papers used for local reference refreshes.
- `scripts/bump-version.sh <major|minor|patch> [--dry-run] [--tag]`
  - Bump the semver version in `pyproject.toml`. Use `--tag` to commit and create a git tag.
- `scripts/cleanup_local_artifacts.py [--group runtime|docs|all] [--apply]`
  - Dry-run or remove ignored local runtime/docs outputs (`data/`, `benchmarks/results/`, and the built docs output under `site/`) without touching tracked files.
- `scripts/evaluate_handoff_pareto.py --baseline-config <path> --out-root <dir> [--stress-profile <name> ...] [--variant-config <path> ...]`
  - Runs matched `dagzoo generate --handoff-root` baseline/variant comparisons and writes RD-005 maintainer summaries for structural diversity, throughput, and the anti-triviality downstream ceiling.
- `scripts/docs/sync_hugo_content.py [--check]`
  - Sync canonical docs from `docs/` into the generated Hugo input area described in `site/README.md` (single-source docs model).
- `scripts/docs/check_links.py [roots...]`
  - Validate local Markdown/HTML links across source docs and generated site content.
- `scripts/docs/check_built_output_links.py [output_dir]`
  - Validate internal links in built Hugo output and enforce base-path-safe absolute links.

## Examples

```bash
.venv/bin/nox -s quick
.venv/bin/nox -s full
.venv/bin/nox -s docs
.venv/bin/nox -s bench_smoke
./scripts/dev bootstrap
./scripts/dev review-base
./scripts/dev impact
./scripts/dev impact --source staged
./scripts/dev impact --files src/dagzoo/core/execution_semantics.py
./scripts/dev contract --source staged
./scripts/fetch-additional-references.sh
./.venv/bin/python scripts/docs/sync_hugo_content.py
./.venv/bin/python scripts/docs/sync_hugo_content.py --check
./.venv/bin/python scripts/docs/check_links.py
./.venv/bin/python scripts/docs/check_built_output_links.py site/public
./.venv/bin/python scripts/cleanup_local_artifacts.py --group all
./.venv/bin/python scripts/cleanup_local_artifacts.py --group runtime --apply
./.venv/bin/python scripts/evaluate_handoff_pareto.py --baseline-config configs/default.yaml --stress-profile anti_memorization_piecewise_classification_graph_breadth_slice_v1 --stress-profile anti_memorization_piecewise_classification_compositional_slice_v1 --out-root benchmarks/results/rd005_pareto --num-datasets 8 --seed 123 --device cpu
uv run dagzoo generate --config configs/default.yaml --num-datasets 50 --device cpu --out data/run_cpu_50
uv run dagzoo generate --config configs/preset_cuda_h100.yaml --num-datasets 500 --device cuda --out data/run_h100_500 --seed 123
uv run dagzoo generate --config configs/preset_many_class_generate_smoke.yaml --num-datasets 25 --device cpu --out data/run_many_class --seed 123
uv run dagzoo generate --config configs/preset_noise_gaussian_generate_smoke.yaml --num-datasets 25 --device cpu --out data/run_noise_gaussian --seed 123
uv run dagzoo generate --config configs/preset_noise_mixture_generate_smoke.yaml --num-datasets 25 --device cpu --out data/run_noise_mixture --seed 124
uv run dagzoo generate --config configs/default.yaml --num-datasets 3 --device cpu --no-dataset-write
uv run dagzoo generate --config configs/preset_missingness_mcar.yaml --num-datasets 25 --device cpu --out data/run_missing_mcar --seed 101
uv run dagzoo generate --config configs/preset_missingness_mar.yaml --num-datasets 25 --device cpu --out data/run_missing_mar --seed 102
uv run dagzoo benchmark --suite smoke --preset cpu --out-dir benchmarks/results/smoke_cpu
uv run dagzoo benchmark --suite standard --preset all --out-dir benchmarks/results/latest
uv run dagzoo benchmark --suite smoke --preset cpu --diagnostics --diagnostics-out-dir benchmarks/results/smoke_cpu_diag --out-dir benchmarks/results/smoke_cpu_diag
./.venv/bin/python scripts/ci/h100_validation.py
./.venv/bin/python scripts/ci/h100_validation.py --out-root benchmarks/results/gpu_h100_manual
uv run dagzoo generate --config configs/preset_diagnostics_on.yaml --num-datasets 25 --diagnostics --out data/run_diag
uv run dagzoo generate --config configs/default.yaml --rows 1024 --num-datasets 25 --out data/run_rows_1024
uv run dagzoo generate --config configs/default.yaml --rows 400..60000 --num-datasets 50 --no-dataset-write
uv run dagzoo generate --config configs/preset_missingness_mnar.yaml --num-datasets 25 --out data/run_missing_mnar
uv run dagzoo generate --config configs/preset_noise_student_t_generate_smoke.yaml --num-datasets 25 --out data/run_noise_student_t
uv run dagzoo benchmark --config configs/preset_missingness_mar.yaml --preset custom --suite smoke --no-memory --out-dir benchmarks/results/smoke_missing_mar
uv run dagzoo benchmark --config configs/preset_noise_benchmark_smoke.yaml --preset custom --suite smoke --no-memory --out-dir benchmarks/results/smoke_noise
./scripts/bump-version.sh patch --dry-run
./scripts/bump-version.sh minor --tag
```

`dagzoo benchmark --preset all` includes CUDA presets and will hard-fail if CUDA is unavailable.

When diagnostics is enabled for benchmark scripts, coverage artifacts are written under:

- `<out_dir>/diagnostics/<sanitized_preset_key>_<hash>/coverage_summary.json`
- `<out_dir>/diagnostics/<sanitized_preset_key>_<hash>/coverage_summary.md`

The diagnostics preset directory is sanitized and hash-suffixed (for example, `cpu_ca49ca4b`) to keep paths unique and filesystem-safe.

Docs workflow note: the built Hugo output lives under `site/`. For the full
single-source docs/rendered-reference model, see
`docs/development/design-decisions.md` and `site/README.md`.

When benchmark scenarios are enabled, summary JSON includes
`preset_results[*].scenarios` with keys for `baseline`, `throughput`, `filtering`,
`missingness`, `shift`, and `noise`.

Scenario entries may include `control_metrics` and `issues`; missingness, shift,
and noise can escalate suite regression status through those scenario issues.

The H100 validation runner writes a root manifest at
`<out_dir>/validation_manifest.json`. Primary H100 performance phases also write:

- `<phase_dir>/gpu_telemetry.csv`
- `<phase_dir>/gpu_telemetry_summary.json`

Use those alongside each phase `summary.json` to inspect generation timing,
write-stage timing, fixed-layout batch/chunking, and GPU memory/utilization
evidence without inferring bottlenecks from datasets/minute alone.
