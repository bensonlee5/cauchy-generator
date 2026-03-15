# Repository Scripts

Run these wrappers and helper scripts from the repo root. CLI-oriented entries
typically use `uv run dagzoo ...`; docs and maintenance helpers are invoked
directly. The `generate-*.sh` files are convenience recipes for local repo
workflows, not the primary interface; prefer `dagzoo generate` and
`dagzoo benchmark` when you need flexible arguments.

## Developer CLI

- `./scripts/dev doctor [code|docs|all]`
  - Verifies local toolchain prerequisites for repo work.
- `./scripts/dev deps [--scope package|hybrid|full] [--format text|json] [--write-docs] [--check]`
  - Builds the repo dependency graph and can refresh the checked-in docs snapshot.
- `./scripts/dev impact [--source working-tree|staged|base] [--base <git-ref>] [--files ...] [--format text|json]`
  - Classifies changed files and shows dependency-aware downstream impact.
- `./scripts/dev contract [--source working-tree|staged|base] [--base <git-ref>] [--files ...] [--strict]`
  - Enforces version/changelog expectations for likely user-facing changes.
- `./scripts/dev verify quick|code|docs|bench|full [--source working-tree|staged|base] [--base <git-ref>] [--files ...] [--dry-run] [--incremental] [--parallel]`
  - Canonical local verification entrypoint for normal code, docs, and benchmark work.

## Scripts

- `scripts/generate-from-config.sh [config] [num_datasets] [device] [out_dir] [seed]`
  - Generic generator wrapper for any config.
- `scripts/generate-default.sh [num_datasets] [device] [out_dir] [seed]`
  - Uses `configs/default.yaml`.
- `scripts/generate-h100.sh [num_datasets] [device] [out_dir] [seed]`
  - Uses `configs/preset_cuda_h100.yaml`.
- `scripts/generate-many-class.sh [num_datasets] [device] [out_dir] [seed]`
  - Uses `configs/preset_many_class_generate_smoke.yaml`.
- `scripts/generate-noise.sh [family] [num_datasets] [device] [out_dir] [seed]`
  - Runs preset-based noise family workflows (`gaussian`, `laplace`, `student_t`, `mixture`).
- `scripts/generate-smoke.sh [config] [num_datasets] [device]`
  - Runs quick in-memory generation with `--no-dataset-write`.
- `scripts/generate-missingness.sh [mechanism] [missing_rate] [num_datasets] [device] [out_dir] [seed]`
  - Runs generation with CLI-level missingness overrides (`mcar`, `mar`, `mnar`).
- `scripts/fetch-additional-references.sh`
  - Downloads a curated hardcoded list of additional arXiv papers used for local reference refreshes.
- `scripts/effective_diversity_audit.py`
  - Convenience wrapper that forwards to the rewritten baseline-vs-variant `dagzoo diversity-audit`; the packaged CLI is the canonical interface.
- `scripts/benchmark-suite.sh [suite] [preset] [out_dir] [diagnostics] [diagnostics_out_dir]`
  - Runs `dagzoo benchmark` with suite/preset selection and optional diagnostics.
- `scripts/benchmark-smoke.sh [preset] [diagnostics] [diagnostics_out_dir]`
  - Quick smoke benchmark for a single preset with optional diagnostics.
- `scripts/benchmark-gpu-validation.sh [out_dir]`
  - Delegates to the Python H100 validation runner, which records Torch CUDA visibility, runs the primary H100 smoke + standard benchmarks, adds a large-shape stress phase plus bounded saturation candidates, saves the standard-run baseline as an artifact, captures `nvidia-smi` telemetry for primary H100 phases, and executes the feature-smoke GPU benchmark matrix.
- `scripts/bump-version.sh <major|minor|patch> [--dry-run] [--tag]`
  - Bump the semver version in `pyproject.toml`. Use `--tag` to commit and create a git tag.
- `scripts/cleanup_local_artifacts.py [--group runtime|docs|all] [--apply]`
  - Dry-run or remove ignored local runtime/docs outputs (`data/`, `benchmarks/results/`, and the built docs output under `site/`) without touching tracked files.
- `scripts/docs/sync_hugo_content.py [--check]`
  - Sync canonical docs from `docs/` into the generated Hugo input area described in `site/README.md` (single-source docs model).
- `scripts/docs/check_links.py [roots...]`
  - Validate local Markdown/HTML links across source docs and generated site content.
- `scripts/docs/check_built_output_links.py [output_dir]`
  - Validate internal links in built Hugo output and enforce base-path-safe absolute links.

## Examples

```bash
./scripts/dev doctor all
./scripts/dev impact
./scripts/dev impact --source staged
./scripts/dev impact --files src/dagzoo/core/execution_semantics.py
./scripts/dev deps --write-docs
./scripts/dev contract --source staged
./scripts/dev verify quick
./scripts/dev verify code --incremental
./scripts/dev verify docs
./scripts/dev verify bench
./scripts/generate-default.sh
./scripts/generate-default.sh 50 cpu data/run_cpu_50
./scripts/generate-h100.sh 500 cuda data/run_h100_500 123
./scripts/generate-many-class.sh 25 cpu data/run_many_class 123
./scripts/generate-noise.sh gaussian 25 cpu data/run_noise_gaussian 123
./scripts/generate-noise.sh mixture 25 cpu data/run_noise_mixture 124
./scripts/generate-from-config.sh configs/benchmark_medium_cuda.yaml 100 cuda data/run_medium 42
./scripts/generate-smoke.sh configs/default.yaml 3 cpu
./scripts/generate-missingness.sh mcar 0.2 25 cpu data/run_missing_mcar 101
./scripts/generate-missingness.sh mar 0.25 25 cpu data/run_missing_mar 102
./scripts/fetch-additional-references.sh
./scripts/benchmark-smoke.sh cpu
./scripts/benchmark-smoke.sh cpu on benchmarks/results/smoke_diag
./scripts/benchmark-suite.sh standard all benchmarks/results/latest
./scripts/benchmark-suite.sh smoke cpu benchmarks/results/smoke_cpu_diag on
./scripts/benchmark-gpu-validation.sh
./scripts/benchmark-gpu-validation.sh benchmarks/results/gpu_h100_manual
./.venv/bin/python scripts/docs/sync_hugo_content.py
./.venv/bin/python scripts/docs/sync_hugo_content.py --check
./.venv/bin/python scripts/docs/check_links.py
./.venv/bin/python scripts/docs/check_built_output_links.py site/public
./.venv/bin/python scripts/cleanup_local_artifacts.py --group all
./.venv/bin/python scripts/cleanup_local_artifacts.py --group runtime --apply
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

`benchmark-suite.sh` with preset `all` includes CUDA presets and will hard-fail if CUDA is unavailable.

When diagnostics is enabled for benchmark scripts, coverage artifacts are written under:

- `<out_dir>/diagnostics/<sanitized_preset_key>_<hash>/coverage_summary.json`
- `<out_dir>/diagnostics/<sanitized_preset_key>_<hash>/coverage_summary.md`

The diagnostics preset directory is sanitized and hash-suffixed (for example, `cpu_ca49ca4b`) to keep paths unique and filesystem-safe.

Docs workflow note: the built Hugo output lives under `site/`. For the full
single-source docs/rendered-reference model, see
`docs/development/design-decisions.md` and `site/README.md`.

When missingness is enabled in benchmark configs, summary JSON includes
`preset_results[*].missingness_guardrails` and may escalate regression status via runtime or acceptance issues.

When non-gaussian noise is enabled in benchmark configs, summary JSON includes
`preset_results[*].noise_guardrails` and may escalate regression status via runtime or metadata validity issues.

The H100 validation runner writes a root manifest at
`<out_dir>/validation_manifest.json`. Primary H100 performance phases also write:

- `<phase_dir>/gpu_telemetry.csv`
- `<phase_dir>/gpu_telemetry_summary.json`

Use those alongside each phase `summary.json` to inspect generation timing,
write-stage timing, fixed-layout batch/chunking, and GPU memory/utilization
evidence without inferring bottlenecks from datasets/minute alone.
