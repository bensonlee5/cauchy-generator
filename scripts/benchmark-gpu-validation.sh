#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-benchmarks/results/gpu_h100_validation_$(date +%Y%m%d_%H%M%S)}"

if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python -m dagzoo.bench.h100_validation --out-root "${OUT_ROOT}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: neither .venv/bin/python nor uv is available" >&2
  exit 1
fi

exec uv run python -m dagzoo.bench.h100_validation --out-root "${OUT_ROOT}"
