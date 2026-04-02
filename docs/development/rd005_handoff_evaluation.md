# RD-005 Handoff Evaluation

`RD-005` should be promoted by downstream quality first, then diversity gain,
then throughput stability. The supported maintainer loop for that comparison is:

```bash
./.venv/bin/python scripts/evaluate_handoff_pareto.py \
  --baseline-config configs/default.yaml \
  --stress-profile anti_memorization_piecewise_classification_graph_breadth_slice_v1 \
  --stress-profile anti_memorization_piecewise_classification_compositional_slice_v1 \
  --out-root benchmarks/results/rd005_pareto \
  --num-datasets 8 \
  --seed 123 \
  --device cpu
```

## What It Does

- Runs `dagzoo generate --handoff-root` for one baseline plus one variant at a
  time under the same seed, dataset budget, device, and hardware policy.
- Scores each generated corpus with a lightweight downstream linear probe:
  ridge regression for regression tasks and one-vs-rest ridge classification
  for classification tasks.
- Reads `internal/diagnostics_artifacts/coverage_summary.json` when available
  and computes diversity-shift summaries against the baseline.
- Writes:
  - `pareto_summary.json`
  - `pareto_summary.md`

## Decision Rule

- Prefer variants with higher downstream mean score.
- Treat diversity-shift summaries as supporting evidence that the change is
  expanding relationship structure rather than only moving noise.
- Use datasets/minute as the third tie-breaker.
- Look at the reported Pareto frontier before promoting any slice into a
  benchmark/default workflow.

## Notes

- The script is repo workflow tooling, not a packaged `dagzoo` CLI command.
- `--stress-profile` variants are synthesized by applying the profile to the
  baseline config and writing a temporary config file under the evaluation root.
- `--reuse-existing` can be used to avoid regenerating already-materialized
  handoff roots.
