# Benchmark Suite Report

- Suite: `smoke`
- Generated at: `<TIMESTAMP>`
- Regression status: `warn`

## Presets
| Preset | Rows | Mode | Device | Backend | Datasets/min | Gen/min | Write/min | Filter/min | Filter Accepted/min | Repro | Workload | Filter Reject % (attempt) | Filter Accept % (dataset) | Filter Reject % (dataset) | Filter Retry % (dataset) | Elapsed (s) | Latency p95 (ms) | Peak RSS (MB) | Diagnostics | Filtering | Missingness | Shift | Noise | Throughput |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| cpu | 1024 | fixed_batched | cpu | cpu | <RATE> | <RATE> | <RATE> | - | - | match | mismatch | - | - | - | - | <SECONDS> | <MS> | <MB> | on | off | off | off | off | pass |

## Bottleneck Evidence

### cpu
- Preparation: `wall=<SECONDS>s`, `cpu=<SECONDS>s`, `cpu_busy_pct=<PERCENT>`
- Generation: `wall=<SECONDS>s`, `cpu=<SECONDS>s`, `cpu_busy_pct=<PERCENT>`
- Raw batch: `wall=<SECONDS>s`, `cpu=<SECONDS>s`, `node_apply_wall=<SECONDS>s`, `converter_wall=<SECONDS>s`, `feature_wall=<SECONDS>s`
- Fixed layout: `target_cells=1024`, `per_dataset_cells=32`, `batch=2`, `chunks=1`, `tail=0`
- Write replay: `sample_datasets=2`, `wall=<SECONDS>s`, `cpu=<SECONDS>s`, `bytes=4096`, `mib_per_s=<RATE>`
- Filter replay: `disabled`
- CUDA memory: `reserved_mb=<MB>`, `reserved_pct=<PERCENT>`, `headroom_mb=<MB>`

## Diagnostics Artifacts
| Preset | Coverage JSON | Coverage Markdown |
|---|---|---|
| cpu | `<ABSOLUTE_PATH>` | `<ABSOLUTE_PATH>` |

## Regression Issues
| Severity | Preset | Metric | Current | Baseline | Degradation % |
|---|---|---|---:|---:|---:|
| warn | cpu | datasets_per_minute | 100.000 | 120.000 | 16.67 |
