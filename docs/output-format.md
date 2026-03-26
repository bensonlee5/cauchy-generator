# Artifacts & API

Consumer-facing specification for generated data. This is a **contract
document** — downstream users can rely on the guarantees described here.

Public config references accepted by the main user-facing surfaces are:

- a YAML path
- a curated recipe reference in the form `recipe:<name>`

That contract applies to `dagzoo generate`, `dagzoo benchmark --preset custom`,
`dagzoo diversity-audit`, and `build_dataloader(...)`.

______________________________________________________________________

## DatasetBundle (in-memory)

Each generated dataset is returned as a `DatasetBundle` with these fields:

| Field           | Type                                | Shape                 |
| --------------- | ----------------------------------- | --------------------- |
| `X_train`       | `torch.Tensor` (float32 or float64) | (n_train, n_features) |
| `y_train`       | `torch.Tensor`                      | (n_train,)            |
| `X_test`        | `torch.Tensor` (float32 or float64) | (n_test, n_features)  |
| `y_test`        | `torch.Tensor`                      | (n_test,)             |
| `feature_types` | `list[str]`                         | length n_features     |
| `metadata`      | `dict[str, Any]`                    | —                     |

**Target dtype**: `int64` for classification, float for regression.

**Feature dtype**: matches the configured torch dtype (float32 or float64).

______________________________________________________________________

## Feature type encoding

Each entry in `feature_types` is one of:

- `"num"` — continuous feature. After postprocessing, values are clipped and
  standardized to approximately zero mean and unit variance.
- `"cat"` — categorical feature. Observed values are integer indices in the
  range `0 .. cardinality - 1`. When missingness is enabled, missing values are
  encoded as `NaN`.

`feature_types[i]` describes column index `i` in X_train and X_test.

______________________________________________________________________

## On-disk directory structure

```
out_dir/
  shard_00000/
    train.parquet
    test.parquet
    metadata.ndjson
    lineage/
      adjacency.bitpack.bin
      adjacency.index.json
  shard_00001/
    ...
```

**Shard naming**: `shard_{id:05d}` — five-digit zero-padded shard ID.
Default: 128 datasets per shard.

**Shard ID calculation**: `dataset_index // shard_size`.

______________________________________________________________________

## Generate handoff layout (`dagzoo generate --handoff-root`)

Generate handoff runs use the supplied handoff root as a stable downstream
entrypoint:

```
handoff_root/
  handoff_manifest.json
  generated/
    shard_00000/
    effective_config.yaml
    effective_config_trace.yaml
```

`generated/` reuses the same shard and metadata contracts documented on this
page. `handoff_manifest.json` is the downstream entrypoint for consumers such
as `tab-foundry`.

### Generate handoff manifest JSON

`handoff_manifest.json` uses this versioned top-level contract:

| Key                   | Type   | Description                                                                       |
| --------------------- | ------ | --------------------------------------------------------------------------------- |
| `schema_name`         | str    | Exact string `dagzoo_generate_handoff_manifest`                                   |
| `schema_version`      | int    | Exact integer `1`                                                                 |
| `identity`            | object | Stable generate-run and corpus ids plus source-family tag                         |
| `generate_invocation` | object | Config reference plus structured user-supplied CLI overrides                      |
| `artifacts`           | object | Absolute paths for the run root, generated output, and effective-config artifacts |
| `artifacts_relative`  | object | Manifest-relative artifact paths for portable downstream consumption              |
| `checksums`           | object | SHA-256 digests for effective-config artifacts                                    |
| `summary`             | object | Generated dataset count                                                           |
| `throughput`          | object | Generation-stage elapsed time plus datasets-per-minute context                    |
| `hardware`            | object | Requested/resolved device context plus applied hardware policy                    |
| `diversity_artifacts` | object | Nullable paths for handoff-associated diversity report artifacts                  |
| `defaults`            | object | Canonical downstream-consumption defaults                                         |

Current `identity` keys:

- `source_family`
- `generate_run_id`
- `generated_corpus_id`

`generate_invocation` contains:

- `config_path`
- `overrides`

`generate_invocation.config_path` records either the literal `recipe:<name>`
reference used on the CLI or the resolved YAML path for path-based runs.

Current `generate_invocation.overrides` keys:

- `num_datasets`
- `seed`
- `rows`
- `device`
- `hardware_policy`
- `missing_rate`
- `missing_mechanism`
- `missing_mar_observed_fraction`
- `missing_mar_logit_scale`
- `missing_mnar_logit_scale`
- `diagnostics`
- `diagnostics_out_dir`
- `handoff_root`

Current `artifacts` keys:

- `run_root`
- `generated_dir`
- `effective_config_path`
- `effective_config_trace_path`

Current `artifacts_relative` keys:

- `run_root`
- `generated_dir`
- `effective_config_path`
- `effective_config_trace_path`

Current `checksums` keys:

- `effective_config_sha256`
- `effective_config_trace_sha256`

Current `diversity_artifacts` keys:

- `summary_json_path`
- `summary_md_path`

Current `defaults` keys:

- `recommended_training_corpus`
- `recommended_training_artifact_key`
- `curation_policy`

`throughput.generation_stage` reports wall-clock timing from
`dagzoo generate --handoff-root`.

`dagzoo generate --handoff-root` does not run a diversity audit automatically,
so the `diversity_artifacts` values are currently `null` unless a separate
workflow persists handoff-associated diversity outputs alongside the run.

Downstream consumers should prefer `artifacts_relative` over absolute-path
`artifacts` when portability matters, and should treat
`defaults.recommended_training_corpus=generated` as the canonical training
target until a separate deferred-filter stage materializes a curated corpus.

______________________________________________________________________

## Parquet column schema

Shard-level `train.parquet` and `test.parquet` both use packed row-wise
records:

| Column          | Type                  | Description                                |
| --------------- | --------------------- | ------------------------------------------ |
| `dataset_index` | int64                 | Global dataset index for this row          |
| `row_index`     | int64                 | Row index within the dataset split         |
| `x`             | list[float32/float64] | Full feature vector for this row           |
| `y`             | int64 or float        | Target value for this row (task-dependent) |

**Compression**: zstd (default).

Feature typing metadata remains per-dataset in `metadata.ndjson` records.

______________________________________________________________________

## Metadata NDJSON structure

Each shard writes one `metadata.ndjson` file with one JSON record per dataset.
Each line contains:

| Key             | Type      | Description                                  |
| --------------- | --------- | -------------------------------------------- |
| `dataset_index` | int       | Global dataset index                         |
| `n_train`       | int       | Train row count for the dataset              |
| `n_test`        | int       | Test row count for the dataset               |
| `n_features`    | int       | Feature count for the dataset                |
| `feature_types` | list[str] | Per-feature type annotations (`num`/`cat`)   |
| `metadata`      | object    | The dataset metadata payload described below |

`metadata` contains the dataset-level generation metadata described below.

### Top-level keys

| Key                          | Type        | Description                                                                                           |
| ---------------------------- | ----------- | ----------------------------------------------------------------------------------------------------- |
| `backend`                    | str         | Always `"torch"`                                                                                      |
| `device`                     | str         | Compute device (e.g., `"cpu"`, `"cuda"`)                                                              |
| `requested_device`           | str         | Requested runtime device after CLI/config normalization (for example `auto`, `cpu`, `cuda`, `mps`)    |
| `resolved_device`            | str         | Runtime backend selected from the requested device for generation                                     |
| `device_fallback_reason`     | str or null | Reserved field retained for artifact-contract stability; currently always `null`                      |
| `compute_backend`            | str         | Implementation variant identifier                                                                     |
| `n_features`                 | int         | Number of features                                                                                    |
| `n_categorical_features`     | int         | Number of categorical features                                                                        |
| `n_classes`                  | int or null | Realized class count in emitted labels (null for regression)                                          |
| `graph_nodes`                | int         | Number of nodes in the DAG                                                                            |
| `graph_edges`                | int         | Number of edges in the DAG                                                                            |
| `graph_depth_nodes`          | int         | Longest path length in the DAG                                                                        |
| `graph_edge_density`         | float       | Edge count / max possible edges                                                                       |
| `seed`                       | int         | Replay seed recorded by the emitting API. Canonical generation stores the shared run seed here.       |
| `dataset_seed`               | int         | Optional canonical per-dataset child seed derived from `seed`; used for deferred replay/diagnostics   |
| `dataset_index`              | int         | Optional canonical dataset position within the run (0-based)                                          |
| `dataset_id`                 | str         | Optional stable canonical dataset identifier derived from run/layout grouping plus dataset provenance |
| `run_num_datasets`           | int         | Optional canonical run length used to replay the saved bundle                                         |
| `attempt_used`               | int         | Generation attempt index (0-based)                                                                    |
| `lineage`                    | object      | DAG lineage record (see Lineage below)                                                                |
| `shift`                      | object      | Resolved shift settings and realized observability signals                                            |
| `noise_distribution`         | object      | Resolved noise-family selection and effective sampling params                                         |
| `config`                     | object      | Full serialized generator configuration                                                               |
| `filter`                     | object      | Filter results (see below)                                                                            |
| `class_structure`            | object      | Present only for classification (see below)                                                           |
| `missingness`                | object      | Present only when missingness is enabled                                                              |
| `layout_mode`                | str         | Optional canonical layout metadata (`"fixed"` for canonical generation outputs)                       |
| `layout_plan_seed`           | int         | Optional internal seed used to sample the shared per-run layout                                       |
| `layout_signature`           | str         | Optional deterministic fingerprint for the shared sampled layout                                      |
| `layout_plan_signature`      | str         | Optional deterministic fingerprint for the internal frozen node execution payload                     |
| `layout_plan_schema_version` | int         | Optional internal metadata version for the canonical shared-layout payload                            |
| `layout_execution_contract`  | str         | Optional internal execution contract identifier for canonical determinism                             |
| `split_groups`               | object      | Optional stable canonical grouping keys for downstream split assignment                               |
| `keyed_replay`               | object      | Optional exact keyed subtree replay paths for canonical layout, execution, and dataset roots          |
| `mechanism_families`         | object      | Optional realized mechanism-family and mechanism-variant coverage for canonical fixed-layout runs     |

For canonical generation (`generate_one`, `generate_batch`, `generate_batch_iter`,
and `dagzoo generate`), replay later bundles with the shared `seed`,
`run_num_datasets`, and `dataset_index` by regenerating the canonical batch and
selecting that index. `dataset_seed` preserves the per-bundle child seed for
deferred replay and diagnostics. `dataset_id` is stable across copies of the
same generated corpus because it is derived from canonical run provenance,
layout-plan provenance, and dataset provenance rather than filesystem
location. `split_groups.request_run` includes stable non-plan run provenance
that can change emitted data, while `split_groups.layout_plan` intentionally
remains the narrower fixed-layout execution-plan grouping key. Exact keyed
subtree replay uses `seed` together with the `keyed_replay` paths.

For downstream train/validation/test assignment, use
`split_groups.request_run` or `split_groups.layout_plan` instead of
reconstructing groups from raw layout metadata. `request_run` keeps all
datasets from one canonical run together; `layout_plan` keeps all datasets that
share the same fixed-layout execution plan together.

### `split_groups` sub-object

Present for canonical generation outputs.

| Key           | Type | Description                                                                  |
| ------------- | ---- | ---------------------------------------------------------------------------- |
| `request_run` | str  | Stable grouping key for one canonical run, including non-plan run provenance |
| `layout_plan` | str  | Stable grouping key for datasets sharing one fixed-layout execution plan     |

### `keyed_replay` sub-object

Present for canonical generation outputs. These paths are interpreted relative
to `KeyedRng(metadata["seed"])`:

| Key                        | Type             | Description                                                 |
| -------------------------- | ---------------- | ----------------------------------------------------------- |
| `layout_root_path`         | list[str \| int] | Exact keyed path for replaying the shared per-run layout    |
| `execution_plan_root_path` | list[str \| int] | Exact keyed path for replaying the shared execution subtree |
| `dataset_root_path`        | list[str \| int] | Exact keyed path for replaying one bundle’s dataset subtree |

### `mechanism_families` sub-object

Present for canonical fixed-layout generation outputs.

| Key                      | Type      | Description                                                              |
| ------------------------ | --------- | ------------------------------------------------------------------------ |
| `sampled_family_counts`  | object    | Realized per-family function-plan counts for the sampled execution plan  |
| `families_present`       | list[str] | Sorted family labels with non-zero realized count                        |
| `sampled_variant_counts` | object    | Realized per-variant counts for internal widened families such as `gp.*` |
| `variants_present`       | list[str] | Sorted variant labels with non-zero realized count                       |
| `total_function_plans`   | int       | Total realized function-plan count across the execution plan             |

Current variant labels are additive observability fields, not separate public
family config names. Today they are emitted for widened `gp` execution as
`gp.standard`, `gp.periodic`, and `gp.multiscale`.

Coverage and diversity-audit summary artifacts reuse these counts inside
`mechanism_family_summary` and add dataset-level presence-rate fields:

- `dataset_presence_rate_by_family`
- `dataset_presence_rate_by_variant`

### Shift sub-object

Present for all generated bundles. When shift is disabled, scales are `0.0` and
multipliers are `1.0`.

| Key                         | Type  | Description                                                 |
| --------------------------- | ----- | ----------------------------------------------------------- |
| `enabled`                   | bool  | Whether shift controls were enabled                         |
| `mode`                      | str   | Resolved shift mode (`off`, `graph_drift`, etc.)            |
| `graph_scale`               | float | Resolved graph drift scale                                  |
| `mechanism_scale`           | float | Resolved mechanism drift scale                              |
| `variance_scale`            | float | Resolved noise drift scale                                  |
| `edge_logit_bias_shift`     | float | Additive shift applied to edge logits                       |
| `mechanism_logit_tilt`      | float | Mechanism-family tilt applied at sampling                   |
| `variance_sigma_multiplier` | float | Sigma multiplier applied to stochastic noise                |
| `edge_odds_multiplier`      | float | Edge-odds multiplier (`exp(edge_logit_bias_shift)`)         |
| `noise_variance_multiplier` | float | Noise-variance multiplier (`variance_sigma_multiplier^2`)   |
| `mechanism_nonlinear_mass`  | float | Probability mass on nonlinear mechanism families (`[0, 1]`) |

### Noise Distribution sub-object

Present for all generated bundles.

| Key                 | Type           | Description                                                             |
| ------------------- | -------------- | ----------------------------------------------------------------------- |
| `family_requested`  | str            | Configured noise family (`gaussian`, `laplace`, `student_t`, `mixture`) |
| `family_sampled`    | str            | Effective family used by the dataset generation runtime                 |
| `sampling_strategy` | str            | Runtime selection strategy (`dataset_level`)                            |
| `base_scale`        | float          | Base noise scale from config                                            |
| `student_t_df`      | float          | Student-t degrees of freedom parameter used by the runtime              |
| `mixture_weights`   | object or null | Effective normalized mixture weights when `family_requested=mixture`    |

### Filter sub-object

Generated bundles guarantee only the deferred-filter trace fields below.

| Key      | Type | Description                                                     |
| -------- | ---- | --------------------------------------------------------------- |
| `mode`   | str  | Filter execution mode. Current value is `deferred`.             |
| `status` | str  | `not_run` on freshly generated outputs before replay filtering. |

When `dagzoo filter` rewrites accepted/rejected metadata, it adds replay
telemetry such as `enabled`, `accepted`, `backend`, `filter_mode`,
`skill_small`, `skill_full`, `skill_gain`, `skill_small_lb95`,
`skill_gain_ub95`, `skill_full_ub95`, `stump_skill`,
`lineage_veto_applied`, and `reason`. The same small-shot-ease fields also
appear in deferred-filter summaries and benchmark filter summaries.

### Class Structure sub-object (classification only)

Present only for classification datasets.

| Key                      | Type        | Description                                        |
| ------------------------ | ----------- | -------------------------------------------------- |
| `n_classes_sampled`      | int         | Layout-sampled class count before postprocessing   |
| `n_classes_realized`     | int         | Unique class count in emitted `y_train` + `y_test` |
| `labels_contiguous`      | bool        | Whether labels form contiguous range `0..K-1`      |
| `train_test_class_match` | bool        | Whether train and test class sets are identical    |
| `min_label`              | int or null | Minimum emitted class label                        |
| `max_label`              | int or null | Maximum emitted class label                        |

### Fixed-layout metadata

Present for all canonical generation outputs. These bundles share one sampled
layout per run and preserve emitted column alignment (feature count, column
order, and lineage feature-to-node mapping) within that run.

| Key                          | Type   | Description                                                            |
| ---------------------------- | ------ | ---------------------------------------------------------------------- |
| `layout_mode`                | str    | `"fixed"`                                                              |
| `layout_plan_seed`           | int    | Internal seed used to sample the shared per-run layout                 |
| `layout_signature`           | str    | Stable fingerprint for the shared sampled layout                       |
| `layout_plan_signature`      | str    | Stable fingerprint for the frozen internal node payload                |
| `layout_plan_schema_version` | int    | Internal canonical layout metadata version                             |
| `layout_execution_contract`  | str    | Internal execution contract (`chunk_batched_v1`)                       |
| `dataset_id`                 | str    | Stable dataset identifier derived from canonical run/layout provenance |
| `split_groups`               | object | Stable downstream grouping keys for request-run and layout-plan splits |
| `keyed_replay`               | object | Exact keyed subtree replay paths for layout/execution/dataset roots    |

Under `chunk_batched_v1`, canonical fixed-layout outputs are deterministic for
the same run seed and realized run shape. Internal plan metadata records the
shared sampled layout and execution-plan fingerprint used for that run, while
`dataset_id` and `split_groups` provide the public path-independent grouping
surface for downstream consumers: `request_run` includes non-plan run
provenance, while `layout_plan` stays scoped to the shared fixed-layout
execution plan. `keyed_replay` records the exact keyed subtree roots needed
for internal replay.

### Missingness sub-object (optional)

Present only when missingness is enabled.

| Key                     | Type  | Description                            |
| ----------------------- | ----- | -------------------------------------- |
| `enabled`               | bool  | Always `true` when present             |
| `mechanism`             | str   | `"mcar"`, `"mar"`, or `"mnar"`         |
| `target_rate`           | float | Configured missing rate                |
| `realized_rate_train`   | float | Actual missing fraction in train split |
| `realized_rate_test`    | float | Actual missing fraction in test split  |
| `realized_rate_overall` | float | Actual missing fraction overall        |
| `missing_count_train`   | int   | Number of missing cells in train       |
| `missing_count_test`    | int   | Number of missing cells in test        |
| `missing_count_overall` | int   | Total missing cells                    |

______________________________________________________________________

## Diagnostics coverage summary artifacts

When diagnostics are enabled, the run root also includes:

- `coverage_summary.json`
- `coverage_summary.md`

These artifacts summarize corpus-level coverage and do not alter the per-dataset
`metadata.ndjson` contract described above.

### `coverage_summary.json` top-level keys

Current top-level keys are:

- `generated_at`
- `num_datasets`
- `task_counts`
- `histogram_bins`
- `quantiles`
- `max_values_per_metric`
- `mechanism_family_summary`
- `steering`
- `metrics`

### `steering` sub-object

Present in `coverage_summary.json`. Always emitted; when steering is disabled it
reports `enabled=false` and an empty `stages` list.

| Key                 | Type         | Description                                                             |
| ------------------- | ------------ | ----------------------------------------------------------------------- |
| `enabled`           | bool         | Whether steering was enabled for the run                                |
| `authoring_form`    | str          | `disabled`, `preset`, or `explicit_stages`                              |
| `preset`            | str or null  | Steering preset name when present                                       |
| `stage_count`       | int          | Number of normalized steering stages                                    |
| `resolution_checks` | object       | Requested-versus-emitted consistency counters for the generated bundles |
| `stages`            | list[object] | Per-stage requested authoring plus realized steering and metric summary |

Each `stages[*]` entry contains:

- `index`, `name`, `fraction`
- `requested`
- `dataset_count`
- `dataset_index_range`
- `progress_range`
- `requested_effective`
- `realized`
- `metrics`

`requested` preserves the normalized authored stage payload. `requested_effective`
summarizes the effective per-dataset steering resolution for bundles assigned to
that stage, while `realized` summarizes the emitted missingness, shift, and
noise metadata actually observed in those bundles. `metrics` reuses the same
coverage-metric summary shape as the top-level `metrics` section, scoped to one
stage.

`coverage_summary.md` renders the same steering analysis as a condensed
human-readable section and does not introduce additional contract fields beyond
the JSON artifact described here.

______________________________________________________________________

## Lineage schema

Schema name: `dagzoo.dag_lineage`

### Version 1.0.0 (dense, in-memory)

Used in the in-memory metadata `lineage` field during generation. When
lineage is persisted to disk, payloads are rewritten to compact version
`1.1.0`.

```json
{
  "schema_name": "dagzoo.dag_lineage",
  "schema_version": "1.0.0",
  "graph": {
    "n_nodes": 8,
    "adjacency": [[0, 1, 0, ...], ...]
  },
  "assignments": {
    "feature_to_node": [2, 3, 5, 7],
    "target_to_node": 7
  }
}
```

- `adjacency` is an n_nodes x n_nodes list of lists. Entries are 0 or 1.
  Upper-triangular only; diagonal is always 0. Direction convention is
  `adjacency[src][dst]` (`src -> dst`), so parents of node `j` are found from column `j`.
- `feature_to_node[i]` is the DAG node index that produces feature `i`.
- `target_to_node` is the DAG node index that produces the target.

### Version 1.1.0 (compact, on-disk)

Used in `metadata.ndjson` dataset records when lineage artifacts are written to disk.
Replaces the dense adjacency matrix with a reference to bitpacked binary
data.

```json
{
  "schema_name": "dagzoo.dag_lineage",
  "schema_version": "1.1.0",
  "graph": {
    "n_nodes": 8,
    "edge_count": 12,
    "adjacency_ref": {
      "encoding": "upper_triangle_bitpack_v1",
      "blob_path": "lineage/adjacency.bitpack.bin",
      "index_path": "lineage/adjacency.index.json",
      "dataset_index": 0,
      "bit_offset": 0,
      "bit_length": 28,
      "sha256": "a1b2c3..."
    }
  },
  "assignments": {
    "feature_to_node": [2, 3, 5, 7],
    "target_to_node": 7
  }
}
```

### Adjacency encoding: `upper_triangle_bitpack_v1`

- Packs the `n_nodes * (n_nodes - 1) / 2` upper-triangle bits into bytes.
- Bit order: little-endian.
- `bit_offset` and `bit_length` locate this dataset's bits within the
  shared shard-level blob file.
- `sha256` is a hex-encoded SHA-256 checksum of the packed bytes for this
  dataset's adjacency data.

### Lineage index file

Each shard contains `lineage/adjacency.index.json` with:

- `schema_name` and `schema_version` — echo the lineage schema identifiers.
- `encoding` — always `"upper_triangle_bitpack_v1"`.
- `records` — array of per-dataset offset/length/checksum entries.

______________________________________________________________________

## Contract guarantees

**Determinism** — seed derivation is deterministic. For fixed seed and
configuration, runs are expected to reproduce metadata and numerical outputs
within tolerance. Strict byte-identical tensors/files are not guaranteed
across all backends.

**Feature alignment** — `feature_types[i]` in each metadata record describes
feature index `i` inside packed `x` row vectors and tensor column index `i`
in `X_train` / `X_test`.

**Lineage integrity** — each dataset's bitpacked adjacency data is
protected by a SHA-256 checksum recorded in the metadata.

**Postprocessing invariants**:

- Canonical generation (`generate_one`, `generate_batch`, `generate_batch_iter`)
  is fixed-layout-backed and preserves emitted feature schema across the run:
  constant-column removal and feature-column permutation are disabled.
- Numeric features are clipped and standardized (approximately zero mean,
  unit variance).
- Classification target classes are randomly permuted (label indices carry
  no ordinal meaning).
