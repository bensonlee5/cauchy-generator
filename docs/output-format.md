# Artifacts & API

Consumer-facing specification for generated data. This is a contract document:
downstream users can rely on the guarantees described here.

Public config references accepted by the main user-facing surfaces are:

- a YAML path
- a curated recipe reference in the form `recipe:<name>`

That contract applies to `dagzoo generate`, `dagzoo benchmark --preset custom`,
`dagzoo diversity-audit`, and `build_dataloader(...)`.

This page is the readable overview. The exhaustive field-by-field catalog lives
in [export-contract-fields.md](export-contract-fields.md) and is generated from
`reference/export_contract_inventory.yaml`.

______________________________________________________________________

## DatasetBundle (in-memory)

Each generated dataset is returned as a `DatasetBundle` with these fields:

| Field           | Type                                | Shape                   |
| --------------- | ----------------------------------- | ----------------------- |
| `X_train`       | `torch.Tensor` (float32 or float64) | `(n_train, n_features)` |
| `y_train`       | `torch.Tensor`                      | `(n_train,)`            |
| `X_test`        | `torch.Tensor` (float32 or float64) | `(n_test, n_features)`  |
| `y_test`        | `torch.Tensor`                      | `(n_test,)`             |
| `feature_types` | `list[str]`                         | length `n_features`     |
| `metadata`      | `dict[str, Any]`                    | —                       |

Target dtype is `int64` for classification and floating-point for regression.
Feature dtype matches the configured torch dtype.

### `metadata` overview

`DatasetBundle.metadata` is the stable in-process metadata payload. Common
top-level keys include:

- runtime and identity fields such as `device`, `requested_device`,
  `resolved_device`, `dataset_index`, `dataset_id`, `dataset_seed`, and
  `run_num_datasets`
- semantic summaries such as `prior`, `lineage`, `shift`, `intervention`,
  `noise_distribution`, `generation_attempts`, and `filter`
- optional task/runtime summaries such as `class_structure`, `missingness`,
  `split_groups`, `keyed_replay`, and `mechanism_families`
- the resolved generator `config` snapshot

The exhaustive recursive contract for `metadata.*` lives in
[export-contract-fields.md](export-contract-fields.md).

Observational bundles omit `metadata.intervention`. Hard-interventional bundles
add only the summary object `{mode, signature}` at the top level; the full
authored selector payload remains in `effective_config.yaml` rather than
`metadata.config`.

### `metadata.prior` sub-object

Present for all generated bundles.

| Key                              | Type | Description                                 |
| -------------------------------- | ---- | ------------------------------------------- |
| `target_derivation`              | str  | Current target-construction contract marker |
| `feature_generator`              | str  | Feature-generation family                   |
| `missingness_stage`              | str  | Stage at which missingness is applied       |
| `classification_validity_policy` | str  | Classification retry policy                 |
| `localization_mode`              | str  | Current localization setting                |
| `n_adaptation`                   | str  | Current `n`-adaptation setting              |

Current emitted bundles use:

- `target_derivation = "tabiclv2_latent_node"`
- `feature_generator = "latent_dag"`
- `missingness_stage = "post_target_observation"`

That means `y` is emitted by converting one selected latent DAG node. There is
no separate observed-feature target mechanism and no soft-label export surface
in the current public contract.

______________________________________________________________________

## Feature type encoding

Each entry in `feature_types` is one of:

- `"num"`: continuous feature. After postprocessing, `X_train` values are
  clipped and standardized to approximately zero mean and unit variance, and
  the same train-fit transform is applied to `X_test`.
- `"cat"`: categorical feature. Observed values are integer indices in the
  range `0 .. cardinality - 1`. When missingness is enabled, missing values are
  encoded as `NaN`.

`feature_types[i]` describes column index `i` in `X_train` and `X_test`.

______________________________________________________________________

## On-Disk Directory Structure

Plain `dagzoo generate --out ...` runs write:

```text
out_dir/
  effective_config.yaml
  effective_config_trace.yaml
  shard_00000/
    train.parquet
    test.parquet
    dataset_catalog.ndjson
  shard_00001/
    ...
  internal/
    shard_00000/
      replay_catalog.ndjson
      lineage/
        adjacency.bitpack.bin
        adjacency.index.json
    shard_00001/
      ...
```

`shard_*` directories are the stable public dataset artifacts. `internal/`
holds dagzoo-only replay and lineage sidecars used by tooling such as
`dagzoo filter`; it is not the stable public contract.

Shard naming is `shard_{id:05d}`. Default shard size is `128` datasets, so the
shard id is `dataset_index // shard_size`.

______________________________________________________________________

## Parquet Column Schema

`train.parquet` and `test.parquet` both use packed row-wise records:

| Column          | Type                  | Description                        |
| --------------- | --------------------- | ---------------------------------- |
| `dataset_index` | int64                 | Global dataset index for this row  |
| `row_index`     | int64                 | Row index within the dataset split |
| `x`             | list[float32/float64] | Full feature vector for this row   |
| `y`             | int64 or float        | Target value for this row          |

Compression is `zstd` by default.

______________________________________________________________________

## Dataset Catalog NDJSON

Each public shard writes one `dataset_catalog.ndjson` file with one JSON record
per dataset. Current record keys are:

| Key                 | Type              | Description                                             |
| ------------------- | ----------------- | ------------------------------------------------------- |
| `dataset_index`     | int               | Global dataset index                                    |
| `dataset_id`        | str               | Stable dataset identifier                               |
| `task`              | str               | `classification` or `regression`                        |
| `n_train`           | int               | Train row count                                         |
| `n_test`            | int               | Test row count                                          |
| `n_features`        | int               | Emitted feature count                                   |
| `feature_types`     | list[str]         | Per-feature type annotations                            |
| `n_classes`         | int or null       | Realized emitted class count (`null` for regression)    |
| `group_ids`         | object (optional) | Stable downstream grouping keys                         |
| `intervention`      | object (optional) | Summary-only intervention regime metadata               |
| `target_derivation` | str (optional)    | Current target-construction marker                      |
| `target_relevance`  | object (optional) | Summary of which emitted features reach the target node |

### `group_ids` sub-object

Present when public grouping ids are available.

| Key           | Type | Description                                                    |
| ------------- | ---- | -------------------------------------------------------------- |
| `request_run` | str  | Stable grouping key for one requested public run               |
| `cohort`      | str  | Stable grouping key for heterogeneous raw-generation cohorts   |
| `layout_plan` | str  | Stable grouping key for datasets sharing one fixed-layout plan |

### `intervention` sub-object

Present when hard-intervention metadata is available.
Observational runs omit this field entirely.

| Key         | Type | Description                             |
| ----------- | ---- | --------------------------------------- |
| `mode`      | str  | Emitted intervention regime             |
| `signature` | str  | Stable summary intervention identifier  |

### `target_relevance` sub-object

Present when lineage target-relevance metadata is available.

| Key                | Type  | Description                                                           |
| ------------------ | ----- | --------------------------------------------------------------------- |
| `feature_count`    | int   | Number of emitted features whose latent node reaches `target_to_node` |
| `feature_fraction` | float | `feature_count / n_features`                                          |

______________________________________________________________________

## Generate Handoff Layout (`dagzoo generate --handoff-root`)

Generate handoff runs use the supplied handoff root as a stable downstream
entrypoint:

```text
handoff_root/
  handoff_manifest.json
  generated/
    shard_00000/
      train.parquet
      test.parquet
      dataset_catalog.ndjson
  internal/
    effective_config.yaml
    effective_config_trace.yaml
    shard_00000/
      replay_catalog.ndjson
      lineage/
        adjacency.bitpack.bin
        adjacency.index.json
  curated/
    ...  # optional, written later by dagzoo filter
```

`generated/` reuses the same public shard contract described above.
`internal/` remains dagzoo-only. `replay_catalog.ndjson` stores the full
per-dataset metadata payload, including the same summary-only `intervention`
object when present.

### `handoff_manifest.json`

`handoff_manifest.json` uses this versioned top-level contract:

| Key                  | Type   | Description                                                                          |
| -------------------- | ------ | ------------------------------------------------------------------------------------ |
| `schema_name`        | str    | Exact string `dagzoo_generate_handoff_manifest`                                      |
| `schema_version`     | int    | Exact integer `5`                                                                    |
| `identity`           | object | Stable generate-run and corpus ids plus source-family tag                            |
| `artifacts_relative` | object | Manifest-relative artifact paths for portable downstream consumption                 |
| `summary`            | object | Generated dataset count                                                              |
| `provenance`         | object | Optional generated-corpus provenance summary derived from the public dataset catalog |

Current `identity` keys:

- `source_family`
- `generate_run_id`
- `generated_corpus_id`

Current `identity.source_family` values:

- `dagzoo.heterogeneous_scm`
- `dagzoo.fixed_layout_scm`

Current `artifacts_relative` keys:

- `generated_dir`
- `curated_dir` (optional; present only after a curated corpus exists)

Current `summary` keys:

- `generated_datasets`

Current `provenance` keys:

- `intervention` (optional)
- `target_derivation`
- `target_relevant_feature_count_range`
- `target_relevant_feature_fraction_range`

Current `provenance.intervention` keys:

- `mode`
- `signature`

`provenance.intervention` is omitted for observational generated corpora.

______________________________________________________________________

## Lineage Schema

Schema name: `dagzoo.dag_lineage`

Older lineage payloads that used target-head or target-parent assignment fields
are intentionally unsupported. The current contract is:

### Version 1.4.0 (dense, in-memory)

Used in `DatasetBundle.metadata["lineage"]` during generation.

```json
{
  "schema_name": "dagzoo.dag_lineage",
  "schema_version": "1.4.0",
  "graph": {
    "n_nodes": 8,
    "adjacency": [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
  },
  "assignments": {
    "feature_to_node": [2, 3, 5, 7],
    "target_to_node": 6,
    "target_relevant_features": [0, 1, 3],
    "target_relevant_feature_count": 3,
    "target_relevant_feature_fraction": 0.75
  }
}
```

### Version 1.5.0 (compact, on-disk)

Used in persisted replay metadata when lineage artifacts are written to disk.

```json
{
  "schema_name": "dagzoo.dag_lineage",
  "schema_version": "1.5.0",
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
    "target_to_node": 6,
    "target_relevant_features": [0, 1, 3],
    "target_relevant_feature_count": 3,
    "target_relevant_feature_fraction": 0.75
  }
}
```

### Adjacency encoding: `upper_triangle_bitpack_v1`

- Packs the `n_nodes * (n_nodes - 1) / 2` upper-triangle bits into bytes.
- Bit order is little-endian.
- `bit_offset` and `bit_length` locate one dataset's adjacency bits inside the
  shared shard-level blob.
- `sha256` is a hex-encoded SHA-256 checksum of the packed bytes for that
  dataset's adjacency data.

Each shard also contains `lineage/adjacency.index.json` with schema identifiers,
the encoding name, and the per-dataset offset/length/checksum entries. Those
artifacts live under the corresponding internal shard directory.

______________________________________________________________________

## Diagnostics Coverage Summary Artifacts

When diagnostics are enabled, the run root also includes:

- `coverage_summary.json`
- `coverage_summary.md`

These artifacts summarize corpus-level coverage and do not alter the public
parquet or `dataset_catalog.ndjson` contract.

The exhaustive field list for diagnostics summaries lives in
[export-contract-fields.md](export-contract-fields.md).

______________________________________________________________________

## Contract Guarantees

**Determinism**: seed derivation is deterministic. For a fixed seed and
configuration, runs are expected to reproduce metadata and numerical outputs
within tolerance. Strict byte-identical tensors/files are not guaranteed across
all backends.

**Feature alignment**: `feature_types[i]` describes feature index `i` inside
packed parquet row vectors and tensor column index `i` in `X_train` /
`X_test`.

**Target semantics**: the current public contract derives `y` from one selected
latent DAG node and applies missingness afterward as an observation process
over emitted features.

**Lineage integrity**: each dataset's bitpacked adjacency data is protected by
a SHA-256 checksum recorded in the compact lineage payload.

**Postprocessing invariants**:

- Default public generation may vary emitted feature schema across one run.
- Stratified mode (`runtime.layout_mode: stratified`) still preserves
  heterogeneous semantics; constant-column removal and feature-column
  permutation remain dataset-local even when compatible strata are batched.
- Numeric features are clipped and standardized using statistics fit on the
  emitted training split, then applied unchanged to the test split.
- Regression targets are clipped and standardized using statistics fit on the
  emitted training split, then applied unchanged to the test split.
- Classification target classes are randomly permuted; label indices carry no
  ordinal meaning.
