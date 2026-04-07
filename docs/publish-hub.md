# Publish to Hugging Face Hub

Use this workflow when you want to share a generated corpus through a public or
private Hugging Face dataset repository.

`dagzoo` publishes from a handoff root, not directly from a plain `--out`
directory. The handoff root is the stable downstream surface for portable
corpora: it keeps the public shard artifacts under `generated/`, adds
`handoff_manifest.json`, and leaves dagzoo-only sidecars under `internal/`.

______________________________________________________________________

## 1. Generate a handoff root

```bash
dagzoo generate \
  --config recipe:default-baseline \
  --num-datasets 25 \
  --handoff-root handoffs/default_baseline
```

That writes a portable corpus layout:

```text
handoffs/default_baseline/
  handoff_manifest.json
  generated/
    shard_00000/
      train.parquet
      test.parquet
      dataset_catalog.ndjson
  internal/
    effective_config.yaml
    effective_config_trace.yaml
    run_context.json
```

If you later run `dagzoo filter` and create a curated accepted-only corpus under
the same handoff root, `dagzoo publish hub` will upload that `curated/`
directory too.

______________________________________________________________________

## 2. Authenticate with Hugging Face

Use the standard Hugging Face authentication flow:

```bash
hf auth login
```

Or set `HF_TOKEN` in the environment before running `dagzoo publish hub`.

______________________________________________________________________

## 3. Publish the corpus

```bash
dagzoo publish hub \
  --handoff-root handoffs/default_baseline \
  --repo-id your-name/default-baseline-corpus
```

Optional flags:

- `--private`: create the dataset repo as private if it does not already exist
- `--license <hf-license-id>`: stamp the dataset card with a Hugging Face
  license identifier such as `apache-2.0`

Successful publishes print the dataset URL:

```text
Published dataset repo: https://huggingface.co/datasets/your-name/default-baseline-corpus
```

______________________________________________________________________

## 4. What gets uploaded

`dagzoo publish hub` uploads only the public handoff artifacts:

- `generated/`
- `curated/` when present
- `handoff_manifest.json`
- a generated root `README.md` dataset card

`internal/` is never uploaded. Those files stay local for dagzoo tooling and
reproducibility.

______________________________________________________________________

## 5. Republishing

You can rerun `dagzoo publish hub` against the same dataset repo after
regenerating the corpus or adding a curated corpus. Republishing replaces only
dagzoo-managed paths:

- `generated/**`
- `curated/**`
- `handoff_manifest.json`
- `README.md`

Other files in the remote dataset repo are left alone.
