# AST-Based-Agent-Skills Retriever Training

This repo supports a 3-step retriever workflow:

1. Build Chroma indexes for `summary` + `description`.
2. Mine negatives parquet using rank window **top-37 minus top-5** (ranks 6..37).
3. Train from the mined negatives parquet.

## Prerequisites

```bash
uv sync
```

Optional for training metrics:

```bash
uv run wandb login
```

## Workflow

### Step 1: Build Chroma (summary + description)

```bash
make build-retriever-chroma
```

Defaults:
- source JSONL: `artifacts/summary_retriever_models.jsonl`
- output Chroma root: `artifacts/chroma`
- fields: `summary,description`

### Step 2: Mine negatives parquet (top-37 minus top-5)

```bash
make build-mined-negatives-parquet
```

This uses:
- `top_k=37`
- `window_start_rank=6`
- `window_end_rank=37`

Output files:
- `artifacts/retriever_training/train.parquet`
- `artifacts/retriever_training/validation.parquet`

### Step 3: Train from mined parquet

```bash
make retriever-train
```

By default this reads `configs/train.config.yaml`.

## Training config

The training config is split into sections:

- `input.mined_parquet_path`
- `model.name`
- `training.{epochs,batch_size,learning_rate,warmup_steps,seed}`
- `output.dir`
- `logging.use_wandb` (default `true`) plus optional W&B metadata

Default config files:
- `configs/train.config.yaml`
- `configs/train.yaml` (same content, compatibility copy)
