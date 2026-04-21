# Agent Skills Retriever

A Python project for building and evaluating retrieval systems over agent skill data.

This repository focuses on **dense retrieval training** with hard-negative mining and
validation workflows. It includes:

- Data preparation and synthetic data generation utilities.
- Chroma-based retrieval indexing.
- Hard-negative mining from retrieval windows.
- Sentence Transformer training (single GPU and FSDP multi-GPU).
- Validation/evaluation pipelines (dense and BM25 baselines).
- Optional Hugging Face dataset/model upload commands.

---

## Repository structure

```text
ast_skills/
  common/          # Shared API + graph + upload helpers
  data_gen/        # Data generation and dataset build tooling
  evaluation/      # Validation and model sweep evaluation entry points
  retriever/       # Retrieval index/search utilities
  train/           # Training, dataset shaping, and annotation tooling
configs/           # Training and runtime YAML configs
scripts/           # Shell wrappers for 2-GPU training/evaluation
tests/             # Unit tests
Makefile           # Primary task runner
```

---

## Prerequisites

- Python environment managed with `uv`.
- GPU + vLLM environment for embedding-serving and high-throughput retrieval tasks.
- Optional Weights & Biases account for experiment logging.

Install dependencies:

```bash
uv sync
```

Optional login for tracking:

```bash
uv run wandb login
```

---

## End-to-end workflow (recommended)

### 1) Build retrieval index (Chroma)

```bash
make build-retriever-chroma
```

Default behavior:

- Input parquet: `artifacts/train.parquet`
- Output index root: `artifacts/chroma`
- Indexed fields: `summary,description`

### 2) Mine hard negatives into train/validation parquet files

```bash
make build-mined-negatives-parquet
```

Default behavior:

- Writes train output to `artifacts/retriever_training/train.parquet`
- Writes validation output to `artifacts/retriever_training/validation.parquet`
- Uses retrieval window and negative sampling controls from Makefile variables.

### 3) Train sentence-transformer retriever

```bash
make retriever-train
```

By default, this uses `TRAIN_CONFIG_PATH=configs/train.config.yaml`.

### 4) Validate/evaluate

```bash
make retriever-evaluate
make retriever-evaluate-validation
make retriever-evaluate-validation-bm25
```

---

## Makefile targets

Run `make help` to print a concise target list.

### Core retrieval pipeline

- `make build-retriever-chroma`  
  Build Chroma indexes from a parquet dataset.

- `make build-mined-negatives-parquet`  
  Produce train/validation parquet files with mined hard negatives.

- `make retriever-train`  
  Train from the YAML config file at `TRAIN_CONFIG_PATH`.

- `make retriever-train-fsdp-2gpu TRAIN_CONFIG_PATH=... GPU_IDS=0,1`  
  Launch FSDP training on two GPUs with the helper shell script.

### Evaluation targets

- `make retriever-evaluate`  
  Run evaluation from `TRAIN_CONFIG_PATH`.

- `make retriever-evaluate-validation`  
  Evaluate a retrieval model against a validation parquet.

- `make retriever-evaluate-validation-bm25`  
  Evaluate BM25 baseline against validation parquet.

- `make retriever-evaluate-model-sweep`  
  Sweep multiple dense models against the validation parquet.

- `make retriever-evaluate-two-gpu`  
  Run a two-GPU model sweep helper script.

- `make smoke-test`  
  Validation evaluation with `EVAL_MAX_VAL_ROWS=5`.

### Embedding server and data preparation

- `make vllm-embd-serve`  
  Launch a local vLLM embedding server.

- `make build-mined-training-data`  
  Build an additional training parquet with generated hard negatives.

### Publishing artifacts

- `make hf-upload-dataset`  
  Upload train/validation/test parquet splits to a Hugging Face dataset repo.

- `make hf-upload-model`  
  Upload a local model directory to a Hugging Face model repo.

---

## High-impact Makefile variables

Variables can be overridden inline:

```bash
make retriever-train TRAIN_CONFIG_PATH=configs/train.yaml
```

Commonly overridden variables:

- `TRAIN_CONFIG_PATH` - training/evaluation YAML config path.
- `TRAINING_DATASET_PARQUET` - input parquet for mining negatives.
- `CHROMA_ROOT` - output location for Chroma indexes.
- `EMBD_BASE_URL`, `EMBD_MODEL`, `EMBD_API_KEY` - embedding endpoint/model settings.
- `MINED_TRAIN_PARQUET`, `MINED_VALIDATION_PARQUET` - mined parquet output paths.
- `EVAL_MODEL`, `EVAL_VAL_PARQUET` - model path + validation parquet for eval commands.
- `HF_*` variables - Hugging Face upload settings.

---

## Configuration files

Primary configs live in `configs/`:

- `configs/train.config.yaml` - default training config used by Makefile.
- `configs/train.yaml` - compatibility or alternate config.
- `configs/fsdp_accelerate.yaml` - Accelerate/FSDP runtime configuration.

The training config controls:

- Input parquet paths (train and optional validation).
- Base model identifier.
- Training hyperparameters (epochs, batch size, learning rate, precision).
- FSDP options.
- Output artifact path.
- W&B logging metadata.

---

## Notes on stale defaults addressed

This repository previously contained stale Makefile defaults and variable collisions.
Current behavior now:

- Uses `TRAIN_CONFIG_PATH` consistently for both training and config-based evaluation.
- Removes hardcoded machine-local absolute model path defaults.
- Splits dataset-mining concurrency variables from training-data-generation variables to avoid accidental overrides.
- Adds a first-class `help` target for discoverability.

---

## Development checks

```bash
uv run pytest
```

