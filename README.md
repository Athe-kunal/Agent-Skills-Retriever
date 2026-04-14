# AST-Based-Agent-Skills Retriever Training

This repository now includes retriever training and evaluation scripts for summary-based
skill retrieval using Sentence Transformers.

## What you need before training

### 1) Python dependencies
Install project dependencies with `uv`:

```bash
uv sync
```

### 2) Training dataset JSONL
Prepare a JSONL file with rows matching `SummaryRetrieverDataModel`:

```python
@dataclass(frozen=True)
class SummaryRetrieverDataModel:
    custom_id: str
    markdown_content: str
    seed_questions: list[str]
    summary: str
    name: str
    description: str
    metadata: dict[str, str]
```

Set fields in `configs/train.yaml` (`train.dataset_jsonl` and
`evaluate.dataset_jsonl`).

### 3) Weights & Biases setup
Training and evaluation both log metrics to W&B.

1. Create/login to a W&B account.
2. Authenticate locally:

```bash
uv run wandb login
```

3. Optionally set these in your shell:

```bash
export WANDB_PROJECT=ast-skills-retriever
export WANDB_ENTITY=your_entity
```

### 4) Model and hardware
- Default base model: `Qwen/Qwen3-Embedding-0.6B`.
- Ensure sufficient GPU memory for your chosen model and batch size.
- For vLLM-based evaluation, start an embeddings endpoint (see Make targets below).

## Makefile commands

### Serve vLLM embeddings endpoint (optional)

```bash
make vllm-embd-serve
```

### Training/evaluation config file

All hyperparameters and runtime settings now live in:

```text
configs/train.yaml
```

This file includes:
- model names
- dataset paths
- optimizer/training hyperparameters
- backend selection (`bi_encoder`, `late_interaction`, `vllm`)
- optional query/document instructions
- W&B metadata

During training, this YAML payload is forwarded into W&B config so runs are tied
to the exact committed config file.

### Train retriever model

```bash
make retriever-train
```

Train with an alternate config path:

```bash
make retriever-train \
  TRAIN_CONFIG_PATH=configs/train.yaml
```

### Evaluate retriever model

```bash
make retriever-evaluate
```

Evaluate with an alternate config path:

```bash
make retriever-evaluate \
  EVAL_CONFIG_PATH=configs/train.yaml
```

## Metrics logged

Both training-time and standalone evaluation log:
- `eval/hit_at_1`
- `eval/hit_at_3`
- `eval/hit_at_5`
- `eval/mrr`
- `eval/queries`

Training additionally logs per-epoch train counts.
