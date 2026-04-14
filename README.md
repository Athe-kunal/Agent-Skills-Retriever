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

Set the path with `TRAIN_DATASET_JSONL` (default:
`artifacts/retriever_models_summary.jsonl`).

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

### Train retriever model

```bash
make retriever-train
```

Common overrides:

```bash
make retriever-train \
  TRAIN_DATASET_JSONL=artifacts/retriever_models_summary.jsonl \
  TRAIN_OUTPUT_DIR=artifacts/sentence_transformers/qwen3-summary \
  TRAIN_BASE_MODEL=Qwen/Qwen3-Embedding-0.6B \
  TRAIN_MINING_BACKEND=late_interaction \
  TRAIN_EVAL_BACKEND=late_interaction \
  TRAIN_QUERY_INSTRUCTION="Represent this query for retrieving the correct skill:" \
  TRAIN_EVAL_INSTRUCTION="Represent this query for retrieving the correct skill:"
```

### Evaluate retriever model

```bash
make retriever-evaluate
```

Examples:

- Evaluate local SentenceTransformer checkpoint as bi-encoder:

```bash
make retriever-evaluate \
  EVAL_BACKEND=bi_encoder \
  EVAL_MODEL=artifacts/sentence_transformers/qwen3-summary
```

- Evaluate local model with late interaction:

```bash
make retriever-evaluate \
  EVAL_BACKEND=late_interaction \
  EVAL_MODEL=Qwen/Qwen3-Embedding-0.6B \
  EVAL_QUERY_INSTRUCTION="Represent this query for retrieving the correct skill:" \
  EVAL_DOCUMENT_INSTRUCTION="Represent this skill summary for retrieval:"
```

- Evaluate embeddings from vLLM server:

```bash
make retriever-evaluate \
  EVAL_BACKEND=vllm \
  EVAL_MODEL=Qwen/Qwen3-Embedding-0.6B \
  EMBD_BASE_URL=http://127.0.0.1:8000/v1 \
  EMBD_API_KEY=EMPTY
```

## Metrics logged

Both training-time and standalone evaluation log:
- `eval/hit_at_1`
- `eval/hit_at_3`
- `eval/hit_at_5`
- `eval/mrr`
- `eval/queries`

Training additionally logs per-epoch train counts.
