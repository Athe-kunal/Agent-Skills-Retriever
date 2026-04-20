"""Training pipeline for summary-based retrieval with Sentence Transformers.

This trainer uses one randomly sampled seed question per row as the query and the
row summary (fallback: description) as the positive document. Hard negatives are
mined with hybrid search (dense + BM25 via RRF), then training runs with
``MultipleNegativesRankingLoss``.

The module supports dense bi-encoder and late-interaction scoring backends for
hard-negative mining, and logs training/evaluation metrics to Weights & Biases.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import fire
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from loguru import logger as log
from rank_bm25 import BM25Okapi
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader

from ast_skills.retriever.datamodels import SummaryRetrieverDataModel


class _RetrievalMetrics(NamedTuple):
    """Aggregated retrieval metrics over seed questions."""

    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float


class _HardNegativeGraph(NamedTuple):
    """Hard-negative neighbor list per training row index."""

    neighbors_by_index: dict[int, list[int]]


@dataclass(frozen=True)
class _TrainingRow:
    """Sampled training row for one epoch."""

    row_index: int
    skill_name: str
    query: str
    positive_text: str


@dataclass(frozen=True)
class _MiningConfig:
    """Configuration for hard-negative mining."""

    backend: str
    dense_weight: float
    sparse_weight: float
    hard_negative_pool_size: int
    instruction: str


@dataclass(frozen=True)
class _EvalConfig:
    """Configuration for in-training evaluation."""

    backend: str
    instruction: str


def _read_jsonl(path: Path) -> list[dict]:
    """Reads non-empty JSONL lines from ``path``."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            normalized_line = line.strip()
            if normalized_line:
                rows.append(json.loads(normalized_line))
    log.info(f"{path=}, {len(rows)=}")
    return rows


def _read_yaml(path: Path) -> dict:
    """Reads YAML file and returns a dictionary payload."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML config root must be a mapping.")
    log.info(f"{path=}, {list(payload.keys())=}")
    return payload


def _load_summary_rows_from_jsonl(dataset_jsonl: str) -> list[SummaryRetrieverDataModel]:
    """Loads summary dataset rows from JSONL."""
    raw_rows = _read_jsonl(Path(dataset_jsonl))
    models = [SummaryRetrieverDataModel(**row) for row in raw_rows]
    log.info(f"{len(models)=}")
    return models


def _load_summary_rows_from_parquet(mined_parquet_path: str) -> list[SummaryRetrieverDataModel]:
    """Loads summary-style training rows from mined negatives parquet."""
    dataframe = pd.read_parquet(mined_parquet_path)
    models: list[SummaryRetrieverDataModel] = []
    for row_index, row in dataframe.iterrows():
        question = str(row.get("question", "")).strip()
        summary = str(row.get("summary", "")).strip()
        description = str(row.get("description", "")).strip()
        name = str(row.get("name", "")).strip() or f"row-{row_index}"
        if not question or not (summary or description):
            continue
        models.append(
            SummaryRetrieverDataModel(
                custom_id=str(row_index),
                markdown_content="",
                seed_questions=[question],
                summary=summary,
                name=name,
                description=description,
                metadata={},
            )
        )
    log.info(f"{mined_parquet_path=}, {len(models)=}")
    return models


def _load_summary_rows(dataset_jsonl: str = "", mined_parquet_path: str = "") -> list[SummaryRetrieverDataModel]:
    """Loads summary dataset rows from JSONL or mined-negatives parquet."""
    if mined_parquet_path.strip():
        return _load_summary_rows_from_parquet(mined_parquet_path=mined_parquet_path)
    if dataset_jsonl.strip():
        return _load_summary_rows_from_jsonl(dataset_jsonl=dataset_jsonl)
    raise ValueError("Either dataset_jsonl or mined_parquet_path must be provided.")


def _train_kwargs_from_nested_config(config: dict) -> dict:
    """Builds train kwargs from nested YAML config sections."""
    input_config = dict(config.get("input", {}))
    model_config = dict(config.get("model", {}))
    training_config = dict(config.get("training", {}))
    output_config = dict(config.get("output", {}))
    logging_config = dict(config.get("logging", {}))
    kwargs = {
        "mined_parquet_path": str(input_config.get("mined_parquet_path", "")),
        "base_model_name": str(model_config.get("name", "Qwen/Qwen3-Embedding-0.6B")),
        "epochs": int(training_config.get("epochs", 3)),
        "batch_size": int(training_config.get("batch_size", 32)),
        "learning_rate": float(training_config.get("learning_rate", 2e-5)),
        "warmup_steps": int(training_config.get("warmup_steps", 200)),
        "seed": int(training_config.get("seed", 13)),
        "output_dir": str(output_config.get("dir", "artifacts/sentence_transformers/qwen3-summary")),
        "use_wandb": bool(logging_config.get("use_wandb", True)),
        "wandb_project": str(logging_config.get("project", "ast-skills-retriever")),
        "wandb_entity": str(logging_config.get("entity", "")),
        "run_name": str(logging_config.get("run_name", "qwen3-summary-train")),
    }
    log.info(f"{kwargs=}")
    return kwargs


def _train_kwargs_from_legacy_config(config: dict) -> dict:
    """Builds train kwargs from legacy flat `train` section."""
    train_config = config.get("train", {})
    if not isinstance(train_config, dict):
        raise ValueError("`train` section must be a mapping.")
    kwargs = dict(train_config)
    return kwargs


def _apply_instruction(text: str, instruction: str) -> str:
    """Prepends instruction text when provided."""
    normalized_text = text.strip()
    normalized_instruction = instruction.strip()
    if not normalized_instruction:
        return normalized_text
    return f"{normalized_instruction}\n{normalized_text}"


def _pick_query(seed_questions: Sequence[str], rng: random.Random) -> str:
    """Returns one random non-empty seed question or an empty string."""
    candidates = [question.strip() for question in seed_questions if question.strip()]
    if not candidates:
        return ""
    return rng.choice(candidates)


def _build_training_rows(
    models: Sequence[SummaryRetrieverDataModel],
    rng: random.Random,
    query_instruction: str,
) -> list[_TrainingRow]:
    """Builds sampled epoch rows from dataset rows."""
    training_rows: list[_TrainingRow] = []
    for row_index, model in enumerate(models):
        query = _pick_query(model.seed_questions, rng)
        positive_text = model.summary.strip() or model.description.strip()
        if not query or not positive_text:
            continue
        training_rows.append(
            _TrainingRow(
                row_index=row_index,
                skill_name=model.name.strip(),
                query=_apply_instruction(query, query_instruction),
                positive_text=positive_text,
            )
        )
    log.info(f"{len(training_rows)=}")
    return training_rows


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenizes text for BM25 lookup."""
    return [token for token in text.lower().split() if token]


def _encode_sentence_embeddings(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    """Encodes sentence-level embeddings with normalization."""
    embeddings = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings)


def _encode_token_embeddings(model: SentenceTransformer, texts: Sequence[str]) -> list[np.ndarray]:
    """Encodes token-level embeddings for late interaction.

    Returns:
      List where each element has shape ``[tokens, dim]``.
    """
    token_embeddings = model.encode(
        list(texts),
        output_value="token_embeddings",
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    output: list[np.ndarray] = []
    for token_matrix in token_embeddings:
        norms = np.linalg.norm(token_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        output.append(token_matrix / norms)
    return output


def _dense_scores(
    model: SentenceTransformer,
    queries: Sequence[str],
    documents: Sequence[str],
    backend: str,
) -> np.ndarray:
    """Computes dense similarity matrix for bi-encoder or late interaction.

    Args:
      model: Sentence-transformers model.
      queries: Query texts.
      documents: Document texts.
      backend: ``bi_encoder`` or ``late_interaction``.

    Returns:
      Similarity matrix of shape ``[num_queries, num_docs]``.
    """
    if backend == "bi_encoder":
        query_embeddings = _encode_sentence_embeddings(model=model, texts=queries)
        doc_embeddings = _encode_sentence_embeddings(model=model, texts=documents)
        return np.matmul(query_embeddings, doc_embeddings.T)

    if backend != "late_interaction":
        raise ValueError("Unsupported backend for dense scoring.")

    query_tokens = _encode_token_embeddings(model=model, texts=queries)
    doc_tokens = _encode_token_embeddings(model=model, texts=documents)
    score_matrix = np.zeros((len(query_tokens), len(doc_tokens)), dtype=np.float32)

    for query_index, query_matrix in enumerate(query_tokens):
        for doc_index, doc_matrix in enumerate(doc_tokens):
            score_matrix[query_index, doc_index] = _late_interaction_score(
                query_tokens=query_matrix,
                doc_tokens=doc_matrix,
            )
    return score_matrix


def _late_interaction_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Computes ColBERT-style MaxSim score for one query-document pair."""
    token_similarities = np.matmul(query_tokens, doc_tokens.T)
    max_per_query_token = np.max(token_similarities, axis=1)
    return float(np.sum(max_per_query_token))


def _build_hybrid_graph(
    training_rows: Sequence[_TrainingRow],
    model: SentenceTransformer,
    config: _MiningConfig,
) -> _HardNegativeGraph:
    """Builds hard-negative neighbors with dense+BM25 RRF fusion."""
    queries = [row.query for row in training_rows]
    documents = [row.positive_text for row in training_rows]
    dense_scores = _dense_scores(
        model=model,
        queries=queries,
        documents=documents,
        backend=config.backend,
    )

    tokenized_docs = [_tokenize_for_bm25(document) for document in documents]
    bm25 = BM25Okapi(tokenized_docs)
    rrf_k = 60.0
    neighbors_by_index: dict[int, list[int]] = {}

    for anchor_index, row in enumerate(training_rows):
        dense_order = np.argsort(-dense_scores[anchor_index]).tolist()
        sparse_scores = bm25.get_scores(_tokenize_for_bm25(row.query))
        sparse_order = np.argsort(-sparse_scores).tolist()

        fused_scores: dict[int, float] = {}
        _merge_rrf_scores(
            fused_scores=fused_scores,
            ordered_indices=dense_order,
            training_rows=training_rows,
            anchor_index=anchor_index,
            anchor_skill_name=row.skill_name,
            weight=config.dense_weight,
            rrf_k=rrf_k,
        )
        _merge_rrf_scores(
            fused_scores=fused_scores,
            ordered_indices=sparse_order,
            training_rows=training_rows,
            anchor_index=anchor_index,
            anchor_skill_name=row.skill_name,
            weight=config.sparse_weight,
            rrf_k=rrf_k,
        )

        ordered = sorted(
            fused_scores.keys(),
            key=lambda candidate_index: fused_scores[candidate_index],
            reverse=True,
        )
        neighbors_by_index[anchor_index] = ordered[: config.hard_negative_pool_size]

    log.info(f"{config.hard_negative_pool_size=}, {len(neighbors_by_index)=}")
    return _HardNegativeGraph(neighbors_by_index=neighbors_by_index)


def _merge_rrf_scores(
    fused_scores: dict[int, float],
    ordered_indices: Sequence[int],
    training_rows: Sequence[_TrainingRow],
    anchor_index: int,
    anchor_skill_name: str,
    weight: float,
    rrf_k: float,
) -> None:
    """Merges one ranked list into reciprocal-rank-fusion scores."""
    for rank, candidate_index in enumerate(ordered_indices, start=1):
        if candidate_index == anchor_index:
            continue
        if training_rows[candidate_index].skill_name == anchor_skill_name:
            continue
        fused_scores[candidate_index] = fused_scores.get(candidate_index, 0.0) + weight * (
            1.0 / (rrf_k + rank)
        )


def _build_batches(graph: _HardNegativeGraph, batch_size: int) -> list[list[int]]:
    """Constructs batches where rows are hard negatives of each other."""
    visited: set[int] = set()
    batches: list[list[int]] = []

    for anchor_index, neighbors in graph.neighbors_by_index.items():
        if anchor_index in visited:
            continue
        batch = [anchor_index]
        for candidate_index in neighbors:
            if candidate_index in visited:
                continue
            batch.append(candidate_index)
            if len(batch) >= batch_size:
                break
        if len(batch) < 2:
            continue
        for row_index in batch:
            visited.add(row_index)
        batches.append(batch)

    log.info(f"{batch_size=}, {len(batches)=}")
    return batches


def _build_input_examples(training_rows: Sequence[_TrainingRow]) -> list[InputExample]:
    """Converts sampled rows to SentenceTransformer input examples."""
    examples = [InputExample(texts=[row.query, row.positive_text]) for row in training_rows]
    log.info(f"{len(examples)=}")
    return examples


def _flatten_batches_to_dataset(
    batches: Sequence[Sequence[int]],
    examples: Sequence[InputExample],
) -> list[InputExample]:
    """Flattens batch index groups into ordered examples."""
    ordered_indices = [index for batch in batches for index in batch]
    dataset = [examples[index] for index in ordered_indices]
    log.info(f"{len(dataset)=}")
    return dataset


def _compute_retrieval_metrics(
    models: Sequence[SummaryRetrieverDataModel],
    model: SentenceTransformer,
    config: _EvalConfig,
) -> _RetrievalMetrics:
    """Computes retrieval metrics for in-training evaluation."""
    corpus_texts = [row.summary.strip() or row.description.strip() for row in models]
    corpus_names = [row.name.strip() for row in models]

    all_questions: list[str] = []
    expected_names: list[str] = []
    for row in models:
        expected_name = row.name.strip()
        for question in row.seed_questions:
            normalized_question = question.strip()
            if not normalized_question:
                continue
            all_questions.append(_apply_instruction(normalized_question, config.instruction))
            expected_names.append(expected_name)

    if not all_questions:
        return _RetrievalMetrics(total_queries=0, hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, mrr=0.0)

    score_matrix = _dense_scores(
        model=model,
        queries=all_questions,
        documents=corpus_texts,
        backend=config.backend,
    )

    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    reciprocal_rank_sum = 0.0

    for query_index, expected_name in enumerate(expected_names):
        top_indices = np.argsort(-score_matrix[query_index])[:5]
        rank = _find_expected_rank(
            top_indices=top_indices,
            corpus_names=corpus_names,
            expected_name=expected_name,
        )
        if rank is None:
            continue
        reciprocal_rank_sum += 1.0 / rank
        if rank <= 1:
            hit_1 += 1
        if rank <= 3:
            hit_3 += 1
        if rank <= 5:
            hit_5 += 1

    total_queries = len(all_questions)
    metrics = _RetrievalMetrics(
        total_queries=total_queries,
        hit_at_1=hit_1 / total_queries,
        hit_at_3=hit_3 / total_queries,
        hit_at_5=hit_5 / total_queries,
        mrr=reciprocal_rank_sum / total_queries,
    )
    log.info(f"{metrics=}")
    return metrics


def _find_expected_rank(
    top_indices: Sequence[int],
    corpus_names: Sequence[str],
    expected_name: str,
) -> int | None:
    """Finds 1-indexed rank of the expected skill name."""
    normalized_expected_name = expected_name.casefold()
    for rank, corpus_index in enumerate(top_indices, start=1):
        if corpus_names[corpus_index].casefold() == normalized_expected_name:
            return rank
    return None


class WandbRetrievalEvaluator(SentenceEvaluator):
    """Evaluator that logs retrieval metrics to W&B after each epoch."""

    def __init__(
        self,
        models: Sequence[SummaryRetrieverDataModel],
        run: wandb.sdk.wandb_run.Run,
        config: _EvalConfig,
    ) -> None:
        self._models = list(models)
        self._run = run
        self._config = config

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        del output_path
        metrics = _compute_retrieval_metrics(
            models=self._models,
            model=model,
            config=self._config,
        )
        payload = {
            "eval/epoch": epoch,
            "eval/steps": steps,
            "eval/queries": metrics.total_queries,
            "eval/hit_at_1": metrics.hit_at_1,
            "eval/hit_at_3": metrics.hit_at_3,
            "eval/hit_at_5": metrics.hit_at_5,
            "eval/mrr": metrics.mrr,
        }
        log.info(f"{payload=}")
        self._run.log(payload)
        return metrics.hit_at_1


def _build_wandb_config(
    dataset_jsonl: str,
    base_model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    seed: int,
    mining_config: _MiningConfig,
    eval_config: _EvalConfig,
    train_config_path: str,
    train_config_payload: dict[str, object] | None,
) -> dict[str, object]:
    """Builds W&B config payload."""
    payload = {
        "dataset_jsonl": dataset_jsonl,
        "base_model_name": base_model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "mining_backend": mining_config.backend,
        "hard_negative_pool_size": mining_config.hard_negative_pool_size,
        "dense_weight": mining_config.dense_weight,
        "sparse_weight": mining_config.sparse_weight,
        "query_instruction": mining_config.instruction,
        "eval_backend": eval_config.backend,
        "eval_instruction": eval_config.instruction,
        "train_config_path": train_config_path,
    }
    if train_config_payload is not None:
        payload["train_config_payload"] = train_config_payload
    log.info(f"{payload=}")
    return payload


def _train_kwargs_from_config(config: dict) -> dict:
    """Builds train kwargs from YAML dictionary.

    Expected format:
      train:
        ...
    """
    if isinstance(config.get("train"), dict):
        kwargs = _train_kwargs_from_legacy_config(config)
    else:
        kwargs = _train_kwargs_from_nested_config(config)
    log.info(f"{kwargs=}")
    return kwargs


def train(
    dataset_jsonl: str = "",
    mined_parquet_path: str = "",
    output_dir: str = "artifacts/sentence_transformers/qwen3-summary",
    base_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_steps: int = 200,
    seed: int = 13,
    hard_negative_pool_size: int = 64,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    mining_backend: str = "bi_encoder",
    eval_backend: str = "bi_encoder",
    query_instruction: str = "",
    eval_instruction: str = "",
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name: str = "qwen3-summary-train",
    use_wandb: bool = True,
    train_config_path: str = "",
    train_config_payload: dict[str, object] | None = None,
) -> None:
    """Trains a retrieval model with hybrid hard-negative batching.

    Args:
      dataset_jsonl: JSONL file with ``SummaryRetrieverDataModel`` rows.
      output_dir: Output path for checkpoints and final model.
      base_model_name: Sentence-transformers model name/path.
      epochs: Number of epochs.
      batch_size: Training batch size.
      learning_rate: Optimizer learning rate.
      warmup_steps: Warmup steps per epoch-run.
      seed: Random seed.
      hard_negative_pool_size: Number of mined neighbors per anchor.
      dense_weight: RRF weight for dense ranking.
      sparse_weight: RRF weight for sparse ranking.
      mining_backend: ``bi_encoder`` or ``late_interaction`` for negative mining.
      eval_backend: ``bi_encoder`` or ``late_interaction`` for eval.
      query_instruction: Optional instruction prepended to training/eval queries.
      eval_instruction: Optional override instruction for evaluation only.
      wandb_project: W&B project.
      wandb_entity: W&B entity.
      run_name: W&B run name.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    models = _load_summary_rows(
        dataset_jsonl=dataset_jsonl,
        mined_parquet_path=mined_parquet_path,
    )
    st_model = SentenceTransformer(base_model_name)

    mining_config = _MiningConfig(
        backend=mining_backend,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        hard_negative_pool_size=hard_negative_pool_size,
        instruction=query_instruction,
    )
    effective_eval_instruction = eval_instruction or query_instruction
    eval_config = _EvalConfig(backend=eval_backend, instruction=effective_eval_instruction)

    wandb_config = _build_wandb_config(
        dataset_jsonl=dataset_jsonl,
        base_model_name=base_model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        seed=seed,
        mining_config=mining_config,
        eval_config=eval_config,
        train_config_path=train_config_path,
        train_config_payload=train_config_payload,
    )

    evaluator: WandbRetrievalEvaluator | None = None
    run = None
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity or None,
            name=run_name,
            config=wandb_config,
        )
        run = wandb.run
        if run is None:
            raise RuntimeError("wandb.run was None after wandb.init.")
        evaluator = WandbRetrievalEvaluator(models=models, run=run, config=eval_config)

    for epoch_index in range(epochs):
        epoch_rng = random.Random(seed + epoch_index)
        training_rows = _build_training_rows(
            models=models,
            rng=epoch_rng,
            query_instruction=mining_config.instruction,
        )
        examples = _build_input_examples(training_rows=training_rows)
        graph = _build_hybrid_graph(
            training_rows=training_rows,
            model=st_model,
            config=mining_config,
        )
        batches = _build_batches(graph=graph, batch_size=batch_size)
        if not batches:
            raise ValueError("No training batches were constructed. Check dataset quality.")

        dataset = _flatten_batches_to_dataset(batches=batches, examples=examples)
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        train_loss = losses.MultipleNegativesRankingLoss(st_model)

        log.info(f"{epoch_index=}, {len(dataset)=}, {len(batches)=}")
        if evaluator is not None:
            st_model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=1,
                warmup_steps=warmup_steps,
                optimizer_params={"lr": learning_rate},
                show_progress_bar=True,
                output_path=output_dir,
                checkpoint_path=output_dir,
                checkpoint_save_total_limit=2,
            )
        else:
            st_model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=warmup_steps,
                optimizer_params={"lr": learning_rate},
                show_progress_bar=True,
                output_path=output_dir,
                checkpoint_path=output_dir,
                checkpoint_save_total_limit=2,
            )
        if run is not None:
            run.log({"train/epoch": epoch_index + 1, "train/examples": len(dataset)})

    st_model.save(output_dir)
    log.info(f"{output_dir=}")
    if use_wandb:
        wandb.finish()


def train_from_config(config_path: str = "configs/train.config.yaml") -> None:
    """Loads YAML config and runs training.

    Args:
      config_path: Path to YAML config file with a `train` section.
    """
    payload = _read_yaml(Path(config_path))
    train_kwargs = _train_kwargs_from_config(payload)
    train_kwargs["train_config_path"] = config_path
    train_kwargs["train_config_payload"] = payload
    train(**train_kwargs)


def main() -> None:
    """CLI entrypoint."""
    fire.Fire({"train": train, "train_from_config": train_from_config})


if __name__ == "__main__":
    main()
