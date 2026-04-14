"""Evaluation script for retriever models with Weights & Biases logging.

Supported retrieval backends:
- ``bi_encoder``: sentence-level embeddings from SentenceTransformer.
- ``late_interaction``: token-level MaxSim scoring (ColBERT-style).
- ``vllm``: OpenAI-compatible embeddings endpoint.

The evaluator supports appending retrieval instructions to query and document text
for instruction-tuned embedding models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Sequence

import fire
import numpy as np
import wandb
import yaml
from loguru import logger as log
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ast_skills.retriever.datamodels import SummaryRetrieverDataModel


class RetrievalMetrics(NamedTuple):
    """Evaluation summary for retrieval quality."""

    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float


class _CorpusPayload(NamedTuple):
    """Prepared corpus texts and labels."""

    texts: list[str]
    names: list[str]


class _QueryPayload(NamedTuple):
    """Prepared query texts and expected labels."""

    texts: list[str]
    expected_names: list[str]


class _EncodedCorpus(NamedTuple):
    """Encoded document representation for retrieval."""

    sentence_embeddings: np.ndarray | None
    token_embeddings: list[np.ndarray] | None


class _EncodedQueries(NamedTuple):
    """Encoded query representation for retrieval."""

    sentence_embeddings: np.ndarray | None
    token_embeddings: list[np.ndarray] | None


def _read_jsonl(path: Path) -> list[dict]:
    """Reads JSONL records from ``path``."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            normalized_line = line.strip()
            if normalized_line:
                rows.append(json.loads(normalized_line))
    log.info(f"{path=}, {len(rows)=}")
    return rows


def _read_yaml(path: Path) -> dict:
    """Reads YAML file and returns dictionary payload."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML config root must be a mapping.")
    log.info(f"{path=}, {list(payload.keys())=}")
    return payload


def _evaluate_kwargs_from_config(config: dict) -> dict:
    """Builds evaluate kwargs from YAML dictionary."""
    evaluate_config = config.get("evaluate", {})
    if not isinstance(evaluate_config, dict):
        raise ValueError("`evaluate` section must be a mapping.")
    kwargs = dict(evaluate_config)
    log.info(f"{kwargs=}")
    return kwargs


def _load_rows(dataset_jsonl: str) -> list[SummaryRetrieverDataModel]:
    """Loads rows as ``SummaryRetrieverDataModel`` objects."""
    raw_rows = _read_jsonl(Path(dataset_jsonl))
    rows = [SummaryRetrieverDataModel(**row) for row in raw_rows]
    log.info(f"{len(rows)=}")
    return rows


def _apply_instruction(text: str, instruction: str) -> str:
    """Prepends instruction text when provided."""
    normalized_text = text.strip()
    normalized_instruction = instruction.strip()
    if not normalized_instruction:
        return normalized_text
    return f"{normalized_instruction}\n{normalized_text}"


def _build_corpus(
    rows: Sequence[SummaryRetrieverDataModel],
    document_instruction: str,
) -> _CorpusPayload:
    """Builds retrieval corpus texts and skill names."""
    texts = [
        _apply_instruction(row.summary.strip() or row.description.strip(), document_instruction)
        for row in rows
    ]
    names = [row.name.strip() for row in rows]
    payload = _CorpusPayload(texts=texts, names=names)
    log.info(f"{len(payload.texts)=}, {len(payload.names)=}")
    return payload


def _build_queries(
    rows: Sequence[SummaryRetrieverDataModel],
    query_instruction: str,
) -> _QueryPayload:
    """Builds query texts and expected labels."""
    texts: list[str] = []
    expected_names: list[str] = []
    for row in rows:
        expected_name = row.name.strip()
        for question in row.seed_questions:
            normalized_question = question.strip()
            if not normalized_question:
                continue
            texts.append(_apply_instruction(normalized_question, query_instruction))
            expected_names.append(expected_name)
    payload = _QueryPayload(texts=texts, expected_names=expected_names)
    log.info(f"{len(payload.texts)=}, {len(payload.expected_names)=}")
    return payload


def _encode_sentence_embeddings(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    """Encodes sentence-level embeddings with normalization."""
    embeddings = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings)


def _encode_token_embeddings(model: SentenceTransformer, texts: Sequence[str]) -> list[np.ndarray]:
    """Encodes token-level embeddings for late interaction."""
    token_embeddings = model.encode(
        list(texts),
        output_value="token_embeddings",
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    normalized: list[np.ndarray] = []
    for token_matrix in token_embeddings:
        norms = np.linalg.norm(token_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized.append(token_matrix / norms)
    return normalized


def _encode_vllm(
    client: OpenAI,
    embedding_model: str,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """Encodes sentence embeddings with an OpenAI-compatible embeddings API."""
    vectors: list[list[float]] = []
    start = 0
    while start < len(texts):
        end = min(start + batch_size, len(texts))
        chunk = list(texts[start:end])
        response = client.embeddings.create(model=embedding_model, input=chunk)
        vectors.extend([item.embedding for item in response.data])
        start = end
    embeddings = np.asarray(vectors)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _late_interaction_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Computes ColBERT-style MaxSim score for one query-document pair."""
    token_similarities = np.matmul(query_tokens, doc_tokens.T)
    max_per_query_token = np.max(token_similarities, axis=1)
    return float(np.sum(max_per_query_token))


def _score_bi_encoder(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
) -> np.ndarray:
    """Returns cosine score matrix for sentence embeddings."""
    return np.matmul(query_embeddings, doc_embeddings.T)


def _score_late_interaction(
    query_token_embeddings: Sequence[np.ndarray],
    doc_token_embeddings: Sequence[np.ndarray],
) -> np.ndarray:
    """Returns score matrix using token-level late interaction."""
    score_matrix = np.zeros(
        (len(query_token_embeddings), len(doc_token_embeddings)),
        dtype=np.float32,
    )
    for query_index, query_tokens in enumerate(query_token_embeddings):
        for doc_index, doc_tokens in enumerate(doc_token_embeddings):
            score_matrix[query_index, doc_index] = _late_interaction_score(
                query_tokens=query_tokens,
                doc_tokens=doc_tokens,
            )
    return score_matrix


def _find_expected_rank(
    top_indices: Sequence[int],
    corpus_names: Sequence[str],
    expected_name: str,
) -> int | None:
    """Finds 1-indexed rank for the expected skill name."""
    normalized_expected_name = expected_name.casefold()
    for rank, corpus_index in enumerate(top_indices, start=1):
        if corpus_names[corpus_index].casefold() == normalized_expected_name:
            return rank
    return None


def _compute_metrics(
    score_matrix: np.ndarray,
    expected_names: Sequence[str],
    corpus_names: Sequence[str],
) -> RetrievalMetrics:
    """Computes hit@k and MRR from score matrix."""
    total_queries = len(expected_names)
    if total_queries == 0:
        return RetrievalMetrics(total_queries=0, hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, mrr=0.0)

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

    metrics = RetrievalMetrics(
        total_queries=total_queries,
        hit_at_1=hit_1 / total_queries,
        hit_at_3=hit_3 / total_queries,
        hit_at_5=hit_5 / total_queries,
        mrr=reciprocal_rank_sum / total_queries,
    )
    log.info(f"{metrics=}")
    return metrics


def _encode_sentence_backend(
    model: SentenceTransformer,
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
) -> tuple[_EncodedQueries, _EncodedCorpus]:
    """Encodes sentence-level vectors for bi-encoder retrieval."""
    query_embeddings = _encode_sentence_embeddings(model=model, texts=query_texts)
    doc_embeddings = _encode_sentence_embeddings(model=model, texts=doc_texts)
    encoded_queries = _EncodedQueries(
        sentence_embeddings=query_embeddings,
        token_embeddings=None,
    )
    encoded_corpus = _EncodedCorpus(
        sentence_embeddings=doc_embeddings,
        token_embeddings=None,
    )
    return encoded_queries, encoded_corpus


def _encode_late_interaction_backend(
    model: SentenceTransformer,
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
) -> tuple[_EncodedQueries, _EncodedCorpus]:
    """Encodes token-level vectors for late-interaction retrieval."""
    query_tokens = _encode_token_embeddings(model=model, texts=query_texts)
    doc_tokens = _encode_token_embeddings(model=model, texts=doc_texts)
    encoded_queries = _EncodedQueries(
        sentence_embeddings=None,
        token_embeddings=query_tokens,
    )
    encoded_corpus = _EncodedCorpus(
        sentence_embeddings=None,
        token_embeddings=doc_tokens,
    )
    return encoded_queries, encoded_corpus


def _encode_vllm_backend(
    client: OpenAI,
    model_name: str,
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
    batch_size: int,
) -> tuple[_EncodedQueries, _EncodedCorpus]:
    """Encodes sentence-level vectors via vLLM embeddings endpoint."""
    query_embeddings = _encode_vllm(
        client=client,
        embedding_model=model_name,
        texts=query_texts,
        batch_size=batch_size,
    )
    doc_embeddings = _encode_vllm(
        client=client,
        embedding_model=model_name,
        texts=doc_texts,
        batch_size=batch_size,
    )
    encoded_queries = _EncodedQueries(
        sentence_embeddings=query_embeddings,
        token_embeddings=None,
    )
    encoded_corpus = _EncodedCorpus(
        sentence_embeddings=doc_embeddings,
        token_embeddings=None,
    )
    return encoded_queries, encoded_corpus


def _score_backend(
    retrieval_backend: str,
    encoded_queries: _EncodedQueries,
    encoded_corpus: _EncodedCorpus,
) -> np.ndarray:
    """Builds query-document score matrix for selected backend."""
    if retrieval_backend in {"bi_encoder", "vllm"}:
        if encoded_queries.sentence_embeddings is None:
            raise ValueError("Sentence embeddings were missing for backend scoring.")
        if encoded_corpus.sentence_embeddings is None:
            raise ValueError("Sentence embeddings were missing for backend scoring.")
        return _score_bi_encoder(
            query_embeddings=encoded_queries.sentence_embeddings,
            doc_embeddings=encoded_corpus.sentence_embeddings,
        )

    if retrieval_backend == "late_interaction":
        if encoded_queries.token_embeddings is None:
            raise ValueError("Token embeddings were missing for late-interaction scoring.")
        if encoded_corpus.token_embeddings is None:
            raise ValueError("Token embeddings were missing for late-interaction scoring.")
        return _score_late_interaction(
            query_token_embeddings=encoded_queries.token_embeddings,
            doc_token_embeddings=encoded_corpus.token_embeddings,
        )

    raise ValueError("Unsupported retrieval_backend.")


def evaluate(
    dataset_jsonl: str,
    retrieval_backend: str = "bi_encoder",
    retrieval_model: str = "Qwen/Qwen3-Embedding-0.6B",
    query_instruction: str = "",
    document_instruction: str = "",
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name: str = "retriever-baseline-eval",
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    vllm_api_key: str = "EMPTY",
    vllm_batch_size: int = 64,
) -> dict[str, float]:
    """Evaluates retrieval and logs metrics to W&B.

    Args:
      dataset_jsonl: JSONL with ``SummaryRetrieverDataModel`` rows.
      retrieval_backend: ``bi_encoder``, ``late_interaction``, or ``vllm``.
      retrieval_model: SentenceTransformer model name/path or vLLM model name.
      query_instruction: Optional instruction appended before each query.
      document_instruction: Optional instruction appended before each document.
      wandb_project: W&B project name.
      wandb_entity: Optional W&B entity.
      run_name: W&B run name.
      vllm_base_url: OpenAI-compatible vLLM endpoint.
      vllm_api_key: API key for vLLM endpoint.
      vllm_batch_size: Batch size for vLLM embedding requests.

    Returns:
      Dict of scalar metrics.
    """
    rows = _load_rows(dataset_jsonl=dataset_jsonl)
    corpus = _build_corpus(rows=rows, document_instruction=document_instruction)
    queries = _build_queries(rows=rows, query_instruction=query_instruction)

    wandb_config = {
        "dataset_jsonl": dataset_jsonl,
        "retrieval_backend": retrieval_backend,
        "retrieval_model": retrieval_model,
        "query_instruction": query_instruction,
        "document_instruction": document_instruction,
        "vllm_base_url": vllm_base_url,
    }
    log.info(f"{wandb_config=}")

    wandb.init(
        project=wandb_project,
        entity=wandb_entity or None,
        name=run_name,
        config=wandb_config,
    )

    if retrieval_backend == "bi_encoder":
        model = SentenceTransformer(retrieval_model)
        encoded_queries, encoded_corpus = _encode_sentence_backend(
            model=model,
            query_texts=queries.texts,
            doc_texts=corpus.texts,
        )
    elif retrieval_backend == "late_interaction":
        model = SentenceTransformer(retrieval_model)
        encoded_queries, encoded_corpus = _encode_late_interaction_backend(
            model=model,
            query_texts=queries.texts,
            doc_texts=corpus.texts,
        )
    elif retrieval_backend == "vllm":
        client = OpenAI(base_url=vllm_base_url, api_key=vllm_api_key)
        encoded_queries, encoded_corpus = _encode_vllm_backend(
            client=client,
            model_name=retrieval_model,
            query_texts=queries.texts,
            doc_texts=corpus.texts,
            batch_size=vllm_batch_size,
        )
    else:
        raise ValueError(
            "Unsupported retrieval_backend. Use bi_encoder, late_interaction, or vllm."
        )

    score_matrix = _score_backend(
        retrieval_backend=retrieval_backend,
        encoded_queries=encoded_queries,
        encoded_corpus=encoded_corpus,
    )
    metrics = _compute_metrics(
        score_matrix=score_matrix,
        expected_names=queries.expected_names,
        corpus_names=corpus.names,
    )

    payload = {
        "eval/queries": metrics.total_queries,
        "eval/hit_at_1": metrics.hit_at_1,
        "eval/hit_at_3": metrics.hit_at_3,
        "eval/hit_at_5": metrics.hit_at_5,
        "eval/mrr": metrics.mrr,
    }
    log.info(f"{payload=}")
    wandb.log(payload)
    wandb.finish()
    return payload


def evaluate_from_config(config_path: str = "configs/train.yaml") -> dict[str, float]:
    """Loads YAML config and runs evaluation.

    Args:
      config_path: Path to YAML config file with an `evaluate` section.
    """
    payload = _read_yaml(Path(config_path))
    evaluate_kwargs = _evaluate_kwargs_from_config(payload)
    return evaluate(**evaluate_kwargs)


def main() -> None:
    """CLI entrypoint."""
    fire.Fire({"evaluate": evaluate, "evaluate_from_config": evaluate_from_config})


if __name__ == "__main__":
    main()
