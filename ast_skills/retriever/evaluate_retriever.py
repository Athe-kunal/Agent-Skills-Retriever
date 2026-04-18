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
import asyncio
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import fire
import numpy as np
import pandas as pd
import wandb
import yaml
from loguru import logger as log
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ast_skills.data_gen.datamodels import TrainingData
from ast_skills.retriever.datamodels import (
    SummaryRetrieverDataModel,
    ValidatedSkillQuestionRow,
)
from ast_skills.retriever.maximal_marginal_relevance_question import (
    MmrSelectionConfig,
    select_diverse_questions_for_rows,
)


class RetrievalMetrics(NamedTuple):
    """Evaluation summary for retrieval quality."""

    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr: float


class _CorpusPayload(NamedTuple):
    """Prepared corpus texts and labels."""

    texts: list[str]
    names: list[str]


class _QueryPayload(NamedTuple):
    """Prepared query texts and expected labels."""

    texts: list[str]
    expected_names: list[str]


class _ValidationPayload(NamedTuple):
    """Loaded validation rows plus query labels."""

    rows: list[TrainingData]
    query_texts: list[str]
    expected_names: list[str]


class _QuestionSplitRows(NamedTuple):
    """Expanded rows after question-level train/validation split."""

    train_rows: list[TrainingData]
    validation_rows: list[TrainingData]


class _EncodedCorpus(NamedTuple):
    """Encoded document representation for retrieval."""

    sentence_embeddings: np.ndarray | None
    token_embeddings: list[np.ndarray] | None


class _ValidationModelConfig(NamedTuple):
    """Model configuration for validation parquet evaluation."""

    model_key: str
    model_type: str
    model_name: str


class _ValidationFieldConfig(NamedTuple):
    """Field configuration for validation parquet evaluation."""

    field_key: str


class _TextChunk(NamedTuple):
    """Chunk of texts and its start index in the original sequence."""

    start_index: int
    texts: list[str]


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


VALIDATION_MODEL_CONFIGS: tuple[_ValidationModelConfig, ...] = (
    _ValidationModelConfig(
        model_key="bert_large_encoder",
        model_type="dense",
        model_name="sentence-transformers/bert-large-nli-mean-tokens",
    ),
    _ValidationModelConfig(
        model_key="qwen3_0_6b",
        model_type="dense",
        model_name="Qwen/Qwen3-Embedding-0.6B",
    ),
    _ValidationModelConfig(
        model_key="bm25",
        model_type="sparse",
        model_name="bm25",
    ),
)

VALIDATION_FIELD_CONFIGS: tuple[_ValidationFieldConfig, ...] = (
    _ValidationFieldConfig(field_key="summary"),
    _ValidationFieldConfig(field_key="description"),
)


def _load_validation_payload(validation_parquet: str) -> _ValidationPayload:
    """Loads ``TrainingData`` rows from validation parquet."""
    validation_path = Path(validation_parquet)
    dataframe = pd.read_parquet(validation_path)
    rows: list[TrainingData] = []
    for record in dataframe.to_dict(orient="records"):
        rows.append(TrainingData(**record))

    query_texts = [row.question.strip() for row in rows if row.question.strip()]
    expected_names = [row.name.strip() for row in rows if row.question.strip()]
    payload = _ValidationPayload(
        rows=rows,
        query_texts=query_texts,
        expected_names=expected_names,
    )
    log.info(f"{validation_path=}, {len(rows)=}, {len(query_texts)=}")
    return payload


def _read_validated_rows(input_jsonl: str) -> list[ValidatedSkillQuestionRow]:
    """Loads validated skill rows from JSONL."""
    raw_rows = _read_jsonl(Path(input_jsonl))
    rows: list[ValidatedSkillQuestionRow] = []
    for row in raw_rows:
        rows.append(
            ValidatedSkillQuestionRow(
                custom_id=str(row["custom_id"]),
                description=str(row["description"]),
                filtered_questions=[
                    str(question).strip()
                    for question in row["filtered_questions"]
                    if str(question).strip()
                ],
                markdown_content=str(row["markdown_content"]),
                name=str(row["name"]),
                num_from_scenario_questions=str(row["num_from_scenario_questions"]),
                num_from_seed_questions=str(row["num_from_seed_questions"]),
                reasoning=str(row["reasoning"]),
                summary=str(row.get("summary", "")),
            )
        )
    log.info(f"{input_jsonl=}, {len(rows)=}")
    return rows


def _summary_for_row(row: ValidatedSkillQuestionRow) -> str:
    """Returns summary text for retrieval, falling back to description."""
    normalized_summary = row.summary.strip()
    if normalized_summary:
        return normalized_summary
    return row.description.strip()


def _build_training_row(question: str, row: ValidatedSkillQuestionRow) -> TrainingData:
    """Builds a ``TrainingData`` row with empty in-batch negatives."""
    return TrainingData(
        question=question.strip(),
        name=row.name.strip(),
        summary=_summary_for_row(row),
        description=row.description.strip(),
        in_batch_negatives_descriptions=[],
        in_batch_negatives_summary=[],
    )


def _split_questions_for_row(
    row: ValidatedSkillQuestionRow,
    train_questions: Sequence[str],
    validation_questions: Sequence[str],
) -> _QuestionSplitRows:
    """Builds split rows from preselected train/validation question groups."""
    split_rows = _QuestionSplitRows(
        train_rows=[
            _build_training_row(question=question, row=row)
            for question in train_questions
        ],
        validation_rows=[
            _build_training_row(question=question, row=row)
            for question in validation_questions
        ],
    )
    log.info(
        f"{row.custom_id=}, {row.name=}, {train_questions=}, {validation_questions=}"
    )
    return split_rows


def _split_questions_dataset(
    rows: Sequence[ValidatedSkillQuestionRow],
    selected_questions_by_custom_id: dict[str, list[str]],
    train_questions_per_skill: int,
    validation_questions_per_skill: int,
) -> _QuestionSplitRows:
    """Splits all validated rows into question-level train and validation rows."""
    total_selected_questions = (
        train_questions_per_skill + validation_questions_per_skill
    )
    train_rows: list[TrainingData] = []
    validation_rows: list[TrainingData] = []
    for row in rows:
        selected_questions = selected_questions_by_custom_id[row.custom_id]
        if len(selected_questions) != total_selected_questions:
            raise ValueError(
                f"Invalid selected question count for {row.custom_id=}, "
                f"{len(selected_questions)=}, {total_selected_questions=}"
            )

        train_questions = selected_questions[:train_questions_per_skill]
        validation_questions = selected_questions[train_questions_per_skill:]
        split_rows = _split_questions_for_row(
            row=row,
            train_questions=train_questions,
            validation_questions=validation_questions,
        )
        train_rows.extend(split_rows.train_rows)
        validation_rows.extend(split_rows.validation_rows)
    payload = _QuestionSplitRows(train_rows=train_rows, validation_rows=validation_rows)
    log.info(f"{len(train_rows)=}, {len(validation_rows)=}")
    return payload


def _write_training_data_parquet(path: Path, rows: Sequence[TrainingData]) -> None:
    """Writes ``TrainingData`` rows into Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([row.__dict__ for row in rows])
    dataframe.to_parquet(path)
    log.info(f"{path=}, {len(rows)=}")


def _build_split_result_payload(
    train_parquet: str,
    validation_parquet: str,
    train_rows: Sequence[TrainingData],
    validation_rows: Sequence[TrainingData],
    validation_metrics: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    """Builds output payload for question-split and validation evaluation."""
    payload: dict[str, Any] = {
        "train_parquet": train_parquet,
        "validation_parquet": validation_parquet,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "validation_metrics": validation_metrics,
    }
    log.info(f"{payload=}")
    return payload


def _text_for_field(row: TrainingData, field_key: str) -> str:
    """Returns text for ``summary`` or ``description`` field retrieval."""
    if field_key == "summary":
        return row.summary.strip()
    if field_key == "description":
        return row.description.strip()
    raise ValueError(f"Unsupported field: {field_key}")


def _build_validation_corpus(
    rows: Sequence[TrainingData],
    field_key: str,
) -> _CorpusPayload:
    """Builds unique corpus for one retrieval field."""
    text_by_name: dict[str, str] = {}
    for row in rows:
        name = row.name.strip()
        if not name or name in text_by_name:
            continue
        text = _text_for_field(row=row, field_key=field_key)
        if not text:
            continue
        text_by_name[name] = text

    names = list(text_by_name.keys())
    texts = [text_by_name[name] for name in names]
    payload = _CorpusPayload(texts=texts, names=names)
    log.info(f"{field_key=}, {len(names)=}, {len(texts)=}")
    return payload


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
        _apply_instruction(
            row.summary.strip() or row.description.strip(), document_instruction
        )
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


def _encode_sentence_embeddings(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """Encodes sentence-level embeddings with normalization."""
    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    log.info(f"{batch_size=}, {len(texts)=}")
    return np.asarray(embeddings)


def _encode_token_embeddings(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
) -> list[np.ndarray]:
    """Encodes token-level embeddings for late interaction."""
    token_embeddings = model.encode(
        list(texts),
        output_value="token_embeddings",
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    normalized: list[np.ndarray] = []
    for token_matrix in token_embeddings:
        norms = np.linalg.norm(token_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized.append(token_matrix / norms)
    return normalized


def _chunk_texts(texts: Sequence[str], batch_size: int) -> list[_TextChunk]:
    """Splits input texts into fixed-size chunks."""
    chunks: list[_TextChunk] = []
    start_index = 0
    while start_index < len(texts):
        end_index = min(start_index + batch_size, len(texts))
        chunks.append(
            _TextChunk(
                start_index=start_index, texts=list(texts[start_index:end_index])
            )
        )
        start_index = end_index
    log.info(f"{len(texts)=}, {batch_size=}, {len(chunks)=}")
    return chunks


async def _fetch_embeddings_chunk(
    client: AsyncOpenAI,
    embedding_model: str,
    text_chunk: _TextChunk,
    semaphore: asyncio.Semaphore,
) -> tuple[int, list[list[float]]]:
    """Requests one embeddings chunk under semaphore control."""
    async with semaphore:
        response = await client.embeddings.create(
            model=embedding_model, input=text_chunk.texts
        )
    embeddings = [item.embedding for item in response.data]
    return text_chunk.start_index, embeddings


async def _encode_vllm_async(
    client: AsyncOpenAI,
    embedding_model: str,
    texts: Sequence[str],
    batch_size: int,
    max_concurrency: int,
) -> np.ndarray:
    """Encodes embeddings with async OpenAI-compatible API and concurrency limits."""
    chunks = _chunk_texts(texts=texts, batch_size=batch_size)
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _fetch_embeddings_chunk(
            client=client,
            embedding_model=embedding_model,
            text_chunk=text_chunk,
            semaphore=semaphore,
        )
        for text_chunk in chunks
    ]
    chunk_results = await asyncio.gather(*tasks)
    ordered_results = sorted(chunk_results, key=lambda item: item[0])
    vectors: list[list[float]] = []
    for _, chunk_vectors in ordered_results:
        vectors.extend(chunk_vectors)
    embeddings = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    log.info(f"{embeddings.shape=}, {max_concurrency=}")
    return embeddings / norms


async def _encode_vllm_pair_async(
    client: AsyncOpenAI,
    model_name: str,
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
    batch_size: int,
    max_concurrency: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encodes query and document texts concurrently with async vLLM calls."""
    encoded_queries_task = _encode_vllm_async(
        client=client,
        embedding_model=model_name,
        texts=query_texts,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )
    encoded_docs_task = _encode_vllm_async(
        client=client,
        embedding_model=model_name,
        texts=doc_texts,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )
    query_embeddings, doc_embeddings = await asyncio.gather(
        encoded_queries_task,
        encoded_docs_task,
    )
    return query_embeddings, doc_embeddings


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


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenizes text for BM25."""
    return text.casefold().split()


def _score_bm25(
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
) -> np.ndarray:
    """Returns BM25 score matrix for all query-document pairs."""
    tokenized_docs = [_tokenize_for_bm25(text) for text in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)
    score_rows: list[np.ndarray] = []
    for query_text in query_texts:
        tokenized_query = _tokenize_for_bm25(query_text)
        score_rows.append(
            np.asarray(bm25.get_scores(tokenized_query), dtype=np.float32)
        )
    score_matrix = np.vstack(score_rows)
    log.info(f"{score_matrix.shape=}")
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
        return RetrievalMetrics(
            total_queries=0,
            hit_at_1=0.0,
            hit_at_3=0.0,
            hit_at_5=0.0,
            hit_at_10=0.0,
            mrr=0.0,
        )

    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0
    reciprocal_rank_sum = 0.0

    for query_index, expected_name in enumerate(expected_names):
        top_indices = np.argsort(-score_matrix[query_index])[:10]
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
        if rank <= 10:
            hit_10 += 1

    metrics = RetrievalMetrics(
        total_queries=total_queries,
        hit_at_1=hit_1 / total_queries,
        hit_at_3=hit_3 / total_queries,
        hit_at_5=hit_5 / total_queries,
        hit_at_10=hit_10 / total_queries,
        mrr=reciprocal_rank_sum / total_queries,
    )
    log.info(f"{metrics=}")
    return metrics


def _encode_sentence_backend(
    model: SentenceTransformer,
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
    sentence_batch_size: int,
) -> tuple[_EncodedQueries, _EncodedCorpus]:
    """Encodes sentence-level vectors for bi-encoder retrieval."""
    query_embeddings = _encode_sentence_embeddings(
        model=model,
        texts=query_texts,
        batch_size=sentence_batch_size,
    )
    doc_embeddings = _encode_sentence_embeddings(
        model=model,
        texts=doc_texts,
        batch_size=sentence_batch_size,
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


def _encode_late_interaction_backend(
    model: SentenceTransformer,
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
    sentence_batch_size: int,
) -> tuple[_EncodedQueries, _EncodedCorpus]:
    """Encodes token-level vectors for late-interaction retrieval."""
    query_tokens = _encode_token_embeddings(
        model=model,
        texts=query_texts,
        batch_size=sentence_batch_size,
    )
    doc_tokens = _encode_token_embeddings(
        model=model,
        texts=doc_texts,
        batch_size=sentence_batch_size,
    )
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
    client: AsyncOpenAI,
    model_name: str,
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
    batch_size: int,
    max_concurrency: int,
) -> tuple[_EncodedQueries, _EncodedCorpus]:
    """Encodes sentence-level vectors via vLLM embeddings endpoint."""
    query_embeddings, doc_embeddings = asyncio.run(
        _encode_vllm_pair_async(
            client=client,
            model_name=model_name,
            query_texts=query_texts,
            doc_texts=doc_texts,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
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
            raise ValueError(
                "Token embeddings were missing for late-interaction scoring."
            )
        if encoded_corpus.token_embeddings is None:
            raise ValueError(
                "Token embeddings were missing for late-interaction scoring."
            )
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
    vllm_max_concurrency: int = 8,
    sentence_batch_size: int = 64,
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
      vllm_max_concurrency: Max concurrent requests for async vLLM embedding calls.
      sentence_batch_size: Batch size for SentenceTransformer encoding.

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
        "vllm_batch_size": vllm_batch_size,
        "vllm_max_concurrency": vllm_max_concurrency,
        "sentence_batch_size": sentence_batch_size,
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
            sentence_batch_size=sentence_batch_size,
        )
    elif retrieval_backend == "late_interaction":
        model = SentenceTransformer(retrieval_model)
        encoded_queries, encoded_corpus = _encode_late_interaction_backend(
            model=model,
            query_texts=queries.texts,
            doc_texts=corpus.texts,
            sentence_batch_size=sentence_batch_size,
        )
    elif retrieval_backend == "vllm":
        client = AsyncOpenAI(base_url=vllm_base_url, api_key=vllm_api_key)
        encoded_queries, encoded_corpus = _encode_vllm_backend(
            client=client,
            model_name=retrieval_model,
            query_texts=queries.texts,
            doc_texts=corpus.texts,
            batch_size=vllm_batch_size,
            max_concurrency=vllm_max_concurrency,
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
        "eval/hit_at_10": metrics.hit_at_10,
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


def _build_validation_wandb_config(
    validation_parquet: str,
    run_name: str,
    sentence_batch_size: int,
) -> dict[str, object]:
    """Builds W&B config for validation parquet evaluation."""
    config = {
        "validation_parquet": validation_parquet,
        "run_name": run_name,
        "sentence_batch_size": sentence_batch_size,
        "fields": [field.field_key for field in VALIDATION_FIELD_CONFIGS],
        "models": [model.model_name for model in VALIDATION_MODEL_CONFIGS],
    }
    log.info(f"{config=}")
    return config


def _build_validation_payload(
    metrics_by_field: dict[str, dict[str, RetrievalMetrics]],
) -> dict[str, float]:
    """Builds flattened payload for validation evaluation metrics."""
    payload: dict[str, float] = {}
    for field_key, metrics_by_model in metrics_by_field.items():
        for model_key, metrics in metrics_by_model.items():
            payload[f"eval/{field_key}/{model_key}/queries"] = float(
                metrics.total_queries
            )
            payload[f"eval/{field_key}/{model_key}/hit_at_1"] = metrics.hit_at_1
            payload[f"eval/{field_key}/{model_key}/hit_at_3"] = metrics.hit_at_3
            payload[f"eval/{field_key}/{model_key}/hit_at_5"] = metrics.hit_at_5
            payload[f"eval/{field_key}/{model_key}/hit_at_10"] = metrics.hit_at_10
            payload[f"eval/{field_key}/{model_key}/mrr"] = metrics.mrr
    log.info(f"{payload=}")
    return payload


def _build_validation_output(
    metrics_by_field: dict[str, dict[str, RetrievalMetrics]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Builds nested JSON-serializable metrics output."""
    output: dict[str, dict[str, dict[str, float]]] = {}
    for field_key, metrics_by_model in metrics_by_field.items():
        output[field_key] = {}
        for model_key, metrics in metrics_by_model.items():
            output[field_key][model_key] = metrics._asdict()
    log.info(f"{output=}")
    return output


def evaluate_validation_parquet(
    validation_parquet: str = "Validation.parquet",
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name: str = "validation-parquet-eval",
    sentence_batch_size: int = 64,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluates summary/description retrieval on validation parquet.

    Runs BERT large encoder, Qwen3-0.6B, and BM25 across both retrieval fields.
    """
    validation_payload = _load_validation_payload(validation_parquet=validation_parquet)
    metrics_by_field: dict[str, dict[str, RetrievalMetrics]] = {}

    wandb.init(
        project=wandb_project,
        entity=wandb_entity or None,
        name=run_name,
        config=_build_validation_wandb_config(
            validation_parquet=validation_parquet,
            run_name=run_name,
            sentence_batch_size=sentence_batch_size,
        ),
    )

    for field_config in VALIDATION_FIELD_CONFIGS:
        field_key = field_config.field_key
        corpus = _build_validation_corpus(
            rows=validation_payload.rows,
            field_key=field_key,
        )
        metrics_by_model: dict[str, RetrievalMetrics] = {}
        for model_config in VALIDATION_MODEL_CONFIGS:
            log.info(f"{field_key=}, {model_config.model_key=}")
            if model_config.model_type == "dense":
                model = SentenceTransformer(model_config.model_name)
                query_embeddings = _encode_sentence_embeddings(
                    model=model,
                    texts=validation_payload.query_texts,
                    batch_size=sentence_batch_size,
                )
                doc_embeddings = _encode_sentence_embeddings(
                    model=model,
                    texts=corpus.texts,
                    batch_size=sentence_batch_size,
                )
                score_matrix = _score_bi_encoder(
                    query_embeddings=query_embeddings,
                    doc_embeddings=doc_embeddings,
                )
            elif model_config.model_type == "sparse":
                score_matrix = _score_bm25(
                    query_texts=validation_payload.query_texts,
                    doc_texts=corpus.texts,
                )
            else:
                raise ValueError(f"Unsupported model type: {model_config.model_type}")

            metrics = _compute_metrics(
                score_matrix=score_matrix,
                expected_names=validation_payload.expected_names,
                corpus_names=corpus.names,
            )
            metrics_by_model[model_config.model_key] = metrics
        metrics_by_field[field_key] = metrics_by_model

    wandb.log(_build_validation_payload(metrics_by_field))
    table_rows = []
    for field_key, metrics_by_model in metrics_by_field.items():
        for model_key, metrics in metrics_by_model.items():
            table_rows.append(
                [
                    field_key,
                    model_key,
                    metrics.total_queries,
                    metrics.hit_at_1,
                    metrics.hit_at_3,
                    metrics.hit_at_5,
                    metrics.hit_at_10,
                    metrics.mrr,
                ]
            )
    table = wandb.Table(
        columns=[
            "field",
            "model",
            "queries",
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "hit_at_10",
            "mrr",
        ],
        data=table_rows,
    )
    wandb.log({"eval/model_comparison": table})
    wandb.finish()
    return _build_validation_output(metrics_by_field)


async def evaluate_validated_skill_questions_async(
    input_jsonl: str,
    output_train_parquet: str = "artifacts/retriever_training/train.parquet",
    output_validation_parquet: str = "artifacts/retriever_training/validation.parquet",
    train_questions_per_skill: int = 2,
    validation_questions_per_skill: int = 1,
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    vllm_api_key: str = "EMPTY",
    vllm_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    vllm_batch_size: int = 64,
    vllm_max_concurrency: int = 8,
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name: str = "validated-questions-parquet-eval",
    sentence_batch_size: int = 64,
) -> dict[str, Any]:
    """Runs MMR question selection, parquet creation, and validation evaluation."""
    total_selected_questions = (
        train_questions_per_skill + validation_questions_per_skill
    )
    if total_selected_questions != 3:
        raise ValueError(
            "This workflow requires exactly 3 selected MMR questions per row."
        )

    validated_rows = _read_validated_rows(input_jsonl=input_jsonl)
    mmr_config = MmrSelectionConfig(
        base_url=vllm_base_url,
        api_key=vllm_api_key,
        embedding_model=vllm_embedding_model,
        mmr_lambda=0.5,
        selected_question_count=total_selected_questions,
        batch_size=vllm_batch_size,
        max_concurrency=vllm_max_concurrency,
    )
    selected_questions_by_custom_id = await select_diverse_questions_for_rows(
        rows=validated_rows,
        config=mmr_config,
    )
    split_rows = _split_questions_dataset(
        rows=validated_rows,
        selected_questions_by_custom_id=selected_questions_by_custom_id,
        train_questions_per_skill=train_questions_per_skill,
        validation_questions_per_skill=validation_questions_per_skill,
    )
    await asyncio.to_thread(
        _write_training_data_parquet,
        Path(output_train_parquet),
        split_rows.train_rows,
    )
    await asyncio.to_thread(
        _write_training_data_parquet,
        Path(output_validation_parquet),
        split_rows.validation_rows,
    )
    validation_metrics = await asyncio.to_thread(
        evaluate_validation_parquet,
        validation_parquet=output_validation_parquet,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_name=run_name,
        sentence_batch_size=sentence_batch_size,
    )
    return _build_split_result_payload(
        train_parquet=output_train_parquet,
        validation_parquet=output_validation_parquet,
        train_rows=split_rows.train_rows,
        validation_rows=split_rows.validation_rows,
        validation_metrics=validation_metrics,
    )


def evaluate_validated_skill_questions(
    input_jsonl: str,
    output_train_parquet: str = "artifacts/retriever_training/train.parquet",
    output_validation_parquet: str = "artifacts/retriever_training/validation.parquet",
    train_questions_per_skill: int = 2,
    validation_questions_per_skill: int = 1,
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    vllm_api_key: str = "EMPTY",
    vllm_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    vllm_batch_size: int = 64,
    vllm_max_concurrency: int = 8,
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name: str = "validated-questions-parquet-eval",
    sentence_batch_size: int = 64,
) -> dict[str, Any]:
    """Sync CLI wrapper for async validated-question evaluation workflow."""
    return asyncio.run(
        evaluate_validated_skill_questions_async(
            input_jsonl=input_jsonl,
            output_train_parquet=output_train_parquet,
            output_validation_parquet=output_validation_parquet,
            train_questions_per_skill=train_questions_per_skill,
            validation_questions_per_skill=validation_questions_per_skill,
            vllm_base_url=vllm_base_url,
            vllm_api_key=vllm_api_key,
            vllm_embedding_model=vllm_embedding_model,
            vllm_batch_size=vllm_batch_size,
            vllm_max_concurrency=vllm_max_concurrency,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            run_name=run_name,
            sentence_batch_size=sentence_batch_size,
        )
    )


def main() -> None:
    """CLI entrypoint."""
    fire.Fire(
        {
            "evaluate": evaluate,
            "evaluate_from_config": evaluate_from_config,
            "evaluate_validation_parquet": evaluate_validation_parquet,
            "evaluate_validated_skill_questions": evaluate_validated_skill_questions,
        }
    )


if __name__ == "__main__":
    main()
