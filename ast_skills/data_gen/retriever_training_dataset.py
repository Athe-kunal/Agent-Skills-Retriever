"""Build training/validation datasets with hybrid-mined in-batch negatives."""

from __future__ import annotations

import asyncio
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import fire
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from loguru import logger as log
from openai import OpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from ast_skills.data_gen.datamodels import SummaryRetrieverDataModel, TrainingData
from ast_skills.retriever.search import (
    bm25_search_ids,
    rrf_merge_ids,
    semantic_search_with_embedding,
)


class _DatasetSplit(NamedTuple):
    """Training/validation dataset split."""

    train_rows: list[SummaryRetrieverDataModel]
    validation_rows: list[SummaryRetrieverDataModel]


class _NegativeTextLookup(NamedTuple):
    """Summary text lookup for hard-negative mining, keyed by custom_id."""

    summary_by_custom_id: dict[str, str]


class _HybridSearchConfig(NamedTuple):
    """Hybrid search configuration for mining hard negatives."""

    chroma_root_dir: str
    embedding_base_url: str
    embedding_model: str
    api_key: str
    retrieval_pool_size: int
    rrf_k: int


class _NegativeWindowConfig(NamedTuple):
    """Start rank and target count for hard-negative accumulation."""

    window_start_rank: int
    negatives_per_row: int


class _RowQuestion(NamedTuple):
    """Row with pre-selected question for async processing."""

    row_index: int
    row: SummaryRetrieverDataModel
    question: str


_REQUIRED_PARQUET_COLUMNS: tuple[str, ...] = (
    "name",
    "markdown_content",
    "summary",
    "description",
    "question",
)

_TRAINING_ROW_FEATURES = Features(
    {
        "question": Value("string"),
        "name": Value("string"),
        "summary": Value("string"),
        "description": Value("string"),
        "negative_documents": Sequence(Value("string")),
    }
)


def _normalize_corpus_field_text(text: str) -> str:
    """Unwraps one layer of JSON string quotes and trims common markdown prefixes.

    Some pipelines store summary/description with surrounding ``"..."`` as literal
    characters; hybrid retrieval then surfaces those quotes in negatives. A leading
    ``|`` plus newline often comes from table-style markdown excerpts.
    """
    result = text.strip()
    if not result:
        return ""

    if len(result) >= 2 and result[0] == '"' and result[-1] == '"':
        try:
            decoded = json.loads(result)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, str):
            result = decoded.strip()
        else:
            result = result[1:-1].strip()

    if result.startswith("|"):
        result = result[1:].lstrip("\n").lstrip(" \t")

    return result


def _is_missing_scalar(value: Any) -> bool:
    """True if ``value`` is None or a pandas/NA missing scalar."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _raise_if_null_parquet_cell(row_index: int, column: str, value: Any) -> None:
    """Raises ``KeyError`` when a required cell is absent or null."""
    if _is_missing_scalar(value):
        raise KeyError(
            f"Row {row_index} has missing or null value for required column {column!r}"
        )


def _coerce_seed_questions_list(value: Any) -> list[str]:
    """Builds ``seed_questions`` from a ``question`` cell (string, list, or JSON string).

    Caller must ensure the cell itself is not null (see ``_raise_if_null_parquet_cell``).
    """
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict, str, bytes)):
        value = value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if not _is_missing_scalar(item)]
        return [str(parsed)]
    if isinstance(value, (list, tuple)):
        return [
            str(item)
            for item in value
            if not _is_missing_scalar(item) and str(item).strip()
        ]
    return [str(value)]


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    """Reads train Parquet rows; each dict has only the file columns (no derived fields).

    Expected columns: ``name``, ``markdown_content``, ``summary``, ``description``, ``question``.
    """
    dataframe = pd.read_parquet(path)
    missing_columns = [
        name for name in _REQUIRED_PARQUET_COLUMNS if name not in dataframe.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Parquet at {path} is missing required columns: {missing_columns}"
        )
    raw_records = dataframe.to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    for row_index, record in enumerate(raw_records):
        for column in _REQUIRED_PARQUET_COLUMNS:
            _raise_if_null_parquet_cell(row_index, column, record[column])
        row = {
            "name": str(record["name"]),
            "markdown_content": str(record["markdown_content"]),
            "summary": str(record["summary"]),
            "description": str(record["description"]),
            "question": record["question"],
        }
        rows.append(row)
    log.info(f"{path=}, {len(rows)=}")
    return rows


def _write_parquet(path: Path, rows: list[TrainingData]) -> None:
    """Writes training rows to Parquet via Hugging Face ``datasets``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        dataset = Dataset.from_dict(
            {
                "question": [],
                "name": [],
                "summary": [],
                "description": [],
                "negative_documents": [],
            },
            features=_TRAINING_ROW_FEATURES,
        )
    else:
        records = [row.__dict__ for row in rows]
        dataset = Dataset.from_list(records, features=_TRAINING_ROW_FEATURES)
    dataset.to_parquet(str(path))
    log.info(f"{path=}, {len(rows)=}")


def _rows_to_summary_models(
    rows: list[dict[str, Any]],
) -> list[SummaryRetrieverDataModel]:
    """Converts train-parquet row dicts (see ``_REQUIRED_PARQUET_COLUMNS``) to models."""
    required_keys = set(_REQUIRED_PARQUET_COLUMNS)
    models: list[SummaryRetrieverDataModel] = []
    for row_index, row in enumerate(rows):
        row_keys = set(row.keys())
        if row_keys != required_keys:
            raise KeyError(
                f"Row {row_index} must have exactly keys {sorted(required_keys)!r}; "
                f"got {sorted(row_keys)!r}"
            )
        seed_questions = _coerce_seed_questions_list(row["question"])
        normalized_seed_questions = [
            _normalize_corpus_field_text(str(question)) for question in seed_questions
        ]
        model = SummaryRetrieverDataModel(
            custom_id=str(row_index),
            markdown_content=str(row["markdown_content"]),
            seed_questions=[
                question for question in normalized_seed_questions if question.strip()
            ],
            summary=_normalize_corpus_field_text(str(row["summary"])),
            name=str(row["name"]),
            description=_normalize_corpus_field_text(str(row["description"])),
            metadata={},
        )
        models.append(model)
    log.info(f"{len(models)=}")
    return models


def _split_rows(
    rows: list[SummaryRetrieverDataModel],
    validation_ratio: float,
    seed: int,
) -> _DatasetSplit:
    """Splits rows into train/validation partitions."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1).")

    shuffled_rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled_rows)

    validation_count = max(1, int(len(shuffled_rows) * validation_ratio))
    if validation_count >= len(shuffled_rows):
        validation_count = len(shuffled_rows) - 1

    validation_rows = shuffled_rows[:validation_count]
    train_rows = shuffled_rows[validation_count:]
    log.info(f"{len(train_rows)=}, {len(validation_rows)=}, {validation_ratio=}")
    return _DatasetSplit(train_rows=train_rows, validation_rows=validation_rows)


def _pick_question(seed_questions: list[str]) -> str:
    """Returns the first non-empty seed question."""
    for question in seed_questions:
        if question and question.strip():
            return question.strip()
    return ""


def _pick_row_query(row: SummaryRetrieverDataModel) -> str:
    """Returns first non-empty question from seed questions, else first of name/summary/description."""
    question = _pick_question(row.seed_questions)
    if question:
        return question
    for text in (row.name, row.summary, row.description):
        if text and text.strip():
            return text.strip()
    log.warning(f"No non-empty seed questions or fallback text for {row.custom_id=}")
    return ""


def _build_negative_lookup(
    rows: list[SummaryRetrieverDataModel],
) -> _NegativeTextLookup:
    """Builds summary lookup map from custom_id for hard-negative mining."""
    summary_by_custom_id = {row.custom_id: row.summary.strip() for row in rows}
    return _NegativeTextLookup(summary_by_custom_id=summary_by_custom_id)


def _batch_embed_queries(
    queries: list[str],
    embedding_base_url: str,
    embedding_model: str,
    api_key: str,
    batch_size: int,
) -> dict[str, list[float]]:
    """Embeds all unique queries in large batches; returns a dict keyed by query string.

    Sending one large batch per request allows the GPU to process a full matrix
    multiply rather than N sequential single-vector forward passes, which is
    the primary reason GPUs sit idle when requests arrive one at a time.
    """
    unique_queries = list(dict.fromkeys(q for q in queries if q))
    if not unique_queries:
        return {}

    client = OpenAI(base_url=embedding_base_url, api_key=api_key)
    embedding_cache: dict[str, list[float]] = {}
    total_batches = math.ceil(len(unique_queries) / batch_size)

    for batch_index in tqdm(range(total_batches), desc="Embedding queries"):
        batch_start = batch_index * batch_size
        batch = unique_queries[batch_start : batch_start + batch_size]
        response = client.embeddings.create(model=embedding_model, input=batch)
        for position, data in enumerate(response.data):
            embedding_cache[batch[position]] = data.embedding

    log.info(f"{len(unique_queries)=}, {total_batches=}")
    return embedding_cache


def _bm25_ranked_ids_for_field(
    question: str,
    field: str,
    config: _HybridSearchConfig,
) -> list[str]:
    """Runs BM25 for one (question, field) pair without touching the ChromaDB collection."""
    return bm25_search_ids(
        query=question,
        field=field,
        root_dir=config.chroma_root_dir,
        limit=config.retrieval_pool_size,
    )


def _run_bm25_pair(
    pair: tuple[str, str],
    config: _HybridSearchConfig,
) -> tuple[tuple[str, str], list[str]]:
    """Runs BM25 for one (question, field) pair.

    Must be a module-level function: ``ProcessPoolExecutor`` serializes submitted
    callables via pickle, and local closures are not picklable.
    ``_HybridSearchConfig`` is a NamedTuple so it pickles fine.
    """
    question, field = pair
    return pair, _bm25_ranked_ids_for_field(question=question, field=field, config=config)


def _batch_bm25_queries(
    query_field_pairs: list[tuple[str, str]],
    config: _HybridSearchConfig,
    max_workers: int,
) -> dict[tuple[str, str], list[str]]:
    """Runs all (question, field) BM25 queries in parallel; returns cache keyed by pair.

    Uses ``ProcessPoolExecutor`` to bypass the GIL for CPU-bound BM25 scoring.
    The worker is a module-level function so it can be pickled for inter-process
    dispatch. ChromaDB is never loaded here — BM25 reads only the JSON index file.
    """
    unique_pairs = list(dict.fromkeys(pair for pair in query_field_pairs if pair[0]))
    if not unique_pairs:
        return {}

    bm25_cache: dict[tuple[str, str], list[str]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_bm25_pair, pair, config) for pair in unique_pairs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="BM25 queries"):
            pair, ids = future.result()
            bm25_cache[pair] = ids

    log.info(f"{len(unique_pairs)=}")
    return bm25_cache


def _parse_hybrid_search_ids(payload_json: str) -> list[str]:
    """Parses ordered document ids from a search JSON payload."""
    rows = json.loads(payload_json)
    if not isinstance(rows, list):
        return []
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id:
            ids.append(row_id)
    return ids


def _hybrid_ranked_ids_precomputed(
    question: str,
    query_embedding: list[float],
    field: str,
    config: _HybridSearchConfig,
    bm25_cache: dict[tuple[str, str], list[str]],
) -> list[str]:
    """Hybrid retrieval using cached embedding (dense) and cached BM25 ids (sparse).

    Both branches are pre-computed before this call; the only work here is
    the ChromaDB ANN lookup and the lightweight RRF merge.
    """
    dense_json = semantic_search_with_embedding(
        query_embedding=query_embedding,
        field=field,
        root_dir=config.chroma_root_dir,
        limit=config.retrieval_pool_size,
    )
    dense_ids = _parse_hybrid_search_ids(dense_json)
    sparse_ids = bm25_cache.get((question, field), [])
    return rrf_merge_ids(
        dense_ids=dense_ids,
        sparse_ids=sparse_ids,
        limit=config.retrieval_pool_size,
        rrf_k=config.rrf_k,
    )


def _mine_field_negatives(
    question: str,
    query_embedding: list[float],
    positive_custom_id: str,
    lookup_by_custom_id: dict[str, str],
    field: str,
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    bm25_cache: dict[tuple[str, str], list[str]],
) -> list[str]:
    """Mines negatives for one field by greedy accumulation starting from window_start_rank.

    The positive document is excluded before windowing. Starting after rank
    ``window_start_rank`` skips the nearest neighbours (easiest negatives).
    We accumulate until ``negatives_per_row`` are collected or the ranked list
    is exhausted — no fixed end-rank, no subsampling.
    """
    if window_config.window_start_rank < 1:
        raise ValueError(f"window_start_rank must be >= 1, got {window_config.window_start_rank=}")

    ranked_ids = _hybrid_ranked_ids_precomputed(
        question=question,
        query_embedding=query_embedding,
        field=field,
        config=hybrid_config,
        bm25_cache=bm25_cache,
    )

    positive_text = lookup_by_custom_id.get(positive_custom_id, "").strip()
    non_positive_ids = [cid for cid in ranked_ids if cid != positive_custom_id]
    candidate_ids = non_positive_ids[window_config.window_start_rank - 1:]

    seen_texts: set[str] = {positive_text} if positive_text else set()
    negatives: list[str] = []
    for candidate_id in candidate_ids:
        text = lookup_by_custom_id.get(candidate_id, "").strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            negatives.append(text)
        if len(negatives) >= window_config.negatives_per_row:
            break

    log.info(f"{field=}, {len(negatives)=}")
    return negatives


def _build_row_questions(
    rows: list[SummaryRetrieverDataModel],
) -> list[_RowQuestion]:
    """Pre-computes one question per row before asynchronous execution."""
    row_questions: list[_RowQuestion] = []
    for row_index, row in enumerate(rows):
        question = _pick_row_query(row)
        row_questions.append(
            _RowQuestion(row_index=row_index, row=row, question=question)
        )
    log.info(f"{len(row_questions)=}")
    return row_questions


def _deduplicate_training_rows(rows: list[TrainingData]) -> list[TrainingData]:
    """Deduplicates training rows by (name, question), keeping first occurrence."""
    seen: set[tuple[str, str]] = set()
    unique_rows: list[TrainingData] = []
    for row in rows:
        key = (row.name, row.question)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    duplicates_removed = len(rows) - len(unique_rows)
    log.info(f"{len(unique_rows)=}, {duplicates_removed=}")
    return unique_rows


async def _mine_field_negatives_async(
    question: str,
    query_embedding: list[float],
    positive_custom_id: str,
    lookup_by_custom_id: dict[str, str],
    field: str,
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    semaphore: asyncio.Semaphore,
    bm25_cache: dict[tuple[str, str], list[str]],
) -> list[str]:
    """Asynchronously mines negatives with concurrency control."""
    async with semaphore:
        return await asyncio.to_thread(
            _mine_field_negatives,
            question,
            query_embedding,
            positive_custom_id,
            lookup_by_custom_id,
            field,
            hybrid_config,
            window_config,
            bm25_cache,
        )


async def _build_training_row_async(
    row_question: _RowQuestion,
    lookup: _NegativeTextLookup,
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    semaphore: asyncio.Semaphore,
    embedding_cache: dict[str, list[float]],
    bm25_cache: dict[tuple[str, str], list[str]],
) -> TrainingData | None:
    """Builds a single ``TrainingData`` row asynchronously."""
    row = row_question.row
    question = row_question.question
    if not question:
        log.warning(f"Skipping row with empty question for {row.custom_id=}")
        return None

    query_embedding = embedding_cache.get(question)
    if query_embedding is None:
        log.warning(f"No pre-computed embedding for {row.custom_id=}; skipping row")
        return None

    negative_documents = await _mine_field_negatives_async(
        question=question,
        query_embedding=query_embedding,
        positive_custom_id=row.custom_id,
        lookup_by_custom_id=lookup.summary_by_custom_id,
        field="summary",
        hybrid_config=hybrid_config,
        window_config=window_config,
        semaphore=semaphore,
        bm25_cache=bm25_cache,
    )

    return TrainingData(
        question=question,
        name=row.name,
        summary=row.summary,
        description=row.description,
        negative_documents=negative_documents,
    )


async def _build_training_dataset_async(
    rows: list[SummaryRetrieverDataModel],
    all_rows: list[SummaryRetrieverDataModel],
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    max_concurrency: int,
    embedding_batch_size: int,
    split_name: str = "split",
) -> list[TrainingData]:
    """Builds ``TrainingData`` rows for one split using async concurrency.

    Pipeline phases (each with a progress bar):
      1. Batch embed all questions (GPU-saturating).
      2. Batch BM25 all (question, "summary") pairs in parallel (CPU-parallel).
      3. ChromaDB ANN lookup + RRF merge per row (async concurrent).
      4. Deduplicate by (name, question).
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1.")

    lookup = _build_negative_lookup(all_rows)
    row_questions = _build_row_questions(rows=rows)

    all_questions = [rq.question for rq in row_questions]
    embedding_cache = await asyncio.to_thread(
        _batch_embed_queries,
        all_questions,
        hybrid_config.embedding_base_url,
        hybrid_config.embedding_model,
        hybrid_config.api_key,
        embedding_batch_size,
    )

    query_field_pairs = [(rq.question, "summary") for rq in row_questions]
    bm25_cache = await asyncio.to_thread(
        _batch_bm25_queries,
        query_field_pairs,
        hybrid_config,
        max_concurrency,
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks: list[asyncio.Task[TrainingData | None]] = [
        asyncio.create_task(
            _build_training_row_async(
                row_question=row_question,
                lookup=lookup,
                hybrid_config=hybrid_config,
                window_config=window_config,
                semaphore=semaphore,
                embedding_cache=embedding_cache,
                bm25_cache=bm25_cache,
            )
        )
        for row_question in row_questions
    ]

    training_rows_or_none = await async_tqdm.gather(
        *tasks, desc=f"Mining negatives [{split_name}]"
    )
    output_rows = [row for row in training_rows_or_none if row is not None]
    deduplicated_rows = _deduplicate_training_rows(output_rows)
    log.info(f"{max_concurrency=}, {len(deduplicated_rows)=}")
    return deduplicated_rows


def _build_training_dataset(
    rows: list[SummaryRetrieverDataModel],
    all_rows: list[SummaryRetrieverDataModel],
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    max_concurrency: int,
    embedding_batch_size: int,
    split_name: str = "split",
) -> list[TrainingData]:
    """Sync wrapper around async split dataset builder."""
    return asyncio.run(
        _build_training_dataset_async(
            rows=rows,
            all_rows=all_rows,
            hybrid_config=hybrid_config,
            window_config=window_config,
            max_concurrency=max_concurrency,
            embedding_batch_size=embedding_batch_size,
            split_name=split_name,
        )
    )


def build_training_and_validation_datasets(
    input_parquet_path: str = "artifacts/train.parquet",
    output_train_parquet_path: str = "artifacts/retriever_training/train.parquet",
    output_validation_parquet_path: str = "artifacts/retriever_training/validation.parquet",
    validation_ratio: float = 0.1,
    random_seed: int = 13,
    chroma_root_dir: str = "artifacts/chroma",
    embedding_base_url: str = "http://127.0.0.1:8000/v1",
    embedding_model: str = "Qwen/Qwen3-Embedding-8B",
    api_key: str = "EMPTY",
    retrieval_pool_size: int = 200,
    rrf_k: int = 60,
    window_start_rank: int = 5,
    negatives_per_row: int = 32,
    max_concurrency: int = 256,
    embedding_batch_size: int = 512,
    smoke_test: bool = False,
) -> None:
    """Builds train/validation datasets with hybrid-mined hard negatives.

    Args:
      input_parquet_path: Parquet with columns ``name``, ``markdown_content``,
        ``summary``, ``description``, ``question`` (see ``_read_parquet``).
      output_train_parquet_path: Output path for train dataset (Parquet).
      output_validation_parquet_path: Output path for validation dataset (Parquet).
      validation_ratio: Fraction of rows allocated to validation split.
      random_seed: Seed for train/validation split shuffling.
      chroma_root_dir: Root directory containing Chroma and BM25 artifacts.
      embedding_base_url: OpenAI-compatible embedding endpoint.
      embedding_model: Embedding model used for dense query encoding.
      api_key: API key for embedding endpoint.
      retrieval_pool_size: Number of candidates retrieved per query (BM25 + dense).
        Must be well above ``negatives_per_row`` so deduplication still leaves
        enough unique candidates after filtering the positive and the window.
      rrf_k: Reciprocal-rank-fusion constant.
      window_start_rank: First rank (1-indexed) to start accumulating summary negatives.
        Ranks below this are skipped (too easy / too similar to the positive).
      negatives_per_row: Target number of summary negatives per row; accumulation
        stops as soon as this count is reached.
      max_concurrency: Maximum number of concurrent row/field retrieval tasks.
      embedding_batch_size: Number of queries per embedding API call. Larger
        batches saturate the GPU more efficiently (default 512).
      smoke_test: If True, only the first input row is processed for outputs; the
        full file is still used to resolve retrieved ids to summary/description text.
    """
    input_path = Path(input_parquet_path)
    train_path = Path(output_train_parquet_path)
    validation_path = Path(output_validation_parquet_path)

    raw_rows = _read_parquet(input_path)
    corpus_rows = _rows_to_summary_models(raw_rows)
    working_rows = corpus_rows[:1] if smoke_test else corpus_rows
    if smoke_test:
        log.info(f"{smoke_test=}, {len(working_rows)=}, {len(corpus_rows)=}")
    split = _split_rows(
        working_rows, validation_ratio=validation_ratio, seed=random_seed
    )

    hybrid_config = _HybridSearchConfig(
        chroma_root_dir=chroma_root_dir,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        api_key=api_key,
        retrieval_pool_size=retrieval_pool_size,
        rrf_k=rrf_k,
    )
    window_config = _NegativeWindowConfig(
        window_start_rank=window_start_rank,
        negatives_per_row=negatives_per_row,
    )

    train_rows = _build_training_dataset(
        rows=split.train_rows,
        all_rows=corpus_rows,
        hybrid_config=hybrid_config,
        window_config=window_config,
        max_concurrency=max_concurrency,
        embedding_batch_size=embedding_batch_size,
        split_name="train",
    )
    validation_rows = _build_training_dataset(
        rows=split.validation_rows,
        all_rows=corpus_rows,
        hybrid_config=hybrid_config,
        window_config=window_config,
        max_concurrency=max_concurrency,
        embedding_batch_size=embedding_batch_size,
        split_name="validation",
    )

    _write_parquet(train_path, train_rows)
    _write_parquet(validation_path, validation_rows)


def main() -> None:
    """CLI entrypoint."""
    fire.Fire(
        {
            "build_retriever_training_dataset": build_training_and_validation_datasets,
        }
    )


if __name__ == "__main__":
    main()
