"""Build training/validation datasets with hybrid-mined in-batch negatives."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any, NamedTuple

import fire
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from loguru import logger as log

from ast_skills.data_gen.datamodels import SummaryRetrieverDataModel, TrainingData
from ast_skills.retriever.search import hybrid_search


class _DatasetSplit(NamedTuple):
    """Training/validation dataset split."""

    train_rows: list[SummaryRetrieverDataModel]
    validation_rows: list[SummaryRetrieverDataModel]


class _NegativeTextLookup(NamedTuple):
    """Text lookup for negatives by custom_id."""

    summary_by_custom_id: dict[str, str]
    description_by_custom_id: dict[str, str]


class _HybridSearchConfig(NamedTuple):
    """Hybrid search configuration for mining hard negatives."""

    chroma_root_dir: str
    embedding_base_url: str
    embedding_model: str
    api_key: str
    top_k: int
    rrf_k: int


class _NegativeWindowConfig(NamedTuple):
    """Rank window and sample size for hard negatives."""

    window_start_rank: int
    window_end_rank: int
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
        "in_batch_negatives_descriptions": Sequence(Value("string")),
        "in_batch_negatives_summary": Sequence(Value("string")),
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
                "in_batch_negatives_descriptions": [],
                "in_batch_negatives_summary": [],
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


def _pick_question(seed_questions: list[str], rng: random.Random) -> str:
    """Picks one random non-empty seed question."""
    candidates = [
        question.strip() for question in seed_questions if question and question.strip()
    ]
    if not candidates:
        return ""
    return rng.choice(candidates)


def _pick_row_query(row: SummaryRetrieverDataModel, rng: random.Random) -> str:
    """Picks a non-empty query from seed questions, else from name/summary/description."""
    question = _pick_question(row.seed_questions, rng)
    if question:
        return question
    fallbacks = [
        text.strip()
        for text in (row.name, row.summary, row.description)
        if text and text.strip()
    ]
    if not fallbacks:
        log.warning(
            f"No non-empty seed questions or fallback text for {row.custom_id=}"
        )
        return ""
    return rng.choice(fallbacks)


def _build_negative_lookup(
    rows: list[SummaryRetrieverDataModel],
) -> _NegativeTextLookup:
    """Builds lookup maps from custom_id to summary/description text."""
    summary_by_custom_id: dict[str, str] = {}
    description_by_custom_id: dict[str, str] = {}
    for row in rows:
        summary_by_custom_id[row.custom_id] = row.summary.strip()
        description_by_custom_id[row.custom_id] = row.description.strip()
    return _NegativeTextLookup(
        summary_by_custom_id=summary_by_custom_id,
        description_by_custom_id=description_by_custom_id,
    )


def _parse_hybrid_search_ids(payload_json: str) -> list[str]:
    """Parses ordered document ids from ``hybrid_search`` output."""
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


def _slice_hard_negative_window(
    ordered_ids: list[str],
    positive_custom_id: str,
    config: _NegativeWindowConfig,
) -> list[str]:
    """Returns candidate ids from the configured hard-negative rank window."""
    filtered_ids = [
        candidate_id
        for candidate_id in ordered_ids
        if candidate_id != positive_custom_id
    ]

    if (
        config.window_start_rank < 1
        or config.window_end_rank < config.window_start_rank
    ):
        raise ValueError("Invalid hard-negative rank window.")

    start_index = config.window_start_rank - 1
    end_index = config.window_end_rank
    return filtered_ids[start_index:end_index]


def _hybrid_ranked_ids(
    question: str,
    field: str,
    config: _HybridSearchConfig,
) -> list[str]:
    """Runs hybrid retrieval and returns ranked ids."""
    payload_json = hybrid_search(
        query=question,
        field=field,
        root_dir=config.chroma_root_dir,
        embedding_base_url=config.embedding_base_url,
        embedding_model=config.embedding_model,
        api_key=config.api_key,
        limit=config.top_k,
        rrf_k=config.rrf_k,
    )
    return _parse_hybrid_search_ids(payload_json)


def _mine_field_negatives(
    question: str,
    positive_custom_id: str,
    lookup_by_custom_id: dict[str, str],
    field: str,
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    rng: random.Random,
) -> list[str]:
    """Mines negatives for one field using hybrid retrieval + rank window."""
    ranked_ids = _hybrid_ranked_ids(
        question=question, field=field, config=hybrid_config
    )
    hard_window_ids = _slice_hard_negative_window(
        ordered_ids=ranked_ids,
        positive_custom_id=positive_custom_id,
        config=window_config,
    )

    negatives: list[str] = []
    for candidate_id in hard_window_ids:
        negative_text = lookup_by_custom_id.get(candidate_id, "").strip()
        if negative_text:
            negatives.append(negative_text)

    if len(negatives) <= window_config.negatives_per_row:
        return negatives

    sampled_negatives = rng.sample(negatives, window_config.negatives_per_row)
    log.info(f"{field=}, {len(negatives)=}, {len(sampled_negatives)=}")
    return sampled_negatives


def _build_row_questions(
    rows: list[SummaryRetrieverDataModel],
    question_pick_seed: int,
) -> list[_RowQuestion]:
    """Pre-computes one question per row before asynchronous execution."""
    question_rng = random.Random(question_pick_seed)
    row_questions: list[_RowQuestion] = []
    for row_index, row in enumerate(rows):
        question = _pick_row_query(row, question_rng)
        row_questions.append(
            _RowQuestion(row_index=row_index, row=row, question=question)
        )
    log.info(f"{len(row_questions)=}")
    return row_questions


async def _mine_field_negatives_async(
    question: str,
    positive_custom_id: str,
    lookup_by_custom_id: dict[str, str],
    field: str,
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    rng: random.Random,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Asynchronously mines negatives with concurrency control."""
    async with semaphore:
        return await asyncio.to_thread(
            _mine_field_negatives,
            question,
            positive_custom_id,
            lookup_by_custom_id,
            field,
            hybrid_config,
            window_config,
            rng,
        )


async def _build_training_row_async(
    row_question: _RowQuestion,
    lookup: _NegativeTextLookup,
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    seed: int,
    semaphore: asyncio.Semaphore,
) -> TrainingData | None:
    """Builds a single ``TrainingData`` row asynchronously."""
    row = row_question.row
    question = row_question.question
    if not question:
        log.warning(f"Skipping row with empty question for {row.custom_id=}")
        return None

    rng_seed_base = seed + (row_question.row_index * 10_000)
    summary_rng = random.Random(rng_seed_base + 1)
    description_rng = random.Random(rng_seed_base + 2)

    negative_summaries_task = _mine_field_negatives_async(
        question=question,
        positive_custom_id=row.custom_id,
        lookup_by_custom_id=lookup.summary_by_custom_id,
        field="summary",
        hybrid_config=hybrid_config,
        window_config=window_config,
        rng=summary_rng,
        semaphore=semaphore,
    )
    negative_descriptions_task = _mine_field_negatives_async(
        question=question,
        positive_custom_id=row.custom_id,
        lookup_by_custom_id=lookup.description_by_custom_id,
        field="description",
        hybrid_config=hybrid_config,
        window_config=window_config,
        rng=description_rng,
        semaphore=semaphore,
    )
    negative_summaries, negative_descriptions = await asyncio.gather(
        negative_summaries_task, negative_descriptions_task
    )

    return TrainingData(
        question=question,
        name=row.name,
        summary=row.summary,
        description=row.description,
        in_batch_negatives_descriptions=negative_descriptions,
        in_batch_negatives_summary=negative_summaries,
    )


async def _build_training_dataset_async(
    rows: list[SummaryRetrieverDataModel],
    all_rows: list[SummaryRetrieverDataModel],
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    seed: int,
    question_pick_seed: int,
    max_concurrency: int,
) -> list[TrainingData]:
    """Builds ``TrainingData`` rows for one split using async concurrency."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1.")

    lookup = _build_negative_lookup(all_rows)
    row_questions = _build_row_questions(
        rows=rows, question_pick_seed=question_pick_seed
    )
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks: list[asyncio.Task[TrainingData | None]] = []
    for row_question in row_questions:
        task = asyncio.create_task(
            _build_training_row_async(
                row_question=row_question,
                lookup=lookup,
                hybrid_config=hybrid_config,
                window_config=window_config,
                seed=seed,
                semaphore=semaphore,
            )
        )
        tasks.append(task)

    training_rows_or_none = await asyncio.gather(*tasks)
    output_rows = [row for row in training_rows_or_none if row is not None]
    log.info(f"{max_concurrency=}, {len(output_rows)=}")
    return output_rows


def _build_training_dataset(
    rows: list[SummaryRetrieverDataModel],
    all_rows: list[SummaryRetrieverDataModel],
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    seed: int,
    question_pick_seed: int,
    max_concurrency: int,
) -> list[TrainingData]:
    """Sync wrapper around async split dataset builder."""
    return asyncio.run(
        _build_training_dataset_async(
            rows=rows,
            all_rows=all_rows,
            hybrid_config=hybrid_config,
            window_config=window_config,
            seed=seed,
            question_pick_seed=question_pick_seed,
            max_concurrency=max_concurrency,
        )
    )


def build_training_and_validation_datasets(
    input_parquet_path: str = "artifacts/train.parquet",
    output_train_parquet_path: str = "artifacts/retriever_training/train.parquet",
    output_validation_parquet_path: str = "artifacts/retriever_training/validation.parquet",
    validation_ratio: float = 0.1,
    random_seed: int = 13,
    question_pick_seed: int = 42,
    chroma_root_dir: str = "artifacts/chroma",
    embedding_base_url: str = "http://127.0.0.1:8000/v1",
    embedding_model: str = "Qwen/Qwen3-Embedding-8B",
    api_key: str = "EMPTY",
    top_k: int = 100,
    rrf_k: int = 60,
    window_start_rank: int = 5,
    window_end_rank: int = 36,
    negatives_per_row: int = 32,
    max_concurrency: int = 256,
    smoke_test: bool = False,
) -> None:
    """Builds train/validation datasets with hybrid-mined hard negatives.

    Args:
      input_parquet_path: Parquet with columns ``name``, ``markdown_content``,
        ``summary``, ``description``, ``question`` (see ``_read_parquet``).
      output_train_parquet_path: Output path for train dataset (Parquet).
      output_validation_parquet_path: Output path for validation dataset (Parquet).
      validation_ratio: Fraction of rows allocated to validation split.
      random_seed: Seed for train/validation split and negative subsampling.
      question_pick_seed: Seed for randomly choosing one ``seed_question`` per row.
      chroma_root_dir: Root directory containing Chroma and BM25 artifacts.
      embedding_base_url: OpenAI-compatible embedding endpoint.
      embedding_model: Embedding model used for dense query encoding.
      api_key: API key for embedding endpoint.
      top_k: Hybrid retrieval top-K before hard-negative windowing.
      rrf_k: Reciprocal-rank-fusion constant.
      window_start_rank: First rank (1-indexed) of hard-negative window.
      window_end_rank: Last rank (1-indexed, inclusive) of hard-negative window.
      negatives_per_row: Number of sampled negatives per field.
      max_concurrency: Maximum number of concurrent row/field retrieval tasks.
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
        top_k=top_k,
        rrf_k=rrf_k,
    )
    window_config = _NegativeWindowConfig(
        window_start_rank=window_start_rank,
        window_end_rank=window_end_rank,
        negatives_per_row=negatives_per_row,
    )

    train_rows = _build_training_dataset(
        rows=split.train_rows,
        all_rows=corpus_rows,
        hybrid_config=hybrid_config,
        window_config=window_config,
        seed=random_seed,
        question_pick_seed=question_pick_seed,
        max_concurrency=max_concurrency,
    )
    validation_rows = _build_training_dataset(
        rows=split.validation_rows,
        all_rows=corpus_rows,
        hybrid_config=hybrid_config,
        window_config=window_config,
        seed=random_seed + 1,
        question_pick_seed=question_pick_seed,
        max_concurrency=max_concurrency,
    )

    _write_parquet(train_path, train_rows)
    _write_parquet(validation_path, validation_rows)


def main() -> None:
    """CLI entrypoint."""
    fire.Fire(
        {
            "build": build_training_and_validation_datasets,
            "build_retriever_training_dataset": build_training_and_validation_datasets,
        }
    )


if __name__ == "__main__":
    main()
