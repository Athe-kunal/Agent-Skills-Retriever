"""Evaluation script for retriever models with Weights & Biases logging.

Supported retrieval backends:
- ``bi_encoder``: sentence-level embeddings from SentenceTransformer.
- ``late_interaction``: token-level MaxSim scoring (ColBERT-style).
- ``vllm``: OpenAI-compatible embeddings endpoint.

The evaluator supports appending retrieval instructions to query and document text
for instruction-tuned embedding models.
"""

from __future__ import annotations

import dataclasses
import json
import asyncio
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chromadb
import fire
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from loguru import logger as log
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from tqdm import tqdm

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

from ast_skills.data_gen.datamodels import TrainingData
from ast_skills.train.datamodels import DataPoint
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

    rows: list[DataPoint]
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


class _HfEncoderAssets(NamedTuple):
    """Tokenizer and model used for Hugging Face embedding inference."""

    tokenizer: AutoTokenizer
    model: AutoModel
    device: str


class _ChromaCorpusArtifacts(NamedTuple):
    """Stored document embeddings and names loaded from Chroma."""

    embeddings: np.ndarray
    names: list[str]


class _BM25CachePayload(NamedTuple):
    """Cached BM25 corpus texts and their corresponding skill names."""

    texts: list[str]
    names: list[str]


def _require_sentence_transformers() -> Any:
    """Loads ``sentence_transformers`` lazily for optional dependency support."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required only for retrieval_backend='bi_encoder'. "
            "Install it or switch to retrieval_backend='vllm'."
        ) from error
    return SentenceTransformer


def _require_hf_transformers() -> tuple[Any, Any]:
    """Loads Hugging Face classes lazily for optional dependency support."""
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as error:
        raise ImportError(
            "transformers is required only when use_hf_encoder=True or "
            "retrieval_backend='late_interaction'."
        ) from error
    return AutoModel, AutoTokenizer


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

_VLLM_STARTUP_POLL_INTERVAL_SEC = 2.0
_VLLM_STARTUP_TIMEOUT_SEC = 600.0
_VLLM_STARTUP_PROBE_TIMEOUT_SEC = 5.0
_CHROMA_ADD_BATCH_SIZE = 5000


def _data_point_from_parquet_record(record: dict[str, Any]) -> DataPoint:
    """Builds a ``DataPoint`` from a Parquet row dict, ignoring unknown columns."""
    kwargs: dict[str, str] = {}
    for field in dataclasses.fields(DataPoint):
        raw = record.get(field.name)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            kwargs[field.name] = ""
        elif isinstance(raw, str):
            kwargs[field.name] = raw
        else:
            kwargs[field.name] = str(raw)
    return DataPoint(**kwargs)


def _load_validation_payload(validation_parquet: str) -> _ValidationPayload:
    """Loads ``DataPoint`` rows from validation parquet."""
    validation_path = Path(validation_parquet)
    dataframe = pd.read_parquet(validation_path)
    rows: list[DataPoint] = []
    for record in dataframe.to_dict(orient="records"):
        rows.append(_data_point_from_parquet_record(record))

    query_texts = [row.question.strip() for row in rows if row.question.strip()]
    expected_names = [row.name.strip() for row in rows if row.question.strip()]
    payload = _ValidationPayload(
        rows=rows,
        query_texts=query_texts,
        expected_names=expected_names,
    )
    log.info(f"{validation_path=}, {len(rows)=}, {len(query_texts)=}")
    return payload


def _slice_validation_payload(
    payload: _ValidationPayload,
    max_validation_rows: int,
) -> _ValidationPayload:
    """Slices validation payload to at most ``max_validation_rows`` rows."""
    if max_validation_rows <= 0:
        return payload

    sliced_rows = payload.rows[:max_validation_rows]
    query_texts = [row.question.strip() for row in sliced_rows if row.question.strip()]
    expected_names = [row.name.strip() for row in sliced_rows if row.question.strip()]
    sliced_payload = _ValidationPayload(
        rows=sliced_rows,
        query_texts=query_texts,
        expected_names=expected_names,
    )
    log.info(f"{max_validation_rows=}, {len(sliced_rows)=}, {len(query_texts)=}")
    return sliced_payload


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
                summary=str(row["summary"]),
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


def _text_for_field(row: DataPoint, field_key: str) -> str:
    """Returns text for ``summary`` or ``description`` field retrieval."""
    if field_key == "summary":
        return row.summary.strip()
    if field_key == "description":
        return row.description.strip()
    raise ValueError(f"Unsupported field: {field_key}")


def _build_validation_corpus(
    rows: Sequence[DataPoint],
    field_key: str,
) -> _CorpusPayload:
    """Builds unique corpus for one retrieval field."""
    text_by_name: dict[str, str] = {}
    empty_name_skips = 0
    duplicate_name_skips = 0
    empty_field_skips = 0
    for row in rows:
        name = row.name.strip()
        if not name:
            empty_name_skips += 1
            continue
        if name in text_by_name:
            duplicate_name_skips += 1
            continue
        text = _text_for_field(row=row, field_key=field_key)
        if not text:
            empty_field_skips += 1
            continue
        text_by_name[name] = text

    names = list(text_by_name.keys())
    texts = [text_by_name[name] for name in names]
    payload = _CorpusPayload(texts=texts, names=names)
    log.info(
        f"{field_key=}, parquet_rows={len(rows)}, unique_corpus_docs={len(texts)}, "
        f"{empty_name_skips=}, {duplicate_name_skips=}, {empty_field_skips=}"
    )
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


def _sanitize_model_name(model_name: str) -> str:
    """Converts model name to a filesystem-safe identifier."""
    sanitized_model_name = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")
    log.info(f"{model_name=}, {sanitized_model_name=}")
    return sanitized_model_name or "model"


def _build_collection_name(
    field_key: str, *, sanitized_model_name: str
) -> str:
    """Builds Chroma collection name: sanitized model id plus field key."""
    collection_name = f"{sanitized_model_name}_{field_key}"
    log.info(f"{field_key=}, {sanitized_model_name=}, {collection_name=}")
    return collection_name


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


def _load_hf_encoder(model_name: str) -> _HfEncoderAssets:
    """Loads tokenizer and model for Hugging Face encoder inference."""
    auto_model_cls, auto_tokenizer_cls = _require_hf_transformers()
    tokenizer = auto_tokenizer_cls.from_pretrained(model_name, trust_remote_code=True)
    model = auto_model_cls.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    assets = _HfEncoderAssets(tokenizer=tokenizer, model=model, device=device)
    log.info(f"{model_name=}, {assets.device=}")
    return assets


def _mean_pool_embeddings(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Computes attention-mask mean pooling for sentence embeddings."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked_hidden = last_hidden_state * mask
    summed_hidden = masked_hidden.sum(dim=1)
    summed_mask = mask.sum(dim=1).clamp(min=1e-9)
    return summed_hidden / summed_mask


def _encode_hf_sentence_embeddings(
    assets: _HfEncoderAssets,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """Encodes normalized sentence embeddings with Hugging Face transformers."""
    all_embeddings: list[np.ndarray] = []
    for start_index in range(0, len(texts), batch_size):
        text_batch = list(texts[start_index : start_index + batch_size])
        tokenized = assets.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        tokenized = {k: v.to(assets.device) for k, v in tokenized.items()}
        with torch.no_grad():
            outputs = assets.model(**tokenized)
        pooled = _mean_pool_embeddings(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=tokenized["attention_mask"],
        )
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(normalized.detach().cpu().numpy().astype(np.float32))
    embeddings = np.vstack(all_embeddings)
    log.info(f"{len(texts)=}, {batch_size=}, {embeddings.shape=}")
    return embeddings


def _encode_hf_token_embeddings(
    assets: _HfEncoderAssets,
    texts: Sequence[str],
    batch_size: int,
) -> list[np.ndarray]:
    """Encodes token-level embeddings for ColBERT-style late interaction."""
    token_embeddings: list[np.ndarray] = []
    for start_index in range(0, len(texts), batch_size):
        text_batch = list(texts[start_index : start_index + batch_size])
        tokenized = assets.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        attention_mask = tokenized["attention_mask"]
        tokenized = {k: v.to(assets.device) for k, v in tokenized.items()}
        with torch.no_grad():
            outputs = assets.model(**tokenized)
        hidden_state = outputs.last_hidden_state.detach().cpu()
        attention_mask = attention_mask.detach().cpu()
        for row_index in range(hidden_state.shape[0]):
            active_tokens = attention_mask[row_index].bool()
            token_matrix = hidden_state[row_index][active_tokens].numpy().astype(
                np.float32
            )
            norms = np.linalg.norm(token_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            token_embeddings.append(token_matrix / norms)
    log.info(f"{len(texts)=}, {batch_size=}, {len(token_embeddings)=}")
    return token_embeddings


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


async def _fetch_embeddings_chunk_with_progress(
    client: AsyncOpenAI,
    embedding_model: str,
    text_chunk: _TextChunk,
    semaphore: asyncio.Semaphore,
    progress_bar: tqdm,
) -> tuple[int, list[list[float]]]:
    """Requests one embeddings chunk and advances the tqdm bar by texts completed."""
    start_index, vectors = await _fetch_embeddings_chunk(
        client=client,
        embedding_model=embedding_model,
        text_chunk=text_chunk,
        semaphore=semaphore,
    )
    progress_bar.update(len(text_chunk.texts))
    return start_index, vectors


async def _encode_vllm_async(
    client: AsyncOpenAI,
    embedding_model: str,
    texts: Sequence[str],
    batch_size: int,
    max_concurrency: int,
    progress_desc: str = "vLLM embeddings",
    progress_position: int | None = None,
) -> np.ndarray:
    """Encodes embeddings with async OpenAI-compatible API and concurrency limits."""
    chunks = _chunk_texts(texts=texts, batch_size=batch_size)
    semaphore = asyncio.Semaphore(max_concurrency)
    tqdm_kwargs: dict[str, Any] = {
        "total": len(texts),
        "desc": progress_desc,
        "unit": "text",
        "mininterval": 0.5,
    }
    if progress_position is not None:
        tqdm_kwargs["position"] = progress_position
    with tqdm(**tqdm_kwargs) as progress_bar:
        tasks = [
            _fetch_embeddings_chunk_with_progress(
                client=client,
                embedding_model=embedding_model,
                text_chunk=text_chunk,
                semaphore=semaphore,
                progress_bar=progress_bar,
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
        progress_desc="vLLM query embeddings",
        progress_position=0,
    )
    encoded_docs_task = _encode_vllm_async(
        client=client,
        embedding_model=model_name,
        texts=doc_texts,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        progress_desc="vLLM document embeddings",
        progress_position=1,
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


def _build_bm25_cache_path(artifacts_root: str, field_key: str) -> Path:
    """Builds path for cached BM25 input corpus for one field."""
    bm25_cache_path = Path(artifacts_root) / "bm25_cache" / f"{field_key}.json"
    log.info(f"{field_key=}, {bm25_cache_path=}")
    return bm25_cache_path


def _load_or_write_bm25_docs(
    artifacts_root: str,
    field_key: str,
    corpus: _CorpusPayload,
    force_reindex: bool,
) -> _BM25CachePayload:
    """Loads cached BM25 docs and names, or writes cache from current corpus."""
    bm25_cache_path = _build_bm25_cache_path(
        artifacts_root=artifacts_root, field_key=field_key
    )
    if bm25_cache_path.exists() and not force_reindex:
        payload = json.loads(bm25_cache_path.read_text(encoding="utf-8"))
        if "names" in payload:
            documents = [str(item) for item in payload["documents"]]
            names = [str(item) for item in payload["names"]]
            log.info(f"{bm25_cache_path=}, loaded_from_cache=True, {len(documents)=}")
            return _BM25CachePayload(texts=documents, names=names)
        log.info(f"{bm25_cache_path=}, names_missing=True, rebuilding cache")

    bm25_cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"documents": corpus.texts, "names": corpus.names}
    bm25_cache_path.write_text(json.dumps(payload), encoding="utf-8")
    log.info(f"{bm25_cache_path=}, loaded_from_cache=False, {len(corpus.texts)=}")
    return _BM25CachePayload(texts=corpus.texts, names=corpus.names)


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


def _encode_hf_sentence_backend(
    model_name: str,
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
    sentence_batch_size: int,
) -> tuple[_EncodedQueries, _EncodedCorpus]:
    """Encodes sentence embeddings via Hugging Face transformers."""
    assets = _load_hf_encoder(model_name=model_name)
    query_embeddings = _encode_hf_sentence_embeddings(
        assets=assets,
        texts=query_texts,
        batch_size=sentence_batch_size,
    )
    doc_embeddings = _encode_hf_sentence_embeddings(
        assets=assets,
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
    model_name: str,
    query_texts: Sequence[str],
    doc_texts: Sequence[str],
    sentence_batch_size: int,
) -> tuple[_EncodedQueries, _EncodedCorpus]:
    """Encodes token-level vectors for late-interaction retrieval."""
    assets = _load_hf_encoder(model_name=model_name)
    query_tokens = _encode_hf_token_embeddings(
        assets=assets,
        texts=query_texts,
        batch_size=sentence_batch_size,
    )
    doc_tokens = _encode_hf_token_embeddings(
        assets=assets,
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
    vllm_port: int = 8002,
    vllm_base_url: str = "",
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
      vllm_port: Port for the vLLM server.
      vllm_base_url: OpenAI-compatible vLLM endpoint. Derived from vllm_port when empty.
      vllm_api_key: API key for vLLM endpoint.
      vllm_batch_size: Batch size for vLLM embedding requests.
      vllm_max_concurrency: Max concurrent requests for async vLLM embedding calls.
      sentence_batch_size: Batch size for SentenceTransformer encoding.

    Returns:
      Dict of scalar metrics.
    """
    if not vllm_base_url:
        vllm_base_url = f"http://127.0.0.1:{vllm_port}/v1"
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
        sentence_transformer_cls = _require_sentence_transformers()
        model = sentence_transformer_cls(retrieval_model)
        encoded_queries, encoded_corpus = _encode_sentence_backend(
            model=model,
            query_texts=queries.texts,
            doc_texts=corpus.texts,
            sentence_batch_size=sentence_batch_size,
        )
    elif retrieval_backend == "late_interaction":
        encoded_queries, encoded_corpus = _encode_late_interaction_backend(
            model_name=retrieval_model,
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
    _log_scalars_to_wandb_summary(payload)
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
    retrieval_model: str,
    force_reindex: bool,
    max_validation_rows: int,
) -> dict[str, object]:
    """Builds W&B config for validation parquet evaluation."""
    config = {
        "validation_parquet": validation_parquet,
        "run_name": run_name,
        "sentence_batch_size": sentence_batch_size,
        "retrieval_model": retrieval_model,
        "force_reindex": force_reindex,
        "max_validation_rows": max_validation_rows,
        "fields": [field.field_key for field in VALIDATION_FIELD_CONFIGS],
        "models": [retrieval_model, "bm25"],
    }
    log.info(f"{config=}")
    return config


def _vllm_openai_models_probe_url(vllm_base_url: str) -> str:
    """Returns the OpenAI-compatible models URL used to verify vLLM is up."""
    return f"{vllm_base_url.rstrip('/')}/models"


def _is_port_in_use(port: int) -> bool:
    """Returns True when a process is already bound to the TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _probe_vllm_http_ready(
    probe_url: str, timeout_sec: float, api_key: str
) -> bool:
    """Returns True when probe_url responds with HTTP 2xx."""
    request = Request(
        probe_url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return 200 <= response.getcode() < 300
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def _wait_for_vllm_server_ready(
    process: subprocess.Popen[bytes],
    vllm_base_url: str,
    api_key: str,
    poll_interval_sec: float,
    timeout_sec: float,
    probe_timeout_sec: float,
) -> None:
    """Blocks until the vLLM OpenAI endpoint answers or startup fails."""
    deadline = time.monotonic() + timeout_sec
    probe_url = _vllm_openai_models_probe_url(vllm_base_url=vllm_base_url)
    log.info(
        f"{probe_url=}, {poll_interval_sec=}, {timeout_sec=}, {probe_timeout_sec=}"
    )
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                "vLLM process exited before the endpoint became ready; "
                f"{exit_code=}, {probe_url=}. Server stderr was not captured."
            )
        if _probe_vllm_http_ready(
            probe_url=probe_url,
            timeout_sec=probe_timeout_sec,
            api_key=api_key,
        ):
            log.info(f"{probe_url=}, {process.pid=}, ready=True")
            return
        time.sleep(poll_interval_sec)
    raise TimeoutError(
        f"vLLM did not become ready within {timeout_sec}s; last probe was {probe_url=}."
    )


def _maybe_start_vllm_server(
    model_name: str,
    vllm_port: int,
    vllm_gpu_device: int,
    vllm_gpu_memory_utilization: float,
    start_vllm_server: bool,
    vllm_base_url: str,
    vllm_api_key: str,
    startup_timeout_sec: float = _VLLM_STARTUP_TIMEOUT_SEC,
    poll_interval_sec: float = _VLLM_STARTUP_POLL_INTERVAL_SEC,
    probe_timeout_sec: float = _VLLM_STARTUP_PROBE_TIMEOUT_SEC,
) -> subprocess.Popen[bytes] | None:
    """Starts vLLM server in the background when requested."""
    if not start_vllm_server:
        return None

    if _is_port_in_use(vllm_port):
        raise RuntimeError(
            f"Port {vllm_port} is already in use. "
            "Stop the existing vLLM server before starting a new one."
        )

    command = [
        "uv",
        "run",
        "vllm",
        "serve",
        model_name,
        "--gpu-memory-utilization",
        str(vllm_gpu_memory_utilization),
        "--runner",
        "pooling",
        "--max-model-len",
        "8192",
        "--port",
        str(vllm_port),
        "--host",
        "0.0.0.0",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(vllm_gpu_device)
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info(f"{command=}, {process.pid=}")
    _wait_for_vllm_server_ready(
        process=process,
        vllm_base_url=vllm_base_url,
        api_key=vllm_api_key,
        poll_interval_sec=poll_interval_sec,
        timeout_sec=startup_timeout_sec,
        probe_timeout_sec=probe_timeout_sec,
    )
    return process


def _chroma_collection_doc_count(
    artifacts_root: str,
    model_name: str,
    field_key: str,
) -> int:
    """Returns doc count for a Chroma collection, or 0 if the collection is missing."""
    sanitized_model_name = _sanitize_model_name(model_name=model_name)
    chroma_path = Path(artifacts_root) / "chroma_validation" / sanitized_model_name
    if not chroma_path.exists():
        log.info(f"{chroma_path=}, exists=False")
        return 0
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection_name = _build_collection_name(
        field_key=field_key, sanitized_model_name=sanitized_model_name
    )
    try:
        count = client.get_collection(name=collection_name).count()
        log.info(f"{collection_name=}, {count=}")
        return count
    except Exception:
        log.info(f"{collection_name=}, collection_not_found=True")
        return 0


def _all_chroma_collections_cached(
    artifacts_root: str,
    model_name: str,
    force_reindex: bool,
) -> bool:
    """Returns True when every field collection has docs and force_reindex is off."""
    if force_reindex:
        return False
    return all(
        _chroma_collection_doc_count(artifacts_root, model_name, fc.field_key) > 0
        for fc in VALIDATION_FIELD_CONFIGS
    )


def get_validation_index_status(
    artifacts_root: str,
    retrieval_model: str,
) -> dict[str, int]:
    """Returns Chroma document counts for validation fields.

    Args:
      artifacts_root: Root folder that stores Chroma validation artifacts.
      retrieval_model: Embedding model name.

    Returns:
      Mapping of field key to Chroma document count.
    """
    index_status: dict[str, int] = {}
    for field_config in VALIDATION_FIELD_CONFIGS:
        field_key = field_config.field_key
        index_status[field_key] = _chroma_collection_doc_count(
            artifacts_root=artifacts_root,
            model_name=retrieval_model,
            field_key=field_key,
        )
    log.info(f"{retrieval_model=}, {index_status=}")
    return index_status


def _encode_texts_for_indexing(
    texts: Sequence[str],
    model_name: str,
    use_hf_encoder: bool,
    batch_size: int,
    vllm_base_url: str,
    vllm_api_key: str,
    vllm_max_concurrency: int,
    progress_desc: str,
) -> np.ndarray:
    """Encodes corpus texts for Chroma indexing via HF transformers or vLLM."""
    if use_hf_encoder:
        assets = _load_hf_encoder(model_name=model_name)
        return _encode_hf_sentence_embeddings(
            assets=assets, texts=texts, batch_size=batch_size
        )
    client = AsyncOpenAI(base_url=vllm_base_url, api_key=vllm_api_key)
    return asyncio.run(
        _encode_vllm_async(
            client=client,
            embedding_model=model_name,
            texts=texts,
            batch_size=batch_size,
            max_concurrency=vllm_max_concurrency,
            progress_desc=progress_desc,
        )
    )


def _load_chroma_corpus_artifacts(
    artifacts_root: str,
    field_key: str,
    model_name: str,
    corpus: _CorpusPayload,
    force_reindex: bool,
    batch_size: int,
    vllm_base_url: str,
    vllm_api_key: str,
    vllm_max_concurrency: int,
    use_hf_encoder: bool = False,
) -> _ChromaCorpusArtifacts:
    """Loads cached corpus embeddings from Chroma or indexes them once."""
    sanitized_model_name = _sanitize_model_name(model_name=model_name)
    chroma_path = Path(artifacts_root) / "chroma_validation" / sanitized_model_name
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection_name = _build_collection_name(
        field_key=field_key, sanitized_model_name=sanitized_model_name
    )
    collection = client.get_or_create_collection(name=collection_name)
    current_count = collection.count()
    log.info(
        f"{model_name=}, {chroma_path=}, {collection_name=}, "
        f"{current_count=}, {force_reindex=}"
    )

    if current_count > 0 and not force_reindex:
        existing = collection.get(include=["embeddings", "metadatas"], limit=current_count)
        embeddings = np.asarray(existing["embeddings"], dtype=np.float32)
        metadatas = existing.get("metadatas") or []
        names = [str(metadata.get("name", "")) for metadata in metadatas]
        return _ChromaCorpusArtifacts(embeddings=embeddings, names=names)

    if current_count > 0:
        existing_ids = collection.get(include=[], limit=current_count).get("ids", [])
        collection.delete(ids=existing_ids)

    embeddings = _encode_texts_for_indexing(
        texts=corpus.texts,
        model_name=model_name,
        use_hf_encoder=use_hf_encoder,
        batch_size=batch_size,
        vllm_base_url=vllm_base_url,
        vllm_api_key=vllm_api_key,
        vllm_max_concurrency=vllm_max_concurrency,
        progress_desc=f"Chroma corpus embeddings ({field_key})",
    )
    ids = [f"{field_key}_{index}" for index in range(len(corpus.texts))]
    metadatas = [{"name": name} for name in corpus.names]
    total_docs = len(ids)
    for batch_start in range(0, total_docs, _CHROMA_ADD_BATCH_SIZE):
        batch_end = min(batch_start + _CHROMA_ADD_BATCH_SIZE, total_docs)
        log.info(
            f"Chroma collection.add batch {batch_start}-{batch_end} of {total_docs}, "
            f"{_CHROMA_ADD_BATCH_SIZE=}"
        )
        collection.add(
            ids=ids[batch_start:batch_end],
            documents=list(corpus.texts[batch_start:batch_end]),
            embeddings=embeddings[batch_start:batch_end].tolist(),
            metadatas=metadatas[batch_start:batch_end],
        )
    return _ChromaCorpusArtifacts(embeddings=embeddings, names=corpus.names)


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


def _log_scalars_to_wandb_summary(payload: dict[str, float]) -> None:
    """Logs scalar metrics directly to W&B run summary for cross-run bar plots."""
    wandb.run.summary.update(payload)
    log.info(f"{payload=}")


def evaluate_validation_parquet(
    validation_parquet: str = "artifacts/val.parquet",
    retrieval_model: str = "Qwen/Qwen3-Embedding-0.6B",
    force_reindex: bool = False,
    artifacts_root: str = "artifacts",
    vllm_base_url: str = "",
    vllm_api_key: str = "EMPTY",
    vllm_batch_size: int = 512,
    vllm_max_concurrency: int = 32,
    start_vllm_server: bool = True,
    vllm_gpu_device: int = 3,
    vllm_port: int = 8002,
    vllm_gpu_memory_utilization: float = 0.9,
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name: str = "validation-parquet-eval",
    sentence_batch_size: int = 64,
    max_validation_rows: int = 0,
    use_hf_encoder: bool = False,
    include_bm25: bool = True,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluates summary/description retrieval on validation parquet.

    Runs the given retrieval_model (vLLM or HuggingFace) across both
    retrieval fields (summary and description). BM25 is optional.
    """
    if not vllm_base_url:
        vllm_base_url = f"http://127.0.0.1:{vllm_port}/v1"
    vllm_process = _maybe_start_vllm_server(
        model_name=retrieval_model,
        vllm_port=vllm_port,
        vllm_gpu_device=vllm_gpu_device,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        start_vllm_server=start_vllm_server,
        vllm_base_url=vllm_base_url,
        vllm_api_key=vllm_api_key,
    )
    validation_payload = _load_validation_payload(validation_parquet=validation_parquet)
    validation_payload = _slice_validation_payload(
        payload=validation_payload,
        max_validation_rows=max_validation_rows,
    )
    metrics_by_field: dict[str, dict[str, RetrievalMetrics]] = {}

    wandb.init(
        project=wandb_project,
        entity=wandb_entity or None,
        name=run_name,
        config=_build_validation_wandb_config(
            validation_parquet=validation_parquet,
            run_name=run_name,
            sentence_batch_size=sentence_batch_size,
            retrieval_model=retrieval_model,
            force_reindex=force_reindex,
            max_validation_rows=max_validation_rows,
        ),
    )

    if use_hf_encoder:
        hf_assets = _load_hf_encoder(model_name=retrieval_model)
        query_embeddings = _encode_hf_sentence_embeddings(
            assets=hf_assets,
            texts=validation_payload.query_texts,
            batch_size=sentence_batch_size,
        )
    else:
        query_client = AsyncOpenAI(base_url=vllm_base_url, api_key=vllm_api_key)
        query_embeddings = asyncio.run(
            _encode_vllm_async(
                client=query_client,
                embedding_model=retrieval_model,
                texts=validation_payload.query_texts,
                batch_size=vllm_batch_size,
                max_concurrency=vllm_max_concurrency,
                progress_desc="Validation query embeddings",
            )
        )

    try:
        for field_config in VALIDATION_FIELD_CONFIGS:
            field_key = field_config.field_key
            corpus = _build_validation_corpus(
                rows=validation_payload.rows,
                field_key=field_key,
            )
            metrics_by_model: dict[str, RetrievalMetrics] = {}

            chroma_artifacts = _load_chroma_corpus_artifacts(
                artifacts_root=artifacts_root,
                field_key=field_key,
                model_name=retrieval_model,
                corpus=corpus,
                force_reindex=force_reindex,
                batch_size=vllm_batch_size,
                vllm_base_url=vllm_base_url,
                vllm_api_key=vllm_api_key,
                vllm_max_concurrency=vllm_max_concurrency,
                use_hf_encoder=use_hf_encoder,
            )
            dense_score_matrix = _score_bi_encoder(
                query_embeddings=query_embeddings,
                doc_embeddings=chroma_artifacts.embeddings,
            )
            dense_metrics = _compute_metrics(
                score_matrix=dense_score_matrix,
                expected_names=validation_payload.expected_names,
                corpus_names=chroma_artifacts.names,
            )
            metrics_by_model[_sanitize_model_name(retrieval_model)] = dense_metrics

            if include_bm25:
                bm25_cache = _load_or_write_bm25_docs(
                    artifacts_root=artifacts_root,
                    field_key=field_key,
                    corpus=corpus,
                    force_reindex=force_reindex,
                )
                bm25_score_matrix = _score_bm25(
                    query_texts=validation_payload.query_texts,
                    doc_texts=bm25_cache.texts,
                )
                bm25_metrics = _compute_metrics(
                    score_matrix=bm25_score_matrix,
                    expected_names=validation_payload.expected_names,
                    corpus_names=bm25_cache.names,
                )
                metrics_by_model["bm25"] = bm25_metrics
            metrics_by_field[field_key] = metrics_by_model
    finally:
        if vllm_process is not None:
            vllm_process.terminate()
            log.info(f"{vllm_process.pid=}, stopped=True")

    _log_scalars_to_wandb_summary(_build_validation_payload(metrics_by_field))
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


def evaluate_validation_bm25_parquet(
    validation_parquet: str = "artifacts/val.parquet",
    force_reindex: bool = False,
    artifacts_root: str = "artifacts",
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name: str = "validation-parquet-eval-bm25",
    max_validation_rows: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluates BM25 on summary/description retrieval for validation parquet."""
    validation_payload = _load_validation_payload(validation_parquet=validation_parquet)
    validation_payload = _slice_validation_payload(
        payload=validation_payload,
        max_validation_rows=max_validation_rows,
    )
    metrics_by_field: dict[str, dict[str, RetrievalMetrics]] = {}

    wandb.init(
        project=wandb_project,
        entity=wandb_entity or None,
        name=run_name,
        config=_build_validation_wandb_config(
            validation_parquet=validation_parquet,
            run_name=run_name,
            sentence_batch_size=0,
            retrieval_model="bm25",
            force_reindex=force_reindex,
            max_validation_rows=max_validation_rows,
        ),
    )

    for field_config in VALIDATION_FIELD_CONFIGS:
        field_key = field_config.field_key
        corpus = _build_validation_corpus(
            rows=validation_payload.rows,
            field_key=field_key,
        )
        bm25_cache = _load_or_write_bm25_docs(
            artifacts_root=artifacts_root,
            field_key=field_key,
            corpus=corpus,
            force_reindex=force_reindex,
        )
        bm25_score_matrix = _score_bm25(
            query_texts=validation_payload.query_texts,
            doc_texts=bm25_cache.texts,
        )
        bm25_metrics = _compute_metrics(
            score_matrix=bm25_score_matrix,
            expected_names=validation_payload.expected_names,
            corpus_names=bm25_cache.names,
        )
        metrics_by_field[field_key] = {"bm25": bm25_metrics}

    _log_scalars_to_wandb_summary(_build_validation_payload(metrics_by_field))
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
    output = _build_validation_output(metrics_by_field)
    log.info(f"{output=}")
    return output


def build_validation_indexes(
    validation_parquet: str = "artifacts/val.parquet",
    retrieval_model: str = "Qwen/Qwen3-Embedding-0.6B",
    force_reindex: bool = False,
    artifacts_root: str = "artifacts",
    vllm_base_url: str = "",
    vllm_api_key: str = "EMPTY",
    vllm_batch_size: int = 512,
    vllm_max_concurrency: int = 32,
    start_vllm_server: bool = True,
    vllm_gpu_device: int = 3,
    vllm_port: int = 8002,
    vllm_gpu_memory_utilization: float = 0.9,
    use_hf_encoder: bool = False,
) -> dict[str, dict[str, int]]:
    """Builds and caches Chroma and BM25 artifacts for validation retrieval."""
    if not vllm_base_url:
        vllm_base_url = f"http://127.0.0.1:{vllm_port}/v1"
    validation_payload = _load_validation_payload(validation_parquet=validation_parquet)
    output: dict[str, dict[str, int]] = {}
    all_cached = _all_chroma_collections_cached(
        artifacts_root=artifacts_root,
        model_name=retrieval_model,
        force_reindex=force_reindex,
    )
    log.info(f"{retrieval_model=}, {all_cached=}, {force_reindex=}")

    if all_cached:
        for field_config in VALIDATION_FIELD_CONFIGS:
            field_key = field_config.field_key
            corpus = _build_validation_corpus(
                rows=validation_payload.rows, field_key=field_key
            )
            bm25_cache = _load_or_write_bm25_docs(
                artifacts_root=artifacts_root,
                field_key=field_key,
                corpus=corpus,
                force_reindex=force_reindex,
            )
            output[field_key] = {
                "chroma_docs": _chroma_collection_doc_count(
                    artifacts_root, retrieval_model, field_key
                ),
                "bm25_docs": len(bm25_cache.texts),
            }
        log.info(f"{output=}")
        return output

    vllm_process = _maybe_start_vllm_server(
        model_name=retrieval_model,
        vllm_port=vllm_port,
        vllm_gpu_device=vllm_gpu_device,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        start_vllm_server=start_vllm_server,
        vllm_base_url=vllm_base_url,
        vllm_api_key=vllm_api_key,
    )
    try:
        for field_config in VALIDATION_FIELD_CONFIGS:
            field_key = field_config.field_key
            corpus = _build_validation_corpus(
                rows=validation_payload.rows,
                field_key=field_key,
            )
            chroma_artifacts = _load_chroma_corpus_artifacts(
                artifacts_root=artifacts_root,
                field_key=field_key,
                model_name=retrieval_model,
                corpus=corpus,
                force_reindex=force_reindex,
                batch_size=vllm_batch_size,
                vllm_base_url=vllm_base_url,
                vllm_api_key=vllm_api_key,
                vllm_max_concurrency=vllm_max_concurrency,
                use_hf_encoder=use_hf_encoder,
            )
            bm25_cache = _load_or_write_bm25_docs(
                artifacts_root=artifacts_root,
                field_key=field_key,
                corpus=corpus,
                force_reindex=force_reindex,
            )
            output[field_key] = {
                "chroma_docs": int(chroma_artifacts.embeddings.shape[0]),
                "bm25_docs": len(bm25_cache.texts),
            }
    finally:
        if vllm_process is not None:
            vllm_process.terminate()
            log.info(f"{vllm_process.pid=}, stopped=True")

    log.info(f"{output=}")
    return output


async def evaluate_validated_skill_questions_async(
    input_jsonl: str,
    output_train_parquet: str = "artifacts/retriever_training/train.parquet",
    output_validation_parquet: str = "artifacts/retriever_training/validation.parquet",
    train_questions_per_skill: int = 2,
    validation_questions_per_skill: int = 1,
    vllm_port: int = 8002,
    vllm_base_url: str = "",
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
    if not vllm_base_url:
        vllm_base_url = f"http://127.0.0.1:{vllm_port}/v1"
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
    vllm_port: int = 8002,
    vllm_base_url: str = "",
    vllm_api_key: str = "EMPTY",
    vllm_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    vllm_batch_size: int = 64,
    vllm_max_concurrency: int = 8,
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name: str = "validated-eval",
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
            vllm_port=vllm_port,
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
            "build_validation_indexes": build_validation_indexes,
            "evaluate_validated_skill_questions": evaluate_validated_skill_questions,
        }
    )


if __name__ == "__main__":
    main()
