"""MMR-based diverse question selection for validated retriever data."""

from __future__ import annotations

import asyncio
from typing import NamedTuple, Sequence

import numpy as np
from loguru import logger as log
from openai import AsyncOpenAI

from ast_skills.retriever.datamodels import ValidatedSkillQuestionRow


class MmrSelectionConfig(NamedTuple):
    """Configuration for vLLM embedding-based MMR selection."""

    base_url: str
    api_key: str
    embedding_model: str
    mmr_lambda: float
    selected_question_count: int
    batch_size: int
    max_concurrency: int


class _TextChunk(NamedTuple):
    """Chunk of texts and start index for async embedding calls."""

    start_index: int
    texts: list[str]


def _chunk_texts(texts: Sequence[str], batch_size: int) -> list[_TextChunk]:
    """Splits texts into fixed-size chunks."""
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
    """Fetches one embedding chunk using concurrency control."""
    async with semaphore:
        response = await client.embeddings.create(
            model=embedding_model,
            input=text_chunk.texts,
        )
    vectors = [item.embedding for item in response.data]
    return text_chunk.start_index, vectors


async def _encode_texts_async(
    client: AsyncOpenAI,
    embedding_model: str,
    texts: Sequence[str],
    batch_size: int,
    max_concurrency: int,
) -> np.ndarray:
    """Encodes texts with async vLLM embeddings and L2 normalization."""
    chunks = _chunk_texts(texts=texts, batch_size=batch_size)
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _fetch_embeddings_chunk(
            client=client,
            embedding_model=embedding_model,
            text_chunk=chunk,
            semaphore=semaphore,
        )
        for chunk in chunks
    ]
    chunk_results = await asyncio.gather(*tasks)
    ordered_results = sorted(chunk_results, key=lambda item: item[0])
    vectors: list[list[float]] = []
    for _, chunk_vectors in ordered_results:
        vectors.extend(chunk_vectors)

    embeddings = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    log.info(f"{normalized.shape=}")
    return normalized


def _compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Computes cosine similarity matrix."""
    similarity_matrix = np.matmul(embeddings, embeddings.T)
    log.info(f"{similarity_matrix.shape=}")
    return similarity_matrix


def _select_mmr_indices(
    similarity_matrix: np.ndarray,
    selected_question_count: int,
    mmr_lambda: float,
) -> list[int]:
    """Selects diverse question indices using MMR."""
    candidate_count = similarity_matrix.shape[0]
    if candidate_count < selected_question_count:
        raise ValueError(
            f"Insufficient candidates for MMR, {candidate_count=}, {selected_question_count=}"
        )

    selected_indices: list[int] = []
    relevance_scores = np.mean(similarity_matrix, axis=1)
    first_index = int(np.argmax(relevance_scores))
    selected_indices.append(first_index)

    while len(selected_indices) < selected_question_count:
        best_score = float("-inf")
        best_index = -1
        for candidate_index in range(candidate_count):
            if candidate_index in selected_indices:
                continue
            relevance = relevance_scores[candidate_index]
            diversity_penalty = float(
                np.max(similarity_matrix[candidate_index, selected_indices])
            )
            mmr_score = (mmr_lambda * relevance) - (
                (1.0 - mmr_lambda) * diversity_penalty
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = candidate_index
        selected_indices.append(best_index)

    log.info(f"{selected_indices=}, {selected_question_count=}, {mmr_lambda=}")
    return selected_indices


def _normalize_questions(questions: Sequence[str]) -> list[str]:
    """Normalizes and removes empty questions."""
    normalized = [question.strip() for question in questions if question.strip()]
    log.info(f"{len(questions)=}, {len(normalized)=}")
    return normalized


def _build_selected_questions(
    questions: Sequence[str],
    selected_indices: Sequence[int],
) -> list[str]:
    """Builds ordered selected questions from chosen indices."""
    selected_questions = [questions[index] for index in selected_indices]
    log.info(f"{selected_questions=}")
    return selected_questions


async def _select_row_questions_async(
    client: AsyncOpenAI,
    row: ValidatedSkillQuestionRow,
    config: MmrSelectionConfig,
) -> tuple[str, list[str]]:
    """Selects diverse questions for one validated row."""
    normalized_questions = _normalize_questions(row.filtered_questions)
    embeddings = await _encode_texts_async(
        client=client,
        embedding_model=config.embedding_model,
        texts=normalized_questions,
        batch_size=config.batch_size,
        max_concurrency=config.max_concurrency,
    )
    similarity_matrix = _compute_similarity_matrix(embeddings=embeddings)
    selected_indices = _select_mmr_indices(
        similarity_matrix=similarity_matrix,
        selected_question_count=config.selected_question_count,
        mmr_lambda=config.mmr_lambda,
    )
    selected_questions = _build_selected_questions(
        questions=normalized_questions,
        selected_indices=selected_indices,
    )
    return row.custom_id, selected_questions


async def select_diverse_questions_for_rows(
    rows: Sequence[ValidatedSkillQuestionRow],
    config: MmrSelectionConfig,
) -> dict[str, list[str]]:
    """Selects diverse questions for all rows via async vLLM embeddings."""
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    tasks = [
        _select_row_questions_async(
            client=client,
            row=row,
            config=config,
        )
        for row in rows
    ]
    task_results = await asyncio.gather(*tasks)
    selected_by_custom_id = dict(task_results)
    log.info(f"{len(selected_by_custom_id)=}")
    return selected_by_custom_id
