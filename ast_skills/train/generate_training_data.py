"""Generate hard-negative training parquet from ``train.parquet``.

Pipeline stages:
  1. Load parquet rows into ``DataPoint`` dataclasses.
  2. Build/refresh Chroma collections for ``summary`` and ``description`` with
     async OpenAI-compatible embeddings.
  3. For each anchor question, run dense + hybrid retrieval over both fields.
  4. Keep candidates ranked 6..37 as mined negatives (32 max).
  5. Write mined rows to parquet for downstream training/analysis.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import chromadb
import fire
import pandas as pd
from chromadb.api.models.Collection import Collection
from loguru import logger as log
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

from ast_skills.retriever.bm25_index import tokenize
from ast_skills.train.datamodels import DataPoint, MinedTrainingDataRow

OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV_VAR = "OPENAI_BASE_URL"
OPENAI_MODEL_ENV_VAR = "OPENAI_MODEL"

DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_API_KEY = "EMPTY"

DEFAULT_INPUT_PARQUET = "artifacts/retriever_training/train.parquet"
DEFAULT_OUTPUT_PARQUET = "artifacts/retriever_training/mined_train.parquet"
DEFAULT_CHROMA_ROOT = "artifacts/chroma_train_builder"

DEFAULT_EMBED_BATCH_SIZE = 256
DEFAULT_MAX_CONCURRENCY = 32
DEFAULT_RETRIEVAL_TOP_K = 37
DEFAULT_DROP_TOP_K = 5
DEFAULT_KEEP_NEGATIVES = 32
DEFAULT_RRF_K = 60


_T = TypeVar("_T")


class _FieldArtifacts(NamedTuple):
    """Artifacts required to retrieve candidates for one text field."""

    field_name: str
    collection: Collection
    bm25_model: BM25Okapi
    doc_ids: list[str]
    doc_text_by_id: dict[str, str]


class _MinedNegatives(NamedTuple):
    """Resolved negatives for one anchor and one field."""

    negative_ids: list[str]
    negative_texts: list[str]


def _build_async_openai_client(api_key: str, base_url: str) -> AsyncOpenAI:
    """Creates an async OpenAI-compatible client."""
    log.info(f"{api_key=}, {base_url=}")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _datapoint_from_record(record: dict[str, Any]) -> DataPoint:
    """Converts one parquet record to ``DataPoint`` with strict required keys."""
    required_fields = ["name", "markdown_content", "summary", "description", "question"]
    for field_name in required_fields:
        if field_name not in record:
            raise KeyError(f"Missing required field in parquet row: {field_name}")
    return DataPoint(
        name=str(record["name"]),
        markdown_content=str(record["markdown_content"]),
        summary=str(record["summary"]),
        description=str(record["description"]),
        question=str(record["question"]),
    )


def load_datapoints_from_parquet(input_parquet: str) -> list[DataPoint]:
    """Loads ``DataPoint`` rows from parquet."""
    input_path = Path(input_parquet)
    dataframe = pd.read_parquet(input_path)
    records = dataframe.to_dict(orient="records")
    datapoints = [_datapoint_from_record(record) for record in records]
    log.info(f"{input_path=}, {len(datapoints)=}")
    return datapoints


def _chunked(items: list[_T], batch_size: int) -> list[list[_T]]:
    """Splits list into fixed-size batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


async def _embed_text_batch(
    client: AsyncOpenAI,
    model_name: str,
    texts: list[str],
    semaphore: asyncio.Semaphore,
) -> list[list[float]]:
    """Embeds one text batch through async OpenAI-compatible API."""
    async with semaphore:
        response = await client.embeddings.create(model=model_name, input=texts)
    embeddings = [item.embedding for item in response.data]
    log.info(f"{len(texts)=}, {len(embeddings)=}")
    return embeddings


async def embed_texts_async(
    client: AsyncOpenAI,
    model_name: str,
    texts: list[str],
    batch_size: int,
    max_concurrency: int,
) -> list[list[float]]:
    """Embeds many texts asynchronously while preserving order."""
    if not texts:
        return []
    batches = _chunked(texts, batch_size)
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        asyncio.create_task(
            _embed_text_batch(
                client=client,
                model_name=model_name,
                texts=batch,
                semaphore=semaphore,
            )
        )
        for batch in batches
    ]
    batch_embeddings = await asyncio.gather(*tasks)
    flattened = [embedding for batch in batch_embeddings for embedding in batch]
    log.info(f"{len(texts)=}, {batch_size=}, {max_concurrency=}, {len(flattened)=}")
    return flattened


def _collection_name_for_field(field_name: str) -> str:
    """Returns deterministic collection name for this builder."""
    return f"train_builder_{field_name}"


def _db_dir_name_for_field(field_name: str) -> str:
    """Returns deterministic db directory for this builder."""
    return f"{field_name}_db"


def _refresh_collection(
    chroma_root: Path,
    field_name: str,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
) -> Collection:
    """Creates/refreshes one Chroma collection with provided vectors."""
    db_path = chroma_root / _db_dir_name_for_field(field_name)
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    collection_name = _collection_name_for_field(field_name)
    try:
        client.delete_collection(name=collection_name)
    except Exception as exc:  # pylint: disable=broad-except
        log.info(f"{field_name=}, {collection_name=}, delete_collection skipped, {exc=}")

    collection = client.get_or_create_collection(name=collection_name)
    collection.add(ids=ids, documents=documents, embeddings=embeddings)
    count = collection.count()
    log.info(f"{field_name=}, {db_path=}, {count=}")
    return collection


def _build_bm25_model(documents: list[str]) -> BM25Okapi:
    """Builds BM25 model for one field corpus."""
    tokenized_documents = [tokenize(document) for document in documents]
    model = BM25Okapi(tokenized_documents)
    log.info(f"{len(documents)=}, {len(tokenized_documents)=}")
    return model


def _build_field_artifacts(
    datapoints: list[DataPoint],
    field_name: str,
    collection: Collection,
) -> _FieldArtifacts:
    """Builds in-memory lookup artifacts for one field."""
    doc_ids = [str(index) for index in range(len(datapoints))]
    doc_texts = [str(getattr(datapoint, field_name)) for datapoint in datapoints]
    doc_text_by_id = {doc_id: text for doc_id, text in zip(doc_ids, doc_texts)}
    bm25_model = _build_bm25_model(doc_texts)
    return _FieldArtifacts(
        field_name=field_name,
        collection=collection,
        bm25_model=bm25_model,
        doc_ids=doc_ids,
        doc_text_by_id=doc_text_by_id,
    )


def _reciprocal_rank_fusion(
    dense_ranked_ids: list[str],
    sparse_ranked_ids: list[str],
    rrf_k: int,
) -> list[str]:
    """Fuses dense and sparse rankings with reciprocal-rank fusion."""
    score_by_id: dict[str, float] = {}

    for rank, doc_id in enumerate(dense_ranked_ids, start=1):
        score_by_id[doc_id] = score_by_id.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    for rank, doc_id in enumerate(sparse_ranked_ids, start=1):
        score_by_id[doc_id] = score_by_id.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    sorted_ids = sorted(score_by_id, key=lambda doc_id: score_by_id[doc_id], reverse=True)
    return sorted_ids


def _dense_ranked_ids(
    artifacts: _FieldArtifacts,
    query_embedding: list[float],
    limit: int,
) -> list[str]:
    """Runs dense retrieval against Chroma and returns ids ordered by rank."""
    result = artifacts.collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        include=["distances"],
    )
    ids = result.get("ids", [[]])[0]
    dense_ids = [str(row_id) for row_id in ids]
    return dense_ids


def _sparse_ranked_ids(
    artifacts: _FieldArtifacts,
    query: str,
    limit: int,
) -> list[str]:
    """Runs sparse BM25 retrieval and returns ids ordered by rank."""
    tokenized_query = tokenize(query)
    scores = artifacts.bm25_model.get_scores(tokenized_query)
    sorted_indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )[:limit]
    return [artifacts.doc_ids[index] for index in sorted_indices]


def _mine_field_negatives(
    artifacts: _FieldArtifacts,
    anchor_id: str,
    query: str,
    query_embedding: list[float],
    retrieval_top_k: int,
    drop_top_k: int,
    keep_negatives: int,
    rrf_k: int,
) -> _MinedNegatives:
    """Runs hybrid retrieval and selects hard negatives for one field."""
    dense_ids = _dense_ranked_ids(
        artifacts=artifacts,
        query_embedding=query_embedding,
        limit=retrieval_top_k,
    )
    sparse_ids = _sparse_ranked_ids(
        artifacts=artifacts,
        query=query,
        limit=retrieval_top_k,
    )
    fused_ids = _reciprocal_rank_fusion(
        dense_ranked_ids=dense_ids,
        sparse_ranked_ids=sparse_ids,
        rrf_k=rrf_k,
    )

    without_anchor = [doc_id for doc_id in fused_ids if doc_id != anchor_id]
    window_ids = without_anchor[drop_top_k : drop_top_k + keep_negatives]

    negative_texts: list[str] = []
    for doc_id in window_ids:
        text = artifacts.doc_text_by_id.get(doc_id, "").strip()
        if text:
            negative_texts.append(text)

    log.info(
        f"{artifacts.field_name=}, {anchor_id=}, {len(dense_ids)=}, "
        f"{len(sparse_ids)=}, {len(fused_ids)=}, {len(window_ids)=}, {len(negative_texts)=}"
    )
    return _MinedNegatives(negative_ids=window_ids, negative_texts=negative_texts)


def _build_mined_row(
    anchor_id: str,
    datapoint: DataPoint,
    summary_negatives: _MinedNegatives,
    description_negatives: _MinedNegatives,
    include_negative_descriptions: bool,
) -> MinedTrainingDataRow:
    """Builds one mined training row from anchor and retrieved negatives."""
    negative_descriptions = None
    if include_negative_descriptions:
        negative_descriptions = description_negatives.negative_texts

    return MinedTrainingDataRow(
        anchor_id=anchor_id,
        name=datapoint.name,
        markdown_content=datapoint.markdown_content,
        summary=datapoint.summary,
        description=datapoint.description,
        question=datapoint.question,
        negative_summaries=summary_negatives.negative_texts,
        negative_descriptions=negative_descriptions,
    )


def _write_mined_parquet(path: Path, rows: list[MinedTrainingDataRow]) -> None:
    """Writes mined rows to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([row.__dict__ for row in rows])
    dataframe.to_parquet(path, index=False)
    log.info(f"{path=}, {len(rows)=}")


async def _build_collections_async(
    datapoints: list[DataPoint],
    chroma_root: Path,
    client: AsyncOpenAI,
    embedding_model: str,
    embedding_batch_size: int,
    max_concurrency: int,
) -> dict[str, _FieldArtifacts]:
    """Embeds corpora and refreshes Chroma collections for summary/description."""
    field_names = ["summary", "description"]
    ids = [str(index) for index in range(len(datapoints))]
    artifacts_by_field: dict[str, _FieldArtifacts] = {}

    for field_name in field_names:
        documents = [str(getattr(datapoint, field_name)) for datapoint in datapoints]
        embeddings = await embed_texts_async(
            client=client,
            model_name=embedding_model,
            texts=documents,
            batch_size=embedding_batch_size,
            max_concurrency=max_concurrency,
        )
        collection = _refresh_collection(
            chroma_root=chroma_root,
            field_name=field_name,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
        )
        artifacts_by_field[field_name] = _build_field_artifacts(
            datapoints=datapoints,
            field_name=field_name,
            collection=collection,
        )

    log.info(f"{list(artifacts_by_field)=}")
    return artifacts_by_field


async def _generate_training_data_async(
    input_parquet: str,
    output_parquet: str,
    chroma_root_dir: str,
    embedding_model: str,
    embedding_base_url: str,
    api_key: str,
    embedding_batch_size: int,
    max_concurrency: int,
    retrieval_top_k: int,
    drop_top_k: int,
    keep_negatives: int,
    rrf_k: int,
    include_negative_descriptions: bool,
) -> None:
    """Async implementation for generating training data parquet."""
    if retrieval_top_k < 1:
        raise ValueError("retrieval_top_k must be >= 1")
    if drop_top_k < 0:
        raise ValueError("drop_top_k must be >= 0")
    if keep_negatives < 1:
        raise ValueError("keep_negatives must be >= 1")

    datapoints = load_datapoints_from_parquet(input_parquet=input_parquet)
    if not datapoints:
        raise ValueError("No datapoints found in input parquet.")

    client = _build_async_openai_client(api_key=api_key, base_url=embedding_base_url)
    chroma_root = Path(chroma_root_dir)

    artifacts_by_field = await _build_collections_async(
        datapoints=datapoints,
        chroma_root=chroma_root,
        client=client,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        max_concurrency=max_concurrency,
    )

    questions = [datapoint.question for datapoint in datapoints]
    question_embeddings = await embed_texts_async(
        client=client,
        model_name=embedding_model,
        texts=questions,
        batch_size=embedding_batch_size,
        max_concurrency=max_concurrency,
    )

    summary_artifacts = artifacts_by_field["summary"]
    description_artifacts = artifacts_by_field["description"]

    mined_rows: list[MinedTrainingDataRow] = []
    for row_index, datapoint in enumerate(datapoints):
        anchor_id = str(row_index)
        query_embedding = question_embeddings[row_index]

        summary_negatives = _mine_field_negatives(
            artifacts=summary_artifacts,
            anchor_id=anchor_id,
            query=datapoint.question,
            query_embedding=query_embedding,
            retrieval_top_k=retrieval_top_k,
            drop_top_k=drop_top_k,
            keep_negatives=keep_negatives,
            rrf_k=rrf_k,
        )
        description_negatives = _mine_field_negatives(
            artifacts=description_artifacts,
            anchor_id=anchor_id,
            query=datapoint.question,
            query_embedding=query_embedding,
            retrieval_top_k=retrieval_top_k,
            drop_top_k=drop_top_k,
            keep_negatives=keep_negatives,
            rrf_k=rrf_k,
        )

        mined_rows.append(
            _build_mined_row(
                anchor_id=anchor_id,
                datapoint=datapoint,
                summary_negatives=summary_negatives,
                description_negatives=description_negatives,
                include_negative_descriptions=include_negative_descriptions,
            )
        )

    _write_mined_parquet(path=Path(output_parquet), rows=mined_rows)


def generate_training_data(
    input_parquet: str = DEFAULT_INPUT_PARQUET,
    output_parquet: str = DEFAULT_OUTPUT_PARQUET,
    chroma_root_dir: str = DEFAULT_CHROMA_ROOT,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_base_url: str = os.environ.get(OPENAI_BASE_URL_ENV_VAR, DEFAULT_OPENAI_BASE_URL),
    api_key: str = os.environ.get(OPENAI_API_KEY_ENV_VAR, DEFAULT_API_KEY),
    embedding_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    drop_top_k: int = DEFAULT_DROP_TOP_K,
    keep_negatives: int = DEFAULT_KEEP_NEGATIVES,
    rrf_k: int = DEFAULT_RRF_K,
    include_negative_descriptions: bool = False,
) -> None:
    """Generates mined training parquet from train parquet input.

    Args:
      input_parquet: Input ``train.parquet`` path containing ``DataPoint`` columns.
      output_parquet: Output parquet for mined negatives.
      chroma_root_dir: Root folder where summary/description Chroma DBs are written.
      embedding_model: Embedding model served by OpenAI-compatible endpoint.
      embedding_base_url: Base URL for vLLM/OpenAI-compatible embeddings API.
      api_key: API key sent to the embedding endpoint.
      embedding_batch_size: Per-request text batch size for embedding calls.
      max_concurrency: Maximum in-flight embedding requests.
      retrieval_top_k: Number of top candidates to retrieve before windowing.
      drop_top_k: Number of leading candidates to drop.
      keep_negatives: Number of negatives to keep after dropping top candidates.
      rrf_k: Reciprocal-rank-fusion constant.
      include_negative_descriptions: Whether to persist description negatives.
    """
    resolved_embedding_model = os.environ.get(OPENAI_MODEL_ENV_VAR, embedding_model)
    log.info(
        f"{input_parquet=}, {output_parquet=}, {chroma_root_dir=}, "
        f"{resolved_embedding_model=}, {embedding_base_url=}, {embedding_batch_size=}, "
        f"{max_concurrency=}, {retrieval_top_k=}, {drop_top_k=}, {keep_negatives=}, "
        f"{rrf_k=}, {include_negative_descriptions=}"
    )
    asyncio.run(
        _generate_training_data_async(
            input_parquet=input_parquet,
            output_parquet=output_parquet,
            chroma_root_dir=chroma_root_dir,
            embedding_model=resolved_embedding_model,
            embedding_base_url=embedding_base_url,
            api_key=api_key,
            embedding_batch_size=embedding_batch_size,
            max_concurrency=max_concurrency,
            retrieval_top_k=retrieval_top_k,
            drop_top_k=drop_top_k,
            keep_negatives=keep_negatives,
            rrf_k=rrf_k,
            include_negative_descriptions=include_negative_descriptions,
        )
    )


if __name__ == "__main__":
    fire.Fire(generate_training_data)
