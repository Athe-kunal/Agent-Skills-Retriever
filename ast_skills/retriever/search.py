"""Retriever search utilities for semantic, sparse, and hybrid ranking."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

import chromadb
import fire
from loguru import logger as log
from openai import OpenAI
from rank_bm25 import BM25Okapi

from ast_skills.retriever.bm25_index import bm25_search, load_bm25_artifacts


class _FieldSearchConfig(NamedTuple):
    """Field-specific storage config."""

    field_name: str
    db_dir_name: str
    collection_name: str


class _ScoredId(NamedTuple):
    """Identifier and score pair."""

    doc_id: str
    score: float


class _SearchArtifacts(NamedTuple):
    """Loaded resources required for searching one field."""

    collection: Any
    bm25_path: Path


class _CachedBm25(NamedTuple):
    """Cached sparse index and BM25 model."""

    index: Any
    model: BM25Okapi


FIELD_CONFIGS: dict[str, _FieldSearchConfig] = {
    "summary": _FieldSearchConfig("summary", "summary_db", "retriever_summary"),
    "description": _FieldSearchConfig(
        "description", "description_db", "retriever_description"
    ),
}

_ARTIFACTS_CACHE: dict[tuple[str, str], _SearchArtifacts] = {}
_ARTIFACTS_CACHE_LOCK = threading.Lock()
_BM25_CACHE: dict[str, _CachedBm25] = {}
_BM25_CACHE_LOCK = threading.Lock()


def _validate_field(field: str) -> str:
    """Validates the requested retrieval field."""
    normalized_field = field.strip().lower()
    if normalized_field in FIELD_CONFIGS:
        return normalized_field

    supported_fields = sorted(FIELD_CONFIGS)
    raise ValueError(
        f"Unsupported field={field!r}. Supported fields: {supported_fields}"
    )


def _make_embedding_client(base_url: str, api_key: str) -> OpenAI:
    """Creates OpenAI-compatible client."""
    log.info(f"{base_url=}")
    return OpenAI(base_url=base_url, api_key=api_key)


def _load_artifacts_uncached(root_dir: Path, field: str) -> _SearchArtifacts:
    """Loads artifacts without cache for one field."""
    config = FIELD_CONFIGS[field]
    db_path = root_dir / config.db_dir_name
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=config.collection_name)
    bm25_path = root_dir / "bm25" / f"{field}.json"
    log.info(f"{db_path=}, {bm25_path=}")
    return _SearchArtifacts(collection=collection, bm25_path=bm25_path)


def _load_artifacts_with_retry(
    root_dir: Path,
    field: str,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.2,
) -> _SearchArtifacts:
    """Loads artifacts with retries for transient Chroma tenant connection errors."""
    last_exception: ValueError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _load_artifacts_uncached(root_dir=root_dir, field=field)
        except ValueError as error:
            error_text = str(error)
            is_tenant_error = "Could not connect to tenant" in error_text
            if not is_tenant_error:
                raise
            last_exception = error
            log.warning(f"{field=}, {attempt=}, {max_attempts=}, {error_text=}")
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)

    if last_exception is None:
        raise RuntimeError("Expected tenant retry exception but none was captured.")
    raise last_exception


def _load_artifacts(root_dir: Path, field: str) -> _SearchArtifacts:
    """Loads Chroma collection and BM25 payload path for a field with cache."""
    cache_key = (str(root_dir.resolve()), field)
    with _ARTIFACTS_CACHE_LOCK:
        cached_artifacts = _ARTIFACTS_CACHE.get(cache_key)
        if cached_artifacts is not None:
            return cached_artifacts

    loaded_artifacts = _load_artifacts_with_retry(root_dir=root_dir, field=field)
    with _ARTIFACTS_CACHE_LOCK:
        cached_artifacts = _ARTIFACTS_CACHE.get(cache_key)
        if cached_artifacts is not None:
            return cached_artifacts
        _ARTIFACTS_CACHE[cache_key] = loaded_artifacts
    return loaded_artifacts


def _load_bm25_with_cache(bm25_path: Path) -> _CachedBm25:
    """Loads BM25 payload/model with in-process cache to avoid rebuild per query."""
    cache_key = str(bm25_path.resolve())
    with _BM25_CACHE_LOCK:
        cached_bm25 = _BM25_CACHE.get(cache_key)
        if cached_bm25 is not None:
            return cached_bm25

    loaded = load_bm25_artifacts(path=bm25_path)
    cached_bm25 = _CachedBm25(
        index=loaded.index,
        model=loaded.model,
    )
    with _BM25_CACHE_LOCK:
        existing_bm25 = _BM25_CACHE.get(cache_key)
        if existing_bm25 is not None:
            return existing_bm25
        _BM25_CACHE[cache_key] = cached_bm25
    log.info(f"{cache_key=}, {len(cached_bm25.index.ids)=}, loaded_from_cache=False")
    return cached_bm25


def semantic_search(
    query: str,
    field: str = "description",
    root_dir: str = "artifacts/chroma",
    embedding_base_url: str = "http://127.0.0.1:8000/v1",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    api_key: str = "EMPTY",
    limit: int = 10,
) -> str:
    """Runs dense semantic search for summary/description and returns JSON."""
    normalized_field = _validate_field(field=field)
    root_path = Path(root_dir)
    artifacts = _load_artifacts(root_dir=root_path, field=normalized_field)
    embedding_client = _make_embedding_client(base_url=embedding_base_url, api_key=api_key)
    query_response = embedding_client.embeddings.create(model=embedding_model, input=[query])
    query_embedding = query_response.data[0].embedding

    result = artifacts.collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        include=["metadatas", "documents", "distances"],
    )

    output_rows = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for row_index, row_id in enumerate(ids):
        output_rows.append(
            {
                "id": row_id,
                "score": float(1.0 - distances[row_index]),
                "document": documents[row_index],
                "metadata": metadatas[row_index],
                "rank": row_index + 1,
            }
        )

    log.info(f"{normalized_field=}, {len(output_rows)=}")
    return json.dumps(output_rows, ensure_ascii=False, indent=2)


def sparse_search(
    query: str,
    field: str = "description",
    root_dir: str = "artifacts/chroma",
    limit: int = 10,
) -> str:
    """Runs sparse BM25 search for summary/description and returns JSON."""
    normalized_field = _validate_field(field=field)
    root_path = Path(root_dir)
    artifacts = _load_artifacts(root_dir=root_path, field=normalized_field)
    bm25_artifacts = _load_bm25_with_cache(bm25_path=artifacts.bm25_path)
    bm25_result = bm25_search(
        index=bm25_artifacts.index,
        query=query,
        limit=limit,
        model=bm25_artifacts.model,
    )

    metadata_by_id = {
        row_id: bm25_artifacts.index.metadatas[index]
        for index, row_id in enumerate(bm25_artifacts.index.ids)
    }
    document_by_id = {
        row_id: bm25_artifacts.index.documents[index]
        for index, row_id in enumerate(bm25_artifacts.index.ids)
    }

    output_rows = []
    for row_index, row_id in enumerate(bm25_result.ids):
        output_rows.append(
            {
                "id": row_id,
                "score": bm25_result.scores[row_index],
                "document": document_by_id.get(row_id, ""),
                "metadata": metadata_by_id.get(row_id, {}),
                "rank": row_index + 1,
            }
        )

    log.info(f"{normalized_field=}, {len(output_rows)=}")
    return json.dumps(output_rows, ensure_ascii=False, indent=2)


def bm25_search_ids(
    query: str,
    field: str = "description",
    root_dir: str = "artifacts/chroma",
    limit: int = 10,
) -> list[str]:
    """Runs BM25 search and returns ranked ids without loading the ChromaDB collection.

    ChromaDB 1.1+ uses internal multiprocessing queues; loading the collection
    from inside a ``ThreadPoolExecutor`` worker triggers pickle errors on local
    closures. This function bypasses the collection entirely — BM25 only needs
    the JSON index file, not the vector store.
    """
    normalized_field = _validate_field(field=field)
    bm25_path = Path(root_dir) / "bm25" / f"{normalized_field}.json"
    bm25_artifacts = _load_bm25_with_cache(bm25_path=bm25_path)
    result = bm25_search(
        index=bm25_artifacts.index,
        query=query,
        limit=limit,
        model=bm25_artifacts.model,
    )
    log.info(f"{normalized_field=}, {len(result.ids)=}")
    return list(result.ids)


def _rrf(scores: list[_ScoredId], k: int) -> dict[str, float]:
    """Computes reciprocal rank fusion map for one ranked list."""
    fused: dict[str, float] = {}
    for rank, scored in enumerate(scores, start=1):
        fused[scored.doc_id] = 1.0 / (k + rank)
    return fused


def rrf_merge_ids(
    dense_ids: list[str],
    sparse_ids: list[str],
    limit: int,
    rrf_k: int,
) -> list[str]:
    """RRF merge of two pre-ranked id lists; returns top-``limit`` ids by fused score.

    Use this when embeddings and BM25 results have already been computed and
    cached separately — it avoids re-parsing JSON and skips the document/metadata
    fields that are only needed for interactive search responses.
    """
    dense_scored = [_ScoredId(doc_id=d, score=0.0) for d in dense_ids]
    sparse_scored = [_ScoredId(doc_id=s, score=0.0) for s in sparse_ids]
    dense_rrf = _rrf(dense_scored, k=rrf_k)
    sparse_rrf = _rrf(sparse_scored, k=rrf_k)
    doc_ids = set(dense_rrf.keys()) | set(sparse_rrf.keys())
    hybrid_scores = {
        doc_id: dense_rrf.get(doc_id, 0.0) + sparse_rrf.get(doc_id, 0.0)
        for doc_id in doc_ids
    }
    return sorted(doc_ids, key=lambda d: hybrid_scores[d], reverse=True)[:limit]


def _merge_rrf_results(
    dense_json: str,
    sparse_json: str,
    limit: int,
    rrf_k: int,
) -> str:
    """Merges dense and sparse result lists via RRF and returns JSON."""
    dense_rows = json.loads(dense_json)
    sparse_rows = json.loads(sparse_json)
    dense_ranked = [
        _ScoredId(doc_id=row["id"], score=float(row["score"])) for row in dense_rows
    ]
    sparse_ranked = [
        _ScoredId(doc_id=row["id"], score=float(row["score"])) for row in sparse_rows
    ]

    dense_rrf = _rrf(dense_ranked, k=rrf_k)
    sparse_rrf = _rrf(sparse_ranked, k=rrf_k)

    doc_ids = set(dense_rrf.keys()) | set(sparse_rrf.keys())
    hybrid_scores = {
        doc_id: dense_rrf.get(doc_id, 0.0) + sparse_rrf.get(doc_id, 0.0)
        for doc_id in doc_ids
    }
    sorted_ids = sorted(
        hybrid_scores.keys(),
        key=lambda doc_id: hybrid_scores[doc_id],
        reverse=True,
    )

    dense_rows_by_id = {row["id"]: row for row in dense_rows}
    sparse_rows_by_id = {row["id"]: row for row in sparse_rows}
    output_rows = []
    for rank, doc_id in enumerate(sorted_ids[:limit], start=1):
        dense_row = dense_rows_by_id.get(doc_id, {})
        sparse_row = sparse_rows_by_id.get(doc_id, {})
        output_rows.append(
            {
                "id": doc_id,
                "rrf_score": hybrid_scores[doc_id],
                "semantic_score": dense_row.get("score"),
                "sparse_score": sparse_row.get("score"),
                "document": dense_row.get("document") or sparse_row.get("document", ""),
                "metadata": dense_row.get("metadata") or sparse_row.get("metadata", {}),
                "rank": rank,
            }
        )

    log.info(f"{len(output_rows)=}, {rrf_k=}")
    return json.dumps(output_rows, ensure_ascii=False, indent=2)


def semantic_search_with_embedding(
    query_embedding: list[float],
    field: str = "description",
    root_dir: str = "artifacts/chroma",
    limit: int = 10,
) -> str:
    """Dense ChromaDB search using a pre-computed query embedding.

    Skips the embedding API call entirely; the caller is responsible for
    computing the embedding (typically in a large batch for GPU efficiency).
    """
    normalized_field = _validate_field(field=field)
    root_path = Path(root_dir)
    artifacts = _load_artifacts(root_dir=root_path, field=normalized_field)

    result = artifacts.collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        include=["metadatas", "documents", "distances"],
    )

    output_rows = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for row_index, row_id in enumerate(ids):
        output_rows.append(
            {
                "id": row_id,
                "score": float(1.0 - distances[row_index]),
                "document": documents[row_index],
                "metadata": metadatas[row_index],
                "rank": row_index + 1,
            }
        )

    log.info(f"{normalized_field=}, {len(output_rows)=}")
    return json.dumps(output_rows, ensure_ascii=False, indent=2)


def hybrid_search_with_embedding(
    query: str,
    query_embedding: list[float],
    field: str = "description",
    root_dir: str = "artifacts/chroma",
    limit: int = 10,
    rrf_k: int = 60,
) -> str:
    """Hybrid search using a pre-computed embedding for dense + BM25 for sparse.

    ``query`` is still needed for the BM25 branch; the embedding API is never
    called, which is the hot path when embeddings are pre-batched by the caller.
    """
    normalized_field = _validate_field(field=field)
    dense_json = semantic_search_with_embedding(
        query_embedding=query_embedding,
        field=normalized_field,
        root_dir=root_dir,
        limit=limit,
    )
    sparse_json = sparse_search(
        query=query,
        field=normalized_field,
        root_dir=root_dir,
        limit=limit,
    )
    log.info(f"{normalized_field=}, {rrf_k=}")
    return _merge_rrf_results(
        dense_json=dense_json,
        sparse_json=sparse_json,
        limit=limit,
        rrf_k=rrf_k,
    )


def hybrid_search(
    query: str,
    field: str = "description",
    root_dir: str = "artifacts/chroma",
    embedding_base_url: str = "http://127.0.0.1:8000/v1",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    api_key: str = "EMPTY",
    limit: int = 10,
    rrf_k: int = 60,
) -> str:
    """Runs hybrid summary/description search via RRF over dense + sparse ranks."""
    normalized_field = _validate_field(field=field)
    dense_json = semantic_search(
        query=query,
        field=normalized_field,
        root_dir=root_dir,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        api_key=api_key,
        limit=limit,
    )
    sparse_json = sparse_search(
        query=query,
        field=normalized_field,
        root_dir=root_dir,
        limit=limit,
    )
    log.info(f"{normalized_field=}, {rrf_k=}")
    return _merge_rrf_results(
        dense_json=dense_json,
        sparse_json=sparse_json,
        limit=limit,
        rrf_k=rrf_k,
    )


def main() -> None:
    """CLI entrypoint."""
    fire.Fire(
        {
            "semantic": semantic_search,
            "sparse": sparse_search,
            "hybrid": hybrid_search,
        }
    )


if __name__ == "__main__":
    main()
