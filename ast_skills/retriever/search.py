"""Retriever search utilities for semantic, sparse, and hybrid ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import chromadb
import fire
from loguru import logger as log
from openai import OpenAI

from ast_skills.retriever.bm25_index import bm25_search, load_bm25_index


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


FIELD_CONFIGS: dict[str, _FieldSearchConfig] = {
    "what": _FieldSearchConfig("what", "what_db", "retriever_what"),
    "why": _FieldSearchConfig("why", "why_db", "retriever_why"),
    "description": _FieldSearchConfig("description", "description_db", "retriever_description"),
}


def _make_embedding_client(base_url: str, api_key: str) -> OpenAI:
    """Creates OpenAI-compatible client."""
    log.info(f"{base_url=}")
    return OpenAI(base_url=base_url, api_key=api_key)


def _load_artifacts(root_dir: Path, field: str) -> _SearchArtifacts:
    """Loads Chroma collection and BM25 payload path for a field."""
    config = FIELD_CONFIGS[field]
    db_path = root_dir / config.db_dir_name
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=config.collection_name)
    bm25_path = root_dir / "bm25" / f"{field}.json"
    log.info(f"{db_path=}, {bm25_path=}")
    return _SearchArtifacts(collection=collection, bm25_path=bm25_path)


def semantic_search(
    query: str,
    field: str = "description",
    root_dir: str = "artifacts/chroma",
    embedding_base_url: str = "http://127.0.0.1:8000/v1",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    api_key: str = "EMPTY",
    limit: int = 10,
) -> str:
    """Runs dense semantic search and returns JSON string."""
    root_path = Path(root_dir)
    artifacts = _load_artifacts(root_dir=root_path, field=field)
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

    log.info(f"{field=}, {len(output_rows)=}")
    return json.dumps(output_rows, ensure_ascii=False, indent=2)


def sparse_search(
    query: str,
    field: str = "description",
    root_dir: str = "artifacts/chroma",
    limit: int = 10,
) -> str:
    """Runs sparse BM25 search and returns JSON string."""
    root_path = Path(root_dir)
    artifacts = _load_artifacts(root_dir=root_path, field=field)
    bm25_index = load_bm25_index(artifacts.bm25_path)
    bm25_result = bm25_search(index=bm25_index, query=query, limit=limit)

    metadata_by_id = {
        row_id: bm25_index.metadatas[index]
        for index, row_id in enumerate(bm25_index.ids)
    }
    document_by_id = {
        row_id: bm25_index.documents[index]
        for index, row_id in enumerate(bm25_index.ids)
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

    log.info(f"{field=}, {len(output_rows)=}")
    return json.dumps(output_rows, ensure_ascii=False, indent=2)


def _rrf(scores: list[_ScoredId], k: int) -> dict[str, float]:
    """Computes reciprocal rank fusion map for one ranked list."""
    fused: dict[str, float] = {}
    for rank, scored in enumerate(scores, start=1):
        fused[scored.doc_id] = 1.0 / (k + rank)
    return fused


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
    """Runs hybrid search via RRF over dense and sparse rankings."""
    dense_json = semantic_search(
        query=query,
        field=field,
        root_dir=root_dir,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        api_key=api_key,
        limit=limit,
    )
    sparse_json = sparse_search(
        query=query,
        field=field,
        root_dir=root_dir,
        limit=limit,
    )

    dense_rows = json.loads(dense_json)
    sparse_rows = json.loads(sparse_json)
    dense_ranked = [_ScoredId(doc_id=row["id"], score=float(row["score"])) for row in dense_rows]
    sparse_ranked = [_ScoredId(doc_id=row["id"], score=float(row["score"])) for row in sparse_rows]

    dense_rrf = _rrf(dense_ranked, k=rrf_k)
    sparse_rrf = _rrf(sparse_ranked, k=rrf_k)

    doc_ids = set(dense_rrf.keys()) | set(sparse_rrf.keys())
    hybrid_scores = {
        doc_id: dense_rrf.get(doc_id, 0.0) + sparse_rrf.get(doc_id, 0.0)
        for doc_id in doc_ids
    }
    sorted_ids = sorted(hybrid_scores.keys(), key=lambda doc_id: hybrid_scores[doc_id], reverse=True)

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

    log.info(f"{field=}, {len(output_rows)=}, {rrf_k=}")
    return json.dumps(output_rows, ensure_ascii=False, indent=2)


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
