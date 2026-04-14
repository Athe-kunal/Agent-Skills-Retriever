"""BM25 index helpers for retriever text fields."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger as log
from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"\w+")


class _Bm25PersistedIndex(NamedTuple):
    """Persisted BM25 payload for one field."""

    ids: list[str]
    documents: list[str]
    tokenized_documents: list[list[str]]
    metadatas: list[dict[str, Any]]


class _Bm25QueryResult(NamedTuple):
    """Sparse search results."""

    ids: list[str]
    scores: list[float]


def tokenize(text: str) -> list[str]:
    """Tokenizes text into lowercase alphanumeric terms."""
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def write_bm25_index(
    *,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Writes BM25-ready payload as JSON."""
    tokenized_documents = [tokenize(document) for document in documents]
    payload = {
        "ids": ids,
        "documents": documents,
        "tokenized_documents": tokenized_documents,
        "metadatas": metadatas,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info(f"{output_path=}, {len(ids)=}")


def load_bm25_index(path: Path) -> _Bm25PersistedIndex:
    """Loads persisted BM25 payload from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    index = _Bm25PersistedIndex(
        ids=list(payload.get("ids", [])),
        documents=list(payload.get("documents", [])),
        tokenized_documents=list(payload.get("tokenized_documents", [])),
        metadatas=list(payload.get("metadatas", [])),
    )
    log.info(f"{path=}, {len(index.ids)=}")
    return index


def bm25_search(index: _Bm25PersistedIndex, query: str, limit: int) -> _Bm25QueryResult:
    """Runs sparse BM25 ranking for a query."""
    if not index.ids:
        return _Bm25QueryResult(ids=[], scores=[])

    tokenized_query = tokenize(query)
    bm25 = BM25Okapi(index.tokenized_documents)
    raw_scores = bm25.get_scores(tokenized_query)

    sorted_pairs = sorted(
        enumerate(raw_scores),
        key=lambda pair: pair[1],
        reverse=True,
    )[:limit]
    ids = [index.ids[row_index] for row_index, _ in sorted_pairs]
    scores = [float(score) for _, score in sorted_pairs]
    return _Bm25QueryResult(ids=ids, scores=scores)
