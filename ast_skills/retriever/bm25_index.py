"""BM25 index helpers for retriever text fields."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger as log
from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"\w+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


class _Bm25PersistedIndex(NamedTuple):
    """Persisted BM25 payload for one field."""

    ids: list[str]
    documents: list[str]
    tokenized_documents: list[list[str]]
    metadatas: list[dict[str, Any]]
    remove_stopwords: bool


class _Bm25QueryResult(NamedTuple):
    """Sparse search results."""

    ids: list[str]
    scores: list[float]


class _LoadedBm25Artifacts(NamedTuple):
    """Loaded persisted payload and instantiated BM25 model."""

    index: _Bm25PersistedIndex
    model: BM25Okapi


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenizes text into lowercase alphanumeric terms."""
    terms = [token.lower() for token in _TOKEN_PATTERN.findall(text)]
    if not remove_stopwords:
        return terms
    return [term for term in terms if term not in _STOPWORDS]


def write_bm25_index(
    *,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    output_path: Path,
    remove_stopwords: bool = True,
) -> None:
    """Writes BM25-ready payload as JSON."""
    tokenized_documents = [
        tokenize(document, remove_stopwords=remove_stopwords) for document in documents
    ]
    payload = {
        "ids": ids,
        "documents": documents,
        "tokenized_documents": tokenized_documents,
        "metadatas": metadatas,
        "remove_stopwords": remove_stopwords,
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
        remove_stopwords=bool(payload.get("remove_stopwords", True)),
    )
    log.info(f"{path=}, {len(index.ids)=}")
    return index


def create_bm25_model(index: _Bm25PersistedIndex) -> BM25Okapi:
    """Creates a BM25 model from persisted tokenized documents."""
    log.info(f"{len(index.tokenized_documents)=}")
    return BM25Okapi(index.tokenized_documents)


def load_bm25_artifacts(path: Path) -> _LoadedBm25Artifacts:
    """Loads persisted BM25 payload and initializes model."""
    index = load_bm25_index(path=path)
    model = create_bm25_model(index=index)
    return _LoadedBm25Artifacts(index=index, model=model)


def bm25_search(
    index: _Bm25PersistedIndex,
    query: str,
    limit: int,
    model: BM25Okapi | None = None,
) -> _Bm25QueryResult:
    """Runs sparse BM25 ranking for a query."""
    if not index.ids:
        return _Bm25QueryResult(ids=[], scores=[])

    tokenized_query = tokenize(query, remove_stopwords=index.remove_stopwords)
    resolved_model = model or create_bm25_model(index=index)
    raw_scores = resolved_model.get_scores(tokenized_query)

    sorted_pairs = sorted(
        enumerate(raw_scores),
        key=lambda pair: pair[1],
        reverse=True,
    )[:limit]
    ids = [index.ids[row_index] for row_index, _ in sorted_pairs]
    scores = [float(score) for _, score in sorted_pairs]
    return _Bm25QueryResult(ids=ids, scores=scores)
