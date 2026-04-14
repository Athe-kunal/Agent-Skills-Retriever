"""Build ChromaDB embedding databases for retriever fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import chromadb
import fire
from chromadb.api.models.Collection import Collection
from loguru import logger as log
from openai import OpenAI

from ast_skills.retriever.bm25_index import write_bm25_index
from ast_skills.retriever.datamodels import RetrieverDataModel


class _FieldBuildConfig(NamedTuple):
    """Configuration used to build one field-level embedding database."""

    field_name: str
    db_dir_name: str
    collection_name: str


class _CollectionPayload(NamedTuple):
    """Payload used by ChromaDB collection.add in one batch."""

    ids: list[str]
    documents: list[str]
    embeddings: list[list[float]]
    metadatas: list[dict[str, Any]]


FIELD_CONFIGS: tuple[_FieldBuildConfig, ...] = (
    _FieldBuildConfig(
        field_name="what",
        db_dir_name="what_db",
        collection_name="retriever_what",
    ),
    _FieldBuildConfig(
        field_name="why",
        db_dir_name="why_db",
        collection_name="retriever_why",
    ),
    _FieldBuildConfig(
        field_name="summary",
        db_dir_name="summary_db",
        collection_name="retriever_summary",
    ),
    _FieldBuildConfig(
        field_name="description",
        db_dir_name="description_db",
        collection_name="retriever_description",
    ),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Reads JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            row = json.loads(stripped_line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected dict row, got {type(row)=} at {line_number=}"
                )
            rows.append(row)
    log.info(f"{path=}, {len(rows)=}")
    return rows


def _rows_to_models(rows: list[dict[str, Any]]) -> list[RetrieverDataModel]:
    """Converts dictionary rows into ``RetrieverDataModel`` objects."""
    models: list[RetrieverDataModel] = []
    for row in rows:
        seed_questions = row.get("seed_questions", [])
        if not isinstance(seed_questions, list):
            seed_questions = []

        model = RetrieverDataModel(
            custom_id=str(row.get("custom_id", "")),
            markdown_content=str(row.get("markdown_content", "")),
            reasoning=str(row.get("reasoning", "")),
            what=str(row.get("what", "")),
            why=str(row.get("why", "")),
            seed_questions=[str(question) for question in seed_questions],
            name=str(row.get("name", "")),
            description=str(row.get("description", "")),
            metadata=dict(row.get("metadata", {})),
            summary=str(row.get("summary", "")),
        )
        models.append(model)
    log.info(f"{len(models)=}")
    return models


def _chunked[T](items: list[T], batch_size: int) -> list[list[T]]:
    """Splits a list into fixed-size chunks."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _build_base_metadata(model: RetrieverDataModel) -> dict[str, Any]:
    """Builds metadata payload stored with each vector."""
    flattened_metadata = {
        f"source_{key}": value for key, value in model.metadata.items()
    }
    return {
        "custom_id": model.custom_id,
        "name": model.name,
        "seed_questions": model.seed_questions,
        "metadata_json": json.dumps(model.metadata, ensure_ascii=False, sort_keys=True),
        **flattened_metadata,
    }


def _make_embedding_client(base_url: str, api_key: str) -> OpenAI:
    """Creates the OpenAI-compatible embeddings client."""
    log.info(f"{base_url=}")
    return OpenAI(base_url=base_url, api_key=api_key)


def _embed_texts(
    client: OpenAI,
    model_name: str,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    """Embeds input texts in batches."""
    all_embeddings: list[list[float]] = []
    text_batches = _chunked(texts, batch_size)
    log.info(f"{len(texts)=}, {batch_size=}, {len(text_batches)=}")
    for batch_index, text_batch in enumerate(text_batches):
        response = client.embeddings.create(model=model_name, input=text_batch)
        batch_embeddings = [item.embedding for item in response.data]
        log.info(f"{batch_index=}, {len(text_batch)=}, {len(batch_embeddings)=}")
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def _clear_collection(collection: Collection, collection_name: str) -> None:
    """Deletes all existing rows from a collection before rebuild."""
    existing_count = collection.count()
    if existing_count <= 0:
        return

    existing_rows = collection.get(include=[], limit=existing_count)
    existing_ids = existing_rows.get("ids") or []
    collection.delete(ids=existing_ids)
    log.info(f"{collection_name=}, {existing_count=}, {len(existing_ids)=}")


def _make_collection(db_path: Path, collection_name: str) -> Collection:
    """Creates or opens the target collection and clears previous rows."""
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name=collection_name)
    _clear_collection(collection=collection, collection_name=collection_name)
    return collection


def _build_payload(
    models: list[RetrieverDataModel],
    embeddings: list[list[float]],
    field_name: str,
) -> _CollectionPayload:
    """Builds Chroma ``add`` payload for one field."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for model in models:
        ids.append(model.custom_id)
        field_value = str(getattr(model, field_name))
        documents.append(field_value)
        metadata = _build_base_metadata(model)
        metadata["field_name"] = field_name
        metadatas.append(metadata)

    return _CollectionPayload(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def _write_collection(
    collection: Collection,
    payload: _CollectionPayload,
    batch_size: int = 5000,
) -> None:
    """Writes the field payload to a Chroma collection in batches."""
    id_batches = _chunked(payload.ids, batch_size)
    document_batches = _chunked(payload.documents, batch_size)
    embedding_batches = _chunked(payload.embeddings, batch_size)
    metadata_batches = _chunked(payload.metadatas, batch_size)

    total = len(payload.ids)
    log.info(f"{total=}, {batch_size=}, num_batches={len(id_batches)}")

    for batch_index, (ids, documents, embeddings, metadatas) in enumerate(
        zip(id_batches, document_batches, embedding_batches, metadata_batches)
    ):
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        log.info(f"{batch_index=}, {len(ids)=}")

    collection_count = collection.count()
    log.info(f"{collection_count=}")


def _write_bm25_artifact(
    output_root: Path,
    field_name: str,
    payload: _CollectionPayload,
) -> None:
    """Writes per-field BM25 index payload."""
    bm25_path = output_root / "bm25" / f"{field_name}.json"
    write_bm25_index(
        ids=payload.ids,
        documents=payload.documents,
        metadatas=payload.metadatas,
        output_path=bm25_path,
    )


def build_chroma_databases(
    input_jsonl_path: str,
    output_root_dir: str = "artifacts/chroma",
    embedding_base_url: str = "http://127.0.0.1:8000/v1",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    api_key: str = "EMPTY",
    embedding_batch_size: int = 4096,
) -> None:
    """Builds ChromaDB databases for what/why/summary/description fields."""
    input_jsonl = Path(input_jsonl_path)
    output_root = Path(output_root_dir)
    log.info(
        f"{input_jsonl=}, {output_root=}, {embedding_base_url=}, "
        f"{embedding_model=}, {embedding_batch_size=}"
    )

    rows = _read_jsonl(input_jsonl)
    models = _rows_to_models(rows)
    embedding_client = _make_embedding_client(
        base_url=embedding_base_url, api_key=api_key
    )

    for field_config in FIELD_CONFIGS:
        field_name = field_config.field_name
        texts = [str(getattr(model, field_name)) for model in models]
        embeddings = _embed_texts(
            client=embedding_client,
            model_name=embedding_model,
            texts=texts,
            batch_size=embedding_batch_size,
        )

        db_path = output_root / field_config.db_dir_name
        db_path.mkdir(parents=True, exist_ok=True)
        log.info(f"{field_name=}, {db_path=}")

        collection = _make_collection(
            db_path=db_path,
            collection_name=field_config.collection_name,
        )
        payload = _build_payload(
            models=models, embeddings=embeddings, field_name=field_name
        )
        _write_collection(collection=collection, payload=payload)
        _write_bm25_artifact(
            output_root=output_root,
            field_name=field_name,
            payload=payload,
        )


def main() -> None:
    """CLI entrypoint."""
    fire.Fire(build_chroma_databases)


if __name__ == "__main__":
    main()
