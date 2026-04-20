"""Build ChromaDB embedding databases for retriever fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import chromadb
import fire
from chromadb.api.models.Collection import Collection
from loguru import logger as log
from openai import OpenAI

from ast_skills.data_gen.retriever_training_dataset import (
    _REQUIRED_PARQUET_COLUMNS,
    _coerce_seed_questions_list,
    _read_parquet,
)
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


def _parse_field_filter(
    only_fields: str | Sequence[str] | None,
) -> set[str]:
    """Parses and validates the optional field filter.

    Accepts a comma-separated string (CLI) or a tuple/list (some Fire call paths).
    """
    if only_fields is None or only_fields == "":
        return set()
    if isinstance(only_fields, str):
        parts = only_fields.split(",")
    elif isinstance(only_fields, (tuple, list)):
        parts = only_fields
    else:
        raise TypeError(
            f"only_fields must be str, tuple, or list, got {type(only_fields).__name__}"
        )
    field_filter = {str(field).strip() for field in parts if str(field).strip()}
    supported_fields = {config.field_name for config in FIELD_CONFIGS}
    invalid_fields = sorted(field_filter - supported_fields)
    if invalid_fields:
        raise ValueError(
            f"Unsupported fields in only_fields: {invalid_fields}. "
            f"Supported fields: {sorted(supported_fields)}"
        )
    return field_filter


def _train_parquet_rows_to_retriever_models(
    rows: list[dict[str, Any]],
) -> list[RetrieverDataModel]:
    """Maps train-parquet rows (see ``_REQUIRED_PARQUET_COLUMNS``) to embedding models."""
    required_keys = set(_REQUIRED_PARQUET_COLUMNS)
    models: list[RetrieverDataModel] = []
    for row_index, row in enumerate(rows):
        row_keys = set(row.keys())
        if row_keys != required_keys:
            raise KeyError(
                f"Row {row_index} must have exactly keys {sorted(required_keys)!r}; "
                f"got {sorted(row_keys)!r}"
            )
        seed_questions = _coerce_seed_questions_list(row["question"])
        model = RetrieverDataModel(
            custom_id=str(row_index),
            seed_questions=[str(question) for question in seed_questions],
            name=str(row["name"]),
            description=str(row["description"]),
            metadata={},
            summary=str(row["summary"]),
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
        "seed_questions_json": json.dumps(model.seed_questions, ensure_ascii=False),
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
    input_parquet_path: str = "artifacts/train.parquet",
    output_root_dir: str = "artifacts/chroma",
    embedding_base_url: str = "http://127.0.0.1:8000/v1",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    api_key: str = "EMPTY",
    embedding_batch_size: int = 4096,
    only_fields: str = "",
) -> None:
    """Builds ChromaDB databases for summary/description fields.

    ``input_parquet_path`` must point to Parquet with columns ``name``,
    ``markdown_content``, ``summary``, ``description``, and ``question`` (same
    layout as ``artifacts/train.parquet``).

    ``only_fields`` is an optional comma-separated list of field names to build
    (e.g. ``"summary"`` or ``"summary,description"``). When empty, all fields are
    built.
    """
    input_parquet = Path(input_parquet_path)
    output_root = Path(output_root_dir)
    log.info(
        f"{input_parquet=}, {output_root=}, {embedding_base_url=}, "
        f"{embedding_model=}, {embedding_batch_size=}, {only_fields=}"
    )

    field_filter = _parse_field_filter(only_fields=only_fields)
    active_configs = [
        config
        for config in FIELD_CONFIGS
        if not field_filter or config.field_name in field_filter
    ]
    log.info(f"{field_filter=}, building fields: {[c.field_name for c in active_configs]}")

    rows = _read_parquet(input_parquet)
    models = _train_parquet_rows_to_retriever_models(rows)
    embedding_client = _make_embedding_client(
        base_url=embedding_base_url, api_key=api_key
    )

    for field_config in active_configs:
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
