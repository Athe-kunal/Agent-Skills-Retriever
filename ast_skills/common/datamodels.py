"""Shared data models for Hugging Face upload workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class ParquetUploadItem:
    """Single parquet artifact plus metadata for dataset upload."""

    local_file_path: str
    path_in_repo: str
    split: str
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetUploadConfig:
    """Configuration for uploading parquet files to a Hugging Face dataset repo."""

    repo_id: str
    parquet_items: list[ParquetUploadItem]
    token: str | None = None
    private: bool = False
    commit_message: str = "Upload parquet dataset artifacts"
    metadata_path_in_repo: str = "metadata/parquet_metadata.json"


@dataclass(frozen=True)
class ModelCheckpointUploadItem:
    """Single model checkpoint directory mapped to a repository path."""

    local_checkpoint_path: str
    path_in_repo: str


@dataclass(frozen=True)
class ModelUploadConfig:
    """Configuration for uploading a model and optional checkpoints."""

    repo_id: str
    local_model_path: str
    checkpoint_items: list[ModelCheckpointUploadItem] = field(default_factory=list)
    token: str | None = None
    private: bool = False
    commit_message: str = "Upload model and checkpoints"
