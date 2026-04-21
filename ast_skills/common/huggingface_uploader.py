"""Utilities to upload datasets and models to Hugging Face Hub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import fire
from huggingface_hub import HfApi
from loguru import logger as log

from ast_skills.common.datamodels import (
    DatasetUploadConfig,
    ModelCheckpointUploadItem,
    ModelUploadConfig,
    ParquetUploadItem,
)


class _UploadSummary(NamedTuple):
    """Tracks uploaded repository paths for reporting."""

    uploaded_paths: list[str]
    metadata_path: str | None = None


def _to_posix_path(value: str) -> str:
    """Converts a path to a normalized POSIX string."""
    posix_path = Path(value).as_posix().lstrip("./")
    log.info(f"{value=}, {posix_path=}")
    return posix_path


def _validate_existing_file(path: str) -> Path:
    """Validates that ``path`` exists and points to a file."""
    file_path = Path(path).expanduser().resolve()
    log.info(f"{path=}, {file_path=}")
    if not file_path.is_file():
        raise ValueError(f"Expected a file, but found: {file_path}")
    return file_path


def _validate_existing_directory(path: str) -> Path:
    """Validates that ``path`` exists and points to a directory."""
    directory_path = Path(path).expanduser().resolve()
    log.info(f"{path=}, {directory_path=}")
    if not directory_path.is_dir():
        raise ValueError(f"Expected a directory, but found: {directory_path}")
    return directory_path


def _create_dataset_metadata_payload(items: list[ParquetUploadItem]) -> dict[str, Any]:
    """Builds a metadata payload for parquet artifacts."""
    dataset_rows: list[dict[str, Any]] = []
    for item in items:
        dataset_row = {
            "path_in_repo": _to_posix_path(item.path_in_repo),
            "split": item.split,
            "metadata": item.metadata,
        }
        dataset_rows.append(dataset_row)
    metadata_payload = {"parquet_items": dataset_rows}
    log.info(f"{metadata_payload=}")
    return metadata_payload


def _write_json_temp_file(payload: dict[str, Any]) -> Path:
    """Writes ``payload`` to a temporary JSON file and returns its path."""
    temp_path = Path("/tmp/hf_dataset_metadata.json")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"{temp_path=}")
    return temp_path


def _ensure_repo_exists(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    private: bool,
    token: str | None,
) -> None:
    """Creates a repository if needed."""
    log.info(f"{repo_id=}, {repo_type=}, {private=}")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True, token=token)


def upload_dataset_parquet_files(config: DatasetUploadConfig) -> _UploadSummary:
    """Uploads parquet files and metadata to a Hugging Face dataset repository."""
    log.info(f"{config=}")
    api = HfApi(token=config.token)
    _ensure_repo_exists(
        api=api,
        repo_id=config.repo_id,
        repo_type="dataset",
        private=config.private,
        token=config.token,
    )

    uploaded_paths: list[str] = []
    for item in config.parquet_items:
        local_file_path = _validate_existing_file(item.local_file_path)
        path_in_repo = _to_posix_path(item.path_in_repo)
        api.upload_file(
            path_or_fileobj=str(local_file_path),
            path_in_repo=path_in_repo,
            repo_id=config.repo_id,
            repo_type="dataset",
            commit_message=config.commit_message,
            token=config.token,
        )
        uploaded_paths.append(path_in_repo)
        log.info(f"{local_file_path=}, {path_in_repo=}")

    metadata_payload = _create_dataset_metadata_payload(config.parquet_items)
    metadata_file_path = _write_json_temp_file(metadata_payload)
    metadata_path_in_repo = _to_posix_path(config.metadata_path_in_repo)
    api.upload_file(
        path_or_fileobj=str(metadata_file_path),
        path_in_repo=metadata_path_in_repo,
        repo_id=config.repo_id,
        repo_type="dataset",
        commit_message=config.commit_message,
        token=config.token,
    )
    log.info(f"{metadata_path_in_repo=}")
    return _UploadSummary(uploaded_paths=uploaded_paths, metadata_path=metadata_path_in_repo)


def _upload_checkpoint_items(
    api: HfApi,
    config: ModelUploadConfig,
    uploaded_paths: list[str],
) -> None:
    """Uploads checkpoint directories to a Hugging Face model repository."""
    checkpoint_item: ModelCheckpointUploadItem
    for checkpoint_item in config.checkpoint_items:
        local_checkpoint_path = _validate_existing_directory(checkpoint_item.local_checkpoint_path)
        path_in_repo = _to_posix_path(checkpoint_item.path_in_repo)
        api.upload_folder(
            folder_path=str(local_checkpoint_path),
            path_in_repo=path_in_repo,
            repo_id=config.repo_id,
            repo_type="model",
            commit_message=config.commit_message,
            token=config.token,
        )
        uploaded_paths.append(path_in_repo)
        log.info(f"{local_checkpoint_path=}, {path_in_repo=}")


def upload_model_and_checkpoints(config: ModelUploadConfig) -> _UploadSummary:
    """Uploads a model directory and optional checkpoints to Hugging Face Hub."""
    log.info(f"{config=}")
    api = HfApi(token=config.token)
    _ensure_repo_exists(
        api=api,
        repo_id=config.repo_id,
        repo_type="model",
        private=config.private,
        token=config.token,
    )

    local_model_path = _validate_existing_directory(config.local_model_path)
    api.upload_folder(
        folder_path=str(local_model_path),
        path_in_repo="",
        repo_id=config.repo_id,
        repo_type="model",
        commit_message=config.commit_message,
        token=config.token,
    )
    uploaded_paths = ["/"]
    _upload_checkpoint_items(api=api, config=config, uploaded_paths=uploaded_paths)
    log.info(f"{uploaded_paths=}")
    return _UploadSummary(uploaded_paths=uploaded_paths)


class HuggingFaceUploaderCLI:
    """CLI wrapper for Hugging Face upload utilities."""

    def upload_dataset(self, config_path: str) -> _UploadSummary:
        """Uploads parquet files using a JSON configuration file."""
        config = load_dataset_upload_config(config_path)
        return upload_dataset_parquet_files(config)

    def upload_model(self, config_path: str) -> _UploadSummary:
        """Uploads model and checkpoints using a JSON configuration file."""
        config = load_model_upload_config(config_path)
        return upload_model_and_checkpoints(config)


def _load_json_payload(config_path: str) -> dict[str, Any]:
    """Loads JSON payload from disk."""
    config_file_path = _validate_existing_file(config_path)
    payload = json.loads(config_file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a JSON object.")
    log.info(f"{config_file_path=}, {list(payload.keys())=}")
    return payload


def _build_parquet_items(raw_items: list[dict[str, Any]]) -> list[ParquetUploadItem]:
    """Builds strongly-typed parquet upload items from JSON payload."""
    parquet_items: list[ParquetUploadItem] = []
    for raw_item in raw_items:
        parquet_item = ParquetUploadItem(
            local_file_path=str(raw_item["local_file_path"]),
            path_in_repo=str(raw_item["path_in_repo"]),
            split=str(raw_item["split"]),
            metadata=dict(raw_item.get("metadata", {})),
        )
        parquet_items.append(parquet_item)
    log.info(f"{parquet_items=}")
    return parquet_items


def _build_checkpoint_items(raw_items: list[dict[str, Any]]) -> list[ModelCheckpointUploadItem]:
    """Builds strongly-typed checkpoint upload items from JSON payload."""
    checkpoint_items: list[ModelCheckpointUploadItem] = []
    for raw_item in raw_items:
        checkpoint_item = ModelCheckpointUploadItem(
            local_checkpoint_path=str(raw_item["local_checkpoint_path"]),
            path_in_repo=str(raw_item["path_in_repo"]),
        )
        checkpoint_items.append(checkpoint_item)
    log.info(f"{checkpoint_items=}")
    return checkpoint_items


def load_dataset_upload_config(config_path: str) -> DatasetUploadConfig:
    """Loads ``DatasetUploadConfig`` from a JSON file."""
    payload = _load_json_payload(config_path)
    parquet_items = _build_parquet_items(list(payload.get("parquet_items", [])))
    config = DatasetUploadConfig(
        repo_id=str(payload["repo_id"]),
        parquet_items=parquet_items,
        token=payload.get("token"),
        private=bool(payload.get("private", False)),
        commit_message=str(payload.get("commit_message", "Upload parquet dataset artifacts")),
        metadata_path_in_repo=str(payload.get("metadata_path_in_repo", "metadata/parquet_metadata.json")),
    )
    log.info(f"{config=}")
    return config


def load_model_upload_config(config_path: str) -> ModelUploadConfig:
    """Loads ``ModelUploadConfig`` from a JSON file."""
    payload = _load_json_payload(config_path)
    checkpoint_items = _build_checkpoint_items(list(payload.get("checkpoint_items", [])))
    config = ModelUploadConfig(
        repo_id=str(payload["repo_id"]),
        local_model_path=str(payload["local_model_path"]),
        checkpoint_items=checkpoint_items,
        token=payload.get("token"),
        private=bool(payload.get("private", False)),
        commit_message=str(payload.get("commit_message", "Upload model and checkpoints")),
    )
    log.info(f"{config=}")
    return config


def main() -> None:
    """Runs the Hugging Face uploader command line interface."""
    fire.Fire(HuggingFaceUploaderCLI)


if __name__ == "__main__":
    main()
