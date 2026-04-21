"""Simple Hugging Face upload helpers for dataset splits and model artifacts."""

from __future__ import annotations

from pathlib import Path

import fire
from huggingface_hub import HfApi
from loguru import logger as log


def _validate_existing_file(file_path: str) -> Path:
    """Validates that a path exists and points to a file."""
    resolved_path = Path(file_path).expanduser().resolve()
    log.info(f"{file_path=}, {resolved_path=}")
    if not resolved_path.is_file():
        raise ValueError(f"Expected file path but found: {resolved_path}")
    return resolved_path


def _validate_existing_directory(directory_path: str) -> Path:
    """Validates that a path exists and points to a directory."""
    resolved_path = Path(directory_path).expanduser().resolve()
    log.info(f"{directory_path=}, {resolved_path=}")
    if not resolved_path.is_dir():
        raise ValueError(f"Expected directory path but found: {resolved_path}")
    return resolved_path


def _create_or_get_repo(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    private: bool,
    token: str | None,
) -> None:
    """Creates the Hugging Face repository when it does not exist."""
    log.info(f"{repo_id=}, {repo_type=}, {private=}")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True, token=token)


def _split_path_in_repo(split_name: str) -> str:
    """Builds a Hugging Face datasets-compatible parquet path for one split."""
    path_in_repo = f"data/{split_name}-00000-of-00001.parquet"
    log.info(f"{split_name=}, {path_in_repo=}")
    return path_in_repo


def upload_dataset_splits(
    dataset_repo_id: str,
    train_parquet_path: str,
    validation_parquet_path: str,
    test_parquet_path: str,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Upload dataset train/validation/test parquet splits",
) -> None:
    """Uploads train/validation/test parquet files to a Hugging Face dataset repo.

    Args:
      dataset_repo_id: Dataset repository on Hugging Face Hub.
      train_parquet_path: Local path to train parquet file.
      validation_parquet_path: Local path to validation parquet file.
      test_parquet_path: Local path to test parquet file.
      token: Optional Hugging Face token.
      private: Whether the repository should be private.
      commit_message: Commit message used for uploads.
    """
    log.info(f"{dataset_repo_id=}, {private=}")
    api = HfApi(token=token)
    _create_or_get_repo(api, dataset_repo_id, "dataset", private, token)

    train_file = _validate_existing_file(train_parquet_path)
    validation_file = _validate_existing_file(validation_parquet_path)
    test_file = _validate_existing_file(test_parquet_path)

    train_path_in_repo = _split_path_in_repo("train")
    validation_path_in_repo = _split_path_in_repo("validation")
    test_path_in_repo = _split_path_in_repo("test")

    api.upload_file(
        path_or_fileobj=str(train_file),
        path_in_repo=train_path_in_repo,
        repo_id=dataset_repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        token=token,
    )
    api.upload_file(
        path_or_fileobj=str(validation_file),
        path_in_repo=validation_path_in_repo,
        repo_id=dataset_repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        token=token,
    )
    api.upload_file(
        path_or_fileobj=str(test_file),
        path_in_repo=test_path_in_repo,
        repo_id=dataset_repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        token=token,
    )
    log.info(
        f"{train_file=}, {validation_file=}, {test_file=}, "
        f"{train_path_in_repo=}, {validation_path_in_repo=}, {test_path_in_repo=}, "
        f"{dataset_repo_id=}, {commit_message=}"
    )


def upload_model(
    model_repo_id: str,
    model_path: str,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Upload model artifacts",
) -> None:
    """Uploads a local model directory to a Hugging Face model repo.

    Args:
      model_repo_id: Model repository on Hugging Face Hub.
      model_path: Local model directory path.
      token: Optional Hugging Face token.
      private: Whether the repository should be private.
      commit_message: Commit message used for uploads.
    """
    log.info(f"{model_repo_id=}, {model_path=}, {private=}")
    api = HfApi(token=token)
    _create_or_get_repo(api, model_repo_id, "model", private, token)

    resolved_model_path = _validate_existing_directory(model_path)
    api.upload_folder(
        folder_path=str(resolved_model_path),
        path_in_repo="",
        repo_id=model_repo_id,
        repo_type="model",
        commit_message=commit_message,
        token=token,
    )
    log.info(f"{resolved_model_path=}, {model_repo_id=}, {commit_message=}")


def main() -> None:
    """Runs the Fire CLI for Hugging Face uploads."""
    fire.Fire(
        {
            "upload_dataset_splits": upload_dataset_splits,
            "upload_model": upload_model,
        }
    )


if __name__ == "__main__":
    main()
