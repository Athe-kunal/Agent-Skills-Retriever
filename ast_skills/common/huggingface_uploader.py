"""Simple Hugging Face upload helpers for dataset splits and model artifacts."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import fire
import pyarrow.parquet as pq
from huggingface_hub import HfApi, login, upload_folder
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


def _coerce_to_bool(flag_value: bool | str) -> bool:
    """Converts Fire CLI or Makefile/env string flags to bool for Hub API fields.

    Fire often passes ``--private false`` as the string ``\"false\"``, which the Hub
    API rejects (it expects a JSON boolean).
    """
    if isinstance(flag_value, bool):
        return flag_value
    if isinstance(flag_value, str):
        normalized = flag_value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    raise ValueError(f"Invalid boolean flag value: {flag_value!r}")


def _create_or_get_repo(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    private: bool | str,
    token: str | None,
) -> str:
    """Creates the Hugging Face repository when it does not exist.

    Returns the fully-qualified ``namespace/name`` repo_id as resolved by the Hub,
    which may differ from the bare ``name`` passed in when no namespace is given.
    """
    private_bool = _coerce_to_bool(private)
    log.info(f"{repo_id=}, {repo_type=}, {private_bool=}, {private=}")
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private_bool,
        exist_ok=True,
        token=token,
    )
    full_repo_id = repo_url.repo_id
    log.info(f"{full_repo_id=}")
    return full_repo_id


def _shard_parquet(src: Path, data_dir: Path, split_name: str, num_shards: int) -> None:
    """Reads ``src`` and writes it as ``num_shards`` equal parquet files into ``data_dir``.

    Output filenames follow the HuggingFace convention:
    ``{split_name}-{index:05d}-of-{num_shards:05d}.parquet``.
    """
    table = pq.read_table(src)
    total_rows = len(table)
    rows_per_shard = math.ceil(total_rows / num_shards)
    log.info(f"{split_name=}, {total_rows=}, {num_shards=}, {rows_per_shard=}")
    for shard_index in range(num_shards):
        row_start = shard_index * rows_per_shard
        row_end = min(row_start + rows_per_shard, total_rows)
        shard_table = table.slice(row_start, row_end - row_start)
        dst = data_dir / f"{split_name}-{shard_index:05d}-of-{num_shards:05d}.parquet"
        pq.write_table(shard_table, dst)
        log.info(f"{shard_index=}, {row_start=}, {row_end=}, {dst=}")


def _stage_three_split_parquets(
    train_file: Path,
    validation_file: Path,
    test_file: Path,
    staging_root: Path,
    train_num_shards: int = 1,
) -> Path:
    """Copies split parquets into a Hub-ready ``data/`` tree under ``staging_root``.

    The train split is written as ``train_num_shards`` equal shards; validation
    and test are always a single shard each.
    """
    data_dir = staging_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    _shard_parquet(train_file, data_dir, "train", train_num_shards)

    for split_name, src in (
        ("validation", validation_file),
        ("test", test_file),
    ):
        dst = data_dir / f"{split_name}-00000-of-00001.parquet"
        shutil.copy2(src, dst)
        log.info(f"{split_name=}, {src=}, {dst=}")

    return staging_root


def upload_dataset_splits(
    dataset_repo_id: str,
    train_parquet_path: str,
    validation_parquet_path: str,
    test_parquet_path: str,
    token: str | None = None,
    private: bool | str = False,
    commit_message: str = "Upload dataset train/validation/test parquet splits",
    train_num_shards: int = 5,
) -> None:
    """Uploads train/validation/test parquet files to a Hugging Face dataset repo.

    The train file is split into ``train_num_shards`` equal shards before upload
    so that large files transfer in parallel-friendly chunks. Validation and test
    are always a single shard each.

    Args:
      dataset_repo_id: Dataset repository on Hugging Face Hub (``namespace/name``).
      train_parquet_path: Local path to train parquet file.
      validation_parquet_path: Local path to validation parquet file.
      test_parquet_path: Local path to test parquet file.
      token: Optional Hugging Face token.
      private: Whether the repository should be private.
      commit_message: Commit message used for uploads.
      train_num_shards: Number of parquet shards to split the train file into.
    """
    log.info(f"{dataset_repo_id=}, {private=}, {train_num_shards=}")
    if token:
        login(token=token)
    else:
        login()

    api = HfApi(token=token)
    full_dataset_repo_id = _create_or_get_repo(api, dataset_repo_id, "dataset", private, token)

    train_file = _validate_existing_file(train_parquet_path)
    validation_file = _validate_existing_file(validation_parquet_path)
    test_file = _validate_existing_file(test_parquet_path)

    with tempfile.TemporaryDirectory(prefix="hf_dataset_splits_") as tmp:
        staging_root = Path(tmp)
        _stage_three_split_parquets(
            train_file, validation_file, test_file, staging_root, train_num_shards
        )
        log.info(f"{staging_root=}, {full_dataset_repo_id=}, {commit_message=}, {train_num_shards=}")
        upload_folder(
            folder_path=str(staging_root),
            repo_id=full_dataset_repo_id,
            repo_type="dataset",
            commit_message=commit_message,
            token=token,
        )


_TRAINING_ARTIFACT_PATTERNS: list[str] = [
    "pytorch_model_fsdp.bin",  # FSDP raw checkpoint — weights already in model.safetensors
    "optimizer.bin",           # optimizer state — only needed to resume training
    "scheduler.pt",            # LR scheduler state
    "rng_state.pth",           # RNG state
    "training_args.bin",       # serialised TrainingArguments
    "trainer_state.json",      # step/loss history used by the Trainer
]


def upload_model(
    model_repo_id: str,
    model_path: str,
    token: str | None = None,
    private: bool | str = False,
    commit_message: str = "Upload model artifacts",
    ignore_patterns: list[str] | None = None,
) -> None:
    """Uploads a local model directory to a Hugging Face model repo.

    Training-resume artifacts (optimizer state, FSDP checkpoint, scheduler,
    RNG state) are excluded by default via ``_TRAINING_ARTIFACT_PATTERNS``.
    Pass ``ignore_patterns=[]`` to upload everything.

    Args:
      model_repo_id: Model repository on Hugging Face Hub.
      model_path: Local model directory path.
      token: Optional Hugging Face token.
      private: Whether the repository should be private.
      commit_message: Commit message used for uploads.
      ignore_patterns: Glob patterns to exclude; defaults to training artifacts.
    """
    if ignore_patterns is None:
        ignore_patterns = _TRAINING_ARTIFACT_PATTERNS

    log.info(f"{model_repo_id=}, {model_path=}, {private=}, {ignore_patterns=}")
    api = HfApi(token=token)
    full_model_repo_id = _create_or_get_repo(api, model_repo_id, "model", private, token)

    resolved_model_path = _validate_existing_directory(model_path)
    api.upload_folder(
        folder_path=str(resolved_model_path),
        path_in_repo="",
        repo_id=full_model_repo_id,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
        token=token,
    )
    log.info(f"{resolved_model_path=}, {full_model_repo_id=}, {commit_message=}")


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
