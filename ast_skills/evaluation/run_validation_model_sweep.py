"""Runs validation indexing + evaluation for one embedding model at a time."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import fire
from loguru import logger as log

from ast_skills.evaluation.evaluate_retriever import (
    build_validation_indexes,
    evaluate_validation_bm25_parquet,
    evaluate_validation_parquet,
    get_validation_index_status,
)


class _SingleEvaluationConfig(NamedTuple):
    """Configuration values for one validation model run."""

    validation_parquet: str
    artifacts_root: str
    retrieval_model: str
    mode: str
    force_reindex: bool
    vllm_base_url: str
    vllm_api_key: str
    vllm_batch_size: int
    vllm_max_concurrency: int
    start_vllm_server: bool
    vllm_gpu_device: int
    vllm_port: int
    vllm_gpu_memory_utilization: float
    wandb_project: str
    wandb_entity: str
    run_name_prefix: str


def _max_validation_rows_for_mode(mode: str) -> int:
    """Returns max rows to evaluate for a given mode."""
    normalized_mode = mode.strip().casefold()
    if normalized_mode == "smoke":
        return 20
    if normalized_mode == "full":
        return 0
    raise ValueError("mode must be either 'smoke' or 'full'.")


def _resolve_retrieval_model(retrieval_model: str, models: str) -> str:
    """Resolves the retrieval model from either `retrieval_model` or `models`."""
    if retrieval_model.strip():
        selected_model = retrieval_model.strip()
        log.info(f"{selected_model=}, source='retrieval_model'")
        return selected_model
    parsed_models = [model.strip() for model in models.split(",") if model.strip()]
    if not parsed_models:
        raise ValueError("Provide --retrieval_model or --models with one model name.")
    if len(parsed_models) > 1:
        raise ValueError("Only one model is supported per run. Provide a single model.")
    selected_model = parsed_models[0]
    log.info(f"{selected_model=}, source='models'")
    return selected_model


def _validation_parquet_exists(validation_parquet: str) -> None:
    """Raises if the validation parquet path does not exist."""
    validation_path = Path(validation_parquet)
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation parquet not found: {validation_path}")
    log.info(f"{validation_path=}, exists=True")


def _indexes_exist_for_all_fields(config: _SingleEvaluationConfig) -> bool:
    """Checks whether validation indexes exist for both summary and description."""
    if config.force_reindex:
        log.info(f"{config.force_reindex=}, skip_cache_check=True")
        return False
    index_status = get_validation_index_status(
        artifacts_root=config.artifacts_root,
        retrieval_model=config.retrieval_model,
    )
    all_indexes_exist = all(doc_count > 0 for doc_count in index_status.values())
    log.info(f"{index_status=}, {all_indexes_exist=}")
    return all_indexes_exist


def _build_indexes_if_missing(config: _SingleEvaluationConfig) -> dict[str, Any]:
    """Builds validation indexes only when missing."""
    if _indexes_exist_for_all_fields(config=config):
        output = {"status": "skipped", "reason": "indexes_already_exist"}
        log.info(f"{output=}")
        return output
    output = build_validation_indexes(
        validation_parquet=config.validation_parquet,
        retrieval_model=config.retrieval_model,
        force_reindex=config.force_reindex,
        artifacts_root=config.artifacts_root,
        vllm_base_url=config.vllm_base_url,
        vllm_api_key=config.vllm_api_key,
        vllm_batch_size=config.vllm_batch_size,
        vllm_max_concurrency=config.vllm_max_concurrency,
        start_vllm_server=config.start_vllm_server,
        vllm_gpu_device=config.vllm_gpu_device,
        vllm_port=config.vllm_port,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
    )
    log.info(f"{output=}")
    return output


def _is_bm25_model(retrieval_model: str) -> bool:
    """Returns True when the requested model maps to BM25 evaluation."""
    normalized_model = retrieval_model.strip().casefold()
    is_bm25 = normalized_model == "bm25"
    log.info(f"{retrieval_model=}, {is_bm25=}")
    return is_bm25


def run_model_sweep(
    validation_parquet: str = "artifacts/val.parquet",
    artifacts_root: str = "artifacts",
    mode: str = "smoke",
    force_reindex: bool = False,
    vllm_base_url: str = "",
    vllm_api_key: str = "EMPTY",
    vllm_batch_size: int = 512,
    vllm_max_concurrency: int = 32,
    start_vllm_server: bool = True,
    vllm_gpu_device: int = 3,
    vllm_port: int = 8002,
    vllm_gpu_memory_utilization: float = 0.9,
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name_prefix: str = "validation-single-model",
    retrieval_model: str = "",
    models: str = "",
) -> dict[str, dict]:
    """Runs one-model validation evaluation on summary and description fields.

    Notes:
      - Runs one embedding model per invocation.
      - If indexes already exist and force_reindex is False, index build is skipped.
      - BM25 is intentionally excluded from this flow.
    """
    if not vllm_base_url:
        vllm_base_url = f"http://127.0.0.1:{vllm_port}/v1"
    _validation_parquet_exists(validation_parquet=validation_parquet)
    selected_model = _resolve_retrieval_model(
        retrieval_model=retrieval_model,
        models=models,
    )
    config = _SingleEvaluationConfig(
        validation_parquet=validation_parquet,
        artifacts_root=artifacts_root,
        retrieval_model=selected_model,
        mode=mode,
        force_reindex=force_reindex,
        vllm_base_url=vllm_base_url,
        vllm_api_key=vllm_api_key,
        vllm_batch_size=vllm_batch_size,
        vllm_max_concurrency=vllm_max_concurrency,
        start_vllm_server=start_vllm_server,
        vllm_gpu_device=vllm_gpu_device,
        vllm_port=vllm_port,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_name_prefix=run_name_prefix,
    )

    max_validation_rows = _max_validation_rows_for_mode(mode=config.mode)
    run_name = f"{config.run_name_prefix}-{config.mode}-{config.retrieval_model}"
    if _is_bm25_model(retrieval_model=config.retrieval_model):
        index_output = {"status": "skipped", "reason": "bm25_does_not_use_chroma"}
        evaluation_output = evaluate_validation_bm25_parquet(
            validation_parquet=config.validation_parquet,
            force_reindex=config.force_reindex,
            artifacts_root=config.artifacts_root,
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
            run_name=run_name,
            max_validation_rows=max_validation_rows,
        )
    else:
        index_output = _build_indexes_if_missing(config=config)
        evaluation_output = evaluate_validation_parquet(
            validation_parquet=config.validation_parquet,
            retrieval_model=config.retrieval_model,
            force_reindex=False,
            artifacts_root=config.artifacts_root,
            vllm_base_url=config.vllm_base_url,
            vllm_api_key=config.vllm_api_key,
            vllm_batch_size=config.vllm_batch_size,
            vllm_max_concurrency=config.vllm_max_concurrency,
            start_vllm_server=config.start_vllm_server,
            vllm_gpu_device=config.vllm_gpu_device,
            vllm_port=config.vllm_port,
            vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
            run_name=run_name,
            max_validation_rows=max_validation_rows,
            include_bm25=False,
        )
    output = {
        config.retrieval_model: {
            "index_output": index_output,
            "evaluation_output": evaluation_output,
        }
    }
    log.info(f"{output=}")
    return output


def main() -> None:
    """CLI entrypoint."""
    fire.Fire({"run_model_sweep": run_model_sweep})


if __name__ == "__main__":
    main()
