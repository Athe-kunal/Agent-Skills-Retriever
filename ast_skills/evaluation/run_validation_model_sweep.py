"""Runs validation index build + evaluation across multiple embedding models."""

from __future__ import annotations

from typing import NamedTuple

import fire
from loguru import logger as log

from ast_skills.evaluation.evaluate_retriever import (
    build_validation_indexes,
    evaluate_validation_parquet,
)


DEFAULT_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "colbert-ir/colbertv2.0",
)


class _SweepConfig(NamedTuple):
    """Configuration values used for one model sweep run."""

    validation_parquet: str
    artifacts_root: str
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
    """Returns max rows to evaluate for given mode."""
    normalized_mode = mode.strip().casefold()
    if normalized_mode == "smoke":
        return 5
    if normalized_mode == "full":
        return 0
    raise ValueError("mode must be either 'smoke' or 'full'.")


def _parse_models(models: str | None) -> tuple[str, ...]:
    """Returns model list from comma-separated string, or DEFAULT_MODELS if empty."""
    if not models or not models.strip():
        return DEFAULT_MODELS
    parsed = tuple(m.strip() for m in models.split(",") if m.strip())
    log.info(f"{models=}, {parsed=}")
    return parsed


def _run_for_model(model_name: str, config: _SweepConfig) -> dict:
    """Builds indexes and runs evaluation for one model."""
    log.info(f"{model_name=}, {config.mode=}, {config.force_reindex=}")
    index_output = build_validation_indexes(
        validation_parquet=config.validation_parquet,
        retrieval_model=model_name,
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
    max_validation_rows = _max_validation_rows_for_mode(mode=config.mode)
    evaluation_output = evaluate_validation_parquet(
        validation_parquet=config.validation_parquet,
        retrieval_model=model_name,
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
        run_name=f"{config.run_name_prefix}-{config.mode}-{model_name}",
        max_validation_rows=max_validation_rows,
    )
    output = {
        "index_output": index_output,
        "evaluation_output": evaluation_output,
    }
    log.info(f"{model_name=}, {output=}")
    return output


def run_model_sweep(
    validation_parquet: str = "artifacts/val.parquet",
    artifacts_root: str = "artifacts",
    mode: str = "smoke",
    force_reindex: bool = False,
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    vllm_api_key: str = "EMPTY",
    vllm_batch_size: int = 64,
    vllm_max_concurrency: int = 8,
    start_vllm_server: bool = True,
    vllm_gpu_device: int = 3,
    vllm_port: int = 8000,
    vllm_gpu_memory_utilization: float = 0.9,
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name_prefix: str = "validation-model-sweep",
    models: str = "",
) -> dict[str, dict]:
    """Runs validation retrieval indexing and evaluation for the given models.

    Args:
      models: Comma-separated model names to sweep. Defaults to all DEFAULT_MODELS.
    """
    model_list = _parse_models(models)
    config = _SweepConfig(
        validation_parquet=validation_parquet,
        artifacts_root=artifacts_root,
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
    output_by_model: dict[str, dict] = {}
    for model_name in model_list:
        try:
            output_by_model[model_name] = _run_for_model(
                model_name=model_name,
                config=config,
            )
        except Exception as exc:
            log.error(f"{model_name=}, sweep_failed=True, {exc=}")
            output_by_model[model_name] = {"error": str(exc)}
    log.info(f"{output_by_model=}")
    return output_by_model


def main() -> None:
    """CLI entrypoint."""
    fire.Fire({"run_model_sweep": run_model_sweep})


if __name__ == "__main__":
    main()
