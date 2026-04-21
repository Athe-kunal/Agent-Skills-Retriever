"""Sentence-transformer trainer for mined-negative parquet datasets."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import fire
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
import yaml
from datasets import Dataset
from loguru import logger as log
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator

from ast_skills.train.datamodels import TrainingParquetRow

_DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class _ParsedTrainingData(NamedTuple):
    """Parsed rows and dropped-count metadata for parquet loading."""

    rows: list[TrainingParquetRow]
    dropped_rows: int


class _RetrievalDataset(NamedTuple):
    """Query/document pairs prepared for retrieval metric reporting."""

    queries: list[str]
    relevant_document_indices: list[int]
    documents: list[str]


@dataclass(frozen=True)
class TrainConfig:
    """Public configuration for parquet-based training."""

    train_parquet: str
    validation_parquet: str
    output_dir: str = "artifacts/sentence_transformers/qwen3-summary"
    base_model_name: str = _DEFAULT_BASE_MODEL
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 200
    eval_strategy: str = "epoch"
    evaluation_steps: int = 200
    save_strategy: str = "epoch"
    checkpoint_save_steps: int = 200
    seed: int = 13
    use_hard_negatives: bool = True
    triplet_margin: float = 0.2
    gradient_accumulation_steps: int = 1
    bf16: bool = True
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    use_wandb: bool = False
    wandb_project: str = "ast-skills-retriever"
    wandb_entity: str = ""
    run_name: str = "qwen3-parquet-train"
    fsdp_mode: str = ""
    fsdp_use_orig_params: bool = True
    fsdp_cpu_ram_efficient_loading: bool = True
    fsdp_activation_checkpointing: bool = False


def _read_yaml(path: Path) -> dict[str, Any]:
    """Reads a YAML config file as a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML config root must be a mapping.")
    log.info(f"{path=}, {list(payload.keys())=}")
    return payload


def _parse_bool(value: Any, default: bool) -> bool:
    """Parses YAML boolean-like values robustly."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        stripped_value = value.strip()
        parsed_value = yaml.safe_load(stripped_value)
        if isinstance(parsed_value, bool):
            return parsed_value
        normalized_value = stripped_value.lower()
        if normalized_value in {"1", "yes", "y", "on"}:
            return True
        if normalized_value in {"0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _parse_eval_strategy(value: Any, default: str = "epoch") -> str:
    """Parses and validates evaluation strategy."""
    normalized_value = str(value or default).strip().lower()
    if normalized_value in {"epoch", "steps", "no"}:
        log.info(f"{normalized_value=}")
        return normalized_value
    raise ValueError("`eval_strategy` must be one of: 'epoch', 'steps', 'no'.")


def _parse_save_strategy(value: Any, default: str = "epoch") -> str:
    """Parses and validates checkpoint save strategy."""
    normalized_value = str(value or default).strip().lower()
    if normalized_value in {"epoch", "steps"}:
        log.info(f"{normalized_value=}")
        return normalized_value
    raise ValueError("`save_strategy` must be either 'epoch' or 'steps'.")


def _train_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extracts kwargs for ``train`` from config payload."""
    if _is_nested_config(config):
        nested_config = _train_config_from_nested_config(config)
        kwargs = asdict(nested_config)
        log.info(f"{kwargs=}")
        return kwargs
    train_config = config.get("train", {})
    if not isinstance(train_config, dict):
        raise ValueError("`train` section must be a mapping.")
    kwargs = dict(train_config)
    _validate_required_paths(kwargs)
    log.info(f"{kwargs=}")
    return kwargs


def _is_nested_config(config: dict[str, Any]) -> bool:
    """Returns true when config payload uses nested sections."""
    nested_keys = {"input", "model", "training", "output", "logging"}
    is_nested = bool(nested_keys.intersection(config.keys()))
    log.info(f"{is_nested=}")
    return is_nested


def _required_non_empty_string(raw_value: Any, field_name: str) -> str:
    """Normalizes and validates required non-empty string fields."""
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"`{field_name}` must be a non-empty string.")
    log.info(f"{field_name=}, {value=}")
    return value


def _validate_required_paths(kwargs: dict[str, Any]) -> None:
    """Validates required parquet paths for training and validation."""
    train_parquet = _required_non_empty_string(kwargs.get("train_parquet", ""), "train_parquet")
    validation_parquet = _required_non_empty_string(
        kwargs.get("validation_parquet", ""),
        "validation_parquet",
    )
    kwargs["train_parquet"] = train_parquet
    kwargs["validation_parquet"] = validation_parquet


def _train_config_from_nested_config(config: dict[str, Any]) -> TrainConfig:
    """Extracts ``TrainConfig`` from nested config sections."""
    input_config = dict(config.get("input", {}))
    model_config = dict(config.get("model", {}))
    training_config = dict(config.get("training", {}))
    output_config = dict(config.get("output", {}))
    logging_config = dict(config.get("logging", {}))
    train_config = TrainConfig(
        train_parquet=_required_non_empty_string(
            input_config.get("mined_parquet_path", ""),
            "input.mined_parquet_path",
        ),
        validation_parquet=_required_non_empty_string(
            input_config.get("validation_parquet_path", ""),
            "input.validation_parquet_path",
        ),
        output_dir=str(output_config.get("dir", "artifacts/sentence_transformers/qwen3-summary")),
        base_model_name=str(model_config.get("name", _DEFAULT_BASE_MODEL)),
        epochs=int(training_config.get("epochs", 3)),
        batch_size=int(training_config.get("batch_size", 32)),
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        warmup_steps=int(training_config.get("warmup_steps", 200)),
        eval_strategy=_parse_eval_strategy(training_config.get("eval_strategy", "epoch")),
        evaluation_steps=int(training_config.get("evaluation_steps", 200)),
        save_strategy=_parse_save_strategy(training_config.get("save_strategy", "epoch")),
        checkpoint_save_steps=int(training_config.get("checkpoint_save_steps", 200)),
        seed=int(training_config.get("seed", 13)),
        use_hard_negatives=_parse_bool(training_config.get("use_hard_negatives", True), default=True),
        triplet_margin=float(training_config.get("triplet_margin", 0.2)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 1)),
        bf16=_parse_bool(training_config.get("bf16", True), default=True),
        gradient_checkpointing=_parse_bool(
            training_config.get("gradient_checkpointing", False), default=False
        ),
        dataloader_num_workers=int(training_config.get("dataloader_num_workers", 0)),
        use_wandb=_parse_bool(logging_config.get("use_wandb", False), default=False),
        wandb_project=str(logging_config.get("project", "ast-skills-retriever")),
        wandb_entity=str(logging_config.get("entity", "")),
        run_name=str(logging_config.get("run_name", "qwen3-parquet-train")),
        fsdp_mode=str(training_config.get("fsdp_mode", "")).strip(),
        fsdp_use_orig_params=_parse_bool(
            training_config.get("fsdp_use_orig_params", True), default=True
        ),
        fsdp_cpu_ram_efficient_loading=_parse_bool(
            training_config.get("fsdp_cpu_ram_efficient_loading", True), default=True
        ),
        fsdp_activation_checkpointing=_parse_bool(
            training_config.get("fsdp_activation_checkpointing", False), default=False
        ),
    )
    log.info(f"{train_config=}")
    return train_config


def _read_parquet_rows(train_parquet: str) -> _ParsedTrainingData:
    """Loads and normalizes rows from mined-negative parquet."""
    train_path = Path(train_parquet)
    dataframe = pd.read_parquet(train_path)
    records = dataframe.to_dict(orient="records")
    log.info(f"{train_parquet=}, {len(records)=}")

    rows: list[TrainingParquetRow] = []
    dropped_rows = 0
    for record in records:
        row = _record_to_training_row(record)
        if row is None:
            dropped_rows += 1
            continue
        rows.append(row)

    log.info(f"{len(rows)=}, {dropped_rows=}")
    return _ParsedTrainingData(rows=rows, dropped_rows=dropped_rows)


def _record_to_training_row(record: dict[str, Any]) -> TrainingParquetRow | None:
    """Converts one parquet record into a normalized training row."""
    question = str(record.get("question", "")).strip()
    positive_summary = str(record.get("summary", "")).strip()
    if not question or not positive_summary:
        return None

    raw_negatives = record.get("negative_documents", [])
    if hasattr(raw_negatives, "tolist"):
        hard_negatives = raw_negatives.tolist()
    else:
        hard_negatives = list(raw_negatives) if raw_negatives is not None else []

    return TrainingParquetRow(
        question=question,
        positive_summary=positive_summary,
        hard_negatives=hard_negatives,
    )


def _build_hf_dataset(rows: list[TrainingParquetRow], use_hard_negatives: bool) -> Dataset:
    """Builds a HuggingFace Dataset for SentenceTransformerTrainer.

    Column names ``anchor``/``positive``/``negative`` are the convention expected
    by ``TripletLoss`` and ``MultipleNegativesRankingLoss`` in sentence-transformers v3+.
    """
    if use_hard_negatives:
        valid_rows = [r for r in rows if r.hard_negatives]
        if not valid_rows:
            raise ValueError("use_hard_negatives=True but no rows contain hard negatives.")
        data: dict[str, list[str]] = {
            "anchor": [r.question for r in valid_rows],
            "positive": [r.positive_summary for r in valid_rows],
            "negative": [r.hard_negatives[0] for r in valid_rows],
        }
    else:
        data = {
            "anchor": [r.question for r in rows],
            "positive": [r.positive_summary for r in rows],
        }
    dataset = Dataset.from_dict(data)
    log.info(f"{len(dataset)=}, {use_hard_negatives=}")
    return dataset


def _build_loss(
    model: SentenceTransformer,
    use_hard_negatives: bool,
    triplet_margin: float,
) -> losses.TripletLoss | losses.MultipleNegativesRankingLoss:
    """Builds training loss, defaulting to in-batch negatives."""
    if use_hard_negatives:
        loss = losses.TripletLoss(model=model, triplet_margin=triplet_margin)
        log.info(f"{use_hard_negatives=}, {triplet_margin=}")
        return loss
    loss = losses.MultipleNegativesRankingLoss(model=model)
    log.info(f"{use_hard_negatives=}")
    return loss


def _set_random_seed(seed: int) -> None:
    """Sets deterministic PyTorch seed."""
    torch.manual_seed(seed)
    log.info(f"{seed=}")


def _build_retrieval_dataset(rows: list[TrainingParquetRow], split_name: str) -> _RetrievalDataset:
    """Builds retrieval dataset used for train/validation metric computation."""
    queries: list[str] = []
    relevant_document_indices: list[int] = []
    documents: list[str] = []
    document_index_by_text: dict[str, int] = {}

    for row in rows:
        positive_summary = row.positive_summary.strip()
        if not positive_summary:
            continue
        if positive_summary not in document_index_by_text:
            document_index_by_text[positive_summary] = len(documents)
            documents.append(positive_summary)
        queries.append(row.question)
        relevant_document_indices.append(document_index_by_text[positive_summary])

    dataset = _RetrievalDataset(
        queries=queries,
        relevant_document_indices=relevant_document_indices,
        documents=documents,
    )
    log.info(f"{split_name=}, {len(dataset.queries)=}, {len(dataset.documents)=}")
    return dataset


def _compute_top_k_hits(ranked_indices: np.ndarray, positive_index: int, k: int) -> float:
    """Computes whether positive index appears in top-k."""
    top_k_indices = ranked_indices[:k]
    hit = float(positive_index in top_k_indices)
    return hit


def _compute_mrr(ranked_indices: np.ndarray, positive_index: int) -> float:
    """Computes reciprocal rank for one query."""
    positions = np.where(ranked_indices == positive_index)[0]
    if len(positions) == 0:
        return 0.0
    reciprocal_rank = 1.0 / float(positions[0] + 1)
    return reciprocal_rank


def _compute_validation_metrics(
    model: SentenceTransformer,
    dataset: _RetrievalDataset,
) -> dict[str, float]:
    """Computes retrieval metrics on validation parquet rows."""
    if not dataset.queries or not dataset.documents:
        return _empty_validation_metrics()

    query_embeddings = model.encode(
        dataset.queries, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    )
    document_embeddings = model.encode(
        dataset.documents, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    )
    similarity_matrix = np.matmul(query_embeddings, document_embeddings.T)

    hit_at_1 = 0.0
    hit_at_3 = 0.0
    hit_at_5 = 0.0
    hit_at_10 = 0.0
    mrr = 0.0
    for row_index, positive_index in enumerate(dataset.relevant_document_indices):
        ranked_indices = np.argsort(-similarity_matrix[row_index])
        hit_at_1 += _compute_top_k_hits(ranked_indices, positive_index, k=1)
        hit_at_3 += _compute_top_k_hits(ranked_indices, positive_index, k=3)
        hit_at_5 += _compute_top_k_hits(ranked_indices, positive_index, k=5)
        hit_at_10 += _compute_top_k_hits(ranked_indices, positive_index, k=10)
        mrr += _compute_mrr(ranked_indices, positive_index)

    num_queries = float(len(dataset.queries))
    metrics = {
        "eval/hit@1": hit_at_1 / num_queries,
        "eval/hit@3": hit_at_3 / num_queries,
        "eval/hit@5": hit_at_5 / num_queries,
        "eval/hit@10": hit_at_10 / num_queries,
        "eval/mrr": mrr / num_queries,
    }
    log.info(f"{metrics=}")
    return metrics


def _empty_validation_metrics() -> dict[str, float]:
    """Builds the zeroed validation metrics payload."""
    return {
        "eval/hit@1": 0.0,
        "eval/hit@3": 0.0,
        "eval/hit@5": 0.0,
        "eval/hit@10": 0.0,
        "eval/mrr": 0.0,
    }


def _is_main_distributed_process() -> bool:
    """Returns true when this process is rank 0 or single-process."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    is_main_process = dist.get_rank() == 0
    log.info(f"{is_main_process=}")
    return is_main_process


def _clear_gradients(module: torch.nn.Module) -> None:
    """Clears gradients on a module before FSDP full-parameter materialization."""
    cleared_gradients = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        parameter.grad = None
        cleared_gradients += 1
    log.info(f"{cleared_gradients=}")


def _is_fsdp_wrapped_model(model: Any) -> bool:
    """Checks whether the incoming model is wrapped by torch FSDP."""
    module_name = model.__class__.__module__
    class_name = model.__class__.__name__
    is_fsdp_module = module_name.startswith("torch.distributed.fsdp")
    is_fsdp_class = class_name == "FullyShardedDataParallel"
    is_fsdp_model = is_fsdp_module and is_fsdp_class
    log.info(f"{module_name=}, {class_name=}, {is_fsdp_model=}")
    return is_fsdp_model


def _compute_validation_metrics_for_trainer_model(
    model: Any,
    dataset: _RetrievalDataset,
) -> dict[str, float]:
    """Computes validation metrics for either plain or FSDP-wrapped models."""
    if not _is_fsdp_wrapped_model(model):
        return _compute_validation_metrics(model=model, dataset=dataset)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    _clear_gradients(model)
    with FSDP.summon_full_params(
        model,
        writeback=False,
        offload_to_cpu=False,
        rank0_only=True,
    ):
        if not _is_main_distributed_process():
            return _empty_validation_metrics()
        sentence_transformer_model = model.module
        return _compute_validation_metrics(model=sentence_transformer_model, dataset=dataset)


class _ParquetValidationEvaluator(SentenceEvaluator):
    """Sentence-transformer evaluator for parquet-based retrieval metrics."""

    primary_metric = "mrr"

    def __init__(self, validation_dataset: _RetrievalDataset, use_wandb: bool):
        self._validation_dataset = validation_dataset
        self._use_wandb = use_wandb

    def __call__(
        self,
        model: Any,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        del output_path
        metrics = _compute_validation_metrics_for_trainer_model(
            model=model,
            dataset=self._validation_dataset,
        )
        metrics["eval/epoch"] = float(epoch)
        metrics["eval/steps"] = float(steps)
        if self._use_wandb and wandb.run is not None:
            wandb.log(metrics)
        return metrics["eval/mrr"]


def _build_fsdp_config(config: TrainConfig) -> tuple[str, dict[str, Any] | None]:
    """Builds FSDP mode and config payload for TrainingArguments."""
    fsdp_mode = config.fsdp_mode.strip().lower()
    if not fsdp_mode:
        log.info(f"{fsdp_mode=}")
        return "", None

    if fsdp_mode not in {"full_shard", "shard_grad_op", "hybrid_shard", "hybrid_shard_zero2"}:
        raise ValueError(
            "`fsdp_mode` must be one of: full_shard, shard_grad_op, "
            "hybrid_shard, hybrid_shard_zero2, or empty string."
        )

    fsdp_config = {
        "backward_prefetch": "backward_pre",
        "forward_prefetch": False,
        "use_orig_params": config.fsdp_use_orig_params,
        "cpu_ram_efficient_loading": config.fsdp_cpu_ram_efficient_loading,
        "activation_checkpointing": config.fsdp_activation_checkpointing,
    }
    log.info(f"{fsdp_mode=}, {fsdp_config=}")
    return fsdp_mode, fsdp_config


class _FSDPSentenceTransformerTrainer(SentenceTransformerTrainer):
    """SentenceTransformerTrainer with an FSDP-safe _save override.

    sentence-transformers' BaseTrainer._save ignores the state_dict argument
    passed by Trainer.save_model() and calls model.save_pretrained() directly.
    With FSDP, that hits a sharded model whose embed_tokens.weight is not 2-D,
    and model card generation also calls model.encode() on the sharded model,
    causing the same RuntimeError on every checkpoint write.

    The fix: FSDP.summon_full_params temporarily all-gathers every shard onto
    rank 0 before calling save_pretrained, then reshards when the context exits.
    SentenceTransformer.save_pretrained does not accept a state_dict argument,
    so the state-dict-based approach used by the base HuggingFace Trainer does
    not apply here.
    """

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        resolved_dir: str = output_dir or self.args.output_dir or ""
        os.makedirs(resolved_dir, exist_ok=True)

        if not self.is_fsdp_enabled:
            super()._save(resolved_dir, state_dict=state_dict)
            return

        # SentenceTransformer.save_pretrained does not accept a state_dict arg.
        # summon_full_params temporarily all-gathers every FSDP shard onto rank 0
        # (kept on GPU so model buffers and parameters share the same device for
        # model.encode() called during model card generation). Resharding happens
        # automatically on context exit.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(
            self.model,
            writeback=False,
            offload_to_cpu=False,
            rank0_only=True,
        ):
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(resolved_dir)
                torch.save(self.args, os.path.join(resolved_dir, "training_args.bin"))
        log.info(f"{resolved_dir=}")

    def _clear_model_gradients(self) -> None:
        """Clears model gradients before FSDP full-parameter save."""
        _clear_gradients(self.model)


def _build_training_args(config: TrainConfig) -> SentenceTransformerTrainingArguments:
    """Builds SentenceTransformerTrainingArguments from TrainConfig."""
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    log.info(
        f"{config.batch_size=}, {config.gradient_accumulation_steps=}, {effective_batch=}, "
        f"{config.bf16=}, {config.gradient_checkpointing=}, {config.fsdp_mode=}"
    )
    fsdp_mode, fsdp_config = _build_fsdp_config(config)
    save_steps = config.checkpoint_save_steps if config.save_strategy == "steps" else None
    eval_steps = config.evaluation_steps if config.eval_strategy == "steps" else None
    log.info(
        f"{config.save_strategy=}, {config.checkpoint_save_steps=}, {save_steps=}, "
        f"{config.eval_strategy=}, {eval_steps=}"
    )
    return SentenceTransformerTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=eval_steps,  # type: ignore[arg-type]  # None is valid when eval_strategy="epoch"
        save_strategy=config.save_strategy,
        save_steps=save_steps,  # type: ignore[arg-type]  # None is valid when save_strategy="epoch"
        save_total_limit=None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.run_name,
        fsdp=fsdp_mode,
        fsdp_config=fsdp_config,
    )


def _train_model(
    model: SentenceTransformer,
    train_dataset: Dataset,
    train_loss: losses.TripletLoss | losses.MultipleNegativesRankingLoss,
    config: TrainConfig,
    evaluator: SentenceEvaluator | None,
) -> None:
    """Runs SentenceTransformerTrainer fit loop."""
    training_args = _build_training_args(config)
    trainer = _FSDPSentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    log.info(f"{config.output_dir=}")


def train(
    train_parquet: str,
    validation_parquet: str,
    output_dir: str = "artifacts/sentence_transformers/qwen3-summary",
    base_model_name: str = _DEFAULT_BASE_MODEL,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_steps: int = 200,
    eval_strategy: str = "epoch",
    evaluation_steps: int = 200,
    save_strategy: str = "epoch",
    checkpoint_save_steps: int = 200,
    seed: int = 13,
    use_hard_negatives: bool = True,
    triplet_margin: float = 0.2,
    gradient_accumulation_steps: int = 1,
    bf16: bool = True,
    gradient_checkpointing: bool = False,
    dataloader_num_workers: int = 0,
    use_wandb: bool = False,
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name: str = "qwen3-parquet-train",
    fsdp_mode: str = "",
    fsdp_use_orig_params: bool = True,
    fsdp_cpu_ram_efficient_loading: bool = True,
    fsdp_activation_checkpointing: bool = False,
) -> None:
    """Trains sentence-transformer from parquet anchors and summaries."""
    config = TrainConfig(
        train_parquet=train_parquet,
        validation_parquet=validation_parquet,
        output_dir=output_dir,
        base_model_name=base_model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        eval_strategy=_parse_eval_strategy(eval_strategy),
        evaluation_steps=evaluation_steps,
        save_strategy=_parse_save_strategy(save_strategy),
        checkpoint_save_steps=checkpoint_save_steps,
        seed=seed,
        use_hard_negatives=use_hard_negatives,
        triplet_margin=triplet_margin,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_name=run_name,
        fsdp_mode=fsdp_mode,
        fsdp_use_orig_params=fsdp_use_orig_params,
        fsdp_cpu_ram_efficient_loading=fsdp_cpu_ram_efficient_loading,
        fsdp_activation_checkpointing=fsdp_activation_checkpointing,
    )
    log.info(f"{config=}")

    if config.use_wandb:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        if config.wandb_entity:
            os.environ["WANDB_ENTITY"] = config.wandb_entity

    _set_random_seed(config.seed)

    parsed = _read_parquet_rows(config.train_parquet)
    log.info(f"{len(parsed.rows)=}, {parsed.dropped_rows=}")

    train_dataset = _build_hf_dataset(parsed.rows, config.use_hard_negatives)

    model = SentenceTransformer(config.base_model_name)
    train_loss = _build_loss(model, config.use_hard_negatives, config.triplet_margin)

    validation_rows = _read_parquet_rows(config.validation_parquet).rows
    validation_dataset = _build_retrieval_dataset(validation_rows, split_name="validation")
    evaluator = _ParquetValidationEvaluator(
        validation_dataset=validation_dataset,
        use_wandb=config.use_wandb,
    )

    _train_model(
        model=model,
        train_dataset=train_dataset,
        train_loss=train_loss,
        config=config,
        evaluator=evaluator,
    )


def train_from_config(config_path: str = "configs/train.config.yaml") -> None:
    """Loads YAML config and runs ``train`` from nested or legacy sections."""
    payload = _read_yaml(Path(config_path))
    train_kwargs = _train_kwargs_from_config(payload)
    train(**train_kwargs)


def main() -> None:
    """CLI entrypoint."""
    fire.Fire({"train": train, "train_from_config": train_from_config})


if __name__ == "__main__":
    main()
