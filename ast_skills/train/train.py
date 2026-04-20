"""Sentence-transformer trainer for mined-negative parquet datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import fire
import pandas as pd
import torch
import wandb
import yaml
from loguru import logger as log
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from ast_skills.train.datamodels import TrainingParquetRow

_DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class _ParsedTrainingData(NamedTuple):
    """Parsed rows and dropped-count metadata for parquet loading."""

    rows: list[TrainingParquetRow]
    dropped_rows: int


@dataclass(frozen=True)
class TrainConfig:
    """Public configuration for parquet-based training."""

    train_parquet: str
    output_dir: str = "artifacts/sentence_transformers/qwen3-summary"
    base_model_name: str = _DEFAULT_BASE_MODEL
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 200
    seed: int = 13
    use_hard_negatives: bool = False
    triplet_margin: float = 0.2
    wandb_project: str = "ast-skills-retriever"
    wandb_entity: str = ""
    run_name: str = "qwen3-parquet-train"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Reads a YAML config file as a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML config root must be a mapping.")
    log.info(f"{path=}, {list(payload.keys())=}")
    return payload


def _train_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extracts kwargs for ``train`` from config payload."""
    train_config = config.get("train", {})
    if not isinstance(train_config, dict):
        raise ValueError("`train` section must be a mapping.")
    kwargs = dict(train_config)
    log.info(f"{kwargs=}")
    return kwargs


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

    positive_summary = str(record.get("positive_summary", "")).strip()
    if not positive_summary:
        positive_summary = str(record.get("summary", "")).strip()

    if not question or not positive_summary:
        return None

    hard_negatives = _normalize_hard_negatives(record)
    return TrainingParquetRow(
        question=question,
        positive_summary=positive_summary,
        hard_negatives=hard_negatives,
    )


def _normalize_hard_negatives(record: dict[str, Any]) -> list[str]:
    """Reads optional hard negatives from known parquet columns."""
    candidates: list[str] = []
    for key in ("hard_negatives", "in_batch_negatives_summary", "in_batch_negatives_descriptions"):
        value = record.get(key, [])
        if not isinstance(value, list):
            continue
        for item in value:
            text = str(item).strip()
            if text:
                candidates.append(text)
    deduplicated = list(dict.fromkeys(candidates))
    return deduplicated


def _build_pair_examples(rows: list[TrainingParquetRow]) -> list[InputExample]:
    """Builds ``InputExample(question, positive_summary)`` rows."""
    examples = [InputExample(texts=[row.question, row.positive_summary]) for row in rows]
    log.info(f"{len(examples)=}")
    return examples


def _build_triplet_examples(rows: list[TrainingParquetRow]) -> list[InputExample]:
    """Builds triplet ``InputExample(question, positive, negative)`` rows."""
    examples: list[InputExample] = []
    for row in rows:
        if not row.hard_negatives:
            continue
        negative = row.hard_negatives[0]
        examples.append(InputExample(texts=[row.question, row.positive_summary, negative]))
    log.info(f"{len(examples)=}")
    return examples


def _build_dataloader(examples: list[InputExample], batch_size: int) -> DataLoader:
    """Builds deterministic dataloader without sample shuffling."""
    if len(examples) < batch_size:
        raise ValueError("Not enough examples for one full batch; lower batch_size.")
    dataloader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    log.info(f"{batch_size=}, {len(dataloader)=}")
    return dataloader


def _build_loss(
    model: SentenceTransformer,
    use_hard_negatives: bool,
    triplet_margin: float,
):
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


def _build_train_examples(
    rows: list[TrainingParquetRow],
    use_hard_negatives: bool,
) -> list[InputExample]:
    """Builds pair or triplet examples based on configuration."""
    if use_hard_negatives:
        examples = _build_triplet_examples(rows)
        if not examples:
            raise ValueError("use_hard_negatives=True but no rows contain hard negatives.")
        return examples
    return _build_pair_examples(rows)


def _build_wandb_config(config: TrainConfig, row_count: int, dropped_rows: int) -> dict[str, Any]:
    """Creates W&B config payload."""
    payload = asdict(config)
    payload["row_count"] = row_count
    payload["dropped_rows"] = dropped_rows
    log.info(f"{payload=}")
    return payload


def _train_model(
    model: SentenceTransformer,
    dataloader: DataLoader,
    train_loss: Any,
    config: TrainConfig,
) -> None:
    """Runs sentence-transformer fit loop."""
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        show_progress_bar=True,
        output_path=config.output_dir,
        checkpoint_path=config.output_dir,
        checkpoint_save_total_limit=2,
    )
    model.save(config.output_dir)
    log.info(f"{config.output_dir=}")


def train(
    train_parquet: str,
    output_dir: str = "artifacts/sentence_transformers/qwen3-summary",
    base_model_name: str = _DEFAULT_BASE_MODEL,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_steps: int = 200,
    seed: int = 13,
    use_hard_negatives: bool = False,
    triplet_margin: float = 0.2,
    wandb_project: str = "ast-skills-retriever",
    wandb_entity: str = "",
    run_name: str = "qwen3-parquet-train",
) -> None:
    """Trains sentence-transformer from parquet anchors and summaries."""
    config = TrainConfig(
        train_parquet=train_parquet,
        output_dir=output_dir,
        base_model_name=base_model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        seed=seed,
        use_hard_negatives=use_hard_negatives,
        triplet_margin=triplet_margin,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_name=run_name,
    )
    log.info(f"{config=}")

    _set_random_seed(config.seed)
    parsed = _read_parquet_rows(config.train_parquet)
    examples = _build_train_examples(parsed.rows, config.use_hard_negatives)
    dataloader = _build_dataloader(examples, config.batch_size)

    model = SentenceTransformer(config.base_model_name)
    train_loss = _build_loss(model, config.use_hard_negatives, config.triplet_margin)

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity or None,
        name=config.run_name,
        config=_build_wandb_config(config, row_count=len(parsed.rows), dropped_rows=parsed.dropped_rows),
    )
    try:
        _train_model(model=model, dataloader=dataloader, train_loss=train_loss, config=config)
    finally:
        wandb.finish()


def train_from_config(config_path: str = "configs/train.yaml") -> None:
    """Loads YAML config and runs ``train`` with its ``train`` section."""
    payload = _read_yaml(Path(config_path))
    train_kwargs = _train_kwargs_from_config(payload)
    train(**train_kwargs)


def main() -> None:
    """CLI entrypoint."""
    fire.Fire({"train": train, "train_from_config": train_from_config})


if __name__ == "__main__":
    main()
