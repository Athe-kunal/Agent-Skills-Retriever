from __future__ import annotations

import asyncio
import dataclasses
import json
import random
from pathlib import Path

import pandas as pd

from ast_skills.retriever.datamodels import ValidatedSkillQuestionRow
from ast_skills.retriever.maximal_marginal_relevance_question import (
    MmrSelectionConfig,
    select_diverse_questions_for_rows,
)
from ast_skills.train.datamodels import DataPoint


def _load_rows(path: str = "artifacts/validated_training_data_list.jsonl") -> list[ValidatedSkillQuestionRow]:
    rows: list[ValidatedSkillQuestionRow] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append(ValidatedSkillQuestionRow(**d))
    return rows


def _datapoints_to_dataframe(points: list[DataPoint]) -> pd.DataFrame:
    return pd.DataFrame([dataclasses.asdict(p) for p in points])


def _row_to_datapoint(row: ValidatedSkillQuestionRow, question: str) -> DataPoint:
    return DataPoint(
        name=row.name,
        markdown_content=row.markdown_content,
        summary=row.summary,
        description=row.description,
        question=question,
    )


async def build_train_val_parquets(
    config: MmrSelectionConfig,
    input_path: str = "artifacts/validated_training_data_list.jsonl",
    output_dir: str = "artifacts",
    random_seed: int | None = None,
) -> None:
    rows = _load_rows(input_path)
    mmr_config = config._replace(selected_question_count=3)
    diverse_by_id = await select_diverse_questions_for_rows(rows=rows, config=mmr_config)

    rng = random.Random(random_seed) if random_seed is not None else random.Random()
    train_points: list[DataPoint] = []
    val_points: list[DataPoint] = []

    for row in rows:
        mmr_questions = diverse_by_id.get(row.custom_id, [])
        if len(mmr_questions) < 3:
            continue
        shuffled = list(mmr_questions)
        rng.shuffle(shuffled)
        train_qs, val_q = shuffled[:2], shuffled[2]
        train_points.extend(_row_to_datapoint(row, q) for q in train_qs)
        val_points.append(_row_to_datapoint(row, val_q))

    output = Path(output_dir)
    _datapoints_to_dataframe(train_points).to_parquet(output / "train.parquet", index=False)
    _datapoints_to_dataframe(val_points).to_parquet(output / "val.parquet", index=False)


def _main(
    base_url: str = "http://127.0.0.1:8001/v1",
    api_key: str = "EMPTY",
    embedding_model: str = "Qwen/Qwen3-Embedding-8B",
    mmr_lambda: float = 0.5,
    batch_size: int = 64,
    max_concurrency: int = 32,
    input_path: str = "artifacts/validated_training_data_list.jsonl",
    output_dir: str = "artifacts",
    random_seed: int | None = None,
) -> None:
    config = MmrSelectionConfig(
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
        mmr_lambda=mmr_lambda,
        selected_question_count=3,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )
    asyncio.run(
        build_train_val_parquets(
            config=config,
            input_path=input_path,
            output_dir=output_dir,
            random_seed=random_seed,
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire(_main)
