from __future__ import annotations

from dataclasses import dataclass

import pydantic


class ModelOutput(pydantic.BaseModel):
    """Structured LLM output for question filtering."""

    model_config = pydantic.ConfigDict(extra="forbid")

    reasoning: str
    filtered_questions: list[str] = pydantic.Field(
        description="Exactly 5 questions copied verbatim from the candidates.",
        min_length=5,
        max_length=5,
    )


@dataclass(frozen=True)
class DataPoint:
    """One training/evaluation row used by retriever fine-tuning."""

    name: str
    markdown_content: str
    summary: str
    description: str
    question: str


@dataclass()
class SeedQuestionsTrainingData:
    question: str
    in_batch_negatives_question_wrt_descriptions: list[str]
    in_batch_negatives_question_wrt_summaries: list[str]

@dataclass()
class ValidatedTrainingData:
    custom_id: str
    name: str
    markdown_content: str
    description: str
    reasoning: str
    filtered_questions: list[str]
    num_from_seed_questions: str
    num_from_scenario_questions: str

@dataclass(frozen=True)
class MinedTrainingDataRow:
    """One output row containing an anchor and mined hard negatives."""

    anchor_id: str
    name: str
    markdown_content: str
    summary: str
    description: str
    question: str
    negative_summaries: list[str]
    negative_descriptions: list[str] | None = None

