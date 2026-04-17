from __future__ import annotations

from dataclasses import dataclass

import pydantic


class ModelOutput(pydantic.BaseModel):
    """Structured LLM output for summary validation and question filtering."""

    model_config = pydantic.ConfigDict(extra="forbid")

    reasoning: str
    filtered_summary: str
    filtered_questions: list[str] = pydantic.Field(
        description="Exactly 5 questions copied verbatim from seed or scenario candidates.",
        min_length=5,
        max_length=5,
    )


@dataclass(frozen=True)
class TrainingData:
    """One training/evaluation row used by retriever fine-tuning."""

    name: str
    markdown_content: str
    summary: str
    description: str
    scenario_training_data: list[ScenarioTrainingData]
    seed_questions_training_data: list[SeedQuestionsTrainingData]

@dataclass()
class ScenarioTrainingData:
    question: str
    in_batch_negatives_question_wrt_descriptions: list[str]
    in_batch_negatives_question_wrt_summaries: list[str]
    in_batch_negatives_question_wrt_scenarios: list[str]
    scenario: str


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
    filtered_summary: str
    description: str
    filtered_questions: list[str]
    num_from_seed_questions: str
    num_from_scenario_questions: str