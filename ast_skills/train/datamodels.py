from __future__ import annotations

from dataclasses import dataclass

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
