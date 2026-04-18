"""Datamodels for retriever embedding and analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrieverDataModel:
    """One retriever training row plus structured extraction fields."""

    custom_id: str
    markdown_content: str
    reasoning: str
    what: str
    why: str
    seed_questions: list[str]
    name: str
    description: str
    metadata: dict[str, str]
    summary: str = ""


@dataclass(frozen=True)
class SummaryRetrieverDataModel:
    """One retriever training row for summary-based retrieval."""

    custom_id: str
    markdown_content: str
    seed_questions: list[str]
    summary: str
    name: str
    description: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class ValidatedSkillQuestionRow:
    """Input row for validated skill-question retrieval evaluation."""

    custom_id: str
    description: str
    filtered_questions: list[str]
    markdown_content: str
    name: str
    num_from_scenario_questions: str
    num_from_seed_questions: str
    reasoning: str
    summary: str = ""
