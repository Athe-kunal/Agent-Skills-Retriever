"""Public data models for the ``ast_skills.data_gen`` package."""

from __future__ import annotations

from dataclasses import dataclass

import pydantic


@dataclass(frozen=True)
class RetrieverDataModel:
    """One retriever training row: SKILL.md source plus structured extraction."""

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
class TrainingData:
    """One training/evaluation row used by retriever fine-tuning.

    Designed for anchor-point loss: ``question`` is the anchor,
    ``summary`` and ``description`` are the positive documents, and
    ``negative_documents`` holds hard negatives mined from both fields combined.
    """

    question: str
    name: str
    summary: str
    description: str
    negative_documents: list[str]


class SkillMdSummaryExtraction(pydantic.BaseModel):
    """Structured output for detailed SKILL.md summaries."""

    model_config = pydantic.ConfigDict(extra="forbid")

    summary: str
    seed_questions: list[str]


class SkillMdExtraction(pydantic.BaseModel):
    """Structured extraction fields used by retriever dataset generation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    reasoning: str
    what: str
    why: str
    seed_questions: list[str]
