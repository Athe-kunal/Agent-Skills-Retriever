"""Public data models for the ``ast_skills.data_gen`` package."""

from __future__ import annotations

from dataclasses import dataclass


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
