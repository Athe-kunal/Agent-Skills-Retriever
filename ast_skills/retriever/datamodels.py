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
    seed_questions: str
    name: str
    description: str
    metadata: dict[str, str]
