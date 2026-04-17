"""Public data models for scenario-driven retrieval data generation."""

from __future__ import annotations

from dataclasses import dataclass

import pydantic


@dataclass(frozen=True)
class ScenarioQueryPromptRow:
    """Prompt row for generating scenario + question outputs from one SKILL.md file."""

    custom_id: str
    relative_path: str
    skill_name: str
    prompt: str


class ScenarioRelatedOutput(pydantic.BaseModel):
    """One scenario with its matching user question."""

    model_config = pydantic.ConfigDict(extra="forbid")

    scenario: str
    question: str

@dataclass(frozen=True)
class ScenarioQueryPromptRowDataModel:
    """One retriever training row for summary-based retrieval."""

    custom_id: str
    markdown_content: str
    seed_questions: list[str]
    summary: str
    name: str
    description: str
    metadata: dict[str, str]
    scenario_output: list[ScenarioRelatedOutput]

class OpenAIOutput(pydantic.BaseModel):
    """Structured model output for scenario-conditioned question generation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    scenario_output: list[ScenarioRelatedOutput] = pydantic.Field(
        min_length=5,
        max_length=5,
    )
