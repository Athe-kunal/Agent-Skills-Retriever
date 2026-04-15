"""Public data models for persona-driven retrieval data generation."""

from __future__ import annotations

from dataclasses import dataclass

import pydantic


@dataclass(frozen=True)
class PersonaProfile:
    """One persona/scenario inferred from a SKILL.md file."""

    role: str
    expertise_level: str
    domain_context: str
    intent_type: str
    scenario: str


@dataclass(frozen=True)
class PersonaGenerationPromptRow:
    """Prompt row for generating personas from a SKILL.md file."""

    custom_id: str
    relative_path: str
    skill_name: str
    prompt: str


@dataclass(frozen=True)
class PersonaQueryPromptRow:
    """Prompt row for generating one query from one persona + SKILL.md."""

    custom_id: str
    relative_path: str
    skill_name: str
    persona: PersonaProfile
    prompt: str


class PersonaGeneration(pydantic.BaseModel):
    """Structured model output for persona generation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    personas: list[str]
