"""Build JSONL prompt inputs for persona and query generation workflows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

import fire
from loguru import logger

from ast_skills.data_gen.skills_data_collect import (
    SkillMdRecord,
    collect_english_skill_md_records,
)
from ast_skills.persona_data_gen.datamodels import (
    PersonaGenerationPromptRow,
    PersonaProfile,
    PersonaQueryPromptRow,
)
from ast_skills.persona_data_gen.persona_prompts import (
    PERSONA_GENERATION_SYSTEM_PROMPT,
    build_persona_generation_prompt,
)
from ast_skills.persona_data_gen.query_prompts import (
    QUERY_GENERATION_SYSTEM_PROMPT,
    build_persona_query_prompt,
)


class _PersonaJsonlRow(NamedTuple):
    """Parsed persona output row with skill metadata and persona list."""

    relative_path: str
    skill_name: str
    personas: list[PersonaProfile]


def load_filtered_skill_md_records(skills_root: str) -> list[SkillMdRecord]:
    """Load skill records using the shared data_gen collector and token filtering."""
    records = collect_english_skill_md_records(skills_root)
    logger.info(f"{skills_root=}")
    logger.info(f"{len(records)=}")
    return records


def _skill_name(skill_record: SkillMdRecord) -> str:
    """Get a skill display name from metadata with a stable fallback."""
    return skill_record.metadata.get("name") or skill_record.relative_path


def build_persona_generation_prompt_rows(
    skill_md_records: list[SkillMdRecord],
) -> list[PersonaGenerationPromptRow]:
    """Build one persona-generation prompt row per skill markdown record."""
    rows: list[PersonaGenerationPromptRow] = []
    for index, skill_md_record in enumerate(skill_md_records):
        custom_id = f"persona-{index}"
        row = PersonaGenerationPromptRow(
            custom_id=custom_id,
            relative_path=skill_md_record.relative_path,
            skill_name=_skill_name(skill_md_record),
            prompt=build_persona_generation_prompt(skill_md_record),
        )
        rows.append(row)
    logger.info(f"{len(rows)=}")
    return rows


def write_persona_generation_prompts_jsonl(
    rows: list[PersonaGenerationPromptRow],
    output_path: str,
) -> None:
    """Write persona-generation prompts to JSONL for batch or offline usage."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            payload = {
                "custom_id": row.custom_id,
                "relative_path": row.relative_path,
                "skill_name": row.skill_name,
                "system_prompt": PERSONA_GENERATION_SYSTEM_PROMPT,
                "user_prompt": row.prompt,
            }
            output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    logger.info(f"{output_path=}")
    logger.info(f"{len(rows)=}")


def _parse_persona_profile(raw_persona: dict) -> PersonaProfile:
    """Parse one raw persona object into ``PersonaProfile``."""
    return PersonaProfile(
        role=str(raw_persona.get("role", "")).strip(),
        expertise_level=str(raw_persona.get("expertise_level", "")).strip(),
        domain_context=str(raw_persona.get("domain_context", "")).strip(),
        intent_type=str(raw_persona.get("intent_type", "")).strip(),
        scenario=str(raw_persona.get("scenario", "")).strip(),
    )


def _persona_from_string(raw_persona: str) -> PersonaProfile:
    """Convert a plain-text persona string into a minimally structured profile."""
    text = raw_persona.strip()
    return PersonaProfile(
        role="unspecified",
        expertise_level="unspecified",
        domain_context="unspecified",
        intent_type="unspecified",
        scenario=text,
    )


def _parse_persona_jsonl_line(raw_row: dict) -> _PersonaJsonlRow:
    """Parse one persona JSONL row supporting both structured and list[str] personas."""
    relative_path = str(raw_row.get("relative_path", "")).strip()
    skill_name = str(raw_row.get("skill_name", "")).strip() or relative_path
    raw_personas = raw_row.get("personas", [])

    personas: list[PersonaProfile] = []
    if isinstance(raw_personas, list):
        for raw_persona in raw_personas:
            if isinstance(raw_persona, dict):
                personas.append(_parse_persona_profile(raw_persona))
            elif isinstance(raw_persona, str):
                personas.append(_persona_from_string(raw_persona))

    if not personas:
        personas_text = _extract_personas_text(raw_row)
        personas = _parse_persona_list_text(personas_text)

    return _PersonaJsonlRow(
        relative_path=relative_path,
        skill_name=skill_name,
        personas=personas,
    )


def _extract_personas_text(raw_row: dict) -> str:
    """Extract persona text from common output keys in a JSONL row."""
    text_keys = ["personas_text", "response_text", "model_output", "personas"]
    for key in text_keys:
        value = raw_row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _clean_persona_line(line: str) -> str:
    """Normalize one raw persona line by removing numbering and markdown bullets."""
    cleaned = line.strip()
    cleaned = cleaned.lstrip("-*").strip()

    if ". " in cleaned:
        prefix, suffix = cleaned.split(". ", maxsplit=1)
        if prefix.isdigit() and suffix:
            return suffix.strip()
    return cleaned


def _parse_persona_list_text(personas_text: str) -> list[PersonaProfile]:
    """Parse a numbered or bulleted persona list from natural-language text."""
    blocks = _split_persona_text_blocks(personas_text)
    personas: list[PersonaProfile] = []

    for block in blocks:
        cleaned = _clean_persona_line(block)
        if not cleaned:
            continue
        personas.append(_persona_from_string(cleaned))

    logger.info(f"{len(personas)=}")
    return personas


def _split_persona_text_blocks(personas_text: str) -> list[str]:
    """Split model output into persona-sized text blocks."""
    numbered_item_re = re.compile(r"(?:^|\n)\s*\d+\.\s+")
    has_numbered_items = bool(numbered_item_re.search(personas_text))
    if has_numbered_items:
        parts = numbered_item_re.split(personas_text)
        return [part.strip() for part in parts if part.strip()]

    paragraph_parts = personas_text.split("\n\n")
    paragraphs = [part.strip() for part in paragraph_parts if part.strip()]
    if paragraphs:
        return paragraphs

    lines = [line.strip() for line in personas_text.splitlines() if line.strip()]
    return lines


def read_persona_generation_output_jsonl(input_path: str) -> list[_PersonaJsonlRow]:
    """Load persona generation outputs from JSONL."""
    path = Path(input_path)
    parsed_rows: list[_PersonaJsonlRow] = []

    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            raw_row = json.loads(stripped)
            parsed_row = _parse_persona_jsonl_line(raw_row)
            if parsed_row.relative_path and parsed_row.personas:
                parsed_rows.append(parsed_row)

    logger.info(f"{input_path=}")
    logger.info(f"{len(parsed_rows)=}")
    return parsed_rows


def _map_skill_records_by_relative_path(
    skill_md_records: list[SkillMdRecord],
) -> dict[str, SkillMdRecord]:
    """Index skill records by relative path for fast joins with persona output."""
    mapping = {record.relative_path: record for record in skill_md_records}
    logger.info(f"{len(mapping)=}")
    return mapping


def build_persona_query_prompt_rows(
    skill_md_records: list[SkillMdRecord],
    persona_rows: list[_PersonaJsonlRow],
) -> list[PersonaQueryPromptRow]:
    """Build one query-generation prompt row per (persona, skill) pair."""
    skill_records_by_path = _map_skill_records_by_relative_path(skill_md_records)
    rows: list[PersonaQueryPromptRow] = []

    for persona_row in persona_rows:
        skill_record = skill_records_by_path.get(persona_row.relative_path)
        if skill_record is None:
            logger.warning(f"Missing skill for {persona_row.relative_path=}")
            continue

        for persona_index, persona in enumerate(persona_row.personas):
            custom_id = f"query-{persona_row.relative_path}-{persona_index}"
            row = PersonaQueryPromptRow(
                custom_id=custom_id,
                relative_path=persona_row.relative_path,
                skill_name=persona_row.skill_name,
                persona=persona,
                prompt=build_persona_query_prompt(skill_record, persona),
            )
            rows.append(row)

    logger.info(f"{len(rows)=}")
    return rows


def write_persona_query_prompts_jsonl(
    rows: list[PersonaQueryPromptRow],
    output_path: str,
) -> None:
    """Write persona-conditioned query prompts to JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            payload = {
                "custom_id": row.custom_id,
                "relative_path": row.relative_path,
                "skill_name": row.skill_name,
                "persona": {
                    "role": row.persona.role,
                    "expertise_level": row.persona.expertise_level,
                    "domain_context": row.persona.domain_context,
                    "intent_type": row.persona.intent_type,
                    "scenario": row.persona.scenario,
                },
                "system_prompt": QUERY_GENERATION_SYSTEM_PROMPT,
                "user_prompt": row.prompt,
            }
            output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    logger.info(f"{output_path=}")
    logger.info(f"{len(rows)=}")


class PersonaPromptJobs:
    """CLI jobs for persona and persona-conditioned query prompt generation."""

    def build_persona_prompts(self, skills_root: str, output_path: str) -> None:
        """Generate JSONL prompts for persona generation from SKILL.md files."""
        skill_md_records = load_filtered_skill_md_records(skills_root)
        rows = build_persona_generation_prompt_rows(skill_md_records)
        write_persona_generation_prompts_jsonl(rows, output_path)

    def build_query_prompts(
        self,
        skills_root: str,
        persona_jsonl_path: str,
        output_path: str,
    ) -> None:
        """Generate JSONL prompts for query generation from persona outputs."""
        skill_md_records = load_filtered_skill_md_records(skills_root)
        persona_rows = read_persona_generation_output_jsonl(persona_jsonl_path)
        rows = build_persona_query_prompt_rows(skill_md_records, persona_rows)
        write_persona_query_prompts_jsonl(rows, output_path)


if __name__ == "__main__":
    fire.Fire(PersonaPromptJobs)
