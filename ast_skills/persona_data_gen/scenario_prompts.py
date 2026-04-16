"""Prompt templates for scenario and question generation from SKILL.md content."""

from __future__ import annotations

from ast_skills.data_gen.skills_data_collect import SkillMdRecord
from ast_skills.persona_data_gen.template_loader import load_template_text, render_template

SCENARIO_GENERATION_SYSTEM_PROMPT = load_template_text(
    "scenario_generation_system.jinja"
)
SCENARIO_GENERATION_USER_PROMPT_TEMPLATE = load_template_text(
    "scenario_generation_user.jinja"
)


def build_scenario_generation_prompt(skill_record: SkillMdRecord) -> str:
    """Build a user prompt for scenario+question generation for one SKILL.md record."""
    skill_name = skill_record.metadata.get("name") or skill_record.relative_path
    variables = {
        "skill_name": skill_name,
        "content": skill_record.content,
    }
    return render_template(SCENARIO_GENERATION_USER_PROMPT_TEMPLATE, variables)
