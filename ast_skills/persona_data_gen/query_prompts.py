"""Prompt templates for persona-conditioned query generation."""

from __future__ import annotations

from ast_skills.data_gen.skills_data_collect import SkillMdRecord
from ast_skills.persona_data_gen.datamodels import PersonaProfile
from ast_skills.persona_data_gen.template_loader import load_template_text, render_template

QUERY_GENERATION_SYSTEM_PROMPT = load_template_text("query_generation_system.jinja")
QUERY_GENERATION_USER_PROMPT_TEMPLATE = load_template_text("query_generation_user.jinja")


def build_persona_query_prompt(skill_record: SkillMdRecord, persona: PersonaProfile) -> str:
    """Build a user prompt for query generation for one persona and skill."""
    skill_name = skill_record.metadata.get("name") or skill_record.relative_path
    variables = {
        "skill_name": skill_name,
        "role": persona.role,
        "expertise_level": persona.expertise_level,
        "domain_context": persona.domain_context,
        "intent_type": persona.intent_type,
        "scenario": persona.scenario,
        "content": skill_record.content,
    }
    return render_template(QUERY_GENERATION_USER_PROMPT_TEMPLATE, variables)
