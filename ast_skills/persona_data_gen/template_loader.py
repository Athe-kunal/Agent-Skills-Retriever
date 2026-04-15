"""Utilities for loading and rendering local Jinja template files."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

_TEMPLATE_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def load_template_text(template_name: str) -> str:
    """Load a template file from the local persona_data_gen template directory."""
    template_path = _TEMPLATE_DIR / template_name
    if not template_path.is_file():
        raise FileNotFoundError(f"Missing template file: {template_path}")
    template_text = template_path.read_text(encoding="utf-8")
    logger.info(f"{template_path=}")
    return template_text


def _replace_template_match(match: re.Match[str], variables: dict[str, str]) -> str:
    """Resolve one ``{{ variable }}`` template token."""
    variable_name = match.group(1)
    return variables.get(variable_name, "")


def render_template(template_text: str, variables: dict[str, str]) -> str:
    """Render a subset of Jinja syntax by replacing ``{{ variable }}`` placeholders."""
    rendered = template_text
    for match in _TEMPLATE_VAR_RE.finditer(template_text):
        matched_text = match.group(0)
        replacement = _replace_template_match(match, variables)
        rendered = rendered.replace(matched_text, replacement)
    logger.info(f"{len(variables)=}")
    return rendered
