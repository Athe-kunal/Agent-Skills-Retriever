"""
Tests for parse_skill_md_frontmatter and its private helpers.

Scenarios covered
-----------------
Fenced (--- ... ---) format
  1.  Full fenced block with name, description, and extra keys.
  2.  Fenced block followed by a markdown body.
  3.  Fenced block with CRLF line endings.
  4.  Fenced block with only the opening --- (no closing delimiter).
  5.  Fenced block with a multi-line description value.
  6.  Fenced block with nested YAML (list / dict) — stored as JSON string.
  7.  Fenced block with missing name key.
  8.  Fenced block with missing description key.

Raw / delimiter-free format
  9.  Raw YAML with name and description, no --- delimiters.
  10. Raw YAML with name only.
  11. Raw YAML with description only.
  12. Raw YAML with extra keys beyond name / description.

Edge / degenerate cases
  13. Completely empty string.
  14. Pure markdown body with no metadata whatsoever.
  15. BOM-prefixed content (UTF-8 BOM before the first ---).
  16. Malformed YAML inside fenced block — still extracts name / description via regex.
  17. name / description values with leading and trailing whitespace.
"""

import pytest

from ast_skills.data_gen.skills_data_collect import (
    _extract_description,
    _extract_fenced_block,
    _extract_name,
    parse_skill_md_frontmatter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta_and_body(raw: str) -> tuple[dict[str, str], str]:
    """Thin alias so tests read like prose."""
    return parse_skill_md_frontmatter(raw)


# ---------------------------------------------------------------------------
# Fenced (--- ... ---) format
# ---------------------------------------------------------------------------


class TestFencedFormat:
    def test_full_fenced_block_name_description_extra_key(self):
        raw = "---\nname: my-skill\ndescription: Does cool things\nauthor: alice\n---\n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "my-skill"
        assert meta["description"] == "Does cool things"
        assert meta["author"] == "alice"
        assert body == ""

    def test_fenced_block_followed_by_markdown_body(self):
        raw = (
            "---\n"
            "name: my-skill\n"
            "description: Does cool things\n"
            "---\n"
            "# Usage\n\n"
            "Call me maybe.\n"
        )
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "my-skill"
        assert meta["description"] == "Does cool things"
        assert "# Usage" in body
        assert "Call me maybe." in body

    def test_fenced_block_crlf_line_endings(self):
        raw = "---\r\nname: crlf-skill\r\ndescription: Windows-style\r\n---\r\n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "crlf-skill"
        assert meta["description"] == "Windows-style"

    def test_fenced_block_only_opening_delimiter(self):
        raw = "---\nname: open-only\ndescription: No closing dash\n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "open-only"
        assert meta["description"] == "No closing dash"
        assert body == ""

    def test_fenced_block_multiline_description(self):
        raw = (
            "---\n"
            "name: multi-line\n"
            "description: |-\n"
            "  First line of description.\n"
            "  Second line of description.\n"
            "version: 1.0\n"
            "---\n"
        )
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "multi-line"
        assert "First line" in meta["description"]
        assert "Second line" in meta["description"]
        assert meta["version"] == "1.0"

    def test_fenced_block_nested_yaml_stored_as_json(self):
        raw = (
            "---\n"
            "name: nested\n"
            "description: Has nested data\n"
            "tags:\n"
            "  - python\n"
            "  - ai\n"
            "---\n"
        )
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "nested"
        # List value must be JSON-encoded into a string.
        import json

        tags = json.loads(meta["tags"])
        assert tags == ["python", "ai"]

    def test_fenced_block_missing_name_defaults_to_empty(self):
        raw = "---\ndescription: No name here\nauthor: bob\n---\n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == ""
        assert meta["description"] == "No name here"

    def test_fenced_block_missing_description_defaults_to_empty(self):
        raw = "---\nname: no-desc-skill\nauthor: bob\n---\n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "no-desc-skill"
        assert meta["description"] == ""


# ---------------------------------------------------------------------------
# Raw / delimiter-free format
# ---------------------------------------------------------------------------


class TestRawYamlFormat:
    def test_raw_yaml_name_and_description(self):
        raw = "name: welight-wechat-layout-publish\ndescription: Turns articles into WeChat HTML\n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "welight-wechat-layout-publish"
        assert meta["description"] == "Turns articles into WeChat HTML"

    def test_raw_yaml_name_only(self):
        raw = "name: just-a-name\n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "just-a-name"
        assert meta["description"] == ""

    def test_raw_yaml_description_only(self):
        raw = "description: Only a description here\n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == ""
        assert meta["description"] == "Only a description here"

    def test_raw_yaml_extra_keys_beyond_name_and_description(self):
        raw = (
            "name: extra-keys\n"
            "description: Has more keys\n"
            "version: 2.3\n"
            "author: carol\n"
        )
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "extra-keys"
        assert meta["description"] == "Has more keys"
        assert meta["version"] == "2.3"
        assert meta["author"] == "carol"


# ---------------------------------------------------------------------------
# Edge / degenerate cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self):
        meta, body = _meta_and_body("")

        assert meta["name"] == ""
        assert meta["description"] == ""
        assert body == ""

    def test_pure_markdown_no_metadata(self):
        raw = "# My Skill\n\nThis skill does something useful.\n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == ""
        assert meta["description"] == ""
        # Body should contain the raw markdown when no metadata was found.
        assert "# My Skill" in body

    def test_bom_prefix_stripped_before_parsing(self):
        raw = "\ufeff---\nname: bom-skill\ndescription: BOM at start\n---\n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "bom-skill"
        assert meta["description"] == "BOM at start"

    def test_malformed_yaml_in_fenced_block_still_extracts_via_regex(self):
        # The YAML is broken (unbalanced quotes) but name/description are
        # still recoverable by direct regex matching.
        raw = (
            "---\n"
            "name: broken-yaml\n"
            'description: "unclosed quote\n'
            "---\n"
        )
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "broken-yaml"
        # description may be empty or partial — what matters is no exception.
        assert "name" in meta
        assert "description" in meta

    def test_whitespace_trimmed_from_name_and_description(self):
        raw = "name:   padded-name   \ndescription:   padded description   \n"
        meta, body = _meta_and_body(raw)

        assert meta["name"] == "padded-name"
        assert meta["description"] == "padded description"


# ---------------------------------------------------------------------------
# Unit tests for private helpers
# ---------------------------------------------------------------------------


class TestExtractFencedBlock:
    def test_both_delimiters(self):
        text = "---\nfoo: bar\n---\nbody here\n"
        yaml_block, body = _extract_fenced_block(text)

        assert yaml_block == "foo: bar"
        assert body == "body here\n"

    def test_only_opening_delimiter(self):
        yaml_block, body = _extract_fenced_block("---\nfoo: bar\n")

        assert "foo: bar" in yaml_block
        assert body == ""

    def test_no_delimiter(self):
        yaml_block, body = _extract_fenced_block("just markdown\n")

        assert yaml_block == ""
        assert body == ""


class TestExtractName:
    def test_finds_name(self):
        assert _extract_name("name: hello-world\nother: x\n") == "hello-world"

    def test_returns_empty_when_absent(self):
        assert _extract_name("description: something\n") == ""

    def test_ignores_indented_name(self):
        # Indented lines are not top-level keys — should not match.
        assert _extract_name("  name: indented\n") == ""


class TestExtractDescription:
    def test_single_line_description(self):
        result = _extract_description("name: x\ndescription: A short desc\nauthor: y\n")
        assert result == "A short desc"

    def test_description_stops_at_next_key(self):
        raw = "description: First line\nauthor: someone\n"
        result = _extract_description(raw)
        assert result == "First line"
        assert "author" not in result

    def test_returns_empty_when_absent(self):
        assert _extract_description("name: only-name\n") == ""
