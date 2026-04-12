from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger as log
from pydantic import ValidationError

from ast_skills.data_gen.synthetic_data_gen import SkillMdExtraction


def _coerce_skill_md_metadata(raw: object) -> dict[str, str]:
    """
    Normalize metadata loaded from JSON (e.g. JSONL) to dict[str, str].

    Same behavior as ``skills_data_collect.coerce_skill_md_metadata`` (kept here so this module
    does not import optional heavy dependencies pulled in by that module).
    """
    if not isinstance(raw, dict):
        return {"name": "", "description": ""}
    out: dict[str, str] = {}
    for key, val in raw.items():
        k = str(key)
        if isinstance(val, str):
            out[k] = val
        elif val is None:
            out[k] = ""
        elif isinstance(val, (dict, list)):
            out[k] = json.dumps(val, ensure_ascii=False, sort_keys=True, default=str)
        else:
            out[k] = str(val)
    out.setdefault("name", "")
    out.setdefault("description", "")
    return out


class BatchTokenUsage(NamedTuple):
    """Token usage from an OpenAI batch output row."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class SkillMdBatchRecord:
    """SKILL.md fields plus batch prompt, usage, and parsed model extraction."""

    relative_path: str
    content: str
    metadata: dict[str, str]
    custom_id: str
    prompt: list[dict[str, Any]]
    usage: BatchTokenUsage | None
    extraction: SkillMdExtraction | None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file; each non-empty line is one JSON object."""
    path_obj = Path(path)
    records: list[dict[str, Any]] = []
    with path_obj.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_jsonl_paths(paths: list[str]) -> list[dict[str, Any]]:
    """Read many JSONL files and return rows in file order (paths order, then line order)."""
    combined: list[dict[str, Any]] = []
    for path in paths:
        combined.extend(read_jsonl(path))
    return combined


def index_rows_by_custom_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Map custom_id to row.

    If the same ``custom_id`` appears more than once, the last row wins (later files / lines
    override earlier ones).
    """
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            continue
        index[custom_id] = row
    return index


def usage_from_batch_output_row(row: dict[str, Any]) -> BatchTokenUsage | None:
    """Extract token usage from a batch API output JSONL row."""
    response = row.get("response")
    if not isinstance(response, dict):
        return None
    body = response.get("body")
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    return BatchTokenUsage(
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        total_tokens=int(usage.get("total_tokens", 0)),
    )


def extraction_from_batch_output_row(row: dict[str, Any]) -> SkillMdExtraction | None:
    """Parse and validate the assistant message as ``SkillMdExtraction``."""
    if row.get("error") is not None:
        return None
    response = row.get("response")
    if not isinstance(response, dict) or response.get("status_code") != 200:
        return None
    body = response.get("body")
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        return SkillMdExtraction.model_validate(parsed)
    except ValidationError:
        return None


def messages_from_batch_input_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return chat messages from a batch input JSONL row (``body.messages``)."""
    body = row.get("body")
    if not isinstance(body, dict):
        return []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def skill_md_record_row_to_fields(
    row: dict[str, Any],
) -> tuple[str, str, dict[str, str]]:
    """Normalize a skill_md_records JSONL row to (relative_path, content, metadata)."""
    relative_path = row.get("relative_path", "")
    content = row.get("content", "")
    metadata_raw = row.get("metadata", {})
    if not isinstance(relative_path, str):
        relative_path = str(relative_path)
    if not isinstance(content, str):
        content = str(content)
    if not isinstance(metadata_raw, dict):
        metadata_raw = {}
    metadata = _coerce_skill_md_metadata(metadata_raw)
    return relative_path, content, metadata


def join_batch_jsonl_files(
    input_jsonl: list[str],
    output_jsonl: list[str],
    *,
    skill_md_records_jsonl: list[str] | None = None,
) -> list[SkillMdBatchRecord]:
    """
    Join OpenAI batch input and output JSONL files on ``custom_id``.

    ``input_jsonl`` and ``output_jsonl`` are lists of file paths; all rows are read, combined,
    then indexed by ``custom_id`` (last row wins for duplicate ids).

    If ``skill_md_records_jsonl`` is provided, each path is read and merged the same way; joined
    rows use ``relative_path``, ``content``, and ``metadata`` from the matching record when
    present.
    """
    log.info(
        f"{len(input_jsonl)=}, {len(output_jsonl)=}, {skill_md_records_jsonl is not None=}"
    )
    input_rows = read_jsonl_paths(input_jsonl)
    output_rows = read_jsonl_paths(output_jsonl)
    log.info(f"{len(input_rows)=}, {len(output_rows)=}")

    input_by_id = index_rows_by_custom_id(input_rows)
    output_by_id = index_rows_by_custom_id(output_rows)

    records_by_id: dict[str, dict[str, Any]] = {}
    if skill_md_records_jsonl:
        record_rows = read_jsonl_paths(skill_md_records_jsonl)
        records_by_id = index_rows_by_custom_id(record_rows)
        log.info(f"{len(record_rows)=}, {len(records_by_id)=}")

    joined: list[SkillMdBatchRecord] = []
    for custom_id in sorted(input_by_id.keys()):
        input_row = input_by_id[custom_id]
        output_row = output_by_id.get(custom_id)
        record_row = records_by_id.get(custom_id, {})

        relative_path, content, metadata = ("", "", _coerce_skill_md_metadata({}))
        if record_row:
            relative_path, content, metadata = skill_md_record_row_to_fields(record_row)

        prompt = messages_from_batch_input_row(input_row)
        usage = (
            usage_from_batch_output_row(output_row) if output_row is not None else None
        )
        extraction = (
            extraction_from_batch_output_row(output_row)
            if output_row is not None
            else None
        )

        joined.append(
            SkillMdBatchRecord(
                relative_path=relative_path,
                content=content,
                metadata=metadata,
                custom_id=custom_id,
                prompt=prompt,
                usage=usage,
                extraction=extraction,
            )
        )

    log.info(f"{len(joined)=}")
    return joined
