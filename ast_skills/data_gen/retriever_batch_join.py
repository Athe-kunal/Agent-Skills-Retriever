"""Join OpenAI batch inputs under ``done/`` with batch output JSONL under ``batch_results/``.

Rows are matched on ``custom_id``. Assistant completions are parsed into
``SkillMdExtraction``; SKILL.md markdown is sliced from the batch input user message.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

from loguru import logger as log

from ast_skills.data_gen.dataset import (
    extraction_from_batch_output_row,
    index_rows_by_custom_id,
    messages_from_batch_input_row,
    read_jsonl,
)
from ast_skills.data_gen.synthetic_data_gen import SkillMdExtraction

_SKILL_MD_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"----- SKILL\.md markdown -----\s*(.*?)\s*----- end -----",
    re.DOTALL,
)


class RedoBatchExportResult(NamedTuple):
    """Summary of ``export_redo_batch_inputs``."""

    written_paths: list[Path]
    missing_output_count: int
    invalid_extraction_count: int
    total_rows_written: int


@dataclass(frozen=True)
class RetrieverDataModel:
    """One retriever training row: SKILL.md source plus structured extraction."""

    custom_id: str
    markdown_content: str
    reasoning: str
    what: str
    why: str
    seed_questions: str
    metadata: dict[str, str]


def _iter_jsonl_files(directory: Path) -> list[Path]:
    """Return ``*.jsonl`` files directly under ``directory``, sorted by path."""
    paths = sorted(directory.glob("*.jsonl"))
    log.info(f"{directory=}, {len(paths)=}")
    return paths


def _iter_jsonl_files_recursive(directory: Path) -> list[Path]:
    """Return every ``*.jsonl`` under ``directory`` (recursive), sorted by path."""
    paths = sorted(directory.rglob("*.jsonl"))
    log.info(f"{directory=}, {len(paths)=}")
    return paths


def _read_jsonl_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _path_source_for_custom_id(paths: list[Path]) -> dict[str, str]:
    """Map each ``custom_id`` to the last JSONL path that contained it."""
    by_id: dict[str, str] = {}
    for path in paths:
        for row in read_jsonl(path):
            custom_id = row.get("custom_id")
            if isinstance(custom_id, str) and custom_id:
                by_id[custom_id] = str(path)
    return by_id


def extract_skill_markdown_from_done_row(row: dict[str, Any]) -> str:
    """Return the SKILL.md markdown body from a batch input JSONL row."""
    messages = messages_from_batch_input_row(row)
    user_text = ""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            user_text = content
            break
    if not user_text:
        log.warning("No user message with string content found")
        return ""
    match = _SKILL_MD_BLOCK_PATTERN.search(user_text)
    if match:
        return match.group(1).strip()
    log.warning("SKILL.md delimiters not found; returning full user message content")
    return user_text.strip()


def _metadata_for_row(
    *,
    custom_id: str,
    done_row: dict[str, Any],
    output_row: dict[str, Any],
    done_source_path: str,
    output_source_path: str,
    extraction: SkillMdExtraction,
) -> dict[str, str]:
    """Normalize join context to ``dict[str, str]`` for downstream consumers."""
    body = done_row.get("body")
    input_model = ""
    if isinstance(body, dict):
        model = body.get("model")
        if isinstance(model, str):
            input_model = model

    status_code = ""
    response_model = ""
    completion_id = ""
    batch_row_id = ""
    request_id = ""
    if isinstance(output_row.get("id"), str):
        batch_row_id = output_row["id"]
    response = output_row.get("response")
    if isinstance(response, dict):
        status = response.get("status_code")
        status_code = str(status) if status is not None else ""
        request_raw = response.get("request_id")
        if isinstance(request_raw, str):
            request_id = request_raw
        resp_body = response.get("body")
        if isinstance(resp_body, dict):
            rid = resp_body.get("id")
            if isinstance(rid, str):
                completion_id = rid
            m = resp_body.get("model")
            if isinstance(m, str):
                response_model = m

    err = output_row.get("error")
    error_summary = ""
    if err is not None:
        error_summary = json.dumps(err, ensure_ascii=False, sort_keys=True, default=str)

    return {
        "custom_id": custom_id,
        "done_jsonl": done_source_path,
        "batch_output_jsonl": output_source_path,
        "batch_row_id": batch_row_id,
        "request_id": request_id,
        "completion_id": completion_id,
        "input_model": input_model,
        "response_status_code": status_code,
        "response_model": response_model,
        "seed_question_count": str(len(extraction.seed_questions)),
        "batch_error": error_summary,
    }


def build_retriever_data_models(
    done_dir: Path,
    batch_results_dir: Path,
    *,
    require_batch_output: bool = True,
) -> list[RetrieverDataModel]:
    """
    Load all ``done/*.jsonl`` inputs and all ``batch_results/**/*.jsonl`` outputs,
    join on ``custom_id`` (last occurrence wins for duplicates), and return rows
    with parsed ``SkillMdExtraction`` fields.

    If ``require_batch_output`` is True, only ``custom_id`` values present in both
    sides with a parseable assistant JSON payload are returned.
    """
    done_paths = _iter_jsonl_files(done_dir)
    output_paths = _iter_jsonl_files_recursive(batch_results_dir)

    done_rows = _read_jsonl_rows(done_paths)
    output_rows = _read_jsonl_rows(output_paths)

    done_by_id = index_rows_by_custom_id(done_rows)
    output_by_id = index_rows_by_custom_id(output_rows)
    done_path_by_id = _path_source_for_custom_id(done_paths)
    output_path_by_id = _path_source_for_custom_id(output_paths)

    log.info(
        f"{len(done_rows)=}, {len(output_rows)=}, {len(done_by_id)=}, {len(output_by_id)=}"
    )

    models: list[RetrieverDataModel] = []
    missing_output = 0
    invalid_extraction = 0
    for custom_id in sorted(done_by_id.keys()):
        done_row = done_by_id[custom_id]
        output_row = output_by_id.get(custom_id)
        if require_batch_output and output_row is None:
            missing_output += 1
            continue

        markdown_content = extract_skill_markdown_from_done_row(done_row)
        extraction: SkillMdExtraction | None = None
        if output_row is not None:
            extraction = extraction_from_batch_output_row(output_row)
        if extraction is None:
            if require_batch_output or output_row is not None:
                invalid_extraction += 1
            continue

        done_source = done_path_by_id.get(custom_id, "")
        output_source = (
            output_path_by_id.get(custom_id, "") if output_row is not None else ""
        )

        metadata = _metadata_for_row(
            custom_id=custom_id,
            done_row=done_row,
            output_row=output_row if output_row is not None else {},
            done_source_path=done_source,
            output_source_path=output_source,
            extraction=extraction,
        )

        models.append(
            RetrieverDataModel(
                custom_id=custom_id,
                markdown_content=markdown_content,
                reasoning=extraction.reasoning,
                what=extraction.what,
                why=extraction.why,
                seed_questions=json.dumps(
                    extraction.seed_questions, ensure_ascii=False
                ),
                metadata=metadata,
            )
        )

    log.info(
        f"{len(models)=}, {missing_output=}, {invalid_extraction=}"
    )
    return models


def _redo_input_rows_for_done_and_outputs(
    done_by_id: dict[str, dict[str, Any]],
    output_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Batch input rows from ``done_by_id`` that need re-submission: no matching output,
    or output exists but does not parse as ``SkillMdExtraction``.

    Returns ``(rows, missing_count, invalid_count)`` in stable ``custom_id`` order.
    """
    redo_rows: list[dict[str, Any]] = []
    missing = 0
    invalid = 0
    for custom_id in sorted(done_by_id.keys()):
        done_row = done_by_id[custom_id]
        output_row = output_by_id.get(custom_id)
        if output_row is None:
            missing += 1
            redo_rows.append(done_row)
            continue
        if extraction_from_batch_output_row(output_row) is None:
            invalid += 1
            redo_rows.append(done_row)
    return redo_rows, missing, invalid


def export_redo_batch_inputs(
    done_dir: Path,
    batch_results_dir: Path,
    output_dir: Path,
    *,
    chunk_size: int = 500,
    start_file_index: int = 74,
    filename_template: str = "batch_inputs_{index}.jsonl",
) -> RedoBatchExportResult:
    """
    Write OpenAI batch **input** JSONL chunks for rows that are missing outputs or
    have outputs that fail ``SkillMdExtraction`` validation.

    Each line is the original request object from ``done_dir`` (same shape as the
    source JSONL). Files are named ``batch_inputs_74.jsonl``, ``batch_inputs_75.jsonl``,
    … when ``start_file_index`` is 74 and ``filename_template`` is the default.
    """
    done_paths = _iter_jsonl_files(done_dir)
    output_paths = _iter_jsonl_files_recursive(batch_results_dir)

    done_rows = _read_jsonl_rows(done_paths)
    output_rows = _read_jsonl_rows(output_paths)

    done_by_id = index_rows_by_custom_id(done_rows)
    output_by_id = index_rows_by_custom_id(output_rows)

    redo_rows, missing_count, invalid_count = _redo_input_rows_for_done_and_outputs(
        done_by_id, output_by_id
    )
    log.info(
        f"{len(redo_rows)=}, {missing_count=}, {invalid_count=}, "
        f"{chunk_size=}, {start_file_index=}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    file_index = start_file_index
    total_written = 0

    for offset in range(0, len(redo_rows), chunk_size):
        chunk = redo_rows[offset : offset + chunk_size]
        name = filename_template.format(index=file_index)
        out_path = output_dir / name
        with out_path.open("w", encoding="utf-8") as handle:
            for row in chunk:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_written += 1
        written.append(out_path)
        log.info(f"wrote {len(chunk)} rows to {out_path=}")
        file_index += 1

    return RedoBatchExportResult(
        written_paths=written,
        missing_output_count=missing_count,
        invalid_extraction_count=invalid_count,
        total_rows_written=total_written,
    )


def retriever_models_to_jsonl(models: list[RetrieverDataModel], path: Path) -> None:
    """Write ``RetrieverDataModel`` instances as JSONL (dataclass fields as JSON keys)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in models:
            payload = {
                "custom_id": item.custom_id,
                "markdown_content": item.markdown_content,
                "reasoning": item.reasoning,
                "what": item.what,
                "why": item.why,
                "seed_questions": item.seed_questions,
                "metadata": item.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    log.info(f"wrote {len(models)} rows to {path=}")
