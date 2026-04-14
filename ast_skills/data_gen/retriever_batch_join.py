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
    skill_md_record_row_to_fields,
)
from ast_skills.data_gen.synthetic_data_gen import SkillMdExtraction

_SKILL_MD_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"----- SKILL\.md markdown -----\s*(.*?)\s*----- end -----",
    re.DOTALL,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SKILL_MD_RECORDS: Final[Path] = _REPO_ROOT / "skill_md_records.jsonl"


class RedoBatchExportResult(NamedTuple):
    """Summary of ``export_redo_batch_inputs``."""

    written_paths: list[Path]
    missing_output_count: int
    invalid_extraction_count: int
    total_rows_written: int


@dataclass(frozen=True)
class RetrieverDataModel:
    """One retriever training row: SKILL.md source plus structured extraction.

    ``metadata`` is only the coerced ``metadata`` object from the matching
    ``skill_md_records.jsonl`` row (empty dict when disabled or missing).
    """

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


def _skill_md_record_metadata_only(record_row: dict[str, Any] | None) -> dict[str, str]:
    """Coerced ``metadata`` from a ``skill_md_records`` row, or empty when missing."""
    if record_row is None:
        return {}
    _, _, skill_metadata = skill_md_record_row_to_fields(record_row)
    return dict(skill_metadata)


def _read_jsonl_dict_rows_lenient(path: Path) -> list[dict[str, Any]]:
    """Like ``read_jsonl`` but skips bad lines with a warning instead of failing."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning(f"{path=} {line_no=} {exc=}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                log.warning(f"{path=} {line_no=} expected dict, got {type(obj)=}")
    return rows


def _load_skill_md_records_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        log.warning(f"skill_md_records file missing or not a file: {path=}")
        return {}
    rows = _read_jsonl_dict_rows_lenient(path)
    by_id = index_rows_by_custom_id(rows)
    log.info(f"{path=}, {len(rows)=}, {len(by_id)=}")
    return by_id


def build_retriever_data_models(
    done_dir: Path,
    batch_results_dir: Path,
    *,
    require_batch_output: bool = True,
    skill_md_records_path: Path | None = None,
    load_skill_md_records: bool = True,
) -> list[RetrieverDataModel]:
    """
    Load all ``done/*.jsonl`` inputs and all ``batch_results/**/*.jsonl`` outputs,
    join on ``custom_id`` (last occurrence wins for duplicates), and return rows
    with parsed ``SkillMdExtraction`` fields.

    If ``require_batch_output`` is True, only ``custom_id`` values present in both
    sides with a parseable assistant JSON payload are returned.

    ``RetrieverDataModel.metadata`` is **only** the coerced ``metadata`` object from
    ``skill_md_records.jsonl`` (matched on ``custom_id``). When ``load_skill_md_records`` is
    False, or no record exists for an id, ``metadata`` is ``{}``.

    If ``skill_md_records_path`` is ``None``, the default is the repository root file
    ``skill_md_records.jsonl``.
    """
    done_paths = _iter_jsonl_files(done_dir)
    output_paths = _iter_jsonl_files_recursive(batch_results_dir)

    done_rows = _read_jsonl_rows(done_paths)
    output_rows = _read_jsonl_rows(output_paths)

    done_by_id = index_rows_by_custom_id(done_rows)
    output_by_id = index_rows_by_custom_id(output_rows)

    records_path = (
        skill_md_records_path
        if skill_md_records_path is not None
        else _DEFAULT_SKILL_MD_RECORDS
    )
    records_by_id: dict[str, dict[str, Any]] = {}
    if load_skill_md_records:
        records_by_id = _load_skill_md_records_by_custom_id(records_path)
    else:
        log.info("skill_md_records merge disabled (load_skill_md_records=False)")

    log.info(
        f"{len(done_rows)=}, {len(output_rows)=}, {len(done_by_id)=}, {len(output_by_id)=}"
    )

    models: list[RetrieverDataModel] = []
    missing_output = 0
    invalid_extraction = 0
    missing_record = 0
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

        record_row = records_by_id.get(custom_id)
        if load_skill_md_records and record_row is None:
            missing_record += 1
        metadata = (
            _skill_md_record_metadata_only(record_row)
            if load_skill_md_records
            else {}
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
        f"{len(models)=}, {missing_output=}, {invalid_extraction=}, {missing_record=}"
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
