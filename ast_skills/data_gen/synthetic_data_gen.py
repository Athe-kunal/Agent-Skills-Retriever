"""OpenAI Batch helpers for SKILL.md extraction and summary generation."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import fire
import tiktoken
from loguru import logger
from openai import OpenAI

from ast_skills.data_gen.datamodels import SkillMdExtraction, SkillMdSummaryExtraction
from ast_skills.data_gen.skills_data_collect import (
    SkillMdRecord,
    collect_english_skill_md_records,
    encode_skill_md_record_batch_custom_id,
    read_skill_md_records_jsonl,
    write_skill_md_records_jsonl,
)

# GIT_ID = 7a7d8fe3f2eab4ab59c53f0f484e68b6dedb2de4
DEFAULT_OPENAI_BATCH_MODEL = "gpt-4o-mini"


SKILL_MD_EXTRACTION_SYSTEM_MESSAGE = """You read Agent Skill files (SKILL.md). Return exactly one JSON object with keys in this order: reasoning, what, why, seed_questions. No markdown fences, no extra keys, and no commentary outside JSON.

Ground every statement in the provided SKILL.md. Use all relevant evidence from both narrative sections and implementation sections (for example: commands, scripts, API/tool usage, configuration, inputs/outputs, constraints, caveats, workflow steps, examples, and failure handling).

Writing requirements:
- `reasoning`: concise justification that maps the extracted `what` and `why` to explicit evidence in the SKILL.md and explains why each seed question is an appropriate retrieval target.
- `what`: comprehensive capability description. Explain the full scope of what the skill enables, key operations, required inputs, expected outputs, important options, and notable constraints.
- `why`: comprehensive situational trigger. Explain when to use the skill, the user problems it solves, operational context, trade-offs, and why this workflow is preferred over generic alternatives.
- `seed_questions`: exactly 5 realistic user tasks that are materially different in intent and wording and together cover the main use-cases implied by the SKILL.md.

Do not invent tools or behaviors that are not present in the SKILL.md. If something is missing, explicitly call out the gap instead of guessing. Before returning, validate that the output is valid JSON and that every required field is present.
"""


SKILL_MD_EXTRACTION_PROMPT = """You are given the complete markdown source of one SKILL.md file.

----- SKILL.md markdown -----
{content}
----- end -----
"""


SKILL_MD_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "skill_md_extraction_response",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                },
                "what": {
                    "type": "string",
                },
                "why": {
                    "type": "string",
                },
                "seed_questions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "minItems": 5,
                    "maxItems": 5,
                },
            },
            "required": ["reasoning", "what", "why", "seed_questions"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


SKILL_MD_SUMMARY_SYSTEM_MESSAGE = """You read Agent Skill files (SKILL.md). Return exactly one JSON object with keys: summary, seed_questions. No markdown fences, no extra keys, and no commentary outside JSON.

Write a highly descriptive, deeply detailed summary grounded only in the provided SKILL.md. Cover the entire markdown, including:
- purpose and scope of the skill;
- all workflow steps and control flow;
- required tools, dependencies, and environment assumptions;
- command-line usage and argument behavior;
- code snippets and what each snippet does step by step;
- debugging guidance, troubleshooting instructions, and failure handling;
- input/output contracts, file paths, data formats, and constraints;
- operational caveats, limitations, and guardrails.

Also generate `seed_questions`: exactly 5 realistic user tasks that are materially different in intent and wording and together cover the main use-cases implied by the SKILL.md.

Do not invent behavior not present in the markdown. If important information is missing, explicitly state that gap.
"""


SKILL_MD_SUMMARY_PROMPT = """You are given the complete markdown source of one SKILL.md file.

Create a very descriptive summary that explains everything in the markdown in practical engineering terms. Also generate exactly 5 seed questions for the skill.

----- SKILL.md markdown -----
{content}
----- end -----
"""


SKILL_MD_SUMMARY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "skill_md_summary_response",
        "schema": SkillMdSummaryExtraction.model_json_schema(),
        "strict": True,
    },
}


# ──────────────────────────────────────────────
# OPENAI BATCH INPUT
# ──────────────────────────────────────────────


def openai_batch_chat_completion_request(
    custom_id: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    response_format: dict | None = None,
) -> dict:
    """
    One OpenAI Batch API input line for POST /v1/chat/completions.

    See https://developers.openai.com/api/docs/guides/batch
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


DEFAULT_BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_BATCH_COMPLETION_WINDOW = "24h"
# Shard batch input JSONL so each file stays under this token budget (tiktoken on ``body.messages``).
OPENAI_BATCH_MAX_FILE_TOKENS = 1_500_000
# Token counts for sharding always use this model's tokenizer, regardless of batch ``body.model``.
BATCH_SHARD_TIKTOKEN_MODEL = "gpt-4o-mini"


@lru_cache(maxsize=1)
def _tiktoken_encoding_for_batch_sharding() -> tiktoken.Encoding:
    """Tiktoken encoding for shard budgets: fixed ``gpt-4o-mini`` (not the per-request model)."""
    return tiktoken.encoding_for_model(BATCH_SHARD_TIKTOKEN_MODEL)


def submit_openai_batch_job(
    batch_input_jsonl_path: str,
    *,
    endpoint: str = DEFAULT_BATCH_ENDPOINT,
    completion_window: str = DEFAULT_BATCH_COMPLETION_WINDOW,
    metadata_description: str | None = None,
) -> str:
    """
    Upload a Batch input .jsonl (purpose=batch) and create a batch job.

    Requires OPENAI_API_KEY. See https://developers.openai.com/api/docs/guides/batch

    ``metadata_description`` sets batch-level metadata on the Batch object (visible
    when you retrieve the job), not per-request fields in completion output lines.

    Returns:
      The batch id (e.g. batch_...) for status checks and result download.
    """
    path = Path(batch_input_jsonl_path).expanduser().resolve()
    logger.info(f"{path=}")
    if not path.is_file():
        raise FileNotFoundError(f"Batch input not found: {path}")

    client = OpenAI()

    with path.open("rb") as batch_file:
        uploaded = client.files.create(file=batch_file, purpose="batch")
    logger.info(f"{uploaded.id=}")

    create_kwargs: dict = {
        "input_file_id": uploaded.id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata_description:
        create_kwargs["metadata"] = {"description": metadata_description}

    batch = client.batches.create(**create_kwargs)
    logger.info(f"{batch.id=}")
    logger.info(f"{batch.status=}")
    logger.info(f"{batch.endpoint=}")
    logger.info(f"{batch.input_file_id=}")
    return batch.id


def build_skill_md_extraction_user_content(skill_md_record: SkillMdRecord) -> str:
    """Build user prompt for extracting what/why/seed questions from SKILL.md."""
    return SKILL_MD_EXTRACTION_PROMPT.format(
        content=skill_md_record.content,
    )


def build_skill_md_summary_user_content(skill_md_record: SkillMdRecord) -> str:
    """Build user prompt for generating a detailed SKILL.md summary."""
    return SKILL_MD_SUMMARY_PROMPT.format(content=skill_md_record.content)


def _batch_chat_messages_token_count(request_row: dict) -> int:
    """
    Return tiktoken length for the chat prompt only: ``body.messages`` from the batch row.

    Uses ``BATCH_SHARD_TIKTOKEN_MODEL`` (``gpt-4o-mini``) for counting, independent of
    ``body.model`` on the request.

    Excludes ``custom_id``, ``method``, ``url``, and other ``body`` keys (``model``,
    ``max_tokens``, ``response_format``, etc.) so sharding tracks prompt size, not the
    full JSONL line.
    """
    body = request_row.get("body")
    if not isinstance(body, dict):
        return 0
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return 0
    prompt_payload = json.dumps(messages, ensure_ascii=False)
    encoding = _tiktoken_encoding_for_batch_sharding()
    return len(encoding.encode(prompt_payload, disallowed_special=set()))


def _would_exceed_batch_file_tokens(
    current_file_tokens: int,
    next_line_tokens: int,
    max_file_tokens: int,
) -> bool:
    """Check whether appending a row would exceed the per-file token budget."""
    return current_file_tokens + next_line_tokens > max_file_tokens


def _batch_jsonl_shard_path(base_path: Path, shard_index: int) -> Path:
    """
    Path for the ``shard_index``-th batch JSONL shard.

    ``shard_index`` 0 is ``base_path``; further shards use ``{stem}_{n}{suffix}``.
    """
    if shard_index == 0:
        return base_path
    return base_path.parent / f"{base_path.stem}_{shard_index}{base_path.suffix}"


def write_openai_batch_skill_md_extraction_jsonl_for_records(
    skill_md_records: list[SkillMdRecord],
    output_path: str,
    *,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_tokens: int = 1024,
    max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
) -> int:
    """
    Emit OpenAI Batch JSONL for SKILL.md extraction, from in-memory records.

    Each request ``custom_id`` is compact (``sm-<record_index>``) and aligns to the
    corresponding record row in ``skill_md_records.jsonl``.

    ``batches.create(..., metadata={...})`` is separate: job-level tags on the batch
    object, not echoed per completion line.

    If the running token total would exceed ``max_file_tokens`` (tiktoken count on each
    request's ``body.messages`` only), additional shards are written as ``{stem}_1{suffix}``,
    ``{stem}_2{suffix}``, ... beside the primary ``output_path``.

    OpenAI still enforces a per-file upload size limit (see Batch API docs); very dense
    lines could hit that limit before the token cap.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written_count = 0
    current_file_tokens = 0
    shard_index = 0
    shard_path = _batch_jsonl_shard_path(out_path, shard_index)
    output_file = shard_path.open("w", encoding="utf-8")

    try:
        for index, skill_md_record in enumerate(skill_md_records):
            custom_id = encode_skill_md_record_batch_custom_id(
                skill_md_record,
                index,
            )
            messages = [
                {"role": "system", "content": SKILL_MD_EXTRACTION_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": build_skill_md_extraction_user_content(skill_md_record),
                },
            ]
            request_line = openai_batch_chat_completion_request(
                custom_id=custom_id,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=SKILL_MD_RESPONSE_FORMAT,
            )
            line_tokens = _batch_chat_messages_token_count(request_line)
            would_exceed = _would_exceed_batch_file_tokens(
                current_file_tokens=current_file_tokens,
                next_line_tokens=line_tokens,
                max_file_tokens=max_file_tokens,
            )
            if would_exceed:
                if current_file_tokens == 0:
                    logger.warning(
                        "Single batch row exceeds max file token budget; writing it anyway: "
                        f"{line_tokens=} {max_file_tokens=} {shard_path=}"
                    )
                else:
                    output_file.close()
                    shard_index += 1
                    shard_path = _batch_jsonl_shard_path(out_path, shard_index)
                    logger.info(
                        "Continuing batch row writes in next shard: "
                        f"{shard_path=} {current_file_tokens=} {line_tokens=} "
                        f"{max_file_tokens=}"
                    )
                    output_file = shard_path.open("w", encoding="utf-8")
                    current_file_tokens = 0

            output_file.write(json.dumps(request_line, ensure_ascii=False) + "\n")
            current_file_tokens += line_tokens
            written_count += 1
    finally:
        output_file.close()

    logger.info(f"{output_path=}")
    logger.info(f"{shard_index=}")
    logger.info(f"{len(skill_md_records)=}")
    logger.info(f"{written_count=}")
    logger.info(f"{current_file_tokens=} (last shard)")
    logger.info(f"{max_file_tokens=}")
    return written_count


def write_openai_batch_skill_md_summary_jsonl_for_records(
    skill_md_records: list[SkillMdRecord],
    output_path: str,
    *,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_tokens: int = 2048,
    max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
) -> int:
    """Emit OpenAI Batch JSONL for detailed SKILL.md summary generation."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written_count = 0
    current_file_tokens = 0
    shard_index = 0
    shard_path = _batch_jsonl_shard_path(out_path, shard_index)
    output_file = shard_path.open("w", encoding="utf-8")

    try:
        for index, skill_md_record in enumerate(skill_md_records):
            custom_id = encode_skill_md_record_batch_custom_id(skill_md_record, index)
            messages = [
                {"role": "system", "content": SKILL_MD_SUMMARY_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": build_skill_md_summary_user_content(skill_md_record),
                },
            ]
            request_line = openai_batch_chat_completion_request(
                custom_id=custom_id,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=SKILL_MD_SUMMARY_RESPONSE_FORMAT,
            )
            line_tokens = _batch_chat_messages_token_count(request_line)
            would_exceed = _would_exceed_batch_file_tokens(
                current_file_tokens=current_file_tokens,
                next_line_tokens=line_tokens,
                max_file_tokens=max_file_tokens,
            )
            if would_exceed:
                if current_file_tokens == 0:
                    logger.warning(
                        "Single batch row exceeds max file token budget; writing it anyway: "
                        f"{line_tokens=} {max_file_tokens=} {shard_path=}"
                    )
                else:
                    output_file.close()
                    shard_index += 1
                    shard_path = _batch_jsonl_shard_path(out_path, shard_index)
                    logger.info(
                        "Continuing batch row writes in next shard: "
                        f"{shard_path=} {current_file_tokens=} {line_tokens=} "
                        f"{max_file_tokens=}"
                    )
                    output_file = shard_path.open("w", encoding="utf-8")
                    current_file_tokens = 0

            output_file.write(json.dumps(request_line, ensure_ascii=False) + "\n")
            current_file_tokens += line_tokens
            written_count += 1
    finally:
        output_file.close()

    logger.info(f"{output_path=}")
    logger.info(f"{shard_index=}")
    logger.info(f"{len(skill_md_records)=}")
    logger.info(f"{written_count=}")
    logger.info(f"{current_file_tokens=} (last shard)")
    logger.info(f"{max_file_tokens=}")
    return written_count


def write_openai_batch_skill_md_extraction_jsonl(
    records_path: str,
    output_path: str,
    *,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_records: int | None = None,
    max_tokens: int = 1024,
    max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
) -> int:
    """Emit OpenAI Batch JSONL for SkillMdRecord extraction (read records from JSONL)."""
    skill_md_records = read_skill_md_records_jsonl(records_path)
    logger.info(f"{records_path=}")
    logger.info(f"{len(skill_md_records)=}")
    if max_records is not None:
        skill_md_records = skill_md_records[:max_records]
    return write_openai_batch_skill_md_extraction_jsonl_for_records(
        skill_md_records,
        output_path,
        model=model,
        max_tokens=max_tokens,
        max_file_tokens=max_file_tokens,
    )


def write_openai_batch_skill_md_summary_jsonl(
    records_path: str,
    output_path: str,
    *,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_records: int | None = None,
    max_tokens: int = 2048,
    max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
) -> int:
    """Emit OpenAI Batch JSONL for detailed SKILL.md summaries from records JSONL."""
    skill_md_records = read_skill_md_records_jsonl(records_path)
    logger.info(f"{records_path=}")
    logger.info(f"{len(skill_md_records)=}")
    if max_records is not None:
        skill_md_records = skill_md_records[:max_records]
    return write_openai_batch_skill_md_summary_jsonl_for_records(
        skill_md_records=skill_md_records,
        output_path=output_path,
        model=model,
        max_tokens=max_tokens,
        max_file_tokens=max_file_tokens,
    )


def has_required_metadata(record: SkillMdRecord) -> bool:
    """Return True only when both name and description are non-empty strings."""
    return bool(record.metadata.get("name")) and bool(
        record.metadata.get("description")
    )


def filter_records_with_metadata(records: list[SkillMdRecord]) -> list[SkillMdRecord]:
    """
    Drop records whose name or description metadata is missing or empty.

    Logs each skipped record so the caller can audit what was removed.
    """
    kept: list[SkillMdRecord] = []
    for record in records:
        if has_required_metadata(record):
            kept.append(record)
        else:
            name = record.metadata.get("name", "")
            description = record.metadata.get("description", "")
            logger.warning(
                f"Skipping record with missing metadata: {record.relative_path=} {name=} {description=}"
            )

    logger.info(f"Records kept: {len(kept)} / {len(records)} total")
    return kept


class SyntheticDataGenCli:
    """Entry point for `python synthetic_data_gen.py <command> ...` via python-fire."""

    def extract_skill_md_batch(
        self,
        skills_root: str,
        batch_output_path: str = "data/openai_skill_md_batch_input.jsonl",
        records_jsonl_path: str = "data/skill_md_records.jsonl",
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_records: int | None = None,
        max_tokens: int = 1024,
        max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
    ) -> None:
        """
        Scan SKILL.md under skills_root, save records JSONL, then write OpenAI Batch input JSONL.

        By default both outputs go under ``data/``. Set ``records_jsonl_path`` to a falsey
        string (e.g. empty ``""``) to skip writing the records file.

        Each records line includes ``record_index`` and ``custom_id`` matching the batch job.
        Each ``custom_id`` follows ``sm-<record_index>`` and is capped at 500 characters.
        Decode with ``decode_skill_md_batch_custom_id`` in ``skills_data_collect``, or join
        batch lines to this file on exact ``custom_id``.
        """
        logger.info(f"{skills_root=}")
        records = collect_english_skill_md_records(skills_root)
        logger.info(f"collected {len(records)} SKILL.md records")

        if max_records is not None:
            records = records[:max_records]

        if records_jsonl_path:
            write_skill_md_records_jsonl(
                records,
                records_jsonl_path,
                include_openai_batch_custom_id=True,
            )
            logger.info(f"{records_jsonl_path=}")
            logger.info(f"{len(records)=}")

        record_count = write_openai_batch_skill_md_extraction_jsonl_for_records(
            records,
            batch_output_path,
            model=model,
            max_tokens=max_tokens,
            max_file_tokens=max_file_tokens,
        )
        logger.info(
            f"Wrote {record_count} batch lines to {batch_output_path} "
            f"(model={model})"
        )

    def extract_skill_md_summary_batch(
        self,
        skills_root: str = "skills/skills",
        batch_output_path: str = "batch_summary_inputs/openai_skill_md_summary_batch_input.jsonl",
        records_jsonl_path: str = "batch_summary_inputs/skill_md_records.jsonl",
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_records: int | None = None,
        max_tokens: int = 2048,
        max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
    ) -> None:
        """Build summary batch JSONL files from scanned SKILL.md files."""
        logger.info(f"{skills_root=}")
        records = collect_english_skill_md_records(skills_root)
        logger.info(f"{len(records)=}")

        records = filter_records_with_metadata(records)

        if max_records is not None:
            records = records[:max_records]
            logger.info(f"{max_records=}")
            logger.info(f"{len(records)=}")

        if records_jsonl_path:
            write_skill_md_records_jsonl(
                records,
                records_jsonl_path,
                include_openai_batch_custom_id=True,
            )
            logger.info(f"{records_jsonl_path=}")

        record_count = write_openai_batch_skill_md_summary_jsonl_for_records(
            skill_md_records=records,
            output_path=batch_output_path,
            model=model,
            max_tokens=max_tokens,
            max_file_tokens=max_file_tokens,
        )
        logger.info(f"{record_count=}")
        logger.info(f"{batch_output_path=}")

    def extract_skill_md_batch_from_jsonl(
        self,
        records_path: str,
        batch_output_path: str = "data/openai_skill_md_batch_input.jsonl",
        records_jsonl_path: str = "data/skill_md_records.jsonl",
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_records: int | None = None,
        max_tokens: int = 4096,
        max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
    ) -> None:
        """
        Same as extract_skill_md_batch but read pre-built records JSONL instead of scanning.

        Input JSONL format per line:
          {"relative_path":"...", "content":"...", "metadata": {...}}
          (metadata optional; used for storage only, not sent in the extraction prompt.)

        After optional ``max_records`` slicing, writes ``records_jsonl_path`` (default under
        ``data/``) with ``record_index`` and ``custom_id`` aligned to the batch file. Pass
        ``records_jsonl_path=""`` to skip that write.
        """
        skill_md_records = read_skill_md_records_jsonl(records_path)
        logger.info(f"{records_path=}")
        logger.info(f"{len(skill_md_records)=}")
        if max_records is not None:
            skill_md_records = skill_md_records[:max_records]

        if records_jsonl_path:
            write_skill_md_records_jsonl(
                skill_md_records,
                records_jsonl_path,
                include_openai_batch_custom_id=True,
            )
            logger.info(f"{records_jsonl_path=}")

        record_count = write_openai_batch_skill_md_extraction_jsonl_for_records(
            skill_md_records,
            batch_output_path,
            model=model,
            max_tokens=max_tokens,
            max_file_tokens=max_file_tokens,
        )
        logger.info(
            f"Wrote {record_count} batch lines to {batch_output_path} "
            f"(model={model})"
        )

    def submit_batch(
        self,
        batch_input_jsonl_path: str,
        endpoint: str = DEFAULT_BATCH_ENDPOINT,
        completion_window: str = DEFAULT_BATCH_COMPLETION_WINDOW,
        metadata_description: str | None = None,
    ) -> None:
        """
        Upload batch_input_jsonl_path to OpenAI (purpose=batch) and enqueue a batch job.

        Set OPENAI_API_KEY in the environment (e.g. from .env in your shell).

        Example:
          uv run python ast_skills/data_gen/synthetic_data_gen.py submit_batch batch_input.jsonl
          uv run python ... submit_batch batch_input.jsonl \\
            --endpoint=/v1/embeddings --metadata-description="skill-md extraction"
        """
        batch_id = submit_openai_batch_job(
            batch_input_jsonl_path,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata_description=metadata_description,
        )
        logger.info(
            "Poll with the API or dashboard; retrieve output when status is completed. "
            f"{batch_id=}"
        )

    def extract_skill_md_summary_batch_from_jsonl(
        self,
        records_path: str,
        batch_output_path: str = "data/batch_summary_inputs/openai_skill_md_summary_batch_input.jsonl",
        records_jsonl_path: str = "data/batch_summary_inputs/skill_md_records.jsonl",
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_records: int | None = None,
        max_tokens: int = 2048,
        max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
    ) -> None:
        """Build summary batch JSONL files from an existing records JSONL file."""
        skill_md_records = read_skill_md_records_jsonl(records_path)
        logger.info(f"{records_path=}")
        logger.info(f"{len(skill_md_records)=}")
        if max_records is not None:
            skill_md_records = skill_md_records[:max_records]
            logger.info(f"{max_records=}")
            logger.info(f"{len(skill_md_records)=}")

        if records_jsonl_path:
            write_skill_md_records_jsonl(
                skill_md_records,
                records_jsonl_path,
                include_openai_batch_custom_id=True,
            )
            logger.info(f"{records_jsonl_path=}")

        record_count = write_openai_batch_skill_md_summary_jsonl_for_records(
            skill_md_records=skill_md_records,
            output_path=batch_output_path,
            model=model,
            max_tokens=max_tokens,
            max_file_tokens=max_file_tokens,
        )
        logger.info(f"{record_count=}")
        logger.info(f"{batch_output_path=}")


if __name__ == "__main__":
    fire.Fire(SyntheticDataGenCli)
