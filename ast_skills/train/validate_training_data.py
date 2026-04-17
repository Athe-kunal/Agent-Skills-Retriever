"""Async validation pipeline for ScenarioQueryPromptRowDataModel rows.

Steps per row:
  1. Ask an LLM (async OpenAI) to validate/complete the summary and pick 5
     relevant questions from the combined seed_questions + scenario questions.
  2. Attribute each filtered question to its source pool (seed vs scenario)
     via Levenshtein distance to the nearest original candidate.
  3. Write ValidatedTrainingData rows to a JSONL file.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import fire
import pydantic
from loguru import logger as log
from openai import AsyncOpenAI

from ast_skills.persona_data_gen.datamodels import (
    ScenarioQueryPromptRowDataModel,
    ScenarioRelatedOutput,
)
from ast_skills.train.datamodels import ModelOutput, ValidatedTrainingData
from ast_skills.train.scenario_query_row_io import (
    read_scenario_query_prompt_rows,
    scenario_query_prompt_row_to_json_dict,
)


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV_VAR = "OPENAI_BASE_URL"
OPENAI_MODEL_ENV_VAR = "OPENAI_MODEL"

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_VALIDATION_MODEL: str = os.environ.get(OPENAI_MODEL_ENV_VAR, "gpt-4o-mini")
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_INPUT_PATH = "artifacts/scenario_query_prompt_row_data_models.jsonl"
DEFAULT_OUTPUT_PATH = "artifacts/validated_training_data.jsonl"
DEFAULT_SMOKE_NUM_SAMPLES = 2
DEFAULT_SMOKE_INPUT_PATH = "artifacts/smoke_input.jsonl"
DEFAULT_SMOKE_OUTPUT_PATH = "artifacts/smoke_output.jsonl"

SUMMARY_APPROVED_SENTINEL = "YES"

VALIDATION_SYSTEM_PROMPT = """You are a quality reviewer for an AI skill knowledge base.

Given a SKILL.md markdown document and its current summary, you must:
1. Validate whether the summary is accurate and complete relative to the markdown.
   - If the summary is already accurate and complete, simply set filtered_summary
     to exactly "YES" — do NOT rewrite or repeat the summary back.
   - Only rewrite the summary when it is genuinely missing key information or
     contains inaccuracies. Ground the rewrite solely in the markdown content.
2. From the provided candidate questions, select exactly 5 that are MOST relevant
   to the skill. Preserve the exact wording — do NOT rephrase or invent questions.

Return a JSON object with:
  - reasoning: explain your summary validation decision and why you picked each question.
  - filtered_summary: exactly "YES" if the summary is already good, otherwise the corrected summary.
  - filtered_questions: exactly 5 questions copied verbatim from the candidates.
"""

VALIDATION_USER_PROMPT_TEMPLATE = """\
----- SKILL.md CONTENT -----
{markdown_content}
----- END CONTENT -----

Current summary (reply "YES" in filtered_summary if this is accurate and complete):
{summary}

Candidate questions:
{questions_formatted}

Select exactly 5 questions from the candidates above that best represent the
range of use-cases for this skill. Copy each question verbatim.
"""

VALIDATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "validation_response",
        "schema": ModelOutput.model_json_schema(),
        "strict": True,
    },
}


# ──────────────────────────────────────────────
# QUESTION SOURCE ATTRIBUTION
# ──────────────────────────────────────────────


class ClosestMatch(NamedTuple):
    """Nearest-candidate lookup result for a single filtered question."""

    candidate: str
    source: str  # "seed" or "scenario"
    distance: int


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _find_closest_match(
    question: str,
    seed_questions: list[str],
    scenario_questions: list[str],
) -> ClosestMatch:
    """Return the nearest original candidate and its source pool.

    Searches both pools using case-folded Levenshtein distance. Ties break
    in favor of ``seed`` since seed candidates are checked first.
    """
    best: ClosestMatch | None = None
    for candidate in seed_questions:
        dist = _levenshtein_distance(question.lower(), candidate.lower())
        if best is None or dist < best.distance:
            best = ClosestMatch(candidate=candidate, source="seed", distance=dist)
    for candidate in scenario_questions:
        dist = _levenshtein_distance(question.lower(), candidate.lower())
        if best is None or dist < best.distance:
            best = ClosestMatch(candidate=candidate, source="scenario", distance=dist)
    if best is None:
        return ClosestMatch(candidate=question, source="seed", distance=0)
    return best


def _count_questions_by_source(
    filtered_questions: list[str],
    seed_questions: list[str],
    scenario_questions: list[str],
) -> tuple[int, int]:
    """Return ``(num_from_seed, num_from_scenario)`` for the filtered question list."""
    num_seed = 0
    num_scenario = 0
    for question in filtered_questions:
        match = _find_closest_match(question, seed_questions, scenario_questions)
        log.info(
            f"{question=} matched {match.candidate=} "
            f"source={match.source} distance={match.distance}"
        )
        if match.source == "seed":
            num_seed += 1
        else:
            num_scenario += 1
    return num_seed, num_scenario


# ──────────────────────────────────────────────
# PROMPT BUILDING
# ──────────────────────────────────────────────


def _extract_scenario_questions(
    scenario_output: list[ScenarioRelatedOutput],
) -> list[str]:
    """Extract just the question string from each ScenarioRelatedOutput."""
    return [item.question for item in scenario_output]


def _format_numbered_list(items: list[str]) -> str:
    """Format a list of strings as a 1-indexed numbered list for a prompt."""
    return "\n".join(f"  {i + 1}. {item}" for i, item in enumerate(items))


def build_validation_user_prompt(row: ScenarioQueryPromptRowDataModel) -> str:
    """Render the user-turn prompt for one ScenarioQueryPromptRowDataModel row."""
    scenario_questions = _extract_scenario_questions(row.scenario_output)
    all_questions = row.seed_questions + scenario_questions
    return VALIDATION_USER_PROMPT_TEMPLATE.format(
        markdown_content=row.markdown_content,
        summary=row.summary,
        questions_formatted=_format_numbered_list(all_questions),
    )


def build_validation_messages(
    row: ScenarioQueryPromptRowDataModel,
) -> list[dict[str, str]]:
    """Return the full messages list for the OpenAI chat completion call."""
    return [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": build_validation_user_prompt(row)},
    ]


# ──────────────────────────────────────────────
# OPENAI ASYNC CALL
# ──────────────────────────────────────────────


async def call_openai_validation(
    client: AsyncOpenAI,
    row: ScenarioQueryPromptRowDataModel,
    model: str,
    max_tokens: int,
) -> ModelOutput | None:
    """Call OpenAI and parse the structured response as ModelOutput.

    Returns ``None`` on network errors or validation failures so the caller
    can skip or retry the row without crashing the pipeline.
    """
    messages = build_validation_messages(row)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            response_format=VALIDATION_RESPONSE_FORMAT,
        )
        raw_content = response.choices[0].message.content
        parsed = json.loads(raw_content)
        return ModelOutput.model_validate(parsed)
    except pydantic.ValidationError as exc:
        log.warning(f"ModelOutput validation failed: {row.custom_id=} {exc=}")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(f"OpenAI call failed: {row.custom_id=} {exc=}")
        return None


async def _validate_row_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    row: ScenarioQueryPromptRowDataModel,
    model: str,
    max_tokens: int,
) -> tuple[ScenarioQueryPromptRowDataModel, ModelOutput | None]:
    """Acquire semaphore, run LLM validation, and return the (row, output) pair."""
    async with semaphore:
        log.info(f"Validating: {row.custom_id=} {row.name=}")
        output = await call_openai_validation(client, row, model, max_tokens)
        return row, output


async def validate_all_rows(
    rows: list[ScenarioQueryPromptRowDataModel],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    max_concurrency: int,
) -> list[tuple[ScenarioQueryPromptRowDataModel, ModelOutput | None]]:
    """Run LLM validation for all rows concurrently, bounded by max_concurrency."""
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _validate_row_with_semaphore(semaphore, client, row, model, max_tokens)
        for row in rows
    ]
    return await asyncio.gather(*tasks)


# ──────────────────────────────────────────────
# BUILDING ValidatedTrainingData
# ──────────────────────────────────────────────


def _resolve_summary(model_filtered_summary: str, original_summary: str) -> str:
    """Return the original summary when the model approved it, otherwise the rewrite.

    The model signals approval by returning exactly ``SUMMARY_APPROVED_SENTINEL``
    (case-insensitive). Any other value is treated as a corrected summary.
    """
    if model_filtered_summary.strip().upper() == SUMMARY_APPROVED_SENTINEL:
        log.info("Summary approved by model; keeping original.")
        return original_summary
    log.info("Summary rewritten by model.")
    return model_filtered_summary


def build_validated_training_data(
    row: ScenarioQueryPromptRowDataModel,
    model_output: ModelOutput,
) -> ValidatedTrainingData:
    """Combine a source row and LLM output into a ValidatedTrainingData record."""
    scenario_questions = _extract_scenario_questions(row.scenario_output)
    num_seed, num_scenario = _count_questions_by_source(
        model_output.filtered_questions,
        row.seed_questions,
        scenario_questions,
    )
    resolved_summary = _resolve_summary(model_output.filtered_summary, row.summary)
    log.info(f"{row.custom_id=} {num_seed=} {num_scenario=}")
    return ValidatedTrainingData(
        custom_id=row.custom_id,
        name=row.name,
        markdown_content=row.markdown_content,
        filtered_summary=resolved_summary,
        description=row.description,
        filtered_questions=model_output.filtered_questions,
        num_from_seed_questions=str(num_seed),
        num_from_scenario_questions=str(num_scenario),
    )


# ──────────────────────────────────────────────
# JSONL I/O
# ──────────────────────────────────────────────


def write_validated_training_data_jsonl(
    rows: list[ValidatedTrainingData],
    path: str,
) -> None:
    """Write ValidatedTrainingData rows to a JSONL file, one JSON object per line."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    log.info(f"{path=} {len(rows)=}")


def sample_rows(
    rows: list[ScenarioQueryPromptRowDataModel],
    num_samples: int,
) -> list[ScenarioQueryPromptRowDataModel]:
    """Return a random sample of ``num_samples`` rows without replacement."""
    count = min(num_samples, len(rows))
    sampled = random.sample(rows, count)
    log.info(f"Sampled {count} of {len(rows)} rows: {[r.custom_id for r in sampled]}")
    return sampled


def write_sampled_rows_jsonl(
    rows: list[ScenarioQueryPromptRowDataModel],
    path: str,
) -> None:
    """Write sampled ScenarioQueryPromptRowDataModel rows to a JSONL file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    scenario_query_prompt_row_to_json_dict(row), ensure_ascii=False
                )
                + "\n"
            )
    log.info(f"{path=} {len(rows)=}")


# ──────────────────────────────────────────────
# CLIENT CONSTRUCTION
# ──────────────────────────────────────────────


def _build_async_openai_client() -> AsyncOpenAI:
    """Build AsyncOpenAI from environment variables.

    Reads ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` as plain strings.
    ``OPENAI_BASE_URL`` falls back to the standard OpenAI endpoint when unset.
    """
    api_key: str = os.environ.get(OPENAI_API_KEY_ENV_VAR, "")
    base_url: str = os.environ.get(OPENAI_BASE_URL_ENV_VAR, DEFAULT_OPENAI_BASE_URL)
    log.info(f"{base_url=}")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


# ──────────────────────────────────────────────
# PIPELINE ORCHESTRATION
# ──────────────────────────────────────────────


async def run_validation_pipeline(
    input_path: str,
    output_path: str,
    model: str,
    max_tokens: int,
    max_concurrency: int,
) -> None:
    """Load rows, validate with LLM, and write results to JSONL."""
    rows = read_scenario_query_prompt_rows(input_path)
    log.info(f"{input_path=} {len(rows)=}")

    client = _build_async_openai_client()
    results = await validate_all_rows(rows, client, model, max_tokens, max_concurrency)

    validated: list[ValidatedTrainingData] = []
    failed_ids: list[str] = []
    for row, model_output in results:
        if model_output is None:
            log.warning(f"Skipping failed row: {row.custom_id=}")
            failed_ids.append(row.custom_id)
            continue
        validated.append(build_validated_training_data(row, model_output))

    log.info(f"{len(validated)=} {len(failed_ids)=}")
    if failed_ids:
        log.warning(f"{failed_ids=}")

    write_validated_training_data_jsonl(validated, output_path)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


class ValidationPipelineCli:
    """CLI entry point for the data validation pipeline."""

    def run(
        self,
        input_path: str = DEFAULT_INPUT_PATH,
        output_path: str = DEFAULT_OUTPUT_PATH,
        model: str = DEFAULT_VALIDATION_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        """Validate ScenarioQueryPromptRowDataModel rows and write validated JSONL.

        Reads rows from ``input_path``, calls the LLM for each row (up to
        ``max_concurrency`` in-flight at once), and writes results to ``output_path``.
        Rows that fail LLM validation are skipped with a warning log.

        Example::

            uv run python ast_skills/train/validate_training_data.py run \\
                --input_path artifacts/scenario_query_prompt_row_data_models.jsonl \\
                --output_path artifacts/validated_training_data.jsonl \\
                --model gpt-4o \\
                --max_concurrency 20
        """
        log.info(
            f"{input_path=} {output_path=} {model=} {max_tokens=} {max_concurrency=}"
        )
        asyncio.run(
            run_validation_pipeline(
                input_path=input_path,
                output_path=output_path,
                model=model,
                max_tokens=max_tokens,
                max_concurrency=max_concurrency,
            )
        )

    def smoke_test(
        self,
        input_path: str = DEFAULT_INPUT_PATH,
        smoke_input_path: str = DEFAULT_SMOKE_INPUT_PATH,
        smoke_output_path: str = DEFAULT_SMOKE_OUTPUT_PATH,
        num_samples: int = DEFAULT_SMOKE_NUM_SAMPLES,
        model: str = DEFAULT_VALIDATION_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """Run the full pipeline on a random sample to verify end-to-end correctness.

        Reads all rows from ``input_path``, randomly picks ``num_samples`` of them,
        writes the sample to ``smoke_input_path``, runs the pipeline, and writes
        results to ``smoke_output_path``.

        Example::

            uv run python ast_skills/train/validate_training_data.py smoke_test \\
                --input_path artifacts/scenario_query_prompt_row_data_models.jsonl \\
                --num_samples 2
        """
        log.info(f"{input_path=} {num_samples=} {smoke_input_path=} {smoke_output_path=}")
        all_rows = read_scenario_query_prompt_rows(input_path)
        log.info(f"Total rows loaded: {len(all_rows)=}")

        sampled = sample_rows(all_rows, num_samples)
        write_sampled_rows_jsonl(sampled, smoke_input_path)

        asyncio.run(
            run_validation_pipeline(
                input_path=smoke_input_path,
                output_path=smoke_output_path,
                model=model,
                max_tokens=max_tokens,
                max_concurrency=num_samples,
            )
        )
        log.info(f"Smoke test complete. Results at {smoke_output_path=}")


if __name__ == "__main__":
    fire.Fire(ValidationPipelineCli)
