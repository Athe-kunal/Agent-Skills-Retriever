"""Build JSONL prompt inputs for scenario and question generation workflows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import fire
from loguru import logger

from ast_skills.data_gen.openai_batch_chat_request import (
    DEFAULT_OPENAI_BATCH_MODEL,
    openai_batch_chat_completion_request,
)
from ast_skills.data_gen.synthetic_data_gen import (
    OPENAI_BATCH_MAX_FILE_TOKENS,
    _batch_chat_messages_token_count,
    _would_exceed_batch_file_tokens,
)
from ast_skills.data_gen.skills_data_collect import (
    SkillMdRecord,
    collect_english_skill_md_records,
)
from ast_skills.persona_data_gen.datamodels import OpenAIOutput, ScenarioQueryPromptRow
from ast_skills.persona_data_gen.scenario_prompts import (
    SCENARIO_GENERATION_SYSTEM_PROMPT,
    build_scenario_generation_prompt,
)

SCENARIO_GENERATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "scenario_generation_response",
        "schema": OpenAIOutput.model_json_schema(),
        "strict": True,
    },
}

SCENARIO_GENERATION_BATCH_STEM = "scenario_generation_batch"
DEFAULT_SCENARIO_BATCH_INPUT_DIR = Path("data/scenario_batch_inputs")
DEFAULT_SCENARIO_BATCH_RESULTS_DIR = Path("data/scenario_batch_results")
DEFAULT_SCENARIO_BATCH_DONE_DIR = Path("data/scenario_batch_done")


def _enumerated_batch_input_path(
    output_dir: Path, filename_stem: str, shard_index: int
) -> Path:
    """Return ``output_dir / {stem}_{shard_index+1}.jsonl`` (1-based shard filenames)."""
    number = shard_index + 1
    return output_dir / f"{filename_stem}_{number}.jsonl"


def scenario_generation_batch_input_base_path(output_dir: str) -> Path:
    """Return path to shard ``1``: ``output_dir/scenario_generation_batch_1.jsonl``."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return _enumerated_batch_input_path(root, SCENARIO_GENERATION_BATCH_STEM, 0)


def _coerce_jsonl_path_list(paths: str | Sequence[str]) -> list[str]:
    """Normalize CLI or API input into a non-empty list of JSONL paths."""
    if isinstance(paths, str):
        stripped = paths.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            loaded = json.loads(stripped)
            if not isinstance(loaded, list):
                raise ValueError(f"Expected JSON array of paths, got {type(loaded)}")
            return [str(p).strip() for p in loaded if str(p).strip()]
        if "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    return [str(p).strip() for p in paths if str(p).strip()]


def load_filtered_skill_md_records(skills_root: str) -> list[SkillMdRecord]:
    """Load skill records using the shared data_gen collector and token filtering."""
    records = collect_english_skill_md_records(skills_root)
    logger.info(f"{skills_root=}")
    logger.info(f"{len(records)=}")
    return records


def _skill_name(skill_record: SkillMdRecord) -> str:
    """Get a skill display name from metadata with a stable fallback."""
    return skill_record.metadata.get("name") or skill_record.relative_path


def build_scenario_generation_prompt_rows(
    skill_md_records: list[SkillMdRecord],
) -> list[ScenarioQueryPromptRow]:
    """Build one scenario-generation prompt row per skill markdown record."""
    rows: list[ScenarioQueryPromptRow] = []
    for index, skill_md_record in enumerate(skill_md_records):
        custom_id = f"scenario-{index}"
        row = ScenarioQueryPromptRow(
            custom_id=custom_id,
            relative_path=skill_md_record.relative_path,
            skill_name=_skill_name(skill_md_record),
            prompt=build_scenario_generation_prompt(skill_md_record),
        )
        rows.append(row)
    logger.info(f"{len(rows)=}")
    return rows


def write_scenario_generation_prompts_jsonl(
    rows: list[ScenarioQueryPromptRow],
    output_dir: str | Path,
    *,
    filename_stem: str = SCENARIO_GENERATION_BATCH_STEM,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_tokens: int = 4096,
    max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
) -> int:
    """Write OpenAI Batch input JSONL for scenario+question generation."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written_count = 0
    current_file_tokens = 0
    shard_index = 0
    shard_path = _enumerated_batch_input_path(out_dir, filename_stem, shard_index)
    output_file = shard_path.open("w", encoding="utf-8")

    try:
        for row in rows:
            messages = [
                {"role": "system", "content": SCENARIO_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": row.prompt},
            ]
            request_line = openai_batch_chat_completion_request(
                custom_id=row.custom_id,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=SCENARIO_GENERATION_RESPONSE_FORMAT,
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
                    shard_path = _enumerated_batch_input_path(
                        out_dir, filename_stem, shard_index
                    )
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

    logger.info(f"{out_dir=} {filename_stem=} {shard_index=}")
    logger.info(f"{written_count=}")
    logger.info(f"{current_file_tokens=} (last shard)")
    logger.info(f"{max_file_tokens=}")
    logger.info(f"{model=}")
    logger.info(f"{max_tokens=}")
    return written_count


class ScenarioPromptJobs:
    """CLI jobs for scenario+question prompt generation and batch submission."""

    def submit_batches(
        self,
        input_dir: str = str(DEFAULT_SCENARIO_BATCH_INPUT_DIR),
        results_dir: str = str(DEFAULT_SCENARIO_BATCH_RESULTS_DIR),
        done_dir: str = str(DEFAULT_SCENARIO_BATCH_DONE_DIR),
    ) -> None:
        """Submit every ``*.jsonl`` in ``input_dir`` through OpenAI Batch."""
        from ast_skills.data_gen.openai_batch_jobs import run_batch_mode

        in_path = Path(input_dir).expanduser()
        in_path.mkdir(parents=True, exist_ok=True)
        jsonl_files = sorted(in_path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {in_path}")

        logger.info(f"{in_path=} {len(jsonl_files)=} {results_dir=} {done_dir=}")
        run_batch_mode(
            jsonl_files,
            batch_output_dir=Path(results_dir).expanduser(),
            input_done_dir=Path(done_dir).expanduser(),
        )

    def submit_batch_files(
        self,
        jsonl_paths: str | Sequence[str],
        results_dir: str = str(DEFAULT_SCENARIO_BATCH_RESULTS_DIR),
        done_dir: str = str(DEFAULT_SCENARIO_BATCH_DONE_DIR),
    ) -> None:
        """Submit explicit batch input JSONL paths."""
        from ast_skills.data_gen.openai_batch_jobs import (
            build_client,
            process_one_jsonl_batch,
            resolve_openai_project,
        )

        path_strings = _coerce_jsonl_path_list(jsonl_paths)
        paths = [Path(p).expanduser().resolve() for p in path_strings]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Batch input not found: {path}")

        res_dir = Path(results_dir).expanduser()
        dn_dir = Path(done_dir).expanduser()
        res_dir.mkdir(parents=True, exist_ok=True)
        dn_dir.mkdir(parents=True, exist_ok=True)

        project = resolve_openai_project()
        logger.info(f"{project=} {paths=} {res_dir=} {dn_dir=}")
        client = build_client(project=project)
        for path in paths:
            try:
                process_one_jsonl_batch(
                    client=client,
                    path=path,
                    batch_output_dir=res_dir,
                    input_done_dir=dn_dir,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(f"{path=} {exc=}")

    def build_scenario_prompts(
        self,
        skills_root: str,
        output_dir: str,
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_tokens: int = 4096,
        max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
    ) -> None:
        """Generate OpenAI Batch input JSONL for scenario+question generation."""
        skill_md_records = load_filtered_skill_md_records(skills_root)
        rows = build_scenario_generation_prompt_rows(skill_md_records)
        first_path = scenario_generation_batch_input_base_path(output_dir)
        written = write_scenario_generation_prompts_jsonl(
            rows,
            output_dir,
            model=model,
            max_tokens=max_tokens,
            max_file_tokens=max_file_tokens,
        )
        logger.info(f"{written=} {output_dir=} {first_path=}")


if __name__ == "__main__":
    fire.Fire(ScenarioPromptJobs)
