from __future__ import annotations
import fire

import json
from collections import defaultdict
from pathlib import Path

from loguru import logger as log
from pydantic import ValidationError

from ast_skills.data_gen.datamodels import SummaryRetrieverDataModel
from ast_skills.data_gen.dataset import parsed_batch_output_content
from ast_skills.persona_data_gen.datamodels import (
    OpenAIOutput,
    ScenarioQueryPromptRowDataModel,
    ScenarioRelatedOutput,
)
from ast_skills.train.scenario_query_row_io import scenario_query_prompt_row_to_json_dict


def _build_summary_retriever_data_models(path: str) -> dict[str,list[SummaryRetrieverDataModel]]:
    """Build summary retriever data models from a JSONL file."""
    data_models: dict[str,list[SummaryRetrieverDataModel]] = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            data_models[obj["custom_id"].replace("sm-","")].append(SummaryRetrieverDataModel(**obj))
    return data_models


def _build_scenario_query_prompt_row_data_models(
    directory: str,
) -> dict[str, list[ScenarioRelatedOutput]]:
    """Load OpenAI batch output JSONL under ``directory`` (recursive) into custom_id → outputs.

    Each JSONL row is parsed like ``dataset._parsed_batch_output_content``; the assistant JSON
    is validated as ``OpenAIOutput`` and the ``scenario_output`` list is stored per ``custom_id``.
    When the same ``custom_id`` appears in multiple rows, the last row with a valid parse wins.
    """
    result: dict[str, list[ScenarioRelatedOutput]] = {}
    root = Path(directory)
    if not root.is_dir():
        log.warning(f"{directory=} is not a directory or does not exist")
        return result

    for path in sorted(root.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                custom_id = row.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id:
                    log.warning(f"skipping row with missing custom_id: {path=} {line_number=}")
                    continue
                parsed = parsed_batch_output_content(row)
                if parsed is None:
                    log.warning(
                        f"could not extract batch assistant JSON: {path=} {line_number=} {custom_id=}"
                    )
                    continue
                try:
                    model_output = OpenAIOutput.model_validate(parsed)
                except ValidationError as exc:
                    log.warning(
                        f"OpenAIOutput validation failed: {path=} {line_number=} {custom_id=} {exc=}"
                    )
                    continue
                result[custom_id.replace("scenario-","")] = list(model_output.scenario_output)
    return result


def build_scenario_query_prompt_row_data_models(
    summary_retriever_jsonl_path: str = "artifacts/summary_retriever_models.jsonl",
    scenario_batch_output_directory: str = "data/scenario_batch_results",
    output_jsonl_path: str = "artifacts/scenario_query_prompt_row_data_models.jsonl",
) -> None:
    """Join summary retriever JSONL rows with OpenAI scenario batch outputs by ``custom_id``.

    ``summary_retriever_jsonl_path`` is read via ``_build_summary_retriever_data_models`` (one
    file). ``scenario_batch_output_directory`` is scanned recursively for batch output JSONL via
    ``_build_scenario_query_prompt_row_data_models``. Only ``custom_id`` values present in both
    sources become ``ScenarioQueryPromptRowDataModel`` rows. If multiple summary rows share a
    ``custom_id``, the first row is used and a warning is logged. Rows are written to
    ``output_jsonl_path`` as JSONL (UTF-8).
    """
    summary_by_custom_id = _build_summary_retriever_data_models(summary_retriever_jsonl_path)
    scenario_by_custom_id = _build_scenario_query_prompt_row_data_models(
        scenario_batch_output_directory
    )

    summary_ids = set(summary_by_custom_id)
    scenario_ids = set(scenario_by_custom_id)
    missing_scenario = summary_ids - scenario_ids
    missing_summary = scenario_ids - summary_ids
    if missing_scenario:
        log.warning(f"{missing_scenario=} have no scenario batch output")
    if missing_summary:
        log.warning(f"{missing_summary=} have no summary retriever row")

    combined: list[ScenarioQueryPromptRowDataModel] = []
    for custom_id in sorted(summary_ids & scenario_ids):
        summary_rows = summary_by_custom_id[custom_id]
        if len(summary_rows) > 1:
            log.warning(f"{custom_id=} has {len(summary_rows)=} summary rows; using first")
        summary_row = summary_rows[0]
        scenario_output = scenario_by_custom_id[custom_id]
        combined.append(
            ScenarioQueryPromptRowDataModel(
                custom_id=custom_id,
                markdown_content=summary_row.markdown_content,
                seed_questions=list(summary_row.seed_questions),
                summary=summary_row.summary,
                name=summary_row.name,
                description=summary_row.description,
                metadata=dict(summary_row.metadata),
                scenario_output=list(scenario_output),
            )
        )
    log.info(f"Built {len(combined)} ScenarioQueryPromptRowDataModel rows")
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(scenario_query_prompt_row_to_json_dict(row), ensure_ascii=False) + "\n")
    # return combined


if __name__ == "__main__":
    fire.Fire({"build": build_scenario_query_prompt_row_data_models})