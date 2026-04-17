"""Read/write helpers for ``ScenarioQueryPromptRowDataModel`` JSONL files."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ast_skills.persona_data_gen.datamodels import (
    ScenarioQueryPromptRowDataModel,
    ScenarioRelatedOutput,
)


def scenario_query_prompt_row_to_json_dict(
    row: ScenarioQueryPromptRowDataModel,
) -> dict[str, object]:
    """Serialize one row for JSONL (``scenario_output`` as plain dicts)."""
    payload: dict[str, object] = asdict(row)
    payload["scenario_output"] = [item.model_dump() for item in row.scenario_output]
    return payload


def scenario_query_prompt_row_from_json_dict(
    payload: dict[str, object],
) -> ScenarioQueryPromptRowDataModel:
    """Build a row from one JSON object; normalize ``scenario_output`` to Pydantic models."""
    raw_scenarios = payload["scenario_output"]
    if not isinstance(raw_scenarios, list):
        raise TypeError(f"scenario_output must be a list, got {type(raw_scenarios)}")
    scenario_output = [
        ScenarioRelatedOutput.model_validate(item) for item in raw_scenarios
    ]
    raw_seeds = payload["seed_questions"]
    if not isinstance(raw_seeds, list):
        raise TypeError(f"seed_questions must be a list, got {type(raw_seeds)}")
    seed_questions = [str(item) for item in raw_seeds]
    raw_meta = payload["metadata"]
    if not isinstance(raw_meta, dict):
        raise TypeError(f"metadata must be a dict, got {type(raw_meta)}")
    metadata = {str(key): str(value) for key, value in raw_meta.items()}
    return ScenarioQueryPromptRowDataModel(
        custom_id=str(payload["custom_id"]),
        markdown_content=str(payload["markdown_content"]),
        seed_questions=seed_questions,
        summary=str(payload["summary"]),
        name=str(payload["name"]),
        description=str(payload["description"]),
        metadata=metadata,
        scenario_output=scenario_output,
    )


def read_scenario_query_prompt_rows(path: str) -> list[ScenarioQueryPromptRowDataModel]:
    """Load JSONL rows and return them sorted by ``custom_id`` (stable order)."""
    rows: list[ScenarioQueryPromptRowDataModel] = []
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(scenario_query_prompt_row_from_json_dict(json.loads(line)))
    rows.sort(key=lambda row: row.custom_id)
    return rows


def read_annotated_rows_map(path: str) -> dict[str, ScenarioQueryPromptRowDataModel]:
    """Load annotation JSONL into ``custom_id`` -> row. Missing file returns empty dict."""
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    result: dict[str, ScenarioQueryPromptRowDataModel] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = scenario_query_prompt_row_from_json_dict(json.loads(line))
            result[row.custom_id] = row
    return result


def write_annotated_jsonl(
    path: str, rows_by_id: dict[str, ScenarioQueryPromptRowDataModel]
) -> None:
    """Write one JSON line per ``custom_id``, sorted by ``custom_id``."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_ids = sorted(rows_by_id)
    with file_path.open("w", encoding="utf-8") as handle:
        for custom_id in ordered_ids:
            row = rows_by_id[custom_id]
            payload = scenario_query_prompt_row_to_json_dict(row)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
