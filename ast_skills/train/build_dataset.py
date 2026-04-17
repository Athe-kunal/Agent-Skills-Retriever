from __future__ import annotations

from ast_skills.persona_data_gen.datamodels import ScenarioQueryPromptRowDataModel
from ast_skills.train.scenario_query_row_io import read_scenario_query_prompt_rows


def _read_scenario_query_prompt_row_data_models(path: str = "artifacts/scenario_query_prompt_row_data_models.jsonl") -> list[ScenarioQueryPromptRowDataModel]:
    return read_scenario_query_prompt_rows(path)


def _filter_seed_questions(seed_questions: list[str]) -> list[str]:
    return list(seed_questions)
