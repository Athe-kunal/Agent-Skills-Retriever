# Run: streamlit run ast_skills/train/scenario_query_annotation_app.py
"""Streamlit UI to annotate scenario query rows (filter seed questions and scenario outputs)."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import streamlit as st
from loguru import logger as log

from ast_skills.persona_data_gen.datamodels import (
    ScenarioQueryPromptRowDataModel,
    ScenarioRelatedOutput,
)
from ast_skills.train.scenario_query_row_io import (
    read_annotated_rows_map,
    read_scenario_query_prompt_rows,
    write_annotated_jsonl,
)


def _custom_id_key_fragment(custom_id: str) -> str:
    return hashlib.sha256(custom_id.encode("utf-8")).hexdigest()[:24]


def _source_row_by_id(rows: list[ScenarioQueryPromptRowDataModel]) -> dict[str, ScenarioQueryPromptRowDataModel]:
    return {row.custom_id: row for row in rows}


def _init_widget_defaults(
    custom_id: str,
    source_row: ScenarioQueryPromptRowDataModel,
    annotated_row: ScenarioQueryPromptRowDataModel | None,
    revision: int,
) -> None:
    """Ensure checkbox keys exist for this row (opt-in: unchecked unless present in annotation)."""
    frag = _custom_id_key_fragment(custom_id)
    annotated_seeds = set(annotated_row.seed_questions) if annotated_row else set()
    annotated_pairs = (
        {(s.scenario, s.question) for s in annotated_row.scenario_output} if annotated_row else set()
    )
    for i, question in enumerate(source_row.seed_questions):
        key = f"seed_{revision}_{frag}_{i}"
        if key not in st.session_state:
            st.session_state[key] = question in annotated_seeds
    for j, item in enumerate(source_row.scenario_output):
        key = f"scen_{revision}_{frag}_{j}"
        if key not in st.session_state:
            pair = (item.scenario, item.question)
            st.session_state[key] = pair in annotated_pairs


def _collect_seed_selection(
    custom_id: str, source_row: ScenarioQueryPromptRowDataModel, revision: int
) -> list[str]:
    frag = _custom_id_key_fragment(custom_id)
    selected: list[str] = []
    for i, question in enumerate(source_row.seed_questions):
        key = f"seed_{revision}_{frag}_{i}"
        if st.session_state.get(key, False):
            selected.append(question)
    return selected


def _collect_scenario_selection(
    custom_id: str, source_row: ScenarioQueryPromptRowDataModel, revision: int
) -> list[ScenarioRelatedOutput]:
    frag = _custom_id_key_fragment(custom_id)
    selected: list[ScenarioRelatedOutput] = []
    for j, item in enumerate(source_row.scenario_output):
        key = f"scen_{revision}_{frag}_{j}"
        if st.session_state.get(key, False):
            selected.append(item)
    return selected


def _sync_checkbox_state_after_save(
    custom_id: str,
    source_row: ScenarioQueryPromptRowDataModel,
    revision: int,
    selected_seeds: list[str],
    selected_scenarios: list[ScenarioRelatedOutput],
) -> None:
    """Align widget keys with the row just written (subset = checked)."""
    frag = _custom_id_key_fragment(custom_id)
    seed_set = set(selected_seeds)
    pair_set = {(s.scenario, s.question) for s in selected_scenarios}
    for i, question in enumerate(source_row.seed_questions):
        st.session_state[f"seed_{revision}_{frag}_{i}"] = question in seed_set
    for j, item in enumerate(source_row.scenario_output):
        st.session_state[f"scen_{revision}_{frag}_{j}"] = (
            item.scenario,
            item.question,
        ) in pair_set


def main() -> None:
    st.set_page_config(page_title="Scenario query annotation", layout="wide")
    st.title("Scenario query row annotation")

    if "data_revision" not in st.session_state:
        st.session_state.data_revision = 0
    if "row_index" not in st.session_state:
        st.session_state.row_index = 0
    if "source_rows" not in st.session_state:
        st.session_state.source_rows: list[ScenarioQueryPromptRowDataModel] = []
    if "annotated_by_id" not in st.session_state:
        st.session_state.annotated_by_id: dict[str, ScenarioQueryPromptRowDataModel] = {}

    default_source = "artifacts/scenario_query_prompt_row_data_models.jsonl"
    default_sink = "artifacts/scenario_query_prompt_row_data_models_annotated.jsonl"

    with st.sidebar:
        st.header("Paths")
        source_path = st.text_input("Source JSONL", value=default_source)
        sink_path = st.text_input("Annotated output JSONL", value=default_sink)
        if st.button("Load / reload from disk"):
            st.session_state.source_rows = read_scenario_query_prompt_rows(source_path)
            st.session_state.annotated_by_id = read_annotated_rows_map(sink_path)
            st.session_state.data_revision += 1
            st.session_state.row_index = 0
            log.info(
                f"{source_path=} {sink_path=} "
                f"{len(st.session_state.source_rows)=} "
                f"{len(st.session_state.annotated_by_id)=}"
            )
            st.rerun()

    rows: list[ScenarioQueryPromptRowDataModel] = st.session_state.source_rows
    if not rows:
        st.info('Click **Load / reload from disk** in the sidebar after setting paths.')
        return

    revision = st.session_state.data_revision
    source_by_id = _source_row_by_id(rows)
    idx = int(st.session_state.row_index)
    idx = max(0, min(idx, len(rows) - 1))
    st.session_state.row_index = idx

    current = rows[idx]
    custom_id = current.custom_id
    annotated = st.session_state.annotated_by_id.get(custom_id)

    st.subheader(f"Row {idx + 1} of {len(rows)} — `{custom_id}`")
    if annotated:
        st.caption("This row has saved annotations (checkbox defaults restored from sink).")

    nav_prev, nav_next, _spacer = st.columns([1, 1, 6])
    with nav_prev:
        if st.button("Prev", disabled=idx <= 0):
            st.session_state.row_index = idx - 1
            st.rerun()
    with nav_next:
        if st.button("Next", disabled=idx >= len(rows) - 1):
            st.session_state.row_index = idx + 1
            st.rerun()

    source_row = source_by_id[custom_id]
    _init_widget_defaults(custom_id, source_row, annotated, revision)

    left, right = st.columns(2)
    with left:
        st.markdown("### Markdown content")
        st.markdown(source_row.markdown_content)
    with right:
        st.markdown("### Summary")
        st.text_area(
            "summary",
            value=source_row.summary,
            height=240,
            disabled=True,
            label_visibility="collapsed",
            key=f"summary_ro_{_custom_id_key_fragment(custom_id)}_{revision}",
        )

    st.markdown("### Seed questions (mark relevant)")
    frag = _custom_id_key_fragment(custom_id)
    for i, question in enumerate(source_row.seed_questions):
        key = f"seed_{revision}_{frag}_{i}"
        preview = question if len(question) <= 200 else question[:200] + "…"
        with st.expander(f"Q{i + 1}: {preview}", expanded=False):
            st.checkbox("Relevant", key=key)
            st.markdown("")

    st.markdown("### Scenario output (mark relevant scenario + question pairs)")
    for j, item in enumerate(source_row.scenario_output):
        key = f"scen_{revision}_{frag}_{j}"
        title = f"Pair {j + 1}"
        with st.expander(title, expanded=False):
            st.checkbox("Relevant", key=key)
            st.markdown("**Scenario**")
            st.markdown(item.scenario)
            st.markdown("**Question**")
            st.markdown(item.question)

    if st.button("Submit for this custom_id"):
        selected_seeds = _collect_seed_selection(custom_id, source_row, revision)
        selected_scenarios = _collect_scenario_selection(custom_id, source_row, revision)
        if not selected_seeds:
            st.error("Select at least one relevant seed question.")
            return
        if not selected_scenarios:
            st.error("Select at least one relevant scenario output pair.")
            return
        updated = replace(
            source_row,
            seed_questions=selected_seeds,
            scenario_output=selected_scenarios,
        )
        st.session_state.annotated_by_id[custom_id] = updated
        write_annotated_jsonl(sink_path, st.session_state.annotated_by_id)
        _sync_checkbox_state_after_save(
            custom_id, source_row, revision, selected_seeds, selected_scenarios
        )
        log.info(
            f"{custom_id=} {sink_path=} {len(selected_seeds)=} {len(selected_scenarios)=}"
        )
        st.success(f"Saved `{custom_id}` to `{sink_path}`.")


if __name__ == "__main__":
    main()
