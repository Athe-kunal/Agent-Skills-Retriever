"""Streamlit app for DBSCAN cluster exploration across retriever fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from loguru import logger as log
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class _DbConfig(NamedTuple):
    """Settings for one field-specific ChromaDB database."""

    field_name: str
    db_dir_name: str
    collection_name: str


class _EmbeddingFrameResult(NamedTuple):
    """Dataframe and diagnostics from ChromaDB collection loading."""

    frame: pd.DataFrame
    row_count: int


DB_CONFIGS: tuple[_DbConfig, ...] = (
    _DbConfig("what", "what_db", "retriever_what"),
    _DbConfig("why", "why_db", "retriever_why"),
    _DbConfig("description", "description_db", "retriever_description"),
)


def _load_field_frame(root_dir: Path, config: _DbConfig) -> _EmbeddingFrameResult:
    db_path = root_dir / config.db_dir_name
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=config.collection_name)
    records = collection.get(include=["embeddings", "metadatas", "documents"])

    ids = records.get("ids") or []
    embeddings = records.get("embeddings") or []
    metadatas = records.get("metadatas") or []
    documents = records.get("documents") or []

    frame = pd.DataFrame(
        {
            "id": ids,
            "text": documents,
            "name": [metadata.get("name", "") for metadata in metadatas],
            "custom_id": [metadata.get("custom_id", "") for metadata in metadatas],
            "seed_questions": [metadata.get("seed_questions", "") for metadata in metadatas],
            "metadata_json": [metadata.get("metadata_json", "") for metadata in metadatas],
            "embedding": embeddings,
        }
    )
    log.info(f"{config.field_name=}, {db_path=}, {len(frame)=}")
    return _EmbeddingFrameResult(frame=frame, row_count=len(frame))


def _project_embeddings(embeddings: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    projection = PCA(n_components=2, random_state=7).fit_transform(scaled)
    return projection


def _run_dbscan(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(embeddings)
    return labels


def _build_cluster_frame(frame: pd.DataFrame, eps: float, min_samples: int) -> pd.DataFrame:
    if frame.empty:
        return frame

    embeddings = np.array(frame["embedding"].tolist(), dtype=np.float64)
    labels = _run_dbscan(embeddings=embeddings, eps=eps, min_samples=min_samples)
    points = _project_embeddings(embeddings=embeddings)

    cluster_frame = frame.copy()
    cluster_frame["cluster"] = labels.astype(str)
    cluster_frame["x"] = points[:, 0]
    cluster_frame["y"] = points[:, 1]
    return cluster_frame


def _plot_clusters(frame: pd.DataFrame, field_name: str) -> None:
    if frame.empty:
        st.warning(f"No records found for `{field_name}`.")
        return

    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster",
        hover_name="name",
        hover_data={
            "custom_id": True,
            "seed_questions": True,
            "text": True,
            "metadata_json": True,
            "x": False,
            "y": False,
        },
        title=f"DBSCAN clusters for {field_name}",
        width=1100,
        height=650,
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_field_tab(
    root_dir: Path,
    config: _DbConfig,
    eps: float,
    min_samples: int,
) -> None:
    result = _load_field_frame(root_dir=root_dir, config=config)
    st.caption(f"Rows loaded: {result.row_count}")
    clustered_frame = _build_cluster_frame(
        frame=result.frame,
        eps=eps,
        min_samples=min_samples,
    )

    if not clustered_frame.empty:
        cluster_count = clustered_frame["cluster"].nunique()
        noise_count = int((clustered_frame["cluster"] == "-1").sum())
        st.write(f"Clusters: {cluster_count} (noise rows: {noise_count})")

    _plot_clusters(frame=clustered_frame, field_name=config.field_name)


def main() -> None:
    st.set_page_config(page_title="Retriever Cluster Visualizer", layout="wide")
    st.title("Retriever Embedding Cluster Explorer")

    root_dir_text = st.sidebar.text_input(
        "Chroma root directory",
        value="artifacts/chroma",
    )
    eps = st.sidebar.slider("DBSCAN eps", min_value=0.05, max_value=1.0, value=0.35)
    min_samples = st.sidebar.slider("DBSCAN min_samples", min_value=2, max_value=50, value=5)

    root_dir = Path(root_dir_text)
    st.sidebar.write(f"Using root directory: `{root_dir}`")
    tabs = st.tabs([config.field_name for config in DB_CONFIGS])

    for tab, config in zip(tabs, DB_CONFIGS):
        with tab:
            try:
                _render_field_tab(
                    root_dir=root_dir,
                    config=config,
                    eps=eps,
                    min_samples=min_samples,
                )
            except Exception as exc:
                log.exception(f"{config.field_name=}, {exc=}")
                st.error(f"Failed to load `{config.field_name}`: {exc}")


if __name__ == "__main__":
    main()
