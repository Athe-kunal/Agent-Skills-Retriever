"""Streamlit app for UMAP + cluster exploration across retriever fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NamedTuple

import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from loguru import logger as log
from sklearn.cluster import DBSCAN, HDBSCAN
from umap import UMAP


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
    _DbConfig("summary", "summary_db", "retriever_summary"),
    _DbConfig("description", "description_db", "retriever_description"),
)

_CHROMA_PAGE_SIZE = 500

AlgorithmChoice = Literal["DBSCAN", "HDBSCAN"]


def _fetch_records_paged(
    collection: Any,
    page_size: int,
) -> dict[str, list[Any]]:
    """Fetches all records from a ChromaDB collection in pages.

    SQLite (used internally by ChromaDB) caps SQL bind variables at 999.
    Fetching everything in one call exceeds that limit for large collections,
    so we paginate with ``limit`` / ``offset`` and merge the pages.
    """
    total = collection.count()
    log.info(f"{total=}, {page_size=}")

    merged: dict[str, list[Any]] = {"ids": [], "embeddings": [], "metadatas": []}

    for offset in range(0, total, page_size):
        page = collection.get(
            include=["embeddings", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        ids_page = page.get("ids")
        embeddings_page = page.get("embeddings")
        metadatas_page = page.get("metadatas")

        merged["ids"].extend(ids_page if ids_page is not None else [])
        merged["embeddings"].extend(embeddings_page if embeddings_page is not None else [])
        merged["metadatas"].extend(metadatas_page if metadatas_page is not None else [])
        log.info(f"{offset=}, fetched={len(ids_page or [])}")

    return merged


@st.cache_data
def _load_field_frame(root_dir: Path, config: _DbConfig) -> _EmbeddingFrameResult:
    """Loads embeddings and names from ChromaDB, cached for the session."""
    db_path = root_dir / config.db_dir_name
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=config.collection_name)
    records = _fetch_records_paged(collection=collection, page_size=_CHROMA_PAGE_SIZE)

    ids = records["ids"]
    embeddings = records["embeddings"]
    metadatas = records["metadatas"]

    frame = pd.DataFrame(
        {
            "id": ids,
            "name": [metadata.get("name", "") for metadata in metadatas],
            "embedding": embeddings,
        }
    )
    log.info(f"{config.field_name=}, {db_path=}, {len(frame)=}")
    return _EmbeddingFrameResult(frame=frame, row_count=len(frame))


@st.cache_data
def _umap_for_clustering(
    root_dir: Path,
    config: _DbConfig,
    n_neighbors: int,
    n_components: int,
) -> np.ndarray:
    """Reduces raw embeddings to ``n_components`` dims with UMAP for clustering.

    Running clustering on raw high-dimensional vectors suffers from the
    curse of dimensionality: all pairwise distances collapse to a narrow
    range, making density-based algorithms see everything as one blob.
    UMAP compresses the manifold into a low-dimensional Euclidean space
    where local neighbourhood structure is preserved, giving clustering
    algorithms meaningful distance information to work with.
    """
    result = _load_field_frame(root_dir, config)
    embeddings = np.array(result.frame["embedding"].tolist(), dtype=np.float32)
    log.info(f"{config.field_name=}, {n_neighbors=}, {n_components=}, {embeddings.shape=}")
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=7,
    )
    return reducer.fit_transform(embeddings)


@st.cache_data
def _umap_2d(root_dir: Path, config: _DbConfig, n_neighbors: int) -> np.ndarray:
    """Reduces raw embeddings to 2D with UMAP for scatter visualization.

    UMAP preserves both local cluster structure and global topology far
    better than PCA (which only captures linear variance) and t-SNE
    (which distorts global distances). The 2D projection is computed
    directly from the original high-dimensional embeddings so the visual
    layout is independent of the clustering reduction step.
    """
    result = _load_field_frame(root_dir, config)
    embeddings = np.array(result.frame["embedding"].tolist(), dtype=np.float32)
    log.info(f"{config.field_name=}, {n_neighbors=}, {embeddings.shape=}")
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=7,
    )
    return reducer.fit_transform(embeddings)


def _run_dbscan(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Runs DBSCAN on already UMAP-reduced embeddings with Euclidean metric."""
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(embeddings)
    log.info(f"{eps=}, {min_samples=}, unique_labels={np.unique(labels).tolist()}")
    return labels


def _run_hdbscan(embeddings: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    """Runs HDBSCAN on already UMAP-reduced embeddings.

    HDBSCAN (Hierarchical DBSCAN) does not require an ``eps`` parameter.
    It builds a cluster hierarchy and extracts the most stable clusters,
    making it far more robust to varying local densities.
    """
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    ).fit_predict(embeddings)
    log.info(f"{min_cluster_size=}, {min_samples=}, unique_labels={np.unique(labels).tolist()}")
    return labels


def _build_cluster_frame(
    root_dir: Path,
    config: _DbConfig,
    algorithm: AlgorithmChoice,
    umap_n_neighbors: int,
    umap_cluster_dims: int,
    eps: float,
    min_samples: int,
    min_cluster_size: int,
) -> pd.DataFrame:
    """Assembles the plot-ready DataFrame with cluster labels and 2D coords."""
    result = _load_field_frame(root_dir=root_dir, config=config)
    if result.frame.empty:
        return result.frame

    reduced = _umap_for_clustering(
        root_dir=root_dir,
        config=config,
        n_neighbors=umap_n_neighbors,
        n_components=umap_cluster_dims,
    )
    points_2d = _umap_2d(root_dir=root_dir, config=config, n_neighbors=umap_n_neighbors)

    if algorithm == "DBSCAN":
        labels = _run_dbscan(embeddings=reduced, eps=eps, min_samples=min_samples)
    else:
        labels = _run_hdbscan(
            embeddings=reduced,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

    cluster_frame = result.frame.drop(columns=["embedding"]).copy()
    cluster_frame["cluster"] = labels.astype(str)
    cluster_frame["x"] = points_2d[:, 0]
    cluster_frame["y"] = points_2d[:, 1]
    return cluster_frame


def _plot_clusters(frame: pd.DataFrame, field_name: str, algorithm: AlgorithmChoice) -> None:
    if frame.empty:
        st.warning(f"No records found for `{field_name}`.")
        return

    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster",
        hover_name="name",
        hover_data={"x": False, "y": False, "cluster": False},
        title=f"{algorithm} clusters for {field_name} (UMAP projection)",
        width=1100,
        height=650,
    )
    figure.update_traces(marker={"size": 4, "opacity": 0.75})
    st.plotly_chart(figure, use_container_width=True)


def _render_field_tab(
    root_dir: Path,
    config: _DbConfig,
    algorithm: AlgorithmChoice,
    umap_n_neighbors: int,
    umap_cluster_dims: int,
    eps: float,
    min_samples: int,
    min_cluster_size: int,
) -> None:
    result = _load_field_frame(root_dir=root_dir, config=config)
    st.caption(f"Rows loaded: {result.row_count}")

    with st.spinner("Running UMAP + clustering…"):
        clustered_frame = _build_cluster_frame(
            root_dir=root_dir,
            config=config,
            algorithm=algorithm,
            umap_n_neighbors=umap_n_neighbors,
            umap_cluster_dims=umap_cluster_dims,
            eps=eps,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        )

    if not clustered_frame.empty:
        cluster_count = clustered_frame["cluster"].nunique()
        noise_count = int((clustered_frame["cluster"] == "-1").sum())
        st.write(f"Clusters: {cluster_count} (noise rows: {noise_count})")

    _plot_clusters(frame=clustered_frame, field_name=config.field_name, algorithm=algorithm)


def main() -> None:
    st.set_page_config(page_title="Retriever Cluster Visualizer", layout="wide")
    st.title("Retriever Embedding Cluster Explorer")

    sb = st.sidebar

    root_dir_text = sb.text_input("Chroma root directory", value="artifacts/chroma")
    root_dir = Path(root_dir_text)
    sb.write(f"Using root directory: `{root_dir}`")

    sb.divider()
    sb.subheader("UMAP")
    umap_n_neighbors = sb.slider("n_neighbors", min_value=5, max_value=100, value=15)
    umap_cluster_dims = sb.slider("Cluster reduction dims", min_value=5, max_value=50, value=20)

    sb.divider()
    sb.subheader("Clustering")
    algorithm: AlgorithmChoice = sb.selectbox("Algorithm", ["HDBSCAN", "DBSCAN"])

    eps = 0.3
    min_samples = 5
    min_cluster_size = 30

    if algorithm == "DBSCAN":
        eps = sb.slider("eps", min_value=0.05, max_value=3.0, value=0.3, step=0.05)
        min_samples = sb.slider("min_samples", min_value=2, max_value=100, value=5)
    else:
        min_cluster_size = sb.slider("min_cluster_size", min_value=5, max_value=500, value=30)
        min_samples = sb.slider("min_samples", min_value=1, max_value=50, value=5)

    tabs = st.tabs([config.field_name for config in DB_CONFIGS])

    for tab, config in zip(tabs, DB_CONFIGS):
        with tab:
            try:
                _render_field_tab(
                    root_dir=root_dir,
                    config=config,
                    algorithm=algorithm,
                    umap_n_neighbors=umap_n_neighbors,
                    umap_cluster_dims=umap_cluster_dims,
                    eps=eps,
                    min_samples=min_samples,
                    min_cluster_size=min_cluster_size,
                )
            except Exception as exc:
                log.exception(f"{config.field_name=}, {exc=}")
                st.error(f"Failed to load `{config.field_name}`: {exc}")


if __name__ == "__main__":
    main()
