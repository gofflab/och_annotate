"""Downstream exploration: load embeddings -> UMAP -> interactive plot.

Embeddings are loaded from the local cache when present (free), otherwise pulled
from Baserow and parsed from the JSON embedding column. Nothing is written to
disk unless you explicitly call ``save_html``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from och_annotate.baserow import BaserowClient
from och_annotate.cache import EmbeddingCache
from och_annotate.config import Config


def load_embeddings(config: Config, *, prefer_cache: bool = True) -> pd.DataFrame:
    """Return a DataFrame of metadata + an ``embedding`` column (list[float])."""
    cache = EmbeddingCache(config.run.cache_dir, id_column=config.baserow.id_column)
    if prefer_cache and len(cache) > 0:
        df = cache.to_frame()
        return df[df["embedding"].notna()].reset_index(drop=True)

    # Fall back to Baserow.
    bw = config.baserow
    client = BaserowClient(bw.base_url, config.baserow_token)
    emb_col = bw.output_columns["embedding"]
    wanted = set(bw.metadata_columns) | {bw.id_column, emb_col}
    records = []
    for row in client.iter_rows(bw.table_id):
        raw = row.get(emb_col)
        if not raw:
            continue
        rec = {k: row.get(k) for k in wanted}
        rec["embedding"] = json.loads(raw) if isinstance(raw, str) else raw
        records.append(rec)
    return pd.DataFrame(records)


def embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].apply(np.asarray).to_numpy())


def sae_feature_activation(
    df: pd.DataFrame, feature_index: int, sae_model: str | None = None
) -> pd.Series:
    """Per-protein activation of one SAE feature, for coloring a UMAP.

    Only the top-K features per protein are stored, so a protein that doesn't
    have ``feature_index`` among its top-K returns ``0.0`` — which is the right
    reading for sparse SAE features (that feature simply isn't active there).
    """
    if "sae_top_features" not in df.columns:
        raise KeyError(
            "No 'sae_top_features' column — embed/sae with SAE enabled first."
        )

    def _activation(cell) -> float:
        if not cell:
            return 0.0
        feats = json.loads(cell) if isinstance(cell, str) else cell
        entries = [feats.get(sae_model)] if sae_model else list(feats.values())
        for entry in entries:
            if not entry:
                continue
            indices = list(entry.get("indices", []))
            if feature_index in indices:
                return float(entry["activations"][indices.index(feature_index)])
        return 0.0

    return df["sae_top_features"].apply(_activation)


def run_umap(
    df: pd.DataFrame,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    random_state: int = 0,
) -> pd.DataFrame:
    """Add UMAP coordinate columns (umap_0..umap_{n-1}) to a copy of ``df``."""
    import umap  # local import; heavy

    matrix = embedding_matrix(df)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(matrix)
    out = df.copy().reset_index(drop=True)
    for i in range(n_components):
        out[f"umap_{i}"] = coords[:, i]
    return out


def plot_umap(
    df: pd.DataFrame,
    *,
    color: str | None = None,
    hover: list[str] | None = None,
    title: str = "Proteome UMAP",
):
    """Interactive 2-D scatter of UMAP coordinates colored by a metadata column."""
    import plotly.express as px

    hover = hover or [c for c in df.columns if not c.startswith("umap_") and c != "embedding"]
    fig = px.scatter(
        df,
        x="umap_0",
        y="umap_1",
        color=color,
        hover_data=hover,
        title=title,
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend_title_text=color or "")
    return fig


def save_html(fig, path: str) -> None:
    """Explicit opt-in export of an interactive plot to a standalone HTML file."""
    fig.write_html(path, include_plotlyjs="cdn")
