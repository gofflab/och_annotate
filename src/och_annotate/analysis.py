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


def _resolve_sae_model(df: pd.DataFrame, sae_model: str | None) -> str | None:
    if sae_model:
        return sae_model
    for cell in df["sae_top_features"]:
        if not cell:
            continue
        feats = json.loads(cell) if isinstance(cell, str) else cell
        if feats:
            return next(iter(feats.keys()))
    return None


def sae_feature_matrix(df: pd.DataFrame, sae_model: str | None = None, n_features: int | None = None):
    """Sparse proteins × features SAE-activation matrix from the stored top-K.

    Column ``j`` is SAE feature ``j`` (so var_names == feature ids). Returns
    ``(scipy.sparse.csr_matrix, resolved_sae_model)``.
    """
    from scipy import sparse

    if "sae_top_features" not in df.columns:
        raise KeyError("No 'sae_top_features' column — embed/sae with SAE enabled first.")
    model = _resolve_sae_model(df, sae_model)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    max_idx = -1
    for i, cell in enumerate(df["sae_top_features"]):
        if not cell:
            continue
        feats = json.loads(cell) if isinstance(cell, str) else cell
        entry = feats.get(model) if model else None
        if not entry:
            continue
        for j, a in zip(entry.get("indices", []), entry.get("activations", [])):
            rows.append(i)
            cols.append(int(j))
            vals.append(float(a))
            max_idx = max(max_idx, int(j))
    width = n_features if n_features is not None else max_idx + 1
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(df), max(width, 1)))
    return matrix, model


def build_anndata(df: pd.DataFrame, sae_model: str | None = None, n_features: int | None = None):
    """Assemble an AnnData for Leiden clustering + SAE-feature enrichment.

    ``X`` = sparse SAE activations (proteins × features), ``obsm['X_esmc']`` =
    the dense ESMC embeddings (cluster on these), ``obs`` = metadata columns.
    """
    import anndata as ad

    matrix, model = sae_feature_matrix(df, sae_model=sae_model, n_features=n_features)
    obs = df[[c for c in df.columns if c not in ("embedding", "sae_top_features")]].copy()
    obs = obs.reset_index(drop=True)
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=matrix.astype("float32"), obs=obs)
    adata.var_names = [str(j) for j in range(matrix.shape[1])]
    adata.obsm["X_esmc"] = embedding_matrix(df)
    if "transcript_id" in obs.columns and obs["transcript_id"].is_unique:
        adata.obs_names = obs["transcript_id"].astype(str).values
    adata.uns["sae_model"] = model
    return adata


def sae_enrichment(adata, groupby: str = "leiden", method: str = "wilcoxon", n: int | None = None):
    """Per-cluster SAE-feature enrichment (marker-gene style). Returns a tidy DataFrame."""
    import scanpy as sc

    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)
    table = sc.get.rank_genes_groups_df(adata, group=None)
    table = table.rename(columns={"group": groupby, "names": "sae_feature"})
    if n is not None:
        table = table.groupby(groupby, observed=True).head(n).reset_index(drop=True)
    return table


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
