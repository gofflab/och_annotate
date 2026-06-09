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
        df = df[df["embedding"].notna()].reset_index(drop=True)
        # Older caches may lack metadata columns added to the config later; pull
        # any missing ones from Baserow (metadata only — no embedding calls).
        missing = [c for c in config.baserow.metadata_columns if c not in df.columns]
        if missing:
            try:
                df = _augment_metadata(df, config, missing)
            except Exception:  # noqa: BLE001 - offline / no token -> skip augmentation
                pass
        return df

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


def _augment_metadata(
    df: pd.DataFrame, config: Config, columns: list[str], *, attempts: int = 5
) -> pd.DataFrame:
    """Left-merge extra Baserow columns into ``df``, keyed by the id column.

    A cheap metadata-only backfill for fields absent from a cached run. Baserow
    can return transient 500s, so the page sweep is retried with backoff; the
    ``include=`` field filter is avoided (it 500s on this table) by fetching full
    rows and selecting columns locally.
    """
    import time

    bw = config.baserow
    id_col = bw.id_column
    client = BaserowClient(bw.base_url, config.baserow_token)
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            rows = list(client.iter_rows(bw.table_id))
            break
        except Exception as err:  # noqa: BLE001 - retry transient server errors
            last_err = err
            time.sleep(2 * (attempt + 1))
    else:
        raise last_err  # type: ignore[misc]

    extra = pd.DataFrame(
        {id_col: row.get(id_col), **{c: row.get(c) for c in columns}} for row in rows
    )
    if extra.empty:
        return df
    return df.merge(extra, on=id_col, how="left")


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


def cluster_feature_profile(adata, groupby: str = "leiden", n: int = 10) -> pd.DataFrame:
    """Per-cluster SAE-feature profile ranked by mean normalized activation.

    A salience + ubiquity view that complements ``sae_enrichment`` (which is
    differential). For each cluster returns the top-``n`` features by
    ``mean_activation`` — the mean *normalized* activation across cluster members
    (0 where the feature is inactive) — alongside ``occurrence``, the fraction of
    members in which the feature is active. Universal features (occurrence ~1)
    define the family; partial ones flag subgroups/domain variants.

    Returns tidy ``[cluster, rank, sae_feature, mean_activation, occurrence,
    n_members]``.
    """
    from scipy import sparse

    matrix = adata.X
    matrix = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
    labels = adata.obs[groupby]
    categories = labels.cat.categories if hasattr(labels, "cat") else sorted(labels.unique())
    feat_ids = np.asarray(adata.var_names)

    rows = []
    for cluster in categories:
        mask = (labels == cluster).to_numpy()
        members = int(mask.sum())
        if members == 0:
            continue
        sub = matrix[mask]
        mean_act = np.asarray(sub.mean(axis=0)).ravel()
        occurrence = np.asarray((sub > 0).sum(axis=0)).ravel() / members
        for rank, f in enumerate(np.argsort(mean_act)[::-1][:n], start=1):
            rows.append({
                "cluster": cluster,
                "rank": rank,
                "sae_feature": int(feat_ids[f]),
                "mean_activation": float(mean_act[f]),
                "occurrence": float(occurrence[f]),
                "n_members": members,
            })
    return pd.DataFrame(rows)


def _pick_column(columns, candidates) -> str | None:
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def load_sae_descriptions(path) -> dict[str, str]:
    """Load a ``feature_index -> natural-language description`` mapping.

    Returns ``{}`` if the file is absent (so the notebook degrades gracefully).
    Accepts CSV/TSV/JSON/parquet and auto-detects the index column
    (feature/index/id/latent/…) and the text column
    (description/label/explanation/annotation/summary/…). Keys are stringified
    feature indices to match the AnnData ``var_names`` / enrichment ``sae_feature``.
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {}
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        df = pd.DataFrame(data)
    elif suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, sep="\t" if suffix == ".tsv" else ",")
    idx_col = _pick_column(
        df.columns, ["feature", "index", "id", "feature_id", "feature_index", "latent", "latent_index"]
    )
    txt_col = _pick_column(
        df.columns, ["description", "label", "explanation", "annotation", "summary", "text", "name"]
    )
    if idx_col is None or txt_col is None:
        raise ValueError(
            f"Could not find index/description columns in {list(df.columns)} — "
            "expected something like 'feature'/'index' and 'description'/'label'."
        )
    return {str(i): str(t) for i, t in zip(df[idx_col], df[txt_col])}


def annotate_enrichment(enrichment: pd.DataFrame, descriptions: dict[str, str],
                        feature_col: str = "sae_feature") -> pd.DataFrame:
    """Add a ``description`` column to an enrichment table from a descriptions map."""
    out = enrichment.copy()
    out["description"] = out[feature_col].astype(str).map(lambda f: descriptions.get(f, ""))
    return out


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
    hover = [h for h in hover if h in df.columns]  # tolerate optional cols being absent
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
