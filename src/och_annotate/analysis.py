"""Downstream exploration: load embeddings -> UMAP -> interactive plot.

Embeddings are loaded from the local cache when present (free), otherwise pulled
from Baserow and parsed from the JSON embedding column. Nothing is written to
disk unless you explicitly call ``save_html``.
"""

from __future__ import annotations

import json
import re

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
    # Pull the SAE column too when configured, so SAE-feature analysis works even
    # without a local cache (the cache path already carries it).
    sae_col = bw.output_columns.get("sae")
    wanted = set(bw.metadata_columns) | {bw.id_column, emb_col}
    if sae_col:
        wanted.add(sae_col)
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


def feature_residue_region(sae_cell, feature_index: int, sae_model: str | None = None):
    """Return ``[start, end, peak]`` residue span for a feature in one protein.

    Reads the optional ``regions`` array stored alongside ``indices``/``activations``
    in ``sae_top_features``. Returns ``None`` when residue regions weren't computed
    (the cache predates a per-residue SAE run), so callers degrade gracefully.
    """
    if not sae_cell:
        return None
    feats = json.loads(sae_cell) if isinstance(sae_cell, str) else sae_cell
    entries = [feats.get(sae_model)] if sae_model else list(feats.values())
    for entry in entries:
        if not entry or "regions" not in entry:
            continue
        indices = list(entry.get("indices", []))
        if feature_index in indices:
            return list(entry["regions"][indices.index(feature_index)])
    return None


def candidate_feature_report(
    sae_cell, *, labels: dict[int, str] | None = None, sae_model: str | None = None, n: int = 10
) -> pd.DataFrame:
    """Per-candidate feature report: top-``n`` SAE features for one protein.

    Columns: ``rank, sae_feature, label, norm_activation, residues`` — ranked by
    the stored (already normalized) activation. ``residues`` is
    ``"<start>-<end> (peak <p>)"`` when residue regions are present, else ``None``
    (so it lights up automatically once a per-residue SAE run populates them).
    """
    feats = json.loads(sae_cell) if isinstance(sae_cell, str) else (sae_cell or {})
    model = sae_model or next(iter(feats), None)
    entry = feats.get(model) if model else None
    columns = ["rank", "sae_feature", "label", "norm_activation", "residues"]
    if not entry:
        return pd.DataFrame(columns=columns)

    labels = labels or {}
    indices = entry.get("indices", [])
    activations = entry.get("activations", [])
    rows = []
    for rank, (fi, act) in enumerate(zip(indices[:n], activations[:n]), start=1):
        region = feature_residue_region(feats, int(fi), model)
        rows.append({
            "rank": rank,
            "sae_feature": int(fi),
            "label": labels.get(int(fi), ""),
            "norm_activation": float(act),
            "residues": f"{region[0]}-{region[1]} (peak {region[2]})" if region else None,
        })
    return pd.DataFrame(rows, columns=columns)


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


def sae_width_from_model(sae_model: str | None) -> int | None:
    """Full SAE feature count parsed from a model id (``...codebook16384`` -> 16384)."""
    if not sae_model:
        return None
    m = re.search(r"codebook(\d+)", sae_model)
    return int(m.group(1)) if m else None


class SaeFeatureStore:
    """Persistent, updatable sparse store of full-width SAE feature vectors.

    Keeps the complete SAE feature embedding (e.g. 16,384-d) for each protein
    **outside Baserow**, one ``float32`` row per protein, in a single compressed
    ``.npz``. Only the non-zeros are stored (the ~k top-K activations from
    ``sae_top_features``), so it is lossless yet ~100× smaller than dense. Rows
    are keyed by the proteome id column and **upserted**, so the store grows
    incrementally as more proteins are embedded — no full rebuild.

    Typical use::

        store, stats = update_sae_feature_store(df, "data/sae_feature_matrix.npz")
        store.to_csr()        # scipy.sparse.csr_matrix (n_proteins x n_features), float32
        store.ids             # row order (protein ids)

    The on-disk ``.npz`` holds the CSR triplet (``data``/``indices``/``indptr``)
    plus ``shape``, the protein ``ids`` (row order), and ``sae_model``.
    """

    def __init__(self, path, *, id_column: str = "transcript_id"):
        from pathlib import Path

        self.path = Path(path)
        self.id_column = id_column
        self.sae_model: str | None = None
        self.n_features: int | None = None
        self._rows: dict = {}    # id -> (int32 indices, float32 activations)
        self._order: list = []   # stable row order (insertion / first-seen)
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        from scipy import sparse

        z = np.load(self.path, allow_pickle=True)
        mat = sparse.csr_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
        )
        self.n_features = int(z["shape"][1])
        model = str(z["sae_model"]) if "sae_model" in z.files else ""
        self.sae_model = model or None
        for i, pid in enumerate(z["ids"]):
            row = mat.getrow(i)
            self._rows[pid] = (row.indices.astype("int32"), row.data.astype("float32"))
            self._order.append(pid)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, pid) -> bool:
        return pid in self._rows

    @property
    def ids(self) -> list:
        return list(self._order)

    def upsert_frame(
        self, df: pd.DataFrame, *, sae_model: str | None = None,
        n_features: int | None = None, sae_column: str = "sae_top_features",
    ) -> dict:
        """Insert/replace rows from ``df`` (keyed by ``id_column``). Returns counts.

        Each row's full SAE vector is taken from its ``sae_column`` cell. Rows
        without an id or SAE entry are skipped. Refuses to mix SAE models or
        feature widths in one store (use a separate file per model).
        """
        if sae_column not in df.columns:
            raise KeyError(f"No {sae_column!r} column — load embeddings with SAE enabled.")
        if self.id_column not in df.columns:
            raise KeyError(f"No id column {self.id_column!r} in the frame.")

        model = sae_model or self.sae_model or _resolve_sae_model(df, None)
        if self.sae_model and model and model != self.sae_model:
            raise ValueError(
                f"Store holds SAE model {self.sae_model!r}; refusing to mix in {model!r}. "
                "Use a separate .npz per SAE model."
            )
        width = n_features or self.n_features or sae_width_from_model(model)

        added = updated = 0
        for pid, cell in zip(df[self.id_column], df[sae_column]):
            if pid is None or (isinstance(pid, float) and np.isnan(pid)) or not cell:
                continue
            feats = json.loads(cell) if isinstance(cell, str) else cell
            entry = feats.get(model) if model else None
            if not entry:
                continue
            idx = np.asarray(entry.get("indices", []), dtype="int32")
            val = np.asarray(entry.get("activations", []), dtype="float32")
            if pid in self._rows:
                updated += 1
            else:
                self._order.append(pid)
                added += 1
            self._rows[pid] = (idx, val)

        self.sae_model = model or self.sae_model
        if width is None:  # last resort: infer from the largest seen index
            width = 1 + max((int(i.max()) for i, _ in self._rows.values() if len(i)), default=0)
        if self.n_features and width != self.n_features:
            raise ValueError(
                f"Store width is {self.n_features}; new data implies {width}. "
                "Different SAE codebooks must use separate .npz files."
            )
        self.n_features = int(width)
        return {"added": added, "updated": updated, "total": len(self._order)}

    def upsert_row(self, pid, indices, activations, *, sae_model: str | None = None,
                   n_features: int | None = None) -> bool:
        """Insert/replace one protein's full sparse vector. Returns True if new.

        Lower-level than :meth:`upsert_frame` — used by the embedding pipeline to
        stream rows in as they are computed. ``indices``/``activations`` are the
        full non-zero pooled features for the protein.
        """
        if sae_model:
            if self.sae_model and sae_model != self.sae_model:
                raise ValueError(
                    f"Store holds SAE model {self.sae_model!r}; refusing {sae_model!r}."
                )
            self.sae_model = sae_model
        width = n_features or self.n_features or sae_width_from_model(self.sae_model)
        idx = np.asarray(indices, dtype="int32")
        val = np.asarray(activations, dtype="float32")
        is_new = pid not in self._rows
        if is_new:
            self._order.append(pid)
        self._rows[pid] = (idx, val)
        if width is None:
            width = 1 + max((int(idx.max()) if len(idx) else 0), (self.n_features or 1) - 1)
        if self.n_features and width != self.n_features:
            raise ValueError(
                f"Store width is {self.n_features}; new data implies {width}."
            )
        self.n_features = int(width)
        return is_new

    def to_csr(self):
        """Return the proteins × features sparse ``float32`` matrix (row order = ``ids``)."""
        from scipy import sparse

        data, indices, indptr = [], [], [0]
        for pid in self._order:
            idx, val = self._rows[pid]
            indices.append(idx)
            data.append(val)
            indptr.append(indptr[-1] + len(idx))
        data = np.concatenate(data) if data else np.zeros(0, dtype="float32")
        indices = np.concatenate(indices) if indices else np.zeros(0, dtype="int32")
        width = self.n_features or 1
        return sparse.csr_matrix(
            (data.astype("float32"), indices.astype("int32"), np.asarray(indptr, dtype="int64")),
            shape=(len(self._order), width),
        )

    def save(self) -> None:
        """Persist to ``self.path`` (compressed ``.npz``)."""
        m = self.to_csr()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            data=m.data.astype("float32"), indices=m.indices.astype("int32"),
            indptr=m.indptr.astype("int64"), shape=np.asarray(m.shape, dtype="int64"),
            ids=np.asarray(self._order, dtype=object),
            sae_model=np.asarray(self.sae_model or ""),
        )


def update_sae_feature_store(
    df: pd.DataFrame, path, *, id_column: str = "transcript_id",
    sae_model: str | None = None, n_features: int | None = None,
):
    """Load (or create) a :class:`SaeFeatureStore`, upsert ``df``, save, and return it.

    Returns ``(store, stats)`` where ``stats`` is ``{added, updated, total}``.
    Safe to call repeatedly — re-running with the same frame is a no-op upsert,
    and a frame with newly embedded proteins just extends the store.
    """
    store = SaeFeatureStore(path, id_column=id_column)
    stats = store.upsert_frame(df, sae_model=sae_model, n_features=n_features)
    store.save()
    return store, stats


def open_sae_feature_store(config: Config) -> "SaeFeatureStore":
    """Open the full-activation store at the config's path (incremental writes).

    Path is ``sae.feature_store_path`` or ``<run.cache_dir>/sae_feature_matrix.npz``.
    """
    from pathlib import Path

    path = config.sae.feature_store_path or (
        Path(config.run.cache_dir) / "sae_feature_matrix.npz"
    )
    return SaeFeatureStore(path, id_column=config.baserow.id_column)


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
    labels: dict[str, str] | None = None,
    title: str = "Proteome UMAP",
):
    """Interactive 2-D scatter of UMAP coordinates colored by a metadata column.

    ``hover`` references the dataframe's real column names (matching Baserow).
    ``labels`` is an optional ``{column: pretty_name}`` map applied at plot time
    for display only, so hover stays robust to Baserow's exact casing without
    renaming dataframe columns.
    """
    import plotly.express as px

    hover = hover or [c for c in df.columns if not c.startswith("umap_") and c != "embedding"]
    hover = [h for h in hover if h in df.columns]  # tolerate optional cols being absent
    fig = px.scatter(
        df,
        x="umap_0",
        y="umap_1",
        color=color,
        hover_data=hover,
        labels=labels or {},
        title=title,
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend_title_text=color or "")
    return fig


def plot_umap_searchable(
    df: pd.DataFrame,
    *,
    color: str | None = None,
    hover: list[str] | None = None,
    search_fields: list[str] | None = None,
    labels: dict[str, str] | None = None,
    title: str = "Proteome UMAP",
    size: int = 820,
    embed_js: bool = True,
):
    """``plot_umap`` plus a client-side gene search box, returned as embeddable HTML.

    Renders the same colored UMAP, then overlays a search box that highlights
    proteins whose metadata matches the query (comma/space-separated terms,
    case-insensitive substring across ``search_fields``). Matches are ringed,
    counted, and can be zoomed to. The search is pure JavaScript baked into the
    figure, so it keeps working in a static ``nbconvert`` HTML export with no
    live kernel.

    ``search_fields`` defaults to the gene-identifying ``hover`` columns. The
    plot is a fixed ``size``×``size`` square (equal axis aspect), centered, and
    non-responsive so it does not stretch with page width. Set ``embed_js=False``
    for every figure after the first in one document so plotly.js is embedded
    only once (it is reused from the page).

    Returns an ``IPython.display.HTML`` object so a notebook cell renders it
    directly.
    """
    import json
    import re

    import plotly.express as px
    import plotly.graph_objects as go
    from IPython.display import HTML

    hover = hover or [c for c in df.columns if not c.startswith("umap_") and c != "embedding"]
    hover = [h for h in hover if h in df.columns]
    search_fields = [f for f in (search_fields or hover) if f in df.columns]

    fig = px.scatter(
        df, x="umap_0", y="umap_1", color=color, hover_data=hover,
        labels=labels or {}, title=title, opacity=0.75,
    )
    fig.update_traces(marker=dict(size=5))
    # Fixed square figure with equal axis aspect (undistorted UMAP geometry).
    fig.update_layout(legend_title_text=color or "", width=size, height=size,
                      autosize=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # Hollow-ring overlay trace (empty until a search populates it via JS).
    fig.add_trace(go.Scattergl(
        x=[], y=[], mode="markers", name="search matches", text=[],
        marker=dict(size=15, color="rgba(0,0,0,0)",
                    line=dict(width=2.5, color="#111")),
        hovertemplate="%{text}<extra>match</extra>", showlegend=True,
    ))
    hi = len(fig.data) - 1  # index of the highlight trace

    # Per-protein search payload: coords, a display label, and a lowercased haystack.
    def _vals(row):
        return [str(row[f]) for f in search_fields if pd.notna(row.get(f)) and str(row[f]) != ""]
    records = [
        {"x": float(r["umap_0"]), "y": float(r["umap_1"]),
         "t": " · ".join(_vals(r)), "h": " ".join(_vals(r)).lower()}
        for _, r in df.iterrows()
    ]

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "umap"
    div_id = f"plot-{slug}"
    # Embed plotly.js inline on the first figure (embed_js=True) so the exported
    # HTML is self-contained / viewable offline; later figures reuse window.Plotly.
    plot_html = fig.to_html(
        include_plotlyjs=(True if embed_js else False),
        full_html=False, div_id=div_id,
        config={"responsive": False},  # don't stretch to container width
    )

    tokens = {
        "__DIV__": div_id, "__HI__": str(hi), "__SIZE__": str(size),
        "__RECORDS__": json.dumps(records, separators=(",", ":")),
        "__FIELDS__": ", ".join(search_fields),
    }
    # Centered, fixed-width wrapper so the whole block (controls + square plot)
    # stays put instead of expanding with the page.
    template = """
<div style="width:__SIZE__px;margin:0 auto;font-family:sans-serif;font-size:13px">
  <div style="margin:6px 0 2px 0">
    <input id="q-__DIV__" type="text" placeholder="search genes (e.g. Pax6, OG0000117)"
           style="padding:5px 8px;width:340px;border:1px solid #aaa;border-radius:4px"/>
    <button id="zoom-__DIV__" style="padding:5px 8px;margin-left:4px">zoom to matches</button>
    <button id="clear-__DIV__" style="padding:5px 8px">clear</button>
    <span id="n-__DIV__" style="margin-left:8px;color:#555"></span>
    <div style="color:#888;font-size:11px;margin-top:2px">searches: __FIELDS__</div>
  </div>
  __PLOT__
</div>
<script>
(function() {
  var RECS = __RECORDS__, HI = __HI__;
  function run() {
    var gd = document.getElementById("__DIV__");
    if (!gd || !window.Plotly) { return setTimeout(run, 120); }
    var q = document.getElementById("q-__DIV__"),
        zoom = document.getElementById("zoom-__DIV__"),
        clr = document.getElementById("clear-__DIV__"),
        nlab = document.getElementById("n-__DIV__"),
        timer = null, hits = [];
    function search() {
      var terms = q.value.toLowerCase().split(/[\\s,]+/).filter(function(t){return t;});
      hits = terms.length ? RECS.filter(function(r){
        return terms.some(function(t){ return r.h.indexOf(t) !== -1; });
      }) : [];
      Plotly.restyle(gd, {
        x: [hits.map(function(r){return r.x;})],
        y: [hits.map(function(r){return r.y;})],
        text: [hits.map(function(r){return r.t;})]
      }, [HI]);
      nlab.textContent = q.value.trim() ? (hits.length + " protein(s) match") : "";
    }
    function debounce() { clearTimeout(timer); timer = setTimeout(search, 150); }
    q.addEventListener("input", debounce);
    zoom.addEventListener("click", function() {
      if (!hits.length) return;
      var xs = hits.map(function(r){return r.x;}), ys = hits.map(function(r){return r.y;});
      var px = (Math.max.apply(null,xs)-Math.min.apply(null,xs))*0.15 || 1,
          py = (Math.max.apply(null,ys)-Math.min.apply(null,ys))*0.15 || 1;
      Plotly.relayout(gd, {
        "xaxis.range": [Math.min.apply(null,xs)-px, Math.max.apply(null,xs)+px],
        "yaxis.range": [Math.min.apply(null,ys)-py, Math.max.apply(null,ys)+py]
      });
    });
    clr.addEventListener("click", function() {
      q.value = ""; search();
      Plotly.relayout(gd, {"xaxis.autorange": true, "yaxis.autorange": true});
    });
  }
  run();
})();
</script>
"""
    html = template.replace("__PLOT__", plot_html)
    for k, v in tokens.items():
        html = html.replace(k, v)
    return HTML(html)


def save_html(fig, path: str) -> None:
    """Explicit opt-in export of an interactive plot to a standalone HTML file."""
    fig.write_html(path, include_plotlyjs="cdn")
