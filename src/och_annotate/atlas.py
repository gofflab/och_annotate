"""Fetch SAE feature descriptions from the public ESM Atlas API (biohub.ai).

The Atlas annotates each SAE codebook feature with a short ``label``, a
``summary``, a longer ``description`` and a ``category``. These calls hit the
Atlas (``/esm/protein/api/v1alpha1/features/{idx}``), not the ESMC inference
API, so they do **not** consume Biohub embedding credits.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ATLAS_BASE = "https://biohub.ai"
FIELDS = ["label", "summary", "description", "category"]


def fetch_all_features(
    *, base_url: str = ATLAS_BASE, timeout: int = 60, cache_path: str | Path | None = None
) -> pd.DataFrame:
    """Fetch the **complete** SAE feature dictionary in a single Atlas call.

    The list endpoint (``/features``) returns every codebook feature's
    ``feature_index``, ``label`` and ``description`` at once (no ``summary`` /
    ``category`` — those are per-feature only). This is the cheapest way to get
    the full dictionary; result is cached to ``cache_path`` if given.
    """
    import requests

    if cache_path and Path(cache_path).exists():
        p = Path(cache_path)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    resp = requests.get(f"{base_url}/esm/protein/api/v1alpha1/features", timeout=timeout)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    df = pd.DataFrame(
        [{"feature": int(d["feature_index"]), "label": d.get("label", ""),
          "description": d.get("description", "")} for d in data]
    )
    if cache_path and len(df):
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False) if p.suffix == ".parquet" else df.to_csv(p, index=False)
    return df


def feature_info(idx: int, *, base_url: str = ATLAS_BASE, timeout: int = 30, session=None) -> dict:
    """Fetch the raw Atlas metadata for a single SAE feature index."""
    import requests

    getter = session or requests
    url = f"{base_url}/esm/protein/api/v1alpha1/features/{int(idx)}"
    resp = getter.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_feature_descriptions(
    indices,
    *,
    base_url: str = ATLAS_BASE,
    max_workers: int = 16,
    cache_path: str | Path | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Return a DataFrame ``[feature, label, summary, description, category]``.

    Concurrent + incremental: if ``cache_path`` exists, only the indices not
    already cached are fetched, and the merged result is written back. Failed
    lookups are kept with an empty label so they aren't retried forever.
    """
    import requests

    wanted = sorted({int(i) for i in indices})
    columns = ["feature", *FIELDS]
    cached = pd.DataFrame(columns=columns)
    if cache_path and Path(cache_path).exists():
        p = Path(cache_path)
        cached = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        cached["feature"] = cached["feature"].astype(int)

    have = set(cached["feature"]) if len(cached) else set()
    todo = [i for i in wanted if i not in have]

    rows: list[dict] = []
    if todo:
        session = requests.Session()

        def _one(i: int) -> dict:
            try:
                d = feature_info(i, base_url=base_url, timeout=timeout, session=session)
                return {"feature": int(d.get("feature_index", i)),
                        **{f: d.get(f, "") for f in FIELDS}}
            except Exception as err:  # noqa: BLE001 - keep going, record the failure
                return {"feature": int(i), "label": "", "summary": "",
                        "description": f"<error: {err}>", "category": ""}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for fut in as_completed([pool.submit(_one, i) for i in todo]):
                rows.append(fut.result())

    fetched = pd.DataFrame(rows, columns=columns)
    full = (
        pd.concat([cached, fetched], ignore_index=True)
        .drop_duplicates("feature", keep="last")
        .sort_values("feature")
        .reset_index(drop=True)
    )
    if cache_path and len(full):
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(p, index=False) if p.suffix == ".parquet" else full.to_csv(p, index=False)

    return full[full["feature"].isin(wanted)].reset_index(drop=True)
