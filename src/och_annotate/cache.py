"""Local parquet cache of embeddings + metadata.

Purpose: make re-runs and downstream UMAP free of Biohub calls. Keyed by the
proteome id column (e.g. ``transcript_id``) plus a sequence hash, so an edited
sequence is correctly treated as stale.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def sequence_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]


class EmbeddingCache:
    """A tiny upsert-able store of one record per protein."""

    def __init__(self, cache_dir: str | Path, *, id_column: str = "transcript_id"):
        self.dir = Path(cache_dir)
        self.id_column = id_column
        self.path = self.dir / "embeddings.parquet"
        self._records: dict[Any, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            df = pd.read_parquet(self.path)
            for rec in df.to_dict(orient="records"):
                self._records[rec[self.id_column]] = rec

    def __contains__(self, key: Any) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, key: Any) -> dict[str, Any] | None:
        return self._records.get(key)

    def is_fresh(self, key: Any, sequence: str) -> bool:
        """True if we already hold an embedding for this id + exact sequence."""
        rec = self._records.get(key)
        return bool(rec and rec.get("seq_hash") == sequence_hash(sequence)
                    and rec.get("embedding") is not None)

    def upsert(self, key: Any, record: dict[str, Any]) -> None:
        record = {self.id_column: key, **record}
        self._records[key] = record

    def save(self) -> None:
        if not self._records:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(list(self._records.values()))
        df.to_parquet(self.path, index=False)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._records.values()))
