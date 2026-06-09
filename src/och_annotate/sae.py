"""Separate, opt-in SAE feature extraction step.

Run after embedding once you have the SAE model id(s) for esmc-6b. Stores the
top-K features per protein (indices + activations) as JSON in Baserow and the
local cache. No-op (with a clear message) until ``sae.models`` is configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tqdm import tqdm

from och_annotate.baserow import BaserowClient
from och_annotate.cache import EmbeddingCache
from och_annotate.config import Config
from och_annotate.esmc import ESMCEmbedder


@dataclass
class SAESummary:
    total_rows: int = 0
    processed: int = 0
    skipped_existing: int = 0
    skipped_empty: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__


class SAEPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.baserow = BaserowClient(
            config.baserow.base_url, config.baserow_token, timeout=config.esmc.request_timeout
        )
        self.cache = EmbeddingCache(config.run.cache_dir, id_column=config.baserow.id_column)
        self.embedder = ESMCEmbedder(config)

    def run(self, *, limit: int | None = None) -> SAESummary:
        cfg = self.config
        summary = SAESummary()
        if not cfg.sae.models:
            print(
                "No SAE models configured (sae.models is empty). Set the SAE id(s) "
                "for your ESMC model in the config to enable this step."
            )
            return summary

        cfg.require_tokens(baserow=True, biohub=True)
        sae_col = cfg.baserow.output_columns["sae"]
        if cfg.run.write_baserow:
            self.baserow.ensure_fields(cfg.baserow.table_id, [sae_col])

        bw = cfg.baserow
        rows = self.baserow.fetch_rows(bw.table_id)
        rows = [
            {"id": r["id"], bw.id_column: r.get(bw.id_column),
             bw.sequence_column: r.get(bw.sequence_column), sae_col: r.get(sae_col)}
            for r in rows
        ]
        summary.total_rows = len(rows)

        pending = []
        for row in rows:
            seq = row.get(bw.sequence_column)
            if not seq or len(seq) > cfg.esmc.max_sequence_length:
                summary.skipped_empty += 1
                continue
            if cfg.run.skip_existing and row.get(sae_col):
                summary.skipped_existing += 1
                continue
            pending.append(row)
        if limit is not None:
            pending = pending[:limit]

        batch: list[dict] = []

        def flush() -> None:
            if cfg.run.use_cache:
                self.cache.save()
            if cfg.run.write_baserow and batch:
                self.baserow.update_rows(bw.table_id, list(batch))
            batch.clear()

        for row in tqdm(pending, desc=f"SAE {cfg.name}", unit="prot"):
            key = row[bw.id_column]
            seq = row[bw.sequence_column]
            try:
                feats = {m: tf.as_dict() for m, tf in self.embedder.sae_one(seq).items()}
            except Exception as err:  # noqa: BLE001
                summary.failed += 1
                summary.errors.append(f"{key}: {err}")
                continue

            if cfg.run.use_cache and key in self.cache:
                rec = self.cache.get(key) or {}
                rec["sae_top_features"] = feats
                self.cache.upsert(key, rec)
            if cfg.run.write_baserow:
                batch.append({"id": row["id"], sae_col: json.dumps(feats)})
            summary.processed += 1
            if len(batch) >= cfg.run.batch_size:
                flush()

        flush()
        return summary
