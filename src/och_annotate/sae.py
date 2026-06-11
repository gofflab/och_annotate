"""Separate, opt-in SAE feature extraction step.

Run after embedding once you have the SAE model id(s) for esmc-6b. Stores the
top-K features per protein (indices + activations) as JSON in Baserow and the
local cache. No-op (with a clear message) until ``sae.models`` is configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

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

        # Optional full feature store (every non-zero pooled feature) outside
        # Baserow. When on, a row still needs processing if it is missing from the
        # store, even if it already has the top_k summary in Baserow.
        store = store_model = store_width = None
        if cfg.sae.store_full:
            from och_annotate.analysis import open_sae_feature_store, sae_width_from_model

            store = open_sae_feature_store(cfg)
            store_model = cfg.sae.models[0]
            store_width = sae_width_from_model(store_model)
            if len(cfg.sae.models) > 1:
                print(f"  store: writing features for {store_model!r} only "
                      f"(first of {len(cfg.sae.models)} SAE models).")

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
            in_store = store is not None and row.get(bw.id_column) in store
            already_done = bool(row.get(sae_col)) and (store is None or in_store)
            if cfg.run.skip_existing and already_done:
                summary.skipped_existing += 1
                continue
            pending.append(row)
        if limit is not None:
            pending = pending[:limit]

        # Concurrent, chunked + checkpointed (same hardening as the embed step).
        chunk_size = max(cfg.run.batch_size, 1)
        total = len(pending)
        processed = 0
        for start in range(0, total, chunk_size):
            chunk = pending[start : start + chunk_size]
            sequences = [row[bw.sequence_column] for row in chunk]
            results = self.embedder.sae_many(sequences)

            batch: list[dict] = []
            for row, result in zip(chunk, results):
                key = row[bw.id_column]
                if not isinstance(result, dict):  # ESMProteinError / Exception
                    summary.failed += 1
                    summary.errors.append(f"{key}: {result}")
                    continue
                feats = result["sae"]  # top-K summary -> Baserow + cache
                if cfg.run.use_cache and key in self.cache:
                    rec = self.cache.get(key) or {}
                    # JSON string (matches Baserow; dicts break the cache parquet).
                    rec["sae_top_features"] = json.dumps(feats)
                    self.cache.upsert(key, rec)
                if cfg.run.write_baserow:
                    batch.append({"id": row["id"], sae_col: json.dumps(feats)})
                if store is not None:
                    rec = (result.get("sae_store") or {}).get(store_model)
                    if rec:
                        store.upsert_row(key, rec["indices"], rec["activations"],
                                         sae_model=store_model, n_features=store_width)
                summary.processed += 1

            if cfg.run.use_cache:
                self.cache.save()
            if store is not None:
                store.save()
            if cfg.run.write_baserow and batch:
                self.baserow.update_rows(bw.table_id, batch)

            processed += len(chunk)
            print(
                f"  checkpoint {processed}/{total} "
                f"(processed={summary.processed} failed={summary.failed})"
            )
        return summary
