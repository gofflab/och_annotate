"""End-to-end embedding pipeline: Baserow -> ESMC -> Baserow + cache.

Designed to be token-frugal and resumable:
  * only embeds proteins missing an embedding (cache or Baserow column)
  * checkpoints to the local cache every batch, so an interrupted run resumes
  * a dry run reports exactly how many Biohub calls a real run would make
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from och_annotate.baserow import BaserowClient
from och_annotate.cache import EmbeddingCache, sequence_hash
from och_annotate.config import Config
from och_annotate.esmc import ESMCEmbedder


@dataclass
class RunSummary:
    total_rows: int = 0
    to_embed: int = 0
    embedded: int = 0
    skipped_existing: int = 0
    skipped_empty: int = 0
    skipped_too_long: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EmbeddingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.baserow = BaserowClient(
            config.baserow.base_url, config.baserow_token, timeout=config.esmc.request_timeout
        )
        self.cache = EmbeddingCache(config.run.cache_dir, id_column=config.baserow.id_column)
        self.embedder = ESMCEmbedder(config)

    # ---- selection ---------------------------------------------------------
    def _needs_embedding(self, row: dict, sequence: str) -> bool:
        if not self.config.run.skip_existing:
            return True
        key = row[self.config.baserow.id_column]
        if self.config.run.use_cache and self.cache.is_fresh(key, sequence):
            return False
        emb_col = self.config.baserow.output_columns["embedding"]
        if row.get(emb_col):  # already populated in Baserow
            return False
        return True

    def _fetch_rows(self) -> list[dict]:
        bw = self.config.baserow
        wanted = set(bw.metadata_columns) | {
            bw.sequence_column,
            bw.id_column,
            bw.output_columns["embedding"],
        }
        rows = self.baserow.fetch_rows(bw.table_id)
        # keep the Baserow integer row id (needed for write-back) + wanted columns
        return [{"id": r["id"], **{k: r.get(k) for k in wanted}} for r in rows]

    # ---- run ---------------------------------------------------------------
    def run(self, *, dry_run: bool = False, limit: int | None = None) -> RunSummary:
        cfg = self.config
        cfg.require_tokens(baserow=True, biohub=not dry_run)
        summary = RunSummary()

        if cfg.run.write_baserow and not dry_run:
            created = self.baserow.ensure_fields(
                cfg.baserow.table_id,
                [cfg.baserow.output_columns[k] for k in ("embedding", "model", "embedded_at")],
            )
            created_names = [n for n, did in created.items() if did]
            if created_names:
                print(f"Created Baserow columns: {', '.join(created_names)}")

        rows = self._fetch_rows()
        summary.total_rows = len(rows)

        # Classify rows up front so a dry run is honest about the workload.
        pending: list[dict] = []
        for row in rows:
            seq = row.get(cfg.baserow.sequence_column)
            if not seq:
                summary.skipped_empty += 1
                continue
            if len(seq) > cfg.esmc.max_sequence_length:
                summary.skipped_too_long += 1
                continue
            if not self._needs_embedding(row, seq):
                summary.skipped_existing += 1
                continue
            pending.append(row)

        if limit is not None:
            pending = pending[:limit]
        summary.to_embed = len(pending)

        if dry_run:
            return summary

        self._embed_rows(pending, summary)
        return summary

    def _embed_rows(self, pending: list[dict], summary: RunSummary) -> None:
        """Embed pending rows concurrently in checkpointed chunks.

        Each chunk is fanned out across the Forge batch executor, then persisted
        to the cache + Baserow before the next chunk starts, so an interrupted
        run resumes at chunk granularity without re-spending Biohub calls.
        """
        cfg = self.config
        out = cfg.baserow.output_columns
        chunk_size = max(cfg.run.batch_size, 1)
        total = len(pending)
        processed = 0

        for start in range(0, total, chunk_size):
            chunk = pending[start : start + chunk_size]
            sequences = [row[cfg.baserow.sequence_column] for row in chunk]
            results = self.embedder.embed_many(sequences)

            stamp = _now_iso()
            batch: list[dict] = []
            for row, result in zip(chunk, results):
                key = row[cfg.baserow.id_column]
                if not isinstance(result, list):  # ESMProteinError / Exception
                    summary.failed += 1
                    summary.errors.append(f"{key}: {result}")
                    continue
                vector = result
                if cfg.run.use_cache:
                    meta = {k: row.get(k) for k in cfg.baserow.metadata_columns}
                    self.cache.upsert(
                        key,
                        {
                            **meta,
                            "seq_hash": sequence_hash(row[cfg.baserow.sequence_column]),
                            "model": cfg.esmc.model,
                            "embedded_at": stamp,
                            "embedding": vector,
                        },
                    )
                if cfg.run.write_baserow:
                    batch.append(
                        {
                            "id": row["id"],
                            out["embedding"]: json.dumps(vector),
                            out["model"]: cfg.esmc.model,
                            out["embedded_at"]: stamp,
                        }
                    )
                summary.embedded += 1

            # Checkpoint this chunk before moving on.
            if cfg.run.use_cache:
                self.cache.save()
            if cfg.run.write_baserow and batch:
                self.baserow.update_rows(cfg.baserow.table_id, batch)

            processed += len(chunk)
            print(
                f"  checkpoint {processed}/{total} "
                f"(embedded={summary.embedded} failed={summary.failed})"
            )
