import os

from och_annotate.config import load_config
from och_annotate.pipeline import EmbeddingPipeline

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "octopus_chierchiae.yaml")


class FakeBaserow:
    def __init__(self, rows):
        self._rows = rows
        self.updated = []
        self.ensured = []

    def ensure_fields(self, table_id, names, **kw):
        self.ensured.extend(names)
        return {n: True for n in names}

    def fetch_rows(self, table_id, **kw):
        return self._rows

    def update_rows(self, table_id, items):
        self.updated.extend(items)


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_one(self, sequence):
        self.calls += 1
        return [float(len(sequence)), 0.0, 1.0]

    def embed_many(self, sequences):
        return [self.embed_one(s) for s in sequences]

    def embed_and_sae_many(self, sequences):
        return [
            {"embedding": self.embed_one(s),
             "sae": {"sae-model": {"indices": [1, 2], "activations": [0.9, 0.8]}}}
            for s in sequences
        ]


def _make_pipeline(tmp_path, rows, *, sae_models=None):
    cfg = load_config(CONFIG)
    cfg.baserow_token = "x"
    cfg.biohub_token = "y"
    cfg.run.cache_dir = str(tmp_path)
    cfg.sae.models = sae_models or []  # default: embedding-only
    pipe = EmbeddingPipeline(cfg)
    pipe.baserow = FakeBaserow(rows)
    pipe.embedder = FakeEmbedder()
    return pipe, cfg


def test_classification_and_dry_run(tmp_path):
    rows = [
        {"id": 1, "transcript_id": "T1", "protein_sequence": "MKTAYIA", "esmc_embedding": None,
         "gene_id": "G1", "chromosome": "1", "start": 1, "end": 2, "strand": "+",
         "Ochierchiae_name": "n1"},
        {"id": 2, "transcript_id": "T2", "protein_sequence": "", "esmc_embedding": None},  # empty
        {"id": 3, "transcript_id": "T3", "protein_sequence": "M" * 9000, "esmc_embedding": None},  # long
        {"id": 4, "transcript_id": "T4", "protein_sequence": "MK", "esmc_embedding": "[0.1]"},  # done
    ]
    pipe, _ = _make_pipeline(tmp_path, rows)
    summary = pipe.run(dry_run=True)
    assert summary.total_rows == 4
    assert summary.to_embed == 1
    assert summary.skipped_empty == 1
    assert summary.skipped_too_long == 1
    assert summary.skipped_existing == 1
    assert pipe.embedder.calls == 0  # dry run never calls the API


def test_embed_writes_cache_and_baserow(tmp_path):
    rows = [
        {"id": 1, "transcript_id": "T1", "protein_sequence": "MKTAYIA", "esmc_embedding": None,
         "gene_id": "G1", "chromosome": "1", "start": 1, "end": 2, "strand": "+",
         "Ochierchiae_name": "n1"},
    ]
    pipe, cfg = _make_pipeline(tmp_path, rows)
    summary = pipe.run()
    assert summary.embedded == 1
    assert pipe.embedder.calls == 1
    # wrote back to Baserow with the configured columns
    assert len(pipe.baserow.updated) == 1
    item = pipe.baserow.updated[0]
    assert item["id"] == 1
    assert cfg.baserow.output_columns["embedding"] in item
    # cached + fresh, so a second run is a no-op (token-frugal)
    pipe2, _ = _make_pipeline(tmp_path, rows)
    summary2 = pipe2.run()
    assert summary2.embedded == 0
    assert summary2.skipped_existing == 1
    assert pipe2.embedder.calls == 0


def test_embed_with_sae_writes_both(tmp_path):
    rows = [
        {"id": 1, "transcript_id": "T1", "protein_sequence": "MKTAYIA", "esmc_embedding": None,
         "sae_top_features": None, "gene_id": "G1", "chromosome": "1", "start": 1, "end": 2,
         "strand": "+", "Ochierchiae_name": "n1"},
    ]
    pipe, cfg = _make_pipeline(tmp_path, rows, sae_models=["sae-model"])
    summary = pipe.run()
    assert summary.embedded == 1
    item = pipe.baserow.updated[0]
    assert cfg.baserow.output_columns["embedding"] in item
    assert cfg.baserow.output_columns["sae"] in item  # SAE written in the same pass
    # SAE column is ensured alongside the embedding columns
    assert cfg.baserow.output_columns["sae"] in pipe.baserow.ensured
    # second run is a no-op: both embedding and SAE present
    pipe2, _ = _make_pipeline(tmp_path, rows, sae_models=["sae-model"])
    summary2 = pipe2.run()
    assert summary2.embedded == 0
    assert summary2.skipped_existing == 1


def test_embedded_without_sae_is_reprocessed_when_sae_enabled(tmp_path):
    # A row already embedded (no SAE) must be reprocessed once SAE is turned on.
    rows = [
        {"id": 1, "transcript_id": "T1", "protein_sequence": "MKTAYIA",
         "esmc_embedding": "[0.1]", "sae_top_features": None},
    ]
    pipe, _ = _make_pipeline(tmp_path, rows, sae_models=["sae-model"])
    summary = pipe.run(dry_run=True)
    assert summary.to_embed == 1
    assert summary.skipped_existing == 0
