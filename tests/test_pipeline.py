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


def _make_pipeline(tmp_path, rows):
    cfg = load_config(CONFIG)
    cfg.baserow_token = "x"
    cfg.biohub_token = "y"
    cfg.run.cache_dir = str(tmp_path)
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
