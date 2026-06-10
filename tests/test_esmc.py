"""Test the ESMC wrapper with a fake `esm` SDK injected, so no torch needed."""

import os
import sys
import types

from och_annotate.config import load_config
from och_annotate.esmc import ESMCEmbedder, _sanitize_sequence

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "octopus_chierchiae.yaml")


class FakeLogitsOutput:
    def __init__(self, mean_embedding=None, embeddings=None, sae_outputs=None):
        self.mean_embedding = mean_embedding
        self.embeddings = embeddings
        self.sae_outputs = sae_outputs
        self.logits = None


class FakeClient:
    def __init__(self, output):
        self.output = output
        self.encoded = []

    def encode(self, protein):
        self.encoded.append(protein.sequence)
        return ("tensor", protein.sequence)

    def logits(self, tensor, config):
        return self.output


def _install_fake_esm(monkeypatch, output):
    sdk = types.ModuleType("esm.sdk")
    api = types.ModuleType("esm.sdk.api")

    class ESMProtein:
        def __init__(self, sequence=None):
            self.sequence = sequence

    class LogitsConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class SAEConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class ESMProteinError(Exception):
        def __init__(self, error_code=500, message=""):
            self.error_code = error_code
            super().__init__(message)

    captured = {}

    def esmc_client(model, url, token, request_timeout=None):
        captured.update(model=model, url=url, token=token)
        return FakeClient(output)

    class _FakeExecutor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute_batch(self, fn, **kwargs):
            # fan a single list kwarg out to per-item calls (synchronously)
            (name, values), = kwargs.items()
            results = []
            for v in values:
                try:
                    results.append(fn(**{name: v}))
                except Exception as err:  # match executor: capture, don't raise
                    results.append(err)
            return results

    def batch_executor(max_attempts=10, show_progress=True):
        return _FakeExecutor()

    sdk.esmc_client = esmc_client
    sdk.batch_executor = batch_executor
    api.ESMProtein = ESMProtein
    api.ESMProteinError = ESMProteinError
    api.LogitsConfig = LogitsConfig
    api.SAEConfig = SAEConfig

    esm = types.ModuleType("esm")
    monkeypatch.setitem(sys.modules, "esm", esm)
    monkeypatch.setitem(sys.modules, "esm.sdk", sdk)
    monkeypatch.setitem(sys.modules, "esm.sdk.api", api)
    return captured


def test_embed_one_uses_mean_embedding(monkeypatch):
    monkeypatch.setenv("BIOHUB_API_TOKEN", "tok")
    output = FakeLogitsOutput(mean_embedding=[0.5, 1.5, 2.5])
    captured = _install_fake_esm(monkeypatch, output)
    cfg = load_config(CONFIG)
    cfg.biohub_token = "tok"

    embedder = ESMCEmbedder(cfg)
    vec = embedder.embed_one("MKTAYIA")
    assert vec == [0.5, 1.5, 2.5]
    # client built with the configured 6B model + Biohub url
    assert captured["model"] == "esmc-6b-2024-12"
    assert captured["url"] == "https://biohub.ai"


class FakeTensor:
    """Minimal stand-in for a torch tensor with leading singleton dims."""

    def __init__(self, flat):
        self._flat = flat

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def reshape(self, _shape):
        return self  # already track the flat view

    def tolist(self):
        return self._flat


def test_sanitize_sequence_strips_stops_and_whitespace():
    # '*' (stop codon), whitespace and case are all normalised away.
    assert _sanitize_sequence("mkt ay\nIA*") == "MKTAYIA"
    assert _sanitize_sequence("AC*DE*") == "ACDE"


def test_embed_one_flattens_nested_mean_embedding(monkeypatch):
    monkeypatch.setenv("BIOHUB_API_TOKEN", "tok")
    # server returns a tensor shaped like [1, 1, d]; embed_one must flatten it.
    output = FakeLogitsOutput(mean_embedding=FakeTensor([0.5, 1.5, 2.5]))
    _install_fake_esm(monkeypatch, output)
    cfg = load_config(CONFIG)
    cfg.biohub_token = "tok"
    vec = ESMCEmbedder(cfg).embed_one("MKTAYIA")
    assert vec == [0.5, 1.5, 2.5]


def test_embed_many_returns_one_result_per_sequence(monkeypatch):
    monkeypatch.setenv("BIOHUB_API_TOKEN", "tok")
    output = FakeLogitsOutput(mean_embedding=FakeTensor([0.5, 1.5, 2.5]))
    _install_fake_esm(monkeypatch, output)
    cfg = load_config(CONFIG)
    cfg.biohub_token = "tok"
    vecs = ESMCEmbedder(cfg).embed_many(["MKTAYIA", "MK*"])
    assert vecs == [[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]]


def test_sae_many_returns_top_features_per_sequence(monkeypatch):
    import torch

    monkeypatch.setenv("BIOHUB_API_TOKEN", "tok")
    acts = torch.zeros(4, 6)  # [L+2, F]; residues 1..2 carry signal
    acts[1, 3] = 2.0
    acts[2, 5] = 1.0
    output = FakeLogitsOutput(sae_outputs={"sae-x": acts})
    _install_fake_esm(monkeypatch, output)
    cfg = load_config(CONFIG)
    cfg.biohub_token = "tok"
    cfg.sae.models = ["sae-x"]
    cfg.sae.top_k = 2
    cfg.run.max_workers = 0  # use the fake batch_executor (no real ForgeBatchExecutor)
    res = ESMCEmbedder(cfg).sae_many(["MKT", "AAA"])
    assert len(res) == 2
    for entry in res:
        assert set(entry["sae-x"]["indices"]) == {3, 5}


def test_sae_one_empty_when_no_models(monkeypatch):
    monkeypatch.setenv("BIOHUB_API_TOKEN", "tok")
    _install_fake_esm(monkeypatch, FakeLogitsOutput())
    cfg = load_config(CONFIG)
    cfg.biohub_token = "tok"
    cfg.sae.models = []  # default: SAE disabled
    embedder = ESMCEmbedder(cfg)
    assert embedder.sae_one("MKTAYIA") == {}


def _embedder_for_topk(pooling="max", residue_regions=False, top_k=8):
    cfg = load_config(CONFIG)
    cfg.sae.pooling = pooling
    cfg.sae.residue_regions = residue_regions
    cfg.sae.top_k = top_k
    return ESMCEmbedder(cfg)


def test_sparse_pooling_matches_dense_topk():
    """The memory-light sparse path must give identical top-K to densifying."""
    import torch

    torch.manual_seed(0)
    L, F, k = 50, 4096, 64  # [1, L, F] with BOS/EOS, k active features per residue
    rows, cols, vals = [], [], []
    for r in range(L):
        feats = torch.randperm(F)[:k]
        for f in feats.tolist():
            rows.append(r); cols.append(f); vals.append(float(torch.rand(1)) + 0.01)
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0] * len(rows), rows, cols]), torch.tensor(vals), size=(1, L, F)
    ).coalesce()
    dense = sparse.to_dense()

    for pooling in ("max", "mean"):
        emb = _embedder_for_topk(pooling=pooling)
        sparse_res = emb._top_features(sparse)                     # sparse fast path
        dense_res = emb._top_features(dense)                       # dense reference path
        assert sparse_res.indices == dense_res.indices
        for a, b in zip(sparse_res.activations, dense_res.activations):
            assert abs(a - b) < 1e-5
        assert sparse_res.regions is None


def test_residue_regions_path_still_densifies_when_enabled():
    """With residue_regions on, the dense path runs and emits spans aligned to indices."""
    import torch

    L, F = 10, 32
    dense = torch.zeros(1, L, F)
    dense[0, 3, 7] = 5.0   # feature 7 peaks at residue index 2 (after dropping BOS)
    dense[0, 4, 7] = 4.0
    emb = _embedder_for_topk(pooling="max", residue_regions=True, top_k=1)
    res = emb._top_features(dense.to_sparse().coalesce())
    assert res.indices == [7]
    assert res.regions is not None and len(res.regions) == 1
