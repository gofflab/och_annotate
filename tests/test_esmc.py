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


def _install_token_tracking_esm(monkeypatch, output, *, credit_limited_tokens=()):
    """Like _install_fake_esm but returns a per-token client registry.

    Each client records the sequences it encoded; clients for tokens in
    ``credit_limited_tokens`` return an ESMProteinError whose message trips the
    credit-limit detector.
    """
    captured = _install_fake_esm(monkeypatch, output)
    import esm.sdk as sdk
    from esm.sdk.api import ESMProteinError

    clients: dict[str, object] = {}

    class TrackingClient:
        def __init__(self, token):
            self.token = token
            self.encoded = []

        def encode(self, protein):
            self.encoded.append(protein.sequence)
            return ("tensor", protein.sequence)

        def logits(self, tensor, config):
            if self.token in credit_limited_tokens:
                return ESMProteinError(429, "Daily credit limit reached")
            return output

    def esmc_client(model, url, token, request_timeout=None):
        captured.update(model=model, url=url, token=token)
        clients[token] = TrackingClient(token)
        return clients[token]

    sdk.esmc_client = esmc_client
    return clients


def test_multiple_tokens_parsed_into_pool(monkeypatch):
    monkeypatch.setenv("BIOHUB_API_TOKENS", "tokA, tokB ,tokC")
    monkeypatch.delenv("BIOHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("ESM_API_KEY", raising=False)
    cfg = load_config(CONFIG)
    assert cfg.biohub_token_pool == ["tokA", "tokB", "tokC"]
    assert cfg.biohub_token == "tokA"  # primary = pool head (back-compat)


def test_round_robin_spreads_requests_across_tokens(monkeypatch):
    monkeypatch.setenv("BIOHUB_API_TOKEN", "tok")
    output = FakeLogitsOutput(mean_embedding=FakeTensor([1.0]))
    clients = _install_token_tracking_esm(monkeypatch, output)
    cfg = load_config(CONFIG)
    cfg.biohub_tokens = ["tokA", "tokB"]  # two-token pool
    ESMCEmbedder(cfg).embed_many(["S0", "S1", "S2", "S3"])
    # One client built per token, and the 4 calls split evenly 2/2.
    assert set(clients) == {"tokA", "tokB"}
    assert [len(clients["tokA"].encoded), len(clients["tokB"].encoded)] == [2, 2]


def test_max_workers_autoscales_with_token_count(monkeypatch):
    output = FakeLogitsOutput(mean_embedding=FakeTensor([1.0]))
    _install_token_tracking_esm(monkeypatch, output)
    cfg = load_config(CONFIG)
    cfg.run.max_workers = 10
    cfg.biohub_tokens = ["a", "b", "c"]
    emb = ESMCEmbedder(cfg)
    emb._ensure_client()
    assert emb._effective_max_workers() == 30  # 10/token × 3 tokens

    # Single token -> unchanged (back-compat).
    solo = load_config(CONFIG)
    solo.biohub_token = "solo"
    solo.run.max_workers = 10
    e_solo = ESMCEmbedder(solo)
    e_solo._ensure_client()
    assert e_solo._effective_max_workers() == 10

    # Per-token concurrency is capped at the SDK's 64 before scaling.
    capped = load_config(CONFIG)
    capped.run.max_workers = 100
    capped.biohub_tokens = ["a", "b"]
    e_cap = ESMCEmbedder(capped)
    e_cap._ensure_client()
    assert e_cap._effective_max_workers() == 128  # min(100, 64) × 2


def test_failover_retires_credit_limited_token(monkeypatch):
    monkeypatch.setenv("BIOHUB_API_TOKEN", "tok")
    output = FakeLogitsOutput(mean_embedding=FakeTensor([1.0]))
    # tokA is capped out: its client returns a credit-limit error.
    clients = _install_token_tracking_esm(monkeypatch, output, credit_limited_tokens={"tokA"})
    cfg = load_config(CONFIG)
    cfg.biohub_tokens = ["tokA", "tokB"]
    emb = ESMCEmbedder(cfg)
    emb._ensure_client()
    # First pick is tokA (index 0); a credit-limit there retires it from the pool.
    try:
        emb._call_with_failover(lambda c: c.logits(("tensor", "S"), None))
    except RuntimeError as err:
        assert "credit limit" in str(err).lower()
    assert 0 in emb._exhausted
    # Subsequent picks now skip tokA and only ever return the healthy tokB.
    for _ in range(4):
        idx, client = emb._next_client()
        assert idx == 1 and client.token == "tokB"


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
        assert set(entry["sae"]["sae-x"]["indices"]) == {3, 5}
        assert entry["sae_store"] is None  # store disabled by default (store_full False)


def _store_acts():
    import torch

    acts = torch.zeros(5, 8)  # [L+2]; residues 1..3 carry signal, F=8
    acts[1, 2] = 3.0
    acts[1, 4] = 0.5
    acts[2, 5] = 2.0
    acts[3, 7] = 1.0
    return acts


def test_store_full_captures_all_nonzero_pooled(monkeypatch):
    """store_full keeps the FULL pooled vector; the tunable top_k sizes only Baserow."""
    monkeypatch.setenv("BIOHUB_API_TOKEN", "tok")
    _install_fake_esm(monkeypatch, FakeLogitsOutput(sae_outputs={"sae-x": _store_acts()}))
    cfg = load_config(CONFIG)
    cfg.biohub_token = "tok"
    cfg.sae.models = ["sae-x"]
    cfg.sae.top_k = 2          # Baserow summary depth (tunable)
    cfg.sae.pooling = "max"
    cfg.sae.store_full = True  # local store = full vector
    cfg.run.max_workers = 0
    entry = ESMCEmbedder(cfg).sae_many(["MKTA"])[0]
    # Baserow/cache summary is the top-2 by pooled max (driven by top_k).
    assert set(entry["sae"]["sae-x"]["indices"]) == {2, 5}
    # Store payload has every non-zero pooled feature, regardless of top_k.
    rec = entry["sae_store"]["sae-x"]
    assert set(rec["indices"]) == {2, 4, 5, 7}
    assert len(rec["indices"]) > cfg.sae.top_k


def test_baserow_top_k_is_tunable_independent_of_store(monkeypatch):
    """Raising top_k widens the Baserow summary; the full store is unaffected."""
    monkeypatch.setenv("BIOHUB_API_TOKEN", "tok")
    _install_fake_esm(monkeypatch, FakeLogitsOutput(sae_outputs={"sae-x": _store_acts()}))
    cfg = load_config(CONFIG)
    cfg.biohub_token = "tok"
    cfg.sae.models = ["sae-x"]
    cfg.sae.top_k = 3          # tune the Baserow summary up to 3
    cfg.sae.pooling = "max"
    cfg.sae.store_full = True
    cfg.run.max_workers = 0
    entry = ESMCEmbedder(cfg).sae_many(["MKTA"])[0]
    assert set(entry["sae"]["sae-x"]["indices"]) == {2, 5, 7}  # top-3 -> Baserow
    assert set(entry["sae_store"]["sae-x"]["indices"]) == {2, 4, 5, 7}  # full -> store


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
