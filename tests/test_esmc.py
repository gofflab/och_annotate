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

    captured = {}

    def esmc_client(model, url, token, request_timeout=None):
        captured.update(model=model, url=url, token=token)
        return FakeClient(output)

    sdk.esmc_client = esmc_client
    api.ESMProtein = ESMProtein
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


def test_sae_one_empty_when_no_models(monkeypatch):
    monkeypatch.setenv("BIOHUB_API_TOKEN", "tok")
    _install_fake_esm(monkeypatch, FakeLogitsOutput())
    cfg = load_config(CONFIG)
    cfg.biohub_token = "tok"
    cfg.sae.models = []  # default: SAE disabled
    embedder = ESMCEmbedder(cfg)
    assert embedder.sae_one("MKTAYIA") == {}
