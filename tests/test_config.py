import os

from och_annotate.config import load_config

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "octopus_chierchiae.yaml")


def test_load_config_parses_structure(monkeypatch):
    monkeypatch.setenv("BASEROW_TOKEN", "bw")
    monkeypatch.setenv("BIOHUB_API_TOKEN", "bh")
    cfg = load_config(CONFIG)
    assert cfg.name == "octopus_chierchiae"
    assert cfg.baserow.table_id == 1026
    assert cfg.baserow.sequence_column == "protein_sequence"
    assert cfg.esmc.model == "esmc-6b-2024-12"
    assert cfg.esmc.url == "https://biohub.ai"
    assert cfg.sae.top_k == 64
    assert cfg.baserow_token == "bw"
    assert cfg.biohub_token == "bh"


def test_biohub_token_falls_back_to_esm_api_key(monkeypatch):
    monkeypatch.delenv("BIOHUB_API_TOKEN", raising=False)
    monkeypatch.setenv("ESM_API_KEY", "fromesm")
    cfg = load_config(CONFIG)
    assert cfg.biohub_token == "fromesm"


def test_require_tokens_raises_when_missing(monkeypatch):
    monkeypatch.delenv("BASEROW_TOKEN", raising=False)
    monkeypatch.delenv("BIOHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("ESM_API_KEY", raising=False)
    cfg = load_config(CONFIG)
    try:
        cfg.require_tokens()
        assert False, "expected EnvironmentError"
    except EnvironmentError as err:
        assert "BASEROW_TOKEN" in str(err)
        assert "BIOHUB_API_TOKEN" in str(err)
