"""Tests for the ESM Atlas feature-description fetcher (no network)."""

import och_annotate.atlas as atlas


def _fake_info(idx, **kw):
    return {
        "feature_index": idx,
        "label": f"label-{idx}",
        "summary": f"summary-{idx}",
        "description": f"desc-{idx}",
        "category": "cat",
    }


def test_fetch_feature_descriptions_dedups_sorts_and_maps(monkeypatch, tmp_path):
    monkeypatch.setattr(atlas, "feature_info", _fake_info)
    df = atlas.fetch_feature_descriptions([3, 1, 1, 2], cache_path=tmp_path / "c.parquet")
    assert list(df["feature"]) == [1, 2, 3]
    assert dict(zip(df["feature"], df["label"])) == {1: "label-1", 2: "label-2", 3: "label-3"}
    assert set(df.columns) == {"feature", "label", "summary", "description", "category"}


def test_fetch_feature_descriptions_uses_cache(monkeypatch, tmp_path):
    calls = {"n": 0}

    def counting(idx, **kw):
        calls["n"] += 1
        return _fake_info(idx)

    monkeypatch.setattr(atlas, "feature_info", counting)
    cache = tmp_path / "c.parquet"
    atlas.fetch_feature_descriptions([1, 2, 3], cache_path=cache)
    assert calls["n"] == 3
    # second call is fully cached -> no further network lookups
    atlas.fetch_feature_descriptions([1, 2, 3], cache_path=cache)
    assert calls["n"] == 3


def test_fetch_feature_descriptions_records_failures(monkeypatch, tmp_path):
    def boom(idx, **kw):
        raise RuntimeError("503")

    monkeypatch.setattr(atlas, "feature_info", boom)
    df = atlas.fetch_feature_descriptions([7], cache_path=tmp_path / "c.parquet")
    assert df.loc[0, "label"] == ""
    assert "<error" in df.loc[0, "description"]


def test_fetch_all_features_bulk_and_cache(monkeypatch, tmp_path):
    import requests

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [
                {"feature_index": 0, "label": "L0", "description": "D0"},
                {"feature_index": 1, "label": "L1", "description": "D1"},
            ]}

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    cache = tmp_path / "dict.parquet"
    df = atlas.fetch_all_features(cache_path=cache)
    assert list(df["feature"]) == [0, 1]
    assert dict(zip(df["feature"], df["label"])) == {0: "L0", 1: "L1"}

    # second call serves from cache without hitting the network
    def _boom(*a, **k):
        raise AssertionError("refetched")

    monkeypatch.setattr(requests, "get", _boom)
    assert len(atlas.fetch_all_features(cache_path=cache)) == 2

