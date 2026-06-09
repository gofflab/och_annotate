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
