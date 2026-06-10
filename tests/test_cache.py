from och_annotate.cache import EmbeddingCache, sequence_hash


def test_upsert_freshness_and_persistence(tmp_path):
    cache = EmbeddingCache(tmp_path, id_column="transcript_id")
    assert "T1" not in cache

    seq = "MKT"
    cache.upsert("T1", {"seq_hash": sequence_hash(seq), "embedding": [1.0, 2.0], "model": "m"})
    assert cache.is_fresh("T1", seq)
    assert not cache.is_fresh("T1", "DIFFERENT")  # changed sequence => stale
    cache.save()

    # reload from disk
    reloaded = EmbeddingCache(tmp_path, id_column="transcript_id")
    assert "T1" in reloaded
    assert reloaded.is_fresh("T1", seq)
    assert list(reloaded.get("T1")["embedding"]) == [1.0, 2.0]


def test_to_frame(tmp_path):
    cache = EmbeddingCache(tmp_path, id_column="transcript_id")
    cache.upsert("T1", {"seq_hash": "a", "embedding": [1.0], "model": "m"})
    cache.upsert("T2", {"seq_hash": "b", "embedding": [2.0], "model": "m"})
    df = cache.to_frame()
    assert set(df["transcript_id"]) == {"T1", "T2"}
