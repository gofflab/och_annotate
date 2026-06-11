"""Tests for SAE-feature coloring helper (no heavy ML deps needed)."""

import json

import pandas as pd

from och_annotate.analysis import sae_feature_activation


def _df():
    return pd.DataFrame(
        {
            "transcript_id": ["T1", "T2", "T3"],
            "sae_top_features": [
                # dict form (from the parquet cache)
                {"sae-x": {"indices": [10, 20, 30], "activations": [0.9, 0.5, 0.1]}},
                # JSON-string form (from Baserow)
                json.dumps({"sae-x": {"indices": [20, 40], "activations": [0.7, 0.2]}}),
                None,  # no SAE for this protein
            ],
        }
    )


def test_feature_present_returns_activation():
    s = sae_feature_activation(_df(), 20, sae_model="sae-x")
    assert list(s) == [0.5, 0.7, 0.0]  # T1 top-K, T2 top-K, T3 missing


def test_feature_absent_returns_zero():
    s = sae_feature_activation(_df(), 30, sae_model="sae-x")
    assert list(s) == [0.1, 0.0, 0.0]  # only T1 has feature 30 in its top-K


def test_default_model_uses_first_stored():
    # no sae_model given -> read whichever model is stored
    s = sae_feature_activation(_df(), 10)
    assert list(s) == [0.9, 0.0, 0.0]


def test_sae_feature_matrix_shapes_and_values():
    from och_annotate.analysis import sae_feature_matrix

    matrix, model = sae_feature_matrix(_df(), n_features=50)
    assert model == "sae-x"
    assert matrix.shape == (3, 50)
    dense = matrix.toarray()
    assert dense[0, 10] == 0.9 and dense[0, 30] == 0.1  # T1 top-K
    assert dense[1, 20] == 0.7 and dense[1, 40] == 0.2  # T2 (JSON-string form)
    assert dense[2].sum() == 0.0                        # T3 had no SAE


def test_sae_feature_matrix_infers_width():
    from och_annotate.analysis import sae_feature_matrix

    matrix, _ = sae_feature_matrix(_df())  # width inferred from max index (40) -> 41
    assert matrix.shape == (3, 41)


def test_load_sae_descriptions_missing_returns_empty(tmp_path):
    from och_annotate.analysis import load_sae_descriptions

    assert load_sae_descriptions(tmp_path / "nope.csv") == {}


def test_load_sae_descriptions_csv_and_json(tmp_path):
    from och_annotate.analysis import load_sae_descriptions

    csv = tmp_path / "d.csv"
    csv.write_text("feature,description\n10,zinc finger\n20,signal peptide\n")
    assert load_sae_descriptions(csv) == {"10": "zinc finger", "20": "signal peptide"}

    js = tmp_path / "d.json"
    js.write_text(json.dumps({"10": "zinc finger", "20": "signal peptide"}))
    assert load_sae_descriptions(js) == {"10": "zinc finger", "20": "signal peptide"}


def test_annotate_enrichment_adds_descriptions():
    from och_annotate.analysis import annotate_enrichment

    enrich = pd.DataFrame({"leiden": ["0", "0"], "sae_feature": ["10", "99"]})
    out = annotate_enrichment(enrich, {"10": "zinc finger"})
    assert list(out["description"]) == ["zinc finger", ""]


# --- SaeFeatureStore (full-activation store) --------------------------------

def _store_df():
    import json as _json
    return pd.DataFrame({
        "transcript_id": ["T1", "T2"],
        "sae_top_features": [
            {"m": {"indices": [1, 3, 5], "activations": [0.9, 0.5, 0.1]}},
            _json.dumps({"m": {"indices": [2, 7], "activations": [0.7, 0.2]}}),
        ],
    })


def test_sae_width_from_model():
    from och_annotate.analysis import sae_width_from_model
    assert sae_width_from_model("esmc-6b-2024-12-sae-layer60-k64-codebook16384") == 16384
    assert sae_width_from_model("no-codebook-here") is None
    assert sae_width_from_model(None) is None


def test_store_upsert_frame_roundtrip_and_idempotent(tmp_path):
    from och_annotate.analysis import update_sae_feature_store, SaeFeatureStore
    p = tmp_path / "sae.npz"
    store, stats = update_sae_feature_store(_store_df(), p, n_features=10)
    assert stats == {"added": 2, "updated": 0, "total": 2}
    m = store.to_csr()
    assert m.shape == (2, 10) and str(m.dtype) == "float32" and m.nnz == 5
    assert m[0, 1] == pytest_approx(0.9) and m[1, 7] == pytest_approx(0.2)
    # reload from disk -> identical
    m2 = SaeFeatureStore(p).to_csr()
    assert (m != m2).nnz == 0
    # re-upsert same frame -> nothing added, matrix unchanged
    s3, st3 = update_sae_feature_store(_store_df(), p, n_features=10)
    assert st3["added"] == 0 and st3["updated"] == 2 and (s3.to_csr() != m).nnz == 0


def test_store_upsert_row_incremental(tmp_path):
    from och_annotate.analysis import SaeFeatureStore
    p = tmp_path / "sae.npz"
    s = SaeFeatureStore(p)
    assert s.upsert_row("A", [0, 4], [1.0, 2.0], sae_model="m", n_features=8) is True
    assert s.upsert_row("A", [4], [9.0], n_features=8) is False  # replace existing
    s.upsert_row("B", [7], [3.0], n_features=8)
    s.save()
    s2 = SaeFeatureStore(p)
    assert s2.ids == ["A", "B"] and s2.n_features == 8
    m = s2.to_csr()
    assert m[0, 4] == pytest_approx(9.0) and m[0, 0] == 0.0 and m[1, 7] == pytest_approx(3.0)


def pytest_approx(x, tol=1e-6):
    class _A:
        def __eq__(self, o): return abs(float(o) - x) <= tol
    return _A()
