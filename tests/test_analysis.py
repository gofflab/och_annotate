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
