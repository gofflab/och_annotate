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
