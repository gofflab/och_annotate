"""Tests for the per-cluster SAE-feature salience/occurrence profile."""

import numpy as np
import pandas as pd
from scipy import sparse

from och_annotate.analysis import cluster_feature_profile


def _toy_adata():
    import anndata as ad

    # 3 proteins x 3 features (normalized activations)
    x = sparse.csr_matrix(np.array([
        [1.0, 0.0, 0.5],
        [0.8, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ]))
    a = ad.AnnData(X=x)
    a.var_names = ["10", "11", "12"]
    a.obs["leiden"] = pd.Categorical(["0", "0", "1"])
    return a


def test_cluster_feature_profile_salience_and_occurrence():
    prof = cluster_feature_profile(_toy_adata(), n=3)
    c0 = prof[prof["cluster"] == "0"].set_index("sae_feature")
    # feature 10: members 1.0 & 0.8 -> mean 0.9, active in both (occurrence 1.0)
    assert c0.loc[10, "mean_activation"] == 0.9
    assert c0.loc[10, "occurrence"] == 1.0
    # feature 12: 0.5 & 0.0 -> mean 0.25, active in one of two (occurrence 0.5)
    assert c0.loc[12, "mean_activation"] == 0.25
    assert c0.loc[12, "occurrence"] == 0.5
    # salience ordering: feature 10 ranked above feature 12
    assert c0.loc[10, "rank"] < c0.loc[12, "rank"]
    assert (prof["n_members"][prof["cluster"] == "0"] == 2).all()


def test_cluster_feature_profile_per_cluster_top_n():
    prof = cluster_feature_profile(_toy_adata(), n=1)
    # one row (top feature) per cluster
    assert set(prof["cluster"]) == {"0", "1"}
    assert (prof["rank"] == 1).all()
    assert prof[prof["cluster"] == "1"].iloc[0]["sae_feature"] == 11
