"""Tests for the residue-region plumbing (extraction, schema, readback)."""

import json

import numpy as np

from och_annotate.analysis import feature_residue_region
from och_annotate.esmc import TopFeatures, _residue_regions


def test_residue_regions_span_and_peak():
    # rows = residues, cols = features
    acts = np.array([
        [0.0, 0.0, 3.0],
        [1.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
    ])
    regions = _residue_regions(acts, [0, 1, 2], threshold_frac=0.5)
    assert regions[0] == [2, 3, 2]     # peak at 2 (=5); span where >= 2.5 is residues 2..3
    assert regions[1] == [0, 0, 0]     # all-zero feature -> degenerate span at 0
    assert regions[2] == [0, 5, 0]     # two peaks (=3) at 0 and 5; span spans both


def test_top_features_as_dict_includes_regions_only_when_present():
    assert TopFeatures([1, 2], [0.5, 0.4]).as_dict() == {
        "indices": [1, 2], "activations": [0.5, 0.4]
    }
    with_regions = TopFeatures([1], [0.5], regions=[[0, 3, 1]]).as_dict()
    assert with_regions["regions"] == [[0, 3, 1]]


def test_candidate_feature_report_with_and_without_regions():
    from och_annotate.analysis import candidate_feature_report

    cell = {"m": {"indices": [10, 11], "activations": [1.5, 1.2],
                  "regions": [[2, 8, 5], [0, 0, 0]]}}
    rep = candidate_feature_report(cell, labels={10: "Foo", 11: "Bar"}, n=2)
    assert list(rep["sae_feature"]) == [10, 11]
    assert rep.loc[0, "residues"] == "2-8 (peak 5)"
    assert rep.loc[0, "label"] == "Foo"
    # cache without regions -> residues None (pending per-residue run)
    rep2 = candidate_feature_report({"m": {"indices": [10], "activations": [1.5]}}, n=1)
    assert rep2.loc[0, "residues"] is None


def test_feature_residue_region_readback_and_graceful_absence():
    cell = {"m": {"indices": [10, 11, 12], "activations": [3, 2, 1],
                  "regions": [[2, 3, 2], [0, 0, 0], [0, 5, 0]]}}
    assert feature_residue_region(cell, 11) == [0, 0, 0]
    assert feature_residue_region(json.dumps(cell), 12) == [0, 5, 0]
    assert feature_residue_region(cell, 99) is None          # feature not in top-K
    # cache without regions (pre per-residue run) -> None, no error
    assert feature_residue_region({"m": {"indices": [10], "activations": [3]}}, 10) is None
    assert feature_residue_region(None, 10) is None
