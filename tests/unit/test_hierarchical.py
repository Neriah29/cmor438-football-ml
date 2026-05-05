"""
Unit Tests — Hierarchical Clustering
=======================================
Tests for football_ml.unsupervised_learning.hierarchical.HierarchicalClustering

Run from repo root:
    pytest tests/unit/test_hierarchical.py -v
"""

import numpy as np
import pytest
from football_ml.unsupervised_learning.hierarchical import HierarchicalClustering


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_blobs():
    """Three tight clusters — all linkage methods should find them."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[0, 0],   scale=0.2, size=(15, 2))
    X1 = rng.normal(loc=[5, 0],   scale=0.2, size=(15, 2))
    X2 = rng.normal(loc=[2.5, 4], scale=0.2, size=(15, 2))
    return np.vstack([X0, X1, X2])


@pytest.fixture
def two_blobs():
    rng = np.random.default_rng(1)
    X0 = rng.normal(loc=[-3, 0], scale=0.3, size=(20, 2))
    X1 = rng.normal(loc=[3,  0], scale=0.3, size=(20, 2))
    return np.vstack([X0, X1])


@pytest.fixture
def tiny():
    """4 points — easy to reason about manually."""
    return np.array([
        [0, 0], [0.5, 0],   # close pair → cluster A
        [5, 5], [5.5, 5],   # close pair → cluster B
    ], dtype=float)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = HierarchicalClustering()
    assert m.n_clusters == 3
    assert m.linkage == 'ward'


def test_custom_hyperparameters():
    m = HierarchicalClustering(n_clusters=5, linkage='average')
    assert m.n_clusters == 5
    assert m.linkage == 'average'


def test_attributes_none_before_fit():
    m = HierarchicalClustering()
    assert m.labels_ is None
    assert m.linkage_matrix_ is None


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(three_blobs):
    m = HierarchicalClustering(n_clusters=3)
    assert m.fit(three_blobs) is m


def test_labels_shape(three_blobs):
    m = HierarchicalClustering(n_clusters=3).fit(three_blobs)
    assert m.labels_.shape == (45,)


def test_labels_range(three_blobs):
    m = HierarchicalClustering(n_clusters=3).fit(three_blobs)
    assert set(m.labels_).issubset({0, 1, 2})


def test_correct_number_of_clusters(three_blobs):
    m = HierarchicalClustering(n_clusters=3).fit(three_blobs)
    assert m.n_clusters_ == 3


def test_linkage_matrix_shape(three_blobs):
    """Linkage matrix should have n-1 rows."""
    m = HierarchicalClustering(n_clusters=3).fit(three_blobs)
    assert m.linkage_matrix_.shape == (44, 4)


def test_linkage_matrix_columns(three_blobs):
    """Each row: [cluster_i, cluster_j, distance, size]."""
    m = HierarchicalClustering(n_clusters=3).fit(three_blobs)
    lm = m.linkage_matrix_
    # Sizes should be >= 2 (merged clusters)
    assert np.all(lm[:, 3] >= 2)
    # Distances should be non-negative
    assert np.all(lm[:, 2] >= 0)


def test_distances_non_decreasing(three_blobs):
    """Merge distances should never decrease — later merges are farther."""
    m = HierarchicalClustering(n_clusters=3).fit(three_blobs)
    dists = m.linkage_matrix_[:, 2]
    assert np.all(np.diff(dists) >= -1e-10)


def test_fit_predict_consistent(three_blobs):
    m = HierarchicalClustering(n_clusters=3)
    labels = m.fit_predict(three_blobs)
    np.testing.assert_array_equal(labels, m.labels_)


# ---------------------------------------------------------------------------
# Clustering quality
# ---------------------------------------------------------------------------

def test_recovers_two_blobs(two_blobs):
    """Clean separation — should get 2 distinct clusters."""
    m = HierarchicalClustering(n_clusters=2, linkage='ward').fit(two_blobs)
    assert m.n_clusters_ == 2
    # Both clusters should have members
    assert np.sum(m.labels_ == 0) > 0
    assert np.sum(m.labels_ == 1) > 0


def test_tiny_correct_grouping(tiny):
    """
    Points [0,0] and [0.5,0] should be in one cluster,
    [5,5] and [5.5,5] in another.
    """
    m = HierarchicalClustering(n_clusters=2, linkage='ward').fit(tiny)
    assert m.labels_[0] == m.labels_[1]   # first pair same cluster
    assert m.labels_[2] == m.labels_[3]   # second pair same cluster
    assert m.labels_[0] != m.labels_[2]   # different clusters


def test_all_linkages_run(three_blobs):
    """All four linkage methods should run without error."""
    for linkage in ['single', 'complete', 'average', 'ward']:
        m = HierarchicalClustering(n_clusters=3, linkage=linkage).fit(three_blobs)
        assert m.n_clusters_ == 3


def test_all_linkages_find_three_blobs(three_blobs):
    """All methods should identify 3 clusters on well-separated data."""
    for linkage in ['single', 'complete', 'average', 'ward']:
        m = HierarchicalClustering(n_clusters=3, linkage=linkage).fit(three_blobs)
        assert len(np.unique(m.labels_)) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_n_clusters_one(three_blobs):
    """n_clusters=1 should assign all to one cluster."""
    m = HierarchicalClustering(n_clusters=1).fit(three_blobs)
    assert np.all(m.labels_ == 0)


def test_n_clusters_equals_n_samples(tiny):
    """n_clusters=n should give each point its own cluster."""
    m = HierarchicalClustering(n_clusters=4).fit(tiny)
    assert len(np.unique(m.labels_)) == 4


def test_invalid_linkage_raises():
    X = np.array([[0, 0], [1, 1], [5, 5]], dtype=float)
    m = HierarchicalClustering(linkage='invalid')
    with pytest.raises(ValueError):
        m.fit(X)


# ---------------------------------------------------------------------------
# Dendrogram data
# ---------------------------------------------------------------------------

def test_get_dendrogram_data_before_fit_raises():
    m = HierarchicalClustering()
    with pytest.raises(RuntimeError):
        m.get_dendrogram_data()


def test_get_dendrogram_data_returns_linkage_matrix(three_blobs):
    m = HierarchicalClustering(n_clusters=3).fit(three_blobs)
    Z = m.get_dendrogram_data()
    np.testing.assert_array_equal(Z, m.linkage_matrix_)


# ---------------------------------------------------------------------------
# Labels are 0-indexed integers
# ---------------------------------------------------------------------------

def test_labels_zero_indexed(three_blobs):
    m = HierarchicalClustering(n_clusters=3).fit(three_blobs)
    assert set(m.labels_) == {0, 1, 2}


def test_labels_dtype(three_blobs):
    m = HierarchicalClustering(n_clusters=3).fit(three_blobs)
    assert m.labels_.dtype in [np.int32, np.int64, np.intp, int]
