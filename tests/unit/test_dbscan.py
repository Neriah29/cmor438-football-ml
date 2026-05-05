"""
Unit Tests — DBSCAN
=====================
Tests for football_ml.unsupervised_learning.dbscan.DBSCAN

Run from repo root:
    pytest tests/unit/test_dbscan.py -v
"""

import numpy as np
import pytest
from football_ml.unsupervised_learning.dbscan import DBSCAN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blobs():
    """Two tight clusters — DBSCAN should find both cleanly."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[-3, 0], scale=0.3, size=(30, 2))
    X1 = rng.normal(loc=[3,  0], scale=0.3, size=(30, 2))
    return np.vstack([X0, X1])


@pytest.fixture
def blobs_with_noise():
    """Two clusters plus scattered noise points."""
    rng = np.random.default_rng(1)
    X0    = rng.normal(loc=[-3, 0], scale=0.3, size=(30, 2))
    X1    = rng.normal(loc=[3,  0], scale=0.3, size=(30, 2))
    noise = rng.uniform(-6, 6, size=(10, 2))
    return np.vstack([X0, X1, noise])


@pytest.fixture
def ring_data():
    """Points arranged in two concentric rings — not separable by K-Means."""
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, 2*np.pi, 50)
    inner = np.c_[np.cos(theta) * 1.0 + rng.normal(0, 0.1, 50),
                  np.sin(theta) * 1.0 + rng.normal(0, 0.1, 50)]
    outer = np.c_[np.cos(theta) * 3.0 + rng.normal(0, 0.1, 50),
                  np.sin(theta) * 3.0 + rng.normal(0, 0.1, 50)]
    return np.vstack([inner, outer])


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = DBSCAN()
    assert m.eps == 0.5
    assert m.min_samples == 5
    assert m.metric == 'euclidean'


def test_custom_hyperparameters():
    m = DBSCAN(eps=1.0, min_samples=3, metric='manhattan')
    assert m.eps == 1.0
    assert m.min_samples == 3
    assert m.metric == 'manhattan'


def test_attributes_none_before_fit():
    m = DBSCAN()
    assert m.labels_ is None
    assert m.core_sample_indices_ is None
    assert m.n_clusters_ is None


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(two_blobs):
    m = DBSCAN(eps=1.0, min_samples=3)
    assert m.fit(two_blobs) is m


def test_labels_shape(two_blobs):
    m = DBSCAN(eps=1.0, min_samples=3).fit(two_blobs)
    assert m.labels_.shape == (60,)


def test_fit_predict_consistent(two_blobs):
    m = DBSCAN(eps=1.0, min_samples=3)
    labels = m.fit_predict(two_blobs)
    np.testing.assert_array_equal(labels, m.labels_)


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------

def test_finds_two_clusters(two_blobs):
    m = DBSCAN(eps=1.0, min_samples=3).fit(two_blobs)
    assert m.n_clusters_ == 2


def test_no_noise_on_clean_blobs(two_blobs):
    """Tight, well-separated blobs with generous eps — no noise expected."""
    m = DBSCAN(eps=1.0, min_samples=3).fit(two_blobs)
    assert m.n_noise_ == 0


def test_detects_noise_points(blobs_with_noise):
    """Scattered noise points should be labeled -1."""
    m = DBSCAN(eps=0.8, min_samples=4).fit(blobs_with_noise)
    assert m.n_noise_ > 0


def test_noise_label_is_minus_one(blobs_with_noise):
    m = DBSCAN(eps=0.8, min_samples=4).fit(blobs_with_noise)
    unique_labels = set(m.labels_)
    assert -1 in unique_labels


def test_cluster_labels_non_negative(two_blobs):
    """Cluster labels (non-noise) should be >= 0."""
    m = DBSCAN(eps=1.0, min_samples=3).fit(two_blobs)
    cluster_labels = m.labels_[m.labels_ != -1]
    assert np.all(cluster_labels >= 0)


def test_finds_ring_clusters(ring_data):
    """
    DBSCAN should separate two concentric rings.
    K-Means cannot do this — DBSCAN can because it follows density.
    """
    m = DBSCAN(eps=0.5, min_samples=3).fit(ring_data)
    assert m.n_clusters_ >= 2


def test_all_noise_with_tiny_eps():
    """If eps is tiny, no points are neighbors — everything is noise."""
    X = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    m = DBSCAN(eps=0.001, min_samples=2).fit(X)
    assert np.all(m.labels_ == -1)
    assert m.n_clusters_ == 0


def test_one_cluster_with_large_eps():
    """If eps is huge, all points are in one cluster."""
    X = np.array([[0,0],[1,0],[0,1],[10,10]], dtype=float)
    m = DBSCAN(eps=100.0, min_samples=2).fit(X)
    assert m.n_clusters_ == 1
    assert m.n_noise_ == 0


# ---------------------------------------------------------------------------
# Core points
# ---------------------------------------------------------------------------

def test_core_sample_indices_shape(two_blobs):
    m = DBSCAN(eps=1.0, min_samples=3).fit(two_blobs)
    assert m.core_sample_indices_.ndim == 1


def test_core_points_have_enough_neighbors(two_blobs):
    """Every core point should have >= min_samples-1 neighbors within eps."""
    m = DBSCAN(eps=1.0, min_samples=3).fit(two_blobs)
    D = m._pairwise_distances(two_blobs)
    for idx in m.core_sample_indices_:
        n_neighbors = np.sum(
            (D[idx] <= m.eps) & (np.arange(len(two_blobs)) != idx)
        )
        assert n_neighbors >= m.min_samples - 1


# ---------------------------------------------------------------------------
# n_clusters_ and n_noise_ consistency
# ---------------------------------------------------------------------------

def test_n_clusters_consistent_with_labels(blobs_with_noise):
    m = DBSCAN(eps=0.8, min_samples=4).fit(blobs_with_noise)
    unique_non_noise = set(m.labels_[m.labels_ != -1])
    assert len(unique_non_noise) == m.n_clusters_


def test_n_noise_consistent_with_labels(blobs_with_noise):
    m = DBSCAN(eps=0.8, min_samples=4).fit(blobs_with_noise)
    assert int(np.sum(m.labels_ == -1)) == m.n_noise_


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------

def test_silhouette_before_fit_raises():
    m = DBSCAN()
    with pytest.raises(RuntimeError):
        m.silhouette_score(np.array([[1,2],[3,4]]))


def test_silhouette_range(two_blobs):
    m = DBSCAN(eps=1.0, min_samples=3).fit(two_blobs)
    s = m.silhouette_score(two_blobs)
    assert -1.0 <= s <= 1.0


def test_silhouette_high_for_separated_blobs(two_blobs):
    m = DBSCAN(eps=1.0, min_samples=3).fit(two_blobs)
    assert m.silhouette_score(two_blobs) > 0.5


def test_silhouette_zero_when_one_cluster():
    """Only one cluster — silhouette not meaningful, returns 0."""
    X = np.array([[0,0],[0.1,0],[0,0.1],[0.1,0.1]], dtype=float)
    m = DBSCAN(eps=1.0, min_samples=2).fit(X)
    assert m.silhouette_score(X) == 0.0


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def test_manhattan_metric_runs(two_blobs):
    m = DBSCAN(eps=2.0, min_samples=3, metric='manhattan').fit(two_blobs)
    assert m.labels_ is not None


def test_invalid_metric_raises():
    X = np.array([[0,0],[1,1]], dtype=float)
    m = DBSCAN(metric='cosine')
    with pytest.raises(ValueError):
        m.fit(X)
