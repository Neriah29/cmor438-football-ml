"""
Unit Tests — K-Means Clustering
=================================
Tests for football_ml.unsupervised_learning.kmeans.KMeans

Run from repo root:
    pytest tests/unit/test_kmeans.py -v
"""

import numpy as np
import pytest
from football_ml.unsupervised_learning.kmeans import KMeans


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blobs():
    """Three well-separated Gaussian clusters."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[0,  0],  scale=0.3, size=(50, 2))
    X1 = rng.normal(loc=[5,  0],  scale=0.3, size=(50, 2))
    X2 = rng.normal(loc=[2.5, 4], scale=0.3, size=(50, 2))
    return np.vstack([X0, X1, X2])


@pytest.fixture
def two_blobs():
    rng = np.random.default_rng(1)
    X0 = rng.normal(loc=[-3, 0], scale=0.5, size=(40, 2))
    X1 = rng.normal(loc=[3,  0], scale=0.5, size=(40, 2))
    return np.vstack([X0, X1])


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = KMeans()
    assert m.k == 3
    assert m.max_iter == 300
    assert m.n_init == 10


def test_custom_hyperparameters():
    m = KMeans(k=5, max_iter=100, n_init=3)
    assert m.k == 5
    assert m.max_iter == 100
    assert m.n_init == 3


def test_attributes_none_before_fit():
    m = KMeans()
    assert m.centroids_ is None
    assert m.labels_ is None
    assert m.inertia_ is None


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(blobs):
    m = KMeans(k=3, n_init=2)
    assert m.fit(blobs) is m


def test_centroids_shape(blobs):
    m = KMeans(k=3, n_init=2).fit(blobs)
    assert m.centroids_.shape == (3, 2)


def test_labels_shape(blobs):
    m = KMeans(k=3, n_init=2).fit(blobs)
    assert m.labels_.shape == (150,)


def test_labels_range(blobs):
    m = KMeans(k=3, n_init=2).fit(blobs)
    assert set(m.labels_).issubset({0, 1, 2})


def test_all_clusters_populated(blobs):
    """Every cluster should have at least one point."""
    m = KMeans(k=3, n_init=3).fit(blobs)
    assert len(np.unique(m.labels_)) == 3


def test_inertia_positive(blobs):
    m = KMeans(k=3, n_init=2).fit(blobs)
    assert m.inertia_ > 0


def test_inertia_history_non_empty(blobs):
    m = KMeans(k=3, n_init=2).fit(blobs)
    assert len(m.inertia_history_) > 0


def test_inertia_history_decreasing(blobs):
    """Inertia should never increase between iterations."""
    m = KMeans(k=3, n_init=1, random_state=0).fit(blobs)
    for i in range(1, len(m.inertia_history_)):
        assert m.inertia_history_[i] <= m.inertia_history_[i-1] + 1e-6


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    m = KMeans()
    with pytest.raises(RuntimeError):
        m.predict(np.array([[1, 2]]))


def test_predict_output_shape(blobs):
    m = KMeans(k=3, n_init=2).fit(blobs)
    assert m.predict(blobs[:10]).shape == (10,)


def test_predict_values_in_range(blobs):
    m = KMeans(k=3, n_init=2).fit(blobs)
    preds = m.predict(blobs)
    assert set(preds).issubset({0, 1, 2})


def test_fit_predict_consistent(blobs):
    """fit_predict should match labels_ from fit."""
    m = KMeans(k=3, n_init=2, random_state=0)
    labels_fp = m.fit_predict(blobs)
    np.testing.assert_array_equal(labels_fp, m.labels_)


# ---------------------------------------------------------------------------
# Clustering quality
# ---------------------------------------------------------------------------

def test_recovers_three_blobs(blobs):
    """
    On well-separated blobs, inertia should be much lower than
    a random partition would give.
    """
    m = KMeans(k=3, n_init=5).fit(blobs)
    # Each blob has std=0.3 and 50 points — expected inertia per blob ≈ 50 * 2 * 0.09 = 9
    # Total expected ≈ 27. Give generous margin.
    assert m.inertia_ < 200


def test_more_clusters_lower_inertia(blobs):
    """More clusters = lower inertia (trivially true — k=n gives inertia=0)."""
    m2 = KMeans(k=2, n_init=3).fit(blobs)
    m3 = KMeans(k=3, n_init=3).fit(blobs)
    assert m3.inertia_ <= m2.inertia_


def test_n_init_doesnt_increase_inertia(blobs):
    """More initializations should never produce worse results."""
    m1 = KMeans(k=3, n_init=1,  random_state=0).fit(blobs)
    m10 = KMeans(k=3, n_init=10, random_state=0).fit(blobs)
    assert m10.inertia_ <= m1.inertia_ + 1e-6


# ---------------------------------------------------------------------------
# Silhouette score
# ---------------------------------------------------------------------------

def test_silhouette_before_fit_raises():
    m = KMeans()
    with pytest.raises(RuntimeError):
        m.silhouette_score(np.array([[1, 2], [3, 4]]))


def test_silhouette_range(two_blobs):
    m = KMeans(k=2, n_init=3).fit(two_blobs)
    score = m.silhouette_score(two_blobs)
    assert -1.0 <= score <= 1.0


def test_silhouette_high_for_separated_blobs(two_blobs):
    """Well-separated clusters should have silhouette > 0.5."""
    m = KMeans(k=2, n_init=5).fit(two_blobs)
    assert m.silhouette_score(two_blobs) > 0.5


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility(blobs):
    m1 = KMeans(k=3, n_init=3, random_state=7).fit(blobs)
    m2 = KMeans(k=3, n_init=3, random_state=7).fit(blobs)
    np.testing.assert_array_almost_equal(m1.centroids_, m2.centroids_)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_k_equals_one(blobs):
    """k=1 should assign everything to one cluster."""
    m = KMeans(k=1, n_init=2).fit(blobs)
    assert np.all(m.labels_ == 0)
    assert m.centroids_.shape == (1, 2)


def test_single_feature():
    X = np.array([[1], [1.1], [5], [5.1], [10], [10.1]])
    m = KMeans(k=3, n_init=3).fit(X)
    assert len(np.unique(m.labels_)) == 3
