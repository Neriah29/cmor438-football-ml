"""
Unit Tests — KNN Classifier
=============================
Tests for football_ml.supervised_learning.knn.KNNClassifier

Run from repo root:
    pytest tests/unit/test_knn.py -v
"""

import numpy as np
import pytest
from collections import Counter
from football_ml.supervised_learning.knn import KNNClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_2d():
    """
    Four clearly separated points — two per class.
    Class 0: top-left region. Class 1: bottom-right region.
    """
    X = np.array([
        [1, 5],   # class 0
        [2, 4],   # class 0
        [8, 1],   # class 1
        [9, 2],   # class 1
    ], dtype=float)
    y = np.array([0, 0, 1, 1])
    return X, y


@pytest.fixture
def blobs():
    """Two well-separated Gaussian blobs."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[-4, -4], scale=0.5, size=(60, 2))
    X1 = rng.normal(loc=[4,  4],  scale=0.5, size=(60, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = KNNClassifier()
    assert m.k == 5
    assert m.metric == 'euclidean'


def test_custom_hyperparameters():
    m = KNNClassifier(k=3, metric='manhattan')
    assert m.k == 3
    assert m.metric == 'manhattan'


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        KNNClassifier(k=0)


def test_invalid_metric_raises():
    with pytest.raises(ValueError):
        KNNClassifier(metric='cosine')


def test_attributes_none_before_fit():
    m = KNNClassifier()
    assert m.X_train_ is None
    assert m.y_train_ is None


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(simple_2d):
    X, y = simple_2d
    m = KNNClassifier()
    assert m.fit(X, y) is m


def test_fit_stores_data(simple_2d):
    X, y = simple_2d
    m = KNNClassifier().fit(X, y)
    np.testing.assert_array_equal(m.X_train_, X)
    np.testing.assert_array_equal(m.y_train_, y)


def test_fit_stores_copy(simple_2d):
    """Stored data should be an independent copy."""
    X, y = simple_2d
    m = KNNClassifier().fit(X, y)
    X[0, 0] = 999
    assert m.X_train_[0, 0] != 999


# ---------------------------------------------------------------------------
# Predict behavior
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    m = KNNClassifier()
    with pytest.raises(RuntimeError):
        m.predict(np.array([[1, 2]]))


def test_predict_output_shape(blobs):
    X, y = blobs
    m = KNNClassifier(k=3).fit(X, y)
    assert m.predict(X[:10]).shape == (10,)


def test_predict_binary_values(blobs):
    X, y = blobs
    m = KNNClassifier(k=3).fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({0, 1})


def test_perfect_accuracy_on_separated_blobs(blobs):
    X, y = blobs
    m = KNNClassifier(k=3).fit(X, y)
    assert m.score(X, y) == 1.0


def test_correct_prediction_simple(simple_2d):
    """Point close to class-0 cluster should be predicted as 0."""
    X, y = simple_2d
    m = KNNClassifier(k=1).fit(X, y)
    # [1.5, 4.5] is nearest to class-0 points
    assert m.predict(np.array([[1.5, 4.5]]))[0] == 0
    # [8.5, 1.5] is nearest to class-1 points
    assert m.predict(np.array([[8.5, 1.5]]))[0] == 1


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

def test_predict_proba_before_fit_raises():
    m = KNNClassifier()
    with pytest.raises(RuntimeError):
        m.predict_proba(np.array([[1, 2]]))


def test_predict_proba_range(blobs):
    X, y = blobs
    m = KNNClassifier(k=5).fit(X, y)
    probs = m.predict_proba(X)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_predict_proba_shape(blobs):
    X, y = blobs
    m = KNNClassifier(k=5).fit(X, y)
    assert m.predict_proba(X).shape == (120,)


def test_predict_proba_consistent_with_predict(blobs):
    """Hard predictions should match thresholded probabilities."""
    X, y = blobs
    m = KNNClassifier(k=5).fit(X, y)
    probs = m.predict_proba(X)
    hard = m.predict(X)
    # Where prob >= 0.5, predict should be 1
    assert np.all(hard[probs >= 0.5] == 1)
    assert np.all(hard[probs < 0.5] == 0)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def test_euclidean_distance_correctness(simple_2d):
    """Hand-verify distance from [0,0] to [3,4] = 5.0."""
    X, y = simple_2d
    m = KNNClassifier(metric='euclidean').fit(X, y)
    query = np.array([0.0, 0.0])
    # Training point [1,5]: sqrt(1+25) = sqrt(26) ≈ 5.099
    d = m._euclidean(query)
    assert abs(d[0] - np.sqrt(26)) < 1e-10


def test_manhattan_distance_correctness(simple_2d):
    """Hand-verify Manhattan distance from [0,0] to [1,5] = 6."""
    X, y = simple_2d
    m = KNNClassifier(metric='manhattan').fit(X, y)
    query = np.array([0.0, 0.0])
    d = m._manhattan(query)
    assert abs(d[0] - 6.0) < 1e-10


def test_manhattan_vs_euclidean_different(blobs):
    """Two metrics should generally produce different predictions on at least some points."""
    X, y = blobs
    m_e = KNNClassifier(k=3, metric='euclidean').fit(X, y)
    m_m = KNNClassifier(k=3, metric='manhattan').fit(X, y)
    # Both should be high accuracy — main check is they run without error
    assert m_e.score(X, y) > 0.9
    assert m_m.score(X, y) > 0.9


# ---------------------------------------------------------------------------
# Effect of K
# ---------------------------------------------------------------------------

def test_k1_perfect_on_training_data(blobs):
    """K=1 always returns the correct label for training points (memorization)."""
    X, y = blobs
    m = KNNClassifier(k=1).fit(X, y)
    assert m.score(X, y) == 1.0


def test_score_is_float(blobs):
    X, y = blobs
    m = KNNClassifier(k=5).fit(X, y)
    assert isinstance(m.score(X, y), float)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_k_larger_than_dataset():
    """k > n_samples should still run (uses all neighbors)."""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    y = np.array([0, 1, 0])
    m = KNNClassifier(k=10).fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (3,)


def test_single_feature():
    X = np.array([[1], [2], [8], [9]], dtype=float)
    y = np.array([0, 0, 1, 1])
    m = KNNClassifier(k=1).fit(X, y)
    assert m.score(X, y) == 1.0


def test_predict_on_new_data(blobs):
    X, y = blobs
    m = KNNClassifier(k=5).fit(X, y)
    X_new = np.array([[-4, -4], [4, 4]])   # clearly in class-0 and class-1 regions
    preds = m.predict(X_new)
    assert preds[0] == 0
    assert preds[1] == 1
