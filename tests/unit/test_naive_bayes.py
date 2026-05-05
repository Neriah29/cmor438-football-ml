"""
Unit Tests — Gaussian Naïve Bayes
===================================
Tests for football_ml.supervised_learning.naive_bayes.GaussianNaiveBayes

Run from repo root:
    pytest tests/unit/test_naive_bayes.py -v
"""

import numpy as np
import pytest
from football_ml.supervised_learning.naive_bayes import GaussianNaiveBayes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[-3, -3], scale=0.5, size=(80, 2))
    X1 = rng.normal(loc=[3,  3],  scale=0.5, size=(80, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 80 + [1] * 80)
    return X, y


@pytest.fixture
def simple():
    """Tiny clearly separated dataset."""
    X = np.array([
        [1.0, 1.0], [1.5, 1.2], [0.8, 0.9],   # class 0
        [9.0, 9.0], [9.5, 8.8], [8.7, 9.1],   # class 1
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def noisy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    logits = 2 * X[:, 0] + X[:, 1] + rng.normal(scale=0.5, size=300)
    y = (logits > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = GaussianNaiveBayes()
    assert m.var_smoothing == 1e-9


def test_attributes_none_before_fit():
    m = GaussianNaiveBayes()
    assert m.classes_ is None
    assert m.means_ == {}
    assert m.variances_ == {}
    assert m.class_priors_ == {}


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(blobs):
    X, y = blobs
    m = GaussianNaiveBayes()
    assert m.fit(X, y) is m


def test_classes_set_after_fit(blobs):
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    np.testing.assert_array_equal(m.classes_, [0, 1])


def test_means_shape(noisy):
    X, y = noisy
    m = GaussianNaiveBayes().fit(X, y)
    assert m.means_[0].shape == (4,)
    assert m.means_[1].shape == (4,)


def test_variances_shape(noisy):
    X, y = noisy
    m = GaussianNaiveBayes().fit(X, y)
    assert m.variances_[0].shape == (4,)
    assert m.variances_[1].shape == (4,)


def test_variances_positive(blobs):
    """Variances must always be positive (var_smoothing ensures this)."""
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    assert np.all(m.variances_[0] > 0)
    assert np.all(m.variances_[1] > 0)


def test_priors_sum_to_one(blobs):
    """Log priors should exponentiate to values that sum to 1."""
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    total = sum(np.exp(m.class_priors_[c]) for c in m.classes_)
    assert abs(total - 1.0) < 1e-10


def test_means_correct(simple):
    """Means should match numpy's computation on each class subset."""
    X, y = simple
    m = GaussianNaiveBayes().fit(X, y)
    np.testing.assert_array_almost_equal(m.means_[0], X[y==0].mean(axis=0))
    np.testing.assert_array_almost_equal(m.means_[1], X[y==1].mean(axis=0))


# ---------------------------------------------------------------------------
# Predict behavior
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    m = GaussianNaiveBayes()
    with pytest.raises(RuntimeError):
        m.predict(np.array([[1, 2]]))


def test_predict_proba_before_fit_raises():
    m = GaussianNaiveBayes()
    with pytest.raises(RuntimeError):
        m.predict_proba(np.array([[1, 2]]))


def test_predict_output_shape(blobs):
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    assert m.predict(X).shape == (160,)


def test_predict_binary_values(blobs):
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({0, 1})


def test_predict_proba_shape(blobs):
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    assert m.predict_proba(X).shape == (160,)


def test_predict_proba_range(blobs):
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    probs = m.predict_proba(X)
    assert np.all(probs >= 0) and np.all(probs <= 1)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def test_perfect_accuracy_on_separated(simple):
    X, y = simple
    m = GaussianNaiveBayes().fit(X, y)
    assert m.score(X, y) == 1.0


def test_high_accuracy_on_blobs(blobs):
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    assert m.score(X, y) > 0.95


def test_better_than_random_on_noisy(noisy):
    X, y = noisy
    m = GaussianNaiveBayes().fit(X, y)
    assert m.score(X, y) > 0.6


# ---------------------------------------------------------------------------
# Probabilistic correctness
# ---------------------------------------------------------------------------

def test_high_prob_for_class1_far_right(simple):
    """A point deep in class-1 territory should have high P(class=1)."""
    X, y = simple
    m = GaussianNaiveBayes().fit(X, y)
    prob = m.predict_proba(np.array([[9.0, 9.0]]))[0]
    assert prob > 0.9


def test_low_prob_for_class1_far_left(simple):
    """A point deep in class-0 territory should have low P(class=1)."""
    X, y = simple
    m = GaussianNaiveBayes().fit(X, y)
    prob = m.predict_proba(np.array([[1.0, 1.0]]))[0]
    assert prob < 0.1


def test_predict_consistent_with_proba(blobs):
    """Hard predictions should match argmax of probabilities."""
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    probs = m.predict_proba(X)
    preds = m.predict(X)
    np.testing.assert_array_equal(preds, (probs >= 0.5).astype(int))


# ---------------------------------------------------------------------------
# Var smoothing
# ---------------------------------------------------------------------------

def test_var_smoothing_prevents_zero_variance():
    """Constant feature should not cause division by zero."""
    X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0],
                  [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    m = GaussianNaiveBayes().fit(X, y)
    # Feature 0 is constant (all 1.0) — var should be > 0 due to smoothing
    assert m.variances_[0][0] > 0
    assert m.variances_[1][0] > 0


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def test_score_is_float(blobs):
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    assert isinstance(m.score(X, y), float)


def test_score_range(blobs):
    X, y = blobs
    m = GaussianNaiveBayes().fit(X, y)
    s = m.score(X, y)
    assert 0.0 <= s <= 1.0
