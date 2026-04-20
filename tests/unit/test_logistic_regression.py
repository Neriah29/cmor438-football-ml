"""
Unit Tests — Logistic Regression
==================================
Tests for football_ml.supervised_learning.logistic_regression.LogisticRegression

Run from repo root:
    pytest tests/unit/test_logistic_regression.py -v
"""

import numpy as np
import pytest
from football_ml.supervised_learning.logistic_regression import LogisticRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linearly_separable():
    """Two well-separated Gaussian blobs — should reach high accuracy."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[-3, -3], scale=0.5, size=(100, 2))
    X1 = rng.normal(loc=[3,  3],  scale=0.5, size=(100, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


@pytest.fixture
def noisy_binary():
    """Overlapping classes with noise — tests robustness."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 4))
    # True boundary: positive when x0 + x1 > 0
    logits = 2 * X[:, 0] + X[:, 1] + rng.normal(scale=0.5, size=400)
    y = (logits > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = LogisticRegression()
    assert m.learning_rate == 0.1
    assert m.n_epochs == 1000
    assert m.threshold == 0.5
    assert m.random_state == 42


def test_weights_none_before_fit():
    m = LogisticRegression()
    assert m.weights_ is None
    assert m.bias_ is None


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(linearly_separable):
    X, y = linearly_separable
    assert LogisticRegression().fit(X, y) is LogisticRegression().fit(X, y).__class__()  or True
    m = LogisticRegression()
    assert m.fit(X, y) is m


def test_weights_shape_after_fit(noisy_binary):
    X, y = noisy_binary
    m = LogisticRegression().fit(X, y)
    assert m.weights_.shape == (4,)


def test_loss_history_length(linearly_separable):
    X, y = linearly_separable
    m = LogisticRegression(n_epochs=300).fit(X, y)
    assert len(m.loss_history_) == 300


def test_loss_decreases(linearly_separable):
    X, y = linearly_separable
    m = LogisticRegression(n_epochs=500).fit(X, y)
    assert m.loss_history_[-1] < m.loss_history_[0]


def test_loss_is_non_negative(noisy_binary):
    X, y = noisy_binary
    m = LogisticRegression().fit(X, y)
    assert all(l >= 0 for l in m.loss_history_)


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

def test_predict_proba_before_fit_raises():
    m = LogisticRegression()
    with pytest.raises(RuntimeError):
        m.predict_proba(np.array([[1, 2]]))


def test_predict_proba_range(noisy_binary):
    """All probabilities must be strictly between 0 and 1."""
    X, y = noisy_binary
    m = LogisticRegression().fit(X, y)
    probs = m.predict_proba(X)
    assert np.all(probs > 0) and np.all(probs < 1)


def test_predict_proba_shape(linearly_separable):
    X, y = linearly_separable
    m = LogisticRegression().fit(X, y)
    assert m.predict_proba(X).shape == (200,)


def test_sigmoid_uncertainty_at_zero():
    """
    When z=0, sigmoid should return exactly 0.5 — maximum uncertainty.
    Test this directly on the internal sigmoid.
    """
    m = LogisticRegression()
    result = m._sigmoid(np.array([0.0]))
    assert abs(result[0] - 0.5) < 1e-10


def test_sigmoid_extremes():
    """Very large positive z → prob near 1; very negative → near 0."""
    m = LogisticRegression()
    assert m._sigmoid(np.array([100.0]))[0] > 0.999
    assert m._sigmoid(np.array([-100.0]))[0] < 0.001


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_binary_output(noisy_binary):
    X, y = noisy_binary
    m = LogisticRegression().fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({0, 1})


def test_predict_shape(linearly_separable):
    X, y = linearly_separable
    m = LogisticRegression().fit(X, y)
    assert m.predict(X).shape == (200,)


def test_custom_threshold():
    """
    With threshold=0.0, everything should be predicted as 1
    (since all probabilities are > 0).
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = rng.integers(0, 2, size=50)
    m = LogisticRegression(threshold=0.0).fit(X, y)
    assert np.all(m.predict(X) == 1)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def test_high_accuracy_on_separable_data(linearly_separable):
    X, y = linearly_separable
    m = LogisticRegression(learning_rate=0.5, n_epochs=500).fit(X, y)
    assert m.score(X, y) > 0.95


def test_better_than_random_on_noisy_data(noisy_binary):
    X, y = noisy_binary
    m = LogisticRegression(n_epochs=500).fit(X, y)
    assert m.score(X, y) > 0.6   # random baseline is 0.5


def test_score_is_float(linearly_separable):
    X, y = linearly_separable
    m = LogisticRegression().fit(X, y)
    assert isinstance(m.score(X, y), float)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def test_log_loss_non_negative(noisy_binary):
    X, y = noisy_binary
    m = LogisticRegression().fit(X, y)
    assert m.log_loss(X, y) >= 0


def test_log_loss_lower_than_untrained(linearly_separable):
    """Trained model should have lower loss than a model with random weights."""
    X, y = linearly_separable
    untrained = LogisticRegression()
    untrained.weights_ = np.zeros(X.shape[1])
    untrained.bias_ = 0.0
    trained = LogisticRegression(n_epochs=500).fit(X, y)
    assert trained.log_loss(X, y) < untrained.log_loss(X, y)


# ---------------------------------------------------------------------------
# Reproducibility & edge cases
# ---------------------------------------------------------------------------

def test_reproducibility(noisy_binary):
    X, y = noisy_binary
    m1 = LogisticRegression(random_state=5).fit(X, y)
    m2 = LogisticRegression(random_state=5).fit(X, y)
    np.testing.assert_array_almost_equal(m1.weights_, m2.weights_)


def test_single_feature():
    X = np.array([[1], [2], [8], [9]], dtype=float)
    y = np.array([0, 0, 1, 1])
    m = LogisticRegression(learning_rate=0.5, n_epochs=1000).fit(X, y)
    assert m.score(X, y) == 1.0


def test_predict_on_new_data(linearly_separable):
    X, y = linearly_separable
    m = LogisticRegression().fit(X, y)
    X_new = np.array([[5, 5], [-5, -5], [0, 0]])
    probs = m.predict_proba(X_new)
    # [5,5] should have high prob, [-5,-5] low prob
    assert probs[0] > probs[2] > probs[1]
