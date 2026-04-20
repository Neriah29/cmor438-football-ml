"""
Unit Tests — Linear Regression
================================
Tests for football_ml.supervised_learning.linear_regression.LinearRegression

Run from repo root:
    pytest tests/unit/test_linear_regression.py -v
"""

import numpy as np
import pytest
from football_ml.supervised_learning.linear_regression import LinearRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_linear():
    """
    y = 3x + 2  with no noise.
    A perfect linear model should recover w=3, b=2 exactly (or very close).
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(200, 1))
    y = 3 * X.ravel() + 2
    return X, y


@pytest.fixture
def multi_feature():
    """y = 2x1 - x2 + 0.5x3 + noise."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.1, size=300)
    return X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = LinearRegression()
    assert m.learning_rate == 0.01
    assert m.n_epochs == 1000
    assert m.random_state == 42


def test_weights_none_before_fit():
    m = LinearRegression()
    assert m.weights_ is None
    assert m.bias_ is None


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(simple_linear):
    X, y = simple_linear
    m = LinearRegression()
    assert m.fit(X, y) is m


def test_weights_shape_after_fit(multi_feature):
    X, y = multi_feature
    m = LinearRegression().fit(X, y)
    assert m.weights_.shape == (3,)


def test_loss_history_length(simple_linear):
    X, y = simple_linear
    m = LinearRegression(n_epochs=200).fit(X, y)
    assert len(m.loss_history_) == 200


def test_loss_decreases_overall(simple_linear):
    """Loss at the end should be lower than loss at the start."""
    X, y = simple_linear
    m = LinearRegression(learning_rate=0.1, n_epochs=500).fit(X, y)
    assert m.loss_history_[-1] < m.loss_history_[0]


def test_loss_is_non_negative(simple_linear):
    X, y = simple_linear
    m = LinearRegression().fit(X, y)
    assert all(l >= 0 for l in m.loss_history_)


# ---------------------------------------------------------------------------
# Predict behavior
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    m = LinearRegression()
    with pytest.raises(RuntimeError):
        m.predict(np.array([[1, 2]]))


def test_predict_output_shape(multi_feature):
    X, y = multi_feature
    m = LinearRegression(learning_rate=0.1).fit(X, y)
    assert m.predict(X).shape == (300,)


def test_predict_continuous_values(simple_linear):
    """Predictions should not be restricted to integers."""
    X, y = simple_linear
    m = LinearRegression(learning_rate=0.1).fit(X, y)
    preds = m.predict(X)
    # At least some predictions should be non-integer
    assert not np.all(preds == preds.astype(int))


# ---------------------------------------------------------------------------
# Accuracy on simple known cases
# ---------------------------------------------------------------------------

def test_recovers_simple_linear_relationship(simple_linear):
    """
    On noise-free y = 3x + 2, the model should get very close.
    We check weight ≈ 3 and bias ≈ 2.
    """
    X, y = simple_linear
    m = LinearRegression(learning_rate=0.1, n_epochs=2000).fit(X, y)
    assert abs(m.weights_[0] - 3.0) < 0.05
    assert abs(m.bias_ - 2.0) < 0.05


def test_r2_near_one_on_clean_data(simple_linear):
    X, y = simple_linear
    m = LinearRegression(learning_rate=0.1, n_epochs=2000).fit(X, y)
    assert m.score(X, y) > 0.999


def test_r2_reasonable_on_noisy_data(multi_feature):
    X, y = multi_feature
    m = LinearRegression(learning_rate=0.1, n_epochs=1000).fit(X, y)
    r2 = m.score(X, y)
    assert r2 > 0.95   # noise is small, should still fit well


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_mse_non_negative(multi_feature):
    X, y = multi_feature
    m = LinearRegression(learning_rate=0.1).fit(X, y)
    assert m.mse(X, y) >= 0


def test_rmse_less_than_mse_when_mse_gt_1(simple_linear):
    """RMSE = sqrt(MSE), so RMSE < MSE when MSE > 1."""
    X, y = simple_linear
    # Use very few epochs so MSE is still large
    m = LinearRegression(learning_rate=0.001, n_epochs=5).fit(X, y)
    if m.mse(X, y) > 1:
        assert m.rmse(X, y) < m.mse(X, y)


def test_score_returns_float(simple_linear):
    X, y = simple_linear
    m = LinearRegression(learning_rate=0.1, n_epochs=500).fit(X, y)
    assert isinstance(m.score(X, y), float)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility(multi_feature):
    X, y = multi_feature
    m1 = LinearRegression(random_state=7).fit(X, y)
    m2 = LinearRegression(random_state=7).fit(X, y)
    np.testing.assert_array_almost_equal(m1.weights_, m2.weights_)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_feature_single_sample():
    X = np.array([[5.0]])
    y = np.array([10.0])
    m = LinearRegression(learning_rate=0.01, n_epochs=500).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (1,)


def test_predict_on_new_data(simple_linear):
    X, y = simple_linear
    m = LinearRegression(learning_rate=0.1, n_epochs=1000).fit(X, y)
    X_new = np.array([[0.0], [1.0], [-1.0]])
    preds = m.predict(X_new)
    assert preds.shape == (3,)
