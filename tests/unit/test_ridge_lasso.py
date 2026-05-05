"""
Unit Tests — Ridge & Lasso Regression
=======================================
Tests for:
  football_ml.supervised_learning.ridge_lasso.RidgeRegression
  football_ml.supervised_learning.ridge_lasso.LassoRegression

Run from repo root:
    pytest tests/unit/test_ridge_lasso.py -v
"""

import numpy as np
import pytest
from football_ml.supervised_learning.ridge_lasso import RidgeRegression, LassoRegression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_linear():
    """y = 3x + 2, no noise."""
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


@pytest.fixture
def sparse_target():
    """
    Only 2 of 10 features truly matter.
    Lasso should zero out the irrelevant 8.
    """
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 10))
    y = 3 * X[:, 0] + 2 * X[:, 1] + rng.normal(scale=0.1, size=500)
    return X, y


# ===========================================================================
# Ridge Regression
# ===========================================================================

class TestRidge:

    def test_default_hyperparameters(self):
        m = RidgeRegression()
        assert m.alpha == 1.0
        assert m.learning_rate == 0.01
        assert m.n_epochs == 1000

    def test_weights_none_before_fit(self):
        m = RidgeRegression()
        assert m.weights_ is None
        assert m.bias_ is None

    def test_fit_returns_self(self, simple_linear):
        X, y = simple_linear
        m = RidgeRegression()
        assert m.fit(X, y) is m

    def test_weights_shape(self, multi_feature):
        X, y = multi_feature
        m = RidgeRegression().fit(X, y)
        assert m.weights_.shape == (3,)

    def test_loss_history_length(self, simple_linear):
        X, y = simple_linear
        m = RidgeRegression(n_epochs=200).fit(X, y)
        assert len(m.loss_history_) == 200

    def test_loss_decreases(self, simple_linear):
        X, y = simple_linear
        m = RidgeRegression(learning_rate=0.05, n_epochs=500).fit(X, y)
        assert m.loss_history_[-1] < m.loss_history_[0]

    def test_predict_before_fit_raises(self):
        m = RidgeRegression()
        with pytest.raises(RuntimeError):
            m.predict(np.array([[1.0]]))

    def test_predict_shape(self, multi_feature):
        X, y = multi_feature
        m = RidgeRegression(learning_rate=0.1).fit(X, y)
        assert m.predict(X).shape == (300,)

    def test_r2_reasonable(self, multi_feature):
        X, y = multi_feature
        m = RidgeRegression(alpha=0.01, learning_rate=0.1, n_epochs=1000).fit(X, y)
        assert m.score(X, y) > 0.90

    def test_alpha_zero_approaches_linear_regression(self, simple_linear):
        """With alpha=0, Ridge reduces to plain Linear Regression."""
        X, y = simple_linear
        m = RidgeRegression(alpha=0.0, learning_rate=0.1, n_epochs=2000).fit(X, y)
        assert abs(m.weights_[0] - 3.0) < 0.1
        assert abs(m.bias_ - 2.0) < 0.1

    def test_large_alpha_shrinks_weights(self, multi_feature):
        """Large alpha → weights closer to zero than small alpha."""
        X, y = multi_feature
        m_small = RidgeRegression(alpha=0.001, learning_rate=0.1, n_epochs=500).fit(X, y)
        m_large = RidgeRegression(alpha=100.0, learning_rate=0.01, n_epochs=500).fit(X, y)
        assert np.sum(m_large.weights_ ** 2) < np.sum(m_small.weights_ ** 2)

    def test_ridge_weights_never_exactly_zero(self, sparse_target):
        """Ridge shrinks but never zeros weights."""
        X, y = sparse_target
        m = RidgeRegression(alpha=10.0, learning_rate=0.01, n_epochs=500).fit(X, y)
        assert np.all(np.abs(m.weights_) > 1e-6)

    def test_rmse_less_than_mse_when_mse_gt_1(self, simple_linear):
        X, y = simple_linear
        m = RidgeRegression(learning_rate=0.001, n_epochs=5).fit(X, y)
        if m.mse(X, y) > 1:
            assert m.rmse(X, y) < m.mse(X, y)

    def test_score_is_float(self, multi_feature):
        X, y = multi_feature
        m = RidgeRegression(learning_rate=0.1).fit(X, y)
        assert isinstance(m.score(X, y), float)

    def test_reproducibility(self, multi_feature):
        X, y = multi_feature
        m1 = RidgeRegression(random_state=5).fit(X, y)
        m2 = RidgeRegression(random_state=5).fit(X, y)
        np.testing.assert_array_almost_equal(m1.weights_, m2.weights_)


# ===========================================================================
# Lasso Regression
# ===========================================================================

class TestLasso:

    def test_default_hyperparameters(self):
        m = LassoRegression()
        assert m.alpha == 0.1
        assert m.learning_rate == 0.01
        assert m.n_epochs == 1000

    def test_weights_none_before_fit(self):
        m = LassoRegression()
        assert m.weights_ is None

    def test_fit_returns_self(self, simple_linear):
        X, y = simple_linear
        assert LassoRegression().fit(X, y) is LassoRegression().fit(X, y).__class__() or True
        m = LassoRegression()
        assert m.fit(X, y) is m

    def test_weights_shape(self, multi_feature):
        X, y = multi_feature
        m = LassoRegression().fit(X, y)
        assert m.weights_.shape == (3,)

    def test_loss_history_length(self, simple_linear):
        X, y = simple_linear
        m = LassoRegression(n_epochs=150).fit(X, y)
        assert len(m.loss_history_) == 150

    def test_loss_decreases(self, simple_linear):
        X, y = simple_linear
        m = LassoRegression(alpha=0.01, learning_rate=0.05, n_epochs=500).fit(X, y)
        assert m.loss_history_[-1] < m.loss_history_[0]

    def test_predict_before_fit_raises(self):
        m = LassoRegression()
        with pytest.raises(RuntimeError):
            m.predict(np.array([[1.0]]))

    def test_predict_shape(self, multi_feature):
        X, y = multi_feature
        m = LassoRegression(learning_rate=0.1).fit(X, y)
        assert m.predict(X).shape == (300,)

    def test_r2_reasonable(self, multi_feature):
        X, y = multi_feature
        m = LassoRegression(alpha=0.001, learning_rate=0.1, n_epochs=1000).fit(X, y)
        assert m.score(X, y) > 0.85

    def test_large_alpha_zeros_weights(self, sparse_target):
        """Large alpha should drive some weights to near-zero."""
        X, y = sparse_target
        m = LassoRegression(alpha=1.0, learning_rate=0.01, n_epochs=1000).fit(X, y)
        assert m.n_zero_weights > 0

    def test_lasso_sparser_than_ridge(self, sparse_target):
        """Lasso should zero more weights than Ridge under same alpha."""
        X, y = sparse_target
        ridge = RidgeRegression(alpha=1.0, learning_rate=0.01, n_epochs=500).fit(X, y)
        lasso = LassoRegression(alpha=1.0, learning_rate=0.01, n_epochs=500).fit(X, y)
        ridge_zeros = int(np.sum(np.abs(ridge.weights_) < 1e-4))
        assert lasso.n_zero_weights >= ridge_zeros

    def test_n_zero_weights_before_fit(self):
        m = LassoRegression()
        assert m.n_zero_weights == 0

    def test_score_is_float(self, multi_feature):
        X, y = multi_feature
        m = LassoRegression(learning_rate=0.1).fit(X, y)
        assert isinstance(m.score(X, y), float)

    def test_reproducibility(self, multi_feature):
        X, y = multi_feature
        m1 = LassoRegression(random_state=9).fit(X, y)
        m2 = LassoRegression(random_state=9).fit(X, y)
        np.testing.assert_array_almost_equal(m1.weights_, m2.weights_)

    def test_rmse_positive(self, multi_feature):
        X, y = multi_feature
        m = LassoRegression(learning_rate=0.1).fit(X, y)
        assert m.rmse(X, y) >= 0
