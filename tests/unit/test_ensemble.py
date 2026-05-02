"""
Unit Tests — Ensemble Methods
===============================
Tests for:
  football_ml.supervised_learning.ensemble.RandomForestClassifier
  football_ml.supervised_learning.ensemble.GradientBoostingClassifier

Run from repo root:
    pytest tests/unit/test_ensemble.py -v
"""

import numpy as np
import pytest
from football_ml.supervised_learning.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)


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
def noisy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    logits = 2 * X[:, 0] + X[:, 1] + rng.normal(scale=0.5, size=300)
    y = (logits > 0).astype(int)
    return X, y


# ===========================================================================
# Random Forest
# ===========================================================================

class TestRandomForest:

    def test_default_hyperparameters(self):
        m = RandomForestClassifier()
        assert m.n_estimators == 100
        assert m.max_features == 'sqrt'
        assert m.random_state == 42

    def test_trees_none_before_fit(self):
        m = RandomForestClassifier()
        assert m.trees_ == []

    def test_fit_returns_self(self, blobs):
        X, y = blobs
        m = RandomForestClassifier(n_estimators=5)
        assert m.fit(X, y) is m

    def test_correct_number_of_trees(self, blobs):
        X, y = blobs
        m = RandomForestClassifier(n_estimators=10).fit(X, y)
        assert len(m.trees_) == 10

    def test_feature_importances_shape(self, noisy):
        X, y = noisy
        m = RandomForestClassifier(n_estimators=10).fit(X, y)
        assert m.feature_importances_.shape == (4,)

    def test_feature_importances_sum_to_one(self, noisy):
        X, y = noisy
        m = RandomForestClassifier(n_estimators=10).fit(X, y)
        assert abs(m.feature_importances_.sum() - 1.0) < 1e-6

    def test_predict_proba_range(self, blobs):
        X, y = blobs
        m = RandomForestClassifier(n_estimators=10).fit(X, y)
        probs = m.predict_proba(X)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_proba_shape(self, blobs):
        X, y = blobs
        m = RandomForestClassifier(n_estimators=10).fit(X, y)
        assert m.predict_proba(X).shape == (160,)

    def test_predict_binary_values(self, blobs):
        X, y = blobs
        m = RandomForestClassifier(n_estimators=10).fit(X, y)
        preds = m.predict(X)
        assert set(preds).issubset({0, 1})

    def test_high_accuracy_on_blobs(self, blobs):
        X, y = blobs
        m = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X, y)
        assert m.score(X, y) > 0.95

    def test_better_than_random_on_noisy(self, noisy):
        X, y = noisy
        m = RandomForestClassifier(n_estimators=30, max_depth=5).fit(X, y)
        assert m.score(X, y) > 0.7

    def test_predict_before_fit_raises(self):
        m = RandomForestClassifier()
        with pytest.raises(RuntimeError):
            m.predict_proba(np.array([[1, 2]]))

    def test_reproducibility(self, blobs):
        X, y = blobs
        m1 = RandomForestClassifier(n_estimators=10, random_state=7).fit(X, y)
        m2 = RandomForestClassifier(n_estimators=10, random_state=7).fit(X, y)
        np.testing.assert_array_almost_equal(
            m1.predict_proba(X), m2.predict_proba(X)
        )

    def test_max_features_sqrt(self, noisy):
        X, y = noisy
        m = RandomForestClassifier(n_estimators=5, max_features='sqrt').fit(X, y)
        # sqrt(4) = 2 features per tree
        assert all(len(fi) == 2 for fi in m.feature_indices_)

    def test_max_features_int(self, noisy):
        X, y = noisy
        m = RandomForestClassifier(n_estimators=5, max_features=3).fit(X, y)
        assert all(len(fi) == 3 for fi in m.feature_indices_)

    def test_score_is_float(self, blobs):
        X, y = blobs
        m = RandomForestClassifier(n_estimators=5).fit(X, y)
        assert isinstance(m.score(X, y), float)


# ===========================================================================
# Gradient Boosting
# ===========================================================================

class TestGradientBoosting:

    def test_default_hyperparameters(self):
        m = GradientBoostingClassifier()
        assert m.n_estimators == 100
        assert m.learning_rate == 0.1
        assert m.max_depth == 3

    def test_fit_returns_self(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=5)
        assert m.fit(X, y) is m

    def test_correct_number_of_trees(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        assert len(m.trees_) == 10

    def test_loss_history_length(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=20).fit(X, y)
        assert len(m.loss_history_) == 20

    def test_loss_decreases(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=50,
                                        learning_rate=0.1).fit(X, y)
        assert m.loss_history_[-1] < m.loss_history_[0]

    def test_loss_non_negative(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        assert all(l >= 0 for l in m.loss_history_)

    def test_predict_proba_range(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        probs = m.predict_proba(X)
        assert np.all(probs > 0) and np.all(probs < 1)

    def test_predict_proba_shape(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        assert m.predict_proba(X).shape == (160,)

    def test_predict_binary_values(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=10).fit(X, y)
        preds = m.predict(X)
        assert set(preds).issubset({0, 1})

    def test_high_accuracy_on_blobs(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=50,
                                        learning_rate=0.1).fit(X, y)
        assert m.score(X, y) > 0.95

    def test_better_than_random_on_noisy(self, noisy):
        X, y = noisy
        m = GradientBoostingClassifier(n_estimators=50,
                                        learning_rate=0.1).fit(X, y)
        assert m.score(X, y) > 0.7

    def test_predict_before_fit_raises(self):
        m = GradientBoostingClassifier()
        with pytest.raises(RuntimeError):
            m.predict_proba(np.array([[1, 2]]))

    def test_more_estimators_lower_loss(self, noisy):
        X, y = noisy
        m_few  = GradientBoostingClassifier(n_estimators=5,  learning_rate=0.1).fit(X, y)
        m_many = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1).fit(X, y)
        assert m_many.score(X, y) >= m_few.score(X, y)

    def test_score_is_float(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=5).fit(X, y)
        assert isinstance(m.score(X, y), float)

    def test_init_pred_set_after_fit(self, blobs):
        X, y = blobs
        m = GradientBoostingClassifier(n_estimators=5).fit(X, y)
        assert m.init_pred_ is not None
