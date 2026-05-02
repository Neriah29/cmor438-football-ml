"""
Unit Tests — Support Vector Machine (SVM)
==========================================
Tests for football_ml.supervised_learning.svm.SVM

Run from repo root:
    pytest tests/unit/test_svm.py -v
"""

import numpy as np
import pytest
from football_ml.supervised_learning.svm import SVM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linearly_separable():
    """Two clearly separated blobs — SVM should find perfect margin."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[-3, -3], scale=0.4, size=(40, 2))
    X1 = rng.normal(loc=[3,  3],  scale=0.4, size=(40, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 40 + [1] * 40)
    return X, y


@pytest.fixture
def small_linear():
    """Tiny dataset for fast tests."""
    X = np.array([
        [1, 1], [2, 1], [1, 2],     # class 0
        [5, 5], [6, 5], [5, 6],     # class 1
    ], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = SVM()
    assert m.C == 1.0
    assert m.kernel == 'rbf'
    assert m.gamma == 'scale'


def test_custom_hyperparameters():
    m = SVM(C=5.0, kernel='linear', gamma=0.1)
    assert m.C == 5.0
    assert m.kernel == 'linear'
    assert m.gamma == 0.1


def test_support_vectors_none_before_fit():
    m = SVM()
    assert m.support_vectors_ is None


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(small_linear):
    X, y = small_linear
    m = SVM(kernel='linear', max_iter=100)
    assert m.fit(X, y) is m


def test_support_vectors_set_after_fit(small_linear):
    X, y = small_linear
    m = SVM(kernel='linear', max_iter=200).fit(X, y)
    assert m.support_vectors_ is not None
    assert m.n_support_ > 0


def test_support_vectors_are_subset_of_training(small_linear):
    """Support vectors must come from the training set."""
    X, y = small_linear
    m = SVM(kernel='linear', max_iter=200).fit(X, y)
    for sv in m.support_vectors_:
        assert any(np.allclose(sv, x) for x in X)


def test_bias_is_scalar(small_linear):
    X, y = small_linear
    m = SVM(kernel='linear', max_iter=200).fit(X, y)
    assert isinstance(float(m.bias_), float)


# ---------------------------------------------------------------------------
# Predict behavior
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    m = SVM()
    with pytest.raises(RuntimeError):
        m.predict(np.array([[1, 2]]))


def test_decision_function_before_fit_raises():
    m = SVM()
    with pytest.raises(RuntimeError):
        m.decision_function(np.array([[1, 2]]))


def test_predict_binary_values(small_linear):
    X, y = small_linear
    m = SVM(kernel='linear', max_iter=500).fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({0, 1})


def test_predict_output_shape(linearly_separable):
    X, y = linearly_separable
    m = SVM(kernel='linear', max_iter=200).fit(X, y)
    assert m.predict(X).shape == (80,)


def test_decision_function_shape(small_linear):
    X, y = small_linear
    m = SVM(kernel='linear', max_iter=200).fit(X, y)
    assert m.decision_function(X).shape == (6,)


def test_decision_function_signs(small_linear):
    """Class-1 points should have positive scores, class-0 negative."""
    X, y = small_linear
    m = SVM(kernel='linear', C=10.0, max_iter=500).fit(X, y)
    scores = m.decision_function(X)
    # Not guaranteed for all points due to soft margin, but should hold
    # for clearly separated data
    assert scores[3] > 0   # class-1 point [5,5]
    assert scores[0] < 0   # class-0 point [1,1]


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def test_high_accuracy_linear_kernel(linearly_separable):
    X, y = linearly_separable
    m = SVM(kernel='linear', C=1.0, max_iter=500).fit(X, y)
    assert m.score(X, y) > 0.90


def test_high_accuracy_rbf_kernel(linearly_separable):
    X, y = linearly_separable
    m = SVM(kernel='rbf', C=1.0, max_iter=500).fit(X, y)
    assert m.score(X, y) > 0.90


def test_perfect_accuracy_small_separable(small_linear):
    X, y = small_linear
    m = SVM(kernel='linear', C=10.0, max_iter=1000).fit(X, y)
    assert m.score(X, y) == 1.0


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

def test_predict_proba_range(linearly_separable):
    X, y = linearly_separable
    m = SVM(kernel='linear', max_iter=200).fit(X, y)
    probs = m.predict_proba(X)
    assert np.all(probs > 0) and np.all(probs < 1)


def test_predict_proba_shape(small_linear):
    X, y = small_linear
    m = SVM(kernel='linear', max_iter=200).fit(X, y)
    assert m.predict_proba(X).shape == (6,)


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def test_linear_kernel_symmetry(small_linear):
    X, y = small_linear
    m = SVM(kernel='linear').fit(X, y)
    K = m._kernel(X, X)
    np.testing.assert_array_almost_equal(K, K.T)


def test_rbf_kernel_diagonal_is_one(small_linear):
    """RBF(x, x) = exp(0) = 1 for all x."""
    X, y = small_linear
    m = SVM(kernel='rbf')
    m._gamma_val = 0.1
    K = m._kernel(X, X)
    np.testing.assert_array_almost_equal(np.diag(K), np.ones(len(X)))


def test_rbf_kernel_symmetry(small_linear):
    X, y = small_linear
    m = SVM(kernel='rbf')
    m._gamma_val = 0.1
    K = m._kernel(X, X)
    np.testing.assert_array_almost_equal(K, K.T)


# ---------------------------------------------------------------------------
# Gamma resolution
# ---------------------------------------------------------------------------

def test_gamma_scale(small_linear):
    X, y = small_linear
    m = SVM(gamma='scale')
    gamma = m._resolve_gamma(X)
    expected = 1.0 / (X.shape[1] * X.var())
    assert abs(gamma - expected) < 1e-10


def test_gamma_auto(small_linear):
    X, y = small_linear
    m = SVM(gamma='auto')
    gamma = m._resolve_gamma(X)
    assert abs(gamma - 0.5) < 1e-10   # 1 / 2 features


def test_gamma_float(small_linear):
    X, y = small_linear
    m = SVM(gamma=0.25)
    assert m._resolve_gamma(X) == 0.25


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def test_score_is_float(small_linear):
    X, y = small_linear
    m = SVM(kernel='linear', max_iter=200).fit(X, y)
    assert isinstance(m.score(X, y), float)


def test_score_range(linearly_separable):
    X, y = linearly_separable
    m = SVM(kernel='linear', max_iter=200).fit(X, y)
    s = m.score(X, y)
    assert 0.0 <= s <= 1.0
