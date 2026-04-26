"""
Unit Tests — Decision Tree Classifier
=======================================
Tests for football_ml.supervised_learning.decision_tree.DecisionTreeClassifier

Run from repo root:
    pytest tests/unit/test_decision_tree.py -v
"""

import numpy as np
import pytest
from football_ml.supervised_learning.decision_tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_separable():
    """Perfectly separable: all class-0 on left, class-1 on right."""
    X = np.array([
        [1.0], [2.0], [3.0],   # class 0
        [7.0], [8.0], [9.0],   # class 1
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[-3, -3], scale=0.5, size=(80, 2))
    X1 = rng.normal(loc=[3,  3],  scale=0.5, size=(80, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 80 + [1] * 80)
    return X, y


@pytest.fixture
def multi_feature():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = DecisionTreeClassifier()
    assert m.max_depth is None
    assert m.min_samples_split == 2
    assert m.min_samples_leaf == 1


def test_custom_hyperparameters():
    m = DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=2)
    assert m.max_depth == 3
    assert m.min_samples_split == 5
    assert m.min_samples_leaf == 2


def test_root_none_before_fit():
    m = DecisionTreeClassifier()
    assert m.root_ is None


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(blobs):
    X, y = blobs
    m = DecisionTreeClassifier(max_depth=3)
    assert m.fit(X, y) is m


def test_root_set_after_fit(blobs):
    X, y = blobs
    m = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert m.root_ is not None


def test_feature_importances_shape(multi_feature):
    X, y = multi_feature
    m = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert m.feature_importances_.shape == (4,)


def test_feature_importances_sum_to_one(multi_feature):
    X, y = multi_feature
    m = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert abs(m.feature_importances_.sum() - 1.0) < 1e-6


def test_feature_importances_non_negative(multi_feature):
    X, y = multi_feature
    m = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert np.all(m.feature_importances_ >= 0)


# ---------------------------------------------------------------------------
# Tree structure
# ---------------------------------------------------------------------------

def test_max_depth_respected(blobs):
    X, y = blobs
    for max_depth in [1, 2, 3, 5]:
        m = DecisionTreeClassifier(max_depth=max_depth).fit(X, y)
        assert m.get_depth() <= max_depth


def test_pure_node_stops_early(simple_separable):
    """A perfectly separable dataset should produce a shallow tree."""
    X, y = simple_separable
    m = DecisionTreeClassifier().fit(X, y)
    assert m.get_depth() <= 2


def test_n_leaves_positive(blobs):
    X, y = blobs
    m = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert m.get_n_leaves() >= 1


def test_deeper_tree_more_leaves(blobs):
    X, y = blobs
    m_shallow = DecisionTreeClassifier(max_depth=2).fit(X, y)
    m_deep    = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert m_deep.get_n_leaves() >= m_shallow.get_n_leaves()


# ---------------------------------------------------------------------------
# Predict behavior
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises():
    m = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        m.predict(np.array([[1, 2]]))


def test_predict_output_shape(blobs):
    X, y = blobs
    m = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert m.predict(X).shape == (160,)


def test_predict_binary_values(blobs):
    X, y = blobs
    m = DecisionTreeClassifier(max_depth=3).fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({0, 1})


def test_perfect_accuracy_on_separable(simple_separable):
    X, y = simple_separable
    m = DecisionTreeClassifier().fit(X, y)
    assert m.score(X, y) == 1.0


def test_high_accuracy_on_blobs(blobs):
    X, y = blobs
    m = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert m.score(X, y) > 0.95


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

def test_predict_proba_before_fit_raises():
    m = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        m.predict_proba(np.array([[1, 2]]))


def test_predict_proba_range(blobs):
    X, y = blobs
    m = DecisionTreeClassifier(max_depth=3).fit(X, y)
    probs = m.predict_proba(X)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_predict_proba_shape(blobs):
    X, y = blobs
    m = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert m.predict_proba(X).shape == (160,)


# ---------------------------------------------------------------------------
# Gini impurity
# ---------------------------------------------------------------------------

def test_gini_pure_node():
    """All same class → Gini = 0."""
    m = DecisionTreeClassifier()
    assert m._gini(np.array([0, 0, 0])) == 0.0
    assert m._gini(np.array([1, 1, 1])) == 0.0


def test_gini_max_impurity():
    """50/50 split → Gini = 0.5."""
    m = DecisionTreeClassifier()
    assert abs(m._gini(np.array([0, 0, 1, 1])) - 0.5) < 1e-10


def test_gini_empty():
    m = DecisionTreeClassifier()
    assert m._gini(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# No scaling needed
# ---------------------------------------------------------------------------

def test_works_without_scaling():
    """Decision trees don't require StandardScaler — test with raw large values."""
    X = np.array([[1000, 0.001], [2000, 0.002], [9000, 0.009], [9500, 0.01]])
    y = np.array([0, 0, 1, 1])
    m = DecisionTreeClassifier().fit(X, y)
    assert m.score(X, y) == 1.0


# ---------------------------------------------------------------------------
# Overfitting behavior
# ---------------------------------------------------------------------------

def test_unlimited_depth_overfits_training(blobs):
    """Unlimited depth should perfectly memorize training data."""
    X, y = blobs
    m = DecisionTreeClassifier(max_depth=None).fit(X, y)
    assert m.score(X, y) == 1.0


def test_score_is_float(blobs):
    X, y = blobs
    m = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert isinstance(m.score(X, y), float)
