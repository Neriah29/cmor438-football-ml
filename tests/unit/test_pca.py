"""
Unit Tests — Principal Component Analysis (PCA)
================================================
Tests for football_ml.unsupervised_learning.pca.PCA

Run from repo root:
    pytest tests/unit/test_pca.py -v
"""

import numpy as np
import pytest
from football_ml.unsupervised_learning.pca import PCA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_2d():
    """Data with clear principal direction along y=x."""
    rng = np.random.default_rng(0)
    t = rng.uniform(-3, 3, 100)
    X = np.c_[t + rng.normal(0, 0.1, 100),
              t + rng.normal(0, 0.1, 100)]
    return X


@pytest.fixture
def multi_feature():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6))
    # Make features 0 and 1 dominant
    X[:, 0] *= 5
    X[:, 1] *= 3
    return X


@pytest.fixture
def correlated():
    """Highly correlated features — PCA should compress well."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=(150, 1))
    X = np.hstack([base + rng.normal(0, 0.1, (150, 1)) for _ in range(5)])
    return X


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = PCA()
    assert m.n_components is None


def test_custom_n_components():
    m = PCA(n_components=3)
    assert m.n_components == 3


def test_attributes_none_before_fit():
    m = PCA()
    assert m.components_ is None
    assert m.mean_ is None
    assert m.explained_variance_ is None


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(simple_2d):
    m = PCA(n_components=2)
    assert m.fit(simple_2d) is m


def test_components_shape(multi_feature):
    m = PCA(n_components=3).fit(multi_feature)
    assert m.components_.shape == (3, 6)


def test_components_shape_all(multi_feature):
    m = PCA().fit(multi_feature)
    assert m.components_.shape == (6, 6)


def test_mean_shape(multi_feature):
    m = PCA().fit(multi_feature)
    assert m.mean_.shape == (6,)


def test_mean_correct(multi_feature):
    m = PCA().fit(multi_feature)
    np.testing.assert_array_almost_equal(m.mean_, multi_feature.mean(axis=0))


def test_explained_variance_shape(multi_feature):
    m = PCA(n_components=4).fit(multi_feature)
    assert m.explained_variance_.shape == (4,)


def test_explained_variance_descending(multi_feature):
    """Eigenvalues should be in descending order."""
    m = PCA().fit(multi_feature)
    assert np.all(np.diff(m.explained_variance_) <= 1e-10)


def test_explained_variance_non_negative(multi_feature):
    m = PCA().fit(multi_feature)
    assert np.all(m.explained_variance_ >= 0)


def test_explained_variance_ratio_sums_to_one(multi_feature):
    m = PCA().fit(multi_feature)
    assert abs(m.explained_variance_ratio_.sum() - 1.0) < 1e-6


def test_cumulative_variance_ratio_ends_at_one(multi_feature):
    m = PCA().fit(multi_feature)
    assert abs(m.cumulative_variance_ratio_[-1] - 1.0) < 1e-6


def test_cumulative_variance_ratio_increasing(multi_feature):
    m = PCA().fit(multi_feature)
    assert np.all(np.diff(m.cumulative_variance_ratio_) >= -1e-10)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def test_transform_before_fit_raises():
    m = PCA()
    with pytest.raises(RuntimeError):
        m.transform(np.array([[1, 2, 3]]))


def test_transform_output_shape(multi_feature):
    m = PCA(n_components=3).fit(multi_feature)
    assert m.transform(multi_feature).shape == (200, 3)


def test_fit_transform_consistent(multi_feature):
    m = PCA(n_components=3)
    X_ft = m.fit_transform(multi_feature)
    X_t  = m.transform(multi_feature)
    np.testing.assert_array_almost_equal(X_ft, X_t)


def test_transform_centered(simple_2d):
    """Transformed data should have zero mean."""
    m = PCA(n_components=2).fit(simple_2d)
    X_t = m.transform(simple_2d)
    np.testing.assert_array_almost_equal(X_t.mean(axis=0), np.zeros(2), decimal=10)


def test_components_orthogonal(multi_feature):
    """Principal components should be orthogonal (dot product ≈ 0)."""
    m = PCA().fit(multi_feature)
    gram = m.components_ @ m.components_.T
    # Off-diagonal elements should be near zero
    off_diag = gram - np.eye(len(m.components_))
    assert np.max(np.abs(off_diag)) < 1e-10


# ---------------------------------------------------------------------------
# Inverse transform
# ---------------------------------------------------------------------------

def test_inverse_transform_before_fit_raises():
    m = PCA()
    with pytest.raises(RuntimeError):
        m.inverse_transform(np.array([[1, 2]]))


def test_inverse_transform_shape(multi_feature):
    m = PCA(n_components=3).fit(multi_feature)
    X_t = m.transform(multi_feature)
    X_r = m.inverse_transform(X_t)
    assert X_r.shape == multi_feature.shape


def test_perfect_reconstruction_all_components(multi_feature):
    """Using all components, reconstruction should be exact."""
    m = PCA().fit(multi_feature)
    X_r = m.inverse_transform(m.transform(multi_feature))
    np.testing.assert_array_almost_equal(X_r, multi_feature, decimal=8)


def test_reconstruction_error_decreases_with_more_components(multi_feature):
    """More components = lower reconstruction error."""
    errors = []
    for k in [1, 2, 3, 4, 6]:
        m = PCA(n_components=k).fit(multi_feature)
        errors.append(m.reconstruction_error(multi_feature))
    assert errors == sorted(errors, reverse=True)


def test_reconstruction_error_zero_all_components(multi_feature):
    m = PCA().fit(multi_feature)
    assert m.reconstruction_error(multi_feature) < 1e-10


# ---------------------------------------------------------------------------
# Dimensionality reduction quality
# ---------------------------------------------------------------------------

def test_first_component_explains_most_variance(simple_2d):
    """For data along y=x, first component should explain >> 50% variance."""
    m = PCA().fit(simple_2d)
    assert m.explained_variance_ratio_[0] > 0.9


def test_highly_correlated_compresses_well(correlated):
    """5 correlated features — 1 component should explain most variance."""
    m = PCA().fit(correlated)
    assert m.explained_variance_ratio_[0] > 0.85


def test_n_components_for_variance(multi_feature):
    m = PCA().fit(multi_feature)
    n = m.n_components_for_variance(threshold=0.95)
    assert 1 <= n <= multi_feature.shape[1]
    assert m.cumulative_variance_ratio_[n-1] >= 0.95


def test_n_components_for_variance_before_fit_raises():
    m = PCA()
    with pytest.raises(RuntimeError):
        m.n_components_for_variance()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_n_components_capped_at_n_features(multi_feature):
    """Requesting more components than features should not crash."""
    m = PCA(n_components=100).fit(multi_feature)
    assert m.components_.shape[0] <= 6


def test_single_component(multi_feature):
    m = PCA(n_components=1).fit(multi_feature)
    assert m.transform(multi_feature).shape == (200, 1)
