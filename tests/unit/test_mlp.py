"""
Unit Tests — Multi-Layer Perceptron (MLP)
==========================================
Tests for football_ml.supervised_learning.mlp.MLP

Run from repo root:
    pytest tests/unit/test_mlp.py -v
"""

import numpy as np
import pytest
from football_ml.supervised_learning.mlp import MLP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blobs():
    """Two well-separated Gaussian blobs — should reach high accuracy."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[-3, -3], scale=0.5, size=(100, 2))
    X1 = rng.normal(loc=[3,  3],  scale=0.5, size=(100, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


@pytest.fixture
def xor_data():
    """
    XOR pattern — not linearly separable.
    A single-layer model (Logistic Regression) cannot solve this.
    An MLP with a hidden layer can.
    """
    X = np.array([
        [0, 0], [0, 1],
        [1, 0], [1, 1],
    ], dtype=float)
    y = np.array([0, 1, 1, 0])   # XOR
    return X, y


@pytest.fixture
def noisy_binary():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    logits = 2 * X[:, 0] + X[:, 1] + rng.normal(scale=0.5, size=300)
    y = (logits > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_default_hyperparameters():
    m = MLP()
    assert m.hidden_layer_sizes == (64, 32)
    assert m.learning_rate == 0.01
    assert m.n_epochs == 1000
    assert m.random_state == 42


def test_custom_architecture():
    m = MLP(hidden_layer_sizes=(16,), learning_rate=0.1, n_epochs=500)
    assert m.hidden_layer_sizes == (16,)
    assert m.learning_rate == 0.1


def test_weights_none_before_fit():
    m = MLP()
    assert m.weights_ is None
    assert m.biases_ is None


# ---------------------------------------------------------------------------
# Fit behavior
# ---------------------------------------------------------------------------

def test_fit_returns_self(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(8,), n_epochs=10)
    assert m.fit(X, y) is m


def test_correct_number_of_weight_matrices(blobs):
    """
    Architecture (2 features) → (8) → (4) → (1 output)
    Should have 3 weight matrices.
    """
    X, y = blobs
    m = MLP(hidden_layer_sizes=(8, 4), n_epochs=5).fit(X, y)
    assert len(m.weights_) == 3
    assert len(m.biases_) == 3


def test_weight_matrix_shapes(noisy_binary):
    """
    Input=4 features, hidden=(16, 8), output=1
    Shapes: (4,16), (16,8), (8,1)
    """
    X, y = noisy_binary
    m = MLP(hidden_layer_sizes=(16, 8), n_epochs=5).fit(X, y)
    assert m.weights_[0].shape == (4, 16)
    assert m.weights_[1].shape == (16, 8)
    assert m.weights_[2].shape == (8, 1)


def test_loss_history_length(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(8,), n_epochs=50).fit(X, y)
    assert len(m.loss_history_) == 50


def test_loss_decreases(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(16,), learning_rate=0.05, n_epochs=200).fit(X, y)
    assert m.loss_history_[-1] < m.loss_history_[0]


def test_loss_non_negative(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(8,), n_epochs=20).fit(X, y)
    assert all(l >= 0 for l in m.loss_history_)


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def test_relu_positive():
    m = MLP()
    z = np.array([1.0, 2.0, 0.5])
    np.testing.assert_array_equal(m._relu(z), z)


def test_relu_negative():
    m = MLP()
    z = np.array([-1.0, -0.5, -3.0])
    np.testing.assert_array_equal(m._relu(z), np.zeros(3))


def test_relu_zero():
    m = MLP()
    assert m._relu(np.array([0.0]))[0] == 0.0


def test_relu_derivative_positive():
    m = MLP()
    z = np.array([1.0, 2.0])
    np.testing.assert_array_equal(m._relu_derivative(z), np.ones(2))


def test_relu_derivative_negative():
    m = MLP()
    z = np.array([-1.0, -2.0])
    np.testing.assert_array_equal(m._relu_derivative(z), np.zeros(2))


def test_sigmoid_at_zero():
    m = MLP()
    assert abs(m._sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10


def test_sigmoid_range(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(8,), n_epochs=10).fit(X, y)
    probs = m.predict_proba(X)
    assert np.all(probs > 0) and np.all(probs < 1)


# ---------------------------------------------------------------------------
# Predict behavior
# ---------------------------------------------------------------------------

def test_predict_proba_before_fit_raises():
    m = MLP()
    with pytest.raises(RuntimeError):
        m.predict_proba(np.array([[1, 2]]))


def test_predict_output_shape(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(8,), n_epochs=10).fit(X, y)
    assert m.predict(X).shape == (200,)


def test_predict_binary_values(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(8,), n_epochs=10).fit(X, y)
    preds = m.predict(X)
    assert set(preds).issubset({0, 1})


def test_predict_proba_shape(noisy_binary):
    X, y = noisy_binary
    m = MLP(hidden_layer_sizes=(8,), n_epochs=10).fit(X, y)
    assert m.predict_proba(X).shape == (300,)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def test_high_accuracy_on_separable_data(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(16,), learning_rate=0.05, n_epochs=500).fit(X, y)
    assert m.score(X, y) > 0.95


def test_better_than_random_on_noisy(noisy_binary):
    X, y = noisy_binary
    m = MLP(hidden_layer_sizes=(16, 8), learning_rate=0.05, n_epochs=300).fit(X, y)
    assert m.score(X, y) > 0.6


def test_xor_solvable_with_hidden_layer(xor_data):
    """
    XOR is not linearly separable — Logistic Regression fails on it.
    An MLP with a hidden layer should be able to solve it.
    """
    X, y = xor_data
    # Train many times with different seeds — XOR convergence is sensitive
    best = 0.0
    for seed in range(20):
        m = MLP(hidden_layer_sizes=(4,), learning_rate=0.5,
                n_epochs=2000, random_state=seed).fit(X, y)
        best = max(best, m.score(X, y))
    assert best == 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility(blobs):
    X, y = blobs
    m1 = MLP(hidden_layer_sizes=(8,), n_epochs=50, random_state=7).fit(X, y)
    m2 = MLP(hidden_layer_sizes=(8,), n_epochs=50, random_state=7).fit(X, y)
    np.testing.assert_array_almost_equal(m1.weights_[0], m2.weights_[0])


# ---------------------------------------------------------------------------
# Architecture flexibility
# ---------------------------------------------------------------------------

def test_single_hidden_layer(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(32,), n_epochs=10).fit(X, y)
    assert len(m.weights_) == 2


def test_three_hidden_layers(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(16, 8, 4), n_epochs=10).fit(X, y)
    assert len(m.weights_) == 4


def test_score_is_float(blobs):
    X, y = blobs
    m = MLP(hidden_layer_sizes=(8,), n_epochs=10).fit(X, y)
    assert isinstance(m.score(X, y), float)
