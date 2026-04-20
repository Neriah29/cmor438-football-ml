"""
Perceptron — Binary Linear Classifier
======================================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


class Perceptron:
    """
    A single-layer Perceptron for binary classification.

    The Perceptron is the simplest neural network — one layer, no hidden units,
    a hard step-function activation. It can only learn linearly separable boundaries.

    Parameters
    ----------
    learning_rate : float
        Step size for weight updates. Default 0.01.
    n_epochs : int
        Number of full passes over the training data. Default 1000.
    random_state : int or None
        Seed for reproducible weight initialization.

    Attributes
    ----------
    weights_ : np.ndarray, shape (n_features,)
        Learned weights after fitting.
    bias_ : float
        Learned bias term after fitting.
    errors_per_epoch_ : list of int
        Number of misclassifications per epoch (useful for plotting convergence).
    """

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1000, random_state: int = 42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

        # Set during fit()
        self.weights_ = None
        self.bias_ = None
        self.errors_per_epoch_ = []

    # ------------------------------------------------------------------
    # Core: activation function
    # ------------------------------------------------------------------

    def _step_function(self, z: np.ndarray) -> np.ndarray:
        """
        Hard threshold activation: returns 1 if z >= 0, else 0.

        This is what makes the Perceptron a *binary* classifier — the output
        is always exactly 0 or 1, never a probability.
        """
        return np.where(z >= 0, 1, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train the Perceptron on labeled data.

        The update rule for each misclassified sample:
            w ← w + lr * (y_true - y_pred) * x
            b ← b + lr * (y_true - y_pred)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — must be binary {0, 1}

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        # Initialize weights near zero
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0
        self.errors_per_epoch_ = []

        for _ in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                z = np.dot(xi, self.weights_) + self.bias_
                y_pred = self._step_function(z)
                delta = self.learning_rate * (int(yi) - int(y_pred))
                if delta != 0:
                    self.weights_ += delta * xi
                    self.bias_ += delta
                    errors += 1
            self.errors_per_epoch_.append(errors)
            # Early stopping: if no errors in this epoch, we've converged
            if errors == 0:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in {0, 1}
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        z = np.dot(X, self.weights_) + self.bias_
        return self._step_function(z)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy: fraction of correctly classified samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float in [0, 1]
        """
        return float(np.mean(self.predict(X) == y))
