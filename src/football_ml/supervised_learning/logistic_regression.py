"""
Logistic Regression with Gradient Descent
==========================================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression trained via Batch Gradient Descent.

    Unlike Linear Regression (which predicts a raw number) or the Perceptron
    (which outputs a hard 0/1), Logistic Regression outputs a PROBABILITY —
    the likelihood that a match belongs to class 1 (home win).

    The key ingredient: the sigmoid function squashes the weighted sum z into
    the range (0, 1), giving us a meaningful probability instead of a raw score.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent. Default 0.1.
    n_epochs : int
        Number of full passes over the training data. Default 1000.
    threshold : float
        Probability cutoff for converting probabilities to hard labels.
        Default 0.5 — predict 1 if p >= 0.5, else 0.
    random_state : int or None
        Seed for reproducible weight initialization.

    Attributes
    ----------
    weights_ : np.ndarray, shape (n_features,)
        Learned weights after fitting.
    bias_ : float
        Learned bias after fitting.
    loss_history_ : list of float
        Cross-entropy loss recorded after every epoch.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_epochs: int = 1000,
        threshold: float = 0.5,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.random_state = random_state

        self.weights_ = None
        self.bias_ = None
        self.loss_history_ = []

    # ------------------------------------------------------------------
    # Core functions
    # ------------------------------------------------------------------

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid (logistic) function — the heart of Logistic Regression.

            σ(z) = 1 / (1 + e^{-z})

        Maps any real number z to (0, 1):
          - Very negative z → probability near 0
          - z = 0           → probability = 0.5 (maximum uncertainty)
          - Very positive z → probability near 1

        Numerically stable implementation to avoid overflow on large |z|.
        """
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _cross_entropy_loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Binary Cross-Entropy Loss (Log Loss).

            L = -(1/n) * sum[ y*log(p) + (1-y)*log(1-p) ]

        Interpretation:
          - If y=1 and p≈1 → loss ≈ 0  (correct and confident)
          - If y=1 and p≈0 → loss → ∞  (wrong and confident — worst case)
          - If p=0.5        → loss = log(2) ≈ 0.693 (maximum uncertainty)

        This is a better loss for probabilities than MSE because it heavily
        penalises confident wrong predictions.
        """
        # Clip probabilities to avoid log(0)
        p = np.clip(y_prob, 1e-12, 1 - 1e-12)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train via Batch Gradient Descent minimising cross-entropy loss.

        The gradient of cross-entropy loss w.r.t. weights turns out to be
        beautifully simple (the calculus works out cleanly):

            dL/dw = (1/n) * X^T @ (p - y)
            dL/db = (1/n) * sum(p - y)

        Where p = sigmoid(X @ w + b) are the predicted probabilities.

        Notice: this looks almost identical to Linear Regression's gradient,
        except we use probabilities p instead of raw predictions y_hat.
        That's gradient descent's elegance — same structure, different loss.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — binary {0, 1}

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_epochs):
            # Forward pass: weighted sum → probability
            z = X @ self.weights_ + self.bias_
            p = self._sigmoid(z)

            # Gradients (derived from d(cross-entropy)/dw)
            error = p - y                                   # shape (n,)
            grad_w = (1 / n_samples) * (X.T @ error)       # shape (n_features,)
            grad_b = (1 / n_samples) * np.sum(error)       # scalar

            # Gradient descent step
            self.weights_ -= self.learning_rate * grad_w
            self.bias_    -= self.learning_rate * grad_b

            # Record loss
            self.loss_history_.append(self._cross_entropy_loss(y, p))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted probabilities for class 1.

        This is what makes Logistic Regression special — instead of just
        a label, you get a confidence score for every prediction.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in (0, 1)
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        z = X @ self.weights_ + self.bias_
        return self._sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return hard binary labels using self.threshold.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in {0, 1}
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return classification accuracy.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float in [0, 1]
        """
        return float(np.mean(self.predict(X) == y))

    def log_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return cross-entropy loss on a given dataset."""
        return self._cross_entropy_loss(y, self.predict_proba(X))
