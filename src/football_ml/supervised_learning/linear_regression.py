"""
Linear Regression with Gradient Descent
=========================================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


class LinearRegression:
    """
    Ordinary Linear Regression trained via Batch Gradient Descent.

    Predicts a continuous output (e.g. goal difference) as a weighted
    linear combination of input features.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent updates. Default 0.01.
    n_epochs : int
        Number of full passes over the training data. Default 1000.
    random_state : int or None
        Seed for reproducible weight initialization.

    Attributes
    ----------
    weights_ : np.ndarray, shape (n_features,)
        Learned weights (coefficients) after fitting.
    bias_ : float
        Learned bias (intercept) after fitting.
    loss_history_ : list of float
        Mean Squared Error recorded after every epoch.
        Use this to plot the loss curve and confirm convergence.
    """

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1000, random_state: int = 42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.weights_ = None
        self.bias_ = None
        self.loss_history_ = []

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------

    def _mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error — the quantity we are minimizing.

            MSE = (1/n) * sum((y_true - y_pred)^2)

        Squaring the errors means:
          - sign doesn't matter (over- and under-predictions both count)
          - large errors are penalized much more than small ones
        """
        return float(np.mean((y_true - y_pred) ** 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Train via Batch Gradient Descent.

        At each epoch:
          1. Compute predictions for ALL samples: y_hat = X @ w + b
          2. Compute gradients of MSE w.r.t. weights and bias
          3. Update weights and bias by stepping opposite the gradient

        Gradient formulas (derived from dMSE/dw and dMSE/db):
            dL/dw = (-2/n) * X^T @ (y - y_hat)
            dL/db = (-2/n) * sum(y - y_hat)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — continuous target values

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
            # Forward pass: predict
            y_hat = X @ self.weights_ + self.bias_

            # Compute gradients
            error = y - y_hat                                  # residuals, shape (n,)
            grad_w = (-2 / n_samples) * (X.T @ error)         # shape (n_features,)
            grad_b = (-2 / n_samples) * np.sum(error)         # scalar

            # Update: step opposite the gradient
            self.weights_ -= self.learning_rate * grad_w
            self.bias_    -= self.learning_rate * grad_b

            # Record loss
            self.loss_history_.append(self._mse(y, y_hat))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous values for samples in X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — real-valued predictions
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        return X @ self.weights_ + self.bias_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return R² (coefficient of determination).

        R² = 1 - (SS_res / SS_tot)

        Interpretation:
          R² = 1.0  → perfect predictions
          R² = 0.0  → model does no better than always predicting the mean
          R² < 0    → model is worse than predicting the mean

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return Mean Squared Error on a given dataset."""
        return self._mse(y, self.predict(X))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return Root Mean Squared Error — same units as the target variable.
        Much easier to interpret than MSE for football (goals, not goals²).
        """
        return float(np.sqrt(self.mse(X, y)))
