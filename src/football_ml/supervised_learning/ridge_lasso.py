"""
Ridge & Lasso Regression
=========================
From-scratch NumPy implementations for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


# =============================================================================
# Ridge Regression (L2 regularization)
# =============================================================================

class RidgeRegression:
    """
    Ridge Regression — Linear Regression with L2 penalty.

    Minimizes: MSE + alpha * sum(w_j^2)

    The L2 penalty shrinks all weights toward zero proportionally.
    No weight reaches exactly zero — all features stay in the model.
    Solved via gradient descent (same as Linear Regression, just with
    an extra gradient term from the penalty).

    Parameters
    ----------
    alpha : float
        Regularization strength. 0 = plain Linear Regression.
        Larger = stronger shrinkage. Default 1.0.
    learning_rate : float
        Gradient descent step size. Default 0.01.
    n_epochs : int
        Number of training epochs. Default 1000.
    random_state : int or None
        Seed for weight initialization.

    Attributes
    ----------
    weights_ : np.ndarray, shape (n_features,)
    bias_ : float
    loss_history_ : list of float  — MSE (without penalty) per epoch
    """

    def __init__(
        self,
        alpha: float = 1.0,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        random_state: int = 42,
    ):
        self.alpha         = alpha
        self.learning_rate = learning_rate
        self.n_epochs      = n_epochs
        self.random_state  = random_state

        self.weights_     = None
        self.bias_        = None
        self.loss_history_ = []

    def _mse(self, y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """
        Train via gradient descent with L2 penalty.

        Gradient of Ridge loss w.r.t. weights:
            dL/dw = (-2/n) * X^T @ (y - y_hat) + 2 * alpha * w

        The extra term `2 * alpha * w` is what distinguishes Ridge
        from plain Linear Regression. It pulls every weight back
        toward zero at every update step.

        The bias is NOT regularized — standard practice, since
        regularizing the bias would shift the entire model up/down.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.weights_ = rng.normal(0, 0.01, size=n_features)
        self.bias_    = 0.0
        self.loss_history_ = []

        for _ in range(self.n_epochs):
            y_hat = X @ self.weights_ + self.bias_
            error = y - y_hat

            # Data gradient (MSE only, no penalty)
            grad_w_data = (-2 / n_samples) * (X.T @ error)
            grad_b      = (-2 / n_samples) * np.sum(error)  # bias not regularized

            # Apply L2 penalty as multiplicative weight decay.
            # Clamped to [0, 1] to prevent oscillation when
            # 2 * lr * alpha >= 1 (e.g. large alpha or large lr).
            decay = max(0.0, 1.0 - 2 * self.learning_rate * self.alpha)
            self.weights_ = decay * self.weights_ - self.learning_rate * grad_w_data
            self.bias_   -= self.learning_rate * grad_b

            self.loss_history_.append(self._mse(y, y_hat))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        return X @ self.weights_ + self.bias_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² (coefficient of determination)."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        return self._mse(y, self.predict(X))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(self.mse(X, y)))


# =============================================================================
# Lasso Regression (L1 regularization)
# =============================================================================

class LassoRegression:
    """
    Lasso Regression — Linear Regression with L1 penalty.

    Minimizes: MSE + alpha * sum(|w_j|)

    The L1 penalty can shrink weights all the way to exactly zero,
    effectively removing features from the model. This makes Lasso
    a form of automatic feature selection.

    The L1 penalty is not differentiable at w=0, so we use the
    subgradient: sign(w). This is a standard approach for Lasso
    via coordinate or gradient descent.

    Parameters
    ----------
    alpha : float
        Regularization strength. 0 = plain Linear Regression.
        Default 0.1.
    learning_rate : float
        Gradient descent step size. Default 0.01.
    n_epochs : int
        Number of training epochs. Default 1000.
    random_state : int or None

    Attributes
    ----------
    weights_ : np.ndarray, shape (n_features,)
    bias_ : float
    loss_history_ : list of float
    """

    def __init__(
        self,
        alpha: float = 0.1,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        random_state: int = 42,
    ):
        self.alpha         = alpha
        self.learning_rate = learning_rate
        self.n_epochs      = n_epochs
        self.random_state  = random_state

        self.weights_      = None
        self.bias_         = None
        self.loss_history_ = []

    def _mse(self, y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        """
        Train via subgradient descent with L1 penalty.

        Gradient of Lasso loss w.r.t. weights:
            dL/dw = (-2/n) * X^T @ (y - y_hat) + alpha * sign(w)

        sign(w) is the subgradient of |w| — it's +1 for positive
        weights, -1 for negative, and 0 for exactly zero.

        This consistently pushes weights toward zero. Unlike Ridge,
        the L1 penalty applies the same absolute force regardless
        of weight magnitude, so small weights get pushed to exactly
        zero rather than just getting smaller.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.weights_ = rng.normal(0, 0.01, size=n_features)
        self.bias_    = 0.0
        self.loss_history_ = []

        for _ in range(self.n_epochs):
            y_hat = X @ self.weights_ + self.bias_
            error = y - y_hat

            # Data gradient (MSE only)
            grad_w_data = (-2 / n_samples) * (X.T @ error)
            grad_b      = (-2 / n_samples) * np.sum(error)

            # Gradient step (data term only)
            self.weights_ -= self.learning_rate * grad_w_data
            self.bias_    -= self.learning_rate * grad_b

            # Proximal operator for L1 (soft-thresholding).
            # Correctly drives small weights to exactly zero —
            # the subgradient sign(w) approach oscillates around
            # zero and never reaches it.
            threshold = self.learning_rate * self.alpha
            self.weights_ = np.sign(self.weights_) * np.maximum(
                0.0, np.abs(self.weights_) - threshold
            )

            self.loss_history_.append(self._mse(y, y_hat))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        return X @ self.weights_ + self.bias_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R²."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        return self._mse(y, self.predict(X))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(self.mse(X, y)))

    @property
    def n_zero_weights(self) -> int:
        """Number of weights driven to (near) zero by Lasso."""
        if self.weights_ is None:
            return 0
        return int(np.sum(np.abs(self.weights_) < 1e-4))
