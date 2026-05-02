"""
Support Vector Machine (SVM)
==============================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Uses a simplified SMO-inspired coordinate descent solver.
Author: Neriah29
"""

import numpy as np


class SVM:
    """
    Binary Support Vector Machine with kernel support.

    Finds the maximum-margin hyperplane that separates two classes.
    Supports linear and RBF kernels. Uses a simplified coordinate
    descent (SMO-inspired) solver to find the optimal dual variables.

    Parameters
    ----------
    C : float
        Soft-margin penalty. Large C = strict margin (less tolerance
        for misclassifications). Small C = wide margin (more tolerance).
        Default 1.0.
    kernel : str
        'linear' or 'rbf'. Default 'rbf'.
    gamma : float or str
        RBF kernel parameter: exp(-gamma * ||x - x'||^2).
        'scale' (default) = 1 / (n_features * X.var())
        'auto'            = 1 / n_features
        float             = exact value
    max_iter : int
        Maximum number of optimization passes. Default 1000.
    tol : float
        Tolerance for stopping criterion. Default 1e-3.
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    support_vectors_ : np.ndarray
        Training samples that became support vectors (alpha > 0).
    support_labels_ : np.ndarray
        Labels of the support vectors.
    alphas_ : np.ndarray
        Dual variables for support vectors.
    bias_ : float
        Bias term of the decision boundary.
    n_support_ : int
        Number of support vectors.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma='scale',
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int = 42,
    ):
        self.C           = C
        self.kernel      = kernel
        self.gamma       = gamma
        self.max_iter    = max_iter
        self.tol         = tol
        self.random_state = random_state

        self.support_vectors_ = None
        self.support_labels_  = None
        self.alphas_          = None
        self.bias_            = 0.0
        self.n_support_       = 0
        self._gamma_val       = None

    # ------------------------------------------------------------------
    # Kernel functions
    # ------------------------------------------------------------------

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, float):
            return self.gamma
        if self.gamma == 'scale':
            var = X.var()
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        if self.gamma == 'auto':
            return 1.0 / X.shape[1]
        return 1.0

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix between X1 (n x d) and X2 (m x d).
        Returns (n x m) matrix.
        """
        if self.kernel == 'linear':
            return X1 @ X2.T

        if self.kernel == 'rbf':
            # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1·x2
            sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)   # (n, 1)
            sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)   # (m, 1)
            dist_sq = sq1 + sq2.T - 2 * (X1 @ X2.T)
            dist_sq = np.maximum(dist_sq, 0)                # numerical safety
            return np.exp(-self._gamma_val * dist_sq)

        raise ValueError(f"Unknown kernel: '{self.kernel}'")

    # ------------------------------------------------------------------
    # Decision function
    # ------------------------------------------------------------------

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw decision scores for samples in X.

        f(x) = sum_i alpha_i * y_i * K(x_i, x) + b

        Positive score → class +1, negative → class -1.
        """
        K = self._kernel(X, self.support_vectors_)           # (n_test, n_sv)
        return (K * (self.alphas_ * self.support_labels_)).sum(axis=1) + self.bias_

    # ------------------------------------------------------------------
    # SMO-inspired solver
    # ------------------------------------------------------------------

    def _solve(self, X: np.ndarray, y: np.ndarray):
        """
        Simplified Sequential Minimal Optimization (SMO).

        SMO solves the SVM dual problem by repeatedly picking two
        alphas and optimizing them analytically while keeping all
        others fixed. The dual problem is:

            maximize  sum(alpha_i) - 0.5 * sum_ij(alpha_i * alpha_j * y_i * y_j * K_ij)
            subject to  0 <= alpha_i <= C
                        sum(alpha_i * y_i) = 0

        Each alpha corresponds to one training sample. Samples with
        alpha > 0 become support vectors.
        """
        n = len(y)
        alphas = np.zeros(n)
        bias   = 0.0
        K = self._kernel(X, X)   # precompute full kernel matrix

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.max_iter):
            alpha_changed = 0

            for i in range(n):
                # Current prediction error for sample i
                f_i = float((alphas * y) @ K[i]) + bias
                E_i = f_i - y[i]

                # KKT violation check
                if not ((y[i] * E_i < -self.tol and alphas[i] < self.C) or
                        (y[i] * E_i >  self.tol and alphas[i] > 0)):
                    continue

                # Pick second alpha randomly (simple heuristic)
                j = rng.integers(0, n)
                while j == i:
                    j = rng.integers(0, n)

                f_j = float((alphas * y) @ K[j]) + bias
                E_j = f_j - y[j]

                alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                # Compute bounds L and H for alpha_j
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(self.C, self.C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - self.C)
                    H = min(self.C, alphas[i] + alphas[j])

                if L >= H:
                    continue

                # Second derivative of objective (eta)
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # Update alpha_j
                alphas[j] -= y[j] * (E_i - E_j) / eta
                alphas[j]  = np.clip(alphas[j], L, H)

                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                # Update alpha_i (to maintain sum(alpha*y) = 0)
                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                # Update bias
                b1 = (bias - E_i
                      - y[i] * (alphas[i] - alpha_i_old) * K[i, i]
                      - y[j] * (alphas[j] - alpha_j_old) * K[i, j])
                b2 = (bias - E_j
                      - y[i] * (alphas[i] - alpha_i_old) * K[i, j]
                      - y[j] * (alphas[j] - alpha_j_old) * K[j, j])

                if 0 < alphas[i] < self.C:
                    bias = b1
                elif 0 < alphas[j] < self.C:
                    bias = b2
                else:
                    bias = (b1 + b2) / 2

                alpha_changed += 1

            if alpha_changed == 0:
                break   # converged

        return alphas, bias

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """
        Fit the SVM.

        Labels must be {0, 1} — internally converted to {-1, +1}
        since SVM theory uses signed labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — binary {0, 1}

        Returns
        -------
        self
        """
        # Convert {0,1} → {-1,+1}
        y_signed = np.where(y == 1, 1.0, -1.0)

        self._gamma_val = self._resolve_gamma(X)

        alphas, bias = self._solve(X, y_signed)

        # Keep only support vectors (alpha > 0)
        sv_mask = alphas > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.support_labels_  = y_signed[sv_mask]
        self.alphas_          = alphas[sv_mask]
        self.bias_            = bias
        self.n_support_       = int(sv_mask.sum())

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in {0, 1}
        """
        if self.support_vectors_ is None:
            raise RuntimeError("Call fit() before predict().")
        scores = self._decision_function(X)
        # Convert {-1,+1} back to {0,1}
        return (scores >= 0).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw decision scores (signed distance from boundary).

        Positive = class 1 side, negative = class 0 side.
        Larger magnitude = more confident.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self.support_vectors_ is None:
            raise RuntimeError("Call fit() before decision_function().")
        return self._decision_function(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return approximate probability estimates via sigmoid on scores.

        Note: SVM is not a probabilistic model. This is an approximation
        (Platt scaling concept) — useful for ROC curves and comparison
        with other classifiers, but not as well-calibrated as Logistic
        Regression output.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in (0, 1)
        """
        scores = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == y))
