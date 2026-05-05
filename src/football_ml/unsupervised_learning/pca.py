"""
Principal Component Analysis (PCA)
=====================================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


class PCA:
    """
    Principal Component Analysis.

    Finds the directions of maximum variance in the data (principal
    components) and projects data onto a lower-dimensional subspace
    defined by the top K components.

    Uses eigendecomposition of the covariance matrix — no iterative
    optimization needed. Training is a single closed-form computation.

    Parameters
    ----------
    n_components : int or None
        Number of principal components to keep.
        None = keep all components.

    Attributes
    ----------
    components_ : np.ndarray, shape (n_components, n_features)
        Principal component directions (eigenvectors), sorted by
        descending explained variance. Each row is one component.
    explained_variance_ : np.ndarray, shape (n_components,)
        Variance explained by each component (eigenvalues).
    explained_variance_ratio_ : np.ndarray, shape (n_components,)
        Fraction of total variance explained by each component.
    cumulative_variance_ratio_ : np.ndarray, shape (n_components,)
        Cumulative explained variance ratio.
    mean_ : np.ndarray, shape (n_features,)
        Per-feature mean computed during fit (used to center data).
    n_features_ : int
        Number of features seen during fit.
    """

    def __init__(self, n_components: int = None):
        self.n_components = n_components

        self.components_               = None
        self.explained_variance_       = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_= None
        self.mean_                     = None
        self.n_features_               = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Compute principal components from training data.

        Steps:
          1. Center the data (subtract mean)
          2. Compute covariance matrix: C = X^T X / (n-1)
          3. Eigendecompose C → eigenvectors and eigenvalues
          4. Sort by descending eigenvalue
          5. Keep top n_components

        No iterative optimization — this is a closed-form solution.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Step 1: center
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Step 2: covariance matrix
        # Divide by (n-1) for unbiased estimate
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        # Step 3: eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns eigenvectors as columns, eigenvalues may be in
        # ascending order — sort descending

        # Step 4: sort descending by eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]   # columns = eigenvectors

        # Clip tiny negative eigenvalues from numerical noise
        eigenvalues = np.maximum(eigenvalues, 0)

        # Step 5: keep top n_components
        k = self.n_components if self.n_components is not None else n_features
        k = min(k, n_features)

        self.components_         = eigenvectors[:, :k].T   # (k, n_features)
        self.explained_variance_ = eigenvalues[:k]

        total_var = eigenvalues.sum()
        if total_var > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
        else:
            self.explained_variance_ratio_ = np.zeros(k)

        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto the principal components.

        Subtracts the training mean, then multiplies by the component
        matrix — each row of the result is one sample's coordinates
        in the new lower-dimensional space.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and return transformed data."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from the reduced representation.

        Projects back from component space to original feature space.
        The reconstruction is approximate — information lost during
        dimensionality reduction cannot be recovered.

        Parameters
        ----------
        X_transformed : np.ndarray, shape (n_samples, n_components)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features) — approximate reconstruction
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return X_transformed @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Mean squared reconstruction error.

        Measures how much information is lost by the dimensionality
        reduction. Lower = better compression quality.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        float
        """
        X_reconstructed = self.inverse_transform(self.transform(X))
        return float(np.mean((X - X_reconstructed) ** 2))

    def n_components_for_variance(self, threshold: float = 0.95) -> int:
        """
        Return the minimum number of components needed to explain
        at least `threshold` fraction of total variance.

        Useful for choosing n_components automatically.

        Parameters
        ----------
        threshold : float
            Target explained variance ratio (e.g. 0.95 = 95%).

        Returns
        -------
        int
        """
        if self.cumulative_variance_ratio_ is None:
            raise RuntimeError("Call fit() first.")
        return int(np.searchsorted(self.cumulative_variance_ratio_, threshold) + 1)
