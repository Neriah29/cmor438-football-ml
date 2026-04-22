"""
K-Nearest Neighbors (KNN) Classifier
======================================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.

    A lazy learner — fit() simply stores the training data.
    All computation happens at prediction time: for each new sample,
    find the K closest training points and take a majority vote.

    Parameters
    ----------
    k : int
        Number of neighbors to consult. Default 5.
    metric : str
        Distance metric — 'euclidean' or 'manhattan'. Default 'euclidean'.

    Attributes
    ----------
    X_train_ : np.ndarray
        Stored training features.
    y_train_ : np.ndarray
        Stored training labels.
    """

    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if metric not in ('euclidean', 'manhattan'):
            raise ValueError(f"metric must be 'euclidean' or 'manhattan', got '{metric}'")
        self.k = k
        self.metric = metric
        self.X_train_ = None
        self.y_train_ = None

    # ------------------------------------------------------------------
    # Distance functions
    # ------------------------------------------------------------------

    def _euclidean(self, x: np.ndarray) -> np.ndarray:
        """
        Euclidean distance from one point x to every training point.

        Euclidean = straight-line distance, like a ruler in feature space.
        For two points a and b with n features:
            d = sqrt( (a1-b1)² + (a2-b2)² + ... + (an-bn)² )

        We compute this for all training points at once using broadcasting.
        """
        diff = self.X_train_ - x          # shape (n_train, n_features)
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def _manhattan(self, x: np.ndarray) -> np.ndarray:
        """
        Manhattan distance from one point x to every training point.

        Manhattan = sum of absolute differences along each dimension.
        Like navigating a city grid — you can only move horizontally
        or vertically, not diagonally.
            d = |a1-b1| + |a2-b2| + ... + |an-bn|
        """
        diff = self.X_train_ - x
        return np.sum(np.abs(diff), axis=1)

    def _distances(self, x: np.ndarray) -> np.ndarray:
        """Dispatch to the correct distance function."""
        if self.metric == 'euclidean':
            return self._euclidean(x)
        return self._manhattan(x)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        'Train' the model — just store the data.

        There is no optimization, no gradient, no weight update.
        KNN is a lazy learner: all work is deferred to predict time.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for each sample in X.

        For each sample:
          1. Compute distance to every training point
          2. Find the K smallest distances (nearest neighbors)
          3. Look up their labels
          4. Return the majority label (most common vote)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before predict().")

        predictions = []
        for x in X:
            # Step 1: compute distances to all training points
            distances = self._distances(x)

            # Step 2: find indices of K nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Step 3: get their labels
            k_labels = self.y_train_[k_indices]

            # Step 4: majority vote
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for class 1.

        Computed as the fraction of K neighbors that belong to class 1.
        e.g. if 3 of 5 neighbors are class 1 → probability = 0.6

        This is a soft vote rather than a hard majority vote.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in [0, 1]
        """
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        probabilities = []
        for x in X:
            distances = self._distances(x)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train_[k_indices]
            prob = np.mean(k_labels == 1)
            probabilities.append(prob)

        return np.array(probabilities)

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
