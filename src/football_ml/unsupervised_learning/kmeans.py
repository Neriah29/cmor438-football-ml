"""
K-Means Clustering
===================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


class KMeans:
    """
    K-Means Clustering.

    Partitions data into K clusters by iteratively assigning points
    to the nearest centroid and updating centroids to the mean of
    their assigned points. Minimizes total inertia (within-cluster
    sum of squared distances).

    Parameters
    ----------
    k : int
        Number of clusters. Must be specified upfront.
    max_iter : int
        Maximum number of assign-update cycles. Default 300.
    tol : float
        Convergence threshold — stop if centroids move less than
        this between iterations. Default 1e-4.
    n_init : int
        Number of times to run with different random initializations.
        Best result (lowest inertia) is kept. Default 10.
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    centroids_ : np.ndarray, shape (k, n_features)
        Final centroid positions after fitting.
    labels_ : np.ndarray, shape (n_samples,)
        Cluster assignment for each training sample.
    inertia_ : float
        Total within-cluster sum of squared distances (final run).
    n_iter_ : int
        Number of iterations until convergence (final run).
    inertia_history_ : list of float
        Inertia after each iteration of the best run.
    """

    def __init__(
        self,
        k: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: int = 42,
    ):
        self.k            = k
        self.max_iter     = max_iter
        self.tol          = tol
        self.n_init       = n_init
        self.random_state = random_state

        self.centroids_       = None
        self.labels_          = None
        self.inertia_         = None
        self.n_iter_          = None
        self.inertia_history_ = []

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        K-Means++ initialization — smarter than pure random.

        Instead of picking K random points, K-Means++ picks the first
        centroid randomly, then picks each subsequent centroid with
        probability proportional to its squared distance from the
        nearest existing centroid.

        This spreads centroids out across the data, leading to better
        convergence and lower final inertia than random initialization.
        """
        n_samples = X.shape[0]

        # Pick first centroid uniformly at random
        idx = rng.integers(0, n_samples)
        centroids = [X[idx]]

        for _ in range(self.k - 1):
            # Squared distance from each point to nearest centroid
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            # Sample next centroid proportional to distance squared
            probs = dists / dists.sum()
            idx = rng.choice(n_samples, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest centroid.

        Computes squared Euclidean distance from every point to every
        centroid and returns the index of the closest one.

        Returns
        -------
        np.ndarray, shape (n_samples,) — cluster index per sample
        """
        # distances[i, j] = squared distance from sample i to centroid j
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, k, d)
        distances = np.sum(diff ** 2, axis=2)                      # (n, k)
        return np.argmin(distances, axis=1)                        # (n,)

    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids to the mean of their assigned points.

        If a cluster becomes empty (no points assigned), reinitialize
        its centroid to a random point to avoid degenerate solutions.
        """
        centroids = np.zeros((self.k, X.shape[1]))
        for j in range(self.k):
            mask = labels == j
            if mask.sum() > 0:
                centroids[j] = X[mask].mean(axis=0)
            else:
                # Empty cluster — reinitialize to random point
                centroids[j] = X[np.random.randint(0, len(X))]
        return centroids

    def _inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Total within-cluster sum of squared distances.

        This is the objective K-Means minimizes. Lower = tighter clusters.
        """
        total = 0.0
        for j in range(self.k):
            mask = labels == j
            if mask.sum() > 0:
                total += np.sum((X[mask] - centroids[j]) ** 2)
        return float(total)

    def _run_once(self, X: np.ndarray, rng: np.random.Generator):
        """
        One full K-Means run from a single initialization.

        Returns centroids, labels, inertia, n_iter, inertia_history.
        """
        centroids = self._init_centroids(X, rng)
        history = []

        for i in range(self.max_iter):
            labels = self._assign(X, centroids)
            inertia = self._inertia(X, labels, centroids)
            history.append(inertia)

            new_centroids = self._update(X, labels)

            # Check convergence: how much did centroids move?
            shift = np.max(np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)))
            centroids = new_centroids

            if shift < self.tol:
                break

        labels = self._assign(X, centroids)
        inertia = self._inertia(X, labels, centroids)
        return centroids, labels, inertia, i + 1, history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Run K-Means n_init times and keep the best result.

        Multiple initializations are important because K-Means can
        converge to local minima — different starting centroids can
        produce very different final clusters. Running many times
        and keeping the lowest inertia gives a more reliable result.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        best_inertia = float('inf')
        best_result  = None

        for _ in range(self.n_init):
            result = self._run_once(X, rng)
            if result[2] < best_inertia:
                best_inertia = result[2]
                best_result  = result

        centroids, labels, inertia, n_iter, history = best_result
        self.centroids_       = centroids
        self.labels_          = labels
        self.inertia_         = inertia
        self.n_iter_          = n_iter
        self.inertia_history_ = history

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new samples to the nearest centroid.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — cluster indices
        """
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self._assign(X, self.centroids_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels for X."""
        return self.fit(X).labels_

    def silhouette_score(self, X: np.ndarray) -> float:
        """
        Compute the mean silhouette score — a measure of cluster quality.

        For each sample, the silhouette score is:
            s = (b - a) / max(a, b)

        Where:
            a = mean distance to other points in the SAME cluster
            b = mean distance to points in the NEAREST OTHER cluster

        s ranges from -1 to +1:
            +1 = sample is well inside its cluster, far from others
             0 = sample is on the boundary between clusters
            -1 = sample is probably in the wrong cluster

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        float — mean silhouette score across all samples
        """
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before silhouette_score().")

        labels = self.labels_
        n = len(X)
        scores = []

        for i in range(n):
            same_cluster = labels == labels[i]
            same_cluster[i] = False   # exclude self

            if same_cluster.sum() == 0:
                scores.append(0.0)
                continue

            # a: mean distance to own cluster
            a = np.mean(np.sqrt(np.sum((X[same_cluster] - X[i]) ** 2, axis=1)))

            # b: mean distance to nearest other cluster
            b = float('inf')
            for j in range(self.k):
                if j == labels[i]:
                    continue
                other = labels == j
                if other.sum() == 0:
                    continue
                mean_dist = np.mean(np.sqrt(np.sum((X[other] - X[i]) ** 2, axis=1)))
                b = min(b, mean_dist)

            if b == float('inf'):
                scores.append(0.0)
            else:
                scores.append((b - a) / max(a, b))

        return float(np.mean(scores))
