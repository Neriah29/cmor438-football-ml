"""
DBSCAN — Density-Based Spatial Clustering of Applications with Noise
======================================================================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


class DBSCAN:
    """
    DBSCAN Clustering.

    Finds clusters as dense regions separated by sparse space.
    Does not require specifying K upfront — the number of clusters
    emerges from the data. Explicitly labels noise points as -1.

    Parameters
    ----------
    eps : float
        Neighborhood radius. Two points are neighbors if their
        distance is <= eps. Default 0.5.
    min_samples : int
        Minimum number of points within eps for a point to be
        considered a core point. Default 5.
    metric : str
        Distance metric — 'euclidean' or 'manhattan'. Default 'euclidean'.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_samples,)
        Cluster label for each point.
        -1 = noise, 0,1,2,... = cluster indices.
    core_sample_indices_ : np.ndarray
        Indices of core points in the training data.
    n_clusters_ : int
        Number of clusters found (excluding noise).
    n_noise_ : int
        Number of noise points.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean',
    ):
        self.eps         = eps
        self.min_samples = min_samples
        self.metric      = metric

        self.labels_              = None
        self.core_sample_indices_ = None
        self.n_clusters_          = None
        self.n_noise_             = None

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute full pairwise distance matrix.

        Returns (n x n) matrix where entry [i,j] is the distance
        between points i and j.
        """
        if self.metric == 'euclidean':
            sq = np.sum(X ** 2, axis=1, keepdims=True)
            dist_sq = sq + sq.T - 2 * (X @ X.T)
            dist_sq = np.maximum(dist_sq, 0)   # numerical safety
            return np.sqrt(dist_sq)

        if self.metric == 'manhattan':
            n = len(X)
            D = np.zeros((n, n))
            for i in range(n):
                D[i] = np.sum(np.abs(X - X[i]), axis=1)
            return D

        raise ValueError(f"Unknown metric: '{self.metric}'")

    def _get_neighbors(self, distances: np.ndarray, idx: int) -> np.ndarray:
        """
        Return indices of all points within eps of point idx.
        Excludes the point itself.
        """
        return np.where(
            (distances[idx] <= self.eps) & (np.arange(len(distances)) != idx)
        )[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Run DBSCAN on X.

        Algorithm:
          1. Compute all pairwise distances
          2. Identify core points (>= min_samples neighbors within eps)
          3. For each unvisited core point, expand a new cluster by
             recursively adding all density-reachable points
          4. Border points get assigned to a cluster but don't expand
          5. Remaining unvisited points are noise (-1)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        n = len(X)
        distances = self._pairwise_distances(X)

        # Find neighbors for every point
        neighbors = [self._get_neighbors(distances, i) for i in range(n)]

        # Identify core points
        is_core = np.array([len(nb) >= self.min_samples - 1 for nb in neighbors])

        # Initialize all labels as unvisited (-2 = unvisited, -1 = noise)
        labels = np.full(n, -2, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -2:
                continue   # already visited

            if not is_core[i]:
                labels[i] = -1   # tentatively noise
                continue

            # Start a new cluster from this core point
            labels[i] = cluster_id
            # Use a queue to expand the cluster
            queue = list(neighbors[i])

            while queue:
                j = queue.pop(0)

                if labels[j] == -1:
                    # Border point — add to cluster but don't expand
                    labels[j] = cluster_id
                    continue

                if labels[j] != -2:
                    continue   # already assigned

                labels[j] = cluster_id

                if is_core[j]:
                    # Core point — expand further
                    for nb in neighbors[j]:
                        if labels[nb] == -2 or labels[nb] == -1:
                            queue.append(nb)

            cluster_id += 1

        # Any remaining unvisited points are noise
        labels[labels == -2] = -1

        self.labels_              = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        self.n_clusters_          = cluster_id
        self.n_noise_             = int(np.sum(labels == -1))

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).labels_

    def silhouette_score(self, X: np.ndarray) -> float:
        """
        Compute mean silhouette score — excludes noise points.

        Only meaningful when there are at least 2 clusters and
        not too many noise points.

        Returns
        -------
        float — mean silhouette score, or 0.0 if not computable
        """
        if self.labels_ is None:
            raise RuntimeError("Call fit() before silhouette_score().")

        # Only score non-noise points
        mask = self.labels_ != -1
        if mask.sum() < 2 or self.n_clusters_ < 2:
            return 0.0

        X_valid  = X[mask]
        labels   = self.labels_[mask]
        unique   = np.unique(labels)
        scores   = []

        for i in range(len(X_valid)):
            same = labels == labels[i]
            same[i] = False

            if same.sum() == 0:
                scores.append(0.0)
                continue

            a = np.mean(np.sqrt(np.sum((X_valid[same] - X_valid[i]) ** 2, axis=1)))

            b = float('inf')
            for c in unique:
                if c == labels[i]:
                    continue
                other = labels == c
                if other.sum() == 0:
                    continue
                d = np.mean(np.sqrt(np.sum((X_valid[other] - X_valid[i]) ** 2, axis=1)))
                b = min(b, d)

            if b == float('inf'):
                scores.append(0.0)
            else:
                scores.append((b - a) / max(a, b))

        return float(np.mean(scores)) if scores else 0.0
