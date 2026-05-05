"""
Hierarchical (Agglomerative) Clustering
=========================================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


class HierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering.

    Builds a cluster tree (dendrogram) bottom-up:
      - Start: every point is its own cluster
      - Repeatedly merge the two closest clusters
      - Stop: everything is one cluster

    The full merge history is stored so you can cut the tree at any
    level to get any number of clusters — no need to specify K upfront.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to extract from the dendrogram. Default 3.
    linkage : str
        How to measure distance between clusters:
        'single'   — minimum pairwise distance (chaining tendency)
        'complete' — maximum pairwise distance (compact clusters)
        'average'  — mean pairwise distance (good balance)
        'ward'     — minimize within-cluster variance increase (most
                     common, similar to K-Means behavior)
        Default 'ward'.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_samples,)
        Cluster assignment for each sample.
    linkage_matrix_ : np.ndarray, shape (n_samples-1, 4)
        The merge history. Each row: [cluster_i, cluster_j, distance, size].
        Same format as scipy's linkage matrix — compatible with
        scipy's dendrogram plotting function.
    n_clusters_ : int
        Number of clusters extracted.
    """

    def __init__(self, n_clusters: int = 3, linkage: str = 'ward'):
        self.n_clusters = n_clusters
        self.linkage    = linkage

        self.labels_         = None
        self.linkage_matrix_ = None
        self.n_clusters_     = n_clusters

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute full Euclidean pairwise distance matrix."""
        sq = np.sum(X ** 2, axis=1, keepdims=True)
        dist_sq = sq + sq.T - 2 * (X @ X.T)
        return np.sqrt(np.maximum(dist_sq, 0))

    def _cluster_distance(
        self,
        ci: list,
        cj: list,
        X: np.ndarray,
        D: np.ndarray,
    ) -> float:
        """
        Distance between two clusters using the chosen linkage method.

        Parameters
        ----------
        ci, cj : lists of point indices in each cluster
        X      : original data (needed for Ward)
        D      : pairwise distance matrix (needed for others)
        """
        if self.linkage == 'single':
            return float(D[np.ix_(ci, cj)].min())

        if self.linkage == 'complete':
            return float(D[np.ix_(ci, cj)].max())

        if self.linkage == 'average':
            return float(D[np.ix_(ci, cj)].mean())

        if self.linkage == 'ward':
            # Ward: increase in total within-cluster variance from merging
            # = n_i * n_j / (n_i + n_j) * ||mean_i - mean_j||^2
            ni, nj = len(ci), len(cj)
            mean_i = X[ci].mean(axis=0)
            mean_j = X[cj].mean(axis=0)
            return float(ni * nj / (ni + nj) * np.sum((mean_i - mean_j) ** 2))

        raise ValueError(f"Unknown linkage: '{self.linkage}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "HierarchicalClustering":
        """
        Build the full dendrogram via agglomerative clustering.

        Algorithm:
          1. Initialize: each point is its own cluster
          2. Compute pairwise distances between all clusters
          3. Find the two closest clusters and merge them
          4. Record the merge in the linkage matrix
          5. Repeat until one cluster remains

        After building the full tree, cut at the level that produces
        self.n_clusters clusters and assign labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        n = len(X)
        D = self._pairwise_distances(X)

        # Each cluster starts as a single point
        # clusters[i] = list of original point indices in cluster i
        clusters = {i: [i] for i in range(n)}

        # Next available cluster ID (merged clusters get new IDs)
        next_id = n

        linkage_rows = []

        # Active cluster IDs
        active = list(range(n))

        for _ in range(n - 1):
            # Find the two closest active clusters
            best_dist = float('inf')
            best_i = best_j = -1

            for idx_a in range(len(active)):
                for idx_b in range(idx_a + 1, len(active)):
                    ci_id = active[idx_a]
                    cj_id = active[idx_b]
                    dist  = self._cluster_distance(
                        clusters[ci_id], clusters[cj_id], X, D
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_i, best_j = ci_id, cj_id

            # Merge the two closest clusters
            new_cluster = clusters[best_i] + clusters[best_j]
            size = len(new_cluster)

            # Record merge
            linkage_rows.append([
                float(best_i),
                float(best_j),
                best_dist,
                float(size),
            ])

            # Update data structures
            clusters[next_id] = new_cluster
            active.remove(best_i)
            active.remove(best_j)
            active.append(next_id)
            next_id += 1

        self.linkage_matrix_ = np.array(linkage_rows)

        # Cut the tree to get n_clusters labels
        self.labels_ = self._cut_tree(n)
        self.n_clusters_ = len(np.unique(self.labels_))

        return self

    def _cut_tree(self, n: int) -> np.ndarray:
        """
        Extract cluster labels by cutting the dendrogram.

        We want n_clusters clusters. The dendrogram has n-1 merges.
        Cutting after (n - n_clusters) merges gives n_clusters clusters.

        Works by replaying the merge history and stopping early.
        """
        # Start: each point in its own cluster
        labels = np.arange(n)

        # Re-label using the merge history
        # After (n - n_clusters) merges, we stop
        n_merges = n - self.n_clusters

        for merge_idx in range(n_merges):
            row = self.linkage_matrix_[merge_idx]
            ci_id = int(row[0])
            cj_id = int(row[1])
            new_id = n + merge_idx

            # Relabel all points in cluster cj to new_id
            # then all points in cluster ci to new_id
            labels[labels == cj_id] = new_id
            labels[labels == ci_id] = new_id

        # Remap to 0-indexed integers
        unique = np.unique(labels)
        remap  = {old: new for new, old in enumerate(unique)}
        return np.array([remap[l] for l in labels])

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).labels_

    def get_dendrogram_data(self) -> np.ndarray:
        """
        Return the linkage matrix for plotting with scipy's dendrogram.

        Compatible with scipy.cluster.hierarchy.dendrogram(Z).

        Returns
        -------
        np.ndarray, shape (n_samples-1, 4)
        """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Call fit() before get_dendrogram_data().")
        return self.linkage_matrix_
