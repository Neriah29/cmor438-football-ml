"""
Decision Tree Classifier
=========================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


class _Node:
    """
    A single node in the Decision Tree.

    Can be either:
    - An internal node: stores a splitting question (feature + threshold)
    - A leaf node:      stores a final prediction (class label)
    """

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        gini=None,
        n_samples=None,
    ):
        self.feature   = feature      # which feature to split on
        self.threshold = threshold    # the value to compare against
        self.left      = left         # left subtree  (feature <= threshold)
        self.right     = right        # right subtree (feature >  threshold)
        self.value     = value        # prediction if this is a leaf node
        self.gini      = gini         # impurity at this node (for inspection)
        self.n_samples = n_samples    # number of samples that reached this node

    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """
    Binary Decision Tree Classifier using Gini impurity.

    Builds a tree by recursively finding the feature and threshold
    that best splits the data at each node. Stops when max_depth is
    reached, a node has fewer than min_samples_split samples, or a
    node is already pure (Gini = 0).

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree. None = grow until pure.
        Controls overfitting — deeper trees overfit more.
    min_samples_split : int
        Minimum samples required to split a node. Default 2.
    min_samples_leaf : int
        Minimum samples required in each leaf after a split. Default 1.

    Attributes
    ----------
    root_ : _Node
        The root node of the fitted tree.
    n_features_ : int
        Number of features seen during fit.
    feature_importances_ : np.ndarray
        How much each feature contributed to reducing impurity.
    """

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root_ = None
        self.n_features_ = None
        self.feature_importances_ = None

    # ------------------------------------------------------------------
    # Gini impurity
    # ------------------------------------------------------------------

    def _gini(self, y: np.ndarray) -> float:
        """
        Gini impurity of a set of labels.

            Gini = 1 - sum(p_i^2)

        Where p_i is the proportion of class i in the node.

        Gini = 0   → perfectly pure (all one class)
        Gini = 0.5 → maximally impure (50/50 split between two classes)

        We want splits that drive Gini toward 0.
        """
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return float(1.0 - np.sum(probs ** 2))

    def _weighted_gini(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Weighted average Gini after a split.

        Larger child nodes carry more weight — a split that creates
        one huge pure node and one tiny impure node is still good.
        """
        n = len(y_left) + len(y_right)
        return (len(y_left) / n) * self._gini(y_left) + \
               (len(y_right) / n) * self._gini(y_right)

    # ------------------------------------------------------------------
    # Finding the best split
    # ------------------------------------------------------------------

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Search every feature and every threshold to find the split
        that minimises weighted Gini impurity.

        Returns
        -------
        best_feature : int
        best_threshold : float
        best_gini : float
        """
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            # Candidate thresholds: midpoints between sorted unique values
            values = np.unique(X[:, feature_idx])
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask  = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # Skip if either side would be empty or too small
                if left_mask.sum() < self.min_samples_leaf or \
                   right_mask.sum() < self.min_samples_leaf:
                    continue

                gini = self._weighted_gini(y[left_mask], y[right_mask])
                if gini < best_gini:
                    best_gini      = gini
                    best_feature   = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    # ------------------------------------------------------------------
    # Tree building (recursive)
    # ------------------------------------------------------------------

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """
        Recursively build the tree.

        Stopping conditions (create a leaf):
          1. Reached max_depth
          2. Too few samples to split
          3. Node is already pure (Gini = 0)
          4. No valid split found
        """
        n_samples = len(y)
        current_gini = self._gini(y)

        # Majority class vote for this node's prediction
        values, counts = np.unique(y, return_counts=True)
        majority_class = int(values[np.argmax(counts)])

        # --- Stopping conditions ---
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           current_gini == 0.0:
            return _Node(value=majority_class, gini=current_gini, n_samples=n_samples)

        # --- Find best split ---
        feature, threshold, split_gini = self._best_split(X, y)

        if feature is None:
            # No valid split found
            return _Node(value=majority_class, gini=current_gini, n_samples=n_samples)

        # --- Accumulate feature importance ---
        # Information gain = impurity reduction weighted by fraction of samples
        gain = (n_samples / self._root_n_samples) * (current_gini - split_gini)
        self.feature_importances_[feature] += gain

        # --- Recurse ---
        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)

        return _Node(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
            gini=current_gini,
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Build the decision tree from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — binary {0, 1}

        Returns
        -------
        self
        """
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self._root_n_samples = len(y)   # needed for importance normalization

        self.root_ = self._build(X, y, depth=0)

        # Normalize importances to sum to 1
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        return self

    def _predict_one(self, x: np.ndarray, node: _Node) -> int:
        """Traverse the tree for a single sample."""
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

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
        if self.root_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.array([self._predict_one(x, self.root_) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for class 1.

        For a Decision Tree, this is the fraction of class-1 samples
        in the leaf node that a sample lands in.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in [0, 1]
        """
        if self.root_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return np.array([self._proba_one(x, self.root_) for x in X])

    def _proba_one(self, x: np.ndarray, node: _Node) -> float:
        """Traverse tree and return leaf class-1 fraction."""
        if node.is_leaf():
            return float(node.value)
        if x[node.feature] <= node.threshold:
            return self._proba_one(x, node.left)
        return self._proba_one(x, node.right)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == y))

    def get_depth(self) -> int:
        """Return the actual depth of the fitted tree."""
        def _depth(node):
            if node is None or node.is_leaf():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root_)

    def get_n_leaves(self) -> int:
        """Return the number of leaf nodes."""
        def _count(node):
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            return _count(node.left) + _count(node.right)
        return _count(self.root_)
