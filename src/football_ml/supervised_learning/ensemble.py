"""
Ensemble Methods — Random Forest & Gradient Boosting
======================================================
From-scratch NumPy implementations for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np
from .decision_tree import DecisionTreeClassifier


# =============================================================================
# Minimal regression tree (used internally by GradientBoostingClassifier)
# =============================================================================

class _RegressionTree:
    """
    Shallow regression tree that stores mean target values in leaves.

    Used by GradientBoostingClassifier to fit continuous pseudo-residuals.
    DecisionTreeClassifier cannot be used here because it stores integer
    majority-class labels in leaves, which truncates float residuals to 0.
    """

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_ = None

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        n = len(y)
        for feature_idx in range(X.shape[1]):
            values = np.unique(X[:, feature_idx])
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask  = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < 1 or right_mask.sum() < 1:
                    continue

                mse = (left_mask.sum() / n) * float(np.var(y[left_mask])) + \
                      (right_mask.sum() / n) * float(np.var(y[right_mask]))

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if (self.max_depth is not None and depth >= self.max_depth) or \
                len(y) < self.min_samples_split:
            return {'leaf': True, 'value': float(np.mean(y))}

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return {'leaf': True, 'value': float(np.mean(y))}

        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left':  self._build(X[left_mask],  y[left_mask],  depth + 1),
            'right': self._build(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RegressionTree":
        self.root_ = self._build(X, y, 0)
        return self

    def _predict_one(self, x: np.ndarray, node: dict) -> float:
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root_) for x in X])


# =============================================================================
# Random Forest
# =============================================================================

class RandomForestClassifier:
    """
    Random Forest — Bagging ensemble of Decision Trees.

    Trains n_estimators trees, each on a bootstrap sample of the data
    and considering only max_features features at each split.
    Prediction = majority vote across all trees.

    Parameters
    ----------
    n_estimators : int
        Number of trees to train. Default 100.
    max_depth : int or None
        Maximum depth of each tree. Default None.
    max_features : str or int
        Features to consider at each split.
        'sqrt' (default) = sqrt(n_features) — standard for classification.
        'log2'           = log2(n_features)
        int              = exact number
    min_samples_split : int
        Minimum samples to split a node. Default 2.
    min_samples_leaf : int
        Minimum samples in each leaf. Default 1.
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    trees_ : list of DecisionTreeClassifier
        The fitted trees.
    feature_importances_ : np.ndarray
        Average feature importances across all trees.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        max_features: str = 'sqrt',
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.max_features     = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.random_state     = random_state

        self.trees_                = []
        self.feature_indices_      = []   # which features each tree used
        self.feature_importances_  = None

    def _get_max_features(self, n_features: int) -> int:
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        return n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """
        Train the forest.

        For each tree:
          1. Bootstrap sample — draw n_samples rows WITH replacement
          2. Feature subsample — pick max_features columns randomly
          3. Train a Decision Tree on this subset

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
        max_feat = self._get_max_features(n_features)

        self.trees_ = []
        self.feature_indices_ = []
        importances_sum = np.zeros(n_features)

        for _ in range(self.n_estimators):
            # Bootstrap sample (sample WITH replacement)
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            # Random feature subset
            feat_idx = rng.choice(n_features, size=max_feat, replace=False)
            feat_idx = np.sort(feat_idx)

            # Train tree on selected features
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_boot[:, feat_idx], y_boot)

            self.trees_.append(tree)
            self.feature_indices_.append(feat_idx)

            # Accumulate importances (mapped back to original feature indices)
            for local_i, global_i in enumerate(feat_idx):
                importances_sum[global_i] += tree.feature_importances_[local_i]

        # Average and normalize importances
        self.feature_importances_ = importances_sum / self.n_estimators
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return average predicted probability of class 1 across all trees.

        Soft voting — average the probability estimates rather than
        just taking majority class. More informative than hard voting.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in [0, 1]
        """
        if not self.trees_:
            raise RuntimeError("Call fit() before predict_proba().")

        proba_sum = np.zeros(X.shape[0])
        for tree, feat_idx in zip(self.trees_, self.feature_indices_):
            proba_sum += tree.predict_proba(X[:, feat_idx])
        return proba_sum / self.n_estimators

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels via majority vote.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in {0, 1}
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == y))


# =============================================================================
# Gradient Boosting
# =============================================================================

class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier using Decision Tree stumps.

    Builds trees sequentially. Each tree fits the pseudo-residuals —
    the gradient of the loss with respect to the current predictions.
    This is gradient descent in function space: instead of updating
    weights, we add a new tree at each step.

    Uses log-loss (binary cross-entropy) as the loss function.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (trees). Default 100.
    learning_rate : float
        Shrinkage factor — scales each tree's contribution.
        Smaller = more conservative, needs more trees. Default 0.1.
    max_depth : int
        Maximum depth of each tree. Shallow trees (2-5) work best
        for boosting — they correct one pattern at a time. Default 3.
    min_samples_split : int
        Minimum samples to split a node. Default 2.
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    trees_ : list of DecisionTreeClassifier
        The fitted trees (one per boosting round).
    init_pred_ : float
        Initial prediction (log-odds of base rate).
    loss_history_ : list of float
        Training loss after each boosting round.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: int = 42,
    ):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.random_state      = random_state

        self.trees_        = []
        self.init_pred_    = None
        self.loss_history_ = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _log_loss(self, y: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        """
        Train via gradient boosting.

        At each round:
          1. Compute pseudo-residuals = y - p  (gradient of log-loss)
          2. Fit a shallow tree to predict those residuals
          3. Add the tree's predictions (scaled by learning_rate) to
             the running log-odds score
          4. Convert to probability and record loss

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        # Initial prediction: log-odds of the base rate
        # (best constant prediction before seeing any features)
        base_rate = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.init_pred_ = float(np.log(base_rate / (1 - base_rate)))

        # Running log-odds scores for all training samples
        F = np.full(len(y), self.init_pred_)

        self.trees_ = []
        self.loss_history_ = []

        for _ in range(self.n_estimators):
            # Current probability estimates
            p = self._sigmoid(F)

            # Pseudo-residuals = negative gradient of log-loss
            # = how much each prediction needs to move up or down
            residuals = y - p

            # Fit a shallow regression tree to the residuals
            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X, residuals)

            # Update scores: add this tree's contribution
            # (tree predicts mean residual per leaf, not class labels)
            F += self.learning_rate * tree.predict(X)

            self.trees_.append(tree)

            # Record loss
            self.loss_history_.append(self._log_loss(y, self._sigmoid(F)))

        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw log-odds scores for samples in X."""
        if self.init_pred_ is None:
            raise RuntimeError("Call fit() before predict().")
        F = np.full(X.shape[0], self.init_pred_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted probability of class 1.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in (0, 1)
        """
        return self._sigmoid(self._decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in {0, 1}
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == y))
