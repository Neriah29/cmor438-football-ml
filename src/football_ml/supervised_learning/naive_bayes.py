"""
Gaussian Naïve Bayes Classifier
=================================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naïve Bayes Classifier.

    Assumes each feature follows a normal (Gaussian) distribution
    within each class. Estimates class-conditional means and variances
    from training data, then applies Bayes' theorem at prediction time.

    The 'naïve' assumption: all features are conditionally independent
    given the class label. This is almost never true in practice, but
    the classifier is surprisingly robust despite it.

    Training is instant — no gradient descent, no iterations.
    Just compute sufficient statistics (mean, variance) per class.

    Parameters
    ----------
    var_smoothing : float
        Small value added to variances for numerical stability.
        Prevents division by zero for constant features. Default 1e-9.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels seen during fit.
    class_priors_ : dict
        Log prior probability of each class: log P(class).
    means_ : dict
        Per-class feature means. means_[c] has shape (n_features,).
    variances_ : dict
        Per-class feature variances. variances_[c] has shape (n_features,).
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing  = var_smoothing
        self.classes_       = None
        self.class_priors_  = {}
        self.means_         = {}
        self.variances_     = {}

    # ------------------------------------------------------------------
    # Gaussian log-probability
    # ------------------------------------------------------------------

    def _log_gaussian(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """
        Log probability density of x under N(mean, var).

        Using log probabilities throughout prevents underflow — multiplying
        many small probabilities together quickly reaches 0 in float64.
        Adding logs is numerically stable.

            log N(x; μ, σ²) = -0.5 * log(2π σ²) - (x - μ)² / (2σ²)

        Parameters
        ----------
        x    : np.ndarray, shape (n_features,) — one sample's features
        mean : np.ndarray, shape (n_features,) — class means
        var  : np.ndarray, shape (n_features,) — class variances

        Returns
        -------
        np.ndarray, shape (n_features,) — log probability per feature
        """
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        """
        Estimate class priors, feature means, and feature variances.

        This is the entire training procedure:
          1. For each class: compute P(class) = count / total
          2. For each class and each feature: compute mean and variance
             of that feature among samples belonging to that class

        No optimization, no epochs — just statistics from data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — binary {0, 1}

        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)
        n_samples = len(y)

        for c in self.classes_:
            X_c = X[y == c]

            # Prior: log P(class) — how common is this class?
            self.class_priors_[c] = np.log(len(X_c) / n_samples)

            # Likelihood parameters: mean and variance per feature
            self.means_[c]     = X_c.mean(axis=0)
            self.variances_[c] = X_c.var(axis=0) + self.var_smoothing

        return self

    def _log_posterior(self, x: np.ndarray) -> dict:
        """
        Compute unnormalized log posterior for each class.

        log P(class | x) ∝ log P(class) + sum_j log P(x_j | class)

        The sum of log Gaussians over all features is the key computation.
        We don't need to normalize (divide by P(x)) because we only
        care about which class has the highest posterior.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)

        Returns
        -------
        dict mapping class → log posterior score
        """
        posteriors = {}
        for c in self.classes_:
            log_prior     = self.class_priors_[c]
            log_likelihood = self._log_gaussian(x, self.means_[c], self.variances_[c]).sum()
            posteriors[c]  = log_prior + log_likelihood
        return posteriors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels — argmax of posterior probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in {0, 1}
        """
        if self.classes_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.array([
            max(self._log_posterior(x), key=self._log_posterior(x).get)
            for x in X
        ])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for class 1.

        Converts log posteriors to normalized probabilities using
        the log-sum-exp trick for numerical stability.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in (0, 1)
        """
        if self.classes_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        probs = []
        for x in X:
            posteriors = self._log_posterior(x)
            log_scores = np.array([posteriors[c] for c in self.classes_])

            # Log-sum-exp for numerical stability
            log_scores -= log_scores.max()
            scores = np.exp(log_scores)
            scores /= scores.sum()

            # Return probability for class 1
            class_1_idx = list(self.classes_).index(1)
            probs.append(scores[class_1_idx])

        return np.array(probs)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == y))
