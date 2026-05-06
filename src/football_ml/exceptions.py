"""
Custom exceptions for the football_ml package.
================================================
Provides clear, specific error messages instead of generic
Python errors. Import and raise these in algorithm classes.
"""


class NotFittedError(Exception):
    """
    Raised when predict/transform is called before fit.

    Example
    -------
    >>> model = LogisticRegression()
    >>> model.predict(X)
    NotFittedError: LogisticRegression is not fitted yet.
    Call fit(X, y) before predict().
    """

    def __init__(self, class_name: str, method: str = "predict"):
        super().__init__(
            f"{class_name} is not fitted yet. "
            f"Call fit() before {method}()."
        )


class InvalidParameterError(Exception):
    """
    Raised when a hyperparameter value is invalid.

    Example
    -------
    >>> KMeans(k=0)
    InvalidParameterError: k must be >= 1, got 0.
    """

    def __init__(self, param: str, value, constraint: str):
        super().__init__(
            f"Invalid value for '{param}': got {value!r}. "
            f"Expected {constraint}."
        )


class ConvergenceWarning(UserWarning):
    """
    Issued when an iterative algorithm does not converge
    within the allowed number of iterations.
    """
    pass
