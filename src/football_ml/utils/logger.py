"""
Logging utility for the football_ml package.
=============================================
Provides a consistent logger for all algorithm classes.
Uses Python's standard logging module — no extra dependencies.

Usage in algorithm classes
--------------------------
    from football_ml.utils.logger import get_logger

    class MyModel:
        def __init__(self, verbose=False):
            self.verbose = verbose
            self._log = get_logger(__name__, verbose)

        def fit(self, X, y):
            self._log.info(f"Training on {X.shape[0]} samples, {X.shape[1]} features")
            for epoch in range(self.n_epochs):
                ...
                if epoch % 100 == 0:
                    self._log.debug(f"Epoch {epoch}: loss={loss:.4f}")
            self._log.info("Training complete.")
            return self

Log levels
----------
    DEBUG   — detailed per-epoch or per-iteration info
    INFO    — high-level training progress (start, end, key metrics)
    WARNING — non-fatal issues (convergence not reached, empty clusters)
    ERROR   — serious problems (will usually be followed by an exception)
"""

import logging
import sys


def get_logger(name: str, verbose: bool = False) -> logging.Logger:
    """
    Return a logger for the given module name.

    Parameters
    ----------
    name : str
        Module name — use __name__ from the calling module.
    verbose : bool
        If True, set level to DEBUG (detailed output).
        If False, set level to WARNING (silent during normal use).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%H:%M:%S'
            )
        )
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    return logger
