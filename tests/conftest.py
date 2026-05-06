"""
conftest.py — Shared pytest configuration for CMOR 438 test suite.

Provides:
  - Timeout marks so slow tests fail loudly
  - Shared fixtures reused across test files
  - Custom pytest markers

Usage in test files:
    @pytest.mark.slow          # expected to take up to 30s
    @pytest.mark.timeout(5)    # fails if test takes > 5 seconds
"""

import time
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (> 10s expected)")
    config.addinivalue_line("markers", "benchmark: mark test as a timing benchmark")


# ---------------------------------------------------------------------------
# Timeout enforcement
# ---------------------------------------------------------------------------

# Maximum allowed time per test (seconds) by category
TIMEOUT_UNIT      = 30    # standard unit tests
TIMEOUT_SLOW      = 120   # slow tests (SVM, DBSCAN, Hierarchical on larger data)

@pytest.fixture(autouse=True)
def enforce_timeout(request):
    """
    Automatically fail any test that exceeds its time budget.

    Standard tests: 30 seconds max
    Tests marked @pytest.mark.slow: 120 seconds max
    """
    limit = TIMEOUT_SLOW if request.node.get_closest_marker('slow') else TIMEOUT_UNIT
    start = time.time()
    yield
    elapsed = time.time() - start
    if elapsed > limit:
        pytest.fail(
            f"Test took {elapsed:.2f}s — exceeded {limit}s limit. "
            f"Consider using a smaller dataset or marking with @pytest.mark.slow."
        )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def rng():
    """Shared random generator — session-scoped for consistency."""
    return np.random.default_rng(42)


@pytest.fixture(scope='session')
def two_blobs():
    """Two well-separated Gaussian blobs — reused across many test files."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[-3, -3], scale=0.5, size=(60, 2))
    X1 = rng.normal(loc=[3,  3],  scale=0.5, size=(60, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


@pytest.fixture(scope='session')
def three_blobs_unlabeled():
    """Three unlabeled clusters for unsupervised tests."""
    rng = np.random.default_rng(1)
    X0 = rng.normal(loc=[0,  0],  scale=0.3, size=(40, 2))
    X1 = rng.normal(loc=[5,  0],  scale=0.3, size=(40, 2))
    X2 = rng.normal(loc=[2.5, 4], scale=0.3, size=(40, 2))
    return np.vstack([X0, X1, X2])


@pytest.fixture(scope='session')
def noisy_binary():
    """Noisy binary classification dataset — 4 features."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 4))
    logits = 2 * X[:, 0] + X[:, 1] + rng.normal(scale=0.5, size=300)
    y = (logits > 0).astype(int)
    return X, y
