"""Shared test fixtures and utilities for FIPS tests."""

import os

# Force a non-interactive matplotlib backend before any test imports pyplot, so
# the visualization tests never reach for an interactive backend (e.g. TkAgg)
# that may be missing on CI runners -- this caused flaky `_tkinter.TclError`
# failures on the Windows runners. matplotlib reads MPLBACKEND when it first
# selects a backend, and the viz modules import pyplot lazily, so setting it
# here is sufficient. Set via env var (rather than `matplotlib.use`) to avoid
# importing matplotlib in conftest, preserving its optional-dependency status.
# `setdefault` lets a developer still override the backend locally.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Pandas version compatibility utilities
# ---------------------------------------------------------------------------


def get_pandas_version():
    """Get pandas version as a tuple of integers."""
    return tuple(int(x) for x in pd.__version__.split(".")[:3])


@pytest.fixture
def pandas_version():
    """Fixture providing pandas version tuple."""
    return get_pandas_version()


def pandas_ge(major, minor=0, patch=0):
    """Check if pandas version is >= specified version."""
    version = get_pandas_version()
    return version >= (major, minor, patch)


def pandas_lt(major, minor=0, patch=0):
    """Check if pandas version is < specified version."""
    version = get_pandas_version()
    return version < (major, minor, patch)


@pytest.fixture
def random_state():
    """Random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def random_prior(random_state):
    """Randomly generated prior."""
    n = 10
    data = random_state.randn(n)
    return pd.Series(
        data,
        index=pd.Index([f"state_{i}" for i in range(n)], name="state_id"),
        name="state",
    )


@pytest.fixture
def random_obs(random_state):
    """Randomly generated observations."""
    m = 15
    data = random_state.randn(m)
    return pd.Series(
        data, index=pd.Index([f"obs_{i}" for i in range(m)], name="obs_id"), name="obs"
    )


@pytest.fixture
def random_forward_operator(random_state):
    """Randomly generated forward operator."""
    H = random_state.randn(15, 10)
    return pd.DataFrame(
        H,
        index=pd.Index([f"obs_{i}" for i in range(15)], name="obs_id"),
        columns=pd.Index([f"state_{i}" for i in range(10)], name="state_id"),
    )


@pytest.fixture
def random_prior_error(random_state):
    """Randomly generated prior error covariance."""
    A = random_state.randn(10, 10)
    S_a = A @ A.T + np.eye(10)
    idx = pd.Index([f"state_{i}" for i in range(10)], name="state_id")
    return pd.DataFrame(
        S_a,
        index=idx,
        columns=idx,
    )


@pytest.fixture
def random_modeldata_mismatch(random_state):
    """Randomly generated model-data mismatch covariance."""
    A = random_state.randn(15, 15)
    S_z = A @ A.T + np.eye(15)
    idx = pd.Index([f"obs_{i}" for i in range(15)], name="obs_id")
    return pd.DataFrame(
        S_z,
        index=idx,
        columns=idx,
    )
