"""Shared test fixtures and utilities for FIPS tests."""

import numpy as np
import pandas as pd
import pytest

from fips.vector import Block, Vector


@pytest.fixture
def simple_prior():
    """Provide a simple prior state vector."""
    return pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.Index(["state_0", "state_1", "state_2"], name="state_id"),
        name="prior",
    )


@pytest.fixture
def simple_obs():
    """Provide a simple observation vector."""
    return pd.Series(
        [1.5, 2.5, 3.5, 4.5],
        index=pd.Index(["obs_0", "obs_1", "obs_2", "obs_3"], name="obs_id"),
        name="obs",
    )


@pytest.fixture
def simple_forward_operator():
    """Provide a simple forward operator matrix."""
    H = pd.DataFrame(
        np.array(
            [[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0], [0.25, 0.25, 0.25]]
        ),
        index=pd.Index(["obs_0", "obs_1", "obs_2", "obs_3"], name="obs_id"),
        columns=pd.Index(["state_0", "state_1", "state_2"], name="state_id"),
    )
    return H


@pytest.fixture
def simple_prior_error():
    """Provide a simple prior error covariance matrix."""
    idx = pd.Index(["state_0", "state_1", "state_2"], name="state_id")
    S_a = pd.DataFrame(
        np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.2], [0.0, 0.2, 1.0]]),
        index=idx,
        columns=idx,
    )
    return S_a


@pytest.fixture
def simple_modeldata_mismatch():
    """Provide a simple model-data mismatch covariance matrix."""
    idx = pd.Index(["obs_0", "obs_1", "obs_2", "obs_3"], name="obs_id")
    S_z = pd.DataFrame(
        np.eye(4) * 0.5,
        index=idx,
        columns=idx,
    )
    return S_z


@pytest.fixture
def simple_vector_multi_block():
    """Provide a simple multi-block vector."""
    block1 = Block(
        pd.Series([1.0, 2.0], index=pd.Index(["x", "y"], name="dim"), name="state_a")
    )
    block2 = Block(
        pd.Series(
            [3.0, 4.0, 5.0], index=pd.Index(["a", "b", "c"], name="dim"), name="state_b"
        )
    )
    return Vector(data=[block1, block2], name="state")


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
        name="prior",
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
