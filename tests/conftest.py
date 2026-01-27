"""Shared test fixtures and utilities for FIPS tests."""

import pytest
import numpy as np
import pandas as pd

from fips.vectors import Vector, Block
from fips.matrices import CovarianceMatrix, ForwardOperator


@pytest.fixture
def simple_prior():
    """Simple prior state vector."""
    return pd.Series([1.0, 2.0, 3.0], 
                    index=['state_0', 'state_1', 'state_2'],
                    name='prior')


@pytest.fixture
def simple_obs():
    """Simple observation vector."""
    return pd.Series([1.5, 2.5, 3.5, 4.5],
                    index=['obs_0', 'obs_1', 'obs_2', 'obs_3'],
                    name='obs')


@pytest.fixture
def simple_forward_operator():
    """Simple forward operator matrix."""
    H = pd.DataFrame(
        np.array([
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.5],
            [0.0, 0.5, 1.0],
            [0.25, 0.25, 0.25]
        ]),
        index=['obs_0', 'obs_1', 'obs_2', 'obs_3'],
        columns=['state_0', 'state_1', 'state_2']
    )
    return H


@pytest.fixture
def simple_prior_error():
    """Simple prior error covariance matrix."""
    S_a = pd.DataFrame(
        np.array([
            [1.0, 0.2, 0.0],
            [0.2, 1.0, 0.2],
            [0.0, 0.2, 1.0]
        ]),
        index=['state_0', 'state_1', 'state_2'],
        columns=['state_0', 'state_1', 'state_2']
    )
    return S_a


@pytest.fixture
def simple_modeldata_mismatch():
    """Simple model-data mismatch covariance matrix."""
    S_z = pd.DataFrame(
        np.eye(4) * 0.5,
        index=['obs_0', 'obs_1', 'obs_2', 'obs_3'],
        columns=['obs_0', 'obs_1', 'obs_2', 'obs_3']
    )
    return S_z


@pytest.fixture
def simple_vector_multi_block():
    """Simple multi-block vector."""
    block1 = Block('state_a', pd.Series([1.0, 2.0], index=['x', 'y']))
    block2 = Block('state_b', pd.Series([3.0, 4.0, 5.0], index=['a', 'b', 'c']))
    return Vector([block1, block2])


@pytest.fixture
def random_state():
    """Random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def random_prior(random_state):
    """Randomly generated prior."""
    n = 10
    data = random_state.randn(n)
    return pd.Series(data, 
                    index=[f'state_{i}' for i in range(n)],
                    name='prior')


@pytest.fixture
def random_obs(random_state):
    """Randomly generated observations."""
    m = 15
    data = random_state.randn(m)
    return pd.Series(data,
                    index=[f'obs_{i}' for i in range(m)],
                    name='obs')


@pytest.fixture
def random_forward_operator(random_state):
    """Randomly generated forward operator."""
    H = random_state.randn(15, 10)
    return pd.DataFrame(
        H,
        index=[f'obs_{i}' for i in range(15)],
        columns=[f'state_{i}' for i in range(10)]
    )


@pytest.fixture
def random_prior_error(random_state):
    """Randomly generated prior error covariance."""
    A = random_state.randn(10, 10)
    S_a = A @ A.T + np.eye(10)
    return pd.DataFrame(
        S_a,
        index=[f'state_{i}' for i in range(10)],
        columns=[f'state_{i}' for i in range(10)]
    )


@pytest.fixture
def random_modeldata_mismatch(random_state):
    """Randomly generated model-data mismatch covariance."""
    A = random_state.randn(15, 15)
    S_z = A @ A.T + np.eye(15)
    return pd.DataFrame(
        S_z,
        index=[f'obs_{i}' for i in range(15)],
        columns=[f'obs_{i}' for i in range(15)]
    )
