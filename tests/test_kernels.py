"""Tests for fips kernels module."""

import numpy as np
import pandas as pd

from fips.kernels import ConstantCorrelation, GridTimeDecay, RaggedTimeDecay


def test_constant_correlation():
    """Test ConstantCorrelation factory."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    kernel = ConstantCorrelation()
    corr = kernel(df)

    assert corr.shape == (3, 3)
    assert np.all(corr == 1.0)


def test_grid_time_decay():
    """Test GridTimeDecay factory."""
    times = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    # 1 day scale. Distance 1 day -> exp(-1) ~= 0.367
    kernel_func = GridTimeDecay(scale="1D")
    cov = kernel_func(times)

    assert cov.shape == (3, 3)
    assert np.isclose(cov[0, 0], 1.0)
    assert np.isclose(cov[0, 1], np.exp(-1.0))
    assert np.isclose(cov[0, 2], np.exp(-2.0))


def test_ragged_time_decay():
    """Test RaggedTimeDecay factory."""
    df = pd.DataFrame(
        {"time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])}
    )

    kernel_func = RaggedTimeDecay(time_dim="time", scale="1D")
    cov = kernel_func(df)

    assert cov.shape == (3, 3)
    assert np.isclose(cov[0, 1], np.exp(-1.0))
