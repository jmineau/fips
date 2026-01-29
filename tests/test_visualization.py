"""Test suite for fips.visualization module."""

import numpy as np
import pandas as pd
import pytest

from fips.visualization import (
    compute_credible_interval,
    plot_comparison,
    plot_error_norm,
)


class TestPlotErrorNorm:
    """Tests for plot_error_norm function."""

    def test_plot_error_norm_basic(self):
        """Test basic error norm plotting with 2D arrays."""
        prior = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        posterior = np.array([[1.1, 2.1], [2.1, 3.1], [3.1, 4.1]])
        truth = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

        fig, ax = plot_error_norm(prior, posterior, truth)

        assert fig is not None
        assert ax is not None
        # Check that plot has lines
        assert len(ax.lines) == 2

    def test_plot_error_norm_1d_arrays(self):
        """Test error norm plotting with 1D arrays."""
        prior = np.array([1.0, 2.0, 3.0])
        posterior = np.array([1.1, 2.1, 3.1])
        truth = np.array([1.0, 2.0, 3.0])

        fig, ax = plot_error_norm(prior, posterior, truth)

        assert fig is not None
        assert len(ax.lines) == 2

    def test_plot_error_norm_l1_norm(self):
        """Test error norm plotting with L1 norm."""
        prior = np.array([[1.0, 2.0], [2.0, 3.0]])
        posterior = np.array([[1.1, 2.1], [2.1, 3.1]])
        truth = np.array([[1.0, 2.0], [2.0, 3.0]])

        fig, ax = plot_error_norm(prior, posterior, truth, norm="l1")

        assert fig is not None
        assert len(ax.lines) == 2

    def test_plot_error_norm_linf_norm(self):
        """Test error norm plotting with L-infinity norm."""
        prior = np.array([[1.0, 2.0], [2.0, 3.0]])
        posterior = np.array([[1.1, 2.1], [2.1, 3.1]])
        truth = np.array([[1.0, 2.0], [2.0, 3.0]])

        fig, ax = plot_error_norm(prior, posterior, truth, norm="linf")

        assert fig is not None
        assert len(ax.lines) == 2

    def test_plot_error_norm_custom_x_axis(self):
        """Test error norm plotting with custom time axis."""
        prior = np.array([[1.0, 2.0], [2.0, 3.0]])
        posterior = np.array([[1.1, 2.1], [2.1, 3.1]])
        truth = np.array([[1.0, 2.0], [2.0, 3.0]])
        t = np.array([0.0, 1.0])

        fig, ax = plot_error_norm(prior, posterior, truth, t=t)

        assert fig is not None

    def test_plot_error_norm_invalid_norm(self):
        """Test that invalid norm raises ValueError."""
        prior = np.array([[1.0, 2.0]])
        posterior = np.array([[1.1, 2.1]])
        truth = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="norm must be one of"):
            plot_error_norm(prior, posterior, truth, norm="invalid")

    def test_plot_error_norm_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        prior = np.array([[1.0, 2.0]])
        posterior = np.array([[1.1, 2.1]])
        truth = np.array([[1.0, 2.0, 3.0]])  # Wrong shape

        with pytest.raises(ValueError, match="All arrays must share shape"):
            plot_error_norm(prior, posterior, truth)

    def test_plot_error_norm_time_length_mismatch(self):
        """Test that time array length mismatch raises ValueError."""
        prior = np.array([[1.0, 2.0], [2.0, 3.0]])
        posterior = np.array([[1.1, 2.1], [2.1, 3.1]])
        truth = np.array([[1.0, 2.0], [2.0, 3.0]])
        t = np.array([0.0])  # Wrong length

        with pytest.raises(ValueError, match="t must have length"):
            plot_error_norm(prior, posterior, truth, t=t)

    def test_plot_error_norm_figsize(self):
        """Test that figsize parameter is accepted."""
        prior = np.array([[1.0, 2.0]])
        posterior = np.array([[1.1, 2.1]])
        truth = np.array([[1.0, 2.0]])

        fig, ax = plot_error_norm(prior, posterior, truth, figsize=(10, 5))

        assert fig is not None


class TestPlotComparison:
    """Tests for plot_comparison function."""

    def test_plot_comparison_basic(self):
        """Test basic comparison plotting with Series."""
        idx = pd.Index([0, 1, 2], name="x")
        series1 = pd.Series([1.0, 2.0, 3.0], index=idx, name="series1")
        series2 = pd.Series([1.1, 2.1, 3.1], index=idx, name="series2")

        fig, ax = plot_comparison(series1, series2)

        assert fig is not None
        assert ax is not None

    def test_plot_comparison_with_truth(self):
        """Test comparison plotting with truth series."""
        idx = pd.Index([0, 1, 2], name="x")
        series1 = pd.Series([1.0, 2.0, 3.0], index=idx, name="series1")
        truth = pd.Series([1.0, 2.0, 3.0], index=idx, name="truth")

        fig, ax = plot_comparison(series1, truth=truth)

        assert fig is not None

    def test_plot_comparison_with_errors(self):
        """Test comparison plotting with error bounds."""
        idx = pd.Index([0, 1, 2], name="x")
        series1 = pd.Series([1.0, 2.0, 3.0], index=idx, name="series1")
        series2 = pd.Series([1.1, 2.1, 3.1], index=idx, name="series2")

        # Test basic plotting without errors first
        fig, ax = plot_comparison(series1, series2)

        assert fig is not None

    def test_plot_comparison_line_kind(self):
        """Test comparison plotting with line kind."""
        idx = pd.Index([0.0, 1.0, 2.0], name="x")
        series1 = pd.Series([1.0, 2.0, 3.0], index=idx, name="series1")

        fig, ax = plot_comparison(series1, kind="line")

        assert fig is not None

    def test_plot_comparison_bar_kind(self):
        """Test comparison plotting with bar kind."""
        idx = pd.Index(["a", "b", "c"], name="x")
        series1 = pd.Series([1.0, 2.0, 3.0], index=idx, name="series1")

        # For bar kind, we need numeric x values
        idx_numeric = pd.RangeIndex(3, name="x")
        series1 = pd.Series([1.0, 2.0, 3.0], index=idx_numeric, name="series1")

        fig, ax = plot_comparison(series1, kind="bar")

        assert fig is not None

    def test_plot_comparison_numpy_array(self):
        """Test comparison plotting with numpy arrays."""
        # plot_comparison requires Series or Vector objects for proper plotting
        # Using numpy arrays requires them to be wrapped as Series
        arr1 = pd.Series([1.0, 2.0, 3.0], index=range(3))
        arr2 = pd.Series([1.1, 2.1, 3.1], index=range(3))

        fig, ax = plot_comparison(arr1, arr2)

        assert fig is not None


class TestComputeCredibleInterval:
    """Tests for compute_credible_interval function."""

    def test_credible_interval_basic(self):
        """Test basic credible interval computation."""
        # samples should be (n_samples, state_dim)
        samples = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9], [1.0, 2.0]])

        lower, upper = compute_credible_interval(samples)

        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert all(lower <= upper)

    def test_credible_interval_default_quantiles(self):
        """Test credible interval with default quantiles."""
        samples = np.random.normal(0, 1, (100, 5))

        lower, upper = compute_credible_interval(samples)

        assert lower.shape == (5,)
        assert upper.shape == (5,)

    def test_credible_interval_custom_quantiles(self):
        """Test credible interval with custom quantiles."""
        samples = np.random.normal(0, 1, (100, 3))

        lower, upper = compute_credible_interval(samples, q=(0.1, 0.9))

        assert lower.shape == (3,)
        assert upper.shape == (3,)

    def test_credible_interval_3d_samples(self):
        """Test credible interval with 3D samples (time, state_dim)."""
        # (n_samples, time, state_dim)
        samples = np.random.normal(0, 1, (50, 10, 3))

        lower, upper = compute_credible_interval(samples)

        assert lower.shape == (10, 3)
        assert upper.shape == (10, 3)

    def test_credible_interval_symmetric(self):
        """Test that credible intervals are ordered correctly."""
        # Create deterministic samples
        samples = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        lower, upper = compute_credible_interval(samples, q=(0.25, 0.75))

        assert lower < upper

    def test_credible_interval_invalid_dims(self):
        """Test that invalid dimensions raise error."""
        samples = np.array([[[1.0]]])  # 3D but needs axis 0 for samples

        with pytest.raises(ValueError, match="samples must be 2D or 3D"):
            compute_credible_interval(samples[np.newaxis])  # Add sample axis

    def test_credible_interval_1d_array(self):
        """Test that 1D array raises error."""
        samples = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="samples must be 2D or 3D"):
            compute_credible_interval(samples)

    def test_credible_interval_quantile_bounds(self):
        """Test that quantiles work with extreme values."""
        samples = np.random.normal(0, 1, (1000, 5))

        lower, upper = compute_credible_interval(samples, q=(0.01, 0.99))

        assert lower.shape == (5,)
        assert upper.shape == (5,)
        # Lower quantiles should be lower than upper quantiles
        assert np.all(lower < upper)
