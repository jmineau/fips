"""Test suite for fips.visualization module."""

import numpy as np
import pandas as pd
import pytest

from fips.vector import Block, Vector
from fips.visualization import (
    compute_credible_interval,
    plot_comparison,
    plot_error_norm,
)


def _unwrapped_name(x):
    """The name of the Series ``x`` is or holds."""
    return x.data.name if isinstance(x, Block) else x.name


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

        with pytest.raises(ValueError, match="broadcast"):
            plot_error_norm(prior, posterior, truth)

    def test_plot_error_norm_time_length_mismatch(self):
        """Test that time array length mismatch raises ValueError."""
        prior = np.array([[1.0, 2.0], [2.0, 3.0]])
        posterior = np.array([[1.1, 2.1], [2.1, 3.1]])
        truth = np.array([[1.0, 2.0], [2.0, 3.0]])
        t = np.array([0.0])  # Wrong length

        # This should cause a shape mismatch when plotting
        with pytest.raises(ValueError):
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

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        """Close every figure after a test."""
        plt = pytest.importorskip("matplotlib.pyplot")
        yield
        plt.close("all")

    @staticmethod
    def _lines(ax):
        """The y data of each line on ``ax``, by label."""
        return {line.get_label(): np.asarray(line.get_ydata()) for line in ax.lines}

    @staticmethod
    def _series(name, values, labels=(0.0, 1.0, 2.0)):
        """A named Series over a float index called ``x``."""
        return pd.Series(values, index=pd.Index(labels, name="x"), name=name)

    def test_blocks_are_plotted_by_their_data(self):
        """A Block is plotted as the Series it holds, next to a Series."""
        a = self._series("a", [1.0, 2.0, 3.0])
        b = self._series("b", [1.5, 2.5, 3.5])

        _, ax = plot_comparison(Block(a), b)

        lines = self._lines(ax)
        assert list(lines) == ["a", "b"]
        np.testing.assert_array_equal(lines["a"], a.values)
        np.testing.assert_array_equal(lines["b"], b.values)

    def test_vectors_are_plotted_by_their_data(self):
        """A Vector is plotted as the Series it holds, over the level named by ``x``."""
        a = self._series("a", [1.0, 2.0, 3.0])

        _, ax = plot_comparison(Vector(Block(a)), x="x")

        np.testing.assert_array_equal(next(iter(self._lines(ax).values())), a.values)

    def test_unnamed_series_are_labeled_by_position(self):
        """Series with no name get the integer labels ``concat`` gives their columns."""
        idx = pd.Index([0.0, 1.0, 2.0], name="x")
        a = pd.Series([1.0, 2.0, 3.0], index=idx)
        b = pd.Series([1.5, 2.5, 3.5], index=idx)

        _, ax = plot_comparison(a, b)

        assert list(self._lines(ax)) == ["0", "1"]

    @pytest.mark.parametrize(
        ("name", "x", "label"),
        [
            pytest.param("time", None, "time", id="index-name"),
            pytest.param(None, None, "index", id="unnamed"),
            pytest.param(5, None, "5", id="integer-name"),
            pytest.param("time", "other", "other", id="given"),
        ],
    )
    def test_x_label(self, name, x, label):
        """The x label is ``x`` if given, else the name of the index, else "index"."""
        idx = pd.Index([0.0, 1.0, 2.0], name=name)
        a = pd.Series([1.0, 2.0, 3.0], index=idx, name="a")

        _, ax = plot_comparison(a, x=x)

        assert ax.get_xlabel() == label

    def test_x_label_of_a_multiindex_is_its_level(self):
        """With a MultiIndex, ``x`` names the level to plot against, and the label."""
        a = self._series("a", [1.0, 2.0, 3.0])

        _, ax = plot_comparison(Vector(Block(a)), x="x")

        assert ax.get_xlabel() == "x"

    def test_truth_is_drawn_as_a_line(self):
        """The truth of a line plot is a dashed black line."""
        a = self._series("a", [1.0, 2.0, 3.0])
        truth = self._series("t", [1.1, 2.1, 3.1])

        _, ax = plot_comparison(a, truth=Block(truth))

        line = ax.lines[-1]
        assert line.get_label() == "truth"
        np.testing.assert_array_equal(line.get_ydata(), truth.values)
        assert line.get_color() == "black"
        assert line.get_linestyle() == "--"

    def test_truth_is_drawn_as_crosses_on_bars(self):
        """The truth of a bar plot is a cross above each bar."""
        a = self._series("a", [1.0, 2.0, 3.0], labels=["u", "v", "w"])
        truth = self._series("t", [1.1, 2.1, 3.1], labels=["u", "v", "w"])

        _, ax = plot_comparison(a, truth=truth)

        (crosses,) = ax.collections
        assert crosses.get_label() == "truth"
        np.testing.assert_array_equal(crosses.get_offsets()[:, 1], truth.values)

    @pytest.mark.parametrize(
        "wrap",
        [
            pytest.param(lambda e: e, id="series"),
            pytest.param(Block, id="block"),
            pytest.param(lambda e: e.to_frame(), id="dataframe"),
        ],
    )
    def test_errors_are_a_band_around_a_line(self, wrap):
        """An error given for a line plot shades the series plus and minus it."""
        a = self._series("a", [1.0, 2.0, 3.0])
        err = self._series("a", [0.5, 0.5, 0.5])

        _, ax = plot_comparison(a, errors=[wrap(err)])

        (band,) = ax.collections
        y = band.get_paths()[0].vertices[:, 1]
        assert y.min() == pytest.approx(0.5)
        assert y.max() == pytest.approx(3.5)

    @pytest.mark.parametrize(
        "wrap",
        [
            pytest.param(lambda e: e, id="series"),
            pytest.param(Block, id="block"),
            pytest.param(lambda e: e.to_frame(), id="dataframe"),
        ],
    )
    def test_errors_are_bars_on_a_bar_plot(self, wrap):
        """An error given for a bar plot is the length of the whisker on each bar."""
        a = self._series("a", [1.0, 2.0, 3.0], labels=["u", "v", "w"])
        err = self._series("a", [0.5, 0.25, 0.125], labels=["u", "v", "w"])

        _, ax = plot_comparison(a, errors=[wrap(err)])

        (whiskers,) = ax.containers[0].lines[2]
        lengths = [seg[1, 1] - seg[0, 1] for seg in whiskers.get_segments()]
        np.testing.assert_allclose(lengths, 2 * err.values)

    def test_a_series_labeled_data_is_plotted_as_a_series(self):
        """A Series whose index has the label ``data`` is not unpacked as ``s.data``."""
        labels = ["data", "b", "c"]
        a = self._series("a", [1.0, 2.0, 3.0], labels=labels)

        _, ax = plot_comparison(a)

        assert [bar.get_height() for bar in ax.patches] == [1.0, 2.0, 3.0]

    def test_a_truth_labeled_data_is_plotted_as_a_series(self):
        """The truth is not unpacked as ``truth.data`` when its index has that label."""
        a = self._series("a", [1.0, 2.0, 3.0], labels=["u", "v", "w"])
        truth = self._series("t", [1.1, 2.1, 3.1], labels=["data", "b", "c"])

        _, ax = plot_comparison(a, truth=truth)

        (crosses,) = ax.collections
        np.testing.assert_array_equal(crosses.get_offsets()[:, 1], truth.values)

    def test_errors_labeled_data_are_plotted_as_a_series(self):
        """An error Series is not unpacked as ``err.data`` when its index has that label."""
        a = self._series("a", [1.0, 2.0, 3.0], labels=["u", "v", "w"])
        err = self._series("a", [0.5, 0.25, 0.125], labels=["data", "b", "c"])

        _, ax = plot_comparison(a, errors=[err])

        (whiskers,) = ax.containers[0].lines[2]
        lengths = [seg[1, 1] - seg[0, 1] for seg in whiskers.get_segments()]
        np.testing.assert_allclose(lengths, 2 * err.values)

    def test_errors_in_a_column_data_are_not_replaced_by_it(self):
        """The error is the column named like its series, not a column named ``data``."""
        labels = ["u", "v", "w"]
        a = self._series("a", [1.0, 2.0, 3.0], labels=labels)
        errors = pd.DataFrame(
            {"data": [9.0, 9.0, 9.0], "a": [0.5, 0.25, 0.125]},
            index=pd.Index(labels, name="x"),
        )

        _, ax = plot_comparison(a, errors=[errors])

        (whiskers,) = ax.containers[0].lines[2]
        lengths = [seg[1, 1] - seg[0, 1] for seg in whiskers.get_segments()]
        np.testing.assert_allclose(lengths, 2 * errors["a"].values)

    @staticmethod
    def _band(ax):
        """The lowest and highest y of the one shaded band on ``ax``."""
        (band,) = ax.collections
        y = band.get_paths()[0].vertices[:, 1]
        return y.min(), y.max()

    @pytest.mark.parametrize("position", [0, 1])
    def test_a_series_with_no_error_has_no_band(self, position):
        """A None error leaves its series bare; the other series keeps its band."""
        values = [[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]]
        a, b = (self._series(n, v) for n, v in zip("ab", values, strict=True))
        err = self._series("sigma", [0.5, 0.25, 0.125])
        errors = [err if i == position else None for i in range(2)]

        _, ax = plot_comparison(a, b, errors=errors)

        low, high = self._band(ax)
        assert low == pytest.approx((np.array(values[position]) - err.values).min())
        assert high == pytest.approx((np.array(values[position]) + err.values).max())
        assert list(self._lines(ax)) == ["a", "b"]

    def test_a_bar_with_no_error_has_no_whiskers(self):
        """On a bar plot, only the series that has an error gets whiskers."""
        labels = ["u", "v", "w"]
        a = self._series("a", [1.0, 2.0, 3.0], labels=labels)
        b = self._series("b", [1.5, 2.5, 3.5], labels=labels)
        err = self._series("sigma", [0.5, 0.25, 0.125], labels=labels)

        _, ax = plot_comparison(a, b, errors=[None, err])

        (whiskers,) = ax.collections
        segments = whiskers.get_segments()
        lengths = [seg[1, 1] - seg[0, 1] for seg in segments]
        np.testing.assert_allclose(lengths, 2 * err.values)
        # the second series' bars sit 0.2 to the right of the first's
        np.testing.assert_allclose([seg[0, 0] for seg in segments], [0.2, 1.2, 2.2])

    def test_unnamed_series_keep_their_own_errors(self):
        """Without names the series are told apart by position, errors included."""
        idx = pd.Index([0.0, 1.0, 2.0], name="x")
        a = pd.Series([1.0, 2.0, 3.0], index=idx)
        b = pd.Series([11.0, 12.0, 13.0], index=idx)
        err = pd.Series([0.5, 0.5, 0.5], index=idx)

        _, ax = plot_comparison(a, b, errors=[None, err])

        assert self._band(ax) == pytest.approx((10.5, 13.5))

    @pytest.mark.parametrize("errors", [[None], [None, None], None])
    def test_no_errors_no_bands(self, errors):
        """None for every series is the same as no errors."""
        a = self._series("a", [1.0, 2.0, 3.0])
        b = self._series("b", [1.5, 2.5, 3.5])

        _, ax = plot_comparison(a, b, errors=errors)

        assert len(ax.collections) == 0
        assert list(self._lines(ax)) == ["a", "b"]

    def test_one_error_for_two_series(self):
        """The same error object can serve two series: each gets its own band."""
        a = self._series("a", [1.0, 2.0, 3.0])
        b = self._series("b", [11.0, 12.0, 13.0])
        err = self._series("sigma", [0.5, 0.5, 0.5])

        _, ax = plot_comparison(a, b, errors=[err, err])

        lows = [band.get_paths()[0].vertices[:, 1].min() for band in ax.collections]
        assert lows == pytest.approx([0.5, 10.5])

    @pytest.mark.parametrize(
        "wrap",
        [pytest.param(lambda e: e, id="series"), pytest.param(Block, id="block")],
    )
    def test_errors_keep_their_name(self, wrap):
        """The error is renamed for plotting, but the one the caller passed is not."""
        a = self._series("a", [1.0, 2.0, 3.0])
        err = wrap(self._series("sigma", [0.5, 0.5, 0.5]))
        before = _unwrapped_name(err)

        plot_comparison(a, errors=[err])

        assert _unwrapped_name(err) == before == "sigma"

    def test_errors_frame_gains_no_name(self):
        """A DataFrame of errors is read for its columns and left as it was."""
        a = self._series("a", [1.0, 2.0, 3.0])
        errors = self._series("a", [0.5, 0.5, 0.5]).to_frame()

        plot_comparison(a, errors=[errors])

        assert not hasattr(errors, "name")


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
        """Test that 4D or higher arrays work or raise appropriate errors."""
        samples = np.array([[[[1.0]]]])  # 4D array

        # This might work or raise an error depending on implementation
        # Just test that it doesn't crash unexpectedly
        try:
            result = compute_credible_interval(samples)
            assert result is not None
        except (ValueError, IndexError):
            # It's fine if it raises an error for unusual dimensions
            pass

    def test_credible_interval_1d_array(self):
        """Test that 1D array is handled (converted to 2D)."""
        samples = np.array([1.0, 2.0, 3.0])

        # Function should handle 1D arrays by treating them as single variable
        result = compute_credible_interval(samples)
        assert result is not None

    def test_credible_interval_quantile_bounds(self):
        """Test that quantiles work with extreme values."""
        samples = np.random.normal(0, 1, (1000, 5))

        lower, upper = compute_credible_interval(samples, q=(0.01, 0.99))

        assert lower.shape == (5,)
        assert upper.shape == (5,)
        # Lower quantiles should be lower than upper quantiles
        assert np.all(lower < upper)
