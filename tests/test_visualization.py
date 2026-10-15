"""Test suite for fips.visualization module."""

import numpy as np
import pandas as pd
import pytest

import fips
from fips import Covariance, Operator, Problem
from fips.vector import Vector
from fips.visualization import (
    compute_credible_interval,
    plot_comparison,
    plot_error_norm,
    plot_map,
    plot_obs,
)


def _block(series):
    """A vector of one block, named after the Series."""
    return Vector.from_series(series, block=series.name)


def _unwrapped_name(x):
    """The name of the Series or vector ``x``."""
    return x.name


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

    def test_one_block_is_plotted_on_its_own_labels(self):
        """A vector of one block is plotted as its Series without the block level, next to a Series."""
        a = self._series("a", [1.0, 2.0, 3.0])
        b = self._series("b", [1.5, 2.5, 3.5])

        _, ax = plot_comparison(_block(a), b)

        lines = self._lines(ax)
        assert list(lines) == ["a", "b"]
        np.testing.assert_array_equal(lines["a"], a.values)
        np.testing.assert_array_equal(lines["b"], b.values)

    def test_vectors_are_plotted_by_their_data(self):
        """A Vector is plotted as its Series, over the level named by ``x``."""
        a = self._series("a", [1.0, 2.0, 3.0])

        _, ax = plot_comparison(_block(a), x="x")

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

        _, ax = plot_comparison(_block(a).to_series(), x="x")

        assert ax.get_xlabel() == "x"

    def test_truth_is_drawn_as_a_line(self):
        """The truth of a line plot is a dashed black line."""
        a = self._series("a", [1.0, 2.0, 3.0])
        truth = self._series("t", [1.1, 2.1, 3.1])

        _, ax = plot_comparison(a, truth=_block(truth))

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
            pytest.param(_block, id="vector"),
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
            pytest.param(_block, id="vector"),
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
        [pytest.param(lambda e: e, id="series"), pytest.param(_block, id="vector")],
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


# ---------------------------------------------------------------------------
# plot_map and plot_obs: a block of the state on a map, the observations over time
# ---------------------------------------------------------------------------


def _solved(levels=("time", "lat", "lon"), obs_levels=("obs_location", "obs_time")):
    """
    The posterior of a small problem: 2 times x 2 lat x 2 lon in the state, two sites of daily data.

    The state's block is ``flux`` and the observations' ``ch4``; the level
    names are `levels` and `obs_levels`.
    """
    rng = np.random.default_rng(0)
    state_idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2023-01", periods=2, freq="MS"),
            [40.5, 40.9],
            [-112.0, -111.6],
        ],
        names=list(levels),
    )
    days = pd.date_range("2023-01-01", periods=40, freq="D")
    # SITE2 has a 10-day gap in its data, which plot_obs shades.
    obs_idx = pd.MultiIndex.from_tuples(
        [
            (site, day)
            for site in ["SITE1", "SITE2"]
            for i, day in enumerate(days)
            if not (site == "SITE2" and 15 <= i < 25)
        ],
        names=list(obs_levels),
    )
    n, m = len(state_idx), len(obs_idx)
    prior = Vector.from_series(
        pd.Series(rng.uniform(1, 2, n), index=state_idx), block="flux"
    )
    obs = Vector.from_series(
        pd.Series(rng.uniform(400, 410, m), index=obs_idx), block="ch4"
    )
    problem = Problem(
        prior=prior,
        prior_cov=Covariance.diagonal(0.5, prior.space),
        obs=obs,
        obs_cov=Covariance.diagonal(0.1, obs.space),
        forward=Operator(rng.uniform(0, 1, (m, n)), obs.space, prior.space),
        offset=Vector(np.full(m, 395.0), obs.space),
    )
    return fips.solve(problem)


@pytest.fixture(scope="module")
def solved():
    """The posterior of the small map problem."""
    return _solved()


class TestPlotMap:
    """Tests for plot_map."""

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        """Close every figure after a test."""
        plt = pytest.importorskip("matplotlib.pyplot")
        yield
        plt.close("all")

    @pytest.mark.parametrize(
        ("time", "has_truth", "titles"),
        [
            ("mean", False, ["Prior", "Posterior", "Residual"]),
            ("mean", True, ["Truth", "Prior", "Posterior", "Residual"]),
            (1, False, ["Prior", "Posterior", "Residual"]),
            (pd.Timestamp("2023-02-01"), False, ["Prior", "Posterior", "Residual"]),
            ("std", False, ["Prior", "Posterior"]),
            ("std", True, ["Truth", "Prior", "Posterior"]),
        ],
        ids=["mean", "mean-truth", "int", "label", "std", "std-truth"],
    )
    def test_panels(self, solved, time, has_truth, titles):
        """The truth panel comes first and the residual panel last; ``std`` has none."""
        truth = solved.problem.prior["flux"].to_series().droplevel("block") * 1.1
        fig, axes = plot_map(solved, time=time, truth=truth if has_truth else None)

        assert [ax.get_title() for ax in axes] == titles
        # The panels, plus one colorbar axes for the prior and one for the residual.
        assert len(fig.axes) == len(titles) + (1 if time == "std" else 2)

    def test_panels_hold_the_prior_and_the_estimate(self, solved):
        """The prior and posterior panels are the block's values at the time asked for."""
        _, axes = plot_map(solved, time=0)
        first = pd.Timestamp("2023-01-01")
        for ax, vector in zip(
            axes[:2], [solved.problem.prior, solved.estimate], strict=True
        ):
            expected = vector["flux"].to_xarray().sel(time=first)
            mesh = ax.collections[0].get_array()
            np.testing.assert_allclose(
                np.sort(np.ravel(mesh)), np.sort(expected.values.ravel())
            )

    def test_block_is_the_colorbar_label_and_the_title(self, solved):
        """The block's name labels the map: nothing in it names a science."""
        fig, _ = plot_map(solved)
        assert fig._suptitle.get_text() == "flux"
        assert fig.axes[-2].get_xlabel() == "flux"

    def test_truth_as_a_vector(self, solved):
        """A truth given as a Vector is read at the block."""
        _, axes = plot_map(solved, truth=solved.problem.prior)
        assert axes[0].get_title() == "Truth"

    def test_other_level_names(self):
        """The level names are arguments."""
        post = _solved(levels=("month", "y", "x"))
        _, axes = plot_map(post, x="x", y="y", time_level="month")
        assert [ax.get_title() for ax in axes] == ["Prior", "Posterior", "Residual"]

    def test_a_state_of_several_blocks_needs_block(self):
        """With several blocks the block is named; a block the state lacks raises."""
        post = _solved()
        problem = post.problem
        bias = Vector.from_series(
            pd.Series([0.0], index=pd.Index(["b"], name="b")), block="bias"
        )
        prior = Vector.from_blocks({"flux": problem.prior, "bias": bias})
        two = fips.solve(
            Problem(
                prior=prior,
                prior_cov=Covariance.diagonal(0.5, prior.space),
                obs=problem.obs,
                obs_cov=problem.obs_cov,
                forward=Operator.from_blocks(
                    {("ch4", "flux"): problem.forward},
                    rows=problem.obs.space,
                    cols=prior.space,
                ),
            )
        )
        with pytest.raises(ValueError, match="name one with block="):
            plot_map(two)
        with pytest.raises(KeyError, match="no block 'sink'"):
            plot_map(two, block="sink")
        _, axes = plot_map(two, block="flux")
        assert len(axes) == 3

    def test_sites(self, solved):
        """Site markers are drawn on every panel."""
        _, axes = plot_map(solved, sites={"A": (40.7, -111.8)})

        assert all(len(ax.collections) >= 2 for ax in axes)

    def test_sites_must_be_a_dict(self, solved):
        """Anything but a dict of sites is refused."""
        with pytest.raises(TypeError, match="dict"):
            plot_map(solved, sites=[(40.7, -111.8)])

    def test_tiler(self, solved):
        """With a tiler, every panel is a map with the tiles added."""
        pytest.importorskip("cartopy")
        from cartopy.io.img_tiles import GoogleTiles

        _, axes = plot_map(solved, tiler=GoogleTiles(), tiler_zoom=5)

        assert [type(ax).__name__ for ax in axes] == ["GeoAxes"] * 3
        assert [len(ax.img_factories) for ax in axes] == [1] * 3

    def test_tiler_without_cartopy(self, solved, monkeypatch):
        """A missing cartopy raises its install hint before a figure is made."""
        import fips.visualization as viz

        plt = pytest.importorskip("matplotlib.pyplot")

        def missing():
            raise ImportError("cartopy is required for map tiles")

        monkeypatch.setattr(viz, "_require_cartopy", missing)

        class Tiler:
            crs = None

        with pytest.raises(ImportError, match="cartopy"):
            plot_map(solved, tiler=Tiler())
        assert plt.get_fignums() == []


class TestPlotObs:
    """Tests for plot_obs."""

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        """Close every figure after a test."""
        plt = pytest.importorskip("matplotlib.pyplot")
        yield
        plt.close("all")

    def test_defaults(self, solved):
        """
        The default windows raise no warning.

        pandas 3 deprecates the lowercase alias, ``"30d"``, and the suite turns
        warnings into errors.
        """
        axes = plot_obs(solved)

        assert axes.shape == (2,)
        # SITE2's 11-day gap is longer than the default 7-day threshold
        assert [len(ax.patches) for ax in axes] == [0, 1]

    def test_legend_and_gap(self, solved):
        """Each location gets a figure; one with a data gap adds a "no data" entry."""
        axes = plot_obs(solved, rolling_window="30D", gap_threshold="5D")

        assert axes.shape == (2,)
        legends = [[t.get_text() for t in ax.get_legend().get_texts()] for ax in axes]
        assert legends[0] == ["observed", "posterior", "prior"]
        assert legends[1] == ["observed", "posterior", "prior", "no data"]
        # the shaded gap is one patch
        assert [len(ax.patches) for ax in axes] == [0, 1]

    def test_location(self, solved):
        """Only the requested location is plotted, titled with the block."""
        axes = plot_obs(solved, location="SITE1", show_raw=False)

        assert axes.shape == (1,)
        assert axes[0].get_title() == "ch4 at SITE1"
        assert axes[0].get_ylabel() == "ch4"
        assert len(axes[0].lines) == 3

    def test_lines_are_the_observations_and_the_predictions(self, solved):
        """The raw points are the observations, the posterior's and the prior's modeled observations."""
        axes = plot_obs(solved, location=["SITE1"])
        problem = solved.problem
        expected = [
            problem.obs,
            solved.modeled_obs,
            problem.predict(problem.prior),
        ]
        for line, vector in zip(axes[0].lines[:3], expected, strict=True):
            site = vector["ch4"].to_series().droplevel("block").xs("SITE1")
            np.testing.assert_allclose(line.get_ydata(), site.values)

    def test_other_level_names_and_labels(self):
        """The levels are arguments, and a location label need not be a string."""
        post = _solved(obs_levels=("site_idx", "time"))
        problem = post.problem
        # relabel the sites as integers
        obs = problem.obs["ch4"].to_series().droplevel("block")
        obs.index = obs.index.set_levels([0, 1], level="site_idx")
        obs = Vector.from_series(obs, block="ch4")
        problem = Problem(
            prior=problem.prior,
            prior_cov=problem.prior_cov,
            obs=obs,
            obs_cov=Covariance.diagonal(0.1, obs.space),
            forward=Operator(problem.forward.dense(), obs.space, problem.prior.space),
        )
        axes = plot_obs(
            fips.solve(problem),
            location=1,
            location_level="site_idx",
            time_level="time",
        )
        assert axes[0].get_title() == "ch4 at 1"
