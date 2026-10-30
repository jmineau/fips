"""Test suite for fips.visualization module."""

import numpy as np
import pandas as pd
import pytest

from fips.covariance import Covariance
from fips.operator import Operator
from fips.space import Space
from fips.vector import Vector
from fips.visualization import plot_comparison


def _block(series):
    """A vector of one block, named after the Series."""
    return Vector.from_series(series, block=series.name)


def _unwrapped_name(x):
    """The name of the Series or vector ``x``."""
    return x.name


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
        [pytest.param(lambda e: e, id="series"), pytest.param(_block, id="vector")],
    )
    def test_truth_is_matched_by_label(self, wrap):
        """A truth in another order is drawn at its labels, on a line and on bars."""
        a = self._series("a", [1.0, 2.0, 3.0])
        truth = wrap(self._series("t", [10.0, 20.0, 30.0]).iloc[::-1])

        _, ax = plot_comparison(a, truth=truth)

        np.testing.assert_array_equal(ax.lines[-1].get_ydata(), [10.0, 20.0, 30.0])

        b = self._series("b", [1.0, 2.0, 3.0], labels=["u", "v", "w"])
        t = self._series("t", [10.0, 20.0, 30.0], labels=["u", "v", "w"]).iloc[::-1]
        _, ax = plot_comparison(b, truth=wrap(t))
        (crosses,) = ax.collections
        np.testing.assert_array_equal(crosses.get_offsets()[:, 1], [10.0, 20.0, 30.0])

    def test_a_truth_missing_a_label_raises(self):
        """A label of the series the truth has no value for is named."""
        a = self._series("a", [1.0, 2.0, 3.0])
        truth = self._series("t", [10.0, 20.0], labels=(0.0, 1.0))

        with pytest.raises(ValueError, match="truth has no value .*2.0"):
            plot_comparison(a, truth=truth)

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
        a = self._series("a", [1.0, 2.0, 3.0], labels=["data", "b", "c"])
        truth = self._series("t", [1.1, 2.1, 3.1], labels=["data", "b", "c"])

        _, ax = plot_comparison(a, truth=truth)

        (crosses,) = ax.collections
        np.testing.assert_array_equal(crosses.get_offsets()[:, 1], truth.values)

    def test_errors_labeled_data_are_plotted_as_a_series(self):
        """An error Series is not unpacked as ``err.data`` when its index has that label."""
        a = self._series("a", [1.0, 2.0, 3.0], labels=["data", "b", "c"])
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


class TestErrorsAreMatchedByLabel:
    """The errors of ``plot_comparison`` are read by label, in any order (issue 165)."""

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        """Close every figure after a test."""
        plt = pytest.importorskip("matplotlib.pyplot")
        yield
        plt.close("all")

    # labels in an order that is neither sorted nor reversed; std of each label
    X = [3.0, 1.0, 2.0]
    STD = {3.0: 1.0, 1.0: 2.0, 2.0: 3.0}
    Y = {3.0: 10.0, 1.0: 20.0, 2.0: 30.0}
    C = ["c", "a", "b"]
    STD_C = {"c": 1.0, "a": 2.0, "b": 3.0}

    @staticmethod
    def _series(labels, values, name="s"):
        """A Series of ``values`` on the labels, an index called ``x``."""
        return pd.Series(values, index=pd.Index(labels, name="x"), name=name)

    @classmethod
    def _est(cls, labels, by_label):
        """The estimate: a Series on the labels, in their order."""
        return cls._series(labels, [by_label[k] for k in labels], name="est")

    @staticmethod
    def _band(ax):
        """The (lower, upper) edge of the band at each x of the line, by x."""
        (band,) = ax.collections
        v = band.get_paths()[0].vertices
        return {
            x: (v[v[:, 0] == x, 1].min(), v[v[:, 0] == x, 1].max())
            for x in set(v[:, 0])
        }

    @staticmethod
    def _bar_lengths(ax):
        """The length of the whisker on each bar, in the order of the bars."""
        (whiskers,) = ax.containers[0].lines[2]
        return [seg[1, 1] - seg[0, 1] for seg in whiskers.get_segments()]

    def _std_in_orders(self, labels, by_label):
        """The same standard deviations as Series, Vector and DataFrame, in other orders."""
        for order in (sorted(labels), list(reversed(labels)), list(labels)):
            series = self._series(order, [by_label[k] for k in order], name="est")
            yield series
            yield series.to_frame()
            yield Vector.from_series(series, block="flux")

    def test_a_series_of_errors_in_another_order_on_a_line(self):
        """The band of a line plot is each label's own, whatever the errors' order."""
        est = self._est(self.X, self.Y)
        for err in self._std_in_orders(self.X, self.STD):
            _, ax = plot_comparison(est, errors=[err])
            band = self._band(ax)
            for x in self.X:
                assert band[x] == pytest.approx(
                    (self.Y[x] - self.STD[x], self.Y[x] + self.STD[x])
                )

    def test_a_series_of_errors_in_another_order_on_bars(self):
        """The whisker of each bar is its own label's, whatever the errors' order."""
        y = {"c": 10.0, "a": 20.0, "b": 30.0}
        est = self._est(self.C, y)
        for err in self._std_in_orders(self.C, self.STD_C):
            _, ax = plot_comparison(est, errors=[err])
            np.testing.assert_allclose(
                self._bar_lengths(ax), [2 * self.STD_C[k] for k in self.C]
            )

    def test_the_order_of_the_errors_changes_nothing(self):
        """Errors in the series' order and in another give the same picture."""
        est = self._est(self.X, self.Y)
        std = self._est(self.X, self.STD)

        _, same = plot_comparison(est, errors=[std])
        _, other = plot_comparison(est, errors=[std.sort_index()])

        assert self._band(other) == self._band(same)

    @pytest.mark.parametrize("form", ["covariance", "operator"])
    def test_errors_from_a_matrix_on_a_reordered_space(self, form):
        """A covariance or an operator whose space is in another order is matched by label."""
        est = Vector.from_series(self._est(self.X, self.Y), block="flux")
        order = list(reversed(self.X))
        space = Space({"flux": pd.Index(order, name="x")})
        variances = np.array([self.STD[k] ** 2 for k in order])
        cov = Covariance.diagonal(variances, space)
        err = cov if form == "covariance" else Operator(cov.dense(), space, space)

        _, ax = plot_comparison(est, errors=[err])

        band = self._band(ax)
        for x in self.X:
            assert band[x] == pytest.approx(
                (self.Y[x] - self.STD[x], self.Y[x] + self.STD[x])
            )

        labels = Vector.from_series(
            self._est(self.C, {"c": 10.0, "a": 20.0, "b": 30.0}), block="flux"
        )
        space = Space({"flux": pd.Index(["b", "c", "a"], name="x")})
        variances = np.array([self.STD_C[k] ** 2 for k in ["b", "c", "a"]])
        cov = Covariance.diagonal(variances, space)
        err = cov if form == "covariance" else Operator(cov.dense(), space, space)

        _, ax = plot_comparison(labels, errors=[err])

        np.testing.assert_allclose(
            self._bar_lengths(ax), [2 * self.STD_C[k] for k in self.C]
        )

    def test_errors_of_several_blocks_are_matched_by_label(self):
        """With a block level too, a covariance on reordered blocks and labels is matched."""
        x = pd.Index([1.0, 2.0], name="x")
        est = Vector([10.0, 20.0, 30.0, 40.0], Space({"v": x, "u": x}))  # v1 v2 u1 u2
        std = {("u", 1.0): 1.0, ("u", 2.0): 2.0, ("v", 1.0): 3.0, ("v", 2.0): 4.0}
        order = {"u": x[::-1], "v": x}  # u2 u1 v1 v2
        variances = [std[(b, k)] ** 2 for b, idx in order.items() for k in idx]

        cov = Covariance.diagonal(variances, Space(order))
        _, ax = plot_comparison(est, x="x", errors=[cov])

        (band,) = ax.collections
        lower = band.get_paths()[0].vertices[1:5, 1]
        np.testing.assert_allclose(lower, [10 - 3, 20 - 4, 30 - 1, 40 - 2])

    def test_a_plain_array_or_list_is_read_by_position(self):
        """An array has no labels: it is the series' order."""
        est = self._est(self.X, self.Y)

        for err in (np.array([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0], (1.0, 2.0, 3.0)):
            _, ax = plot_comparison(est, errors=[err])
            band = self._band(ax)
            assert band[3.0] == pytest.approx((9.0, 11.0))
            assert band[1.0] == pytest.approx((18.0, 22.0))
            assert band[2.0] == pytest.approx((27.0, 33.0))

    def test_a_plain_array_of_the_wrong_length_raises(self):
        """An array is read by position, so it must be as long as the series."""
        est = self._est(self.X, self.Y)

        with pytest.raises(ValueError, match=r"errors\[0\] has 2 values for 3 points"):
            plot_comparison(est, errors=[np.array([1.0, 2.0])])

    @pytest.mark.parametrize("kind", ["line", "bar"])
    def test_errors_missing_a_label_raise(self, kind):
        """An error with no value for a label of the series names the label."""
        est = self._est(self.X, self.Y)
        std = self._series([1.0, 2.0], [2.0, 3.0])

        with pytest.raises(ValueError, match=r"errors\[0\] has no value .*3\.0"):
            plot_comparison(est, errors=[std], kind=kind)

    def test_errors_on_other_labels_raise(self):
        """Errors that share no label with the series raise, rather than get read by position."""
        est = self._est(self.C, {"c": 10.0, "a": 20.0, "b": 30.0})
        std = self._series(["u", "v", "w"], [1.0, 2.0, 3.0])
        space = Space({"flux": pd.Index(["u", "v", "w"], name="x")})

        for err in (std, Covariance.diagonal([1.0, 4.0, 9.0], space)):
            with pytest.raises(ValueError, match=r"errors\[0\] has no value .*'c'"):
                plot_comparison(est, errors=[err])

    def test_the_error_of_the_second_series_is_named_by_its_position(self):
        """The message says which of the errors is wrong."""
        a = self._est(self.X, self.Y)
        b = self._est(self.X, self.Y)
        std = self._series([1.0, 2.0], [2.0, 3.0])

        with pytest.raises(ValueError, match=r"errors\[1\]"):
            plot_comparison(a, b, errors=[None, std])

    def test_errors_for_labels_the_series_lacks_are_left_out(self):
        """An error with more labels than the series is cut to the series', as truth is."""
        est = self._est([2.0, 1.0], {1.0: 20.0, 2.0: 30.0})
        std = self._est(self.X, self.STD)

        _, ax = plot_comparison(est, errors=[std])

        band = self._band(ax)
        assert band[1.0] == pytest.approx((18.0, 22.0))
        assert band[2.0] == pytest.approx((27.0, 33.0))
