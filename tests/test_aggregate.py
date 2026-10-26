"""
Tests for fips.aggregate: the aggregation operators and the problem of aggregated observations.

Ported from the tests of ``ObsAggregator`` and ``integrate_over_time_bins``,
which the operators replace; the numbers each old test checked are checked
here on the new objects.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import fips
from fips import Covariance, Operator, Problem, Space, Vector
from fips.linalg import SparseOperator

# ---------------------------------------------------------------------------
# Sums over time bins (what integrate_over_time_bins did)
# ---------------------------------------------------------------------------


def _bins(start: str, periods: int, freq: str, name: str = "time", unit=None):
    """Left-closed time bins, named after the level they bin."""
    first = pd.Timestamp(start) if unit is None else pd.Timestamp(start).as_unit(unit)
    return pd.interval_range(
        start=first, periods=periods, freq=freq, closed="left", name=name
    )


def test_sum_over_time_bins():
    """Hourly values summed into two-hour bins, labeled by each bin's start."""
    idx = pd.date_range("2023-01-01", periods=10, freq="h", name="time")
    v = Vector.from_series(pd.Series(1.0, index=idx), block="flux")
    W = fips.aggregate.sum(v.space, by=_bins("2023-01-01", 5, "2h"))
    total = W @ v
    assert len(total) == 5
    labels = W.rows.select("flux")
    assert labels.name == "time"
    assert list(labels) == list(
        pd.date_range("2023-01-01", periods=5, freq="2h", name="time")
    )
    np.testing.assert_array_equal(total.values, 2.0)


def test_sum_over_time_bins_keeps_the_other_levels():
    """A (time, loc) block summed over time bins at each loc."""
    times = pd.date_range("2023-01-01", periods=4, freq="h")
    idx = pd.MultiIndex.from_product([times, ["A", "B"]], names=["time", "loc"])
    v = Vector.from_series(pd.Series(1.0, index=idx), block="flux")
    W = fips.aggregate.sum(v.space, by=[_bins("2023-01-01", 2, "2h"), "loc"])
    assert len(W.rows) == 4
    assert list(W.rows.select("flux").names) == ["time", "loc"]
    np.testing.assert_array_equal((W @ v).values, 2.0)


def test_a_missing_level_raises():
    """A key on a level the block lacks names the block and its levels."""
    v = Vector.from_series(
        pd.Series([1.0, 2.0], index=pd.Index([1, 2], name="other")), block="x"
    )
    with pytest.raises(
        ValueError, match=r"Block 'x' has no level 'time'; its levels are \['other'\]"
    ):
        fips.aggregate.sum(v.space, by=_bins("2023-01-01", 2, "1h"))


@pytest.mark.parametrize(
    "data_unit, bin_unit", [("ns", "us"), ("ms", "s"), ("us", "ns")]
)
def test_time_bins_of_another_resolution(data_unit, bin_unit):
    """Labels and bins with different datetime units bin as the same instants (pandas 2 refuses the lookup)."""
    times = pd.DatetimeIndex(
        pd.date_range("2023-01-01", periods=6, freq="h").as_unit(data_unit),
        name="time",
    )
    v = Vector.from_series(pd.Series(1.0, index=times), block="flux")
    W = fips.aggregate.sum(v.space, by=_bins("2023-01-01", 3, "2h", unit=bin_unit))
    np.testing.assert_array_equal((W @ v).values, 2.0)


def test_labels_outside_every_bin_raise():
    """A label no interval holds is named, not dropped."""
    idx = pd.date_range("2023-01-01", periods=4, freq="h", name="time")
    space = Space({"flux": idx})
    with pytest.raises(ValueError, match="outside every interval: Timestamp"):
        fips.aggregate.sum(space, by=_bins("2023-01-01", 3, "1h"))


# ---------------------------------------------------------------------------
# Observations: 4 obs over two days, 2 per day (what ObsAggregator's tests used)
# ---------------------------------------------------------------------------

TIMES = pd.DatetimeIndex(
    ["2023-01-01 00:00", "2023-01-01 12:00", "2023-01-02 00:00", "2023-01-02 12:00"],
    name="obs_time",
)
DAILY = pd.Grouper(level="obs_time", freq="D")
STATE = Space({"state": pd.Index(["s0", "s1", "s2"], name="state_id")})


@pytest.fixture
def obs():
    """The observations, one block."""
    return Vector.from_series(
        pd.Series([1.0, 2.0, 3.0, 4.0], index=TIMES, name="obs"), block="obs"
    )


@pytest.fixture
def problem(obs):
    """A problem on those observations: H = 0.5 everywhere, S_z = 2 I."""
    return Problem(
        prior=Vector(np.zeros(3), STATE, name="prior"),
        prior_cov=Covariance.diagonal(1.0, STATE),
        obs=obs,
        obs_cov=Covariance.from_frame(
            pd.DataFrame(np.eye(4) * 2.0, index=obs.index, columns=obs.index)
        ),
        forward=Operator(np.full((4, 3), 0.5), obs.space, STATE),
    )


def _with(problem: Problem, **inputs) -> Problem:
    """Return `problem` with some inputs replaced."""
    args = {
        "prior": problem.prior,
        "prior_cov": problem.prior_cov,
        "obs": problem.obs,
        "obs_cov": problem.obs_cov,
        "forward": problem.forward,
        "offset": problem.offset,
    }
    args.update(inputs)
    return Problem(**args)


class TestOperators:
    """The matrices ``sum`` and ``mean`` build."""

    def test_mean_weights_sum_to_one(self, obs):
        """Each row of a mean averages its group: its weights sum to 1."""
        W = fips.aggregate.mean(obs.space, by=DAILY)
        np.testing.assert_allclose(W.dense().sum(axis=1), 1.0)
        np.testing.assert_allclose(W.dense(), [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])

    def test_sum_weights(self, obs):
        """Each row of a sum has a 1 for each of its 2 observations."""
        W = fips.aggregate.sum(obs.space, by=DAILY)
        np.testing.assert_allclose(W.dense().sum(axis=1), 2.0)

    def test_the_operator_is_sparse_on_the_space(self, obs):
        """W is a sparse Operator whose columns are the space and rows the groups."""
        W = fips.aggregate.mean(obs.space, by=DAILY)
        assert isinstance(W, Operator)
        assert isinstance(W.data, SparseOperator)
        assert W.cols == obs.space
        assert W.rows.names == ("obs",)
        assert list(W.rows.select("obs")) == list(
            pd.DatetimeIndex(["2023-01-01", "2023-01-02"])
        )
        assert W.rows.select("obs").name == "obs_time"

    def test_four_ways_to_the_same_days(self, obs):
        """A Grouper, an IntervalIndex, a Series mapping and a level of days give one W."""
        days = pd.DatetimeIndex(["2023-01-01", "2023-01-02"], name="obs_time")
        bins = pd.IntervalIndex.from_breaks(
            pd.date_range("2023-01-01", periods=3, freq="D"),
            closed="left",
            name="obs_time",
        )
        mapping = pd.Series(days.repeat(2), index=TIMES, name="obs_time")
        expected = fips.aggregate.mean(obs.space, by=DAILY)
        for W in (
            fips.aggregate.mean(obs.space, by=bins),
            fips.aggregate.mean(obs.space, by=mapping),
        ):
            assert list(W.rows.select("obs")) == list(expected.rows.select("obs"))
            np.testing.assert_array_equal(W.dense(), expected.dense())
        # Equal labels are one group: the level itself
        with_days = Vector(
            obs.values,
            Space(
                {
                    "obs": pd.MultiIndex.from_arrays(
                        [TIMES.floor("D"), TIMES], names=["day", "obs_time"]
                    )
                }
            ),
        )
        W = fips.aggregate.mean(with_days.space, by="day")
        np.testing.assert_array_equal((W @ with_days).values, [1.5, 3.5])

    def test_an_empty_day_makes_no_row(self):
        """Observations on days 1 and 3 give two rows, not a third of zeros for day 2."""
        times = pd.DatetimeIndex(
            [
                "2023-01-01 00:00",
                "2023-01-01 12:00",
                "2023-01-03 00:00",
                "2023-01-03 12:00",
            ],
            name="obs_time",
        )
        z = Vector.from_series(
            pd.Series([1.0, 2.0, 3.0, 4.0], index=times), block="obs"
        )
        W = fips.aggregate.mean(z.space, by=DAILY)
        assert list(W.rows.select("obs")) == list(
            pd.DatetimeIndex(["2023-01-01", "2023-01-03"])
        )
        np.testing.assert_array_equal((W @ z).values, [1.5, 3.5])

    def test_labels_out_of_time_order(self):
        """Labels whose times are not sorted are binned by their own times."""
        labels = pd.MultiIndex.from_product(
            [["A", "B"], pd.date_range("2020-01-01", periods=4, freq="12h")],
            names=["site", "obs_time"],
        )
        z = Vector.from_series(pd.Series(np.arange(8.0), index=labels), block="co2")
        W = fips.aggregate.mean(z.space, by=["site", DAILY])
        np.testing.assert_array_equal((W @ z).values, [0.5, 2.5, 4.5, 6.5])
        W_days = fips.aggregate.mean(z.space, by=DAILY)
        np.testing.assert_array_equal((W_days @ z).values, [2.5, 4.5])

    def test_a_mapping_relabels(self):
        """A Series maps a level's values to groups, named after the Series."""
        cells = pd.Index(["a", "b", "c", "d"], name="cell")
        x = Vector([1.0, 2.0, 3.0, 4.0], Space({"flux": cells}))
        regions = pd.Series(
            ["east", "west", "east", "west"], index=cells, name="region"
        )
        W = fips.aggregate.sum(x.space, by=regions)
        assert list(W.rows.select("flux")) == ["east", "west"]
        assert W.rows.select("flux").name == "region"
        np.testing.assert_array_equal((W @ x).values, [4.0, 6.0])

    def test_blocks_pass_through(self):
        """Blocks not named keep one row per label."""
        bias_times = pd.DatetimeIndex(
            ["2023-01-03 00:00", "2023-01-03 12:00"], name="obs_time"
        )
        space = Space({"obs": TIMES, "bias": bias_times})
        W = fips.aggregate.mean(space, by=DAILY, blocks="obs")
        assert W.shape == (4, 6)
        assert W.rows.names == ("obs", "bias")
        assert W.rows.select("bias").equals(bias_times)
        np.testing.assert_array_equal(W["bias", "bias"].dense(), np.eye(2))
        np.testing.assert_array_equal(W["obs", "bias"].dense(), 0.0)

    def test_every_block_on_its_own(self):
        """Two blocks with the same days are aggregated apart."""
        space = Space({"co2": TIMES, "ch4": TIMES})
        W = fips.aggregate.sum(space, by=DAILY)
        assert W.rows.names == ("co2", "ch4")
        assert W.shape == (4, 8)
        np.testing.assert_array_equal(W["co2", "ch4"].dense(), 0.0)

    def test_state_side_weighted_total(self):
        """A weighted total per month: the sum times a diagonal of the weights."""
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=2, freq="MS"), ["a", "b"]],
            names=["time", "cell"],
        )
        x = Vector([1.0, 2.0, 3.0, 4.0], Space({"flux": idx}))
        area = Operator(sp.diags_array([10.0, 20.0, 10.0, 20.0]), x.space, x.space)
        W = fips.aggregate.sum(x.space, by="time") @ area
        assert isinstance(W.data, SparseOperator)
        np.testing.assert_array_equal((W @ x).values, [50.0, 110.0])

    @pytest.mark.parametrize(
        "by, error, match",
        [
            ([], ValueError, "at least one key"),
            (pd.Grouper(level="obs_time"), ValueError, "needs a `level`"),
            (pd.Grouper(freq="D"), ValueError, "needs a `level`"),
            (pd.IntervalIndex.from_breaks([0, 1]), ValueError, "must be named"),
            (pd.Series(["x"]), ValueError, "name its index"),
            (lambda index: index, TypeError, "level name, a pandas Grouper"),
            (["obs_time", "obs_time"], ValueError, "label the same level"),
        ],
        ids=[
            "empty",
            "no-freq",
            "no-level",
            "unnamed-bins",
            "unnamed-map",
            "callable",
            "twice",
        ],
    )
    def test_bad_keys_raise(self, obs, by, error, match):
        """Each kind of key that cannot work says why."""
        with pytest.raises(error, match=match):
            fips.aggregate.sum(obs.space, by=by)

    def test_a_mapping_missing_a_value_raises(self, obs):
        """A value the mapping does not have is named; a mapping must be one to one on its index."""
        mapping = pd.Series(["day 1"], index=TIMES[:1], name="day")
        with pytest.raises(ValueError, match="not in the mapping"):
            fips.aggregate.sum(obs.space, by=mapping)
        duplicated = pd.Series(["a", "b"], index=TIMES[[0, 0]])
        with pytest.raises(ValueError, match="duplicate entries"):
            fips.aggregate.sum(obs.space, by=duplicated)

    def test_a_mapping_to_nan_raises_about_the_mapping(self):
        """A value mapped to NaN is named, with the mapping, not as missing labels of the user's block."""
        sites = Space({"co2": pd.Index(["A", "B", "C"], name="site")})
        for networks in (["n1", "n1", np.nan], [np.nan, "n1", "n1"]):
            by = pd.Series(networks, index=sites.select("co2"), name="network")
            for aggregate in (fips.aggregate.sum, fips.aggregate.mean):
                with pytest.raises(
                    ValueError,
                    match=r"maps 'site' '[AC]' \(on labels of block 'co2'\) to a missing",
                ):
                    aggregate(sites, by=by)

    def test_unknown_blocks_raise(self, obs):
        """A block that is not in the space is named, with the blocks there are."""
        with pytest.raises(
            ValueError, match="No block named 'co2'; the blocks are 'obs'"
        ):
            fips.aggregate.sum(obs.space, by=DAILY, blocks="co2")
        with pytest.raises(ValueError, match="names no block"):
            fips.aggregate.sum(obs.space, by=DAILY, blocks=[])
        with pytest.raises(TypeError, match="space must be a Space"):
            fips.aggregate.sum(obs.index, by=DAILY)


class TestProblemAggregate:
    """``Problem.aggregate(W)``: the observations, ``H``, ``S_z`` and the offset through one ``W``."""

    def test_obs_mean_values(self, problem):
        """The daily means of the observations: [1.5, 3.5]."""
        W = fips.aggregate.mean(problem.obs.space, by=DAILY)
        np.testing.assert_allclose(problem.aggregate(W).obs.values, [1.5, 3.5])

    def test_reduces_the_obs_count(self, problem):
        """4 observations become 2, on every input of the observations."""
        W = fips.aggregate.mean(problem.obs.space, by=DAILY)
        daily = problem.aggregate(W)
        assert len(daily.obs) == 2
        assert daily.forward.shape == (2, 3)
        assert daily.obs_cov.shape == (2, 2)
        assert daily.offset is None
        assert daily.prior is problem.prior
        assert daily.prior_cov.space == problem.prior_cov.space

    def test_types_and_names(self, problem):
        """A Problem of a Vector, an Operator and a Covariance, each named as before."""
        named = _with(
            problem,
            obs_cov=Covariance.diagonal(2.0, problem.obs.space, name="S_z"),
            forward=Operator(problem.forward.data, problem.obs.space, STATE, name=5),
        )
        daily = named.aggregate(fips.aggregate.mean(named.obs.space, by=DAILY))
        assert type(daily) is Problem
        assert isinstance(daily.obs, Vector)
        assert type(daily.forward) is Operator
        assert isinstance(daily.obs_cov, Covariance)
        assert (daily.obs.name, daily.forward.name, daily.obs_cov.name) == (
            "obs",
            "5",
            "S_z",
        )

    def test_mismatch_of_a_mean(self, problem):
        """S_z = 2 I averaged two at a time is I, diagonal and symmetric."""
        W = fips.aggregate.mean(problem.obs.space, by=DAILY)
        S = problem.aggregate(W).obs_cov.dense()
        np.testing.assert_allclose(S, np.eye(2), atol=1e-15)
        np.testing.assert_array_equal(S, S.T)

    def test_offset_vector_is_aggregated(self, problem):
        """W c, on the aggregated observations."""
        p = _with(
            problem,
            offset=Vector(problem.obs.values, problem.obs.space, name="background"),
        )
        offset = p.aggregate(fips.aggregate.mean(p.obs.space, by=DAILY)).offset
        assert isinstance(offset, Vector) and offset.name == "background"
        np.testing.assert_allclose(offset.values, [1.5, 3.5])

    def test_scalar_offset_stays_the_same_under_a_mean(self, problem):
        """A number for every observation averages to itself."""
        p = _with(problem, offset=5.0)
        W = fips.aggregate.mean(p.obs.space, by=DAILY)
        np.testing.assert_allclose(p.aggregate(W).offset.values, 5.0)

    def test_columns_matched_by_label(self, problem):
        """W with its columns in another order aggregates by label."""
        values = np.arange(12.0).reshape(4, 3)
        p = _with(
            problem,
            obs_cov=Covariance.diagonal([1.0, 2.0, 3.0, 4.0], problem.obs.space),
            forward=Operator(values, problem.obs.space, STATE),
        )
        W = fips.aggregate.mean(p.obs.space, by=DAILY)
        backwards = Space.from_index(p.obs.index[::-1])
        daily = p.aggregate(W.align(W.rows, backwards))
        np.testing.assert_allclose(
            daily.forward.dense(), [values[:2].mean(axis=0), values[2:].mean(axis=0)]
        )
        np.testing.assert_allclose(daily.obs_cov.dense(), np.diag([0.75, 1.75]))

    def test_other_labels_raise(self, problem):
        """W whose columns are not exactly the observations raises, naming them."""
        W = fips.aggregate.mean(Space({"obs": TIMES[:3]}), by=DAILY)
        with pytest.raises(
            ValueError, match="The columns of the aggregation and the observations"
        ):
            problem.aggregate(W)
        with pytest.raises(TypeError, match="An aggregation is an Operator"):
            problem.aggregate(np.eye(4))

    def test_aggregated_problem_solves(self, problem):
        """The aggregated problem solves like any other."""
        daily = problem.aggregate(fips.aggregate.mean(problem.obs.space, by=DAILY))
        assert len(fips.solve(daily).modeled_obs) == 2
