"""
Tests for fips.offset_block: a block of offsets in the state, one per group of observations.

Ported from the downstream package's tests of its bias block (one offset per
time interval, per site or per group of sites), with synthetic sites and
groups of the same shape: there the groups came from its site configuration,
here they are a mapping given to ``by``.
"""

import numpy as np
import pandas as pd
import pytest

import fips
from fips import Covariance, Exponential, Independent, Operator, Problem, Space, Vector
from fips.linalg import DiagonalOperator, SparseOperator

# Three months, as the downstream tests' 2020-01-01 to 2020-04-01 monthly bins
MONTHS = pd.interval_range(
    start=pd.Timestamp("2020-01-01"),
    end=pd.Timestamp("2020-04-01"),
    freq="MS",
    closed="left",
    name="obs_time",
)
TIME = pd.Series(MONTHS.left, index=MONTHS, name="time")
"""The months as a key: each observation's month, by its start, on a level named ``time``."""

GROUPS = pd.Series({"wbb": "UATAQ", "hw": "DAQ"}, name="site_group").rename_axis(
    "obs_location"
)
"""Sites to the organization that runs them."""


def _obs(locations, times, block="concentration") -> Space:
    """The observations' labels: (obs_location, obs_time), one block."""
    index = pd.MultiIndex.from_arrays(
        [locations, pd.to_datetime(times)], names=["obs_location", "obs_time"]
    )
    return Space({block: index})


class TestOffsetsByTime:
    """One offset per month (the default grouping downstream)."""

    @pytest.fixture
    def offsets(self):
        """The offsets of observations in each of the three months."""
        obs = _obs(["a", "b", "c"], ["2020-01-15", "2020-02-10", "2020-03-20"])
        return fips.offset_block(obs, by=TIME)

    def test_prior_is_a_vector_of_zeros(self, offsets):
        """The prior: a Vector of zeros, one block named "offset"."""
        prior, _ = offsets
        assert isinstance(prior, Vector)
        assert prior.space.names == ("offset",)
        np.testing.assert_array_equal(prior.values, 0.0)

    def test_one_offset_per_month(self, offsets):
        """Three months with observations, three offsets, labeled by each month's start."""
        prior, _ = offsets
        labels = prior.space.select("offset")
        assert labels.name == "time"
        assert len(labels) == 3
        assert labels[0] == pd.Timestamp("2020-01-01")
        assert labels[-1] == pd.Timestamp("2020-03-01")

    def test_each_obs_maps_to_exactly_one_offset(self, offsets):
        """Every row of the forward block has one 1."""
        _, forward = offsets
        assert isinstance(forward, Operator)
        # sparse, or its diagonal when each offset has one observation
        assert isinstance(forward.data, (SparseOperator, DiagonalOperator))
        np.testing.assert_array_equal(forward.dense().sum(axis=1), 1.0)

    def test_obs_maps_to_its_month(self):
        """An observation of 2020-02-15 has February's offset, not January's."""
        obs = _obs(["a", "a"], ["2020-01-03", "2020-02-15"])
        prior, forward = fips.offset_block(obs, by=TIME)
        frame = forward.to_frame().droplevel("block", axis=1)
        feb = frame[pd.Timestamp("2020-02-01")]
        assert feb.iloc[1] == 1.0 and feb.iloc[0] == 0.0
        assert frame[pd.Timestamp("2020-01-01")].iloc[1] == 0.0

    def test_columns_are_the_offsets(self, offsets):
        """The forward block's columns are the prior's space."""
        prior, forward = offsets
        assert forward.cols == prior.space
        assert forward.cols.names == ("offset",)

    def test_a_month_without_observations_has_no_offset(self):
        """Observations in January and March: two offsets (one for February would see nothing)."""
        obs = _obs(["a", "a"], ["2020-01-15", "2020-03-15"])
        prior, _ = fips.offset_block(obs, by=TIME)
        assert list(prior.space.select("offset")) == [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-03-01"),
        ]

    def test_an_observation_outside_every_month_raises(self):
        """An observation in April is named, not left without an offset."""
        obs = _obs(["a", "a"], ["2020-01-15", "2020-04-02"])
        with pytest.raises(ValueError, match="outside every interval"):
            fips.offset_block(obs, by=TIME)


class TestOffsetsBySite:
    """One offset per month and site, and per month and group of sites."""

    def test_per_site(self):
        """(time, obs_location): each observation's month at its site."""
        obs = _obs(["wbb", "hw", "wbb"], ["2020-01-15", "2020-01-20", "2020-02-05"])
        prior, forward = fips.offset_block(obs, by=[TIME, "obs_location"])
        labels = prior.space.select("offset")
        assert list(labels.names) == ["time", "obs_location"]
        assert len(labels) == 3
        frame = forward.to_frame().droplevel("block", axis=1)
        assert frame[(pd.Timestamp("2020-01-01"), "hw")].tolist() == [0.0, 1.0, 0.0]

    def test_per_site_group(self):
        """(time, site_group): the sites of one organization share an offset."""
        obs = _obs(
            ["wbb", "hw", "wbb", "hw", "wbb", "hw"],
            [
                "2020-01-15",
                "2020-01-15",
                "2020-02-10",
                "2020-02-10",
                "2020-03-05",
                "2020-03-05",
            ],
        )
        prior, forward = fips.offset_block(obs, by=[TIME, GROUPS])
        labels = prior.space.select("offset")
        assert list(labels.names) == ["time", "site_group"]
        assert len(labels) == 2 * 3
        assert set(labels.get_level_values("site_group")) == {"UATAQ", "DAQ"}
        frame = forward.to_frame().droplevel("block", axis=1)
        assert frame[(pd.Timestamp("2020-01-01"), "UATAQ")].iloc[0] == 1.0
        assert frame[(pd.Timestamp("2020-01-01"), "DAQ")].iloc[0] == 0.0
        assert frame[(pd.Timestamp("2020-02-01"), "DAQ")].iloc[3] == 1.0
        np.testing.assert_array_equal(forward.dense().sum(axis=1), 1.0)

    def test_a_site_without_a_group_raises(self):
        """A site the mapping lacks is named (downstream put it in an "unknown" group)."""
        obs = _obs(["wbb", "nonexistent"], ["2020-01-15", "2020-01-15"])
        with pytest.raises(ValueError, match="'nonexistent'"):
            fips.offset_block(obs, by=[TIME, GROUPS])

    def test_through_a_location_map(self):
        """Receptor locations to sites to groups: compose the mappings, one key."""
        location_site = pd.Series({"loc_a": "wbb"}).rename_axis("obs_location")
        location_group = location_site.map(GROUPS).rename("site_group")
        obs = _obs(["loc_a"], ["2020-01-05"])
        _, forward = fips.offset_block(obs, by=[TIME, location_group])
        frame = forward.to_frame().droplevel("block", axis=1)
        assert frame.loc[:, (pd.Timestamp("2020-01-01"), "UATAQ")].tolist() == [1.0]


class TestBlocks:
    """Which observations get an offset."""

    def test_other_blocks_get_none(self):
        """Observations of a block not named have rows of zeros."""
        times = pd.DatetimeIndex(["2020-01-15", "2020-02-15"], name="obs_time")
        obs = Space({"co2": times, "ch4": times})
        prior, forward = fips.offset_block(obs, by=TIME, blocks="co2")
        assert len(prior) == 2
        np.testing.assert_array_equal(forward["ch4", "offset"].dense(), 0.0)
        np.testing.assert_array_equal(forward["co2", "offset"].dense(), np.eye(2))

    def test_named_blocks_share_offsets(self):
        """Two blocks of one quantity share the offset of each month."""
        times = pd.DatetimeIndex(["2020-01-15", "2020-02-15"], name="obs_time")
        obs = Space({"tower": times, "mobile": times[:1]})
        prior, forward = fips.offset_block(obs, by=TIME, block="background")
        assert prior.space.names == ("background",)
        assert len(prior) == 2
        np.testing.assert_array_equal(
            forward.dense(), [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        )

    def test_not_a_space_raises(self):
        """The observations' Space, not its index."""
        with pytest.raises(TypeError, match="obs_space must be a Space"):
            fips.offset_block(_obs(["a"], ["2020-01-15"]).index, by=TIME)


def test_offsets_in_a_problem():
    """Offsets beside a flux, correlated between months (#28), recover a background shift."""
    obs = _obs(["a"] * 6, pd.date_range("2020-01-10", periods=6, freq="12D"))
    prior_offsets, H_offsets = fips.offset_block(obs, by=TIME)
    cells = pd.Index(["c0", "c1"], name="cell")
    flux = pd.Series([1.0, 1.0], index=cells)
    prior = Vector.from_blocks({"flux": flux, "offset": prior_offsets})
    assert prior.space.names == ("flux", "offset")
    H_flux = np.array([[1.0, 0.0], [0.0, 1.0]] * 3)
    forward = Operator.from_blocks(
        {
            ("concentration", "flux"): H_flux,
            ("concentration", "offset"): H_offsets["concentration", "offset"],
        },
        rows=obs,
        cols=prior.space,
    )
    truth = np.concatenate([[1.0, 1.0], [0.5, 1.0, 1.5]])
    z = Vector(forward.dense() @ truth, obs)
    prior_cov = Covariance.block_diag(
        flux=Independent().with_std(1e-3),
        offset=Exponential("time", "60D").with_std(2.0),
    )
    problem = Problem(
        prior=prior,
        prior_cov=prior_cov,
        obs=z,
        obs_cov=Independent().with_std(1e-3),
        forward=forward,
    )
    S = problem.prior_cov["offset", "offset"].dense()
    assert S[0, 1] > 0.0  # the months are correlated
    post = fips.solve(problem)
    np.testing.assert_allclose(
        post.estimate["offset"].values, [0.5, 1.0, 1.5], atol=1e-4
    )
