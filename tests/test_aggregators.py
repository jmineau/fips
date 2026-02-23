"""Tests for fips aggregators module."""

import numpy as np
import pandas as pd
import pytest

from fips.aggregators import ObsAggregator, integrate_over_time_bins
from fips.covariance import CovarianceMatrix
from fips.indexes import assign_block
from fips.operators import ForwardOperator
from fips.vector import Vector

# ---------------------------------------------------------------------------
# integrate_over_time_bins tests
# ---------------------------------------------------------------------------


def test_integrate_over_time_bins_series():
    """Test integrating a Series over time bins."""
    idx = pd.date_range("2023-01-01", periods=10, freq="h", name="time")
    data = pd.Series(1.0, index=idx, name="flux")

    bins = pd.interval_range(
        start=pd.Timestamp("2023-01-01 00:00"), periods=5, freq="2h", closed="left"
    )

    result = integrate_over_time_bins(data, bins, time_dim="time")

    assert len(result) == 5
    assert result.index.name == "time"
    assert (result.values == 2.0).all()


def test_integrate_over_time_bins_dataframe_multiindex():
    """Test integrating DataFrame with MultiIndex."""
    times = pd.date_range("2023-01-01", periods=4, freq="h")
    locs = ["A", "B"]
    idx = pd.MultiIndex.from_product([times, locs], names=["time", "loc"])
    df = pd.DataFrame({"value": 1.0}, index=idx)

    bins = pd.interval_range(
        start=pd.Timestamp("2023-01-01 00:00"), periods=2, freq="2h", closed="left"
    )

    result = integrate_over_time_bins(df, bins, time_dim="time")

    assert len(result) == 4
    assert set(result.index.names) == {"time", "loc"}
    assert (result["value"] == 2.0).all()


def test_integrate_raises_missing_dim():
    """Test error when time_dim is missing."""
    data = pd.Series([1, 2], index=pd.Index([1, 2], name="other"))
    bins = pd.interval_range(start=0, periods=2, freq=1)

    with pytest.raises(ValueError, match="not found"):
        integrate_over_time_bins(data, bins, time_dim="time")


# ---------------------------------------------------------------------------
# Fixtures for ObsAggregator tests
# ---------------------------------------------------------------------------
# 4 hourly obs, 2 per day — block-wrapped as InverseProblem would see them


@pytest.fixture
def obs_idx():
    # 2 obs per day across 2 days
    times = pd.DatetimeIndex(
        [
            "2023-01-01 00:00",
            "2023-01-01 12:00",
            "2023-01-02 00:00",
            "2023-01-02 12:00",
        ],
        name="obs_time",
    )
    return assign_block(times, "obs")


@pytest.fixture
def state_idx():
    return assign_block(pd.Index(["s0", "s1", "s2"], name="state_id"), "state")


@pytest.fixture
def obs_vec(obs_idx):
    return Vector(pd.Series([1.0, 2.0, 3.0, 4.0], index=obs_idx, name="obs"))


@pytest.fixture
def fwd_op(obs_idx, state_idx):
    return ForwardOperator(
        pd.DataFrame(np.ones((4, 3)) * 0.5, index=obs_idx, columns=state_idx)
    )


@pytest.fixture
def cov_Sz(obs_idx):
    return CovarianceMatrix(
        pd.DataFrame(np.eye(4) * 2.0, index=obs_idx, columns=obs_idx)
    )


def daily_groupby(idx):
    """Group obs_time to daily frequency."""
    return idx.get_level_values("obs_time").floor("D")


# ---------------------------------------------------------------------------
# ObsAggregator tests
# ---------------------------------------------------------------------------


class TestObsAggregator:
    def test_invalid_func_raises(self):
        with pytest.raises(ValueError, match="func"):
            ObsAggregator(by="obs_time", func="median")

    def test_missing_by_and_level_raises(self):
        with pytest.raises(ValueError):
            ObsAggregator()

    def test_mean_weights_sum_to_one(self, obs_idx):
        agg = ObsAggregator(by=daily_groupby, func="mean")
        W, _ = agg._build_operator(obs_idx)
        row_sums = np.asarray(W.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 1.0)

    def test_sum_weights(self, obs_idx):
        """Each row of W should equal n_obs_in_group for sum."""
        agg = ObsAggregator(by=daily_groupby, func="sum")
        W, _ = agg._build_operator(obs_idx)
        # 2 obs per day
        row_sums = np.asarray(W.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 2.0)

    def test_apply_obs_mean_values(self, obs_vec, fwd_op, cov_Sz):
        """Aggregated obs should be daily means: [1.5, 3.5]."""
        agg = ObsAggregator(by=daily_groupby, func="mean")
        new_obs, _, _, _ = agg.apply(obs_vec, fwd_op, cov_Sz)
        np.testing.assert_allclose(sorted(new_obs.values), [1.5, 3.5])

    def test_apply_reduces_obs_count(self, obs_vec, fwd_op, cov_Sz):
        agg = ObsAggregator(by=daily_groupby, func="mean")
        new_obs, new_H, new_Sz, new_c = agg.apply(obs_vec, fwd_op, cov_Sz)
        assert new_obs.shape[0] == 2
        assert new_H.shape == (2, 3)
        assert new_Sz.shape == (2, 2)
        assert new_c is None

    def test_apply_returns_correct_types(self, obs_vec, fwd_op, cov_Sz):
        agg = ObsAggregator(by=daily_groupby, func="mean")
        new_obs, new_H, new_Sz, _ = agg.apply(obs_vec, fwd_op, cov_Sz)
        assert isinstance(new_obs, Vector)
        assert isinstance(new_H, ForwardOperator)
        assert isinstance(new_Sz, CovarianceMatrix)

    def test_Sz_diagonal_mean(self, obs_vec, fwd_op, cov_Sz):
        """Diagonal S_z=2I mean-aggregated with 2 obs/group → S_z_agg = I (=2/2)."""
        agg = ObsAggregator(by=daily_groupby, func="mean")
        _, _, new_Sz, _ = agg.apply(obs_vec, fwd_op, cov_Sz)
        np.testing.assert_allclose(np.diag(new_Sz.values), 1.0)
        off = new_Sz.values - np.diag(np.diag(new_Sz.values))
        np.testing.assert_allclose(off, 0.0, atol=1e-12)

    def test_Sz_is_symmetric(self, obs_vec, fwd_op, cov_Sz):
        agg = ObsAggregator(by=daily_groupby, func="mean")
        _, _, new_Sz, _ = agg.apply(obs_vec, fwd_op, cov_Sz)
        np.testing.assert_allclose(new_Sz.values, new_Sz.values.T, atol=1e-12)

    def test_apply_aggregates_constant_vector(self, obs_vec, fwd_op, cov_Sz):
        agg = ObsAggregator(by=daily_groupby, func="mean")
        c = Vector(pd.Series(obs_vec.values, index=obs_vec.index, name="constant"))
        _, _, _, new_c = agg.apply(obs_vec, fwd_op, cov_Sz, constant=c)
        assert isinstance(new_c, Vector)
        assert new_c.shape[0] == 2

    def test_apply_scalar_constant_passthrough(self, obs_vec, fwd_op, cov_Sz):
        """Scalar constant should be returned unchanged."""
        agg = ObsAggregator(by=daily_groupby, func="mean")
        _, _, _, new_c = agg.apply(obs_vec, fwd_op, cov_Sz, constant=5.0)
        assert new_c == 5.0

    def test_level_freq_aggregation(self, obs_vec, fwd_op, cov_Sz):
        """level + freq interface should aggregate the same as by= callable."""
        agg_by = ObsAggregator(by=daily_groupby, func="mean")
        agg_freq = ObsAggregator(level="obs_time", freq="D", func="mean")
        obs_by, _, _, _ = agg_by.apply(obs_vec, fwd_op, cov_Sz)
        obs_freq, _, _, _ = agg_freq.apply(obs_vec, fwd_op, cov_Sz)
        np.testing.assert_allclose(sorted(obs_by.values), sorted(obs_freq.values))

    def test_blocks_filter_only_aggregates_target_block(self, obs_idx, state_idx):
        """With blocks=, untargeted blocks should pass through as identity rows."""
        # Build a two-block obs index: "obs" (datetime) + "bias" (also datetime, different day)
        bias_times = pd.DatetimeIndex(
            ["2023-01-03 00:00", "2023-01-03 12:00"], name="obs_time"
        )
        bias_idx = assign_block(bias_times, "bias")
        combined_idx = obs_idx.append(bias_idx)

        n_total = len(combined_idx)
        z = Vector(pd.Series(np.ones(n_total), index=combined_idx, name="obs"))
        H = ForwardOperator(
            pd.DataFrame(
                np.ones((n_total, 3)) * 0.5, index=combined_idx, columns=state_idx
            )
        )
        S_z = CovarianceMatrix(
            pd.DataFrame(np.eye(n_total), index=combined_idx, columns=combined_idx)
        )

        agg = ObsAggregator(by=daily_groupby, func="mean", blocks="obs")
        new_obs, new_H, new_Sz, _ = agg.apply(z, H, S_z)

        # "obs" block: 4 → 2 rows. "bias" block: 2 pass-through rows. Total = 4.
        assert new_obs.shape[0] == 4
        assert new_H.shape[0] == 4

    def test_build_operator_returns_sparse(self, obs_idx):
        """_build_operator should return a scipy sparse matrix."""
        from scipy.sparse import issparse

        agg = ObsAggregator(by=daily_groupby, func="mean")
        W, _ = agg._build_operator(obs_idx)
        assert issparse(W)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_obs_aggregator_hook():
    """Pipeline subclass overriding aggregate_obs_space should solve with n_obs=2."""
    from fips.pipeline import InversionPipeline
    from fips.problem import InverseProblem

    # 2 obs per day over 2 days → 4 total, 2 groups
    times = pd.DatetimeIndex(
        [
            "2023-01-01 00:00",
            "2023-01-01 12:00",
            "2023-01-02 00:00",
            "2023-01-02 12:00",
        ],
        name="obs_time",
    )
    state_idx = pd.Index(["s0", "s1", "s2"], name="state_id")

    class SimplePipeline(InversionPipeline):
        def get_obs(self):
            return pd.Series([1.0, 2.0, 3.0, 4.0], index=times, name="obs")

        def get_prior(self):
            return pd.Series([0.0, 0.0, 0.0], index=state_idx, name="prior")

        def get_forward_operator(self, obs, prior):
            return pd.DataFrame(
                np.ones((len(obs), len(prior))) * 0.5,
                index=obs.index,
                columns=prior.index,
            )

        def get_prior_error(self, prior):
            return pd.DataFrame(
                np.eye(len(prior)), index=prior.index, columns=prior.index
            )

        def get_modeldata_mismatch(self, obs):
            return pd.DataFrame(
                np.eye(len(obs)) * 2.0, index=obs.index, columns=obs.index
            )

        def aggregate_obs_space(self, obs, forward_operator, mdm, constant):
            agg = ObsAggregator(by=daily_groupby, func="mean")
            return agg.apply(obs, forward_operator, mdm, constant)

    result = SimplePipeline(
        config=None, problem=InverseProblem, estimator="bayesian"
    ).run()

    assert result.n_obs == 2, f"Expected 2 aggregated obs, got {result.n_obs}"
