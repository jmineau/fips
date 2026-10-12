"""Test suite for fips.problems.flux.pipeline module."""

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fips.matrix import MatrixBlock
from fips.problems.flux.pipeline import FluxInversionPipeline
from fips.problems.flux.problem import FluxProblem
from fips.vector import Block, Vector


class _Pipeline(FluxInversionPipeline):
    """A flux pipeline whose abstract steps are never called."""

    def get_obs(self):
        raise NotImplementedError

    def get_prior(self):
        raise NotImplementedError

    def get_forward_operator(self, obs, prior):
        raise NotImplementedError

    def get_prior_error(self, prior):
        raise NotImplementedError

    def get_modeldata_mismatch(self, obs):
        raise NotImplementedError


def _vectors(times):
    """A prior with the given ``time`` level, and six observations 12 hours apart."""
    prior_idx = pd.MultiIndex.from_product([times, [0, 1]], names=["time", "cell"])
    prior = Vector(
        Block(pd.Series(np.ones(len(prior_idx)), index=prior_idx, name="flux"))
    )
    obs_idx = pd.MultiIndex.from_arrays(
        [["A"] * 6, pd.date_range("2023-01-01", periods=6, freq="12h")],
        names=["obs_location", "obs_time"],
    )
    obs = Vector(
        Block(pd.Series(np.arange(6.0) + 400, index=obs_idx, name="concentration"))
    )
    return obs, prior


DAYS = pd.date_range("2023-01-01", periods=3, freq="D")

# The daily bins are closed on the right, so they hold two observations each
# except the last, which holds the one at noon on January 3; that one is dropped.
KEPT = pd.date_range("2023-01-01 12:00", periods=4, freq="12h")


class TestFilterStateSpace:
    """Tests for FluxInversionPipeline.filter_state_space."""

    def test_nothing_to_filter_by_default(self):
        """Without a threshold, the vectors come back as they went in."""
        obs, prior = _vectors(DAYS)

        obs_out, prior_out = _Pipeline(SimpleNamespace()).filter_state_space(obs, prior)

        assert obs_out is obs
        assert prior_out is prior

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="inferred"),
            pytest.param({"flux_freq": "D"}, id="given"),
        ],
    )
    def test_intervals_with_too_few_observations_are_dropped(self, config):
        """A flux interval needs ``min_obs_per_interval`` observations to keep them."""
        obs, prior = _vectors(DAYS)
        pipeline = _Pipeline(SimpleNamespace(min_obs_per_interval=2, **config))

        obs_out, prior_out = pipeline.filter_state_space(obs, prior)

        assert list(obs_out.index.get_level_values("obs_time")) == list(KEPT)
        assert prior_out is prior

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({}, id="inferred"),
            pytest.param({"flux_freq": "D"}, id="given"),
        ],
    )
    def test_object_timestamps_are_read_as_datetimes(self, config):
        """A ``time`` level of timestamps stored as objects is datetimes in a Vector."""
        obs, prior = _vectors(pd.Index(list(DAYS), dtype=object))
        pipeline = _Pipeline(SimpleNamespace(min_obs_per_interval=2, **config))

        obs_out, _ = pipeline.filter_state_space(obs, prior)

        assert isinstance(prior.index.get_level_values("time"), pd.DatetimeIndex)
        assert list(obs_out.index.get_level_values("obs_time")) == list(KEPT)

    def test_interval_times_are_the_bins(self):
        """A ``time`` level of intervals is used as the bins."""
        bins = pd.interval_range(pd.Timestamp("2023-01-01"), periods=3, freq="D")
        obs, prior = _vectors(bins)
        pipeline = _Pipeline(SimpleNamespace(min_obs_per_interval=2))

        obs_out, _ = pipeline.filter_state_space(obs, prior)

        assert list(obs_out.index.get_level_values("obs_time")) == list(KEPT)

    def test_irregular_times_cannot_be_inferred(self):
        """The frequency of an irregular index is not inferred."""
        times = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-05"])
        obs, prior = _vectors(times)
        pipeline = _Pipeline(SimpleNamespace(min_obs_per_interval=2))

        with pytest.raises(ValueError, match="Could not infer frequency"):
            pipeline.filter_state_space(obs, prior)


def _solved_problem(times):
    """A solved flux problem: ``times`` x 2 cells of flux, one site of daily data."""
    rng = np.random.default_rng(0)
    flux_idx = pd.MultiIndex.from_product([times, [0, 1]], names=["time", "cell"])
    obs_idx = pd.MultiIndex.from_arrays(
        [["A"] * 8, pd.date_range("2023-01-01", periods=8, freq="D")],
        names=["obs_location", "obs_time"],
    )
    n_flux, n_obs = len(flux_idx), len(obs_idx)
    return FluxProblem(
        obs=Block(
            pd.Series(rng.uniform(400, 410, n_obs), index=obs_idx, name="concentration")
        ),
        prior=Block(pd.Series(rng.uniform(1, 2, n_flux), index=flux_idx, name="flux")),
        forward_operator=MatrixBlock(
            pd.DataFrame(
                rng.uniform(0, 1, (n_obs, n_flux)), index=obs_idx, columns=flux_idx
            ),
            row_block="concentration",
            col_block="flux",
        ),
        prior_error=MatrixBlock(
            pd.DataFrame(np.eye(n_flux) * 0.5, index=flux_idx, columns=flux_idx),
            row_block="flux",
            col_block="flux",
        ),
        modeldata_mismatch=MatrixBlock(
            pd.DataFrame(np.eye(n_obs) * 0.1, index=obs_idx, columns=obs_idx),
            row_block="concentration",
            col_block="concentration",
        ),
        constant=Block(
            pd.Series(np.full(n_obs, 395.0), index=obs_idx, name="concentration")
        ),
    ).solve()


def _summary(pipeline, problem, capsys):
    """Print ``problem``'s summary through ``pipeline`` and return what it printed."""
    pipeline.problem = problem
    pipeline.summarize()
    return capsys.readouterr().out


class TestSummarize:
    """Tests for FluxInversionPipeline.summarize."""

    def test_report(self, capsys):
        """The report gives the sizes of the problem and the flux per timestep."""
        times = pd.date_range("2023-01-01", periods=4, freq="MS")
        out = _summary(_Pipeline(SimpleNamespace()), _solved_problem(times), capsys)

        assert "FLUX INVERSION SUMMARY REPORT" in out
        assert "State Vector Size (n):       8" in out
        assert "Observation Vector Size (m): 8" in out
        assert "MEAN FLUX BY TIMESTEP:" in out
        assert "2023-04-01" in out

    def test_trend_is_in_the_flux_per_year(self, capsys):
        """The trend is a regression of the mean posterior flux on the time in years."""
        times = pd.date_range("2023-01-01", periods=4, freq="MS")
        problem = _solved_problem(times)
        out = _summary(_Pipeline(SimpleNamespace()), problem, capsys)

        by_time = problem.posterior_fluxes.groupby(level="time").mean()
        years = by_time.index.year + (by_time.index.day_of_year - 1) / 365.25
        slope, intercept = np.polyfit(years, by_time.values, 1)
        assert "POSTERIOR FLUX TREND:" in out
        assert float(
            re.search(r"Slope: +([-+.\d]+) /yr", out).group(1)
        ) == pytest.approx(slope, abs=1e-4)

    def test_no_trend_from_two_timesteps(self, capsys):
        """A trend needs more than two timesteps."""
        times = pd.date_range("2023-01-01", periods=2, freq="MS")
        out = _summary(_Pipeline(SimpleNamespace()), _solved_problem(times), capsys)

        assert "MEAN FLUX BY TIMESTEP:" in out
        assert "POSTERIOR FLUX TREND:" not in out

    def test_no_trend_needed_for_other_times(self, capsys):
        """With two timesteps the times are not read, so any kind of index works."""
        out = _summary(_Pipeline(SimpleNamespace()), _solved_problem([0, 1]), capsys)

        assert "MEAN FLUX BY TIMESTEP:" in out
        assert "POSTERIOR FLUX TREND:" not in out
