"""Test suite for fips.problems.flux.problem module."""

import numpy as np
import pandas as pd
import pytest

import fips
from fips import Covariance, Operator
from fips.problems.flux.problem import FluxProblem
from fips.vector import Vector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flux_idx():
    """4-cell (2 time × 2 location) flux state index."""
    return pd.MultiIndex.from_product(
        [pd.date_range("2023-01", periods=2, freq="MS"), ["A", "B"]],
        names=["time", "loc"],
    )


@pytest.fixture
def obs_idx():
    """4-observation index."""
    return pd.MultiIndex.from_tuples(
        [(t, "SITE") for t in pd.date_range("2023-01-01", periods=4, freq="2W")],
        names=["time", "site"],
    )


@pytest.fixture
def flux_prior(flux_idx):
    """Prior flux, one block."""
    return Vector.from_series(
        pd.Series([1.0, 2.0, 3.0, 4.0], index=flux_idx, name="flux"), block="flux"
    )


@pytest.fixture
def flux_obs(obs_idx):
    """Observed concentrations, one block."""
    return Vector.from_series(
        pd.Series([400.0, 401.0, 402.0, 403.0], index=obs_idx, name="concentration"),
        block="concentration",
    )


@pytest.fixture
def flux_jacobian(obs_idx, flux_idx):
    """Jacobian Operator mapping flux → concentration."""
    H = np.eye(4)
    return Operator.from_frame(
        pd.DataFrame(H, index=obs_idx, columns=flux_idx),
        row_block="concentration",
        col_block="flux",
    )


@pytest.fixture
def flux_prior_error(flux_idx):
    """Prior flux error Covariance (diagonal)."""
    return Covariance.from_frame(
        pd.DataFrame(np.eye(4) * 0.5, index=flux_idx, columns=flux_idx),
        row_block="flux",
        col_block="flux",
    )


@pytest.fixture
def flux_mismatch(obs_idx):
    """Observation error Covariance (diagonal)."""
    return Covariance.from_frame(
        pd.DataFrame(np.eye(4) * 0.1, index=obs_idx, columns=obs_idx),
        row_block="concentration",
        col_block="concentration",
    )


@pytest.fixture
def flux_problem(flux_obs, flux_prior, flux_jacobian, flux_prior_error, flux_mismatch):
    """Unsolved FluxProblem."""
    return FluxProblem(
        obs=flux_obs,
        prior=flux_prior,
        forward=flux_jacobian,
        prior_cov=flux_prior_error,
        obs_cov=flux_mismatch,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFluxProblemConstruction:
    """Tests for FluxProblem initialization."""

    def test_creates_successfully(self, flux_problem):
        """Test FluxProblem creation succeeds."""
        assert isinstance(flux_problem, FluxProblem)

    def test_n_state(self, flux_problem):
        """Test n_state property returns correct count."""
        assert len(flux_problem.prior) == 4

    def test_n_obs(self, flux_problem):
        """Test n_obs property returns correct count."""
        assert len(flux_problem.obs) == 4

    def test_has_no_solve(self, flux_problem):
        """A flux problem is solved by fips.solve, as any problem."""
        assert not hasattr(flux_problem, "solve")
        assert not hasattr(flux_problem, "result")
        assert isinstance(fips.solve(flux_problem), fips.Posterior)


# ---------------------------------------------------------------------------
# Observation-space properties (before solving)
# ---------------------------------------------------------------------------


class TestFluxProblemObsProperties:
    """Tests for FluxProblem observation-space accessors."""

    def test_concentrations_is_series(self, flux_problem):
        """Test concentrations property returns a Series."""
        result = flux_problem.concentrations
        assert isinstance(result, pd.Series)
        assert len(result) == 4

    def test_concentrations_values(self, flux_problem):
        """Test concentrations values are correct."""
        result = flux_problem.concentrations
        np.testing.assert_allclose(result.values, [400.0, 401.0, 402.0, 403.0])

    def test_enhancement_no_background(self, flux_problem):
        """Test enhancement without background equals concentrations."""
        """Without a background, enhancement equals concentrations."""
        result = flux_problem.enhancement
        pd.testing.assert_series_equal(result, flux_problem.concentrations)

    def test_enhancement_with_background(
        self,
        flux_obs,
        flux_prior,
        flux_jacobian,
        flux_prior_error,
        flux_mismatch,
        obs_idx,
    ):
        """With a background, enhancement = concentrations - background."""
        background = Vector.from_series(
            pd.Series(np.ones(4) * 390.0, index=obs_idx, name="concentration"),
            block="concentration",
        )
        problem = FluxProblem(
            obs=flux_obs,
            prior=flux_prior,
            forward=flux_jacobian,
            prior_cov=flux_prior_error,
            obs_cov=flux_mismatch,
            offset=background,
        )
        result = problem.enhancement
        expected = flux_obs.values - 390.0
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_background_none_when_not_set(self, flux_problem):
        """Test that background is None when not set."""
        assert flux_problem.background is None

    def test_background_returns_series_when_set(
        self,
        flux_obs,
        flux_prior,
        flux_jacobian,
        flux_prior_error,
        flux_mismatch,
        obs_idx,
    ):
        """Test that background returns a Series when set."""
        background = Vector.from_series(
            pd.Series(np.ones(4) * 390.0, index=obs_idx, name="concentration"),
            block="concentration",
        )
        problem = FluxProblem(
            obs=flux_obs,
            prior=flux_prior,
            forward=flux_jacobian,
            prior_cov=flux_prior_error,
            obs_cov=flux_mismatch,
            offset=background,
        )
        result = problem.background
        assert isinstance(result, pd.Series)
        np.testing.assert_allclose(result.to_numpy(), 390.0)


# ---------------------------------------------------------------------------
# State-space properties (before solving)
# ---------------------------------------------------------------------------


class TestFluxProblemStateProperties:
    """Tests for FluxProblem state-space accessors."""

    def test_prior_fluxes_is_series(self, flux_problem):
        """Test that prior_fluxes returns a Series."""
        result = flux_problem.prior_fluxes
        assert isinstance(result, pd.Series)
        assert len(result) == 4

    def test_prior_fluxes_values(self, flux_problem):
        """Test that prior_fluxes has correct values."""
        np.testing.assert_allclose(
            flux_problem.prior_fluxes.values, [1.0, 2.0, 3.0, 4.0]
        )

    def test_jacobian_is_operator(self, flux_problem):
        """Test that jacobian returns the Operator block concentration x flux."""
        result = flux_problem.jacobian
        assert isinstance(result, Operator)
        assert result.shape == (4, 4)
        assert (result.rows.names, result.cols.names) == (("concentration",), ("flux",))

    def test_prior_flux_error_is_covariance(self, flux_problem):
        """Test that prior_flux_error returns the Covariance block of the fluxes."""
        result = flux_problem.prior_flux_error
        assert isinstance(result, Covariance)
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result.dense(), np.eye(4) * 0.5)

    def test_concentration_error_is_covariance(self, flux_problem):
        """Test that concentration_error returns the Covariance block of the obs."""
        result = flux_problem.concentration_error
        assert isinstance(result, Covariance)
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result.dense(), np.eye(4) * 0.1)


# ---------------------------------------------------------------------------
# What the problem gives without a solve
# ---------------------------------------------------------------------------


class TestFluxProblemPriorProperties:
    """The prior's concentrations and their error, from the problem alone."""

    def test_prior_concentration_error_is_on_the_obs_index(self, flux_problem):
        """The prior flux error mapped to concentrations, block-indexed like the obs."""
        S = flux_problem.prior_concentration_error
        assert isinstance(S, Covariance)
        n_obs = len(flux_problem.obs)
        assert S.shape == (n_obs, n_obs)
        assert S.space == flux_problem.jacobian.rows
        H = flux_problem.jacobian.dense()
        S_0 = flux_problem.prior_flux_error.dense()
        np.testing.assert_allclose(S.dense(), H @ S_0 @ H.T)

    def test_prior_concentrations_are_the_prediction(self, flux_problem):
        """``prior_concentrations`` is the concentration block of ``predict(prior)``."""
        result = flux_problem.prior_concentrations
        assert isinstance(result, pd.Series)
        assert len(result) == 4
        np.testing.assert_allclose(
            result.to_numpy(), flux_problem.predict(flux_problem.prior).values
        )


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestFluxProblemRepr:
    """Tests for FluxProblem __repr__."""

    def test_repr(self, flux_problem):
        """Test repr of a FluxProblem."""
        r = repr(flux_problem)
        assert r == "FluxProblem(n_obs=4, n_flux=4, n_state=4)"
