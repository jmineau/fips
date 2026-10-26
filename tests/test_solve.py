"""Tests for fips.solve: the methods, their options, and what a solve leaves alone."""

import numpy as np
import pandas as pd
import pytest

import fips
from fips import Covariance, Operator, Posterior, Problem, Space, Vector
from fips.methods.analytic import AnalyticOptions


def _problem(seed: int = 0, n: int = 3, m: int = 4, obs_var: float = 0.01) -> Problem:
    """A small dense problem with the true state 1, 2, 3."""
    rng = np.random.default_rng(seed)
    state = Space({"flux": pd.Index([f"x{k}" for k in range(n)], name="cell")})
    obs = Space({"co2": pd.Index([f"o{k}" for k in range(m)], name="obs_id")})
    H = rng.standard_normal((m, n))
    z = H @ np.arange(1.0, n + 1) + rng.standard_normal(m) * 0.1
    return Problem(
        prior=Vector(np.zeros(n), state),
        prior_cov=Covariance.diagonal(2.0, state),
        obs=Vector(z, obs),
        obs_cov=Covariance.diagonal(obs_var, obs),
        forward=Operator(H, obs, state),
    )


class TestSolve:
    def test_returns_a_posterior_with_its_estimate_computed(self):
        post = fips.solve(_problem())
        assert isinstance(post, Posterior)
        assert "estimate" in vars(post)

    def test_the_default_is_analytic_with_gamma_one(self):
        problem = _problem()
        post = fips.solve(problem)
        assert post.options == AnalyticOptions(gamma=1.0)
        np.testing.assert_array_equal(
            post.estimate.values,
            fips.solve(problem, "analytic", gamma=1.0).estimate.values,
        )

    def test_options_reach_the_posterior(self):
        post = fips.solve(_problem(), gamma=0.5)
        assert post.options.gamma == 0.5
        assert isinstance(post.options.gamma, float)

    def test_does_not_change_the_problem(self):
        problem = _problem()
        before = (problem.prior.values.copy(), problem.obs.values.copy())
        post = fips.solve(problem, gamma=3.0)
        assert post.problem is problem
        np.testing.assert_array_equal(problem.prior.values, before[0])
        np.testing.assert_array_equal(problem.obs.values, before[1])

    def test_available_methods(self):
        assert fips.available_methods() == ["analytic"]

    def test_unknown_method_lists_the_methods(self):
        with pytest.raises(
            ValueError, match=r"Unknown method 'analytc'; the methods are: analytic\."
        ):
            fips.solve(_problem(), "analytc")

    def test_unknown_option_lists_the_options(self):
        with pytest.raises(
            ValueError,
            match=r"Method 'analytic' has no option 'gama'; its options are: gamma\.",
        ):
            fips.solve(_problem(), gama=2.0)

    @pytest.mark.parametrize("gamma", [0.0, -1.0, np.inf, np.nan, "a", None])
    def test_gamma_must_be_positive_and_finite(self, gamma):
        with pytest.raises(ValueError, match="gamma must be"):
            fips.solve(_problem(), gamma=gamma)

    def test_a_zero_dimensional_array_is_a_gamma(self):
        assert fips.solve(_problem(), gamma=np.array(2.0)).options.gamma == 2.0

    def test_takes_only_a_problem(self):
        with pytest.raises(TypeError, match="solve takes a Problem"):
            fips.solve("problem")  # type: ignore[arg-type]

    def test_options_are_frozen(self):
        options = AnalyticOptions(gamma=2.0)
        with pytest.raises(AttributeError):
            options.gamma = 3.0  # type: ignore[misc]


class TestGamma:
    """gamma divides the model-data mismatch: larger fits the observations more."""

    @pytest.mark.parametrize("gamma", [0.1, 0.5, 2.0, 10.0])
    def test_is_the_mismatch_divided_by_gamma(self, gamma):
        problem = _problem()
        scaled = Problem(
            prior=problem.prior,
            prior_cov=problem.prior_cov,
            obs=problem.obs,
            obs_cov=problem.obs_cov / gamma,
            forward=problem.forward,
        )
        got, expected = fips.solve(problem, gamma=gamma), fips.solve(scaled)
        # Two routes to one posterior agree to rounding, and the rounding depends
        # on the BLAS (1.2e-12 apart on CI's runners): 1e-10, as the oracle uses.
        np.testing.assert_allclose(
            got.estimate.values, expected.estimate.values, rtol=1e-10
        )
        np.testing.assert_allclose(
            got.variances.values, expected.variances.values, rtol=1e-10
        )
        assert got.dofs == pytest.approx(expected.dofs, rel=1e-10)

    def test_directions(self):
        """More gamma: smaller variances, more degrees of freedom, a closer fit."""
        problem = _problem(obs_var=0.5)
        posts = [fips.solve(problem, gamma=g) for g in (0.1, 1.0, 10.0)]
        variances = [p.variances.values.sum() for p in posts]
        dofs = [p.dofs for p in posts]
        rmse = [p.rmse for p in posts]
        reduction = [p.uncertainty_reduction for p in posts]
        assert variances[0] > variances[1] > variances[2]
        assert dofs[0] < dofs[1] < dofs[2]
        assert rmse[0] > rmse[1] > rmse[2]
        assert reduction[0] < reduction[1] < reduction[2]

    def test_extreme_values_stay_finite(self):
        problem = _problem()
        for gamma in (1e-8, 1e8):
            post = fips.solve(problem, gamma=gamma)
            assert np.isfinite(post.estimate.values).all()
            assert np.isfinite(post.variances.values).all()
        # A tiny gamma keeps the prior (zero)
        np.testing.assert_allclose(
            fips.solve(problem, gamma=1e-8).estimate.values, 0.0, atol=1e-4
        )


def test_a_roundoff_indefinite_problem_solves():
    """
    A sub-yearly problem whose ``H S_0 H^T + S_z`` is positive-definite only in exact arithmetic solves.

    A smooth time correlation over many bins, a strong Jacobian and a tiny
    mismatch make the Cholesky factorization of the sum fail (that was the
    "singular matrix" of sub-yearly runs); the whitened solve never forms it.
    """
    rng = np.random.default_rng(0)
    n_time, n_grid, m = 10, 15, 200
    t, g = np.arange(n_time), np.arange(n_grid)
    T = np.exp(-((t[:, None] - t[None, :]) ** 2) / (2 * n_time**2))
    G = np.exp(-np.abs(g[:, None] - g[None, :]) / (n_grid / 3))
    S_0 = 100.0 * np.kron(T, G)
    H = np.abs(rng.standard_normal((m, n_time * n_grid))) * 50.0
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(H @ S_0 @ H.T + 1e-7 * np.eye(m))

    state = Space({"flux": pd.Index(range(n_time * n_grid), name="cell")})
    obs = Space({"co2": pd.Index(range(m), name="obs_id")})
    z = H @ rng.standard_normal(n_time * n_grid) + rng.standard_normal(m) * 1e-3
    post = fips.solve(
        Problem(
            prior=Vector(np.zeros(len(state)), state),
            prior_cov=Covariance(S_0, state),
            obs=Vector(z, obs),
            obs_cov=Covariance.diagonal(1e-7, obs),
            forward=Operator(H, obs, state),
        )
    )
    assert np.isfinite(post.estimate.values).all()
    assert np.isfinite(post.dofs)
    assert np.isfinite(post.cov().dense()).all()
