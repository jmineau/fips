"""
Tests for the prior side of the reduction of uncertainty: the prior the method used.

``prior_std`` and ``reduction`` (:meth:`fips.Posterior.uncertainty`) and
:attr:`fips.Posterior.uncertainty_reduction` compare the posterior with the
prior the method used, so observations that tell nothing reduce nothing:

1. **Nothing told, nothing reduced.** With ``H = 0`` (DOFS 0), every method
   gives a reduction of 0, for each element, for a total and for the whole
   state. Measured against the problem's Gaussian prior, eight correlated
   elements near a lower bound gave 0.19 (``uncertainty_reduction``) and
   0.28 (the total) for truncated draws, 0.16 and 0.24 for rectified ones,
   and up to 0.11 for a 20-member ensemble.
2. **The bounded prior, against exact answers.** One element: the prior's
   spread is ``scipy.stats.truncnorm``'s (truncated draws) or the clipped
   normal's (rectified draws), within Monte Carlo error. Several: it is the
   spread of the bounded problem with no observations, the same draws.
3. **The ensemble's prior** is its members' spread.
4. **Observations that inform** still reduce, for every method; the analytic
   method's prior side is still :math:`S_0`.
"""

import math
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import norm, truncnorm

import fips

N, M = 8, 6

METHODS = {
    "analytic": ("analytic", {}),
    "truncated": ("bounded", {"lower": 0.0}),
    "rectified": ("bounded", {"lower": 0.0, "distribution": "rectified"}),
    "ensemble": ("ensemble", {"members": 20, "seed": 1}),
}


def correlated(H_scale: float, seed: int = 0) -> fips.Problem:
    """Eight elements with a correlated prior of mean 0.5 and standard deviation 1, near a lower bound of 0, observed through ``H_scale`` times a random H."""
    rng = np.random.default_rng(seed)
    i = np.arange(N)
    S_0 = np.exp(-np.abs(i[:, None] - i[None, :]) / 3.0)
    state = fips.Space({"flux": pd.Index([f"c{k}" for k in range(N)], name="cell")})
    observed = fips.Space({"conc": pd.RangeIndex(M, name="obs_id")})
    return fips.Problem(
        prior=fips.Vector(np.full(N, 0.5), state),
        prior_cov=fips.Covariance(S_0, state),
        obs=fips.Vector(rng.normal(0.0, 1.0, M), observed),
        obs_cov=fips.Covariance(0.1 * np.eye(M), observed),
        forward=fips.Operator(H_scale * rng.uniform(0.0, 1.0, (M, N)), observed, state),
    )


def total(problem: fips.Problem) -> fips.Operator:
    """The sum of the state."""
    return fips.Operator(
        np.ones((1, len(problem.prior))),
        fips.Space({"total": pd.Index(["all"], name="region")}),
        problem.prior.space,
    )


@pytest.mark.parametrize("name", list(METHODS))
def test_observations_that_tell_nothing_reduce_nothing(name):
    """With ``H = 0`` every method's reduction is 0: for each element, for the total and for the whole state."""
    method, options = METHODS[name]
    problem = correlated(0.0)
    post = fips.solve(problem, method, **options)
    assert post.dofs == pytest.approx(0.0, abs=1e-12)
    # Within the Monte Carlo error of the draws: their prior and posterior
    # draws share their normals, so they come out the same draws
    atol = 1e-12 if name in ("analytic", "ensemble") else 0.02
    by_element = post.uncertainty()
    np.testing.assert_allclose(by_element["reduction"], 0.0, atol=atol)
    np.testing.assert_allclose(
        post.uncertainty(total(problem))["reduction"], 0.0, atol=atol
    )
    assert post.uncertainty_reduction == pytest.approx(0.0, abs=atol)


@pytest.mark.parametrize("name", list(METHODS))
def test_observations_that_inform_still_reduce(name):
    """With an informative H every method reduces the uncertainty of the total and of the whole state, and ``summary`` reports the same reduction."""
    method, options = METHODS[name]
    problem = correlated(1.0)
    post = fips.solve(problem, method, **options)
    assert post.uncertainty(total(problem))["reduction"].iloc[0] > 0.5
    assert post.uncertainty_reduction > 0.3
    assert post.summary()["uncertainty_reduction"].iloc[0] == post.uncertainty_reduction


@pytest.mark.parametrize("distribution", ["truncated", "rectified"])
def test_the_bounded_prior_is_the_bounded_problem_with_no_observations(distribution):
    """The bounded prior's spread, for each element and for the total, is the posterior spread of the same problem with ``H = 0``: the same draws."""
    informed = fips.solve(
        correlated(1.0), "bounded", lower=0.0, distribution=distribution
    )
    blind = fips.solve(correlated(0.0), "bounded", lower=0.0, distribution=distribution)
    W = total(informed.problem)
    np.testing.assert_allclose(
        informed.uncertainty()["prior_std"],
        blind.uncertainty()["posterior_std"],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        informed.uncertainty(W)["prior_std"],
        blind.uncertainty(W)["posterior_std"],
        rtol=1e-8,
    )
    # Not the problem's Gaussian prior: cut or set to the bound, it is narrower
    assert np.all(informed.uncertainty()["prior_std"] < 0.9)


def clipped_std(mean: float, sd: float, lower: float) -> float:
    """The standard deviation of ``max(x, lower)`` for ``x ~ N(mean, sd^2)``, by integration."""
    below = norm.cdf(lower, mean, sd)

    def moment(k: int) -> float:
        inside, _ = integrate.quad(
            lambda x: x**k * norm.pdf(x, mean, sd), lower, np.inf, epsabs=1e-13
        )
        return lower**k * below + inside

    first = moment(1)
    return math.sqrt(moment(2) - first**2)


@pytest.mark.parametrize("distribution", ["truncated", "rectified"])
def test_one_elements_prior_spread_is_the_cut_or_clipped_normal(distribution):
    """One element, observed: its ``prior_std`` is that of the prior cut at the bound (``truncnorm``) or set to it (clipped), within Monte Carlo error, and the data still reduce it."""
    mean, sd, lower = 0.3, 1.0, 0.0
    state = fips.Space({"flux": pd.Index(["a"], name="cell")})
    observed = fips.Space({"conc": pd.RangeIndex(3, name="obs_id")})
    problem = fips.Problem(
        prior=fips.Vector(np.array([mean]), state),
        prior_cov=fips.Covariance(np.array([[sd**2]]), state),
        obs=fips.Vector(np.array([0.8, 1.1, 0.9]), observed),
        obs_cov=fips.Covariance(0.5 * np.eye(3), observed),
        forward=fips.Operator(np.ones((3, 1)), observed, state),
    )
    post = fips.solve(
        problem, "bounded", lower=lower, draws=20_000, distribution=distribution
    )
    table = post.uncertainty()
    if distribution == "truncated":
        expected = float(truncnorm.std((lower - mean) / sd, np.inf, loc=mean, scale=sd))
    else:
        expected = clipped_std(mean, sd, lower)
    # 20000 draws: about 1% for independent ones, a few times that for a chain
    assert table["prior_std"].iloc[0] == pytest.approx(expected, rel=0.03)
    assert table["reduction"].iloc[0] > 0.2


def test_the_ensembles_prior_is_its_members_spread():
    """The ensemble's ``prior_std`` is the standard deviation of its prior members, for each element and for the total."""
    problem = correlated(1.0)
    members = problem.ensemble(15, rng=3)
    post = fips.solve(problem, "ensemble", members=members)
    np.testing.assert_allclose(
        post.uncertainty()["prior_std"].to_numpy(),
        members.loc[problem.prior.index].std(axis=1, ddof=1).to_numpy(),
        rtol=1e-12,
    )
    assert post.uncertainty(total(problem))["prior_std"].iloc[0] == pytest.approx(
        members.sum(axis=0).std(ddof=1), rel=1e-12
    )


def test_the_analytic_prior_side_is_the_problems_prior():
    """The analytic method still compares with :math:`S_0`: each element, the total, and the trace."""
    problem = correlated(1.0)
    post = fips.solve(problem)
    S_0 = problem.prior_cov.dense()
    np.testing.assert_allclose(post.uncertainty()["prior_std"], 1.0, rtol=1e-14)
    assert post.uncertainty(total(problem))["prior_std"].iloc[0] == pytest.approx(
        math.sqrt(S_0.sum()), rel=1e-14
    )
    assert post.uncertainty_reduction == pytest.approx(
        1.0 - math.sqrt(post.variances.values.sum() / np.trace(S_0)), rel=1e-14
    )


def test_the_bounded_priors_draws_are_shared_and_drawn_again_after_a_pickle():
    """``with_gamma`` and ``with_data`` share the prior's draws; a pickle leaves them out, and draws them again, the same, from the seed."""
    problem = correlated(1.0)
    post = fips.solve(problem, "bounded", lower=0.0, draws=300)
    prior_std = post.uncertainty()["prior_std"]
    draws = post._solution.prior_draws()
    assert not draws.flags.writeable
    assert post.with_gamma(2.0)._solution.prior_draws() is draws
    assert post.with_data(problem.obs * 2.0)._solution.prior_draws() is draws
    assert "_prior_draws" not in post._solution.__getstate__()
    back = pickle.loads(pickle.dumps(post))
    pd.testing.assert_series_equal(back.uncertainty()["prior_std"], prior_std)
