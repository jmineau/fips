"""
Tests for ``Posterior.sample``: exact draws from the analytic posterior, by perturbing the prior and the data and solving.

A draw with the right variances and the wrong correlations (or the right
shape and the wrong scale) still has the right labels and passes a casual
look, so the draws are held to the closed form three ways: their mean and
covariance against ``post.estimate`` and ``post.cov()`` within Monte Carlo
standard errors, the squared Mahalanobis lengths against the chi-squared
distribution (the whole shape, correlations included), and the gain applied to a block
of innovations against ``post.kalman_gain``. A reference sampler written with
dense matrices checks that the statistics have teeth: it passes them, and the
same sampler with a term left out or the scaling by gamma forgotten fails.
Most tests run in both analytic forms (the ``analytic_form`` fixture).
"""

import pickle
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.linalg import cholesky
from scipy.stats import chi2, kstest

import fips
from fips import (
    Covariance,
    Exponential,
    Kronecker,
    Operator,
    Problem,
    Space,
    Vector,
)
from fips.linalg import DiagonalOperator, KroneckerOperator, SparseOperator
from fips.methods import analytic
from fips.methods.analytic import AnalyticOptions, Information, Whitened
from fips.posterior import Posterior

N_DRAWS = 20_000


def _spd(rng, k):
    """A well-conditioned correlated SPD matrix."""
    B = rng.standard_normal((k, k))
    return B @ B.T + k * np.eye(k)


def _tridiagonal(rng, k):
    """A sparse SPD matrix: errors correlated with the next one's."""
    return sp.csr_matrix(
        np.diag(rng.uniform(1.0, 3.0, k))
        + np.diag(np.full(k - 1, 0.4), 1)
        + np.diag(np.full(k - 1, 0.4), -1)
    )


# name -> (n_time, n_cell, m, gamma): the prior and the mismatch of each are below
CASES = {
    "dense": (2, 3, 10, 0.5),
    "kronecker prior, sparse mismatch": (2, 4, 10, 2.0),
    "dense prior, diagonal mismatch": (3, 2, 9, 3.0),
}


def _problem(case: str, seed: int = 0) -> tuple[Problem, float]:
    """Return a labeled problem of `case` (a flux block over time and cell) and its gamma."""
    n_time, n_cell, m, gamma = CASES[case]
    rng = np.random.default_rng(seed)
    n = n_time * n_cell
    state = Space(
        {
            "flux": pd.MultiIndex.from_product(
                [pd.date_range("2024-01-01", periods=n_time, freq="MS"), range(n_cell)],
                names=["time", "cell"],
            )
        }
    )
    obs = Space({"conc": pd.Index(range(m), name="obs_id")})
    H = rng.standard_normal((m, n))
    S_0, S_z = _spd(rng, n), _spd(rng, m)
    if case == "kronecker prior, sparse mismatch":
        S_0 = KroneckerOperator(
            [_spd(rng, n_time) / n_time, _spd(rng, n_cell) / n_cell],
            std=rng.uniform(0.5, 2.0, n),
        )
        S_z = _tridiagonal(rng, m)
        H = sp.csr_matrix(np.where(np.abs(H) < 0.6, 0.0, H))
    elif case == "dense prior, diagonal mismatch":
        S_z = DiagonalOperator(rng.uniform(0.5, 2.0, m))
    problem = Problem(
        prior=Vector(rng.normal(2.0, 1.0, n), state),
        prior_cov=Covariance(S_0, state),
        obs=Vector(rng.normal(2.0, 2.0, m), obs),
        obs_cov=Covariance(S_z, obs),
        forward=Operator(H, obs, state),
        offset=Vector(rng.normal(0.5, 0.1, m), obs),
    )
    return problem, gamma


def _standard_errors(draws: np.ndarray, post: Posterior) -> tuple[float, float, float]:
    """
    Return the largest standard error the mean and the covariance of `draws` are off by, and the KS p-value of their Mahalanobis lengths.

    The mean of each element is compared with ``post.estimate`` in units of
    ``sqrt(S_ii / N)``, each covariance entry with ``post.cov()`` in units of
    ``sqrt((S_ii S_jj + S_ij^2) / N)`` (the standard error of a sample
    covariance of Gaussian draws), and the squared Mahalanobis lengths
    ``(x - x_hat)^T S^-1 (x - x_hat)`` with the chi-squared distribution of
    ``n`` degrees of freedom.
    """
    n, N = draws.shape
    x_hat, S = post.estimate.values, post.cov().dense()
    variance = np.diag(S)
    mean_off = np.abs(draws.mean(axis=1) - x_hat) / np.sqrt(variance / N)
    cov_off = np.abs(np.cov(draws) - S) / np.sqrt(
        (np.outer(variance, variance) + S**2) / N
    )
    centered = draws - x_hat[:, None]
    L = cholesky(S, lower=True)
    whitened = np.linalg.solve(L, centered)
    lengths = (whitened**2).sum(axis=0)
    return (
        float(mean_off.max()),
        float(cov_off.max()),
        kstest(lengths, chi2(n).cdf).pvalue,
    )


def _holds(post: Posterior, draws: np.ndarray) -> bool:
    """Whether `draws` are within the limits of an exact sampler of `post` (the tests' thresholds)."""
    mean_off, cov_off, p = _standard_errors(draws, post)
    return mean_off < 4.5 and cov_off < 5.0 and p > 0.001


def _reference(post, N, rng, *, prior_term=True, scale=True) -> np.ndarray:
    """
    Perturb and solve with dense matrices: the sampler, written the long way.

    ``prior_term=False`` leaves out ``- K H (x_p - x_0)`` and ``scale=False``
    forgets to divide the mismatch's draw by ``sqrt(gamma)``: the two ways the
    method can be wrong that its shapes and labels would not show.
    """
    problem = post.problem
    H = problem.forward.dense()
    L_0 = cholesky(problem.prior_cov.dense(), lower=True)
    L_z = cholesky(problem.obs_cov.dense(), lower=True)
    n, m = H.shape[1], H.shape[0]
    delta = L_0 @ rng.standard_normal((n, N))
    error = L_z @ rng.standard_normal((m, N))
    if scale:
        error = error / np.sqrt(post.options.gamma)
    innovations = (H @ delta if prior_term else 0.0) + error
    K = post.kalman_gain.dense()
    return post.estimate.values[:, None] + delta - K @ innovations


# ---------------------------------------------------------------------------
# The draws are the posterior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_draws_have_the_posterior_mean_and_covariance(case, analytic_form):
    """20,000 draws: the mean within 4.5 and the covariance within 5 standard errors of the closed form, in both forms."""
    problem, gamma = _problem(case)
    post = fips.solve(problem, gamma=gamma)
    assert isinstance(post._solution, analytic_form)
    draws = post.sample(N_DRAWS, rng=11).values
    mean_off, cov_off, _ = _standard_errors(draws, post)
    assert mean_off < 4.5, mean_off
    assert cov_off < 5.0, cov_off


@pytest.mark.parametrize("case", CASES)
def test_squared_mahalanobis_lengths_are_chi_squared(case, analytic_form):
    """The whole distribution: ``(x - x_hat)^T S^-1 (x - x_hat)`` is chi-squared with ``n`` degrees of freedom (KS, p > 0.001)."""
    problem, gamma = _problem(case)
    post = fips.solve(problem, gamma=gamma)
    draws = post.sample(N_DRAWS, rng=12).values
    _, _, p = _standard_errors(draws, post)
    assert p > 0.001, p


def test_a_semidefinite_prior_draws_its_mean_and_covariance(monkeypatch):
    """
    A prior of rank n - 2 still gives draws with the posterior's moments.

    Whether the prior's Cholesky fails in floating point depends on the
    platform's LAPACK (on macOS it can succeed, with pivots of roundoff), so
    the solve would take either form. The test holds the whitened form, the
    one a semidefinite prior is meant for, whichever root the platform
    finds for the prior.
    """
    rng = np.random.default_rng(3)
    n, m = 6, 9
    state = Space({"flux": pd.Index(range(n), name="cell")})
    obs = Space({"conc": pd.Index(range(m), name="obs_id")})
    B = rng.standard_normal((n, n - 2))
    problem = Problem(
        prior=Vector(rng.normal(size=n), state),
        prior_cov=Covariance(B @ B.T, state),
        obs=Vector(rng.normal(size=m), obs),
        obs_cov=Covariance(_spd(rng, m), obs),
        forward=Operator(rng.standard_normal((m, n)), obs, state),
    )
    monkeypatch.setattr(
        analytic, "_choose", lambda problem: (Whitened, "forced by a test")
    )
    post = fips.solve(problem, gamma=1.5)
    assert isinstance(post._solution, Whitened)
    draws = post.sample(N_DRAWS, rng=4).values
    x_hat, S = post.estimate.values, post.cov().dense()
    variance = np.diag(S)
    assert (
        np.abs(draws.mean(axis=1) - x_hat) / np.sqrt(variance / N_DRAWS)
    ).max() < 4.5
    se = np.sqrt((np.outer(variance, variance) + S**2) / N_DRAWS)
    assert (np.abs(np.cov(draws) - S) / se).max() < 5.0


def test_the_statistics_have_teeth():
    """A dense reference sampler passes them; without ``- K H (x_p - x_0)`` or without the ``1/sqrt(gamma)`` it fails."""
    problem, gamma = _problem("dense")
    post = fips.solve(problem, gamma=gamma)
    assert gamma != 1.0
    assert _holds(post, _reference(post, N_DRAWS, np.random.default_rng(5)))
    no_prior_term = _reference(
        post, N_DRAWS, np.random.default_rng(5), prior_term=False
    )
    assert not _holds(post, no_prior_term)
    no_scaling = _reference(post, N_DRAWS, np.random.default_rng(5), scale=False)
    assert not _holds(post, no_scaling)


# ---------------------------------------------------------------------------
# The block of innovations, and one form against the other
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_the_gain_applied_to_a_block_is_the_gain(case, analytic_form):
    """``_gain_times(D)`` is ``K D`` for a block, and for the innovation it is the increment ``estimate - prior``."""
    problem, gamma = _problem(case)
    post = fips.solve(problem, gamma=gamma)
    solution = post._solution
    K = post.kalman_gain.dense()
    D = np.random.default_rng(1).standard_normal((K.shape[1], 7))
    np.testing.assert_allclose(solution._gain_times(D), K @ D, rtol=1e-10, atol=1e-12)
    d = problem.obs.values - problem.predict(problem.prior).values
    increment = solution._gain_times(d[:, None])[:, 0]
    np.testing.assert_allclose(
        increment,
        post.estimate.values - problem.prior.values,
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("gamma", [0.4, 1.0])
def test_a_seed_gives_the_same_draws_in_both_forms(case, gamma):
    """Both forms draw the same normals and apply the same gain, so a seed gives the same draws to roundoff."""
    problem, _ = _problem(case)
    posts = [
        Posterior(problem, form.factor(problem, gamma), AnalyticOptions(gamma))
        for form in (Whitened, Information)
    ]
    whitened, information = (post.sample(50, rng=7) for post in posts)
    pd.testing.assert_frame_equal(whitened, information, rtol=1e-9, atol=1e-11)


def test_a_draw_after_a_gamma_sweep_is_the_posterior_at_that_gamma():
    """``with_gamma`` changes the posterior, and the draws follow: the same seed gives the same draws as a fresh solve."""
    problem, _ = _problem("dense")
    swept = fips.solve(problem).with_gamma(3.0)
    fresh = fips.solve(problem, gamma=3.0)
    pd.testing.assert_frame_equal(
        swept.sample(30, rng=2), fresh.sample(30, rng=2), rtol=1e-8, atol=1e-10
    )
    assert _holds(swept, swept.sample(N_DRAWS, rng=9).values)


# ---------------------------------------------------------------------------
# What it returns
# ---------------------------------------------------------------------------


def test_labels_and_shape(analytic_form):
    """Rows are the estimate's labels, columns a ``RangeIndex`` named ``draw`` from 0, and n = 1 and a NumPy integer work."""
    problem, gamma = _problem("dense")
    post = fips.solve(problem, gamma=gamma)
    draws = post.sample(5, rng=0)
    assert isinstance(draws, pd.DataFrame)
    assert draws.shape == (len(problem.prior), 5)
    assert draws.index.equals(post.estimate.to_series().index)
    assert draws.index.names == post.estimate.to_series().index.names
    assert isinstance(draws.columns, pd.RangeIndex)
    assert draws.columns.name == "draw" and list(draws.columns) == list(range(5))
    assert np.isfinite(draws.values).all()
    assert post.sample(1, rng=0).shape == (len(problem.prior), 1)
    assert post.sample(np.int64(3), rng=0).shape == (len(problem.prior), 3)


def test_groupby_totals_the_draws():
    """The rows keep the state's levels, so pandas totals the draws: ``groupby(level="time").sum()``."""
    problem, gamma = _problem("dense")
    post = fips.solve(problem, gamma=gamma)
    draws = post.sample(40, rng=0)
    totals = draws.groupby(level="time").sum()
    W = fips.aggregate.sum(problem.prior.space, by="time")
    aggregated = post.sample(40, rng=0, W=W)
    np.testing.assert_allclose(totals.values, aggregated.values, rtol=1e-10)


def test_an_aggregation_gives_the_draws_of_its_rows(analytic_form):
    """``sample(n, rng, W=W)`` is ``W @ sample(n, rng)`` to roundoff, labeled by the rows of ``W``, and its spread is ``cov(W)``."""
    problem, gamma = _problem("dense")
    post = fips.solve(problem, gamma=gamma)
    W = fips.aggregate.sum(problem.prior.space, by="time")
    plain = post.sample(60, rng=8)
    totals = post.sample(60, rng=8, W=W)
    np.testing.assert_allclose(
        totals.values, W.data @ plain.values, rtol=1e-12, atol=1e-12
    )
    assert totals.index.equals(W.rows.index)
    assert totals.index.names == W.rows.index.names
    assert totals.columns.equals(plain.columns)
    # and the totals have the aggregated covariance, within Monte Carlo error
    big = post.sample(N_DRAWS, rng=10, W=W).values
    S = post.cov(W).dense()
    se = np.sqrt((np.outer(np.diag(S), np.diag(S)) + S**2) / N_DRAWS)
    assert (np.abs(np.cov(big) - S) / se).max() < 5.0
    assert (
        np.abs(big.mean(axis=1) - (W.data @ post.estimate.values))
        / np.sqrt(np.diag(S) / N_DRAWS)
    ).max() < 4.5


def test_a_seed_is_reproducible_and_a_generator_moves_on(analytic_form):
    """The same seed gives the same draws; a generator passed twice gives different ones; no seed draws fresh."""
    problem, gamma = _problem("kronecker prior, sparse mismatch")
    post = fips.solve(problem, gamma=gamma)
    pd.testing.assert_frame_equal(post.sample(10, rng=3), post.sample(10, rng=3))
    first = post.sample(10, rng=3)
    assert not np.allclose(first.values, post.sample(10, rng=4).values)
    generator = np.random.default_rng(3)
    a, b = post.sample(10, rng=generator), post.sample(10, rng=generator)
    pd.testing.assert_frame_equal(a, first)  # a generator is used as given
    assert not np.allclose(a.values, b.values)
    assert not np.allclose(post.sample(10).values, post.sample(10).values)


def test_the_draw_order_is_the_prior_then_the_mismatch():
    """The prior's normals come first, then the mismatch's: the draws are what that order gives with the same roots."""
    problem, gamma = _problem("dense")
    post = fips.solve(problem, gamma=gamma)
    n, m = len(problem.prior), len(problem.obs)
    normals = np.random.default_rng(6)
    delta = problem.prior_cov.data._root @ normals.standard_normal((n, 4))
    error = problem.obs_cov.data._root @ normals.standard_normal((m, 4))
    D = problem.forward.data @ delta + error / np.sqrt(gamma)
    expected = (
        post.estimate.values[:, None] + delta - post.kalman_gain.dense() @ np.asarray(D)
    )
    np.testing.assert_allclose(
        post.sample(4, rng=6).values, expected, rtol=1e-10, atol=1e-12
    )


def test_a_pickled_posterior_draws_the_same():
    """The roots are a cache the pickle leaves out; the loaded posterior rebuilds them and draws the same for a seed."""
    problem, gamma = _problem("kronecker prior, sparse mismatch")
    post = fips.solve(problem, gamma=gamma)
    before = post.sample(8, rng=1)
    loaded = pickle.loads(pickle.dumps(post))
    pd.testing.assert_frame_equal(loaded.sample(8, rng=1), before)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, -3, 2.5, 3.0, True, "4", None])
def test_n_is_a_positive_integer(n):
    """``n`` is a positive integer; anything else raises, as ``trials`` does."""
    post = fips.solve(_problem("dense")[0])
    with pytest.raises(
        ValueError, match="n is the number of draws, a positive integer"
    ):
        post.sample(n)


def test_an_aggregation_is_checked_as_cov_checks_it():
    """``W`` must be an Operator on the state's labels."""
    problem, _ = _problem("dense")
    post = fips.solve(problem)
    with pytest.raises(TypeError, match="An aggregation is an Operator"):
        post.sample(3, W=np.ones((1, len(problem.prior))))
    wrong = Operator(
        np.ones((1, 2)),
        Space({"total": pd.Index(["all"], name="region")}),
        Space({"flux": pd.Index([0, 1], name="cell")}),
    )
    with pytest.raises(ValueError, match="aggregation"):
        post.sample(3, W=wrong)


# ---------------------------------------------------------------------------
# Nothing n x n, nothing m x m
# ---------------------------------------------------------------------------


def test_sampling_a_large_state_allocates_nothing_n_by_n():
    """
    With ``n = 2,400`` and a Kronecker prior, a block of draws forms no ``n x n`` array.

    One ``n x n`` array is 46 MB. The draws are ``n x N``, the prior drawn
    through its factors, the gain applied through ``W`` (``m x n``). The traced
    peak must stay under half of one ``n x n`` array.
    """
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=48, freq="D")
    lats, lons = np.arange(10) * 0.1, np.arange(5) * 0.1
    state = Space.from_index(
        pd.MultiIndex.from_product(
            [["flux"], times, lats, lons], names=["block", "time", "lat", "lon"]
        )
    )
    n, m = len(state.index), 30
    obs = Space.from_index(
        pd.MultiIndex.from_product([["conc"], range(m)], names=["block", "obs_id"])
    )
    problem = Problem(
        obs=Vector(rng.normal(size=m), obs),
        prior=Vector(rng.normal(size=n), state),
        forward=Operator(
            sp.random(m, n, density=0.05, random_state=1, format="csr"), obs, state
        ),
        prior_cov=Kronecker(
            Exponential("time", "5D"),
            Exponential(["lat", "lon"], 20.0, metric="haversine"),
        ).with_std(np.sqrt(0.5)),
        obs_cov=Covariance.diagonal(0.1, obs),
    )
    assert isinstance(problem.prior_cov.data, KroneckerOperator)
    post = fips.solve(problem)
    assert isinstance(post._solution, Whitened)
    tracemalloc.start()
    try:
        draws = post.sample(20, rng=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert draws.shape == (n, 20)
    n_by_n = n * n * 8
    assert peak < n_by_n / 2, (
        f"peak {peak / 1e6:.1f} MB; one n x n is {n_by_n / 1e6:.1f} MB"
    )


def test_sampling_with_many_observations_allocates_nothing_m_by_m():
    """With ``n = 30`` and ``m = 2,400`` (the information form), a Kronecker mismatch is drawn and applied through its factors."""
    rng = np.random.default_rng(0)
    n, m = 30, 2400
    state = Space({"flux": pd.Index(range(n), name="cell")})
    obs = Space({"conc": pd.Index(range(m), name="obs_id")})
    problem = Problem(
        obs=Vector(rng.normal(size=m), obs),
        prior=Vector(rng.normal(size=n), state),
        forward=Operator(
            sp.random(m, n, density=0.2, random_state=1, format="csr"), obs, state
        ),
        prior_cov=Covariance(_spd(rng, n), state),
        obs_cov=Covariance(
            KroneckerOperator(
                [_spd(rng, 40), _spd(rng, 60)], std=rng.uniform(0.5, 2.0, m)
            ),
            obs,
        ),
    )
    post = fips.solve(problem)
    assert isinstance(post._solution, Information)
    assert not isinstance(problem.obs_cov.data, SparseOperator)
    tracemalloc.start()
    try:
        draws = post.sample(20, rng=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert draws.shape == (n, 20)
    m_by_m = m * m * 8
    assert peak < m_by_m / 2, (
        f"peak {peak / 1e6:.1f} MB; one m x m is {m_by_m / 1e6:.1f} MB"
    )
