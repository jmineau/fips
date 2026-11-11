"""
Tests for ``Posterior.log_evidence``: the log density of the observations given the hyperparameters, the state integrated out.

The analytic value is held to SciPy's density of the observations,
``multivariate_normal(H x_0 + c, H S_0 H^T + S_z / gamma).logpdf(z)``, written
with dense matrices and nothing from fips:

1. at sizes each of the analytic method's two forms takes, in both forms
   (the ``analytic_form`` fixture), at several gammas and after ``with_gamma``
   (``test_oracle.py`` also holds every method to it, on every oracle problem,
   and after reordering every input);
2. with each form a covariance is kept in (Kronecker, block diagonal, sparse,
   diagonal), as the prior error and as the model-data mismatch;
3. with a singular prior error covariance;
4. through ``with_gamma`` and ``with_data``, against a fresh solve, and
   through a pickle;
5. wrong variants of the formula (no ``m log gamma``, the information content
   not doubled, no ``log |S_z|``) fail the same comparison.

``"gauss_newton"`` gives Laplace's approximation: exact for a linear model
(to roundoff, against the analytic value), and, for a lognormal model in a few
cells, compared with an importance-sampling estimate of the evidence known to
1%; the error is recorded, not held to a tolerance. ``"bounded"`` with a
finite bound raises. A slow test: over worlds drawn at known hyperparameters, the
hyperparameters that maximize the evidence scatter about the true ones.
"""

import logging
import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

import fips
from fips import (
    Covariance,
    Exponential,
    ForwardModel,
    Independent,
    Operator,
    Problem,
    Space,
    Vector,
)
from fips.linalg import (
    BlockDiagonalOperator,
    DiagonalOperator,
    KroneckerOperator,
    SparseOperator,
)

RTOL = 1e-10

logger = logging.getLogger(__name__)


def _spd(rng: np.random.Generator, k: int) -> np.ndarray:
    """A well-conditioned SPD matrix."""
    B = rng.standard_normal((k, k))
    return B @ B.T + k * np.eye(k)


def _scipy_evidence(
    z: np.ndarray,
    x_0: np.ndarray,
    H: np.ndarray,
    S_0: np.ndarray,
    S_z: np.ndarray,
    c: float | np.ndarray,
    gamma: float = 1.0,
) -> float:
    """The density of `z` under ``N(H x_0 + c, H S_0 H^T + S_z / gamma)``."""
    G = H @ S_0 @ H.T + S_z / gamma
    return float(multivariate_normal(H @ x_0 + c, G).logpdf(z))


def _close(actual: float, expected: float, what: str, rtol: float = RTOL) -> None:
    np.testing.assert_allclose(actual, expected, rtol=rtol, err_msg=what)


def _line(n: int, m: int, seed: int, density: float = 0.4):
    """Spaces, a sparse-ish forward matrix, a prior and observations on a line of `n` cells."""
    rng = np.random.default_rng(seed)
    state = Space({"flux": pd.Index(np.arange(n) + 0.5, name="x")})
    observed = Space({"ch4": pd.RangeIndex(m, name="obs_id")})
    H = rng.uniform(0.0, 1.0, (m, n)) * (rng.uniform(size=(m, n)) < density)
    x_0 = rng.normal(1.0, 0.5, n)
    z = H @ x_0 + 2.0 + rng.normal(0.0, 1.5, m)
    return rng, state, observed, H, x_0, z


def _problem(state, observed, H, x_0, z, prior_cov, obs_cov, offset=2.0) -> Problem:
    return Problem(
        prior=Vector(x_0, state),
        prior_cov=prior_cov,
        obs=Vector(z, observed),
        obs_cov=obs_cov,
        forward=Operator(H, observed, state),
        offset=offset,
    )


class LinearModel(ForwardModel):
    """A linear map written as a model, so that ``"gauss_newton"`` iterates on it."""

    def __init__(self, H: Operator):
        super().__init__(H.rows, H.cols)
        self.H = H

    def __call__(self, x):
        return self.H @ x

    def jacobian(self, x):
        return self.H


# ---------------------------------------------------------------------------
# 1. The analytic value against SciPy, in both forms
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("n, m", [(30, 80), (80, 30)])
@pytest.mark.parametrize("gamma", [0.4, 1.0, 2.5])
def test_against_scipy(n, m, gamma):
    """The evidence is SciPy's density of the observations, at the solve's gamma and after ``with_gamma``."""
    _, state, observed, H, x_0, z = _line(n, m, seed=n + m)
    std = np.random.default_rng(1).uniform(0.3, 1.0, m)
    problem = _problem(
        state,
        observed,
        H,
        x_0,
        z,
        Exponential("x", 3.0).with_std(0.8),
        Independent().with_std(Vector(std, observed)),
    )
    S_0, S_z = problem.prior_cov.dense(), np.diag(std**2)
    post = fips.solve(problem, gamma=gamma)
    _close(
        post.log_evidence,
        _scipy_evidence(z, x_0, H, S_0, S_z, 2.0, gamma),
        f"n={n} m={m} gamma={gamma}",
    )
    other = 3.0 * gamma
    _close(
        post.with_gamma(other).log_evidence,
        _scipy_evidence(z, x_0, H, S_0, S_z, 2.0, other),
        f"n={n} m={m}: with_gamma({other})",
    )


# ---------------------------------------------------------------------------
# 2. Every form a covariance is kept in
# ---------------------------------------------------------------------------

STRUCTURES = [
    "Kronecker S_0",
    "permuted Kronecker S_0",
    "block-diagonal S_0",
    "sparse S_0",
    "diagonal S_0",
    "Kronecker S_z",
    "permuted Kronecker S_z",
    "part of a Kronecker S_z",
    "block-diagonal S_z",
    "sparse S_z",
    "diagonal S_z",
]

FORM_OF = {
    "Kronecker": KroneckerOperator,
    "permuted": KroneckerOperator,
    "part": KroneckerOperator,
    "block-diagonal": BlockDiagonalOperator,
    "sparse": SparseOperator,
    "diagonal": DiagonalOperator,
}


def _structured(rng: np.random.Generator, structure: str, k: int):
    """Return a form of `structure` on `k` rows (a multiple of 3) and its dense matrix."""
    kind = structure.split()[0]
    if kind in ("Kronecker", "permuted", "part"):
        # The whole product of two factors, or k of its rows
        sizes = (3, k // 3 + (kind == "part"))
        total = sizes[0] * sizes[1]
        chosen = {
            "Kronecker": np.arange(total),
            "permuted": rng.permutation(total),
            "part": np.sort(rng.permutation(total)[:k]),
        }[kind][:k]
        factors = [_spd(rng, s) for s in sizes]
        std = rng.uniform(0.5, 2.0, k)
        form = KroneckerOperator(
            factors, std=std, positions=np.unravel_index(chosen, sizes)
        )
        dense = std[:, None] * np.kron(*factors)[np.ix_(chosen, chosen)] * std[None, :]
        return form, dense
    if kind == "block-diagonal":
        half = k // 2
        block, variances = _spd(rng, half), rng.uniform(0.2, 2.0, k - half)
        dense = np.zeros((k, k))
        dense[:half, :half], dense[half:, half:] = block, np.diag(variances)
        return BlockDiagonalOperator([block, DiagonalOperator(variances)]), dense
    if kind == "sparse":
        # A tridiagonal covariance, diagonally dominant
        off = rng.uniform(-0.4, 0.4, k - 1)
        dense = np.diag(rng.uniform(1.0, 2.0, k)) + np.diag(off, 1) + np.diag(off, -1)
        return SparseOperator(sp.csr_array(dense)), dense
    if kind == "diagonal":
        variances = rng.uniform(0.2, 2.0, k)
        return DiagonalOperator(variances), np.diag(variances)
    raise ValueError(structure)


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize("n, m", [(9, 12), (12, 9)])
def test_covariance_forms(structure, n, m):
    """Each form a covariance is kept in gives SciPy's value, as the prior error and as the mismatch."""
    rng, state, observed, H, x_0, z = _line(n, m, seed=7)
    on_prior = structure.endswith("S_0")
    form, dense = _structured(rng, structure, n if on_prior else m)
    if on_prior:
        prior_cov, S_0 = Covariance(form, state), dense
        S_z = _spd(rng, m) / m
        obs_cov = Covariance(S_z, observed)
    else:
        S_0 = _spd(rng, n) / n
        prior_cov, obs_cov, S_z = (
            Covariance(S_0, state),
            Covariance(form, observed),
            dense,
        )
    problem = _problem(state, observed, H, x_0, z, prior_cov, obs_cov)
    held = problem.prior_cov if on_prior else problem.obs_cov
    assert isinstance(held.data, FORM_OF[structure.split()[0]])
    post = fips.solve(problem, gamma=0.7)
    _close(
        post.log_evidence,
        _scipy_evidence(z, x_0, H, S_0, S_z, 2.0, 0.7),
        f"{structure}, n={n} m={m}",
    )


# ---------------------------------------------------------------------------
# 3. A singular prior error covariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n, m", [(10, 25), (25, 10)])
@pytest.mark.parametrize("singular", ["zero std", "constant block"])
def test_singular_prior(n, m, singular):
    """A prior error covariance without a Cholesky factor gives SciPy's value too."""
    _, state, observed, H, x_0, z = _line(n, m, seed=3)
    if singular == "zero std":
        std = np.full(n, 0.8)
        std[[1, n // 2]] = 0.0  # two cells known exactly
        prior_cov = Exponential("x", 2.0).with_std(Vector(std, state))
    else:
        # one shared error for every cell: rank one, plus nothing else
        prior_cov = fips.Constant().with_std(0.6)
    problem = _problem(
        state, observed, H, x_0, z, prior_cov, Independent().with_std(0.5)
    )
    S_0 = problem.prior_cov.dense()
    assert np.linalg.matrix_rank(S_0) < n
    S_z = 0.25 * np.eye(m)
    post = fips.solve(problem)
    _close(post.log_evidence, _scipy_evidence(z, x_0, H, S_0, S_z, 2.0), singular)
    _close(
        post.with_gamma(2.0).log_evidence,
        _scipy_evidence(z, x_0, H, S_0, S_z, 2.0, 2.0),
        f"{singular}: with_gamma",
    )


# ---------------------------------------------------------------------------
# 4. The twins: with_gamma, with_data, a pickle
# ---------------------------------------------------------------------------


def _correlated_problem(n: int, m: int, seed: int = 11) -> Problem:
    rng, state, observed, H, x_0, z = _line(n, m, seed)
    return _problem(
        state,
        observed,
        H,
        x_0,
        z,
        Exponential("x", 2.0).with_std(0.7),
        Covariance(_spd(rng, m) / m, observed),
    )


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("n, m", [(8, 20), (20, 8)])
def test_with_gamma_and_with_data_match_a_fresh_solve(n, m):
    """The twins of a posterior give the evidence a fresh solve does, without solving again."""
    problem = _correlated_problem(n, m)
    post = fips.solve(problem)
    for gamma in (0.2, 1.0, 5.0):
        _close(
            post.with_gamma(gamma).log_evidence,
            fips.solve(problem, gamma=gamma).log_evidence,
            f"with_gamma({gamma})",
        )
    rng = np.random.default_rng(5)
    _, obs = problem.simulate(rng)
    shuffled = Vector.from_series(obs.to_series().sample(frac=1.0, random_state=1))
    fresh = Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=obs,
        obs_cov=problem.obs_cov,
        forward=problem.forward,
        offset=problem.offset,
    )
    _close(
        post.with_data(shuffled).log_evidence,
        fips.solve(fresh).log_evidence,
        "with_data, labels shuffled",
    )
    _close(
        post.with_gamma(3.0).with_data(obs).log_evidence,
        fips.solve(fresh, gamma=3.0).log_evidence,
        "with_gamma, then with_data",
    )


def test_a_sweep_factors_the_mismatch_once(monkeypatch):
    """The posteriors of a gamma sweep share the problem's ``log |S_z|``, computed once."""
    problem = _correlated_problem(8, 20)
    calls = []
    original = Covariance.logdet

    def counted(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(Covariance, "logdet", counted)
    post = fips.solve(problem)
    values = [post.with_gamma(g).log_evidence for g in (0.5, 1.0, 2.0, 4.0)]
    assert len(calls) == 1
    assert np.all(np.isfinite(values))


def test_pickle():
    """A posterior read back gives the same evidence; the kept ``log |S_z|`` is not pickled."""
    post = fips.solve(_correlated_problem(8, 20))
    value = post.log_evidence
    assert "_obs_logdet" in post.problem.__dict__
    assert "_obs_logdet" not in post.problem.__getstate__()
    back = pickle.loads(pickle.dumps(post))
    assert back.log_evidence == value


# ---------------------------------------------------------------------------
# 5. Wrong variants of the formula fail
# ---------------------------------------------------------------------------


def test_wrong_variants_fail():
    """Each term matters: leaving one out, or not doubling the information content, misses SciPy's value."""
    problem = _correlated_problem(12, 30)
    gamma = 2.5
    post = fips.solve(problem, gamma=gamma)
    m = len(problem.obs)
    chi2, info = post.chi2, post.information_content
    logdet = problem.obs_cov.logdet()
    exact = _scipy_evidence(
        problem.obs.values,
        problem.prior.values,
        problem.forward.dense(),
        problem.prior_cov.dense(),
        problem.obs_cov.dense(),
        2.0,
        gamma,
    )
    _close(post.log_evidence, exact, "the formula")
    variants = {
        "no m log gamma": chi2 + 2 * info + logdet + m * np.log(2 * np.pi),
        "information not doubled": chi2
        + info
        + logdet
        - m * np.log(gamma)
        + m * np.log(2 * np.pi),
        "no log |S_z|": chi2 + 2 * info - m * np.log(gamma) + m * np.log(2 * np.pi),
        "+ m log gamma": chi2
        + 2 * info
        + logdet
        + m * np.log(gamma)
        + m * np.log(2 * np.pi),
    }
    for name, twice_negative in variants.items():
        wrong = -0.5 * twice_negative
        assert abs(wrong - exact) > 1e3 * RTOL * abs(exact), name


# ---------------------------------------------------------------------------
# The other methods
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("n, m", [(5, 12), (12, 5)])
def test_gauss_newton_on_a_linear_model_is_exact(n, m):
    """Laplace's approximation of a linear model is the exact evidence, to roundoff."""
    problem = _correlated_problem(n, m, seed=21)
    model = Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=problem.obs,
        obs_cov=problem.obs_cov,
        forward=LinearModel(problem.forward),
        offset=problem.offset,
    )
    analytic = fips.solve(problem, gamma=0.6)
    laplace = fips.solve(model, "gauss_newton", gamma=0.6, draws=10)
    _close(laplace.log_evidence, analytic.log_evidence, "gauss_newton, linear")


def test_gauss_newton_with_data_matches_a_fresh_solve():
    """
    ``with_data`` of a lognormal model's posterior gives the evidence a fresh solve does.

    Both iterate to the mode, from different starts, and the Gaussian is the
    one at the last iterate: they agree to the size of the last step, so the
    iterations here go on until it is negligible (at the default ``tol`` they
    agreed to 4e-5 of the value).
    """
    problem, _, _ = _lognormal_problem(0, sigma=0.8)
    post = fips.solve(problem, "gauss_newton", draws=10, tol=1e-12)
    rng = np.random.default_rng(9)
    obs = Vector(
        problem.obs.values + rng.normal(0.0, 0.3, len(problem.obs)), problem.obs.space
    )
    fresh = Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=obs,
        obs_cov=problem.obs_cov,
        forward=problem.forward,
        offset=problem.offset,
    )
    _close(
        post.with_data(obs).log_evidence,
        fips.solve(fresh, "gauss_newton", draws=10, tol=1e-12).log_evidence,
        "gauss_newton with_data",
        rtol=1e-6,
    )


def test_bounded():
    """A bounded posterior with no finite bound has the analytic evidence; with a bound it raises."""
    problem = _correlated_problem(8, 20)
    analytic = fips.solve(problem).log_evidence
    free = fips.solve(problem, "bounded", lower=-np.inf, draws=10)
    _close(free.log_evidence, analytic, "bounded, no finite bound")
    _close(fips.solve(problem, "bounded", draws=10).log_evidence, analytic, "default")
    bounded = fips.solve(problem, "bounded", lower=0.0, draws=10)
    with pytest.raises(TypeError, match="orthant probabilities"):
        _ = bounded.log_evidence
    with pytest.raises(TypeError, match=r"fips\.solve\(problem\)\.log_evidence"):
        _ = bounded.with_gamma(2.0).log_evidence


# ---------------------------------------------------------------------------
# Laplace's approximation against importance sampling
# ---------------------------------------------------------------------------


def _lognormal_problem(seed: int, sigma: float, n: int = 4, m: int = 10):
    """A lognormal model in `n` cells, the data from a truth drawn from its prior."""
    rng = np.random.default_rng(seed)
    state = Space({"flux": pd.RangeIndex(n, name="cell")})
    observed = Space({"ch4": pd.RangeIndex(m, name="obs_id")})
    H = rng.uniform(0.0, 1.0, (m, n)) * (rng.uniform(size=(m, n)) < 0.6)
    reference = rng.lognormal(0.0, 0.5, n)
    model = fips.LogNormalForward(
        Operator(H, observed, state),
        pd.Series(reference, index=state.select("flux")),
    )
    u = rng.normal(0.0, sigma, n)
    noise = 0.3
    z = H @ (reference * np.exp(u)) + 2.0 + rng.normal(0.0, noise, m)
    problem = Problem(
        prior=Vector(np.zeros(n), state),
        prior_cov=Exponential("cell", 1.5).with_std(sigma),
        obs=Vector(z, observed),
        obs_cov=Independent().with_std(noise),
        forward=model,
        offset=2.0,
    )
    return problem, H, reference


def _importance_evidence(problem, F, center, cov, N, rng):
    """
    Estimate ``log p(z)`` by importance sampling from ``N(center, cov)``.

    `F` maps a block of states (``n x N``) to the modeled observations,
    offset included. Returns the estimate and its standard error (in log
    units, the relative standard error of the mean weight).
    """
    n, m = len(problem.prior), len(problem.obs)
    x_0, z = problem.prior.values, problem.obs.values
    L = np.linalg.cholesky(cov)
    w = rng.standard_normal((n, N))
    X = center[:, None] + L @ w
    log_q = (
        -0.5 * np.sum(w**2, axis=0)
        - np.log(np.diag(L)).sum()
        - 0.5 * n * np.log(2 * np.pi)
    )
    L_0 = np.linalg.cholesky(problem.prior_cov.dense())
    a = np.linalg.solve(L_0, X - x_0[:, None])
    log_prior = (
        -0.5 * np.sum(a**2, axis=0)
        - np.log(np.diag(L_0)).sum()
        - 0.5 * n * np.log(2 * np.pi)
    )
    L_z = np.linalg.cholesky(problem.obs_cov.dense())
    b = np.linalg.solve(L_z, z[:, None] - F(X))
    log_likelihood = (
        -0.5 * np.sum(b**2, axis=0)
        - np.log(np.diag(L_z)).sum()
        - 0.5 * m * np.log(2 * np.pi)
    )
    log_w = log_likelihood + log_prior - log_q
    weights = np.exp(log_w - log_w.max())
    se = weights.std() / np.sqrt(N) / weights.mean()
    return float(logsumexp(log_w) - np.log(N)), float(se)


def test_importance_sampler_recovers_a_linear_evidence():
    """The importance sampler the Laplace test uses gives the exact evidence of a linear problem, within its error."""
    problem = _correlated_problem(4, 10, seed=2)
    post = fips.solve(problem)
    Hd = problem.forward.dense()
    estimate, se = _importance_evidence(
        problem,
        lambda X: Hd @ X + 2.0,
        post.estimate.values,
        1.3**2 * post.cov().dense(),  # wider than the posterior: unequal weights
        200_000,
        np.random.default_rng(0),
    )
    assert se < 0.01
    assert abs(estimate - post.log_evidence) < 4 * se


@pytest.mark.parametrize("seed", range(3))
def test_laplace_against_importance_sampling(seed):
    """
    For a lognormal model, Laplace's approximation is near the importance-sampling evidence; the error is recorded.

    The proposal is the Gaussian at the mode, 1.3 times wider; 200,000 draws
    know the evidence to 1%. The approximation leaves out the skew of the
    posterior, so it is not exact: with a log-std of 0.8 in four cells its
    error was 0.07 to 0.11 nats in these seeds (0.01 to 0.2 over log-stds of
    0.3 to 1.5). The test logs it (``pytest -o log_cli=true``) and asserts
    only that it is well under one nat.
    """
    problem, H, reference = _lognormal_problem(seed, sigma=0.8)
    post = fips.solve(problem, "gauss_newton", draws=10)
    gaussian = fips.solve(post._linear, gamma=post.options.gamma)
    estimate, se = _importance_evidence(
        problem,
        lambda U: H @ (reference[:, None] * np.exp(U)) + 2.0,
        post.estimate.values,
        1.3**2 * gaussian.cov().dense(),
        200_000,
        np.random.default_rng(seed),
    )
    error = post.log_evidence - estimate
    logger.info(
        "seed %d: Laplace %.4f, importance sampling %.4f +- %.4f, error %+.4f",
        seed,
        post.log_evidence,
        estimate,
        se,
        error,
    )
    assert se < 0.01
    assert np.isfinite(post.log_evidence)
    assert abs(error) < 0.5, f"Laplace {post.log_evidence:.4f}, sampled {estimate:.4f}"


# ---------------------------------------------------------------------------
# Worlds at known hyperparameters
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_evidence_maximum_scatters_about_the_true_hyperparameters():
    """
    Over worlds drawn at known hyperparameters, the prior std and the mismatch std that maximize the evidence scatter about the true ones.

    100 worlds on one problem's labels (40 cells, 80 observations, the
    correlation length known); each fit maximizes the evidence over the log of
    the two standard deviations. The mean log ratio of fit to truth is within
    three standard errors of zero, and about half the fits are below the truth.
    The evidence without its information content, the chi-squared and
    ``log |S_z|`` alone, has its maximum at an infinite prior std: it fails.
    """
    n, m, trials = 40, 80, 100
    true = np.array([1.0, 0.5])  # prior std, mismatch std
    rng = np.random.default_rng(0)
    state = Space({"flux": pd.Index(np.arange(n) + 0.5, name="x")})
    observed = Space({"ch4": pd.RangeIndex(m, name="obs_id")})
    H = np.zeros((m, n))
    for i, start in enumerate(rng.integers(0, n - 6, m)):
        H[i, start : start + rng.integers(2, 7)] = rng.uniform(0.5, 1.5)
    forward = Operator(H, observed, state)

    def build(std: float, noise: float, obs: Vector) -> Problem:
        return Problem(
            prior=Vector(np.full(n, 2.0), state),
            prior_cov=Exponential("x", 4.0).with_std(std),
            obs=obs,
            obs_cov=Independent().with_std(noise),
            forward=forward,
        )

    world = build(*true, obs=Vector(np.zeros(m), observed))
    log_ratios = []
    for _ in range(trials):
        _, obs = world.simulate(rng)
        fit = minimize(
            lambda s, obs=obs: -fips.solve(build(*np.exp(s), obs)).log_evidence,
            np.log([0.5, 1.0]),
            method="L-BFGS-B",
        )
        assert fit.success, fit.message
        log_ratios.append(fit.x - np.log(true))
    log_ratios = np.array(log_ratios)
    mean = log_ratios.mean(axis=0)
    se = log_ratios.std(axis=0, ddof=1) / np.sqrt(trials)
    assert np.all(np.abs(mean) < 3 * se), (mean, se)
    below = (log_ratios < 0).mean(axis=0)
    assert np.all((below > 0.35) & (below < 0.65)), below

    # The wrong variant: no information content. Its cost falls as the prior std grows.
    _, obs = world.simulate(rng)

    def wrong(std: float) -> float:
        post = fips.solve(build(std, true[1], obs))
        return -0.5 * (post.chi2 + post.problem.obs_cov.logdet())

    values = [wrong(s) for s in (1.0, 10.0, 100.0)]
    assert values[0] < values[1] < values[2]
