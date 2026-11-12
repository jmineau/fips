"""
The nonlinear methods against a sampler fips did not write.

Every other test of the draws holds them to a reference written beside the
code: a grid, importance sampling, SciPy's distributions, or fips's own
simulated world. Here each posterior's log density is written in NumPy from
its definition (the prior times the likelihood, nothing from fips), and drawn
by emcee's ensemble sampler (Foreman-Mackey, Hogg, Lang and Goodman, 2013)
with its differential-evolution move (ter Braak, 2006), which shares no code
with fips. fips's posterior mean, its
standard deviations, the correlation of the two most correlated elements, and
the 5% and 95% quantiles of the total of the block's values are then held to
emcee's, within their Monte Carlo standard errors:

- ``"gauss_newton"`` with :class:`~fips.LogNormalForward` and with
  :class:`~fips.SoftplusForward`, on ten state elements, in a mild regime (a
  narrow prior, more observations than elements) and a strongly nonlinear one
  (a wide prior, fewer observations than elements, each tight);
- ``"bounded"`` on ten elements with a lower bound of zero, several of them on
  it at the mode;
- :func:`fips.hierarchical` with two hyperparameters (a factor on the prior's
  standard deviation and one on the observation error's), on both its paths:
  emcee draws the hyperparameters from the evidence written in NumPy, and the
  state from its Gaussian at each draw.

**The Monte Carlo errors.** Each side is run as independent replicates (fips:
chains of ``Posterior.sample`` with their own seeds, or ``hierarchical`` with
its own seeds; emcee: ensembles with their own seeds and starts), each
replicate gives every statistic, and a statistic's error is the spread of its
replicates over the root of their number. That counts each chain's
autocorrelation, whatever it is. A fips value agrees when it is within
:data:`WITHIN` combined errors of emcee's, and the test also asserts that each
emcee run is longer than 50 of its integrated autocorrelation times (emcee's
own guidance) and that both sides' errors are small against the posterior's
spread, so that agreement is not bought with noise.

In the two strongly nonlinear regimes, fips's ``"gauss_newton"`` chains of
the default length do not yet agree (issue #203): those cases are expected to
fail (:data:`ISSUE_203`), and strictly, so that they turn the suite red once
they pass.

These tests are ``slow``; ``just test-slow`` runs them, and so does the
weekly Slow tests workflow.

References
----------
Foreman-Mackey, D., Hogg, D. W., Lang, D., and Goodman, J. (2013). emcee:
the MCMC hammer. *Publications of the Astronomical Society of the Pacific*,
125(925), 306-312. https://doi.org/10.1086/670067

ter Braak, C. J. F. (2006). A Markov chain Monte Carlo version of the genetic
algorithm Differential Evolution: easy Bayesian computing for real parameter
spaces. *Statistics and Computing*, 16(3), 239-249.
https://doi.org/10.1007/s11222-006-8769-1
"""

import dataclasses
import functools
import math
from collections.abc import Callable

import emcee
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import fips
from fips import Covariance, Operator, Problem, Space, Vector
from fips.distributions import LogNormal

pytestmark = pytest.mark.slow

WITHIN = 4.5
"""A fips statistic agrees with emcee's when within this many combined Monte Carlo errors (about 20 statistics a case: a false alarm in one of 10^4 runs)."""

N = 10
"""The number of state elements of every problem."""

MOVES = emcee.moves.DEMove()
"""
emcee's differential-evolution move (ter Braak, 2006), alone.

On the mildly nonlinear lognormal problem it mixes faster than emcee's
default stretch move (an autocorrelation time of 38 steps against 138) and
draws the same distribution. emcee 3.1.6's ``DESnookerMove``, which its
documentation pairs with it, does not: alone, its standard deviations there
were 30% narrower than the stretch move's and the Gaussian at the mode's.
"""


# ---------------------------------------------------------------------------
# The two samplers, each as independent replicates
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Estimates:
    """Each statistic's mean over the replicates and its Monte Carlo standard error."""

    value: np.ndarray
    error: np.ndarray

    @classmethod
    def of(cls, replicates: list) -> "Estimates":
        """The mean of the replicates' statistics and their spread over the root of their number."""
        values = np.array(replicates, dtype=float)
        return cls(
            values.mean(axis=0), values.std(axis=0, ddof=1) / math.sqrt(len(values))
        )


def emcee_draws(
    log_prob: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    spread: np.ndarray,
    seed: int,
    walkers: int,
    steps: int,
    burn: int,
) -> tuple[np.ndarray, float]:
    """
    One emcee ensemble: its draws after the burn-in (``dim x draws``), and the longest integrated autocorrelation time of a coordinate, in steps.

    The walkers start in a ball of `spread` about `start`, which a
    `log_prob` of minus infinity (outside a bound) does not accept: `start`
    and the ball are then moved inside by the caller.
    """
    rng = np.random.default_rng(seed)
    dim = len(start)
    p0 = start + spread * rng.standard_normal((walkers, dim))
    sampler = emcee.EnsembleSampler(walkers, dim, log_prob, vectorize=True, moves=MOVES)
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(p0, burn + steps)
    chain = sampler.get_chain(discard=burn)  # steps x walkers x dim
    tau = float(np.max(emcee.autocorr.integrated_time(chain, quiet=True)))
    return chain.reshape(-1, dim).T, tau


def emcee_estimates(
    log_prob, start, spread, statistic, runs=8, walkers=32, steps=30_000, burn=5_000
) -> Estimates:
    """`statistic` of `runs` independent emcee ensembles; asserts each ran for more than 50 autocorrelation times."""
    replicates = []
    for k in range(runs):
        draws, tau = emcee_draws(
            log_prob, start, spread, 1000 + k, walkers, steps, burn
        )
        assert steps > 50 * tau, f"emcee run {k}: tau = {tau:.0f} steps of {steps}"
        replicates.append(statistic(draws))
    return Estimates.of(replicates)


def fips_estimates(post: fips.Posterior, statistic, chains=16, draws=2000) -> Estimates:
    """`statistic` of `chains` chains of the posterior's draws, each with its own seed, its first 50 draws left out."""
    return Estimates.of(
        [
            statistic(post.sample(draws + 50, rng=100 + k).to_numpy()[:, 50:])
            for k in range(chains)
        ]
    )


class Disagrees(AssertionError):
    """fips's statistics are not emcee's: too far from them, or too uncertain to tell."""


def assert_agree(
    names: list[str],
    ours: Estimates,
    theirs: Estimates,
    scale: np.ndarray,
    unchecked: tuple[str, ...] = (),
    precision: float = 0.1,
) -> None:
    """
    Assert that emcee's errors are below a twentieth of `scale`, fips's below `precision` of it, and each of fips's statistics within :data:`WITHIN` combined errors of emcee's.

    `scale` is each statistic's yardstick (the posterior's standard deviation
    for a mean or a quantile, the standard deviation itself for a standard
    deviation, 1 for a correlation). A reference too noisy to test against
    fails with a plain ``AssertionError``; fips's side fails with
    :class:`Disagrees`, so that an expected failure of fips's names only it.
    The statistics named in `unchecked` are printed but not compared: those
    the reference cannot measure (the case says why). The table is printed,
    so a run with ``-s`` shows how close each came.
    """
    assert set(unchecked) <= set(names), unchecked
    checked = np.array([name not in unchecked for name in names])
    error = np.hypot(ours.error, theirs.error)
    z = (ours.value - theirs.value) / error
    print()
    print(f"{'statistic':<17}{'fips':>10}{'error':>8}{'emcee':>10}{'error':>8}{'z':>7}")
    for k, name in enumerate(names):
        print(
            f"{name:<17}{ours.value[k]:>10.4f}{ours.error[k]:>8.4f}"
            f"{theirs.value[k]:>10.4f}{theirs.error[k]:>8.4f}{z[k]:>7.2f}"
            + ("" if checked[k] else "  not compared")
        )

    def over(errors, part):
        return [
            (n, float(e), float(s))
            for n, e, s, c in zip(names, errors, scale, checked, strict=True)
            if c and e >= part * s
        ]

    assert not over(theirs.error, 0.05), over(theirs.error, 0.05)
    if over(ours.error, precision):
        raise Disagrees(f"fips's errors are large: {over(ours.error, precision)}")
    off = checked & (np.abs(z) > WITHIN)
    if off.any():
        raise Disagrees(
            "fips is off: "
            f"{[(n, float(v)) for n, v, o in zip(names, z, off, strict=True) if o]}"
        )


def statistic_names(n: int) -> list[str]:
    """The names of what :func:`state_statistic` returns."""
    return [
        *(f"mean[{i}]" for i in range(n)),
        *(f"std[{i}]" for i in range(n)),
        "corr",
        "total q05",
        "total q95",
    ]


def state_statistic(value: Callable, pair: tuple[int, int]):
    """
    The statistic of draws of the state (``n x draws``): each element's mean and standard deviation, the correlation of `pair`, and the 5% and 95% quantiles of the total of the values.

    `value` maps the state to the quantity summed (the transform of the block,
    or the state itself).
    """
    i, j = pair

    def statistic(draws: np.ndarray) -> list[float]:
        total = value(draws).sum(axis=0)
        return [
            *draws.mean(axis=1),
            *draws.std(axis=1, ddof=1),
            float(np.corrcoef(draws[i], draws[j])[0, 1]),
            *np.quantile(total, [0.05, 0.95]),
        ]

    return statistic


def state_scale(values: np.ndarray) -> np.ndarray:
    """
    Each statistic's yardstick for :func:`assert_agree`, from the values of :func:`state_statistic`.

    Each element's standard deviation for its mean and its standard
    deviation, 1 for the correlation, and the total's standard deviation (its
    90% interval over 3.29) for its quantiles.
    """
    std = values[N : 2 * N]
    width = (values[-1] - values[-2]) / 3.29
    return np.concatenate([std, std, [1.0], [width, width]])


def most_correlated(cov: np.ndarray) -> tuple[int, int]:
    """The two elements whose correlation is furthest from zero."""
    d = np.sqrt(np.diag(cov))
    corr = np.abs(cov / np.outer(d, d))
    np.fill_diagonal(corr, 0.0)
    i, j = np.unravel_index(np.argmax(corr), corr.shape)
    return int(i), int(j)


# ---------------------------------------------------------------------------
# The problems, as arrays, and their log densities in NumPy
# ---------------------------------------------------------------------------


def exponential_cov(n: int, std: float, length: float = 2.0) -> np.ndarray:
    """A prior covariance of standard deviation `std` whose correlation falls as ``exp(-|i - j| / length)`` along the elements."""
    i = np.arange(n)
    return std**2 * np.exp(-np.abs(i[:, None] - i[None, :]) / length)


def jacobian_matrix(rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    """A nonnegative forward operator, about half its entries zero, every element seen at least once."""
    H = rng.uniform(0.0, 1.0, (m, n)) * (rng.uniform(size=(m, n)) < 0.5)
    for j in np.flatnonzero(~H.any(axis=0)):
        H[rng.integers(m), j] = rng.uniform(0.5, 1.0)
    return H


@dataclasses.dataclass
class Transformed:
    """
    ``z = H T(u) + e``, ``u ~ N(u_0, S_0)``, ``e ~ N(0, S_z)``, with ``T`` element-wise: the arrays, ``T`` and ``T'`` in NumPy, and the fips problem.

    `kind` is ``"lognormal"`` (``T(u) = r e^u``, ``r = 1``) or
    ``"softplus"`` (``T(u) = s log(1 + e^{u/s})``).
    """

    kind: str
    u_0: np.ndarray
    S_0: np.ndarray
    H: np.ndarray
    S_z: np.ndarray
    z: np.ndarray
    scale: float = 1.0

    @classmethod
    def drawn(
        cls,
        kind: str,
        m: int,
        mean: float,
        std: float,
        noise: float,
        seed: int,
        scale: float = 1.0,
    ) -> "Transformed":
        """A truth drawn from the prior and observations of it with independent errors of standard deviation `noise`."""
        rng = np.random.default_rng(seed)
        S_0 = exponential_cov(N, std)
        u_0 = np.full(N, mean)
        H = jacobian_matrix(rng, m, N)
        made = cls(kind, u_0, S_0, H, noise**2 * np.eye(m), np.zeros(m), scale)
        truth = u_0 + np.linalg.cholesky(S_0) @ rng.standard_normal(N)
        made.z = H @ made.value(truth) + noise * rng.standard_normal(m)
        return made

    def value(self, u: np.ndarray) -> np.ndarray:
        """``T(u)``, element-wise."""
        if self.kind == "lognormal":
            return np.exp(u)
        return self.scale * np.logaddexp(0.0, u / self.scale)

    def slope(self, u: np.ndarray) -> np.ndarray:
        """``T'(u)``, element-wise."""
        if self.kind == "lognormal":
            return np.exp(u)
        return 0.5 * (1.0 + np.tanh(0.5 * u / self.scale))  # the logistic function

    @functools.cached_property
    def precisions(self) -> tuple[np.ndarray, np.ndarray]:
        """The inverses of the prior covariance and of the observation covariance."""
        return np.linalg.inv(self.S_0), np.linalg.inv(self.S_z)

    def log_prob(self, u: np.ndarray) -> np.ndarray:
        """The log posterior of each row of `u` (``k x n``), up to a constant: the prior's Gaussian and the likelihood's."""
        P_0, P_z = self.precisions
        d = u - self.u_0
        r = self.z - self.value(u) @ self.H.T
        return -0.5 * (np.einsum("ki,ij,kj->k", d, P_0, d)) - 0.5 * np.einsum(
            "ki,ij,kj->k", r, P_z, r
        )

    def mode(self) -> np.ndarray:
        """The posterior mode, by BFGS on the negative log posterior with its gradient."""
        P_0, P_z = np.linalg.inv(self.S_0), np.linalg.inv(self.S_z)

        def cost(u):
            d, r = u - self.u_0, self.z - self.H @ self.value(u)
            grad = P_0 @ d - self.slope(u) * (self.H.T @ (P_z @ r))
            return 0.5 * d @ P_0 @ d + 0.5 * r @ P_z @ r, grad

        found = minimize(cost, self.u_0, jac=True, method="BFGS", tol=1e-12)
        return found.x

    def laplace(self, u: np.ndarray) -> np.ndarray:
        """The covariance of the Gaussian at `u` (Gauss-Newton's: the prior's precision and the linearized data's)."""
        J = self.H * self.slope(u)
        return np.linalg.inv(
            np.linalg.inv(self.S_0) + J.T @ np.linalg.inv(self.S_z) @ J
        )

    def problem(self) -> Problem:
        """The fips problem of these arrays."""
        state = Space({"x": pd.RangeIndex(N, name="element")})
        observed = Space({"z": pd.RangeIndex(len(self.z), name="obs")})
        H = Operator(self.H, observed, state)
        if self.kind == "lognormal":
            model = fips.LogNormalForward(H, 1.0)
        else:
            model = fips.SoftplusForward(H, scale=self.scale)
        return Problem(
            prior=Vector(self.u_0, state),
            prior_cov=Covariance(self.S_0, state),
            obs=Vector(self.z, observed),
            obs_cov=Covariance(self.S_z, observed),
            forward=model,
        )


REGIMES = {
    # the prior's spread, in u, is small against the curvature of T: nearly linear
    "lognormal-mild": dict(kind="lognormal", m=20, mean=0.0, std=0.3, noise=0.1),
    # a factor of e^1.5 either way, six sums of ten elements, each to 0.03 (3% to 10%)
    "lognormal-strong": dict(kind="lognormal", m=6, mean=0.0, std=1.5, noise=0.03),
    # the values are six scales above zero: nearly linear
    "softplus-mild": dict(
        kind="softplus", m=20, mean=3.0, std=0.5, noise=0.2, scale=0.5
    ),
    # values within a few scales of zero, a prior wide against them, tight sums
    "softplus-strong": dict(
        kind="softplus", m=6, mean=0.3, std=1.5, noise=0.03, scale=0.3
    ),
}


# ---------------------------------------------------------------------------
# "gauss_newton" with a transform
# ---------------------------------------------------------------------------


UNCHECKED = {"softplus-strong": ("mean[0]", "std[0]")}
"""
The statistics emcee cannot measure, by regime.

softplus-strong: element 0's posterior has a thin arm towards large negative
``u`` (a value near zero, which the tight sums allow), visited rarely. emcee's
runs disagree on it: the standard deviation of element 0 was 0.14 to 0.43 run
by run with 32 walkers and 120,000 steps, and 0.14 to 0.35 with 64 walkers
and 240,000; with the stretch move and 200,000 steps the estimate was
0.25 +- 0.03, against 0.19 +- 0.04 here. No run length the suite can afford
measures it.
"""

PRECISION = 0.025
"""
fips's errors, from 16 chains of 2000 draws, must be below this fraction of each statistic's scale: one chain of the default length within 10%.

A user reads a solve's spread from one such chain, so a chain that lands
somewhere else from seed to seed is a wrong answer for them even when many
chains average to the right one. In the mild regimes the errors are 0.6% to
1.4% of the scale.
"""

ISSUE_203 = pytest.mark.xfail(
    strict=True,
    raises=Disagrees,
    reason="issue #203: a chain of gauss_newton's draws can miss part of a curved "
    "posterior, so its standard deviations are low and differ from seed to seed",
)
"""
The strongly nonlinear regimes: fips's chains of 2000 draws miss parts of the posterior.

Measured against eight emcee runs of 120,000 steps (and, separately, eight
of the stretch move's of 200,000, which agree with them): with 16 chains of
2000 draws, fips's errors are up to 11% of the scale (lognormal, the mean of
element 0) and 5% (softplus, element 8), so one chain is off by up to 45% and
20%; its standard deviations are low in 18 of the 20 elements, up to 12% and
8% where emcee measures them. With chains of 8000 the standard deviations are
still low (all ten for the lognormal, z to -3.2; the softplus's element 9 by
6%, z = -5.2). With chains of 64,000 they come within 1% to 4% of emcee's
(|z| below 3.4): the posterior fips draws is the right one, and its chains
mix too slowly for their default length, which is #203. When it is fixed
these pass, and ``strict`` turns the suite red until the mark comes off.
"""


@pytest.mark.parametrize(
    "regime",
    [
        pytest.param(name, marks=ISSUE_203) if name.endswith("strong") else name
        for name in REGIMES
    ],
)
def test_gauss_newton_draws_are_emcee_draws(regime):
    """
    ``"gauss_newton"`` with each transform, in each regime: the draws' means, standard deviations, a correlation, and the 5% and 95% quantiles of the total value are emcee's.

    The mode is also held to BFGS on the NumPy cost, to 1e-5 of the
    posterior's standard deviation.
    """
    case = Transformed.drawn(**REGIMES[regime], seed=7)
    mode = case.mode()
    post = fips.solve(case.problem(), "gauss_newton", tol=1e-10)
    laplace = case.laplace(mode)
    spread = np.sqrt(np.diag(laplace))
    np.testing.assert_allclose(post.estimate.values, mode, atol=1e-5 * spread.min())

    statistic = state_statistic(case.value, most_correlated(laplace))
    steps = 120_000 if regime.endswith("strong") else 30_000
    theirs = emcee_estimates(case.log_prob, mode, 0.1 * spread, statistic, steps=steps)
    ours = fips_estimates(post, statistic)
    assert_agree(
        statistic_names(N),
        ours,
        theirs,
        state_scale(theirs.value),
        UNCHECKED.get(regime, ()),
        PRECISION,
    )


# ---------------------------------------------------------------------------
# "bounded"
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Linear:
    """``z = H x + e``, ``x ~ N(x_0, S_0)``, ``e ~ N(0, S_z)``, as arrays."""

    x_0: np.ndarray
    S_0: np.ndarray
    H: np.ndarray
    S_z: np.ndarray
    z: np.ndarray

    def problem(self) -> Problem:
        """The fips problem of these arrays."""
        state = Space({"x": pd.RangeIndex(len(self.x_0), name="element")})
        observed = Space({"z": pd.RangeIndex(len(self.z), name="obs")})
        return Problem(
            prior=Vector(self.x_0, state),
            prior_cov=Covariance(self.S_0, state),
            obs=Vector(self.z, observed),
            obs_cov=Covariance(self.S_z, observed),
            forward=Operator(self.H, observed, state),
        )

    def analytic(self) -> tuple[np.ndarray, np.ndarray]:
        """The Gaussian posterior without bounds: its mean and covariance, from the dense formulas."""
        P_z = np.linalg.inv(self.S_z)
        cov = np.linalg.inv(np.linalg.inv(self.S_0) + self.H.T @ P_z @ self.H)
        return self.x_0 + cov @ self.H.T @ P_z @ (self.z - self.H @ self.x_0), cov


def bounded_case() -> Linear:
    """Ten elements of prior mean 0.2 and standard deviation 1, the truth zero at six, twelve observations of error 0.5: four elements on the bound at the mode."""
    rng = np.random.default_rng(11)
    m = 12
    H = jacobian_matrix(rng, m, N)
    truth = np.array([0.0, 0.0, 1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 2.5, 1.2])
    noise = 0.5
    z = H @ truth + noise * rng.standard_normal(m)
    x_0 = np.full(N, 0.2)
    return Linear(x_0, exponential_cov(N, 1.0), H, noise**2 * np.eye(m), z)


def test_bounded_draws_are_emcee_draws():
    """
    ``"bounded"`` with a lower bound of zero, several elements on it at the mode: the truncated posterior's means, standard deviations, a correlation, and the total's 5% and 95% quantiles are emcee's.

    emcee draws the prior times the likelihood, set to zero where an element
    is below zero.
    """
    case = bounded_case()
    mean, cov = case.analytic()
    P_0, P_z = np.linalg.inv(case.S_0), np.linalg.inv(case.S_z)

    def log_prob(x):
        d, r = x - case.x_0, case.z - x @ case.H.T
        value = -0.5 * np.einsum("ki,ij,kj->k", d, P_0, d) - 0.5 * np.einsum(
            "ki,ij,kj->k", r, P_z, r
        )
        return np.where((x >= 0.0).all(axis=1), value, -np.inf)

    post = fips.solve(case.problem(), "bounded", lower=0.0)
    on_bound = post.estimate.values == 0.0
    assert on_bound.sum() >= 4 and (mean < 0.0).sum() >= 2  # several on the bound

    spread = np.sqrt(np.diag(cov))
    start = np.maximum(mean, 0.0) + 0.5 * spread
    statistic = state_statistic(lambda x: x, most_correlated(cov))
    theirs = emcee_estimates(log_prob, start, 0.1 * spread, statistic)
    ours = fips_estimates(post, statistic)
    assert_agree(statistic_names(N), ours, theirs, state_scale(theirs.value))


# ---------------------------------------------------------------------------
# fips.hierarchical with two hyperparameters
# ---------------------------------------------------------------------------

PRIORS = {
    "prior_scale": LogNormal(0.0, 0.5),
    "obs_scale": LogNormal(math.log(0.5), 0.5),
}
"""The hyperparameters' priors: a factor on the prior's standard deviation, and one on the observation error's."""


def hierarchical_case() -> Linear:
    """Ten elements, fifteen observations; the truth drawn with the prior's std doubled, the errors with half the observation error's."""
    rng = np.random.default_rng(5)
    m = 15
    H = jacobian_matrix(rng, m, N)
    S_0 = exponential_cov(N, 1.0)
    truth = 2.0 * np.linalg.cholesky(S_0) @ rng.standard_normal(N)
    z = H @ truth + 0.5 * rng.standard_normal(m)
    return Linear(np.zeros(N), S_0, H, np.eye(m), z)


def hierarchical_names() -> list[str]:
    """The names of what the hierarchical statistic returns."""
    return [
        *(f"{name} {s}" for name in PRIORS for s in ("mean", "std", "q05", "q95")),
        *statistic_names(N),
    ]


def hierarchical_statistic(pair: tuple[int, int]):
    """The statistic of draws of the hyperparameters (``2 x draws``) with the state's (``n x draws``): each hyperparameter's mean, std and 5% and 95% quantiles, then the state's statistic."""
    state = state_statistic(lambda x: x, pair)

    def statistic(s: np.ndarray, x: np.ndarray) -> list[float]:
        first = []
        for row in s:
            first += [row.mean(), row.std(ddof=1), *np.quantile(row, [0.05, 0.95])]
        return first + state(x)

    return statistic


def emcee_hierarchical(case: Linear, statistic, runs=8, walkers=32, steps=20_000):
    """
    The hierarchical posterior by emcee: the two log factors drawn from their posterior, the evidence written in NumPy, then the state drawn from its Gaussian at each kept draw.

    In the log of each factor, a lognormal prior is a normal one, so the log
    posterior is the two normals' log densities plus the evidence,
    ``log N(z; H x_0, a^2 H S_0 H^T + b^2 S_z)``.
    """
    mus = np.array([p.mu for p in PRIORS.values()])
    sigmas = np.array([p.sigma for p in PRIORS.values()])
    HSH = case.H @ case.S_0 @ case.H.T
    r = case.z - case.H @ case.x_0
    m = len(r)

    def log_prob(t: np.ndarray) -> np.ndarray:
        a2, b2 = np.exp(2 * t[:, :1, None]), np.exp(2 * t[:, 1:, None])
        C = a2 * HSH + b2 * case.S_z  # k x m x m
        _, logdet = np.linalg.slogdet(C)
        quad = np.einsum(
            "i,ki->k",
            r,
            np.linalg.solve(C, np.broadcast_to(r, (len(t), m))[..., None])[..., 0],
        )
        prior = -0.5 * (((t - mus) / sigmas) ** 2).sum(axis=1)
        return prior - 0.5 * (logdet + quad + m * math.log(2 * math.pi))

    P_0, P_z = np.linalg.inv(case.S_0), np.linalg.inv(case.S_z)
    replicates = []
    for k in range(runs):
        t, tau = emcee_draws(
            log_prob, mus, 0.1 * sigmas, 2000 + k, walkers, steps, 2000
        )
        assert steps > 50 * tau, f"emcee run {k}: tau = {tau:.0f} steps of {steps}"
        t = t[:, :: walkers * 5]  # every fifth step of the ensemble, one walker each
        rng = np.random.default_rng(3000 + k)
        x = np.empty((N, t.shape[1]))
        for i, (la, lb) in enumerate(t.T):
            cov = np.linalg.inv(
                np.exp(-2 * la) * P_0 + np.exp(-2 * lb) * case.H.T @ P_z @ case.H
            )
            mean = case.x_0 + cov @ case.H.T @ (np.exp(-2 * lb) * P_z) @ r
            x[:, i] = mean + np.linalg.cholesky(cov) @ rng.standard_normal(N)
        replicates.append(statistic(np.exp(t), x))
    return Estimates.of(replicates)


@pytest.mark.parametrize("path", ["fast", "general"])
def test_hierarchical_draws_are_emcee_draws(path):
    """
    ``fips.hierarchical`` with two hyperparameters, on each path: the hyperparameters' means, standard deviations and 5% and 95% quantiles, and the state's statistic, are emcee's.

    fips's errors are the spread of eight runs with their own seeds.
    """
    case = hierarchical_case()
    problem = case.problem()

    def build(s):
        return Problem(
            problem.prior,
            Covariance(s["prior_scale"] ** 2 * case.S_0, problem.prior.space),
            problem.obs,
            Covariance(s["obs_scale"] ** 2 * case.S_z, problem.obs.space),
            problem.forward,
        )

    _, cov = case.analytic()
    statistic = hierarchical_statistic(most_correlated(cov))
    replicates = []
    for seed in range(8):
        result = fips.hierarchical(
            problem if path == "fast" else build,
            PRIORS,
            draws=4000,
            burn_in=500,
            seed=seed,
        )
        s = result.hyperparameters[list(PRIORS)].to_numpy().T
        x = result.sample(s.shape[1], rng=0).to_numpy()
        replicates.append(statistic(s, x))
    ours = Estimates.of(replicates)
    theirs = emcee_hierarchical(case, statistic)
    scale = np.concatenate(
        [
            np.repeat(theirs.value[[1, 5]], 4),  # each hyperparameter's std
            state_scale(theirs.value[8:]),
        ]
    )
    assert_agree(hierarchical_names(), ours, theirs, scale)
