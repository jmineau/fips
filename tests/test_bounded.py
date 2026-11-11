"""
Tests for the ``"bounded"`` method: the mode of the posterior with bounds on the state, and draws of the truncated posterior.

A wrong mode or a sampler that has not mixed still gives a labeled map of the
right shape, so the method is held to answers that do not use its algorithm,
in layers that each catch what the one before misses:

1. **Reduction.** With no bound the posterior is the analytic one, bit for
   bit (the oracle in ``test_oracle.py`` runs ``"bounded"`` with its default
   options on every problem); with a bound no element reaches it is the
   analytic estimate to roundoff, with nothing on a bound.
2. **The mode, by certificates.** The optimality conditions, with the
   posterior precision formed explicitly; SciPy's bounded least squares
   (``lsq_linear(method="bvls")``) on the whitened, stacked system; the one
   element case, where the mode is the analytic estimate clipped. In both
   analytic forms, with a Kronecker prior, a state of two blocks with one
   unbounded, and the inputs in any order.
3. **The sampler, against exact answers.** One element against
   ``scipy.stats.truncnorm``; two correlated elements against numerical
   integration; within Monte Carlo standard errors that count the draws'
   autocorrelation.
4. **Its own world.** A truth drawn from the prior truncated to the bounds,
   and data from it: the truth's rank among the draws is uniform, and their
   intervals cover it as often as they claim (an exact posterior does, on
   average over its prior).
5. **Twins.** ``with_gamma`` and ``with_data`` against a fresh solve; the
   covariance of aggregates against the same draws aggregated; the Gaussian
   at the mode (gain, DOFS, the information content) and the chi-squared
   terms against dense formulas; a pickle round trip; the options' checks.
"""

import dataclasses
import math
import pickle

import numpy as np
import pandas as pd
import pytest
from numpy.linalg import LinAlgError
from scipy import integrate
from scipy.linalg import cholesky
from scipy.optimize import lsq_linear
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import norm, truncnorm

import fips
from fips import Covariance, Operator, Problem, Space, Vector
from fips.linalg import KroneckerOperator
from fips.methods import _draws, bounded
from fips.methods.analytic import Information, Whitened
from fips.methods.bounded import Bounded, BoundedOptions

# ---------------------------------------------------------------------------
# Problems, written as arrays, and the references written with them
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Arrays:
    """A problem as arrays: ``z = H x + c + e``, ``x ~ N(x_0, S_0)``, ``e ~ N(0, S_z)``."""

    x_0: np.ndarray
    S_0: np.ndarray
    H: np.ndarray
    S_z: np.ndarray
    z: np.ndarray
    c: np.ndarray

    @classmethod
    def random(cls, seed: int, n: int, m: int, corr: float = 3.0) -> "Arrays":
        """A prior near zero with correlated errors and a truth near zero: many analytic estimates below zero."""
        rng = np.random.default_rng(seed)
        i = np.arange(n)
        C = np.exp(-np.abs(i[:, None] - i[None, :]) / corr)
        std = rng.uniform(0.5, 1.5, n)
        S_0 = std[:, None] * C * std[None, :]
        x_0 = rng.uniform(0.0, 0.3, n)
        H = rng.uniform(0.0, 1.0, (m, n)) * (rng.uniform(0.0, 1.0, (m, n)) < 0.5)
        S_z = np.diag(rng.uniform(0.05, 0.2, m))
        c = rng.normal(1.0, 0.1, m)
        truth = np.exp(rng.normal(-1.5, 1.0, n)) - 0.3  # a few below zero
        z = H @ truth + c + rng.normal(0.0, np.sqrt(np.diag(S_z)))
        return cls(x_0, S_0, H, S_z, z, c)

    def problem(self, state: Space | None = None, prior_cov=None) -> Problem:
        """The labeled problem, on `state` (one ``flux`` block of string labels by default)."""
        n, m = len(self.x_0), len(self.z)
        state = state or Space(
            {"flux": pd.Index([f"c{k}" for k in range(n)], name="cell")}
        )
        obs = Space({"conc": pd.Index([f"o{k}" for k in range(m)], name="obs_id")})
        return Problem(
            prior=Vector(self.x_0, state),
            prior_cov=prior_cov
            if prior_cov is not None
            else Covariance(self.S_0, state),
            obs=Vector(self.z, obs),
            obs_cov=Covariance(self.S_z, obs),
            forward=Operator(self.H, obs, state),
            offset=Vector(self.c, obs),
        )

    def posterior(
        self, gamma: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The analytic posterior: its mean, covariance and precision, from the formulas."""
        S_z = self.S_z / gamma
        P = np.linalg.inv(self.S_0) + self.H.T @ np.linalg.inv(S_z) @ self.H
        S = np.linalg.inv(P)
        x_hat = S @ (
            np.linalg.solve(self.S_0, self.x_0)
            + self.H.T @ np.linalg.solve(S_z, self.z - self.c)
        )
        return x_hat, 0.5 * (S + S.T), P

    def bvls(self, lower, upper, gamma: float = 1.0) -> np.ndarray:
        """The bounded mode by SciPy's bounded least squares on the whitened, stacked system."""
        L_z = cholesky(self.S_z / gamma, lower=True)
        L_0 = cholesky(self.S_0, lower=True)
        A = np.vstack([np.linalg.solve(L_z, self.H), np.linalg.inv(L_0)])
        b = np.concatenate(
            [np.linalg.solve(L_z, self.z - self.c), np.linalg.solve(L_0, self.x_0)]
        )
        return lsq_linear(A, b, bounds=(lower, upper), method="bvls", tol=1e-15).x


def kkt(
    x: np.ndarray, x_hat: np.ndarray, P: np.ndarray, lower, upper
) -> dict[str, float]:
    """
    Return how far `x` is from the optimality conditions of min (x - x_hat)^T P (x - x_hat) over the box, relative to the scale of the state.

    The gradient ``g = P (x - x_hat)`` must be zero at a free element, at
    least zero at one on its lower bound and at most zero at one on its upper
    bound; `x` must be in the box.
    """
    lower = np.broadcast_to(np.asarray(lower, dtype=float), x.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), x.shape)
    scale = max(np.abs(x).max(), np.abs(x_hat).max())
    g = P @ (x - x_hat) * np.sqrt(np.diag(np.linalg.inv(P)))  # in the state's units
    at_lower = np.isclose(x, lower, rtol=0.0, atol=1e-12 * scale)
    at_upper = np.isclose(x, upper, rtol=0.0, atol=1e-12 * scale)
    free = ~(at_lower | at_upper)
    return {
        "feasible": float(max((lower - x).max(), (x - upper).max(), 0.0)) / scale,
        "free gradient": float(np.abs(g[free]).max(initial=0.0)) / scale,
        "lower sign": float(max(-g[at_lower].min(initial=0.0), 0.0)) / scale,
        "upper sign": float(max(g[at_upper].max(initial=0.0), 0.0)) / scale,
    }


def assert_kkt(x, x_hat, P, lower, upper, tol=1e-9):
    """Assert that `x` meets the optimality conditions to `tol` of the scale of the state."""
    conditions = kkt(x, x_hat, P, lower, upper)
    assert max(conditions.values()) < tol, conditions


# ---------------------------------------------------------------------------
# Monte Carlo standard errors with autocorrelated draws
# ---------------------------------------------------------------------------


def iat(series: np.ndarray) -> float:
    """The integrated autocorrelation time of `series`, by Sokal's adaptive window (the window ends at five times the estimate)."""
    x = series - series.mean()
    n = len(x)
    if not np.any(x):
        return 1.0
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for window in range(1, n):
        tau = 1.0 + 2.0 * acf[1 : window + 1].sum()
        if window >= 5 * tau:
            break
    return max(tau, 1.0)


def standard_error(series: np.ndarray) -> float:
    """The Monte Carlo standard error of the mean of `series`, with its autocorrelation counted."""
    return float(series.std(ddof=1) * np.sqrt(iat(series) / len(series)))


def assert_moments(
    draws: np.ndarray, mean: np.ndarray, cov: np.ndarray, within: float = 4.5
):
    """Assert that the draws' mean, variances and covariances are within `within` Monte Carlo standard errors of the exact ones."""
    n = len(mean)
    centered = (
        draws - mean[:, None]
    )  # about the exact mean: the series' means are the moments
    for i in range(n):
        se = standard_error(draws[i])
        assert abs(draws[i].mean() - mean[i]) < within * se, (
            i,
            draws[i].mean(),
            mean[i],
            se,
        )
        for j in range(i, n):
            product = centered[i] * centered[j]
            se = standard_error(product)
            assert abs(product.mean() - cov[i, j]) < within * se, (
                i,
                j,
                product.mean(),
                cov[i, j],
                se,
            )


def gaussian_problem(x_hat: np.ndarray, S: np.ndarray) -> Problem:
    """A problem whose analytic posterior is exactly ``N(x_hat, S)``: a prior ``N(0, 2 S)`` and one observation of each element, ``2 x_hat`` with error ``2 S``."""
    n = len(x_hat)
    state = Space({"flux": pd.RangeIndex(n, name="cell")})
    obs = Space({"conc": pd.RangeIndex(n, name="obs_id")})
    return Problem(
        prior=Vector(np.zeros(n), state),
        prior_cov=Covariance(2.0 * S, state),
        obs=Vector(2.0 * x_hat, obs),
        obs_cov=Covariance(2.0 * S, obs),
        forward=Operator(np.eye(n), obs, state),
    )


def truncated_moments_2d(mean, S, lower, upper):
    """
    The mean and covariance of ``N(mean, S)`` truncated to a box in two dimensions, by integration.

    Over the first element numerically; over the second in closed form,
    given the first (its conditional is normal).
    """
    sd1 = math.sqrt(S[0, 0])
    lo1 = max(lower[0], mean[0] - 12 * sd1)
    hi1 = min(upper[0], mean[0] + 12 * sd1)
    slope = S[0, 1] / S[0, 0]
    sc = math.sqrt(S[1, 1] - S[0, 1] ** 2 / S[0, 0])

    def inner(x1):
        mc = mean[1] + slope * (x1 - mean[0])
        a, b = (lower[1] - mc) / sc, (upper[1] - mc) / sc
        P = norm.cdf(b) - norm.cdf(a)
        pa, pb = norm.pdf(a), norm.pdf(b)
        apa = a * pa if np.isfinite(a) else 0.0
        bpb = b * pb if np.isfinite(b) else 0.0
        first = mc * P + sc * (pa - pb)
        second = (mc**2 + sc**2) * P + 2 * mc * sc * (pa - pb) + sc**2 * (apa - bpb)
        return norm.pdf(x1, mean[0], sd1), P, first, second

    def moment(f):
        return integrate.quad(
            lambda x1: f(x1, *inner(x1)), lo1, hi1, epsabs=0, epsrel=1e-12, limit=200
        )[0]

    Z = moment(lambda x1, w, P, first, second: w * P)
    m1 = moment(lambda x1, w, P, first, second: w * P * x1) / Z
    m2 = moment(lambda x1, w, P, first, second: w * first) / Z
    s11 = moment(lambda x1, w, P, first, second: w * P * x1**2) / Z - m1**2
    s22 = moment(lambda x1, w, P, first, second: w * second) / Z - m2**2
    s12 = moment(lambda x1, w, P, first, second: w * first * x1) / Z - m1 * m2
    return np.array([m1, m2]), np.array([[s11, s12], [s12, s22]])


# ---------------------------------------------------------------------------
# 1. Reduction
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize(
    "bounds",
    [
        {},
        {"lower": None, "upper": None},
        {"lower": -math.inf, "upper": math.inf},
        {"lower": {"flux": -math.inf}},
        {"upper": "vector of inf"},
    ],
    ids=["default", "none", "infinite numbers", "infinite block", "infinite vector"],
)
def test_no_bound_is_the_analytic_posterior_bit_for_bit(bounds):
    """With no finite bound the solution is the analytic one itself, and every output is the same, bit for bit."""
    arrays = Arrays.random(0, 6, 9)
    problem = arrays.problem()
    if bounds.get("upper") == "vector of inf":
        bounds = {"upper": Vector(np.full(6, np.inf), problem.prior.space)}
    analytic = fips.solve(problem, gamma=2.0)
    post = fips.solve(problem, "bounded", gamma=2.0, **bounds)
    assert isinstance(post._solution, (Whitened, Information))
    assert isinstance(post.options, BoundedOptions)
    assert post.mean is post.estimate
    np.testing.assert_array_equal(post.estimate.values, analytic.estimate.values)
    np.testing.assert_array_equal(post.variances.values, analytic.variances.values)
    np.testing.assert_array_equal(post.cov().dense(), analytic.cov().dense())
    np.testing.assert_array_equal(
        post.kalman_gain.dense(), analytic.kalman_gain.dense()
    )
    assert post.dofs == analytic.dofs
    assert post.chi2 == analytic.chi2
    with pytest.raises(fips.UnavailableError, match="No finite bound was given"):
        _ = post.at_bound
    assert not hasattr(post, "at_bound")
    # with_gamma keeps the bounded options
    assert post.with_gamma(0.5).options == dataclasses.replace(post.options, gamma=0.5)


@pytest.mark.usefixtures("analytic_form")
def test_a_bound_no_element_reaches_is_the_analytic_estimate():
    """A bound far below every estimate holds nothing: the analytic estimate, gain and diagnostics, to roundoff."""
    arrays = Arrays.random(1, 6, 9)
    problem = arrays.problem()
    analytic = fips.solve(problem)
    post = fips.solve(problem, "bounded", lower=-1e9, upper=1e9)
    assert isinstance(post._solution, Bounded)
    assert not post.at_bound.any()
    np.testing.assert_allclose(
        post.estimate.values, analytic.estimate.values, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        post.kalman_gain.dense(), analytic.kalman_gain.dense(), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        post.dofs_per_element.values, analytic.dofs_per_element.values, rtol=1e-12
    )
    np.testing.assert_allclose(
        post.information_content, analytic.information_content, rtol=1e-12
    )
    np.testing.assert_allclose(post.chi2, analytic.chi2, rtol=1e-10)
    np.testing.assert_allclose(post.data_misfit, analytic.data_misfit, rtol=1e-10)
    np.testing.assert_allclose(post.state_misfit, analytic.state_misfit, rtol=1e-10)
    tests, expected = post.chi2_tests(), analytic.chi2_tests()
    assert list(tests["dof"]) == list(expected["dof"])
    np.testing.assert_allclose(tests["chi2"], expected["chi2"], rtol=1e-9)


# ---------------------------------------------------------------------------
# 2. The mode, by certificates
# ---------------------------------------------------------------------------

BOX = {
    "lower": (0.0, math.inf),
    "upper": (-math.inf, 0.4),
    "both": (0.0, 0.4),
}


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("box", list(BOX))
@pytest.mark.parametrize(
    "n, m, corr",
    [(12, 20, 3.0), (15, 6, 3.0), (10, 10, 0.01)],
    ids=["n<m", "n>m", "independent"],
)
def test_mode_is_the_constrained_minimum(box, n, m, corr):
    """The mode meets the optimality conditions with the precision formed explicitly, and equals SciPy's bounded least squares."""
    lower, upper = BOX[box]
    held = 0
    for seed in range(4):
        arrays = Arrays.random(seed, n, m, corr=corr)
        post = fips.solve(
            arrays.problem(), "bounded", lower=lower, upper=upper, gamma=1.5
        )
        x_hat, _, P = arrays.posterior(gamma=1.5)
        x = post.estimate.values
        assert_kkt(x, x_hat, P, lower, upper)
        reference = arrays.bvls(lower, upper, gamma=1.5)
        scale = max(np.abs(reference).max(), np.abs(x_hat).max())
        np.testing.assert_allclose(x, reference, rtol=0, atol=1e-9 * scale)
        assert np.all(x >= lower) and np.all(x <= upper)
        # at_bound is the active set of the reference's optimality conditions
        at = np.isclose(reference, lower, atol=1e-9 * scale) | np.isclose(
            reference, upper, atol=1e-9 * scale
        )
        np.testing.assert_array_equal(post.at_bound.values, at)
        held += int(at.sum())
    assert held > 0  # the cases hold elements on a bound


@pytest.mark.parametrize(
    "x_hat, lower, upper",
    [
        (-0.7, 0.0, None),
        (0.3, 0.0, None),
        (1.2, None, 1.0),
        (-2.0, -1.0, 1.0),
        (2.0, -1.0, 1.0),
        (0.5, -1.0, 1.0),
    ],
)
def test_one_element_is_its_estimate_clipped(x_hat, lower, upper):
    """With one element, the nearest point of the box in any metric is the analytic estimate clipped."""
    problem = gaussian_problem(np.array([x_hat]), np.array([[0.3]]))
    post = fips.solve(problem, "bounded", lower=lower, upper=upper, draws=10)
    lo = -np.inf if lower is None else lower
    hi = np.inf if upper is None else upper
    assert post.estimate.values[0] == pytest.approx(np.clip(x_hat, lo, hi), abs=1e-15)
    assert bool(post.at_bound.iloc[0]) == (not lo <= x_hat <= hi)


@pytest.mark.usefixtures("analytic_form")
def test_kronecker_prior_gives_the_dense_prior_mode():
    """A prior kept as Kronecker factors gives the mode of the same prior written dense, and of the reference."""
    rng = np.random.default_rng(3)
    times, cells = 3, 5
    T = np.exp(-np.abs(np.subtract.outer(np.arange(times), np.arange(times))) / 2.0)
    X = np.exp(-np.abs(np.subtract.outer(np.arange(cells), np.arange(cells))) / 1.5)
    std = rng.uniform(0.5, 1.2, times * cells)
    kron = KroneckerOperator([T, X], std=std)
    arrays = Arrays.random(3, times * cells, 12)
    arrays.S_0 = kron.dense()
    state = Space(
        {
            "flux": pd.MultiIndex.from_product(
                [range(times), range(cells)], names=["time", "cell"]
            )
        }
    )
    dense = fips.solve(arrays.problem(state), "bounded", lower=0.0)
    factored = fips.solve(
        arrays.problem(state, Covariance(kron, state)), "bounded", lower=0.0
    )
    assert dense.at_bound.any()
    np.testing.assert_array_equal(factored.at_bound.values, dense.at_bound.values)
    np.testing.assert_allclose(
        factored.estimate.values, dense.estimate.values, rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        dense.estimate.values, arrays.bvls(0.0, np.inf), rtol=0, atol=1e-10
    )


def _two_blocks(seed: int) -> tuple[Arrays, Space]:
    """A flux block of 8 cells and a bias block of 2 sites, the bias negative in the data."""
    arrays = Arrays.random(seed, 10, 16)
    arrays.H[:8, 8] = 1.0  # the first site's observations see its bias
    arrays.H[8:, 9] = 1.0
    arrays.H[:, 8:][arrays.H[:, 8:] != 1.0] = 0.0
    arrays.z = arrays.z - 1.5 * arrays.H[:, 8:].sum(axis=1)
    arrays.x_0[8:] = 0.0
    state = Space(
        {
            "flux": pd.Index([f"c{k}" for k in range(8)], name="cell"),
            "bias": pd.Index(["A", "B"], name="site"),
        }
    )
    return arrays, state


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("given", ["number", "vector"])
def test_bounds_by_block_leave_the_other_blocks_free(given):
    """``lower={"flux": ...}`` bounds the flux and leaves the bias free to go negative; the reference bounds the same elements."""
    arrays, state = _two_blocks(4)
    problem = arrays.problem(state)
    flux_bound = (
        0.0
        if given == "number"
        else Vector(np.zeros(8), Space({"flux": state.select("flux")}))
    )
    post = fips.solve(problem, "bounded", lower={"flux": flux_bound})
    lower = np.r_[np.zeros(8), -np.inf, -np.inf]
    x_hat, _, P = arrays.posterior()
    x = post.estimate.values
    assert_kkt(x, x_hat, P, lower, np.inf)
    np.testing.assert_allclose(x, arrays.bvls(lower, np.inf), rtol=0, atol=1e-9)
    assert np.all(post.estimate["flux"].values >= 0.0)
    assert np.all(post.estimate["bias"].values < 0.0)  # free, and negative
    assert post.at_bound["flux"].any() and not post.at_bound["bias"].any()


def test_a_number_on_several_blocks_bounds_them_all_and_warns(caplog):
    """``lower=0.0`` on a flux and a bias block holds the bias at zero too, which is rarely meant: the solve warns, naming the blocks; a dict, a single block or an infinite number does not."""
    arrays, state = _two_blocks(4)
    problem = arrays.problem(state)
    with caplog.at_level("WARNING", logger="fips.methods.bounded"):
        post = fips.solve(problem, "bounded", lower=0.0, draws=20)
    assert "lower=0 bounds every block of the state: 'flux', 'bias'" in caplog.text
    assert "lower={'flux': 0}" in caplog.text
    assert np.all(post.estimate["bias"].values == 0.0)  # what the warning is about
    caplog.clear()
    with caplog.at_level("WARNING", logger="fips.methods.bounded"):
        fips.solve(problem, "bounded", lower={"flux": 0.0}, draws=20)
        fips.solve(problem, "bounded", lower=-math.inf, upper={"flux": 5.0}, draws=20)
        fips.solve(Arrays.random(0, 6, 8).problem(), "bounded", lower=0.0, draws=20)
    assert "every block" not in caplog.text
    with caplog.at_level("WARNING", logger="fips.methods.bounded"):
        fips.solve(problem, "bounded", upper=9.0, draws=20)
    assert "upper=9 bounds every block" in caplog.text


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("seed", range(3))
def test_input_order_does_not_change_the_labeled_mode(seed):
    """Each input, and the bound, in its own random order gives the same mode and active set, by label."""
    arrays, state = _two_blocks(seed)
    problem = arrays.problem(state)
    bound = Vector(np.r_[np.zeros(8), -np.inf, -np.inf], state)
    ref = fips.solve(problem, "bounded", lower=bound)
    rng = np.random.default_rng(100 + seed)

    def shuffled(obj):
        if isinstance(obj, pd.Series):
            return obj.iloc[rng.permutation(len(obj))]
        return obj.iloc[rng.permutation(obj.shape[0]), rng.permutation(obj.shape[1])]

    other = fips.solve(
        Problem(
            prior=Vector.from_series(shuffled(problem.prior.to_series())),
            prior_cov=Covariance.from_frame(shuffled(problem.prior_cov.to_frame())),
            obs=Vector.from_series(shuffled(problem.obs.to_series())),
            obs_cov=Covariance.from_frame(shuffled(problem.obs_cov.to_frame())),
            forward=Operator.from_frame(shuffled(problem.forward.to_frame())),
            offset=Vector.from_series(shuffled(problem.offset.to_series())),
        ),
        "bounded",
        lower=Vector.from_series(shuffled(bound.to_series())),
    )
    got = other.estimate.to_series().reindex(ref.estimate.index)
    np.testing.assert_allclose(got.values, ref.estimate.values, rtol=0, atol=1e-10)
    pd.testing.assert_series_equal(
        other.at_bound.reindex(ref.at_bound.index), ref.at_bound
    )
    assert ref.at_bound.any()


def test_a_mode_that_fails_its_conditions_raises(monkeypatch):
    """The solve checks the optimality conditions of the mode the active set returns, and refuses a wrong one."""
    problem = Arrays.random(0, 8, 10).problem()
    real = bounded._mode

    def wrong(*args):
        mode, side, multipliers = real(*args)
        held = np.flatnonzero(side)
        side = side.copy()
        side[held[0]] = 0  # release one element without moving it: not stationary
        multipliers = multipliers.copy()
        multipliers[held[0]] = 0.0
        return mode, side, multipliers

    monkeypatch.setattr(bounded, "_mode", wrong)
    with pytest.raises(LinAlgError, match="optimality conditions"):
        fips.solve(problem, "bounded", lower=0.0)


# ---------------------------------------------------------------------------
# 3. The sampler, against exact answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_hat, lower, upper",
    [(-0.6, 0.0, None), (0.0, 0.0, None), (0.4, None, 0.0), (-0.2, -0.5, 0.8)],
    ids=[
        "mean below the lower bound",
        "mean at it",
        "mean above the upper bound",
        "both",
    ],
)
def test_one_element_draws_are_the_truncated_normal(x_hat, lower, upper):
    """The draws of one element have the mean and variance of ``scipy.stats.truncnorm``, within their Monte Carlo errors."""
    variance = 0.3
    post = fips.solve(
        gaussian_problem(np.array([x_hat]), np.array([[variance]])),
        "bounded",
        lower=lower,
        upper=upper,
    )
    draws = post.sample(20_000, rng=1).to_numpy()
    sd = math.sqrt(variance)
    lo = -np.inf if lower is None else lower
    hi = np.inf if upper is None else upper
    mean, var = truncnorm.stats(
        (lo - x_hat) / sd, (hi - x_hat) / sd, loc=x_hat, scale=sd, moments="mv"
    )
    assert np.all(draws >= lo) and np.all(draws <= hi)
    assert_moments(draws, np.array([float(mean)]), np.array([[float(var)]]))


TWO = {
    "rho 0.8, both on lower bounds": (
        [[1.0, 0.8], [0.8, 1.0]],
        [-0.5, -0.2],
        [0.0, 0.0],
        [np.inf, np.inf],
    ),
    "rho -0.8, one on a lower bound": (
        [[1.0, -0.8], [-0.8, 1.0]],
        [-0.5, 0.3],
        [0.0, -np.inf],
        [np.inf, np.inf],
    ),
    "rho 0.6, a box and an upper bound": (
        [[0.5, 0.3], [0.3, 0.5]],
        [0.2, 0.6],
        [-0.4, -np.inf],
        [0.3, 0.2],
    ),
    "rho -0.95, both bounded": (
        [[1.0, -0.95], [-0.95, 1.0]],
        [0.1, 0.1],
        [0.0, 0.0],
        [np.inf, np.inf],
    ),
}


@pytest.mark.parametrize("case", list(TWO))
def test_two_correlated_elements_match_integration(case):
    """Two correlated elements: the draws' mean and covariance match those integrated numerically, within their Monte Carlo errors."""
    S, x_hat, lower, upper = (np.array(v, dtype=float) for v in TWO[case])
    state = Space({"flux": pd.RangeIndex(2, name="cell")})
    post = fips.solve(
        gaussian_problem(x_hat, S),
        "bounded",
        lower=Vector(lower, state),
        upper=Vector(upper, state),
    )
    draws = post.sample(20_000, rng=2).to_numpy()
    assert np.all(draws >= lower[:, None]) and np.all(draws <= upper[:, None])
    mean, cov = truncated_moments_2d(x_hat, S, lower, upper)
    assert_moments(draws, mean, cov)


def test_the_sampler_reflects_in_the_posterior_metric(monkeypatch):
    """
    The teeth of the tests above: a sampler that reflects off a bound in the Euclidean metric fails them.

    Reflecting only the element's own velocity (as for an uncorrelated
    posterior) keeps the draws in the box and their energy, but not the
    distribution of a correlated one.
    """

    def euclidean(velocity, column, i):
        out = velocity.copy()
        out[i] = -out[i]
        return out

    monkeypatch.setattr(bounded, "_reflect", euclidean)
    S, x_hat, lower, upper = (
        np.array(v, dtype=float) for v in TWO["rho 0.8, both on lower bounds"]
    )
    state = Space({"flux": pd.RangeIndex(2, name="cell")})
    post = fips.solve(
        gaussian_problem(x_hat, S),
        "bounded",
        lower=Vector(lower, state),
        upper=Vector(upper, state),
    )
    draws = post.sample(20_000, rng=2).to_numpy()
    mean, cov = truncated_moments_2d(x_hat, S, lower, upper)
    with pytest.raises(AssertionError):
        assert_moments(draws, mean, cov)


def test_the_error_of_a_chains_standard_deviation():
    """``_std_errors`` is ``1/sqrt(2N)`` for independent Gaussian draws and ``sqrt(tau/(2N))`` for a Gaussian AR(1) chain (``tau = (1 + rho)/(1 - rho)`` for its squares, ``rho^2``), the spread of the standard deviation over many such chains."""
    rng = np.random.default_rng(0)
    N, chains = 2000, 400
    independent = rng.standard_normal((chains, N))
    error, _ = _draws._std_errors(independent)
    assert np.median(error) == pytest.approx(1 / np.sqrt(2 * N), rel=0.1)
    rho = 0.9
    ar = np.empty((chains, N))
    ar[:, 0] = rng.standard_normal(chains)
    for k in range(1, N):
        ar[:, k] = rho * ar[:, k - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(
            chains
        )
    error, tau = _draws._std_errors(ar)
    expected = np.sqrt((1 + rho**2) / (1 - rho**2) / (2 * N))
    assert np.median(error) == pytest.approx(expected, rel=0.15)
    spread = ar.std(axis=1, ddof=1).std() / ar.std(axis=1, ddof=1).mean()
    assert spread == pytest.approx(expected, rel=0.15)
    assert np.all(_draws._std_errors(np.ones((2, 50)))[0] == 0.0)  # never moves


def test_draws_piled_against_a_bound_warn_how_many_would_do(caplog):
    """
    Cells whose analytic estimates are 2.7 standard deviations below their bound: their draws pile against it and mix slowly, so a standard deviation from 300 draws is uncertain by 10% to 40%, and the solve warns, saying how many draws would bring it to 10%; cells away from a bound do not warn.

    One chain's estimate of that error varies (from 5% to 30% at 500 draws
    over ten seeds for one such cell), so four are drawn, and the warning
    needs one. At the default 2000 draws the error of such a cell is about
    12% (rms over 24 solves of a correlated 3-cell problem, where the free
    cells were good to 1.4%), near the warning's threshold.
    """
    x_hat = np.array([3.0, -2.7, -2.7, -2.7, -2.7, 4.0])
    with caplog.at_level("WARNING", logger="fips.methods.bounded"):
        post = fips.solve(
            gaussian_problem(x_hat, np.eye(6)), "bounded", lower=0.0, draws=300
        )
        _ = post.variances
    assert "The posterior's draws mix slowly" in caplog.text
    assert "draws would bring it to 10%" in caplog.text
    error, _ = _draws._std_errors(np.asarray(post._solution.kept()))
    assert error[1:5].max() > 0.1 and error[[0, 5]].max() < 0.07
    caplog.clear()
    with caplog.at_level("WARNING", logger="fips.methods.bounded"):
        far = fips.solve(
            gaussian_problem(-x_hat, np.eye(6)), "bounded", lower=-10.0, draws=300
        )
        _ = far.variances
    assert "mix slowly" not in caplog.text


def test_kept_draws_give_the_variances_and_covariances():
    """``variances`` and ``cov(W)`` are those of the kept draws: ``sample(draws, rng=seed)``, aggregated the same way."""
    arrays = Arrays.random(5, 8, 10)
    problem = arrays.problem()
    post = fips.solve(problem, "bounded", lower=0.0, draws=300, seed=7)
    draws = post.sample(300, rng=7).to_numpy()
    np.testing.assert_allclose(
        post.variances.values, draws.var(axis=1, ddof=1), rtol=1e-12
    )
    np.testing.assert_allclose(
        post.cov().dense(), np.cov(draws), rtol=1e-12, atol=1e-15
    )
    W = _halves(problem.prior.space)
    np.testing.assert_allclose(
        post.cov(W).dense(),
        np.atleast_2d(np.cov(W.dense() @ draws)),
        rtol=1e-12,
        atol=1e-15,
    )
    uncertainty = post.uncertainty(W)
    np.testing.assert_allclose(
        uncertainty["posterior_std"].values,
        np.sqrt(np.diag(np.cov(W.dense() @ draws))),
        rtol=1e-12,
    )
    assert np.all(draws >= 0.0)
    # a solve is reproducible, and another seed gives other draws
    np.testing.assert_array_equal(
        fips.solve(problem, "bounded", lower=0.0, draws=300, seed=7).variances.values,
        post.variances.values,
    )
    assert not np.array_equal(
        fips.solve(problem, "bounded", lower=0.0, draws=300, seed=8).variances.values,
        post.variances.values,
    )


def test_draws_of_a_larger_problem_have_the_moments_of_a_long_reference_chain():
    """
    Ten correlated elements, many on a bound: two independent chains agree on every mean and variance within their errors.

    No exact answer exists in ten dimensions; two chains from other seeds,
    one long, are compared, so a chain that has not mixed (or is biased by its
    start at the mode) shows.
    """
    arrays = Arrays.random(6, 10, 6)
    post = fips.solve(arrays.problem(), "bounded", lower=0.0)
    assert post.at_bound.sum() >= 2
    a = post.sample(4000, rng=11).to_numpy()
    b = post.sample(16000, rng=12).to_numpy()
    for i in range(10):
        se = math.hypot(standard_error(a[i]), standard_error(b[i]))
        assert abs(a[i].mean() - b[i].mean()) < 4 * se, i
        ca, cb = (a[i] - a[i].mean()) ** 2, (b[i] - b[i].mean()) ** 2
        se = math.hypot(standard_error(ca), standard_error(cb))
        assert abs(ca.mean() - cb.mean()) < 4 * se, i


# ---------------------------------------------------------------------------
# 4. Its own world
# ---------------------------------------------------------------------------


def _calibration(trials: int, draws: int, seed: int) -> tuple[np.ndarray, float]:
    """
    Return, in the method's own world, the p-value of the uniformity of the truth's rank among the draws, for each quantity, and the coverage of the 90% intervals.

    The quantities are each element, their total and the difference of the
    first two. The rank of the truth among a trial's draws (the fraction
    below it) is uniform over trials when the draws are of the posterior
    (simulation-based calibration, Talts et al. 2018); its histogram in
    tenths is tested with a chi-squared test of 9 degrees of freedom.
    """
    rng = np.random.default_rng(seed)
    n, m = 5, 4
    state = Space({"flux": pd.RangeIndex(n, name="cell")})
    obs = Space({"conc": pd.RangeIndex(m, name="obs_id")})
    x_0 = np.array([0.1, 0.4, -0.2, 0.0, 0.8])
    std = np.array([0.5, 0.3, 0.6, 0.4, 0.5])
    H = np.random.default_rng(99).uniform(0.0, 1.0, (m, n))
    sz = np.full(m, 0.3)
    problem = Problem(
        prior=Vector(x_0, state),
        prior_cov=Covariance.diagonal(std**2, state),
        obs=Vector(np.zeros(m), obs),
        obs_cov=Covariance.diagonal(sz**2, obs),
        forward=Operator(H, obs, state),
    )
    post = fips.solve(problem, "bounded", lower=0.0, draws=draws)
    Q = np.vstack([np.eye(n), np.ones(n), np.eye(n)[0] - np.eye(n)[1]])
    ranks = np.empty((trials, len(Q)))
    for k in range(trials):
        # the prior truncated to the bounds, element by element (it is diagonal)
        truth = truncnorm.rvs(-x_0 / std, np.inf, loc=x_0, scale=std, random_state=rng)
        z = H @ truth + rng.normal(0.0, sz)
        kept = post.with_data(Vector(z, obs))._solution.kept()
        ranks[k] = ((Q @ kept) < (Q @ truth)[:, None]).mean(axis=1)
    counts = np.stack([np.histogram(r, bins=10, range=(0.0, 1.0))[0] for r in ranks.T])
    expected = trials / 10
    p_values = chi2_distribution.sf(
        ((counts - expected) ** 2 / expected).sum(axis=1), 9
    )
    coverage = float(np.mean((ranks > 0.05) & (ranks < 0.95)))
    return p_values, coverage


def test_draws_are_calibrated_in_their_own_world():
    """
    In the world the posterior assumes, the truth is a draw like the others: its rank among the draws is uniform.

    The truth is drawn from the prior truncated to the bounds, and the
    observations from it: the posterior with the bounds is then exact, so
    the rank of the truth among its draws is uniform over the trials, for
    each element and any function of them (the total, a difference), and
    the 90% intervals cover 90% of the time. A sampler that draws another
    distribution fails it: one that reflects off the bounds in the Euclidean
    metric gives p-values below 1e-20 here.
    """
    p_values, coverage = _calibration(trials=150, draws=300, seed=0)
    assert p_values.min() > 1e-4, p_values
    assert abs(coverage - 0.9) < 0.04, coverage


@pytest.mark.slow
def test_draws_are_calibrated_in_their_own_world_many_trials():
    """The calibration test with four times the trials and three times the draws."""
    p_values, coverage = _calibration(trials=600, draws=1000, seed=1)
    assert p_values.min() > 1e-4, p_values
    assert abs(coverage - 0.9) < 0.02, coverage


def test_trials_read_a_bounded_totals_interval_from_the_draws():
    """
    In its own world a bounded total's 95% interval from ``trials`` holds the truth about 95% of the time; read about the mode it would not.

    Twelve cells with a prior mean of 0.2 and a standard deviation of 1,
    truncated at zero (a mean of about 0.87 a cell), seen weakly: the mode
    stays near the prior's 0.2 a cell, far below the truths' total, and the
    draws are where the posterior is. ``trials``' ``lower`` and ``upper`` are
    each trial's draws' quantiles, so the coverage is near 0.95 (40 trials
    know it to about 0.035; 0.8 is four of those below). The mode plus or
    minus 1.96 of the draws' standard deviations covers about 0.05 here, as
    the comparison page's 0.025, which the test checks too, so that a
    ``trials`` that went back to that reading fails.
    """
    n, m, mu, sd, noise = 12, 4, 0.2, 1.0, 10.0
    state = Space({"flux": pd.RangeIndex(n, name="cell")})
    observed = Space({"conc": pd.RangeIndex(m, name="obs_id")})
    H = np.random.default_rng(0).uniform(0.0, 0.3, (m, n))
    problem = Problem(
        prior=Vector(np.full(n, mu), state),
        prior_cov=fips.Independent().with_std(sd),
        obs=Vector(np.zeros(m), observed),
        obs_cov=fips.Independent().with_std(noise),
        forward=Operator(H, observed, state),
    )
    total = Operator(
        np.ones((1, n)), Space({"total": pd.Index(["all"], name="what")}), state
    )

    def world(rng):
        """A truth from the prior truncated at zero, and its observations."""
        x = truncnorm.rvs(-mu / sd, np.inf, loc=mu, scale=sd, size=n, random_state=rng)
        return Vector(x, state), Vector(H @ x + rng.normal(0.0, noise, m), observed)

    post = fips.solve(problem, "bounded", lower=0.0, draws=300)
    table = post.trials(world, 40, rng=0, W=total)
    covered = (table["lower"] <= table["truth"]) & (table["truth"] <= table["upper"])
    assert covered.mean() >= 0.8, covered.mean()
    about_the_mode = (table["estimate"] - table["truth"]).abs() <= 1.96 * table[
        "posterior_std"
    ]
    assert about_the_mode.mean() <= 0.25, about_the_mode.mean()


# ---------------------------------------------------------------------------
# 5. Twins: the Gaussian at the mode, the chi-squared terms, reuse, pickles
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("n, m", [(12, 20), (15, 6)])
def test_the_gaussian_at_the_mode(n, m):
    """The gain, averaging kernel, DOFS and information content are those of the analytic posterior conditioned on the elements on a bound, by dense formulas."""
    arrays = Arrays.random(2, n, m)
    problem = arrays.problem()
    gamma = 1.5
    post = fips.solve(problem, "bounded", lower=0.0, gamma=gamma)
    a = np.flatnonzero(post.at_bound.values)
    assert len(a)
    x_hat, S, P = arrays.posterior(gamma)
    K = S @ arrays.H.T @ np.linalg.inv(arrays.S_z / gamma)
    E = np.eye(n)[a]
    K_a = (np.eye(n) - S[:, a] @ np.linalg.solve(S[np.ix_(a, a)], E)) @ K
    A_a = K_a @ arrays.H
    np.testing.assert_allclose(post.kalman_gain.dense(), K_a, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        post.averaging_kernel().dense(), A_a, rtol=1e-9, atol=1e-11
    )
    np.testing.assert_allclose(post.dofs, np.trace(A_a), rtol=1e-9)
    np.testing.assert_allclose(
        post.dofs_per_element.values, np.diag(A_a), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(post.dofs_per_element.values[a], 0.0, atol=1e-12)
    sign, logdet = np.linalg.slogdet(np.eye(n) - A_a)
    assert sign > 0
    np.testing.assert_allclose(post.information_content, -0.5 * logdet, rtol=1e-9)
    # the gain of a total, without the n x m gain
    W = _halves(problem.prior.space)
    np.testing.assert_allclose(
        post.averaging_kernel(W).dense(),
        _aggregated_kernel(W.dense(), A_a),
        rtol=1e-9,
        atol=1e-12,
    )


def _halves(state: Space) -> Operator:
    """The totals of the first half of the state and of the second."""
    n = len(state)
    W = np.zeros((2, n))
    W[0, : n // 2] = W[1, n // 2 :] = 1.0
    return Operator(
        W, Space({"total": pd.Index(["first", "second"], name="half")}), state
    )


def _aggregated_kernel(W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """``W A W^+``, the averaging kernel of aggregates."""
    return W @ A @ W.T @ np.linalg.inv(W @ W.T)


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("n, m", [(8, 14), (14, 6)])
def test_chi2_terms_at_the_mode(n, m):
    """The misfits and Rodgers' tests are those of the mode, by dense formulas; the cost is the analytic minimum plus the mode's distance from it."""
    arrays = Arrays.random(3, n, m)
    gamma = 0.7
    post = fips.solve(arrays.problem(), "bounded", lower=0.0, gamma=gamma)
    analytic = fips.solve(arrays.problem(), gamma=gamma)
    assert post.at_bound.any()
    x = post.estimate.values
    x_hat, S, P = arrays.posterior(gamma)
    r = arrays.z - arrays.H @ x - arrays.c
    dx = x - arrays.x_0
    data = r @ np.linalg.solve(arrays.S_z, r)
    state = dx @ np.linalg.solve(arrays.S_0, dx)
    np.testing.assert_allclose(post.data_misfit, data, rtol=1e-9)
    np.testing.assert_allclose(post.state_misfit, state, rtol=1e-9)
    np.testing.assert_allclose(post.chi2, gamma * data + state, rtol=1e-9)
    np.testing.assert_allclose(
        post.chi2, analytic.chi2 + (x - x_hat) @ P @ (x - x_hat), rtol=1e-9
    )
    # Rodgers' tests at the mode
    G = arrays.H @ arrays.S_0 @ arrays.H.T + arrays.S_z / gamma
    S_z = arrays.S_z / gamma
    residual = r @ np.linalg.solve(S_z @ np.linalg.solve(G, S_z), r)
    # the increment, in the directions the observations constrain: its whitened
    # components projected on the range of its covariance there
    L_0 = cholesky(arrays.S_0, lower=True)
    xi = np.linalg.solve(L_0, dx)
    C = np.linalg.solve(
        L_0,
        np.linalg.solve(
            L_0, arrays.S_0 @ arrays.H.T @ np.linalg.solve(G, arrays.H @ arrays.S_0)
        ).T,
    )
    eigenvalues, V = np.linalg.eigh(0.5 * (C + C.T))
    kept = eigenvalues > 1e-9
    increment = np.sum((V[:, kept].T @ xi) ** 2 / eigenvalues[kept])
    tests = post.chi2_tests()
    np.testing.assert_allclose(
        tests.loc["innovation", "chi2"], analytic.chi2, rtol=1e-9
    )
    np.testing.assert_allclose(tests.loc["residual", "chi2"], residual, rtol=1e-9)
    np.testing.assert_allclose(tests.loc["increment", "chi2"], increment, rtol=1e-8)
    np.testing.assert_allclose(
        tests.loc["modeled_increment", "chi2"], increment, rtol=1e-8
    )
    assert list(tests["dof"]) == [m, m, min(n, m), min(n, m)]


@pytest.mark.usefixtures("analytic_form")
def test_with_gamma_and_with_data_are_a_fresh_solve():
    """Another gamma, or other observations, from the kept factors give the solve's own mode, active set and Gaussian at the mode."""
    arrays = Arrays.random(7, 10, 14)
    problem = arrays.problem()
    post = fips.solve(problem, "bounded", lower=0.0, upper=1.5, draws=1000)
    other = Vector(
        arrays.z + np.random.default_rng(0).normal(0.0, 0.5, len(arrays.z)),
        problem.obs.space,
    )
    pairs = [
        (
            post.with_gamma(3.0),
            fips.solve(problem, "bounded", lower=0.0, upper=1.5, gamma=3.0, draws=1000),
        ),
        (
            post.with_data(other),
            fips.solve(
                post.with_data(other).problem,
                "bounded",
                lower=0.0,
                upper=1.5,
                draws=1000,
            ),
        ),
    ]
    for reused, fresh in pairs:
        assert isinstance(reused._solution, Bounded)
        assert reused.options == fresh.options
        np.testing.assert_array_equal(reused.at_bound.values, fresh.at_bound.values)
        np.testing.assert_allclose(
            reused.estimate.values, fresh.estimate.values, rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            reused.kalman_gain.dense(), fresh.kalman_gain.dense(), rtol=1e-9, atol=1e-11
        )
        np.testing.assert_allclose(reused.dofs, fresh.dofs, rtol=1e-9)
        np.testing.assert_allclose(reused.chi2, fresh.chi2, rtol=1e-9)
        # the draws: chains of the same posterior from factors equal to roundoff,
        # which part after some bounces: the same moments within their errors
        a, b = reused._solution.kept(), fresh._solution.kept()
        for i in range(len(a)):
            se = math.hypot(standard_error(a[i]), standard_error(b[i]))
            assert abs(a[i].mean() - b[i].mean()) < 5 * se, i


def test_trials_runs_the_bounded_posterior_of_each_trial():
    """``trials`` solves each trial's data with the bounds, and reports each trial's own spread from its draws."""
    arrays = Arrays.random(8, 6, 8)
    post = fips.solve(arrays.problem(), "bounded", lower=0.0, draws=100)
    table = post.trials(n=4, rng=0)
    assert table.index.names == ["trial", "block", "cell"]
    estimates = table["estimate"].to_numpy().reshape(4, 6)
    assert np.all(estimates >= 0.0)
    spreads = table["posterior_std"].to_numpy().reshape(4, 6)
    assert not np.allclose(spreads[0], spreads[1])  # each trial's own


def test_what_a_bounded_posterior_cannot_give():
    """The mode is neither the mean nor linear in the data: the mean is the draws', and no recovery, no data worth."""
    problem = Arrays.random(0, 6, 8).problem()
    post = fips.solve(problem, "bounded", lower=0.0, draws=50)
    kept = post._solution.kept()
    np.testing.assert_array_equal(post.mean.values, kept.mean(axis=1))
    assert post.mean.space == post.estimate.space
    assert not np.allclose(post.mean.values, post.estimate.values)
    with pytest.raises(TypeError, match="estimate_is_linear"):
        post.recovery()
    W = _halves(problem.prior.space)
    with pytest.raises(TypeError, match="innovation_precision"):
        post.data_worth("obs_id", W)
    with pytest.raises(TypeError, match="estimate_is_linear"):
        post.candidate_worth(problem.forward, problem.obs_cov, "obs_id", W)
    with pytest.raises(TypeError, match="no at_bound"):
        _ = fips.solve(problem).at_bound
    analytic = fips.solve(problem)
    assert not hasattr(analytic, "at_bound") and hasattr(post, "at_bound")
    assert not hasattr(analytic, "iterations") and not hasattr(analytic, "eigenvalues")
    with pytest.raises(fips.UnavailableError, match="orthant probabilities"):
        _ = post.log_evidence
    with pytest.raises(TypeError, match="obs must be a Vector, got ndarray"):
        post.with_data(problem.obs.values)


def test_pickle_round_trip():
    """A bounded posterior pickles without its caches and gives the same answers back, its draws made again from the seed."""
    problem = Arrays.random(9, 7, 9).problem()
    post = fips.solve(problem, "bounded", lower=0.0, draws=150)
    variances = post.variances.values
    state = post._solution.__getstate__()
    assert not {"_columns_", "_draws", "_sampler_", "_conditioned_"} & set(state)
    back = pickle.loads(pickle.dumps(post))
    np.testing.assert_array_equal(back.estimate.values, post.estimate.values)
    pd.testing.assert_series_equal(back.at_bound, post.at_bound)
    np.testing.assert_array_equal(back.variances.values, variances)
    assert not back._solution.mode.flags.writeable
    assert back.options == post.options


# ---------------------------------------------------------------------------
# The options
# ---------------------------------------------------------------------------


def test_options_are_frozen_hashable_and_keep_a_dict_through_replace():
    """The options hash and pickle; a dict of bounds by block survives ``with_gamma``'s copy and shows as the dict."""
    vector = Vector(np.zeros(2), Space({"flux": pd.Index(["a", "b"], name="cell")}))
    options = BoundedOptions(lower={"flux": 0.0, "other": vector}, upper=3, draws=10)
    hash(options)
    assert options.upper == 3.0 and isinstance(options.upper, float)
    assert repr(options.lower).startswith("{'flux': 0.0, 'other': ")
    copied = dataclasses.replace(options, gamma=2.0)
    assert copied.lower == options.lower and copied.gamma == 2.0
    assert pickle.loads(pickle.dumps(options)).lower.parts[0] == ("flux", 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.draws = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "options, error, match",
    [
        ({"lower": math.nan}, ValueError, "lower is NaN"),
        ({"upper": {"flux": math.nan}}, ValueError, "upper of block 'flux' is NaN"),
        ({"lower": math.inf}, ValueError, "lower is inf, a bound no value can satisfy"),
        (
            {"upper": -math.inf},
            ValueError,
            "upper is -inf, a bound no value can satisfy",
        ),
        (
            {"lower": 1.0, "upper": 1.0},
            ValueError,
            r"lower \(1.0\) must be below upper \(1.0\)",
        ),
        (
            {"lower": 2.0, "upper": 1.0},
            ValueError,
            r"lower \(2.0\) must be below upper",
        ),
        ({"lower": pd.Series([0.0])}, TypeError, "Vector.from_series"),
        ({"lower": "0"}, TypeError, "lower is a number or a Vector, not str"),
        ({"lower": True}, TypeError, "not bool"),
        ({"lower": {0: 0.0}}, TypeError, "block names, strings"),
        ({"draws": 0}, ValueError, "draws must be an integer of at least 2"),
        ({"draws": 1}, ValueError, "draws must be an integer of at least 2"),
        ({"burn_in": -1}, ValueError, "burn_in must be a nonnegative integer"),
        ({"seed": 1.5}, ValueError, "seed must be a nonnegative integer"),
        ({"gamma": -1.0}, ValueError, "gamma must be positive"),
    ],
)
def test_options_refuse(options, error, match):
    """Each option refuses what it cannot be, naming the problem."""
    with pytest.raises(error, match=match):
        BoundedOptions(**options)


def test_bounds_are_checked_against_the_state():
    """A bound on other labels, a block the state does not have, or bounds that cross at an element are refused by the solve, naming them."""
    arrays, state = _two_blocks(0)
    problem = arrays.problem(state)
    with pytest.raises(
        ValueError, match="names a block the state does not have, 'flx'"
    ):
        fips.solve(problem, "bounded", lower={"flx": 0.0})
    wrong = Vector(
        np.zeros(8), Space({"flux": pd.Index([f"d{k}" for k in range(8)], name="cell")})
    )
    with pytest.raises(
        ValueError,
        match="The lower bounds of block 'flux' and the block do not have the same labels",
    ):
        fips.solve(problem, "bounded", lower={"flux": wrong})
    with pytest.raises(
        ValueError,
        match="The upper bounds and the state do not have the same labels: blocks missing",
    ):
        fips.solve(problem, "bounded", upper=wrong)
    lower = Vector(np.r_[np.zeros(8), -np.inf, 1.0], state)
    with pytest.raises(
        ValueError,
        match=r"lower must be below upper at every state element, and is not in block 'bias': 'B'",
    ):
        fips.solve(problem, "bounded", lower=lower, upper={"bias": 0.5})
    infinite = Vector(np.r_[np.zeros(8), -np.inf, np.inf], state)
    with pytest.raises(ValueError, match="lower is inf somewhere"):
        fips.solve(problem, "bounded", lower=infinite)
