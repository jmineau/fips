"""
Tests for Problem.simulate: the synthetic truth and its observations.

``simulate`` draws a truth from the prior and observations from the problem's
own model, so an inversion run on them must show the statistics it claims.
Each test holds one of those claims to a Monte Carlo of `TRIALS` seeded trials
(a seed makes them deterministic), in both forms the analytic method factors
in (the ``analytic_form`` fixture):

- the 95% interval of each state element and of each total holds the truth
  `0.95` of the time, within `COVERAGE_TOL`: the binomial standard deviation of
  a frequency over 2,000 trials is sqrt(0.95 * 0.05 / 2000) = 0.0049, so the
  tolerance is six of them, so no frequency fails by chance, and a draw from
  the wrong distribution misses by far more;
- the reduced chi-squared averages 1 within 4 standard errors, sqrt(2 / m) per
  trial;
- the Desroziers ratio of each block averages 1 within 4 standard errors of
  its own trials;
- when the world's noise is four times the assumed one, the chi-squared
  averages the closed form tr(G^-1 (H S_0 H^T + S_z,world)), G = H S_0 H^T +
  S_z / gamma, within 4 standard errors of the trials.

The rest is bookkeeping: a given truth is put on the prior's labels, a seed
repeats, a generator is used as given.

``Posterior.recovery`` is the exact answer of those loops: the mean and spread
of the error, and the coverage, when the data come from a world. It is held to
the closed form in ``test_oracle.py``; here, to the loop itself, for a
mis-specified world and a fixed truth, within `STANDARD_ERRORS` of the trials;
and to what it refuses and what it allocates.
"""

import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import norm

import fips
from fips import Covariance, Operator, Posterior, Problem, Space, Vector
from fips.linalg import KroneckerOperator
from fips.methods import analytic

TRIALS = 2000
COVERAGE_TOL = 0.03
STANDARD_ERRORS = 4

TIMES = pd.Index([0, 1, 2], name="time")
CELLS = pd.Index(["a", "b"], name="cell")
FLUX = pd.MultiIndex.from_product([TIMES, CELLS])
SITES = pd.Index(["A", "B"], name="site")
STATE = Space({"flux": FLUX, "bias": SITES})
CONC = pd.MultiIndex.from_product([SITES, [0, 1, 2, 3]], names=["site", "hour"])
OTHER = pd.Index([0, 1, 2, 3], name="obs_id")
OBS = Space({"co2": CONC, "ch4": OTHER})
N, M = len(STATE), len(OBS)


def _spd(rng: np.random.Generator, size: int, scale: float) -> np.ndarray:
    """A dense symmetric positive-definite matrix, with correlations."""
    A = rng.normal(size=(size, size))
    return scale * (A @ A.T / size + np.eye(size))


class World:
    """The arrays of a problem, and the Problem built on them."""

    def __init__(self, seed: int = 0, noise_factor: float = 1.0):
        rng = np.random.default_rng(seed)
        self.H = rng.uniform(0.0, 1.0, (M, N))
        self.x0 = rng.normal(1.0, 0.5, N)
        self.c = rng.normal(2.0, 0.5, M)
        self.S0 = _spd(rng, N, 1.0)
        self.Sz = _spd(rng, M, 0.5)
        self.z = rng.normal(size=M)
        self.noise_factor = noise_factor

    def problem(self, noise_factor: float = 1.0, **inputs) -> Problem:
        """The problem; `noise_factor` times the mismatch is another world."""
        args = {
            "prior": Vector(self.x0, STATE, name="prior"),
            "prior_cov": Covariance(self.S0, STATE),
            "obs": Vector(self.z, OBS, name="obs"),
            "obs_cov": Covariance(noise_factor * self.Sz, OBS),
            "forward": Operator(self.H, OBS, STATE),
            "offset": Vector(self.c, OBS),
        }
        args.update(inputs)
        return Problem(**args)


def _flux_totals() -> Operator:
    """The total of the flux block in each time step; the bias block passes through."""
    return fips.aggregate.sum(STATE, by="time", blocks="flux")


# ---------------------------------------------------------------------------
# The statistics of an honest inversion
# ---------------------------------------------------------------------------


def test_coverage_of_each_element_and_each_total(analytic_form):
    """The 95% interval holds the truth 95% of the time, per element and per total."""
    problem = World().problem()
    post = fips.solve(problem)
    W = _flux_totals()
    sd_state = np.sqrt(post.variances.values)
    sd_totals = np.sqrt(post.cov(W).variances.values)
    rng = np.random.default_rng(0)
    state_in, totals_in = [], []
    for _ in range(TRIALS):
        truth, obs = problem.simulate(rng)
        estimate = post.with_data(obs).estimate
        state_in.append(np.abs(estimate.values - truth.values) <= 1.96 * sd_state)
        totals_in.append(
            np.abs((W @ estimate).values - (W @ truth).values) <= 1.96 * sd_totals
        )
    state_coverage = np.mean(state_in, axis=0)
    totals_coverage = np.mean(totals_in, axis=0)
    assert state_coverage.shape == (N,)
    assert totals_coverage.shape == (len(W.rows),)
    np.testing.assert_allclose(state_coverage, 0.95, atol=COVERAGE_TOL)
    np.testing.assert_allclose(totals_coverage, 0.95, atol=COVERAGE_TOL)


def test_reduced_chi2_and_desroziers_are_one(analytic_form):
    """In its own world the reduced chi-squared and the Desroziers ratios average 1."""
    problem = World().problem()
    post = fips.solve(problem)
    rng = np.random.default_rng(1)
    chi2, ratios = [], []
    for _ in range(TRIALS):
        _, obs = problem.simulate(rng)
        trial = post.with_data(obs)
        chi2.append(trial.reduced_chi2)
        ratios.append(trial.desroziers(by="block")["ratio"].to_numpy())
    chi2, ratios = np.array(chi2), np.array(ratios)
    standard_error = np.sqrt(2.0 / M / TRIALS)
    assert abs(chi2.mean() - 1.0) < STANDARD_ERRORS * standard_error
    assert ratios.shape == (TRIALS, 2)
    # the ratio's spread is not 2 / m: its own trials give its standard error
    ratio_error = ratios.std(axis=0, ddof=1) / np.sqrt(TRIALS)
    assert np.all(np.abs(ratios.mean(axis=0) - 1.0) < STANDARD_ERRORS * ratio_error)


@pytest.mark.parametrize("gamma", [1.0, 4.0])
def test_chi2_of_a_noisier_world_is_its_closed_form(analytic_form, gamma):
    """Data with four times the assumed noise give the chi-squared the formula says."""
    arrays = World()
    post = fips.solve(arrays.problem(), gamma=gamma)
    world = arrays.problem(noise_factor=4.0)
    HS0H = arrays.H @ arrays.S0 @ arrays.H.T
    G = HS0H + arrays.Sz / gamma
    expected = np.trace(np.linalg.solve(G, HS0H + 4.0 * arrays.Sz))
    rng = np.random.default_rng(2)
    chi2 = np.array(
        [post.with_data(world.simulate(rng)[1]).chi2 for _ in range(TRIALS)]
    )
    standard_error = chi2.std(ddof=1) / np.sqrt(TRIALS)
    assert abs(chi2.mean() - expected) < STANDARD_ERRORS * standard_error
    # the check can tell this world from the honest one, whose mean is M
    assert abs(expected - M) > 20 * standard_error


# ---------------------------------------------------------------------------
# What simulate returns
# ---------------------------------------------------------------------------


def test_draw_is_the_prior_and_the_mismatch_sampled_in_turn():
    """Truth = x0 + a prior draw, then obs = H truth + c + a mismatch draw, from one generator."""
    arrays = World()
    problem = arrays.problem()
    truth, obs = problem.simulate(np.random.default_rng(7))
    rng = np.random.default_rng(7)
    expected_truth = arrays.x0 + problem.prior_cov.sample(rng).values
    expected_obs = (
        arrays.H @ expected_truth + arrays.c + problem.obs_cov.sample(rng).values
    )
    np.testing.assert_array_equal(truth.values, expected_truth)
    np.testing.assert_array_equal(obs.values, expected_obs)
    assert truth.space == problem.prior.space
    assert obs.space == problem.obs.space
    assert obs.name == "obs"


def test_a_given_truth_comes_back_on_the_priors_labels():
    """A truth in any order is put in the prior's; only the error is drawn."""
    arrays = World()
    problem = arrays.problem()
    given = Vector(arrays.x0 * 1.5, STATE, name="inventory")
    on_other_order = Vector.from_blocks(
        {
            "bias": given.to_series(block="bias").iloc[::-1],
            "flux": given.to_series(block="flux").iloc[::-1],
        },
        name="inventory",
    )
    assert not on_other_order.index.equals(given.index)
    truth, obs = problem.simulate(np.random.default_rng(3), truth=on_other_order)
    assert truth.space == problem.prior.space
    assert truth.name == "inventory"
    np.testing.assert_array_equal(truth.values, given.values)
    noise = problem.obs_cov.sample(np.random.default_rng(3)).values
    np.testing.assert_allclose(obs.values, problem.predict(given).values + noise)
    # the same truth in the prior's order gives the same observations
    _, again = problem.simulate(np.random.default_rng(3), truth=given)
    np.testing.assert_array_equal(obs.values, again.values)


def test_a_given_truth_is_checked():
    """Not a Vector raises a TypeError; other labels raise a ValueError."""
    problem = World().problem()
    with pytest.raises(TypeError, match="truth must be a Vector"):
        problem.simulate(0, truth=np.ones(N))  # type: ignore[arg-type]
    short = Vector(np.ones(len(FLUX) - 1), Space({"flux": FLUX[:-1]}))
    with pytest.raises(ValueError, match="truth and the prior"):
        problem.simulate(0, truth=short)


def test_a_seed_repeats_and_a_generator_is_used_as_given():
    """The same seed gives the same pair; one generator gives trials in sequence."""
    problem = World().problem()
    first = problem.simulate(5)
    again = problem.simulate(5)
    other = problem.simulate(6)
    np.testing.assert_array_equal(first[0].values, again[0].values)
    np.testing.assert_array_equal(first[1].values, again[1].values)
    assert not np.allclose(first[0].values, other[0].values)

    generator = np.random.default_rng(5)
    one = problem.simulate(generator)
    two = problem.simulate(generator)
    np.testing.assert_array_equal(one[1].values, first[1].values)
    assert not np.allclose(one[1].values, two[1].values)
    # the second trial continues the stream the first left
    rng = np.random.default_rng(5)
    problem.prior_cov.sample(rng)
    problem.obs_cov.sample(rng)
    next_truth = problem.prior.values + problem.prior_cov.sample(rng).values
    np.testing.assert_array_equal(two[0].values, next_truth)


def test_simulate_without_an_offset_or_a_seed():
    """No offset is zero; None draws from fresh entropy and still has the right labels."""
    arrays = World()
    problem = arrays.problem(offset=None)
    truth, obs = problem.simulate(1)
    rng = np.random.default_rng(1)
    prior_draw = problem.prior_cov.sample(rng).values
    np.testing.assert_allclose(
        obs.values,
        arrays.H @ (arrays.x0 + prior_draw) + problem.obs_cov.sample(rng).values,
    )
    fresh_truth, fresh_obs = problem.simulate()
    assert fresh_truth.space == problem.prior.space
    assert fresh_obs.space == problem.obs.space
    assert not np.allclose(fresh_truth.values, truth.values)


def test_structured_covariances_give_the_same_statistics(analytic_form):
    """A Kronecker prior and a within-group mismatch (the forms a flux inversion has) hold the claim."""
    n_trials = 500
    grid = pd.MultiIndex.from_product(
        [pd.Index(range(5), name="time"), pd.Index(range(4), name="cell")]
    )
    state = Space({"flux": grid[np.r_[0:7, 9:20]]})  # a gap in the grid
    obs_space = Space({"co2": CONC})
    rng = np.random.default_rng(4)
    H = rng.uniform(0.0, 1.0, (len(obs_space), len(state)))
    problem = Problem(
        prior=Vector(rng.normal(1.0, 0.3, len(state)), state),
        prior_cov=fips.Kronecker(
            fips.Exponential("time", 1.5), fips.Exponential("cell", 2.0)
        ).with_std(0.7),
        obs=Vector(np.zeros(len(obs_space)), obs_space),
        obs_cov=fips.Exponential("hour", 1.5, within="site").with_std(0.5),
        forward=Operator(H, obs_space, state),
    )
    assert isinstance(problem.prior_cov.data, KroneckerOperator)
    post = fips.solve(problem)
    sd = np.sqrt(post.variances.values)
    gen = np.random.default_rng(5)
    inside = []
    chi2 = []
    for _ in range(n_trials):
        truth, obs = problem.simulate(gen)
        trial = post.with_data(obs)
        inside.append(np.abs(trial.estimate.values - truth.values) <= 1.96 * sd)
        chi2.append(trial.reduced_chi2)
    # 500 trials: the binomial standard deviation is 0.0097, so 0.04 is four of them
    np.testing.assert_allclose(np.mean(inside, axis=0), 0.95, atol=0.04)
    standard_error = np.sqrt(2.0 / len(obs_space) / n_trials)
    assert abs(np.mean(chi2) - 1.0) < STANDARD_ERRORS * standard_error


# ---------------------------------------------------------------------------
# Posterior.recovery against the loop it replaces
# ---------------------------------------------------------------------------


def _misspecified_world(arrays: World) -> Problem:
    """The noise four times the assumed one, the ch4 offset 1 too high, and another forward operator."""
    rng = np.random.default_rng(8)
    offset = arrays.c.copy()
    offset[OBS.slice("ch4")] += 1.0
    return arrays.problem(
        noise_factor=4.0,
        forward=Operator(arrays.H + rng.normal(0.0, 0.2, arrays.H.shape), OBS, STATE),
        offset=Vector(offset, OBS),
    )


@pytest.mark.parametrize("gamma", [1.0, 4.0])
@pytest.mark.parametrize("kind", ["misspecified", "fixed truth"])
def test_recovery_is_the_mean_of_the_loop(analytic_form, kind, gamma):
    """
    The bias, spread and coverage of ``recovery`` are what the trials show.

    For each state element and each total: the mean error within 4 standard
    errors of ``bias`` (its trials' standard deviation over sqrt(2,000)), the
    standard deviation of the error within 4 standard errors of ``std``
    (``std / sqrt(2 (2,000 - 1))``), and the frequency with which the 95%
    interval holds the truth within 4 binomial standard deviations of
    ``coverage``. The world is mis-specified (noise 4x, an offset on one
    block, another forward operator) or the problem itself with a fixed truth
    of 1.5 times the prior.
    """
    arrays = World()
    post = fips.solve(arrays.problem(), gamma=gamma)
    W = _flux_totals()
    if kind == "misspecified":
        world, truth = _misspecified_world(arrays), None
    else:
        world, truth = arrays.problem(), Vector(1.5 * arrays.x0, STATE)
    tables = [post.recovery(world, truth=truth), post.recovery(world, truth=truth, W=W)]
    errors: list[list[np.ndarray]] = [[], []]
    rng = np.random.default_rng(9)
    for _ in range(TRIALS):
        x, z = world.simulate(rng, truth=truth)
        estimate = post.with_data(z).estimate
        errors[0].append(estimate.values - x.values)
        errors[1].append((W @ estimate).values - (W @ x).values)
    half = norm.ppf(0.975)
    far = False
    for table, error in zip(tables, errors, strict=True):
        error = np.array(error)
        mean, spread = error.mean(axis=0), error.std(axis=0, ddof=1)
        bias, std = table["bias"].to_numpy(), table["std"].to_numpy()
        se_mean = spread / np.sqrt(TRIALS)
        se_std = std / np.sqrt(2 * (TRIALS - 1))
        np.testing.assert_array_less(np.abs(mean - bias), STANDARD_ERRORS * se_mean)
        np.testing.assert_array_less(np.abs(spread - std), STANDARD_ERRORS * se_std)
        claimed = table["posterior_std"].to_numpy()
        inside = np.abs(error) <= half * claimed
        p = table["coverage"].to_numpy()
        np.testing.assert_array_less(
            np.abs(inside.mean(axis=0) - p),
            STANDARD_ERRORS * np.sqrt(p * (1 - p) / TRIALS),
        )
        # the check has power: this world is far from the honest one (bias 0,
        # std = posterior_std) in its bias or its spread
        far |= bool(
            np.any(np.abs(bias) > 20 * se_mean)
            or np.any(np.abs(std - claimed) > 20 * se_std)
        )
    assert far


def test_recovery_refuses_what_it_cannot_answer():
    """A method not linear in the data, other labels, and bad arguments raise, naming the problem."""
    arrays = World()
    problem = arrays.problem()
    post = fips.solve(problem)

    class NotLinear:
        """A solution like the analytic one, without the claim of linearity."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "estimate_is_linear":
                raise AttributeError(name)
            return getattr(self._inner, name)

    other = Posterior(problem, NotLinear(post._solution), post.options)
    with pytest.raises(TypeError, match="not linear|linear in the observations"):
        other.recovery()

    keep = Space({"flux": FLUX[1:], "bias": SITES})
    with pytest.raises(ValueError, match="world's state.*'flux'.*\\(0, 'a'\\)"):
        post.recovery(problem.restrict(keep))
    daily = problem.aggregate(fips.aggregate.mean(OBS, by="site", blocks="co2"))
    with pytest.raises(ValueError, match="world's observations.*'co2'"):
        post.recovery(daily)
    with pytest.raises(TypeError, match="world must be a Problem"):
        post.recovery(post)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="truth must be a Vector"):
        post.recovery(truth=arrays.x0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="truth and the prior"):
        post.recovery(truth=Vector(np.ones(len(FLUX)), Space({"flux": FLUX})))
    with pytest.raises(ValueError, match="aggregation"):
        post.recovery(W=fips.aggregate.sum(Space({"flux": FLUX}), by="time"))
    for level in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="level must be between 0 and 1"):
            post.recovery(level=level)


def test_an_error_with_no_spread_is_inside_or_not():
    """A total the data do not see, with a fixed truth, has std 0 and coverage 0 or 1."""
    arrays = World()
    H = arrays.H.copy()
    H[:, STATE.slice("bias")] = 0.0  # the observations see no bias
    S0 = np.diag(np.diag(arrays.S0))  # and the prior does not tie it to the flux
    problem = arrays.problem(
        forward=Operator(H, OBS, STATE), prior_cov=Covariance(S0, STATE)
    )
    post = fips.solve(problem)
    near = arrays.x0.copy()
    near[STATE.slice("bias")] += [0.1, 10.0]  # inside and far outside the interval
    table = post.recovery(truth=Vector(near, STATE)).xs("bias", level="block")
    np.testing.assert_array_equal(table["std"].to_numpy(), 0.0)
    np.testing.assert_array_equal(table["coverage"].to_numpy(), [1.0, 0.0])


@pytest.mark.parametrize("form", ["whitened", "information"])
def test_recovery_allocates_nothing_n_by_n_or_m_by_m(form):
    """
    For an aggregation of k rows, recovery allocates nothing n x n or m x m.

    Whitened: n = 2,400 (a Kronecker prior) and m = 30, so one n x n array is
    46 MB; information: n = 30 and m = 2,400, so one m x m array is 46 MB.
    The world has its own forward operator and mismatch. The traced peak of
    ``recovery`` alone (after the solve, which holds the posterior's own
    arrays) must stay under half of one such array.
    """
    rng = np.random.default_rng(0)
    small = 30  # and the grid has 2,400 labels
    times = pd.date_range("2024-01-01", periods=48, freq="D")
    lats, lons = np.arange(10) * 0.1, np.arange(5) * 0.1
    grid = pd.MultiIndex.from_product([times, lats, lons], names=["time", "lat", "lon"])
    if form == "whitened":
        state = Space({"flux": grid})
        obs = Space({"conc": pd.RangeIndex(small, name="obs_id")})
        prior_cov = fips.Kronecker(
            fips.Exponential("time", "5D"),
            fips.Exponential(["lat", "lon"], 20.0, metric="haversine"),
        ).with_std(np.sqrt(0.5))
    else:
        state = Space({"flux": pd.RangeIndex(small, name="cell")})
        obs = Space({"conc": grid})
        prior_cov = Covariance.diagonal(0.5, state)
    n, m = len(state), len(obs)

    def problem(H, variance):
        return Problem(
            prior=Vector(rng.normal(size=n), state),
            prior_cov=prior_cov,
            obs=Vector(np.zeros(m), obs),
            obs_cov=Covariance.diagonal(variance, obs),
            forward=Operator(H, obs, state),
        )

    H = sp.random(m, n, density=0.05, random_state=1, format="csr")
    post = fips.solve(problem(H, 0.1))
    expected = analytic.Whitened if form == "whitened" else analytic.Information
    assert isinstance(post._solution, expected)
    if form == "whitened":
        assert isinstance(post.problem.prior_cov.data, KroneckerOperator)
    world = problem(sp.random(m, n, density=0.05, random_state=2, format="csr"), 0.4)
    k = 5
    W = Operator(
        sp.csr_array(rng.uniform(size=(k, n))),
        Space({"total": pd.RangeIndex(k, name="region")}),
        state,
    )
    too_big = max(n, m) ** 2 * 8
    tracemalloc.start()
    try:
        table = post.recovery(world, W=W)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert table.shape == (k, 7)
    assert peak < too_big / 2, (
        f"peak {peak / 1e6:.1f} MB; one {max(n, m)} x {max(n, m)} is {too_big / 1e6:.1f} MB"
    )
