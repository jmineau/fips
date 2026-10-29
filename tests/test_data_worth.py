"""
Tests for the uncertainty of linear quantities of the state (#26).

Every number is held to a brute-force calculation on small problems: the dense
posterior covariance from the closed form, in both analytic forms (the
``analytic_form`` fixture), at a gamma of 1 and of 2.5, for priors and
model-data mismatches of each form (diagonal, dense, block diagonal, Kronecker).
"""

import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import fips
from fips import Covariance, Operator, Problem, Space, Vector
from fips.linalg import DiagonalOperator, KroneckerOperator

# ---------------------------------------------------------------------------
# Small problems with every form of covariance
# ---------------------------------------------------------------------------

MONTHS = pd.Index([1, 2, 3], name="month")
CELLS = pd.Index([0, 1, 2, 3], name="cell")
SITES = pd.Index(["A", "B", "C"], name="site")
HOURS = pd.Index(range(5), name="hour")

# The state: three months of four cells (month slowest, as a Kronecker form
# lays out its rows), and a bias for sites A and B
FLUX = Space({"flux": pd.MultiIndex.from_product([MONTHS, CELLS])})
BIAS = Space({"bias": pd.Index(["A", "B"], name="site")})
STATE = Space({"flux": FLUX.select("flux"), "bias": BIAS.select("bias")})
# The observations: five hours at three sites (site slowest), and six from a train
INSITU = Space({"insitu": pd.MultiIndex.from_product([SITES, HOURS])})
TRAX = Space({"trax": pd.Index(range(6), name="trip")})
OBS = Space({"insitu": INSITU.select("insitu"), "trax": TRAX.select("trax")})

PRIORS = ["diagonal", "dense", "kronecker"]
MISMATCHES = ["diagonal", "dense", "block", "kronecker"]


def _correlation(size: int, scale: float) -> np.ndarray:
    """An exponential correlation along a line of `size` points."""
    d = np.abs(np.subtract.outer(np.arange(size), np.arange(size)))
    return np.exp(-d / scale)


def _spd(size: int, rng: np.random.Generator) -> np.ndarray:
    """A random dense covariance, well conditioned."""
    A = rng.normal(size=(size, size))
    return A @ A.T / size + np.diag(rng.uniform(0.5, 1.5, size))


def prior_cov(kind: str, rng: np.random.Generator) -> Covariance:
    """The prior error covariance, of one form."""
    n = len(STATE)
    if kind == "diagonal":
        return Covariance(DiagonalOperator(rng.uniform(0.3, 1.0, n)), STATE)
    if kind == "dense":
        return Covariance(_spd(n, rng), STATE)
    if kind == "kronecker":
        flux = KroneckerOperator(
            [_correlation(len(MONTHS), 1.5), _correlation(len(CELLS), 2.0)],
            std=rng.uniform(0.5, 1.0, len(FLUX)),
        )
        return Covariance.block_diag(
            Covariance(flux, FLUX), Covariance.diagonal(0.09, BIAS)
        )
    raise ValueError(kind)


def obs_cov(kind: str, rng: np.random.Generator) -> Covariance:
    """The model-data mismatch, of one form: `dense` and `kronecker` correlate the sites."""
    m = len(OBS)
    if kind == "diagonal":
        return Covariance(DiagonalOperator(rng.uniform(0.2, 0.6, m)), OBS)
    if kind == "dense":
        return Covariance(_spd(m, rng) * 0.4, OBS)
    if kind == "block":  # each site's hours correlated, the train independent
        hours = _correlation(len(HOURS), 2.0) * 0.3
        insitu = sp.block_diag([sp.csr_array(hours)] * len(SITES), format="csr")
        return Covariance.block_diag(
            Covariance(insitu, INSITU), Covariance.diagonal(0.2, TRAX)
        )
    if kind == "kronecker":  # the sites correlated with one another, and the hours
        site = np.array([[1.0, 0.4, 0.1], [0.4, 1.0, 0.3], [0.1, 0.3, 1.0]])
        insitu = KroneckerOperator(
            [site, _correlation(len(HOURS), 1.5)],
            std=rng.uniform(0.4, 0.7, len(INSITU)),
        )
        return Covariance.block_diag(
            Covariance(insitu, INSITU), Covariance.diagonal(0.25, TRAX)
        )
    raise ValueError(kind)


def forward(rng: np.random.Generator) -> Operator:
    """The forward operator: footprints on the cells, and each site's bias."""
    H = np.zeros((len(OBS), len(STATE)))
    H[:, : len(FLUX)] = rng.uniform(0.0, 1.0, (len(OBS), len(FLUX)))
    sites = INSITU.select("insitu").get_level_values("site")
    for column, site in enumerate(["A", "B"]):
        H[: len(INSITU), len(FLUX) + column] = sites == site
    return Operator(H, OBS, STATE)


def problem_of(prior: str, mismatch: str, seed: int = 0) -> Problem:
    """A problem with a prior error covariance and a mismatch of the given forms."""
    rng = np.random.default_rng(seed)
    return Problem(
        prior=Vector(rng.uniform(0.5, 1.5, len(STATE)), STATE),
        prior_cov=prior_cov(prior, rng),
        obs=Vector(rng.normal(3.0, 1.0, len(OBS)), OBS),
        obs_cov=obs_cov(mismatch, rng),
        forward=forward(rng),
    )


def closed_form(problem: Problem, gamma: float) -> np.ndarray:
    """The posterior covariance, dense: ``S_0 - S_0 H^T (H S_0 H^T + S_z/gamma)^-1 H S_0``."""
    S0 = problem.prior_cov.dense()
    H = problem.forward.dense()
    G = H @ S0 @ H.T + problem.obs_cov.dense() / gamma
    S0Ht = S0 @ H.T
    return S0 - S0Ht @ np.linalg.solve(G, S0Ht.T)


def totals(space: Space) -> Operator:
    """The flux total of each month, and of cells 0 and 1 over the three months (the bias left out)."""
    monthly = fips.aggregate.sum(space, by="month", blocks="flux", dropna=True)
    region = pd.Series("west", index=pd.Index([0, 1], name="cell"), name="region")
    west = fips.aggregate.sum(space, by=region, blocks="flux", dropna=True)
    W = np.vstack([monthly.align(monthly.rows, space).dense(), west.dense()])
    rows = Space(
        {
            "monthly": monthly.rows.select("flux"),
            "west": west.rows.select("flux"),
        }
    )
    return Operator(sp.csr_array(W), rows, space)


def _scale(S: np.ndarray) -> float:
    """The size of a covariance's entries, for a relative tolerance."""
    return float(np.abs(S).max())


# ---------------------------------------------------------------------------
# Part 1: prior and posterior uncertainty of aggregates
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("analytic_form", ["whitened", "information"], indirect=True)
@pytest.mark.parametrize("gamma", [1.0, 2.5])
@pytest.mark.parametrize("mismatch", MISMATCHES)
@pytest.mark.parametrize("prior", PRIORS)
def test_uncertainty_matches_the_closed_form(prior, mismatch, gamma):
    """prior_std, posterior_std and reduction of each total and each element, from the dense closed form."""
    problem = problem_of(prior, mismatch)
    post = fips.solve(problem, gamma=gamma)
    S_hat = closed_form(problem, gamma)
    S0 = problem.prior_cov.dense()

    W = totals(STATE)
    Wd = W.dense()
    table = post.uncertainty(W)
    assert table.index.equals(W.rows.index)
    assert list(table.columns) == ["prior_std", "posterior_std", "reduction"]
    prior_std = np.sqrt(np.diag(Wd @ S0 @ Wd.T))
    posterior_std = np.sqrt(np.diag(Wd @ S_hat @ Wd.T))
    np.testing.assert_allclose(table["prior_std"], prior_std, rtol=1e-10)
    np.testing.assert_allclose(table["posterior_std"], posterior_std, rtol=1e-10)
    np.testing.assert_allclose(
        table["reduction"], 1 - posterior_std / prior_std, rtol=1e-9
    )

    each = post.uncertainty()
    assert each.index.equals(STATE.index)
    np.testing.assert_allclose(each["prior_std"], np.sqrt(np.diag(S0)), rtol=1e-10)
    np.testing.assert_allclose(
        each["posterior_std"], np.sqrt(np.diag(S_hat)), rtol=1e-10
    )


@pytest.mark.parametrize("prior", PRIORS)
def test_the_prior_covariance_of_aggregates(prior):
    """Covariance.aggregate is W S W^T, on the aggregation's rows, its columns matched by label."""
    problem = problem_of(prior, "diagonal")
    W = totals(STATE)
    expected = W.dense() @ problem.prior_cov.dense() @ W.dense().T
    got = problem.prior_cov.aggregate(W)
    assert isinstance(got, Covariance)
    assert got.space == W.rows
    assert got.name == problem.prior_cov.name
    np.testing.assert_allclose(got.dense(), expected, rtol=1e-12, atol=1e-14)
    # the columns of W in another order give the same covariance
    reordered = W.align(
        W.rows, Space({"bias": BIAS.select("bias"), "flux": FLUX.select("flux")})
    )
    np.testing.assert_allclose(
        problem.prior_cov.aggregate(reordered).dense(), expected, rtol=1e-12
    )


def test_aggregate_refuses_other_labels_and_arrays():
    """W must be an Operator on exactly the covariance's labels."""
    S = Covariance.diagonal(1.0, STATE)
    with pytest.raises(TypeError, match="An aggregation is an Operator"):
        S.aggregate(np.ones((1, len(STATE))))  # pyrefly: ignore[bad-argument-type]
    W = fips.aggregate.sum(FLUX, by="month")
    with pytest.raises(ValueError, match="the covariance"):
        S.aggregate(W)


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("analytic_form", ["whitened", "information"], indirect=True)
@pytest.mark.parametrize("prior", PRIORS)
def test_a_combination_of_totals_uses_their_covariance(prior):
    """The change between two months: its variance is c^T (W S W^T) c, covariance included."""
    problem = problem_of(prior, "dense")
    post = fips.solve(problem)
    monthly = fips.aggregate.sum(STATE, by="month", blocks="flux", dropna=True)
    change = Operator(
        np.array([[-1.0, 1.0, 0.0]]),
        Space({"change": pd.Index(["2 - 1"], name="months")}),
        monthly.rows,
    )
    table = post.uncertainty(change @ monthly)
    c = np.array([-1.0, 1.0, 0.0])
    covariance = post.cov(monthly).dense()
    np.testing.assert_allclose(
        table["posterior_std"].iloc[0], np.sqrt(c @ covariance @ c), rtol=1e-10
    )
    Wd = monthly.align(monthly.rows, STATE).dense()
    cW = c @ Wd
    S_hat = closed_form(problem, 1.0)
    np.testing.assert_allclose(
        table["posterior_std"].iloc[0], np.sqrt(cW @ S_hat @ cW), rtol=1e-10
    )
    # and not the root of the sum of the variances, which ignores the covariance
    without = np.sqrt(np.diag(covariance)[:2].sum())
    assert abs(table["posterior_std"].iloc[0] - without) > 1e-3


def test_a_row_of_zeros_has_no_reduction():
    """An aggregate of nothing has prior std 0, and reduction NaN, not a division warning."""
    problem = problem_of("diagonal", "diagonal")
    post = fips.solve(problem)
    W = Operator(
        np.vstack([np.zeros(len(STATE)), np.ones(len(STATE))]),
        Space({"agg": pd.Index(["none", "all"], name="what")}),
        STATE,
    )
    table = post.uncertainty(W)
    assert table["prior_std"].iloc[0] == 0.0
    assert np.isnan(table["reduction"].iloc[0])
    assert np.isfinite(table["reduction"].iloc[1])


def test_uncertainty_refuses_an_aggregation_on_other_labels():
    """W's columns must be the state's labels."""
    post = fips.solve(problem_of("diagonal", "diagonal"))
    with pytest.raises(ValueError, match="the state"):
        post.uncertainty(fips.aggregate.sum(FLUX, by="month"))


def test_problem_aggregate_is_unchanged():
    """Problem.aggregate's mismatch is still W S_z W^T (now through Covariance.aggregate)."""
    problem = problem_of("diagonal", "block")
    W = fips.aggregate.mean(OBS, by="site", blocks="insitu")
    agg = problem.aggregate(W)
    Wd = W.align(W.rows, OBS).dense()
    np.testing.assert_allclose(
        agg.obs_cov.dense(), Wd @ problem.obs_cov.dense() @ Wd.T, rtol=1e-12
    )
    assert agg.obs_cov.name == problem.obs_cov.name


@pytest.mark.parametrize("prior", ["Kronecker", "diagonal"])
def test_uncertainty_allocates_nothing_n_by_n(prior):
    """Solving and reading the uncertainty of a few totals stays far below one ``n x n`` array."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=48, freq="D")
    cells = pd.MultiIndex.from_product(
        [times, np.arange(10) * 0.1, np.arange(5) * 0.1], names=["time", "lat", "lon"]
    )
    state = Space({"flux": cells})
    n, m = len(state), 30
    observed = Space({"conc": pd.RangeIndex(m, name="obs_id")})
    if prior == "Kronecker":
        S_0 = fips.Kronecker(
            fips.Exponential("time", "5D"),
            fips.Exponential(["lat", "lon"], 20.0, metric="haversine"),
        ).with_std(np.sqrt(0.5))
    else:
        S_0 = Covariance(DiagonalOperator(rng.uniform(0.5, 1.0, n)), state)
    problem = Problem(
        prior=Vector(rng.normal(size=n), state),
        prior_cov=S_0,
        obs=Vector(rng.normal(size=m), observed),
        obs_cov=Covariance.diagonal(0.1, observed),
        forward=Operator(
            sp.random(m, n, density=0.05, random_state=1, format="csr"),
            observed,
            state,
        ),
    )
    weekly = fips.aggregate.sum(state, by=pd.Grouper(level="time", freq="7D"))
    n_by_n = n * n * 8
    tracemalloc.start()
    try:
        post = fips.solve(problem)
        table = post.uncertainty(weekly)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(table) == 7
    assert peak < n_by_n / 2, (
        f"peak {peak / 1e6:.1f} MB, n x n is {n_by_n / 1e6:.1f} MB"
    )
