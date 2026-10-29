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


# ---------------------------------------------------------------------------
# Part 2: what each group of observations adds (removal by Woodbury)
# ---------------------------------------------------------------------------


def without(problem: Problem, rows: np.ndarray) -> Problem:
    """The problem without the observations at `rows`, built by hand from the dense matrices."""
    keep = np.setdiff1d(np.arange(len(problem.obs)), rows)
    space = Space.from_index(problem.obs.index[keep])
    S_z = problem.obs_cov.dense()[np.ix_(keep, keep)]
    return Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=Vector(problem.obs.values[keep], space),
        obs_cov=Covariance(0.5 * (S_z + S_z.T), space),
        forward=Operator(problem.forward.dense()[keep], space, problem.prior.space),
    )


def self_sensitivity(problem: Problem, gamma: float) -> np.ndarray:
    """diag(H K), dense: ``K = S^ H^T (S_z/gamma)^-1``."""
    H = problem.forward.dense()
    K = (
        closed_form(problem, gamma)
        @ H.T
        @ np.linalg.inv(problem.obs_cov.dense() / gamma)
    )
    return np.diag(H @ K)


def _group_rows(problem: Problem, level: str) -> dict:
    """The positions of each value of `level` among the observations (those that have it)."""
    values = problem.obs.index.get_level_values(level)
    return {
        value: np.flatnonzero(values == value) for value in pd.unique(values.dropna())
    }


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("analytic_form", ["whitened", "information"], indirect=True)
@pytest.mark.parametrize("gamma", [1.0, 2.5])
@pytest.mark.parametrize("mismatch", MISMATCHES)
@pytest.mark.parametrize("prior", PRIORS)
@pytest.mark.parametrize("read", ["solve", "with_gamma"])
def test_removing_a_group_matches_solving_without_it(prior, mismatch, gamma, read):
    """std_without of each total is the posterior std of the problem solved again without the group."""
    problem = problem_of(prior, mismatch)
    if read == "solve":  # the Cholesky at gamma
        post = fips.solve(problem, gamma=gamma)
    else:  # the eigendecomposition, read at gamma
        post = fips.solve(problem, gamma=0.7).with_gamma(gamma)
    W = totals(STATE)
    Wd = W.dense()
    table = post.data_worth(by="site", W=W)
    assert list(table.columns) == [
        "n_obs",
        "dofs",
        "std_without",
        "std_with",
        "reduction",
    ]
    assert list(table.index.names) == ["site", *W.rows.index.names]
    sensitivity = self_sensitivity(problem, gamma)
    S_hat = closed_form(problem, gamma)
    std_with = np.sqrt(np.diag(Wd @ S_hat @ Wd.T))
    groups = _group_rows(problem, "site")
    assert sorted(table.index.get_level_values("site").unique()) == ["A", "B", "C"]
    for site, rows in groups.items():
        rest = without(problem, rows)
        S_rest = closed_form(rest, gamma)
        expected = np.sqrt(np.diag(Wd @ S_rest @ Wd.T))
        part = table.xs(site, level="site")
        assert part.index.equals(W.rows.index)
        np.testing.assert_allclose(part["std_without"], expected, rtol=1e-9)
        np.testing.assert_allclose(part["std_with"], std_with, rtol=1e-10)
        np.testing.assert_allclose(
            part["reduction"], 1 - std_with / expected, rtol=1e-8, atol=1e-12
        )
        assert (part["n_obs"] == len(rows)).all()
        np.testing.assert_allclose(
            part["dofs"], sensitivity[rows].sum(), rtol=1e-9, atol=1e-12
        )
        # and the solve of the smaller problem agrees, in the form it chooses
        np.testing.assert_allclose(
            np.sqrt(fips.solve(rest, gamma=gamma).cov(W).variances.values),
            expected,
            rtol=1e-9,
        )


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("analytic_form", ["whitened", "information"], indirect=True)
@pytest.mark.parametrize("gamma", [1.0, 2.5])
@pytest.mark.parametrize("mismatch", MISMATCHES)
@pytest.mark.parametrize("prior", PRIORS)
@pytest.mark.parametrize("read", ["solve", "with_gamma"])
def test_the_self_sensitivity_and_dofs_shares(prior, mismatch, gamma, read):
    """diag(H K) from the factors equals the dense one; the shares of the blocks sum to DOFS."""
    problem = problem_of(prior, mismatch)
    if read == "solve":
        post = fips.solve(problem, gamma=gamma)
    else:
        post = fips.solve(problem, gamma=0.7).with_gamma(gamma)
    expected = self_sensitivity(problem, gamma)
    np.testing.assert_allclose(
        post._solution.self_sensitivity(), expected, rtol=1e-9, atol=1e-12
    )
    W = totals(STATE)
    table = post.data_worth(by="block", W=W)
    per_block = table.groupby(level="obs_block")["dofs"].first()
    assert sorted(per_block.index) == ["insitu", "trax"]
    np.testing.assert_allclose(per_block.sum(), post.dofs, rtol=1e-9)
    # the trax block alone: removing it is solving the in-situ problem
    rows = np.flatnonzero(problem.obs.index.get_level_values("block") == "trax")
    Wd = W.dense()
    S_rest = closed_form(without(problem, rows), gamma)
    np.testing.assert_allclose(
        table.xs("trax", level="obs_block")["std_without"],
        np.sqrt(np.diag(Wd @ S_rest @ Wd.T)),
        rtol=1e-9,
    )


@pytest.mark.parametrize("analytic_form", ["whitened", "information"], indirect=True)
def test_removing_every_observation_gives_the_prior(analytic_form):
    """One group of all the observations: std_without is the prior std, and its DOFS share is DOFS."""
    problem = problem_of("kronecker", "kronecker")
    post = fips.solve(problem)
    W = totals(STATE)
    table = post.data_worth(by=np.zeros(len(OBS)), W=W)
    np.testing.assert_allclose(
        table["std_without"].to_numpy(),
        post.uncertainty(W)["prior_std"].to_numpy(),
        rtol=1e-9,
    )
    np.testing.assert_allclose(table["dofs"].iloc[0], post.dofs, rtol=1e-9)
    assert table["n_obs"].iloc[0] == len(OBS)


@pytest.mark.parametrize("analytic_form", ["whitened", "information"], indirect=True)
def test_groups_of_unsorted_times_by_a_grouper(analytic_form):
    """A pd.Grouper on a time level in no order: each day's observations, as the brute force finds them."""
    rng = np.random.default_rng(3)
    times = pd.to_datetime(
        ["2024-01-02 06:00", "2024-01-01 12:00", "2024-01-03 01:00", "2024-01-01 03:00"]
        + ["2024-01-02 18:00", "2024-01-03 20:00", "2024-01-01 22:00"]
    )
    observed = Space({"co2": pd.Index(times, name="obs_time")})
    state = Space({"flux": pd.Index(range(4), name="cell")})
    m, n = len(observed), len(state)
    problem = Problem(
        prior=Vector(np.zeros(n), state),
        prior_cov=Covariance(_spd(n, rng), state),
        obs=Vector(rng.normal(size=m), observed),
        obs_cov=Covariance(_spd(m, rng) * 0.3, observed),
        forward=Operator(rng.uniform(0, 1, (m, n)), observed, state),
    )
    post = fips.solve(problem)
    W = Operator(np.eye(n), state, state)
    table = post.data_worth(by=pd.Grouper(level="obs_time", freq="D"), W=W)
    days = pd.DatetimeIndex(times).normalize()
    for day in days.unique():
        rows = np.flatnonzero(days == day)
        S_rest = closed_form(without(problem, rows), 1.0)
        np.testing.assert_allclose(
            table.xs(day, level="obs_time")["std_without"],
            np.sqrt(np.diag(S_rest)),
            rtol=1e-9,
        )
        assert (table.xs(day, level="obs_time")["n_obs"] == len(rows)).all()


def test_two_grouping_levels():
    """By a list: the groups are the pairs, each removed on its own."""
    problem = problem_of("dense", "block")
    post = fips.solve(problem)
    W = totals(STATE)
    table = post.data_worth(by=["site", "hour"], W=W)
    assert list(table.index.names)[:2] == ["site", "hour"]
    assert len(table) == len(INSITU) * len(W.rows)
    values = problem.obs.index
    rows = np.flatnonzero(
        (values.get_level_values("site") == "B")
        & (values.get_level_values("hour") == 3)
    )
    Wd = W.dense()
    S_rest = closed_form(without(problem, rows), 1.0)
    np.testing.assert_allclose(
        table.xs(("B", 3), level=["site", "hour"])["std_without"],
        np.sqrt(np.diag(Wd @ S_rest @ Wd.T)),
        rtol=1e-9,
    )


def test_data_worth_refuses_what_it_cannot_use():
    """No group, an aggregation on other labels, a method without the members."""
    post = fips.solve(problem_of("diagonal", "diagonal"))
    W = totals(STATE)
    with pytest.raises(ValueError, match="no observation in a group"):
        post.data_worth(by=np.full(len(OBS), np.nan), W=W)
    with pytest.raises(ValueError, match="the state"):
        post.data_worth(by="site", W=fips.aggregate.sum(FLUX, by="month"))

    class Bare:
        """A solution with only the required members."""

        def __getattr__(self, name):
            if name in ("innovation_precision", "self_sensitivity"):
                raise AttributeError(name)
            return getattr(post._solution, name)

    bare = fips.Posterior(post.problem, Bare(), post.options)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(TypeError, match="innovation_precision"):
        bare.data_worth(by="site", W=W)


def test_data_worth_allocates_nothing_n_by_n():
    """Per-group removal for a few totals, in the whitened form, stays far below one ``n x n`` array (n = 2,400)."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=48, freq="D")
    cells = pd.MultiIndex.from_product(
        [times, np.arange(10) * 0.1, np.arange(5) * 0.1], names=["time", "lat", "lon"]
    )
    state = Space({"flux": cells})
    n, m = len(state), 30
    observed = Space(
        {
            "conc": pd.MultiIndex.from_arrays(
                [np.repeat(["A", "B", "C"], m // 3), np.tile(np.arange(m // 3), 3)],
                names=["site", "hour"],
            )
        }
    )
    problem = Problem(
        prior=Vector(rng.normal(size=n), state),
        prior_cov=fips.Kronecker(
            fips.Exponential("time", "5D"),
            fips.Exponential(["lat", "lon"], 20.0, metric="haversine"),
        ).with_std(np.sqrt(0.5)),
        obs=Vector(rng.normal(size=m), observed),
        obs_cov=Covariance.diagonal(0.1, observed),
        forward=Operator(
            sp.random(m, n, density=0.2, random_state=1, format="csr"),
            observed,
            state,
        ),
    )
    total = fips.aggregate.sum(state, by=pd.Grouper(level="time", freq="7D"))
    n_by_n = n * n * 8
    tracemalloc.start()
    try:
        post = fips.solve(problem)
        table = post.data_worth(by="site", W=total)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(table) == 3 * len(total.rows)
    assert peak < n_by_n / 2, (
        f"peak {peak / 1e6:.1f} MB, n x n is {n_by_n / 1e6:.1f} MB"
    )


# ---------------------------------------------------------------------------
# Part 3: candidate observations, not in the problem
# ---------------------------------------------------------------------------

# Four hours at two new sites and at site A, as a block of their own
CANDIDATES = Space(
    {
        "new": pd.MultiIndex.from_product(
            [["A", "D", "E"], range(10, 14)], names=["site", "hour"]
        )
    }
)


def candidates(kind: str, seed: int = 1) -> tuple[Operator, Covariance]:
    """The candidates' forward operator and error covariance (`dense` correlates the sites)."""
    rng = np.random.default_rng(seed)
    m = len(CANDIDATES)
    H = np.zeros((m, len(STATE)))
    H[:, : len(FLUX)] = rng.uniform(0.0, 1.0, (m, len(FLUX)))
    H[:4, len(FLUX)] = 1.0  # site A's bias
    if kind == "diagonal":
        R = Covariance(DiagonalOperator(rng.uniform(0.2, 0.6, m)), CANDIDATES)
    else:
        R = Covariance(_spd(m, rng) * 0.4, CANDIDATES)
    return Operator(H, CANDIDATES, STATE), R


def with_added(problem: Problem, H_c: Operator, R_c: Covariance, rows) -> Problem:
    """The problem with the candidates at `rows` added, their errors independent of the others."""
    observed = problem.obs.space
    blocks = {name: observed.select(name) for name in observed.names}
    space = Space({**blocks, "new": H_c.rows.select("new")[rows]})
    m = len(problem.obs)
    S_z = np.zeros((len(space), len(space)))
    S_z[:m, :m] = problem.obs_cov.dense()
    S_z[m:, m:] = R_c.dense()[np.ix_(rows, rows)]
    return Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=Vector(np.concatenate([problem.obs.values, np.zeros(len(rows))]), space),
        obs_cov=Covariance(0.5 * (S_z + S_z.T), space),
        forward=Operator(
            np.vstack([problem.forward.dense(), H_c.dense()[rows]]),
            space,
            problem.prior.space,
        ),
    )


@pytest.mark.usefixtures("analytic_form")
@pytest.mark.parametrize("analytic_form", ["whitened", "information"], indirect=True)
@pytest.mark.parametrize("gamma", [1.0, 2.5])
@pytest.mark.parametrize("errors", ["diagonal", "dense"])
@pytest.mark.parametrize("mismatch", MISMATCHES)
@pytest.mark.parametrize("prior", PRIORS)
def test_adding_candidates_matches_solving_with_them(prior, mismatch, errors, gamma):
    """std_with of each total is the posterior std of the problem solved again with the group added."""
    problem = problem_of(prior, mismatch)
    post = fips.solve(problem, gamma=gamma)
    H_c, R_c = candidates(errors)
    W = totals(STATE)
    Wd = W.dense()
    table = post.candidate_worth(H_c, R_c, by="site", W=W)
    assert list(table.columns) == [
        "n_obs",
        "dofs",
        "std_without",
        "std_with",
        "reduction",
    ]
    assert sorted(table.index.get_level_values("site").unique()) == ["A", "D", "E"]
    std_without = np.sqrt(np.diag(Wd @ closed_form(problem, gamma) @ Wd.T))
    sites = CANDIDATES.select("new").get_level_values("site")
    for site in ["A", "D", "E"]:
        rows = np.flatnonzero(sites == site)
        bigger = with_added(problem, H_c, R_c, rows)
        expected = np.sqrt(np.diag(Wd @ closed_form(bigger, gamma) @ Wd.T))
        part = table.xs(site, level="site")
        assert part.index.equals(W.rows.index)
        np.testing.assert_allclose(part["std_with"], expected, rtol=1e-9)
        np.testing.assert_allclose(part["std_without"], std_without, rtol=1e-10)
        np.testing.assert_allclose(
            part["reduction"], 1 - expected / std_without, rtol=1e-8, atol=1e-12
        )
        assert (part["n_obs"] == len(rows)).all()
        # the candidates' share of DOFS in the problem with them
        share = self_sensitivity(bigger, gamma)[len(problem.obs) :].sum()
        np.testing.assert_allclose(part["dofs"], share, rtol=1e-9)


@pytest.mark.parametrize("analytic_form", ["whitened", "information"], indirect=True)
def test_candidates_after_with_gamma_and_as_a_model(analytic_form):
    """The eigendecomposition's weights, and an error given as a model built on the candidates."""
    problem = problem_of("kronecker", "dense")
    post = fips.solve(problem, gamma=0.7).with_gamma(2.0)
    H_c, _ = candidates("diagonal")
    W = totals(STATE)
    # the candidates' columns in another order are matched by label
    reordered = Space({"bias": BIAS.select("bias"), "flux": FLUX.select("flux")})
    table = post.candidate_worth(
        H_c.align(CANDIDATES, reordered),
        fips.Independent().with_std(0.5),
        by="site",
        W=W,
    )
    R_c = Covariance.diagonal(0.25, CANDIDATES)
    rows = np.flatnonzero(CANDIDATES.select("new").get_level_values("site") == "D")
    Wd = W.dense()
    expected = np.sqrt(
        np.diag(Wd @ closed_form(with_added(problem, H_c, R_c, rows), 2.0) @ Wd.T)
    )
    np.testing.assert_allclose(
        table.xs("D", level="site")["std_with"], expected, rtol=1e-9
    )


def test_a_candidate_already_observed_twice_equals_data_worth():
    """Adding a group that equals one in the problem, to the problem without it, is removing it seen from the other side."""
    problem = problem_of("dense", "diagonal")
    rows = _group_rows(problem, "site")["B"]
    rest = without(problem, rows)
    labels = problem.obs.index[rows]
    space = Space.from_index(labels)
    H_c = Operator(problem.forward.dense()[rows], space, STATE)
    R_c = Covariance(problem.obs_cov.dense()[np.ix_(rows, rows)], space)
    W = totals(STATE)
    added = fips.solve(rest).candidate_worth(H_c, R_c, by="site", W=W)
    removed = fips.solve(problem).data_worth(by="site", W=W).xs("B", level="site")
    for column in ["std_without", "std_with", "reduction", "dofs"]:
        np.testing.assert_allclose(
            added.xs("B", level="site")[column], removed[column], rtol=1e-9
        )


def test_candidate_worth_refuses_what_it_cannot_use():
    """Arrays, other labels, an error on other labels, and a method whose estimate is not linear."""
    post = fips.solve(problem_of("diagonal", "diagonal"))
    H_c, R_c = candidates("diagonal")
    W = totals(STATE)
    with pytest.raises(TypeError, match="forward must be an Operator"):
        post.candidate_worth(H_c.dense(), R_c, by="site", W=W)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(ValueError, match="the state"):
        post.candidate_worth(
            Operator(H_c.dense()[:, : len(FLUX)], CANDIDATES, FLUX), R_c, by="site", W=W
        )
    with pytest.raises(ValueError, match="the candidates"):
        post.candidate_worth(H_c, Covariance.diagonal(1.0, OBS), by="site", W=W)

    class NotLinear:
        """A solution whose estimate is not linear in the observations."""

        estimate_is_linear = False

        def __getattr__(self, name):
            return getattr(post._solution, name)

    other = fips.Posterior(post.problem, NotLinear(), post.options)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(TypeError, match="estimate_is_linear"):
        other.candidate_worth(H_c, R_c, by="site", W=W)


def test_candidate_worth_allocates_nothing_n_by_n():
    """Three candidate sites for a few totals stay far below one ``n x n`` array (n = 2,400)."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=48, freq="D")
    cells = pd.MultiIndex.from_product(
        [times, np.arange(10) * 0.1, np.arange(5) * 0.1], names=["time", "lat", "lon"]
    )
    state = Space({"flux": cells})
    n, m = len(state), 30
    observed = Space({"conc": pd.RangeIndex(m, name="obs_id")})
    problem = Problem(
        prior=Vector(rng.normal(size=n), state),
        prior_cov=fips.Kronecker(
            fips.Exponential("time", "5D"),
            fips.Exponential(["lat", "lon"], 20.0, metric="haversine"),
        ).with_std(np.sqrt(0.5)),
        obs=Vector(rng.normal(size=m), observed),
        obs_cov=Covariance.diagonal(0.1, observed),
        forward=Operator(
            sp.random(m, n, density=0.2, random_state=1, format="csr"),
            observed,
            state,
        ),
    )
    new = Space(
        {
            "conc": pd.MultiIndex.from_product(
                [["X", "Y", "Z"], range(10)], names=["site", "hour"]
            )
        }
    )
    H_c = Operator(
        sp.random(len(new), n, density=0.2, random_state=2, format="csr"), new, state
    )
    total = fips.aggregate.sum(state, by=pd.Grouper(level="time", freq="7D"))
    n_by_n = n * n * 8
    post = fips.solve(problem)
    tracemalloc.start()
    try:
        table = post.candidate_worth(
            H_c, fips.Independent().with_std(0.3), by="site", W=total
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(table) == 3 * len(total.rows)
    assert peak < n_by_n / 2, (
        f"peak {peak / 1e6:.1f} MB, n x n is {n_by_n / 1e6:.1f} MB"
    )
