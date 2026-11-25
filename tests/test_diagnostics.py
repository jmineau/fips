"""
The checks of a sampled posterior's draws, as PyMC and Stan users read them.

:meth:`fips.Posterior.diagnostics` gives, for each element or aggregate, the
numbers of ArviZ's ``summary``: the rank-normalized split R-hat, the bulk and
tail effective draws and the Monte Carlo errors of the mean and of the
standard deviation (Vehtari et al., 2021). Wrong numbers still come out as a
table of the right shape, so:

1. **Against ArviZ.** fips computes them itself (no ArviZ at run time); on
   fixed chains of several kinds (independent, autocorrelated, one chain
   shifted, one narrower, heavy-tailed, tied values, an odd length, one
   chain) they equal ``arviz_stats``' to roundoff. ArviZ needs Python 3.12,
   so these tests are skipped without it.
2. **What they are of.** The table of a ``"gauss_newton"`` posterior is of
   its kept draws and its chains, on the state's labels or an aggregation's
   rows; one chain (``"bounded"``) has no R-hat; an analytic posterior has
   nothing to check.
3. **The warning and the log.** A path whose energy rose by more than 1000
   diverged, and a solve with any among its kept draws logs how many at
   INFO (it does not warn: they did not mark the solves that were off); an interval
   read from draws with fewer than 100 effective draws in the tails for each
   chain warns (any aggregate, or 5% of the elements), and one read from
   enough does not.
"""

import logging

import numpy as np
import pandas as pd
import pytest

import fips
from fips import _draws
from fips.methods import gauss_newton

COLUMNS = ["mean", "sd", "mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]


def chains_of_each_kind() -> dict[str, np.ndarray]:
    """Fixed chains, ``rows x chains x draws``, of the kinds a check must tell apart."""
    rng = np.random.default_rng(20261116)
    independent = rng.standard_normal((6, 4, 300))
    ar = np.empty((6, 4, 300))
    ar[:, :, 0] = rng.standard_normal((6, 4))
    for t in range(1, 300):  # AR(1), rho 0.9: about 19 draws a independent one
        ar[:, :, t] = 0.9 * ar[:, :, t - 1] + np.sqrt(1 - 0.81) * rng.standard_normal(
            (6, 4)
        )
    shifted = independent.copy()
    shifted[:, 0] += 0.5
    narrower = independent.copy()
    narrower[:, 0] *= 0.5
    return {
        "independent": independent,
        "autocorrelated": ar,
        "one chain shifted": shifted,
        "one chain narrower": narrower,
        "heavy tails": np.exp(2.0 * independent),
        "tied values": np.round(independent, 1),
        "odd length": rng.standard_normal((6, 4, 301)),
        "one chain": rng.standard_normal((6, 1, 400)),
    }


def side_by_side(chains: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """`chains` (``rows x chains x draws``) as fips keeps them: the chains side by side, and their lengths."""
    rows, M, N = chains.shape
    return chains.reshape(rows, M * N), (N,) * M


@pytest.mark.parametrize("kind", list(chains_of_each_kind()))
def test_the_checks_are_arviz_s(kind):
    """Each number of :func:`fips._draws._diagnostics` equals ``arviz_stats``' for the same chains, to roundoff."""
    array_stats = pytest.importorskip("arviz_stats.base").array_stats
    chains = chains_of_each_kind()[kind]
    draws, lengths = side_by_side(chains)
    ours = _draws._diagnostics(draws, lengths)
    theirs = {
        "ess_bulk": array_stats.ess(chains, method="bulk"),
        "ess_tail": array_stats.ess(chains, method="tail", prob=0.05),
        "mcse_mean": array_stats.mcse(chains, method="mean"),
        "mcse_sd": array_stats.mcse(chains, method="sd"),
        "r_hat": array_stats.rhat(chains),
    }
    for name, values in theirs.items():
        np.testing.assert_allclose(ours[name], values, rtol=1e-9, err_msg=name)
    np.testing.assert_allclose(ours["mean"], draws.mean(axis=1))
    np.testing.assert_allclose(ours["sd"], draws.std(axis=1, ddof=1))


def test_the_checks_tell_the_kinds_apart():
    """R-hat is under 1.01 for chains of one distribution and above it (Stan's cutoff) when one of four is shifted by half a standard deviation or half as wide; autocorrelation lowers the effective draws by about its time, ``(1 + rho) / (1 - rho)`` = 19; a row that never moves has every draw and an R-hat of 1."""
    kinds = chains_of_each_kind()
    rhat = {k: _draws._rhat(*side_by_side(c)) for k, c in kinds.items()}
    assert np.all(rhat["independent"] < 1.01)
    assert np.all(rhat["one chain shifted"] > 1.01)
    assert np.all(rhat["one chain narrower"] > 1.01)
    assert np.all(np.isnan(rhat["one chain"]))
    bulk = _draws._ess_bulk(*side_by_side(kinds["autocorrelated"]))
    assert np.median(bulk) == pytest.approx(1200 / 19, rel=0.35)
    constant = np.ones((2, 400))
    table = _draws._diagnostics(constant, (100,) * 4)
    np.testing.assert_array_equal(table["r_hat"], 1.0)
    np.testing.assert_array_equal(table["ess_bulk"], 400.0)


@pytest.mark.parametrize(
    "scale", [2.0**-30, 2.0**-66, 2.0**66]
)  # about 1e-9, 1e-20, 1e20; exact
def test_the_checks_are_the_same_in_any_units(scale):
    """The checks of draws in other units (a state in mol m-2 s-1, say) are those of the same draws at unit scale, the Monte Carlo errors scaled with them; a row that only moves by roundoff about its value still has every draw."""
    draws, lengths = side_by_side(chains_of_each_kind()["autocorrelated"])
    unit = _draws._diagnostics(draws, lengths)
    scaled = _draws._diagnostics(draws * scale, lengths)
    for name in ("ess_bulk", "ess_tail", "r_hat"):
        np.testing.assert_allclose(scaled[name], unit[name], rtol=1e-9, err_msg=name)
    for name in ("mcse_mean", "mcse_sd"):
        np.testing.assert_allclose(
            scaled[name] / scale, unit[name], rtol=1e-9, err_msg=name
        )
    jitter = scale * (1.0 + np.finfo(float).eps * (np.arange(400) % 3))
    assert np.ptp(jitter) > 0
    assert _draws._ess_of(jitter.reshape(1, 4, 100))[0] == 400.0


def test_unequal_chains_are_cut_to_the_shortest():
    """Fips splits its draws among its chains as evenly as they go, the first ones a draw more: the checks read each chain cut to the shortest, as ArviZ would read equal ones."""
    rng = np.random.default_rng(1)
    draws = rng.standard_normal((3, 10 + 10 + 9 + 9))
    lengths = (10, 10, 9, 9)
    equal = np.concatenate(
        [draws[:, 0:9], draws[:, 10:19], draws[:, 20:29], draws[:, 29:38]], axis=1
    )
    np.testing.assert_array_equal(
        _draws._rhat(draws, lengths), _draws._rhat(equal, (9,) * 4)
    )
    np.testing.assert_array_equal(
        _draws._ess_tail(draws, lengths), _draws._ess_tail(equal, (9,) * 4)
    )


# ---------------------------------------------------------------------------
# 2. What the table is of
# ---------------------------------------------------------------------------


def two_cells(noise: float) -> fips.Problem:
    """Two cells with lognormal priors (mean 1, log standard deviation 1) whose total is observed as 2 with error `noise`: a curved posterior where the error is small."""
    state = fips.Space({"flux": pd.Index(["a", "b"], name="cell")})
    obs = fips.Space({"conc": pd.RangeIndex(1, name="obs_id")})
    model = fips.LogNormalForward(fips.Operator(np.ones((1, 2)), obs, state), 1.0)
    mu, sigma = model.prior_for(1.0, np.sqrt(np.e - 1.0))
    return fips.Problem(
        prior=mu,
        prior_cov=fips.Covariance(np.diag(sigma.values**2), state),
        obs=fips.Vector([2.0], obs),
        obs_cov=fips.Covariance(np.array([[noise**2]]), obs),
        forward=model,
    )


def total(problem: fips.Problem) -> fips.Operator:
    """The sum of the state's elements, one row."""
    return fips.Operator(
        np.ones((1, 2)),
        fips.Space({"total": pd.Index(["all"], name="region")}),
        problem.prior.space,
    )


class Linear(fips.ForwardModel):
    """A linear map written as a model, ``F(x) = H x``: its posterior is the Gaussian at the mode, and the sampler's draws are independent."""

    def __init__(self, H: fips.Operator):
        super().__init__(H.rows, H.cols)
        self.H = H

    def __call__(self, x):
        return self.H @ x

    def jacobian(self, x):
        return self.H


def linear_problem(n: int = 30) -> fips.Problem:
    """`n` elements, half as many observations of pairs of them, through :class:`Linear`."""
    state = fips.Space({"x": pd.RangeIndex(n, name="i")})
    obs = fips.Space({"z": pd.RangeIndex(n // 2, name="j")})
    H = np.random.default_rng(3).normal(size=(n // 2, n))
    return fips.Problem(
        prior=fips.Vector(np.zeros(n), state),
        prior_cov=fips.Independent().with_std(1.0),
        obs=fips.Vector(np.ones(n // 2), obs),
        obs_cov=fips.Independent().with_std(0.5),
        forward=Linear(fips.Operator(H, obs, state)),
    )


def test_the_table_is_of_the_kept_draws_and_their_chains():
    """A ``"gauss_newton"`` posterior's table is :func:`fips._draws._diagnostics` of its kept draws and its chains' lengths, on the state's labels; its mean and sd are the posterior's own."""
    problem = two_cells(noise=0.5)
    post = fips.solve(problem, "gauss_newton", draws=403)
    table = post.diagnostics()
    assert list(table.columns) == COLUMNS
    pd.testing.assert_index_equal(table.index, problem.prior.index)
    solution = post._solution
    assert solution.lengths() == (101, 101, 101, 100)
    expected = _draws._diagnostics(np.asarray(solution.kept()), solution.lengths())
    for name in COLUMNS:
        np.testing.assert_array_equal(table[name].to_numpy(), expected[name])
    np.testing.assert_allclose(table["mean"].to_numpy(), post.mean.values)
    np.testing.assert_allclose(table["sd"].to_numpy() ** 2, post.variances.values)
    assert np.all(table["ess_bulk"] > 0)


def test_the_table_of_an_aggregation_is_of_its_rows():
    """With `W` the rows are the aggregates of the same draws, on `W`'s rows."""
    problem = two_cells(noise=0.5)
    post = fips.solve(problem, "gauss_newton", draws=400)
    W = total(problem)
    table = post.diagnostics(W)
    pd.testing.assert_index_equal(table.index, W.rows.index)
    summed = np.asarray(post._solution.kept()).sum(axis=0, keepdims=True)
    expected = _draws._diagnostics(summed, post._solution.lengths())
    np.testing.assert_allclose(table["r_hat"].to_numpy(), expected["r_hat"])
    np.testing.assert_allclose(table["mean"].to_numpy(), summed.mean(axis=1))


def test_one_chain_has_no_r_hat_and_an_analytic_posterior_nothing_to_check():
    """``"bounded"`` keeps one chain: its R-hat is NaN and its effective draws are there. An analytic posterior is exact and raises :class:`fips.UnavailableError`."""
    problem = linear_problem()
    problem = fips.Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=problem.obs,
        obs_cov=problem.obs_cov,
        forward=problem.forward.H,
    )
    bounded = fips.solve(problem, "bounded", lower=0.0, draws=200)
    table = bounded.diagnostics()
    assert table["r_hat"].isna().all()
    assert (table["ess_bulk"] > 0).all()
    with pytest.raises(fips.UnavailableError, match="checks of its draws"):
        fips.solve(problem).diagnostics()


# ---------------------------------------------------------------------------
# 3. The warnings
# ---------------------------------------------------------------------------


def test_a_divergent_path_among_the_kept_draws_is_logged(caplog, monkeypatch):
    """With the energy's limit set below any rise every path diverges, and the solve logs at INFO how many of its kept draws came from one, without a warning; with Stan's 1000 a mild solve has none and says nothing."""
    with caplog.at_level(logging.INFO, logger="fips.methods.gauss_newton"):
        fips.solve(two_cells(noise=0.5), "gauss_newton", draws=200).variances  # noqa: B018
    assert "diverged" not in caplog.text
    caplog.clear()
    monkeypatch.setattr(gauss_newton, "_DIVERGENT", -np.inf)
    with caplog.at_level(logging.INFO, logger="fips.methods.gauss_newton"):
        fips.solve(two_cells(noise=0.5), "gauss_newton", draws=200).variances  # noqa: B018
    (record,) = [r for r in caplog.records if "diverged" in r.getMessage()]
    assert (
        "200 of the 200 kept draws came from a path that diverged"
        in record.getMessage()
    )
    assert record.levelno == logging.INFO


def test_a_path_the_model_cannot_follow_diverged():
    """A path along which the model fails is rejected and counted as divergent."""
    post = fips.solve(two_cells(noise=0.5), "gauss_newton", draws=8)
    sampler = post._solution._sampler()
    here = sampler.point(np.asarray(post._solution.mode))
    huge = np.full_like(here.x, 1e6)
    end, alpha, change = sampler.move(here, huge, 0.5, 0.5, 3)
    assert end is here
    assert alpha == 0.0
    assert change == np.inf
    assert gauss_newton._diverged(
        np.array([change, np.nan, 1000.0, 1000.5])
    ).tolist() == [
        True,
        True,
        False,
        True,
    ]


def test_an_interval_from_too_few_tail_draws_warns(caplog):
    """Forty draws in four chains have far fewer than 100 effective draws in the tails for each chain: reading the intervals warns, for the elements and for an aggregate, naming the floor."""
    problem = two_cells(noise=0.5)
    post = fips.solve(problem, "gauss_newton", draws=40, burn_in=50)
    with caplog.at_level(logging.WARNING, logger="fips.posterior"):
        post.uncertainty()
    assert (
        "have fewer than 400 effective draws in their tails (4 chains)" in caplog.text
    )
    assert "state elements" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="fips.posterior"):
        post.uncertainty(total(problem))
    assert "1 of the 1 aggregates have fewer than 400" in caplog.text


def test_an_interval_from_enough_draws_does_not_warn(caplog):
    """A linear model's draws are independent: 2000 of them have about 2000 effective draws in the tails, and its intervals say nothing."""
    post = fips.solve(linear_problem(), "gauss_newton")
    with caplog.at_level(logging.INFO, logger="fips.posterior"):
        post.uncertainty()
    assert "effective draws in their tails" not in caplog.text
    assert (post.diagnostics()["ess_tail"] > 1000).all()


def test_a_few_elements_with_thin_tails_of_many_are_logged_not_warned(caplog):
    """Of more than 20 elements, fewer than 5% with thin tails are logged at INFO, as the chains' disagreement is."""
    rng = np.random.default_rng(5)
    draws = rng.standard_normal((40, 4000))
    draws[0] = np.repeat(
        rng.standard_normal(40), 100
    )  # one element of 40 with ~40 independent draws
    logger = logging.getLogger("fips.test")
    with caplog.at_level(logging.INFO, logger="fips.test"):
        _draws._warn_if_tails_thin(draws, (1000,) * 4, logger, aggregates=False)
    assert "Too few elements to warn of" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
