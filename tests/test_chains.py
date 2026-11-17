"""
A sampled posterior's chains, public: as a DataFrame, with the sampler's record, and for ArviZ.

:meth:`fips.Posterior.chains` gives the kept draws by chain and draw,
:attr:`fips.Posterior.sample_stats` the sampler's record of the path that
made each (ArviZ's names), and :meth:`fips.Posterior.to_inference_data` both
as ArviZ's ``DataTree``. Each is held to what it is of:

1. **The chains** are the kept draws the posterior reads its mean, spread and
   intervals from, labeled ``(chain, draw)`` by the chains' lengths, or their
   aggregates; one chain for ``"bounded"``; none for an analytic posterior.
2. **The record** is of the same draws: the acceptance probability is the
   Metropolis test's of the energy error, a path diverged where the error is
   above 1000 or not finite, each chain keeps its tuned steps, and a pickled
   posterior draws the same record again from its seed.
3. **For ArviZ** (skipped without it: it needs Python 3.12): the ``posterior``
   group holds the chains cut to one length, a variable per block over the
   block's labels (renamed where two blocks share a name), the
   ``sample_stats`` group the record, and ArviZ's own ``summary`` of it equals
   :meth:`~fips.Posterior.diagnostics`.
"""

import math
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

import fips
from fips.methods import gauss_newton


def two_cells(noise: float = 0.5) -> fips.Problem:
    """Two cells with lognormal priors (mean 1, log standard deviation 1) whose total is observed as 2 with error `noise`: a curved posterior."""
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
        np.ones((1, len(problem.prior.index))),
        fips.Space({"total": pd.Index(["all"], name="region")}),
        problem.prior.space,
    )


def linear_problem(state: fips.Space) -> fips.Problem:
    """A linear problem on `state`: each element observed once, with noise."""
    n = len(state.index)
    obs = fips.Space({"z": pd.RangeIndex(n, name="j")})
    return fips.Problem(
        prior=fips.Vector(np.zeros(n), state),
        prior_cov=fips.Independent().with_std(1.0),
        obs=fips.Vector(np.full(n, -1.0), obs),
        obs_cov=fips.Independent().with_std(0.5),
        forward=fips.Operator(np.eye(n), obs, state),
    )


# ---------------------------------------------------------------------------
# 1. The chains
# ---------------------------------------------------------------------------


def test_the_chains_are_the_kept_draws_labeled_by_chain_and_draw():
    """403 draws in four chains are 101, 101, 101 and 100: the columns count each chain's draws from 0, the rows are the state's labels, and the values are the kept draws the mean is read from."""
    problem = two_cells()
    post = fips.solve(problem, "gauss_newton", draws=403)
    chains = post.chains()
    pd.testing.assert_index_equal(chains.index, problem.prior.index)
    assert chains.columns.names == ["chain", "draw"]
    assert chains.columns.get_level_values(
        "chain"
    ).value_counts().sort_index().tolist() == [
        101,
        101,
        101,
        100,
    ]
    assert chains.columns[101] == (1, 0)
    np.testing.assert_array_equal(chains.to_numpy(), post._solution.kept())
    np.testing.assert_allclose(chains.mean(axis=1).to_numpy(), post.mean.values)


def test_the_chains_of_an_aggregation_are_its_rows():
    """With `W`, the rows are the aggregates of the same draws."""
    problem = two_cells()
    post = fips.solve(problem, "gauss_newton", draws=200)
    W = total(problem)
    chains = post.chains(W)
    pd.testing.assert_index_equal(chains.index, W.rows.index)
    np.testing.assert_allclose(
        chains.to_numpy()[0], post.chains().to_numpy().sum(axis=0)
    )


def test_bounded_has_one_chain_and_an_analytic_posterior_none():
    """``"bounded"`` keeps one chain, and records nothing of its exact paths; an analytic posterior has no chains."""
    state = fips.Space({"x": pd.RangeIndex(3, name="i")})
    problem = linear_problem(state)
    bounded = fips.solve(problem, "bounded", lower=0.0, draws=50)
    chains = bounded.chains()
    assert set(chains.columns.get_level_values("chain")) == {0}
    assert chains.shape == (3, 50)
    with pytest.raises(fips.UnavailableError, match="record of its draws"):
        bounded.sample_stats  # noqa: B018
    analytic = fips.solve(problem)
    with pytest.raises(fips.UnavailableError, match="chains of draws"):
        analytic.chains()
    with pytest.raises(fips.UnavailableError):
        analytic.sample_stats  # noqa: B018


# ---------------------------------------------------------------------------
# 2. The record
# ---------------------------------------------------------------------------


def test_the_record_is_of_each_kept_draws_path():
    """One row per kept draw, labeled as the chains' columns; the acceptance probability is ``min(1, exp(-energy_error))``, a path diverged exactly where its error is above 1000 or not finite, each chain takes one number of steps, and its step sizes are its tuned step, jittered."""
    post = fips.solve(two_cells(noise=0.2), "gauss_newton", draws=403)
    stats = post.sample_stats
    assert list(stats.columns) == list(gauss_newton._STATS)
    pd.testing.assert_index_equal(stats.index, post.chains().columns)
    error = stats["energy_error"].to_numpy()
    expected = np.where(error <= 0, 1.0, np.exp(-np.clip(error, 0, None)))
    np.testing.assert_allclose(stats["acceptance_rate"].to_numpy(), expected)
    np.testing.assert_array_equal(
        stats["diverging"].to_numpy(), ~(error <= gauss_newton._DIVERGENT)
    )
    assert stats["diverging"].dtype == bool
    tunings = post._solution._tunings()
    for k, tuning in enumerate(tunings):
        chain = stats.xs(k, level="chain")
        steps = gauss_newton._steps(tuning.step)
        assert (chain["n_steps"] == steps).all()
        size = chain["step_size"] * steps / gauss_newton._QUARTER
        assert size.between(1 - gauss_newton._JITTER, 1 + gauss_newton._JITTER).all()
    assert 0.5 < stats["acceptance_rate"].mean() <= 1.0


def test_divergent_paths_are_in_the_record(monkeypatch):
    """With the energy's limit set below any rise, every path diverged, and the record says so."""
    monkeypatch.setattr(gauss_newton, "_DIVERGENT", -np.inf)
    post = fips.solve(two_cells(), "gauss_newton", draws=40)
    assert post.sample_stats["diverging"].all()


def test_a_pickled_posterior_draws_the_same_record_again():
    """The record is kept with the draws, outside the pickle, and drawn again, the same, from the seed."""
    post = fips.solve(two_cells(), "gauss_newton", draws=100)
    stats = post.sample_stats
    assert "_stats" not in post._solution.__getstate__()
    again = pickle.loads(pickle.dumps(post))
    pd.testing.assert_frame_equal(again.sample_stats, stats)
    pd.testing.assert_frame_equal(again.chains(), post.chains())


# ---------------------------------------------------------------------------
# 3. For ArviZ
# ---------------------------------------------------------------------------


def test_the_inference_data_holds_the_chains_and_the_record():
    """The ``posterior`` group has the block over its labels, ``(chain, draw, cell)``, each chain cut to the shortest (100 of 403); ``sample_stats`` has the record, cut the same way."""
    pytest.importorskip("arviz_base")
    problem = two_cells()
    post = fips.solve(problem, "gauss_newton", draws=403)
    data = post.to_inference_data()
    flux = data.posterior["flux"]
    assert flux.dims == ("chain", "draw", "cell")
    assert flux.shape == (4, 100, 2)
    chains = post.chains()
    for k in range(4):
        np.testing.assert_array_equal(
            flux.sel(chain=k).values.T,
            chains.xs(k, axis=1, level="chain").to_numpy()[:, :100],
        )
    np.testing.assert_array_equal(
        flux.sel(cell="b").values[1], chains.loc[("flux", "b"), 1].to_numpy()[:100]
    )
    stats = post.sample_stats
    assert set(data.sample_stats.data_vars) == set(gauss_newton._STATS)
    np.testing.assert_array_equal(
        data.sample_stats["energy_error"].values[3],
        stats.xs(3, level="chain")["energy_error"].to_numpy(),
    )
    assert data.sample_stats["diverging"].dtype == bool


def test_arvizs_summary_of_it_is_the_diagnostics():
    """ArviZ's own ``summary`` of the ``DataTree`` gives the numbers :meth:`~fips.Posterior.diagnostics` computes, for the elements and for an aggregate, with its ``stats.ci_prob`` at 0.95: ArviZ 1 reads the tail's effective draws at the quantiles ``1 - ci_prob`` and ``ci_prob`` (11% and 89% at its default), fips at 5% and 95%, as Vehtari et al. (2021) and Stan do."""
    arviz_stats = pytest.importorskip("arviz_stats")
    arviz_base = pytest.importorskip("arviz_base")
    problem = two_cells()
    post = fips.solve(problem, "gauss_newton", draws=400)
    for W in (None, total(problem)):
        with arviz_base.rc_context({"stats.ci_prob": 0.95}):
            theirs = arviz_stats.summary(
                post.to_inference_data(W), kind="diagnostics", round_to="none"
            )
        ours = post.diagnostics(W)
        for name in ("ess_bulk", "ess_tail", "r_hat", "mcse_mean", "mcse_sd"):
            np.testing.assert_allclose(
                theirs[name].to_numpy(dtype=float), ours[name].to_numpy(), rtol=1e-9
            )


def test_blocks_are_named_by_their_labels_and_renamed_where_they_share_them():
    """A block with one level of labels is over that level's name; one with several over ``<block>_dim`` with a coordinate per level; two blocks labeled ``site`` become ``flux_site`` and ``bias_site``. Each element is where its labels say."""
    pytest.importorskip("arviz_base")
    cells = pd.MultiIndex.from_product(
        [["2020", "2021"], ["a", "b"]], names=["year", "cell"]
    )
    state = fips.Space(
        {
            "flux": cells,
            "bias": pd.Index(["x", "y"], name="site"),
            "scale": pd.Index(["p", "q", "r"], name="site"),
        }
    )
    post = fips.solve(linear_problem(state), "bounded", lower=0.0, draws=20)
    data = post.to_inference_data()
    assert data.posterior["flux"].dims == ("chain", "draw", "flux_dim")
    assert data.posterior["bias"].dims == ("chain", "draw", "bias_site")
    assert data.posterior["scale"].dims == ("chain", "draw", "scale_site")
    assert "sample_stats" not in data.children
    chains, space = post.chains().to_numpy(), state
    picked = data.posterior["flux"].sel(year="2021", cell="a").values[0]
    np.testing.assert_array_equal(picked, chains[space.slice("flux")][2])
    picked = data.posterior["scale"].sel(scale_site="q").values[0]
    np.testing.assert_array_equal(picked, chains[space.slice("scale")][1])


def test_without_arviz_it_says_how_to_install_it(monkeypatch):
    """With ArviZ missing, the error names the extra; the chains need no ArviZ."""
    monkeypatch.setitem(sys.modules, "arviz_base", None)
    post = fips.solve(two_cells(), "gauss_newton", draws=20)
    with pytest.raises(ImportError, match=r"fips\[arviz\]"):
        post.to_inference_data()
    assert post.chains().shape == (2, 20)
    assert math.isfinite(post.sample_stats["step_size"].iloc[0])
