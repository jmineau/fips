"""
Tests for ``fips.hierarchical``: the covariance settings drawn from their posterior, and the state at each draw.

How it is held to being right:

1. **Quadrature.** One setting on a tiny problem: the posterior of the
   setting, ``p(s) p(z | s)`` on a fine grid with the evidence of a fresh
   solve at each point, gives the setting's mean, std and 5% and 95%
   quantiles, and the state's mixed mean and variance (the law of total
   variance). The chain's draws match them within four Monte Carlo standard
   errors (from the chain's effective sample size), on the general path and
   on the fast path.
2. **Two wrong variants fail the same checks**: a chain with no
   accept/reject step, and the state drawn at the settings' mode instead of
   each draw's settings.
3. **The fast path is the general path**: its evidence equals the scaled
   problem's ``log_evidence`` in both forms, and with the same seed the two
   paths give the same chain and the same draws of the state.
4. **The priors**: their densities against SciPy's, the map to unbounded
   coordinates and its Jacobian.
5. Seeds reproduce, the result pickles, labels and refusals.

A slow test: simulation-based calibration. Worlds draw the settings from
their priors, then the state, then the data; the rank of the true settings
among the draws is uniform over the worlds, and a total's 90% interval holds
the truth in 90% of them.
"""

import math
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.integrate import trapezoid

import fips
import fips.settings
from fips import HalfNormal, LogNormal, Uniform
from fips.methods import analytic
from fips.settings import _effective_size, _Scales


def _world(n=3, m=4, seed=0, std=1.5, noise=0.5):
    """A tiny problem: `n` cells, `m` observations, data from a truth of std `std`."""
    rng = np.random.default_rng(seed)
    cells = fips.Space({"flux": pd.Index([f"c{i}" for i in range(n)], name="cell")})
    observed = fips.Space({"ch4": pd.RangeIndex(m, name="obs_id")})
    H = fips.Operator(rng.uniform(0.2, 1.0, (m, n)), observed, cells)
    prior = fips.Vector(np.ones(n), cells, name="prior")
    truth = prior.values + std * rng.standard_normal(n)
    obs = fips.Vector(H.dense() @ truth + noise * rng.standard_normal(m), observed)
    return prior, obs, H


PRIOR, OBS, H = _world()
PRIOR_STD = LogNormal(0.0, 0.7)


def build_tiny(s):
    """The tiny problem at prior std ``s["std"]`` (top level, so a result pickles)."""
    return fips.Problem(
        prior=PRIOR,
        prior_cov=fips.Independent().with_std(s["std"]),
        obs=OBS,
        obs_cov=fips.Independent().with_std(0.5),
        forward=H,
    )


def _quadrature():
    """The setting's posterior on a grid of log s, and the state's mixed mean and variance."""
    u = np.linspace(-5.0, 5.0, 1201) * PRIOR_STD.sigma + PRIOR_STD.mu
    log_post = np.empty(len(u))
    means, variances = [], []
    for k, uk in enumerate(u):
        post = fips.solve(build_tiny({"std": math.exp(uk)}))
        log_post[k] = PRIOR_STD._log_density_unbounded(uk) + post.log_evidence
        means.append(post.estimate.values)
        variances.append(post.variances.values)
    weights = np.exp(log_post - log_post.max())
    weights /= trapezoid(weights, u)
    s = np.exp(u)
    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (weights[1:] + weights[:-1]) * np.diff(u))]
    )
    means, variances = np.array(means), np.array(variances)
    mean_x = trapezoid(weights[:, None] * means, u, axis=0)
    var_x = trapezoid(weights[:, None] * (variances + (means - mean_x) ** 2), u, axis=0)
    mean_s = trapezoid(weights * s, u)
    return {
        "mean": mean_s,
        "std": math.sqrt(trapezoid(weights * (s - mean_s) ** 2, u)),
        "q05": float(np.exp(np.interp(0.05, cdf, u))),
        "q95": float(np.exp(np.interp(0.95, cdf, u))),
        "mean_x": mean_x,
        "var_x": var_x,
    }


@pytest.fixture(scope="module")
def exact():
    """The quadrature, computed once."""
    return _quadrature()


def _setting_errors(draws, ess, exact):
    """Each check of the setting's draws, in Monte Carlo standard errors (ESS from the correct chain)."""
    se_mean = exact["std"] / math.sqrt(ess)
    errors = {
        "mean": (draws.mean() - exact["mean"]) / se_mean,
        # the std of a lognormal-like posterior: its SE, roughly that of a normal's, times 2 for the skew
        "std": (draws.std() - exact["std"])
        / (2.0 * exact["std"] / math.sqrt(2.0 * ess)),
    }
    for q in ("q05", "q95"):
        p = 0.05 if q == "q05" else 0.95
        errors[q] = (np.mean(draws < exact[q]) - p) / math.sqrt(p * (1 - p) / ess)
    return errors


def _state_errors(states, ess, exact):
    """The state's mixed mean and variance, each element in Monte Carlo standard errors."""
    var = exact["var_x"]
    mean_err = (states.mean(axis=1) - exact["mean_x"]) / np.sqrt(var / ess)
    var_err = (states.var(axis=1, ddof=1) - var) / (var * math.sqrt(2.0 / ess) * 1.5)
    return np.concatenate([mean_err, var_err])


@pytest.mark.parametrize("path", ["general", "fast"])
def test_the_setting_and_the_state_match_quadrature(path, exact):
    if path == "general":
        result = fips.hierarchical(build_tiny, {"std": PRIOR_STD}, draws=3000, seed=1)
        draws = result.settings["std"].values
    else:
        problem = build_tiny({"std": 1.0})
        result = fips.hierarchical(
            problem, {"prior_scale": PRIOR_STD}, draws=20000, seed=1
        )
        draws = result.settings["prior_scale"].values
    ess = float(result.ess.iloc[0])
    assert ess > 0.1 * len(draws)
    assert 0.2 < result.acceptance < 0.5
    errors = _setting_errors(draws, ess, exact)
    assert all(abs(e) < 4.0 for e in errors.values()), errors
    states = result.sample().values
    assert np.all(np.abs(_state_errors(states, ess, exact)) < 4.0)
    # the result's own summaries are those of the draws
    np.testing.assert_allclose(result.estimate.values, states.mean(axis=1))
    np.testing.assert_allclose(result.variances.values, states.var(axis=1, ddof=1))


def test_a_chain_with_no_accept_reject_step_fails_quadrature(exact, monkeypatch):
    """The wrong variant: every proposal accepted (where the posterior has a density), a random walk, not the posterior."""
    monkeypatch.setattr(
        fips.settings, "_accept", lambda log_ratio, rng: math.isfinite(log_ratio)
    )
    problem = build_tiny({"std": 1.0})
    result = fips.hierarchical(problem, {"prior_scale": PRIOR_STD}, draws=4000, seed=1)
    errors = _setting_errors(result.settings["prior_scale"].values, 1000.0, exact)
    assert max(abs(e) for e in errors.values()) > 10.0, errors


def test_the_state_drawn_at_the_mode_fails_the_law_of_total_variance(exact):
    """
    The wrong variant: the state's draws at the settings' mode leave out the settings' uncertainty.

    Its 20,000 draws are independent, so their standard errors are their own
    (about 1% of a variance); the mixed variance is about 30% larger.
    """
    problem = build_tiny({"std": 1.0})
    result = fips.hierarchical(problem, {"prior_scale": PRIOR_STD}, draws=4000, seed=1)
    at_mode = result.at(result.mode).sample(20000, rng=2).values
    errors = _state_errors(at_mode, 20000.0, exact)
    assert np.max(np.abs(errors)) > 10.0
    # the mixture's variance is the larger, in every cell
    assert np.all(result.variances.values > at_mode.var(axis=1))


# -- the fast path ------------------------------------------------------------


def _problem(n, m, seed=3):
    """A problem with a correlated prior and mismatch, in the whitened (n >= m) or the information form."""
    rng = np.random.default_rng(seed)
    state = fips.Space({"flux": pd.RangeIndex(n, name="cell")})
    observed = fips.Space({"ch4": pd.RangeIndex(m, name="obs")})
    Hm = rng.uniform(0, 1, (m, n)) * (rng.uniform(size=(m, n)) < 0.4)
    x0 = rng.normal(size=n)
    z = Hm @ x0 + 2.0 * rng.normal(size=m)
    return fips.Problem(
        fips.Vector(x0, state),
        fips.Exponential("cell", 3.0).with_std(1.0),
        fips.Vector(z, observed),
        fips.Independent().with_std(0.5) + fips.Exponential("obs", 2.0).with_std(0.2),
        fips.Operator(Hm, observed, state),
    )


@pytest.mark.parametrize(("n", "m"), [(12, 20), (20, 12)])
@pytest.mark.parametrize(
    "values",
    [
        {"prior_scale": 1.0, "obs_scale": 1.0},
        {"prior_scale": 2.3, "obs_scale": 0.4},
        {"prior_scale": 0.3, "gamma": 5.0},
        {"obs_scale": 3.0},
    ],
)
def test_the_fast_evidence_is_the_scaled_problems(n, m, values):
    problem = _problem(n, m)
    target = _Scales(problem)
    expected_form = analytic.Information if n < m else analytic.Whitened
    assert isinstance(target.factors, expected_form)
    log_evidence, _ = target.evaluate(values)
    scaled = target.posterior(values)
    assert log_evidence == pytest.approx(scaled.log_evidence, rel=1e-10)
    s = values.get("prior_scale", 1.0)
    c = values.get("obs_scale", 1.0) ** 2 / values.get("gamma", 1.0)
    np.testing.assert_allclose(
        scaled.problem.prior_cov.dense(), s * s * problem.prior_cov.dense(), rtol=1e-14
    )
    np.testing.assert_allclose(
        scaled.problem.obs_cov.dense(), c * problem.obs_cov.dense(), rtol=1e-14
    )


@pytest.mark.parametrize(("n", "m"), [(12, 20), (20, 12)])
def test_the_fast_path_gives_the_general_paths_chain_and_draws(n, m):
    """The same seed: the same settings and the same draws of the state, to the optimizer's tolerance."""
    problem = _problem(n, m)

    def build(s):
        return fips.solve(
            fips.Problem(
                problem.prior,
                problem.prior_cov * s["prior_scale"] ** 2,
                problem.obs,
                problem.obs_cov,
                problem.forward,
            ),
            gamma=s["gamma"],
        )

    priors = {"prior_scale": LogNormal(0.0, 1.0), "gamma": LogNormal(0.0, 1.0)}
    fast = fips.hierarchical(problem, priors, draws=300, burn_in=100, seed=4)
    general = fips.hierarchical(build, priors, draws=300, burn_in=100, seed=4)
    np.testing.assert_allclose(fast.mode.values, general.mode.values, rtol=1e-4)
    np.testing.assert_allclose(fast.settings.values, general.settings.values, rtol=1e-3)
    assert fast.acceptance == general.acceptance
    scale = np.sqrt(fast.variances.values)
    np.testing.assert_allclose(
        fast.sample().values / scale[:, None],
        general.sample().values / scale[:, None],
        atol=1e-3,
    )


def test_the_fast_path_refuses_other_settings():
    problem = build_tiny({"std": 1.0})
    with pytest.raises(ValueError, match="'length'"):
        fips.hierarchical(problem, {"length": LogNormal()})
    with pytest.raises(ValueError, match="give one"):
        fips.hierarchical(problem, {"obs_scale": LogNormal(), "gamma": LogNormal()})


# -- the priors ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("prior", "reference"),
    [
        (LogNormal(0.4, 0.7), stats.lognorm(0.7, scale=math.exp(0.4))),
        (HalfNormal(2.5), stats.halfnorm(scale=2.5)),
        (Uniform(1.0, 4.0), stats.uniform(1.0, 3.0)),
    ],
)
def test_the_priors_densities_and_maps(prior, reference):
    for x in reference.ppf([0.01, 0.2, 0.5, 0.8, 0.99]):
        assert prior.logpdf(float(x)) == pytest.approx(reference.logpdf(x), rel=1e-12)
        u = prior._to_unbounded(float(x))
        assert prior._from_unbounded(u) == pytest.approx(x, rel=1e-12)
        # the density of u is that of x times |dx/du|, by a central difference
        h = 1e-6
        dxdu = (prior._from_unbounded(u + h) - prior._from_unbounded(u - h)) / (2 * h)
        assert prior._log_density_unbounded(u) == pytest.approx(
            reference.logpdf(x) + math.log(dxdu), abs=1e-7
        )
    assert prior.median == pytest.approx(reference.median(), rel=1e-12)
    draws = prior.sample(20000, rng=0)
    assert stats.kstest(draws, reference.cdf).pvalue > 1e-3
    assert prior.logpdf(-1.0) == -math.inf


@pytest.mark.parametrize(
    "make",
    [
        lambda: LogNormal(0.0, 0.0),
        lambda: LogNormal(math.inf, 1.0),
        lambda: HalfNormal(-1.0),
        lambda: Uniform(2.0, 1.0),
    ],
    ids=[
        "lognormal-sigma-0",
        "lognormal-mu-inf",
        "halfnormal-negative",
        "uniform-reversed",
    ],
)
def test_a_prior_refuses_bad_parameters(make):
    with pytest.raises(ValueError):
        make()


# -- the result ---------------------------------------------------------------


@pytest.fixture(scope="module")
def result():
    """A general-path result on the tiny problem."""
    return fips.hierarchical(build_tiny, {"std": PRIOR_STD}, draws=400, burn_in=100)


def test_seeds_reproduce(result):
    again = fips.hierarchical(build_tiny, {"std": PRIOR_STD}, draws=400, burn_in=100)
    pd.testing.assert_frame_equal(result.settings, again.settings)
    pd.testing.assert_frame_equal(result.sample(), again.sample())
    other = fips.hierarchical(
        build_tiny, {"std": PRIOR_STD}, draws=400, burn_in=100, seed=1
    )
    assert not np.allclose(result.settings.values, other.settings.values)


class _WithoutSpawn:
    """A Generator as NumPy 1.24 has it: no ``spawn`` (it came in 1.25)."""

    def __init__(self, generator):
        self._generator = generator

    def __getattr__(self, name):
        if name == "spawn":
            raise AttributeError("'Generator' object has no attribute 'spawn'")
        return getattr(self._generator, name)


def test_the_chains_are_seeded_without_generator_spawn(monkeypatch):
    """The declared floor, NumPy 1.24.2, has no ``Generator.spawn``: the streams come from ``SeedSequence(seed).spawn``, so a run is the same with a Generator that lacks it."""
    before = fips.hierarchical(build_tiny, {"std": PRIOR_STD}, draws=30, burn_in=20)
    real = np.random.default_rng

    def default_rng(seed=None):
        generator = real(seed)
        return (
            generator
            if isinstance(seed, np.random.SeedSequence)
            else _WithoutSpawn(generator)
        )

    monkeypatch.setattr(np.random, "default_rng", default_rng)
    after = fips.hierarchical(build_tiny, {"std": PRIOR_STD}, draws=30, burn_in=20)
    pd.testing.assert_frame_equal(before.settings, after.settings)


def test_the_result_pickles(result):
    loaded = pickle.loads(pickle.dumps(result))
    pd.testing.assert_frame_equal(loaded.settings, result.settings)
    pd.testing.assert_frame_equal(loaded.sample(), result.sample())
    assert loaded.at(result.mode).log_evidence == pytest.approx(
        result.at(result.mode).log_evidence, rel=1e-12
    )
    fast = fips.hierarchical(
        build_tiny({"std": 1.0}), {"prior_scale": PRIOR_STD}, draws=50
    )
    pd.testing.assert_frame_equal(
        pickle.loads(pickle.dumps(fast)).sample(), fast.sample()
    )


def test_the_results_labels_and_aggregates(result):
    total = fips.aggregate.sum(
        PRIOR.space,
        by=pd.Series("all", index=PRIOR.space.select("flux"), name="region"),
    )
    assert list(result.settings.columns) == ["std"]
    assert result.settings.index.name == "draw"
    draws = result.sample(100)
    assert draws.shape == (3, 100)
    assert draws.index.equals(PRIOR.index)
    assert draws.columns[0] == 0 and draws.columns[-1] == 399
    totals = result.sample(W=total)
    np.testing.assert_allclose(
        totals.values, result.sample().values.sum(axis=0, keepdims=True)
    )
    assert result.cov(total).dense()[0, 0] == pytest.approx(totals.values.var(ddof=1))
    np.testing.assert_allclose(result.cov().dense(), np.cov(result.sample().values))
    table = result.uncertainty(total, level=0.9)
    assert list(table.columns) == ["estimate", "posterior_std", "lower", "upper"]
    assert table["lower"].iloc[0] == pytest.approx(np.quantile(totals.values, 0.05))
    assert result.estimate.space == PRIOR.space
    assert "Hierarchical(settings=['std']" in repr(result)
    with pytest.raises(ValueError, match="kept"):
        result.sample(401)
    with pytest.raises(ValueError, match="missing"):
        result.at({})
    with pytest.raises(TypeError):
        result.cov(np.ones((1, 3)))  # pyrefly: ignore[bad-argument-type]


def test_a_built_state_in_another_order_is_put_in_the_first_problems():
    """
    A function that returns the prior in another order at some settings: each draw is put in the order of the first problem built, at the priors' medians.

    The prior means are 0, 10 and 20, far apart next to their std, so a draw
    put in the wrong order would move a cell's mean by 10.
    """
    distinct = fips.Vector([0.0, 10.0, 20.0], PRIOR.space)
    reversed_prior = fips.Vector(
        distinct.values[::-1], fips.Space({"flux": PRIOR.space.select("flux")[::-1]})
    )

    rng = np.random.default_rng(5)
    truth = distinct.values + rng.standard_normal(3)
    obs = fips.Vector(H.dense() @ truth + 0.5 * rng.standard_normal(4), OBS.space)

    def build(s, prior=distinct):
        return fips.Problem(
            prior,
            fips.Independent().with_std(s["std"]),
            obs,
            fips.Independent().with_std(0.5),
            H,
        )

    def mixed(s):
        return build(s, reversed_prior if s["std"] > 1.0 else distinct)

    plain = fips.hierarchical(build, {"std": PRIOR_STD}, draws=400, burn_in=50)
    reordered = fips.hierarchical(mixed, {"std": PRIOR_STD}, draws=400, burn_in=50)
    np.testing.assert_allclose(
        reordered.settings.values, plain.settings.values, rtol=1e-4
    )
    assert (reordered.settings["std"] > 1.0).any() and (
        reordered.settings["std"] < 1.0
    ).any()
    assert reordered.sample().index.equals(PRIOR.index)
    np.testing.assert_allclose(
        reordered.estimate.values, plain.estimate.values, atol=0.5
    )


@pytest.mark.parametrize(
    ("build", "settings", "error", "match"),
    [
        (build_tiny, {}, ValueError, "empty"),
        (build_tiny, {"std": stats.lognorm(1.0)}, TypeError, "prior of setting 'std'"),
        (3.0, {"std": PRIOR_STD}, TypeError, "build is a function"),
        (lambda s: 1.0, {"std": PRIOR_STD}, TypeError, "returned float"),
        (
            lambda s: fips.solve(build_tiny(s), "bounded", lower=0.0),
            {"std": PRIOR_STD},
            TypeError,
            "BoundedOptions",
        ),
    ],
    ids=["empty", "scipy-prior", "a-number", "returns-a-float", "bounded"],
)
def test_refusals(build, settings, error, match):
    with pytest.raises(error, match=match):
        fips.hierarchical(build, settings, draws=10, burn_in=0)


@pytest.mark.parametrize(
    ("name", "value"), [("draws", 0), ("burn_in", -1), ("seed", 1.5)]
)
def test_counts_are_checked(name, value):
    with pytest.raises(ValueError, match=name):
        fips.hierarchical(build_tiny, {"std": PRIOR_STD}, **{name: value})


def test_the_effective_size_of_independent_and_correlated_chains():
    rng = np.random.default_rng(0)
    white = rng.standard_normal(20000)
    assert _effective_size(white) == pytest.approx(20000, rel=0.1)
    ar = np.empty(20000)
    ar[0] = 0.0
    for t in range(1, len(ar)):
        ar[t] = 0.9 * ar[t - 1] + rng.standard_normal()
    # an AR(1) chain of coefficient 0.9 has tau = (1 + 0.9) / (1 - 0.9) = 19
    assert _effective_size(ar) == pytest.approx(20000 / 19, rel=0.2)


# -- simulation-based calibration ---------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("path", ["fast", "general"])
def test_simulation_based_calibration(path):
    """
    Worlds drawn from the settings' priors: the true settings' ranks are uniform, and a total's 90% interval covers.

    Each world draws a prior-std factor and a mismatch factor from their
    priors, a truth from the prior at the first, and the observations with
    the error at the second; the hierarchical posterior's draws (every
    fifth, 100 of them) rank the true settings. Over the worlds the ranks are
    uniform (a chi-squared test over 10 bins), and the total's 90% interval
    holds the true total in 90% of them (within three binomial standard
    errors). The general path builds an exponential prior covariance whose
    length is a third setting.
    """
    rng = np.random.default_rng(11)
    n, m = 12, 16
    state = fips.Space({"flux": pd.Index(np.arange(n) * 1.0, name="x")})
    observed = fips.Space({"ch4": pd.RangeIndex(m, name="obs")})
    Hm = rng.uniform(0, 1, (m, n)) * (rng.uniform(size=(m, n)) < 0.5)
    forward = fips.Operator(Hm, observed, state)
    prior = fips.Vector(np.full(n, 2.0), state)
    total = fips.aggregate.sum(
        state, by=pd.Series("all", index=state.select("flux"), name="region")
    )
    priors = {
        "prior_scale": LogNormal(0.0, 0.5),
        "obs_scale": LogNormal(math.log(0.5), 0.5),
    }
    if path == "general":
        priors["length"] = LogNormal(math.log(3.0), 0.5)
    trials = 200 if path == "fast" else 100

    def problem_at(s, obs):
        return fips.Problem(
            prior,
            fips.Exponential("x", s.get("length", 3.0)).with_std(s["prior_scale"]),
            obs,
            fips.Independent().with_std(s["obs_scale"]),
            forward,
        )

    ranks, covered = [], 0
    for trial in range(trials):
        true = {name: float(p.sample(1, rng)[0]) for name, p in priors.items()}
        world = problem_at(true, fips.Vector(np.zeros(m), observed))
        truth, obs = world.simulate(rng)
        if path == "fast":
            result = fips.hierarchical(
                problem_at({"prior_scale": 1.0, "obs_scale": 1.0}, obs),
                priors,
                draws=500,
                burn_in=200,
                seed=trial,
            )
        else:
            result = fips.hierarchical(
                lambda s, obs=obs: problem_at(s, obs),
                priors,
                draws=500,
                burn_in=200,
                seed=trial,
            )
        thinned = result.settings.iloc[::5]
        ranks.append([int((thinned[name] < true[name]).sum()) for name in priors])
        interval = result.uncertainty(total, level=0.9).iloc[0]
        true_total = float((total @ truth).values[0])
        covered += interval["lower"] <= true_total <= interval["upper"]
    ranks = np.array(ranks)  # 0..100
    for k, name in enumerate(priors):
        counts = np.bincount(np.minimum(ranks[:, k] // 10, 9), minlength=10)
        p = stats.chisquare(counts).pvalue
        assert p > 1e-3, (name, counts)
    coverage = covered / trials
    assert abs(coverage - 0.9) < 3.0 * math.sqrt(0.09 / trials), coverage
