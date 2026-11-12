"""
Tests for ``fips.sweep``: a grid of hyperparameters, each member solved.

How it is held to being right:

1. **Each member is its own solve.** Every row of the table, every estimate
   and every variance equals those of a fresh ``fips.solve`` of the problem
   built at that member, on the general path and on a problem's scales.
2. **On a problem, the members that differ only in gamma share one solve**:
   a grid over gamma solves once and gives ``fips.solve(problem, gamma=g)``
   at each gamma to 1e-10.
3. **The uncertainty** is the mean of the kept members' estimates, the root
   of the mean of their variances, and the population standard deviation of
   their estimates.
4. Labels, a member whose state is in another order, a method that cannot
   give a column, pickling, and refusals.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

import fips
import fips.hyperparameters


def _problem(n=12, m=20, seed=3):
    """A problem with a correlated prior and mismatch."""
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


PROBLEM = _problem()


def build(s):
    """The problem at a prior std and a correlation length (top level, so a result pickles)."""
    return fips.Problem(
        PROBLEM.prior,
        fips.Exponential("cell", s["length"]).with_std(s["std"]),
        PROBLEM.obs,
        PROBLEM.obs_cov,
        PROBLEM.forward,
    )


GRID = {"std": [0.5, 1.0, 2.0], "length": [1.0, 3.0]}


@pytest.fixture(scope="module")
def result():
    return fips.sweep(build, GRID)


def _same_as_solve(result, label, post):
    """Assert that a member's row, estimate and variances are those of `post`."""
    row = result.table.loc[label]
    for name in row.index:
        value = post._gamma if name == "gamma" else getattr(post, name)
        assert row[name] == pytest.approx(value, rel=1e-10, abs=1e-12), name
    np.testing.assert_allclose(
        result.estimates[label].to_numpy(), post.estimate.values, rtol=1e-10
    )
    np.testing.assert_allclose(
        result.variances[label].to_numpy(), post.variances.values, rtol=1e-10
    )


# -- each member is its own solve ---------------------------------------------


def test_each_member_is_the_problem_built_there_solved(result):
    assert list(result.table.index.names) == ["std", "length"]
    assert list(result.table.index) == [
        (s, length) for s in GRID["std"] for length in GRID["length"]
    ]
    assert result.table.columns.tolist() == list(fips.hyperparameters._COLUMNS)
    assert result.estimates.columns.equals(result.table.index)
    assert result.estimates.index.equals(PROBLEM.prior.space.index)
    for label in result.table.index:
        values = dict(zip(result.table.index.names, label, strict=True))
        _same_as_solve(result, label, fips.solve(build(values)))


def test_a_function_may_return_a_posterior_of_any_method():
    def at_gamma(s):
        return fips.solve(PROBLEM, gamma=s["gamma"])

    result = fips.sweep(at_gamma, {"gamma": [0.5, 2.0]})
    assert result.table.index.equals(pd.Index([0.5, 2.0], name="gamma"))
    assert result.table["gamma"].tolist() == [0.5, 2.0]
    _same_as_solve(result, 2.0, fips.solve(PROBLEM, gamma=2.0))


@pytest.mark.parametrize(
    "grid",
    [
        {"prior_scale": [0.5, 2.0], "obs_scale": [0.4, 1.0, 3.0]},
        {"prior_scale": [0.3, 1.0], "gamma": [0.2, 5.0]},
        {"obs_scale": [0.5, 2.0]},
    ],
)
def test_a_problems_scales_are_the_scaled_problem_solved(grid):
    result = fips.sweep(PROBLEM, grid)
    names = list(grid)
    for label in result.table.index:
        values = dict(zip(names, label if len(names) > 1 else (label,), strict=True))
        s = values.get("prior_scale", 1.0)
        o = values.get("obs_scale", 1.0)
        scaled = fips.Problem(
            PROBLEM.prior,
            PROBLEM.prior_cov * (s * s),
            PROBLEM.obs,
            PROBLEM.obs_cov * (o * o),
            PROBLEM.forward,
        )
        _same_as_solve(
            result, label, fips.solve(scaled, gamma=values.get("gamma", 1.0))
        )


@pytest.mark.parametrize(("n", "m"), [(12, 20), (20, 12)])
def test_a_grid_over_gamma_solves_once(n, m, monkeypatch):
    problem = _problem(n, m)
    calls = []
    real = fips.hyperparameters.solve

    def counted(*args, **kwargs):
        calls.append(kwargs.get("gamma"))
        return real(*args, **kwargs)

    monkeypatch.setattr(fips.hyperparameters, "solve", counted)
    gammas = [0.1, 1.0, 3.0, 10.0]
    result = fips.sweep(problem, {"gamma": gammas})
    assert calls == [0.1]
    monkeypatch.setattr(fips.hyperparameters, "solve", real)
    for g in gammas:
        _same_as_solve(result, g, fips.solve(problem, gamma=g))


# -- choosing members ---------------------------------------------------------


def test_the_uncertainty_of_the_kept_members(result):
    kept = result.table.index[result.table.index.get_level_values("std") >= 1.0]
    assert 1 < len(kept) < len(result.table)
    table = result.uncertainty(kept)
    estimates = np.column_stack([result.estimates[k] for k in kept])
    variances = np.column_stack([result.variances[k] for k in kept])
    assert table.columns.tolist() == ["estimate", "posterior_std", "spread"]
    assert table.index.equals(PROBLEM.prior.space.index)
    np.testing.assert_allclose(table["estimate"], estimates.mean(axis=1), rtol=1e-14)
    np.testing.assert_allclose(
        table["posterior_std"], np.sqrt(variances.mean(axis=1)), rtol=1e-14
    )
    np.testing.assert_allclose(
        table["spread"], estimates.std(axis=1, ddof=0), rtol=1e-14
    )
    every = result.uncertainty()
    np.testing.assert_allclose(
        every["estimate"], result.estimates.to_numpy().mean(axis=1), rtol=1e-14
    )


def test_one_member_has_no_spread(result):
    best = result.table["log_evidence"].idxmax()
    table = result.uncertainty([best])
    np.testing.assert_array_equal(table["spread"], 0.0)
    np.testing.assert_allclose(table["estimate"], result.estimates[best], rtol=0)


def test_at_solves_one_member_again(result):
    best = result.table["log_evidence"].idxmax()
    post = result.at(best)
    assert isinstance(post, fips.Posterior)
    assert post.log_evidence == pytest.approx(
        result.table.loc[best, "log_evidence"], rel=1e-12
    )
    gammas = fips.sweep(PROBLEM, {"gamma": [0.5, 2.0]})
    assert gammas.at(2.0).log_evidence == pytest.approx(
        fips.solve(PROBLEM, gamma=2.0).log_evidence, rel=1e-12
    )


@pytest.mark.parametrize("members", [[(9.0, 1.0)], [(0.5, 1.0), (0.5, 7.0)], []])
def test_unknown_members_are_refused(result, members):
    with pytest.raises(ValueError, match="labels of the sweep's table"):
        result.uncertainty(members)


def test_at_refuses_an_unknown_member(result):
    with pytest.raises(ValueError, match="labels of the sweep's table"):
        result.at((9.0, 1.0))


# -- labels, methods, pickling, refusals ------------------------------------------


def test_a_member_whose_state_is_in_another_order_is_aligned():
    def shuffled(s):
        problem = build({"std": 1.0, "length": s["length"]})
        if s["length"] == 1.0:
            return problem
        order = problem.prior.space.index[::-1]
        return fips.Problem(
            fips.Vector.from_series(problem.prior.to_series().reindex(order)),
            problem.prior_cov,
            problem.obs,
            problem.obs_cov,
            problem.forward,
        )

    result = fips.sweep(shuffled, {"length": [1.0, 3.0]})
    expected = fips.solve(build({"std": 1.0, "length": 3.0}))
    np.testing.assert_allclose(
        result.estimates[3.0].to_numpy(), expected.estimate.values, rtol=1e-10
    )


def test_a_column_the_method_cannot_give_is_nan():
    def bounded(s):
        return fips.solve(build(s), method="bounded", lower=0.0)

    result = fips.sweep(bounded, {"std": [1.0], "length": [3.0]})
    assert np.isnan(result.table["log_evidence"]).all()
    assert np.isfinite(result.table["reduced_chi2"]).all()


def test_the_result_pickles(result):
    back = pickle.loads(pickle.dumps(result))
    pd.testing.assert_frame_equal(back.table, result.table)
    pd.testing.assert_frame_equal(back.estimates, result.estimates)
    assert repr(back) == "Sweep(hyperparameters=['std', 'length'], members=6)"


@pytest.mark.parametrize(
    ("grid", "error", "match"),
    [
        ({}, ValueError, "empty or not a mapping"),
        ([("std", [1.0])], ValueError, "empty or not a mapping"),
        ({1: [1.0]}, TypeError, "name is a str"),
        ({"std": "abc"}, TypeError, "are a list"),
        ({"std": 1.0}, TypeError, "are a list"),
        ({"std": []}, ValueError, "has no values"),
        ({"std": [1.0, 2.0, 1.0]}, ValueError, "repeats values"),
    ],
)
def test_a_bad_grid_is_refused(grid, error, match):
    with pytest.raises(error, match=match):
        fips.sweep(build, grid)


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        ({"length": [1.0]}, "varies only the scales"),
        ({"obs_scale": [1.0], "gamma": [1.0]}, "give one"),
        ({"prior_scale": [1.0, 0.0]}, "positive numbers"),
        ({"gamma": [-1.0]}, "positive numbers"),
        ({"gamma": ["a"]}, "positive numbers"),
    ],
)
def test_a_bad_grid_on_a_problem_is_refused(grid, match):
    with pytest.raises(ValueError, match=match):
        fips.sweep(PROBLEM, grid)


def test_build_must_make_a_problem_or_a_posterior():
    with pytest.raises(TypeError, match="it returned int"):
        fips.sweep(lambda s: 3, {"std": [1.0]})
    with pytest.raises(TypeError, match="got int"):
        fips.sweep(3, {"std": [1.0]})  # pyrefly: ignore[bad-argument-type]
