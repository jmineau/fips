"""Tests for fips.solve and the Posterior it returns."""

import pickle

import numpy as np
import pandas as pd
import pytest

import fips
from fips import (
    Covariance,
    Estimator,
    InverseProblem,
    Operator,
    Posterior,
    Space,
    Vector,
)
from fips.estimators import BayesianSolver
from fips.problems.flux import FluxProblem


def _spd(rng, n):
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _problem(cls=InverseProblem, seed=0):
    """A two-block state (flux, bias), one obs block, and a background."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2023-01", periods=3, freq="MS")
    flux_idx = pd.MultiIndex.from_product([times, ["A", "B"]], names=["time", "loc"])
    bias_idx = pd.Index(["S1", "S2"], name="site")
    obs_idx = pd.MultiIndex.from_tuples(
        [
            (t, s)
            for t in pd.date_range("2023-01-01", periods=4, freq="20D")
            for s in ["S1", "S2"]
        ],
        names=["time", "site"],
    )
    prior = Vector.from_blocks(
        {
            "flux": pd.Series(rng.normal(size=6), index=flux_idx),
            "bias": pd.Series([0.1, -0.1], index=bias_idx),
        }
    )
    obs = Vector.from_series(
        pd.Series(400 + rng.normal(size=8), index=obs_idx, name="concentration"),
        block="concentration",
    )
    H = Operator.from_blocks(
        {
            ("concentration", "flux"): pd.DataFrame(
                rng.random((8, 6)), index=obs_idx, columns=flux_idx
            ),
            ("concentration", "bias"): pd.DataFrame(
                np.tile(np.eye(2), (4, 1)), index=obs_idx, columns=bias_idx
            ),
        }
    )
    S_0 = Covariance.from_blocks(
        {
            ("flux", "flux"): pd.DataFrame(
                _spd(rng, 6), index=flux_idx, columns=flux_idx
            ),
            ("bias", "bias"): Covariance.diagonal(0.01, Space({"bias": bias_idx})),
        }
    )
    S_z = Covariance.from_frame(
        pd.DataFrame(_spd(rng, 8), index=obs_idx, columns=obs_idx),
        row_block="concentration",
        col_block="concentration",
    )
    background = Vector.from_series(
        pd.Series(np.full(8, 399.0), index=obs_idx, name="concentration"),
        block="concentration",
    )
    return cls(
        obs=obs,
        prior=prior,
        forward_operator=H,
        prior_error=S_0,
        modeldata_mismatch=S_z,
        constant=background,
    )


@pytest.fixture
def problem():
    """An unsolved two-block problem."""
    return _problem()


@pytest.fixture
def post(problem):
    """The posterior of ``problem``."""
    return fips.solve(problem)


# (Posterior attribute, InverseProblem property, estimator attribute, row
# space, column space or None, type, name). The labels are what
# InverseProblem built for each property before Posterior existed.
OUTPUTS = [
    ("estimate", "posterior", "x_hat", "state", None, Vector, "posterior"),
    (
        "covariance",
        "posterior_error",
        "S_hat",
        "state",
        "state",
        Covariance,
        "posterior_error",
    ),
    ("modeled_obs", "posterior_obs", "y_hat", "obs", None, Vector, "modeled_obs"),
    ("prior_modeled_obs", "prior_obs", "y_0", "obs", None, Vector, "prior_obs"),
    (
        "prior_obs_covariance",
        "prior_obs_error",
        "S_y0",
        "obs",
        "obs",
        Covariance,
        "prior_obs_error",
    ),
    ("kalman_gain", "kalman_gain", "K", "state", "obs", Operator, "kalman_gain"),
    (
        "averaging_kernel",
        "averaging_kernel",
        "A",
        "state",
        "state",
        Operator,
        "averaging_kernel",
    ),
    (
        "desroziers",
        "desroziers",
        "desroziers",
        "obs",
        "obs",
        Covariance,
        "desroziers",
    ),
]
OUTPUT_IDS = [o[0] for o in OUTPUTS]

SCALARS = [
    ("dofs", "DOFS"),
    ("reduced_chi2", "reduced_chi2"),
    ("r2", "R2"),
    ("rmse", "RMSE"),
    ("uncertainty_reduction", "uncertainty_reduction"),
]


def _expected(problem, est, est_attr, row, col, cls, name):
    """Label an estimator array the way InverseProblem labeled it."""
    indexes = {"state": problem.state_index, "obs": problem.obs_index}
    raw = getattr(est, est_attr)
    if col is None:
        return Vector.from_series(pd.Series(raw, index=indexes[row], name=name))
    spaces = {name: Space.from_index(index) for name, index in indexes.items()}
    if cls is Covariance:
        return Covariance(np.asarray(raw), spaces[row], name=name)
    return cls(np.asarray(raw), spaces[row], spaces[col], name=name)


def _assert_same(result, expected):
    assert type(result) is type(expected)
    assert result.name == expected.name
    if isinstance(expected, Vector):
        pd.testing.assert_series_equal(
            result.to_series(), expected.to_series(), check_exact=True
        )
    else:
        assert result.rows.index.equals(expected.rows.index)
        assert result.cols.index.equals(expected.cols.index)
        np.testing.assert_array_equal(result.dense(), expected.dense())


def _output(post, attr):
    """Return an output of `post`; ``averaging_kernel`` is a method, called with no aggregation."""
    value = getattr(post, attr)
    return value() if callable(value) else value


class TestOutputs:
    """Each output has the values, labels, type and name the problem gave."""

    @pytest.mark.parametrize(
        ("attr", "prop", "est_attr", "row", "col", "cls", "name"),
        OUTPUTS,
        ids=OUTPUT_IDS,
    )
    def test_output_is_labeled_as_before(
        self, problem, post, attr, prop, est_attr, row, col, cls, name
    ):
        expected = _expected(problem, post.estimator, est_attr, row, col, cls, name)
        _assert_same(_output(post, attr), expected)

    @pytest.mark.parametrize(
        ("attr", "prop", "est_attr", "row", "col", "cls", "name"),
        OUTPUTS,
        ids=OUTPUT_IDS,
    )
    def test_problem_property_forwards_to_the_posterior(
        self, problem, attr, prop, est_attr, row, col, cls, name
    ):
        problem.solve("bayesian")
        result = getattr(problem, prop)
        assert result is _output(problem.result, attr)
        expected = _expected(problem, problem.estimator, est_attr, row, col, cls, name)
        _assert_same(result, expected)

    def test_labels_are_the_problem_indexes(self, problem, post):
        assert post.estimate.index.equals(problem.state_index)
        assert post.modeled_obs.index.equals(problem.obs_index)
        assert post.kalman_gain.rows.index.equals(problem.state_index)
        assert post.kalman_gain.cols.index.equals(problem.obs_index)
        assert post.state_space == Space.from_index(problem.state_index)
        assert post.obs_space == Space.from_index(problem.obs_index)

    def test_state_index_with_a_block_apart_raises(self, problem, post):
        """A state index with a block's rows apart has no state space to label the arrays."""
        index = problem.state_index
        assert index.get_level_values("block")[[5, 6]].tolist() == ["flux", "bias"]
        apart = index[[0, 1, 2, 6, 3, 4, 5, 7]]  # flux, bias, flux, bias
        broken = Posterior(post.estimator, apart, problem.obs_index)
        with pytest.raises(ValueError, match="has the rows of a block apart"):
            _ = broken.state_space
        with pytest.raises(ValueError, match="has the rows of a block apart"):
            _ = broken.covariance
        with pytest.raises(ValueError, match="has the rows of a block apart"):
            _ = broken.kalman_gain

    def test_mean_is_the_estimate(self, post):
        assert post.mean is post.estimate

    def test_variances_are_the_diagonal_of_the_covariance(self, problem, post):
        # Computed without the covariance, so equal to roundoff, not bit for bit
        assert isinstance(post.variances, Vector)
        assert post.variances.index.equals(problem.state_index)
        np.testing.assert_allclose(
            post.variances.values, np.diag(post.covariance.dense()), rtol=1e-12
        )

    @pytest.mark.parametrize(("attr", "est_attr"), SCALARS)
    def test_scalars_are_the_estimators(self, post, attr, est_attr):
        value = getattr(post, attr)
        assert isinstance(value, float)
        assert value == getattr(post.estimator, est_attr)

    def test_modeled_obs_include_the_constant(self, problem, post):
        est = post.estimator
        np.testing.assert_allclose(
            post.modeled_obs.values, est.H @ est.x_hat + problem.constant.values
        )
        np.testing.assert_allclose(
            post.prior_modeled_obs.values, est.H @ est.x_0 + problem.constant.values
        )


class TestCaching:
    """Each labeled output is built once."""

    @pytest.mark.parametrize("attr", [*OUTPUT_IDS, "mean", "variances"])
    def test_same_object_twice(self, post, attr):
        assert _output(post, attr) is _output(post, attr)

    def test_problem_property_same_object_twice(self, problem):
        problem.solve("bayesian")
        assert problem.posterior is problem.posterior
        assert problem.posterior_error is problem.posterior_error

    def test_a_new_solve_gives_new_outputs(self, problem):
        problem.solve("bayesian")
        first = problem.posterior
        problem.solve("bayesian", gamma=0.5)
        assert problem.posterior is not first
        assert problem.estimator.gamma == 0.5


class TestSolve:
    """fips.solve and InverseProblem.solve."""

    def test_returns_a_posterior(self, post):
        assert isinstance(post, Posterior)
        assert isinstance(post.estimator, BayesianSolver)

    def test_estimate_is_computed_before_it_returns(self, post):
        assert "x_hat" in vars(post.estimator)

    def test_does_not_change_the_problem(self, problem):
        fips.solve(problem)
        assert problem._estimator is None
        with pytest.raises(RuntimeError, match="not been solved"):
            _ = problem.result

    def test_options_reach_the_estimator(self, problem):
        assert fips.solve(problem, gamma=0.5).estimator.gamma == 0.5

    def test_estimator_by_class(self, problem):
        post = fips.solve(problem, BayesianSolver)
        assert type(post.estimator) is BayesianSolver

    def test_same_as_the_problems_solve(self, problem):
        post = fips.solve(problem, "bayesian", gamma=2.0)
        problem.solve("bayesian", gamma=2.0)
        _assert_same(problem.posterior, post.estimate)
        _assert_same(problem.posterior_error, post.covariance)

    def test_unregistered_name_raises(self, problem):
        with pytest.raises(ValueError, match="is not registered"):
            fips.solve(problem, "nope")

    def test_not_an_estimator_raises(self, problem):
        with pytest.raises(TypeError, match="subclass of Estimator"):
            fips.solve(problem, object)  # type: ignore[arg-type]

    def test_problem_solve_returns_the_problem(self, problem):
        assert problem.solve("bayesian") is problem
        assert isinstance(problem.result, Posterior)
        assert problem.estimator is problem.result.estimator
        assert problem._estimator is problem.result.estimator

    def test_unsolved_problem_raises(self, problem):
        with pytest.raises(RuntimeError, match="not been solved"):
            _ = problem.result
        with pytest.raises(RuntimeError, match="not been solved"):
            _ = problem.posterior

    def test_flux_problem(self):
        problem = _problem(FluxProblem).solve()
        post = problem.result

        def block(vector, name):
            return vector[name].to_series().droplevel("block")

        pd.testing.assert_series_equal(
            problem.posterior_fluxes, block(post.estimate, "flux")
        )
        pd.testing.assert_series_equal(
            problem.posterior_concentrations, block(post.modeled_obs, "concentration")
        )
        pd.testing.assert_series_equal(
            problem.prior_concentrations,
            block(post.prior_modeled_obs, "concentration"),
        )


class TestPickle:
    """A Posterior and a solved problem round-trip through pickle."""

    def test_posterior_round_trip(self, post):
        for attr in OUTPUT_IDS:
            _output(post, attr)
        loaded = pickle.loads(pickle.dumps(post))
        assert isinstance(loaded, Posterior)
        for attr in OUTPUT_IDS:
            _assert_same(_output(loaded, attr), _output(post, attr))
        assert loaded.dofs == post.dofs

    def test_labeled_outputs_are_not_pickled(self, post):
        for attr in OUTPUT_IDS:
            _output(post, attr)
        assert set(post.__getstate__()) == {"estimator", "state_index", "obs_index"}
        loaded = pickle.loads(pickle.dumps(post))
        assert not set(OUTPUT_IDS) & set(vars(loaded))

    def test_solved_problem_round_trip(self, problem, tmp_path):
        problem.solve("bayesian")
        _ = problem.posterior_error  # cache an n x n output
        path = tmp_path / "problem.pkl"
        problem.to_file(path)
        loaded = InverseProblem.from_file(path)
        assert isinstance(loaded.result, Posterior)
        assert loaded.estimator is loaded.result.estimator
        for attr in ("posterior", "posterior_error", "kalman_gain", "desroziers"):
            _assert_same(getattr(loaded, attr), getattr(problem, attr))

    def test_problem_pickle_holds_the_estimator_once(self, problem):
        problem.solve("bayesian")
        state = problem.__getstate__()
        assert state["_estimator"] is problem.estimator
        assert not any(isinstance(v, Posterior) for v in state.values())

    def test_unsolved_problem_round_trip(self, problem):
        loaded = pickle.loads(pickle.dumps(problem))
        assert loaded._estimator is None


def test_output_meta_is_gone():
    assert not hasattr(Estimator, "_output_meta")


def test_exported():
    assert fips.solve is fips.problem.solve
    assert "solve" in fips.__all__
    assert "Posterior" in fips.__all__
