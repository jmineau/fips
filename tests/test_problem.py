"""Tests for fips.problem: a Problem checks its labels once, by an exact join."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import fips
from fips import (
    Covariance,
    Exponential,
    Independent,
    Kronecker,
    Operator,
    Problem,
    Space,
    Vector,
)
from fips.linalg import DiagonalOperator, KroneckerOperator, SparseOperator

CELLS = pd.Index([1, 2, 3], name="cell")
SITES = pd.Index(["A", "B"], name="site")
OBS_A = pd.Index(["o1", "o2"], name="obs_id")
OBS_B = pd.Index(["o3"], name="obs_id")

STATE = Space({"flux": CELLS, "bias": SITES})
OBS = Space({"a": OBS_A, "b": OBS_B})

H_VALUES = np.array(
    [
        [1.0, 0.5, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.5, 1.0, 0.0],
        [0.2, 0.0, 1.0, 0.0, 1.0],
    ]
)


def _reversed(space: Space) -> Space:
    """The same labels, every block reversed and the blocks in reverse order."""
    return Space(
        {name: space.select(name)[::-1] for name in reversed(space.names)},
    )


def _on(values: np.ndarray, rows: Space, cols: Space, new_rows: Space, new_cols: Space):
    """Return `values` (on `rows` x `cols`) reordered onto `new_rows` x `new_cols`."""
    return values[np.ix_(new_rows.align(rows), new_cols.align(cols))]


def _inputs(**changes):
    """The inputs of a two-block problem; keyword arguments replace one."""
    inputs = {
        "prior": Vector([1.0, 2.0, 3.0, 0.0, 0.0], STATE, name="prior"),
        "prior_cov": Covariance.diagonal([4.0, 4.0, 4.0, 1.0, 1.0], STATE),
        "obs": Vector([5.0, 6.0, 7.0], OBS, name="obs"),
        "obs_cov": Covariance.diagonal(0.5, OBS),
        "forward": Operator(H_VALUES, OBS, STATE, name="H"),
        "offset": None,
    }
    inputs.update(changes)
    return inputs


class TestConstruction:
    def test_inputs_are_kept_on_their_spaces(self):
        problem = Problem(**_inputs())
        assert problem.prior.space is STATE and problem.obs.space is OBS
        assert problem.forward.rows is OBS and problem.forward.cols is STATE
        assert problem.prior_cov.space is STATE and problem.obs_cov.space is OBS
        np.testing.assert_array_equal(problem.forward.dense(), H_VALUES)
        assert problem.offset is None
        assert problem.forward.name == "H"

    def test_repr(self):
        assert repr(Problem(**_inputs())) == (
            "Problem(state: Space(flux: 3 x ['cell'], bias: 2 x ['site']), "
            "obs: Space(a: 2 x ['obs_id'], b: 1 x ['obs_id']))"
        )

    def test_builds_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Problem(**_inputs())

    @pytest.mark.parametrize(
        ("argument", "value", "match"),
        [
            ("prior", pd.Series([1.0]), "prior must be a Vector.*Vector.from_series"),
            ("obs", np.ones(3), "obs must be a Vector"),
            ("forward", H_VALUES, "forward must be an Operator"),
            (
                "forward",
                pd.DataFrame(H_VALUES),
                "forward must be an Operator.*Operator.from_frame",
            ),
            ("prior_cov", 1.0, "prior_cov must be a Covariance or a covariance model"),
            (
                "obs_cov",
                Operator(np.eye(3), OBS, OBS),
                "obs_cov must be a Covariance",
            ),
            ("offset", "1", "offset must be a Vector, a number or None"),
        ],
    )
    def test_refused_types_name_the_way_in(self, argument, value, match):
        with pytest.raises(TypeError, match=match):
            Problem(**_inputs(**{argument: value}))

    def test_has_no_solve_method(self):
        """A problem is solved by fips.solve, which returns the posterior."""
        assert not hasattr(Problem(**_inputs()), "solve")


class TestZeroPriorVariances:
    """Some prior variances may be zero; all of them may not."""

    STATE = Space({"flux": pd.Index(list("abc"), name="cell")})
    OBS = Space({"co2": pd.Index(range(4), name="k")})

    def _problem(self, prior_cov) -> Problem:
        """A problem with this prior error covariance."""
        rng = np.random.default_rng(0)
        return Problem(
            prior=Vector(np.ones(3), self.STATE),
            prior_cov=prior_cov,
            obs=Vector(rng.standard_normal(4), self.OBS),
            obs_cov=Independent().with_std(0.5),
            forward=Operator(rng.uniform(size=(4, 3)), self.OBS, self.STATE),
        )

    @pytest.mark.parametrize(
        "prior_cov",
        [
            Independent().with_std(0.0),
            Exponential("cell_number", 1.0).with_std(0.0),
        ],
        ids=["independent", "correlated"],
    )
    def test_all_zero_raises(self, prior_cov):
        """A prior no observation could change is refused where it is given."""
        state = Space(
            {
                "flux": pd.MultiIndex.from_arrays(
                    [list("abc"), [0.0, 1.0, 2.0]], names=["cell", "cell_number"]
                )
            }
        )
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Every variance of prior_cov is zero"):
            Problem(
                prior=Vector(np.ones(3), state),
                prior_cov=prior_cov,
                obs=Vector(rng.standard_normal(4), self.OBS),
                obs_cov=Independent().with_std(0.5),
                forward=Operator(rng.uniform(size=(4, 3)), self.OBS, state),
            )

    def test_all_zero_covariance_raises(self):
        """A covariance given as numbers is refused the same way."""
        with pytest.raises(ValueError, match="Every variance of prior_cov is zero"):
            self._problem(Covariance.diagonal(np.zeros(3), self.STATE))

    def test_some_zero_solves(self):
        """An element known exactly stays at its prior; the others move."""
        std = pd.Series([1.0, 0.0, 1.0], index=pd.Index(list("abc"), name="cell"))
        estimate = fips.solve(
            self._problem(Independent().with_std(std))
        ).estimate.values
        assert estimate[1] == 1.0
        assert estimate[0] != 1.0

    def test_restricting_to_the_zero_elements_raises(self):
        """A restriction goes through the same check."""
        std = pd.Series([1.0, 0.0, 1.0], index=pd.Index(list("abc"), name="cell"))
        problem = self._problem(Independent().with_std(std))
        keep = Space({"flux": pd.Index(["b"], name="cell")})
        with pytest.raises(ValueError, match="Every variance of prior_cov is zero"):
            problem.restrict(keep)


class TestExactJoin:
    """Every input is put on the prior's or the observations' space by its labels."""

    def test_inputs_in_another_order_are_reordered(self):
        state, obs = _reversed(STATE), _reversed(OBS)
        given = Problem(**_inputs())
        S_0 = given.prior_cov.dense() + 0.1  # not diagonal, so an order shows
        S_z = given.obs_cov.dense() + 0.01
        problem = Problem(
            **_inputs(
                forward=Operator(_on(H_VALUES, OBS, STATE, obs, state), obs, state),
                prior_cov=Covariance(_on(S_0, STATE, STATE, state, state), state),
                obs_cov=Covariance(_on(S_z, OBS, OBS, obs, obs), obs),
                offset=Vector([0.3, 0.1, 0.2], Space({"b": OBS_B, "a": OBS_A[::-1]})),
            )
        )
        np.testing.assert_array_equal(problem.forward.dense(), H_VALUES)
        np.testing.assert_array_equal(problem.prior_cov.dense(), S_0)
        np.testing.assert_array_equal(problem.obs_cov.dense(), S_z)
        assert problem.offset is not None
        np.testing.assert_array_equal(problem.offset.values, [0.2, 0.1, 0.3])
        assert problem.offset.space is OBS

    def test_a_jacobian_on_part_of_the_state_raises(self):
        """A column missing (a cell no footprint touched) raises, naming it: no zero fill."""
        cols = Space({"flux": CELLS[:2], "bias": SITES})
        H = Operator(np.delete(H_VALUES, 2, axis=1), OBS, cols)
        with pytest.raises(
            ValueError,
            match=r"The columns of forward and the prior do not have the same labels: "
            r"labels of block 'flux' missing from the columns of forward: 3\.",
        ):
            Problem(**_inputs(forward=H))

    def test_a_label_only_forward_has_raises(self):
        cols = Space({"flux": pd.Index([1, 2, 3, 4], name="cell"), "bias": SITES})
        H = Operator(np.insert(H_VALUES, 3, 1.0, axis=1), OBS, cols)
        with pytest.raises(
            ValueError, match="block 'flux' only in the columns of forward: 4"
        ):
            Problem(**_inputs(forward=H))

    def test_issue_80_a_misspelled_block_raises(self):
        """The reproduction of #80: a column block that matches no state block raises."""
        cols = Space({"fluxx": CELLS, "bias": SITES})
        with pytest.raises(ValueError) as excinfo:
            Problem(**_inputs(forward=Operator(H_VALUES, OBS, cols)))
        message = str(excinfo.value)
        assert "blocks missing from the columns of forward: 'flux'" in message
        assert "blocks only in the columns of forward: 'fluxx'" in message

    def test_rows_of_forward_are_the_observations(self):
        rows = Space({"a": OBS_A, "c": OBS_B})
        with pytest.raises(
            ValueError, match="The rows of forward and the observations do not have"
        ):
            Problem(**_inputs(forward=Operator(H_VALUES, rows, STATE)))

    @pytest.mark.parametrize(
        ("argument", "space", "what"),
        [
            (
                "prior_cov",
                Space({"flux": CELLS, "bias": pd.Index(["A", "C"], name="site")}),
                "prior_cov",
            ),
            (
                "obs_cov",
                Space({"a": OBS_A, "b": pd.Index(["o4"], name="obs_id")}),
                "obs_cov",
            ),
        ],
    )
    def test_a_covariance_on_other_labels_raises(self, argument, space, what):
        with pytest.raises(ValueError, match=f"The rows of {what} and the"):
            Problem(**_inputs(**{argument: Covariance.diagonal(1.0, space)}))

    def test_an_offset_on_other_labels_raises(self):
        offset = Vector([1.0, 2.0], Space({"a": OBS_A}))
        with pytest.raises(ValueError, match="blocks missing from the offset: 'b'"):
            Problem(**_inputs(offset=offset))

    def test_float_noise_in_labels_is_not_matched(self):
        """Labels are matched as given (#78): noise raises, and rounding on purpose matches them."""
        lats = pd.Index([40.1, 40.2], name="lat")
        noisy = pd.Index([40.1 + 1e-9, 40.2], name="lat")
        state, obs = Space({"flux": lats}), Space({"co2": pd.Index([0], name="obs_id")})
        inputs = {
            "prior": Vector([0.0, 0.0], state),
            "prior_cov": Covariance.diagonal(1.0, state),
            "obs": Vector([1.0], obs),
            "obs_cov": Covariance.diagonal(1.0, obs),
            "forward": Operator(np.ones((1, 2)), obs, Space({"flux": noisy})),
        }
        with pytest.raises(
            ValueError, match="missing from the columns of forward: 40.1"
        ):
            Problem(**inputs)
        inputs["forward"] = inputs["forward"].round_labels(6)
        inputs["prior"] = inputs["prior"].round_labels(6)
        inputs["prior_cov"] = inputs["prior_cov"].round_labels(6)
        Problem(**inputs)

    def test_labels_are_kept_as_given(self):
        """Site codes stay strings (#77)."""
        codes = pd.Index(["001", "010"], name="site")
        state, obs = (
            Space({"bias": codes}),
            Space({"co2": pd.Index([0], name="obs_id")}),
        )
        problem = Problem(
            prior=Vector([0.0, 0.0], state),
            prior_cov=Independent().with_std(1.0),
            obs=Vector([1.0], obs),
            obs_cov=Independent().with_std(1.0),
            forward=Operator(np.ones((1, 2)), obs, state),
        )
        assert problem.prior_cov.space.select("bias").tolist() == ["001", "010"]


class TestForms:
    """The forms are kept: a sparse, diagonal or Kronecker input stays one."""

    def test_sparse_forward_and_diagonal_covariances(self):
        problem = Problem(
            **_inputs(forward=Operator(sp.csr_array(H_VALUES), OBS, STATE))
        )
        assert isinstance(problem.forward.data, SparseOperator)
        assert isinstance(problem.prior_cov.data, DiagonalOperator)
        assert isinstance(problem.obs_cov.data, DiagonalOperator)

    def test_sparse_forward_reordered_stays_sparse(self):
        state, obs = _reversed(STATE), _reversed(OBS)
        H = Operator(sp.csr_array(_on(H_VALUES, OBS, STATE, obs, state)), obs, state)
        problem = Problem(**_inputs(forward=H))
        assert isinstance(problem.forward.data, SparseOperator)
        np.testing.assert_array_equal(problem.forward.dense(), H_VALUES)


class TestModels:
    """A covariance model is built on the problem's spaces."""

    @staticmethod
    def _grid():
        times = pd.date_range("2024-01-01", periods=3, freq="D")
        cells = pd.MultiIndex.from_product([times, [40.0, 40.5]], names=["time", "lat"])
        state = Space({"flux": cells})
        obs = Space({"co2": pd.Index(range(4), name="obs_id")})
        return state, obs

    def test_a_model_is_built_on_the_prior_space(self):
        state, obs = self._grid()
        model = Kronecker(Exponential("time", "2D"), Exponential("lat", 1.0)).with_std(
            2.0
        )
        problem = Problem(
            prior=Vector(np.zeros(6), state),
            prior_cov=model,
            obs=Vector(np.ones(4), obs),
            obs_cov=Independent().with_std(0.5),
            forward=Operator(np.ones((4, 6)), obs, state),
        )
        assert isinstance(problem.prior_cov.data, KroneckerOperator)
        assert (
            problem.prior_cov.space is state and problem.prior_cov.name == "prior_cov"
        )
        np.testing.assert_array_equal(
            problem.prior_cov.dense(), model.build(state).dense()
        )
        np.testing.assert_array_equal(
            problem.obs_cov.variances.values, np.full(4, 0.25)
        )
        assert problem.obs_cov.name == "obs_cov"

    def test_a_model_reading_a_level_the_space_lacks_raises(self):
        state, obs = self._grid()
        with pytest.raises(KeyError, match="'obs_time'"):
            Problem(
                prior=Vector(np.zeros(6), state),
                prior_cov=Independent().with_std(1.0),
                obs=Vector(np.ones(4), obs),
                obs_cov=Exponential("obs_time", "1h").with_std(1.0),
                forward=Operator(np.ones((4, 6)), obs, state),
            )

    def test_a_model_given_a_vector_is_checked_by_the_join(self):
        """``with_std(Vector)`` is a Covariance already: its labels must be the space's."""
        state, obs = self._grid()
        std = Vector(np.ones(5), Space({"flux": state.select("flux")[:5]}))
        with pytest.raises(ValueError, match="The rows of prior_cov and the prior"):
            Problem(
                prior=Vector(np.zeros(6), state),
                prior_cov=Independent().with_std(std),
                obs=Vector(np.ones(4), obs),
                obs_cov=Independent().with_std(1.0),
                forward=Operator(np.ones((4, 6)), obs, state),
            )

    def test_block_diag_of_models_by_block(self):
        problem = Problem(
            **_inputs(
                prior_cov=Covariance.block_diag(
                    flux=Exponential("cell", 1.0).with_std(2.0),
                    bias=Independent().with_std(0.1),
                )
            )
        )
        np.testing.assert_allclose(
            problem.prior_cov.variances.values, [4.0, 4.0, 4.0, 0.01, 0.01]
        )


class TestOffset:
    def test_a_number_is_the_same_for_every_observation(self):
        problem = Problem(**_inputs(offset=2.5))
        assert problem.offset is not None
        assert problem.offset.space is OBS
        np.testing.assert_array_equal(problem.offset.values, [2.5, 2.5, 2.5])

    def test_an_offset_enters_the_modeled_observations(self):
        offset = Vector([1.0, 2.0, 3.0], OBS, name="background")
        with_offset = Problem(**_inputs(offset=offset))
        without = Problem(**_inputs())
        np.testing.assert_allclose(
            with_offset.predict(with_offset.prior).values
            - without.predict(without.prior).values,
            [1.0, 2.0, 3.0],
        )
        assert with_offset.offset is not None
        assert with_offset.offset.name == "background"


class TestPredict:
    def test_is_forward_times_the_state_plus_the_offset(self):
        problem = Problem(**_inputs(offset=0.5))
        x = Vector([1.0, -1.0, 2.0, 0.5, 0.0], STATE)
        predicted = problem.predict(x)
        assert isinstance(predicted, Vector) and predicted.space is OBS
        np.testing.assert_allclose(predicted.values, H_VALUES @ x.values + 0.5)

    def test_the_state_is_matched_by_label(self):
        problem = Problem(**_inputs())
        x = Vector([1.0, -1.0, 2.0, 0.5, 0.0], STATE)
        reordered = Vector(x.values[_reversed(STATE).align(STATE)], _reversed(STATE))
        np.testing.assert_array_equal(
            problem.predict(reordered).values, problem.predict(x).values
        )

    def test_a_state_on_other_labels_raises(self):
        problem = Problem(**_inputs())
        short = Space({"flux": CELLS[:2], "bias": SITES})
        with pytest.raises(
            ValueError,
            match=r"x and the prior do not have the same labels: "
            r"labels of block 'flux' missing from x: 3",
        ):
            problem.predict(Vector(np.ones(4), short))

    def test_refuses_an_array(self):
        with pytest.raises(TypeError, match="x must be a Vector"):
            Problem(**_inputs()).predict(np.ones(5))  # type: ignore[arg-type]


class TestStiltShapedForward:
    """A Jacobian of the cells a footprint touched, placed on the whole state by ``Operator.from_blocks``."""

    @staticmethod
    def _inputs(rng):
        cells = pd.Index(range(10), name="cell")
        prior = Vector.from_series(
            pd.Series(rng.uniform(1, 2, 10), index=cells), block="flux"
        )
        obs_ids = pd.Index(range(4), name="obs_id")
        obs = Vector.from_series(
            pd.Series(rng.uniform(0, 5, 4), index=obs_ids), block="co2"
        )
        return prior, obs, obs_ids

    @pytest.mark.parametrize(
        "given", ["sparse tuple", "pandas-sparse frame", "dense frame"]
    )
    def test_placed_by_label(self, given):
        rng = np.random.default_rng(0)
        prior, obs, obs_ids = self._inputs(rng)
        touched = pd.Index([7, 2, 5], name="cell")  # in the Jacobian's own order
        values = rng.uniform(0, 1, (4, 3))
        rows = obs_ids[::-1]  # and its rows in theirs
        if given == "sparse tuple":
            J = (sp.csr_array(values), rows, touched)
        else:
            J = pd.DataFrame(values, index=rows, columns=touched)
            if given == "pandas-sparse frame":
                J = J.astype(pd.SparseDtype(float, 0.0))

        forward = Operator.from_blocks(
            {("co2", "flux"): J}, rows=obs.space, cols=prior.space
        )
        problem = Problem(
            prior=prior,
            prior_cov=Independent().with_std(0.5),
            obs=obs,
            obs_cov=Independent().with_std(0.1),
            forward=forward,
        )

        assert isinstance(problem.forward.data, SparseOperator) == (
            given != "dense frame"
        )
        dense = np.zeros((4, 10))
        dense[:, [7, 2, 5]] = values[::-1]
        np.testing.assert_array_equal(problem.forward.dense(), dense)
        post = fips.solve(problem)
        untouched = [c for c in range(10) if c not in (7, 2, 5)]
        np.testing.assert_array_equal(
            post.estimate.values[untouched], prior.values[untouched]
        )

    def test_a_label_not_in_the_state_raises(self):
        rng = np.random.default_rng(1)
        prior, obs, obs_ids = self._inputs(rng)
        J = pd.DataFrame(
            np.ones((4, 2)), index=obs_ids, columns=pd.Index([3, 12], name="cell")
        )
        with pytest.raises(
            ValueError,
            match=r"The columns of the matrix of \('co2', 'flux'\) of block 'flux' "
            r"have labels the block does not have: 12\.",
        ):
            Operator.from_blocks({("co2", "flux"): J}, rows=obs.space, cols=prior.space)


class TestPickle:
    def test_round_trip_keeps_the_forms_and_spaces(self):
        problem = Problem(
            **_inputs(forward=Operator(sp.csr_array(H_VALUES), OBS, STATE), offset=1.0)
        )
        loaded = pickle.loads(pickle.dumps(problem))
        assert loaded.prior.space == STATE and loaded.obs.space == OBS
        assert isinstance(loaded.forward.data, SparseOperator)
        np.testing.assert_array_equal(loaded.forward.dense(), H_VALUES)
        np.testing.assert_array_equal(
            loaded.prior_cov.dense(), problem.prior_cov.dense()
        )
        assert loaded.offset is not None
        np.testing.assert_array_equal(loaded.offset.values, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(
            fips.solve(loaded).estimate.values, fips.solve(problem).estimate.values
        )

    def test_pickle_shares_the_spaces(self):
        loaded = pickle.loads(pickle.dumps(Problem(**_inputs())))
        assert loaded.forward.cols is loaded.prior.space

    def test_pickle_keeps_the_operators_read_only(self):
        loaded = pickle.loads(pickle.dumps(Problem(**_inputs())))
        assert not loaded.forward.data.array.flags.writeable
        assert not loaded.prior_cov.data.values.flags.writeable


class TestOperatorsAreNotCopiedAgain:
    def test_the_problem_holds_the_forms_it_was_given(self):
        """An operator already holds its own copy: the problem takes its form as it is."""
        inputs = _inputs()
        problem = Problem(**inputs)
        assert problem.forward.data is inputs["forward"].data
        assert problem.prior_cov.data is inputs["prior_cov"].data
        assert problem.obs_cov.data is inputs["obs_cov"].data

    def test_editing_the_matrix_after_the_operator_leaves_the_problem(self):
        H = H_VALUES.copy()
        problem = Problem(**_inputs(forward=Operator(H, OBS, STATE)))
        before = fips.solve(problem).estimate.values
        H.fill(-1.0)
        np.testing.assert_array_equal(fips.solve(problem).estimate.values, before)


class Quadratic(fips.ForwardModel):
    """``F(x) = H x + (H x)^2 / 2``, the Jacobian ``diag(1 + H x) H``; at the top level, so it pickles."""

    def __init__(self, H: Operator, rows: Space | None = None):
        super().__init__(H.rows if rows is None else rows, H.cols)
        self.H = H

    def __call__(self, x):
        Hx = self.H @ x
        return (Hx + Hx * Hx / 2.0).align(self.rows)

    def jacobian(self, x):
        Hx = self.H @ x
        scale = Operator(DiagonalOperator(1.0 + Hx.values), self.H.rows, self.H.rows)
        return scale @ self.H


@pytest.mark.parametrize("kind", ["operator", "model"])
def test_a_problems_forward_is_a_forward_model_of_either_kind(kind):
    """``problem.forward(x)`` and ``problem.forward.jacobian(x)`` mean the same for an operator and a model, so one piece of code reads both."""
    H = Operator(H_VALUES, OBS, STATE)
    problem = Problem(
        **_inputs(forward=H if kind == "operator" else Quadratic(H), offset=0.5)
    )
    x = TestForwardModel.X
    forward = problem.forward
    np.testing.assert_allclose(
        (forward(x) + 0.5).align(OBS).values, problem.predict(x).values
    )
    np.testing.assert_allclose(
        forward.jacobian(x).dense(), problem.linearized(x).forward.dense()
    )
    if kind == "operator":
        assert forward.jacobian(x) is forward
        np.testing.assert_array_equal((forward @ x).values, forward(x).values)


class TestForwardModel:
    """A problem whose forward is a model: its labels are checked, and predict, simulate, linearized and aggregate run it."""

    X = Vector([1.0, -1.0, 2.0, 0.5, 0.2], STATE)

    def problem(self, **changes) -> Problem:
        H = Operator(H_VALUES, OBS, STATE, name="H")
        return Problem(**_inputs(forward=Quadratic(H), offset=0.5, **changes))

    @staticmethod
    def F(x: np.ndarray) -> np.ndarray:
        Hx = H_VALUES @ x
        return Hx + Hx**2 / 2.0

    def test_predict_and_simulate_run_the_model(self):
        problem = self.problem()
        np.testing.assert_allclose(
            problem.predict(self.X).values, self.F(self.X.values) + 0.5
        )
        truth, obs = problem.simulate(rng=3, truth=self.X)
        noise = problem.obs_cov.sample(np.random.default_rng(3)).values
        np.testing.assert_allclose(obs.values, self.F(self.X.values) + 0.5 + noise)
        assert problem.forward.__class__ is Quadratic

    def test_a_model_on_labels_in_another_order_is_called_in_its_own(self):
        H = Operator(H_VALUES, OBS, STATE).align(_reversed(OBS), _reversed(STATE))
        problem = Problem(**_inputs(forward=Quadratic(H), offset=0.5))
        np.testing.assert_allclose(
            problem.predict(self.X).values, self.F(self.X.values) + 0.5
        )
        linear = problem.linearized(self.X)
        assert linear.forward.rows is OBS and linear.forward.cols is STATE

    def test_linearized_is_the_tangent_problem(self):
        problem = self.problem()
        linear = problem.linearized(self.X)
        J = (1.0 + H_VALUES @ self.X.values)[:, None] * H_VALUES
        assert type(linear) is Problem and isinstance(linear.forward, Operator)
        np.testing.assert_allclose(linear.forward.dense(), J)
        np.testing.assert_allclose(
            linear.offset.values, 0.5 + self.F(self.X.values) - J @ self.X.values
        )
        # at x itself it predicts what the model does
        np.testing.assert_allclose(
            linear.predict(self.X).values, problem.predict(self.X).values
        )
        assert linear.prior is problem.prior and linear.obs is problem.obs
        reordered = self.X.align(_reversed(STATE))
        np.testing.assert_array_equal(
            problem.linearized(reordered).offset.values, linear.offset.values
        )

    def test_a_linear_problem_is_its_own_linearization(self):
        problem = Problem(**_inputs())
        assert problem.linearized(self.X) is problem

    def test_aggregate_follows_the_model_with_the_aggregation(self):
        problem = self.problem()
        W = Operator(
            np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]),
            Space({"daily": pd.Index(["d1", "d2"], name="day")}),
            OBS,
        )
        daily = problem.aggregate(W)
        assert isinstance(daily.forward, fips.ForwardModel)
        np.testing.assert_allclose(
            daily.predict(self.X).values,
            W.dense() @ (self.F(self.X.values) + 0.5),
        )
        J = (1.0 + H_VALUES @ self.X.values)[:, None] * H_VALUES
        np.testing.assert_allclose(
            daily.linearized(self.X).forward.dense(), W.dense() @ J
        )
        assert fips.taylortest(daily.forward, self.X)
        loaded = pickle.loads(pickle.dumps(daily))
        np.testing.assert_array_equal(
            loaded.predict(self.X).values, daily.predict(self.X).values
        )

    def test_restrict_refuses_a_model(self):
        with pytest.raises(TypeError, match=r"restrict needs a linear.*linearized"):
            self.problem().restrict(Space({"flux": CELLS}))

    def test_what_needs_an_operator_refuses_a_model(self):
        with pytest.raises(TypeError, match=r"gauss_newton.*linearized"):
            _ = self.problem()._operator

    def test_a_model_on_other_labels_raises(self):
        H = Operator(H_VALUES[:2], Space({"a": OBS_A}), STATE)
        with pytest.raises(
            ValueError,
            match=r"The rows of forward and the observations do not have the same "
            r"labels: blocks missing from the rows of forward: 'b'",
        ):
            Problem(**_inputs(forward=Quadratic(H)))
        other_state = Space({"flux": CELLS, "bias": pd.Index(["A", "C"], name="site")})
        H = Operator(H_VALUES, OBS, other_state)
        with pytest.raises(ValueError, match=r"The columns of forward and the prior"):
            Problem(**_inputs(forward=Quadratic(H)))

    def test_a_model_that_returns_other_labels_raises(self):
        H = Operator(H_VALUES, OBS, STATE)
        other = Space({"a": OBS_A, "b": pd.Index(["o9"], name="obs_id")})

        class Wrong(Quadratic):
            def __call__(self, x):
                return Vector(super().__call__(x).values, other)

        problem = Problem(**_inputs(forward=Wrong(H)))
        with pytest.raises(
            ValueError, match=r"What Wrong.__call__ returns and its rows do not"
        ):
            problem.predict(self.X)

    def test_a_model_that_returns_other_types_raises(self):
        H = Operator(H_VALUES, OBS, STATE)

        class Array(Quadratic):
            def __call__(self, x):
                return super().__call__(x).values

        class Matrix(Quadratic):
            def jacobian(self, x):
                return super().jacobian(x).dense()

        with pytest.raises(TypeError, match=r"Array.__call__ must return a Vector"):
            Problem(**_inputs(forward=Array(H))).predict(self.X)
        with pytest.raises(TypeError, match=r"Matrix.jacobian must return an Operator"):
            Problem(**_inputs(forward=Matrix(H))).linearized(self.X)

    def test_a_model_without_its_spaces_or_a_wrong_forward_raises(self):
        class NoInit(Quadratic):
            def __init__(self, H):
                self.H = H

        with pytest.raises(TypeError, match=r"super\(\).__init__\(rows, cols\)"):
            Problem(**_inputs(forward=NoInit(Operator(H_VALUES, OBS, STATE))))
        with pytest.raises(TypeError, match="an Operator or a ForwardModel"):
            Problem(**_inputs(forward=H_VALUES))
        with pytest.raises(TypeError, match="rows must be a Space"):
            Quadratic(Operator(H_VALUES, OBS, STATE), rows=OBS_A)
