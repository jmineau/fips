"""
Tests for Problem.normalized_innovations and Problem.keep_obs: each observation against the prior, and a problem on the observations kept.

Every case is held to the dense calculation from the matrices written out by
hand, for each form the covariances come in.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator

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
from fips.linalg import BlockDiagonalOperator, KroneckerOperator, SparseOperator

TIMES = pd.date_range("2020-01-01", periods=4, freq="D", name="obs_time")
SITES = pd.Index([1.0, 2.5, 4.0], name="x")
GRID = pd.MultiIndex.from_product([TIMES, SITES])  # 12 labels, time-major
OBS = Space({"co2": GRID, "ch4": pd.Index(list("pqrst"), name="gas_id")})
FLUX = pd.MultiIndex.from_product(
    [pd.date_range("2020-01-01", periods=2, freq="2D", name="time"), SITES]
)
STATE = Space({"flux": FLUX, "bias": pd.Index(["A", "B"], name="site")})
M, N = len(OBS), len(STATE)  # 17 and 8

KRONECKER = Kronecker(Exponential("obs_time", "2D"), Exponential("x", 2.0))
STATE_KRONECKER = Kronecker(Exponential("time", "3D"), Exponential("x", 2.0))


def _prior_cov(form: str) -> Covariance:
    cov = Covariance.block_diag(
        flux=STATE_KRONECKER.with_std(0.7), bias=Independent().with_std(0.3)
    ).build(STATE)
    if form == "dense":
        return Covariance(cov.dense(), STATE)
    return cov


def _obs_cov(form: str) -> Covariance:
    if form == "diagonal":
        return Independent().with_std(0.4).build(OBS)
    if form == "block":
        return Covariance.block_diag(
            co2=KRONECKER.with_std(0.4), ch4=Independent().with_std(0.2)
        ).build(OBS)
    if form == "kronecker":
        only = Space({"co2": GRID})
        return KRONECKER.with_std(0.4).build(only)
    rng = np.random.default_rng(4)
    if form == "sparse":
        band = sp.diags_array(
            [np.full(M, 0.3), np.full(M - 1, 0.05), np.full(M - 1, 0.05)],
            offsets=[0, 1, -1],
        )
        return Covariance(sp.csr_array(band), OBS)
    A = rng.normal(size=(M, M))
    return Covariance(A @ A.T / M + 0.1 * np.eye(M), OBS)  # dense


def _forward(kind: str, space: Space) -> Operator:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(len(space), N))
    if kind == "sparse":
        values[rng.random(values.shape) < 0.6] = 0.0
        return Operator(sp.csr_array(values), space, STATE)
    return Operator(values, space, STATE)


def make(
    obs_cov: str = "diagonal",
    prior_cov: str = "block",
    forward: str = "dense",
    offset: bool = True,
) -> Problem:
    """A problem on 17 observations (two blocks) and 8 state elements."""
    cov = _obs_cov(obs_cov)
    space = cov.space  # "kronecker" is the co2 block alone
    rng = np.random.default_rng(2)
    return Problem(
        prior=Vector(rng.normal(size=N), STATE, name="prior"),
        prior_cov=_prior_cov(prior_cov),
        obs=Vector(rng.normal(size=len(space)), space, name="obs"),
        obs_cov=cov,
        forward=_forward(forward, space),
        offset=Vector(rng.normal(size=len(space)), space) if offset else None,
    )


def reference(problem: Problem) -> np.ndarray:
    """Normalized innovations from the dense matrices of a problem with an operator."""
    H = problem.forward.dense()
    S0, Sz = problem.prior_cov.dense(), problem.obs_cov.dense()
    c = 0.0 if problem.offset is None else problem.offset.values
    d = problem.obs.values - H @ problem.prior.values - c
    return d / np.sqrt(np.diag(H @ S0 @ H.T + Sz))


OBS_COVS = ["diagonal", "block", "kronecker", "sparse", "dense"]


class TestNormalizedInnovations:
    @pytest.mark.parametrize("obs_cov", OBS_COVS)
    @pytest.mark.parametrize("prior_cov", ["block", "dense"])
    @pytest.mark.parametrize("forward", ["dense", "sparse"])
    def test_equals_the_dense_calculation(self, obs_cov, prior_cov, forward):
        problem = make(obs_cov, prior_cov, forward)
        d = problem.normalized_innovations()
        assert d.space is problem.obs.space
        np.testing.assert_allclose(d.values, reference(problem), rtol=1e-12)

    def test_without_an_offset(self):
        problem = make(offset=False)
        np.testing.assert_allclose(
            problem.normalized_innovations().values, reference(problem), rtol=1e-12
        )

    def test_rows_in_blocks_give_the_same_values(self, monkeypatch):
        """A block of 2 rows (the last one a single row) is the whole calculation."""
        problem = make("block")
        whole = problem.normalized_innovations().values
        monkeypatch.setattr("fips.problem._INNOVATION_CHUNK", 2 * N)
        np.testing.assert_allclose(
            problem.normalized_innovations().values, whole, rtol=1e-12
        )
        monkeypatch.setattr("fips.problem._INNOVATION_CHUNK", 1)  # one row at a time
        np.testing.assert_allclose(
            problem.normalized_innovations().values, whole, rtol=1e-12
        )

    def test_a_matrix_free_forward_and_prior_covariance(self):
        problem = make("diagonal", "dense")
        H = problem.forward.dense()
        S0 = problem.prior_cov.dense()
        free = Problem(
            prior=problem.prior,
            prior_cov=Covariance(
                fips.linalg.MatrixFreeOperator(aslinearoperator(S0)), STATE
            ),
            obs=problem.obs,
            obs_cov=problem.obs_cov,
            forward=Operator(
                fips.linalg.MatrixFreeOperator(aslinearoperator(H)), OBS, STATE
            ),
            offset=problem.offset,
        )
        np.testing.assert_allclose(
            free.normalized_innovations().values,
            problem.normalized_innovations().values,
            rtol=1e-10,
        )

    def test_labels_in_another_order(self):
        """The inputs given in another order are matched by label."""
        problem = make("sparse")
        reversed_space = Space({name: OBS.select(name)[::-1] for name in OBS.names})
        other = Problem(
            prior=problem.prior,
            prior_cov=problem.prior_cov,
            obs=problem.obs.align(reversed_space),
            obs_cov=problem.obs_cov.align(reversed_space, reversed_space),
            forward=problem.forward.align(reversed_space, STATE),
            offset=problem.offset.align(reversed_space),
        )
        d = problem.normalized_innovations().to_series()
        d_other = other.normalized_innovations().to_series()
        pd.testing.assert_series_equal(d_other.reindex(d.index), d)

    def test_a_forward_model_is_taken_at_the_prior(self):
        """With a model, F(x_0) and the Jacobian there stand in for H x_0 and H."""

        class Square(fips.ForwardModel):
            def __init__(self, H: Operator):
                super().__init__(H.rows, H.cols)
                self.H = H

            def __call__(self, x):
                Hx = self.H @ x
                return Hx + Hx * Hx / 2.0

            def jacobian(self, x):
                Hx = self.H @ x
                scale = fips.linalg.DiagonalOperator(1.0 + Hx.values)
                return Operator(scale, self.H.rows, self.H.rows) @ self.H

        linear = make("block")
        model = Problem(
            prior=linear.prior,
            prior_cov=linear.prior_cov,
            obs=linear.obs,
            obs_cov=linear.obs_cov,
            forward=Square(linear.forward),
            offset=linear.offset,
        )
        tangent = model.linearized(model.prior)
        expected = reference(tangent)
        d_tangent = tangent.obs.values - tangent.predict(model.prior).values
        d_model = model.obs.values - model.predict(model.prior).values
        np.testing.assert_allclose(d_tangent, d_model, atol=1e-12)
        np.testing.assert_allclose(
            model.normalized_innovations().values, expected, rtol=1e-12
        )

    def test_a_large_departure_is_a_large_value(self):
        problem = make("diagonal")
        values = problem.obs.values.copy()
        values[3] += 100.0
        shifted = Problem(
            prior=problem.prior,
            prior_cov=problem.prior_cov,
            obs=Vector(values, OBS),
            obs_cov=problem.obs_cov,
            forward=problem.forward,
            offset=problem.offset,
        )
        d = shifted.normalized_innovations().values
        assert np.argmax(np.abs(d)) == 3 and abs(d[3]) > 20

    def test_nothing_is_rejected_and_the_problem_is_unchanged(self):
        problem = make("block")
        before = problem.obs.values.copy()
        d = problem.normalized_innovations()
        assert isinstance(d, Vector) and len(d) == M
        np.testing.assert_array_equal(problem.obs.values, before)

    def test_no_spread_is_not_finite_and_does_not_warn(self, recwarn):
        problem = Problem(
            prior=Vector([1.0], Space({"s": pd.Index(["a"], name="c")})),
            prior_cov=Independent().with_std(1.0),
            obs=Vector([1.0, 2.0], Space({"o": pd.Index([0, 1], name="i")})),
            obs_cov=Covariance.diagonal(0.0, Space({"o": pd.Index([0, 1], name="i")})),
            forward=Operator(
                np.array([[0.0], [1.0]]),
                Space({"o": pd.Index([0, 1], name="i")}),
                Space({"s": pd.Index(["a"], name="c")}),
            ),
        )
        d = problem.normalized_innovations().values
        assert not np.isfinite(d[0]) and np.isfinite(d[1])
        assert not recwarn.list


def _by_hand(problem: Problem, keep: Space) -> Problem:
    """The problem cut to `keep` from the dense matrices, with no fips selection."""
    positions = np.array(
        [problem.obs.space.index.get_loc(label) for label in keep.index]
    )
    Sz = problem.obs_cov.dense()[np.ix_(positions, positions)]
    H = problem.forward.dense()[positions]
    c = None if problem.offset is None else problem.offset.values[positions]
    return Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=Vector(problem.obs.values[positions], keep),
        obs_cov=Covariance(Sz, keep),
        forward=Operator(H, keep, STATE),
        offset=None if c is None else Vector(c, keep),
    )


def _keep(problem: Problem) -> Space:
    """Some labels of each block, not in order."""
    space = problem.obs.space
    blocks = {"co2": space.select("co2")[[7, 1, 2, 8, 5, 11]]}
    if "ch4" in space:
        blocks["ch4"] = space.select("ch4")[[3, 0, 4]]
    return Space(blocks)


class TestKeepObs:
    @pytest.mark.parametrize("obs_cov", OBS_COVS)
    @pytest.mark.parametrize("forward", ["dense", "sparse"])
    def test_equals_the_problem_cut_by_hand(self, obs_cov, forward):
        problem = make(obs_cov, forward=forward)
        keep = _keep(problem)
        kept = problem.keep_obs(keep)
        hand = _by_hand(problem, keep)
        assert type(kept) is Problem and kept.obs.space is keep
        assert kept.forward.rows is keep and kept.obs_cov.space is keep
        np.testing.assert_array_equal(kept.obs.values, hand.obs.values)
        np.testing.assert_array_equal(kept.offset.values, hand.offset.values)
        np.testing.assert_allclose(kept.forward.dense(), hand.forward.dense())
        np.testing.assert_allclose(
            kept.obs_cov.dense(), hand.obs_cov.dense(), atol=1e-15
        )
        np.testing.assert_array_equal(kept.prior.values, problem.prior.values)
        np.testing.assert_array_equal(kept.prior_cov.dense(), problem.prior_cov.dense())
        post, post_hand = fips.solve(kept), fips.solve(hand)
        np.testing.assert_allclose(
            post.estimate.values, post_hand.estimate.values, rtol=1e-10
        )
        np.testing.assert_allclose(post.chi2, post_hand.chi2, rtol=1e-10)

    def test_the_covariance_keeps_its_form(self):
        """A Kronecker covariance stays factors when the kept labels are in its order, and a block-diagonal one stays blocks."""
        problem = make("kronecker")
        space = problem.obs.space
        kron = problem.keep_obs(Space({"co2": space.select("co2")[[1, 2, 5, 8]]}))
        assert isinstance(kron.obs_cov.data, KroneckerOperator)

        problem = make("block")
        block = problem.keep_obs(_keep(problem))
        assert isinstance(block.obs_cov.data, BlockDiagonalOperator)

        sparse = make("sparse")
        assert isinstance(sparse.keep_obs(_keep(sparse)).obs_cov.data, SparseOperator)

    def test_a_block_left_out_whole(self):
        problem = make("block")
        kept = problem.keep_obs(Space({"ch4": OBS.select("ch4")}))
        assert kept.obs.space.names == ("ch4",)
        np.testing.assert_array_equal(kept.obs.values, problem.obs["ch4"].values)

    def test_the_new_problem_has_the_order_of_keep(self):
        problem = make("sparse")
        keep = Space({"ch4": OBS.select("ch4")[::-1]})
        kept = problem.keep_obs(keep)
        np.testing.assert_array_equal(kept.obs.values, problem.obs["ch4"].values[::-1])

    def test_without_an_offset(self):
        problem = make("diagonal", offset=False)
        assert problem.keep_obs(_keep(problem)).offset is None

    def test_keeping_everything_changes_nothing_of_the_solve(self):
        problem = make("block")
        kept = problem.keep_obs(OBS)
        np.testing.assert_allclose(
            fips.solve(kept).estimate.values,
            fips.solve(problem).estimate.values,
            rtol=1e-10,
        )

    def test_screening_with_the_innovations(self):
        """The use the two are for: leave out the observation that disagrees, then solve."""
        problem = make("diagonal")
        values = problem.obs.values.copy()
        values[0] += 50.0
        bad = Problem(
            prior=problem.prior,
            prior_cov=problem.prior_cov,
            obs=Vector(values, OBS),
            obs_cov=problem.obs_cov,
            forward=problem.forward,
            offset=problem.offset,
        )
        d = bad.normalized_innovations().to_series()
        screened = bad.keep_obs(Space.from_index(d[d.abs() < 5].index))
        assert 0 < len(screened.obs) <= M - 1
        assert GRID[0] not in screened.obs.space.select("co2")
        # the same problem as the one built without that observation
        np.testing.assert_allclose(
            fips.solve(screened).estimate.values,
            fips.solve(problem.keep_obs(screened.obs.space)).estimate.values,
        )

    def test_a_forward_model_follows_the_selection(self):
        class Linear(fips.ForwardModel):
            def __init__(self, H: Operator):
                super().__init__(H.rows, H.cols)
                self.H = H

            def __call__(self, x):
                return self.H @ x

            def jacobian(self, x):
                return self.H

        linear = make("block")
        model = Problem(
            prior=linear.prior,
            prior_cov=linear.prior_cov,
            obs=linear.obs,
            obs_cov=linear.obs_cov,
            forward=Linear(linear.forward),
            offset=linear.offset,
        )
        keep = _keep(model)
        kept = model.keep_obs(keep)
        hand = linear.keep_obs(keep)
        assert isinstance(kept.forward, fips.ForwardModel)
        np.testing.assert_allclose(
            kept.predict(kept.prior).values, hand.predict(hand.prior).values
        )
        np.testing.assert_allclose(
            kept.linearized(kept.prior).forward.dense(), hand.forward.dense()
        )
        np.testing.assert_allclose(
            kept.normalized_innovations().values,
            hand.normalized_innovations().values,
            rtol=1e-12,
        )

    def test_a_kept_problem_pickles(self):
        problem = make("block")
        kept = problem.keep_obs(_keep(problem))
        loaded = pickle.loads(pickle.dumps(kept))
        np.testing.assert_array_equal(
            fips.solve(loaded).estimate.values, fips.solve(kept).estimate.values
        )

    def test_errors(self):
        problem = make("diagonal")
        with pytest.raises(TypeError, match="keep must be a Space, got list"):
            problem.keep_obs(["o1"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="blocks not in the space: 'no'"):
            problem.keep_obs(Space({"no": pd.Index(["a"], name="x")}))
        with pytest.raises(ValueError, match="labels of block 'ch4' not in the space"):
            problem.keep_obs(Space({"ch4": pd.Index(["zz"], name="gas_id")}))

    def test_screened_problem_reads_low_on_reduced_chi2(self):
        """Screening removes the large departures, so the screened problem's reduced chi-squared reads low."""
        rng = np.random.default_rng(11)
        m, n = 400, 3
        space = Space({"o": pd.RangeIndex(m, name="i")})
        state = Space({"s": pd.RangeIndex(n, name="j")})
        H = rng.normal(size=(m, n))
        z = H @ rng.normal(size=n) + rng.normal(size=m)
        problem = Problem(
            prior=Vector(np.zeros(n), state),
            prior_cov=Independent().with_std(1.0),
            obs=Vector(z, space),
            obs_cov=Independent().with_std(1.0),
            forward=Operator(H, space, state),
        )
        d = problem.normalized_innovations().to_series()
        screened = problem.keep_obs(Space.from_index(d[d.abs() < 1.0].index))
        assert (
            fips.solve(screened).reduced_chi2 < 0.8 * fips.solve(problem).reduced_chi2
        )
