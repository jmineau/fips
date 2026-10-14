"""
Oracle tests: the posterior against formulas written in this file.

The other estimator tests check properties: shapes, labels, signs, the
direction in which ``gamma`` moves a diagnostic, and ``K`` against its own
equation. The tests here pin the numbers. Each output of
:meth:`fips.InverseProblem.solve` is compared with the closed-form Gaussian
posterior, written below with ``numpy.linalg`` and nothing from
``fips.estimators``. They are parametrized over the estimator registry, so an
estimator registered later is held to them too.

Also here: a permutation test (the labeled posterior does not depend on the
order in which any input is given) and dot tests (the adjoint of each linear
map fips applies is its transpose).
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from fips import (
    Block,
    CovarianceMatrix,
    ForwardOperator,
    InverseProblem,
    Matrix,
    MatrixBlock,
    Vector,
    available_estimators,
    convolve,
)
from fips.aggregators import ObsAggregator
from fips.covariance import KroneckerOperator
from fips.estimators import ESTIMATOR_REGISTRY

ESTIMATORS = available_estimators()

RTOL = 1e-10
"""
Relative tolerance of every comparison with the closed form.

An absolute floor goes with it: ``RTOL`` times the size of the terms the
quantity is computed from (see :func:`assert_close`). Roundoff in a sum of
products is a multiple of machine epsilon times the sum of the absolute values
of its terms, not of the result, so an entry that is near zero by cancellation
(an off-diagonal of the posterior covariance, or ``x_hat - x_0`` where the data
barely move the state) can only be held to ``1e-16`` of those terms. The test
matrices are well conditioned (condition numbers of order 10), so both sides
are good to about ``1e-14`` of that scale, four orders inside the tolerance,
while a sign error or a transposed matrix is wrong at order one.
"""

SEEDS = range(20)
SIZES = [(n, m) for n in (1, 3, 7) for m in (1, 2, 10)]


# ---------------------------------------------------------------------------
# The closed form, written independently of fips.estimators
# ---------------------------------------------------------------------------


@dataclass
class ClosedForm:
    """The Gaussian posterior and its diagnostics, from the formulas."""

    x_hat: np.ndarray
    S_hat_cov: np.ndarray
    S_hat_info: np.ndarray
    K: np.ndarray
    A: np.ndarray
    DOFS: float
    y_hat: np.ndarray
    y_0: np.ndarray
    S_y0: np.ndarray
    reduced_chi2: float
    scale: dict[str, float]


def closed_form(z, x_0, H, S_0, S_z, c) -> ClosedForm:
    """
    Return the posterior of ``z = H x + c + e`` with ``x ~ N(x_0, S_0)``, ``e ~ N(0, S_z)``.

    Written with explicit inverses, the formulas as they are printed, so that
    the reference does not share an algorithm with the estimator (which uses
    Cholesky solves).
    """
    inv = np.linalg.inv
    G = H @ S_0 @ H.T + S_z
    K = S_0 @ H.T @ inv(G)
    r_0 = z - H @ x_0 - c
    x_hat = x_0 + S_0 @ H.T @ inv(G) @ r_0
    S_hat_cov = S_0 - S_0 @ H.T @ inv(G) @ H @ S_0
    S_hat_info = inv(H.T @ inv(S_z) @ H + inv(S_0))
    A = K @ H
    r = z - H @ x_hat - c
    d = x_hat - x_0
    chi2 = (r @ inv(S_z) @ r + d @ inv(S_0) @ d) / len(z)

    a = np.abs
    scale = {
        "x_hat": max(a(x_0).max(), (a(K) @ a(r_0)).max()),
        "S_hat": max(a(S_0).max(), (a(K) @ a(H @ S_0)).max()),
        "K": (a(S_0 @ H.T) @ a(inv(G))).max(),
        "A": (a(K) @ a(H)).max(),
        "DOFS": np.trace(a(K) @ a(H)),
        "y": (a(H) @ a(x_hat)).max() + a(c).max(),
        "S_y0": (a(H) @ a(S_0) @ a(H.T)).max(),
        "chi2": chi2,
    }
    return ClosedForm(
        x_hat=x_hat,
        S_hat_cov=S_hat_cov,
        S_hat_info=S_hat_info,
        K=K,
        A=A,
        DOFS=float(np.trace(A)),
        y_hat=H @ x_hat + c,
        y_0=H @ x_0 + c,
        S_y0=H @ S_0 @ H.T,
        reduced_chi2=float(chi2),
        scale=scale,
    )


def assert_close(actual, expected, scale: float, what: str) -> None:
    """Compare at ``RTOL``, with an absolute floor of ``RTOL * scale``."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=RTOL,
        atol=RTOL * scale,
        err_msg=what,
    )


# ---------------------------------------------------------------------------
# Random problems
# ---------------------------------------------------------------------------


def random_spd(rng: np.random.Generator, k: int) -> np.ndarray:
    """Return a dense, well-conditioned SPD matrix with a random scale."""
    B = rng.standard_normal((k, k))
    return (B @ B.T + k * np.eye(k)) * 10 ** rng.uniform(-1, 1)


@dataclass
class Case:
    """A random single-block problem: the arrays and the labels that go with them."""

    z: np.ndarray
    x_0: np.ndarray
    H: np.ndarray
    S_0: np.ndarray
    S_z: np.ndarray
    c: np.ndarray
    constant: float | pd.Series
    state_index: pd.MultiIndex
    obs_index: pd.MultiIndex

    @classmethod
    def random(cls, seed: int, n: int, m: int, constant: str = "vector") -> "Case":
        """
        Draw a problem with ``n`` states and ``m`` observations.

        Labels are strings that are not digits, in a random order, so neither
        ``round_index`` nor the numeric conversion of labels changes them.
        """
        rng = np.random.default_rng(seed)
        state_index = pd.MultiIndex.from_arrays(
            [["flux"] * n, [f"c{k}" for k in rng.permutation(n)]],
            names=["block", "cell"],
        )
        obs_index = pd.MultiIndex.from_arrays(
            [["conc"] * m, [f"o{k}" for k in rng.permutation(m)]],
            names=["block", "obs_id"],
        )
        x_0 = rng.normal(5.0, 2.0, n)
        H = rng.standard_normal((m, n))
        S_0 = random_spd(rng, n)
        S_z = random_spd(rng, m)
        z = H @ x_0 + rng.standard_normal(m) * 3.0
        if constant == "scalar":
            value = float(rng.normal(1.0, 0.5))
            c, const = np.full(m, value), value
        else:
            c = rng.normal(1.0, 0.5, m)
            const = pd.Series(c, index=obs_index)
        return cls(z, x_0, H, S_0, S_z, c, const, state_index, obs_index)

    def problem(self, **inputs) -> InverseProblem:
        """Build the labeled problem; keyword arguments replace an input."""
        args = {
            "obs": pd.Series(self.z, index=self.obs_index),
            "prior": pd.Series(self.x_0, index=self.state_index),
            "forward_operator": pd.DataFrame(
                self.H, index=self.obs_index, columns=self.state_index
            ),
            "prior_error": pd.DataFrame(
                self.S_0, index=self.state_index, columns=self.state_index
            ),
            "modeldata_mismatch": pd.DataFrame(
                self.S_z, index=self.obs_index, columns=self.obs_index
            ),
            "constant": self.constant,
        }
        args.update(inputs)
        return InverseProblem(**args)

    def sparse_inputs(self) -> dict[str, Matrix]:
        """Return ``H``, ``S_0`` and ``S_z`` stored sparse, as inputs of :meth:`problem`."""
        s, o = self.state_index, self.obs_index
        return {
            "forward_operator": ForwardOperator(
                pd.DataFrame(self.H, index=o, columns=s), sparse=True
            ),
            "prior_error": CovarianceMatrix(
                pd.DataFrame(self.S_0, index=s, columns=s), sparse=True
            ),
            "modeldata_mismatch": CovarianceMatrix(
                pd.DataFrame(self.S_z, index=o, columns=o), sparse=True
            ),
        }


# ---------------------------------------------------------------------------
# Reading the labeled outputs
# ---------------------------------------------------------------------------


def vector_on(v: Vector, index: pd.Index, what: str) -> np.ndarray:
    """Return the values of `v` after checking that it is labeled by `index`, in order."""
    assert isinstance(v, Vector), what
    assert v.index.equals(index), f"{what}: wrong index"
    assert list(v.index.names) == list(index.names), f"{what}: wrong level names"
    return v.values


def matrix_on(M: Matrix, index: pd.Index, columns: pd.Index, what: str) -> np.ndarray:
    """Return the values of `M` after checking its labels on both axes, in order."""
    assert isinstance(M, Matrix), what
    assert M.index.equals(index), f"{what}: wrong row index"
    assert M.columns.equals(columns), f"{what}: wrong column index"
    assert list(M.index.names) == list(index.names), f"{what}: wrong row level names"
    assert list(M.columns.names) == list(columns.names), (
        f"{what}: wrong column level names"
    )
    return M.to_dense().values


def check_posterior(
    problem: InverseProblem,
    expected: ClosedForm,
    state_index: pd.Index,
    obs_index: pd.Index,
    what: str,
) -> None:
    """Compare every output of a solved problem with the closed form, by label."""
    s, o, sc = state_index, obs_index, expected.scale

    def at(name):
        return f"{what}: {name}"

    assert_close(
        vector_on(problem.posterior, s, at("posterior")),
        expected.x_hat,
        sc["x_hat"],
        at("posterior"),
    )
    S_hat = matrix_on(problem.posterior_error, s, s, at("posterior_error"))
    assert_close(S_hat, expected.S_hat_cov, sc["S_hat"], at("S_hat, covariance form"))
    assert_close(S_hat, expected.S_hat_info, sc["S_hat"], at("S_hat, information form"))
    assert_close(
        matrix_on(problem.kalman_gain, s, o, at("kalman_gain")),
        expected.K,
        sc["K"],
        at("kalman_gain"),
    )
    assert_close(
        matrix_on(problem.averaging_kernel, s, s, at("averaging_kernel")),
        expected.A,
        sc["A"],
        at("averaging_kernel"),
    )
    assert_close(problem.estimator.DOFS, expected.DOFS, sc["DOFS"], at("DOFS"))
    assert_close(
        vector_on(problem.posterior_obs, o, at("posterior_obs")),
        expected.y_hat,
        sc["y"],
        at("posterior_obs"),
    )
    assert_close(
        vector_on(problem.prior_obs, o, at("prior_obs")),
        expected.y_0,
        sc["y"],
        at("prior_obs"),
    )
    assert_close(
        matrix_on(problem.prior_obs_error, o, o, at("prior_obs_error")),
        expected.S_y0,
        sc["S_y0"],
        at("prior_obs_error"),
    )
    assert_close(
        problem.estimator.reduced_chi2,
        expected.reduced_chi2,
        sc["chi2"],
        at("reduced_chi2"),
    )


def takes_gamma(estimator: str) -> bool:
    """Return True if the registered estimator takes a ``gamma`` argument."""
    return "gamma" in inspect.signature(ESTIMATOR_REGISTRY[estimator]).parameters


# ---------------------------------------------------------------------------
# 1. The closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("n, m", SIZES)
def test_closed_form(estimator, n, m, sparse):
    """
    The posterior equals the closed form, in both covariance forms, for 20 seeds.

    The constant is a scalar for the even seeds and a vector for the odd ones,
    so each size sees both forms without solving every problem twice. With
    ``sparse``, ``H``, ``S_0`` and ``S_z`` are stored sparse (#79).
    """
    for seed in SEEDS:
        case = Case.random(seed, n, m, "scalar" if seed % 2 == 0 else "vector")
        expected = closed_form(case.z, case.x_0, case.H, case.S_0, case.S_z, case.c)
        # The reference itself: the two forms of S_hat, and A = K H, agree
        assert_close(
            expected.S_hat_cov,
            expected.S_hat_info,
            expected.scale["S_hat"],
            f"seed {seed}: the covariance and information forms",
        )
        problem = case.problem(**case.sparse_inputs() if sparse else {})
        check_posterior(
            problem.solve(estimator),
            expected,
            case.state_index,
            case.obs_index,
            f"seed {seed}",
        )


@pytest.mark.parametrize("estimator", [e for e in ESTIMATORS if takes_gamma(e)])
@pytest.mark.parametrize("gamma", [0.25, 3.0])
@pytest.mark.parametrize("n, m", [(3, 10), (7, 2)])
def test_gamma_scales_mismatch(estimator, gamma, n, m):
    """``gamma`` gives the closed form with the model-data mismatch divided by ``gamma``."""
    for seed in range(5):
        case = Case.random(seed, n, m)
        expected = closed_form(
            case.z, case.x_0, case.H, case.S_0, case.S_z / gamma, case.c
        )
        problem = case.problem().solve(estimator, gamma=gamma)
        check_posterior(
            problem, expected, case.state_index, case.obs_index, f"seed {seed}"
        )


def _diagonal_inputs(case: Case, form: str, v_0: np.ndarray, v_z: np.ndarray):
    """Return the prior error and mismatch given as variances, in one of several forms."""
    s, o = case.state_index, case.obs_index
    if form == "series":
        return pd.Series(v_0, index=s), pd.Series(v_z, index=o)
    if form == "vector":
        return Vector(pd.Series(v_0, index=s)), Vector(pd.Series(v_z, index=o))
    if form == "scalar":
        return float(v_0[0]), float(v_z[0])
    if form == "diagonal":
        return (
            CovarianceMatrix.diagonal(v_0, index=s),
            CovarianceMatrix.diagonal(v_z, index=o),
        )
    if form == "sparse":
        return (
            CovarianceMatrix.diagonal(v_0, index=s, sparse=True),
            CovarianceMatrix.diagonal(v_z, index=o, sparse=True),
        )
    raise ValueError(form)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize(
    "form",
    [
        "series",
        "vector",
        "scalar",
        "diagonal",
        "sparse",
    ],
)
def test_diagonal_given_as_variances(estimator, form):
    """A diagonal covariance given as its variances solves as the full matrix does."""
    for seed in range(5):
        case = Case.random(seed, 7, 10)
        rng = np.random.default_rng(seed)
        if form == "scalar":
            v_0, v_z = np.full(7, 2.5), np.full(10, 0.4)
        else:
            v_0, v_z = rng.uniform(0.5, 4.0, 7), rng.uniform(0.1, 1.0, 10)
        S_0, S_z = np.diag(v_0), np.diag(v_z)
        expected = closed_form(case.z, case.x_0, case.H, S_0, S_z, case.c)
        prior_error, mismatch = _diagonal_inputs(case, form, v_0, v_z)
        problem = case.problem(
            prior_error=prior_error, modeldata_mismatch=mismatch
        ).solve(estimator)
        check_posterior(
            problem, expected, case.state_index, case.obs_index, f"seed {seed}"
        )


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("rows", ["product", "permuted", "ragged"])
def test_kronecker_prior_error(estimator, rows):
    """
    A prior error kept as Kronecker factors gives the closed form of its matrix.

    The matrix for the closed form is ``np.kron`` of the factors, with the rows
    picked and scaled here: the whole product in order, a permutation of it, or
    9 of its 12 rows.
    """
    sizes = (3, 4)
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        chosen = {
            "product": np.arange(12),
            "permuted": rng.permutation(12),
            "ragged": rng.permutation(12)[:9],
        }[rows]
        case = Case.random(seed, len(chosen), 7)
        factors = [random_spd(rng, k) for k in sizes]
        std = rng.uniform(0.5, 2.0, len(chosen))
        S_0 = std[:, None] * np.kron(*factors)[np.ix_(chosen, chosen)] * std[None, :]
        expected = closed_form(case.z, case.x_0, case.H, S_0, case.S_z, case.c)
        op = KroneckerOperator(
            factors, std=std, positions=np.unravel_index(chosen, sizes)
        )
        problem = case.problem(
            prior_error=CovarianceMatrix.from_operator(op, case.state_index)
        ).solve(estimator)
        assert isinstance(problem.estimator._S_0_op, KroneckerOperator)
        check_posterior(
            problem, expected, case.state_index, case.obs_index, f"seed {seed}"
        )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_sparse_forward_operator(estimator):
    """A sparse forward operator gives the closed form, as the dense one does."""
    for seed in range(5):
        case = Case.random(seed, 7, 10)
        case.H[np.abs(case.H) < 0.8] = 0.0  # make it sparse
        expected = closed_form(case.z, case.x_0, case.H, case.S_0, case.S_z, case.c)
        H = ForwardOperator(
            pd.DataFrame(case.H, index=case.obs_index, columns=case.state_index),
            sparse=True,
        )
        assert H.is_sparse
        problem = case.problem(forward_operator=H).solve(estimator)
        check_posterior(
            problem, expected, case.state_index, case.obs_index, f"seed {seed}"
        )


# ---------------------------------------------------------------------------
# 2. Several blocks
# ---------------------------------------------------------------------------


@dataclass
class MultiBlockCase:
    """
    Two state blocks and two observation blocks, each with its own index levels.

    The arrays are the assembled problem, with the blocks in the order
    ``flux, bias`` (state) and ``co2, ch4`` (observations).
    """

    state: dict[str, pd.Index]
    obs: dict[str, pd.Index]
    z: np.ndarray
    x_0: np.ndarray
    H: np.ndarray
    S_0: np.ndarray
    S_z: np.ndarray
    c: np.ndarray

    @classmethod
    def random(cls, seed: int) -> "MultiBlockCase":
        """Draw the arrays; the ``(co2, bias)`` block of ``H`` is left out (zero)."""
        rng = np.random.default_rng(seed)
        state = {
            "flux": pd.MultiIndex.from_product([[0, 1, 2], [0, 1]], names=["x", "y"]),
            "bias": pd.Index(["north", "south"], name="site"),
        }
        obs = {
            "co2": pd.Index([f"a{k}" for k in rng.permutation(5)], name="obs_id"),
            "ch4": pd.MultiIndex.from_product(
                [["north", "south"], [6, 18]], names=["site", "hour"]
            ),
        }
        n, m = 8, 9
        H = rng.standard_normal((m, n))
        H[:5, 6:] = 0.0  # no (co2, bias) block
        x_0 = rng.normal(5.0, 2.0, n)
        z = H @ x_0 + rng.standard_normal(m)
        c = rng.normal(1.0, 0.5, m)
        S_0, S_z = random_spd(rng, n), random_spd(rng, m)
        return cls(state, obs, z, x_0, H, S_0, S_z, c)

    @staticmethod
    def _slices(blocks: dict[str, pd.Index]) -> dict[str, slice]:
        """Return where each block sits in the assembled arrays."""
        out, start = {}, 0
        for name, idx in blocks.items():
            out[name] = slice(start, start + len(idx))
            start += len(idx)
        return out

    @property
    def state_slices(self) -> dict[str, slice]:
        """Where each state block sits in the assembled state arrays."""
        return self._slices(self.state)

    @property
    def obs_slices(self) -> dict[str, slice]:
        """Where each observation block sits in the assembled observation arrays."""
        return self._slices(self.obs)

    def problem(self) -> InverseProblem:
        """Build the problem from Blocks and MatrixBlocks."""
        ss, os_ = self.state_slices, self.obs_slices

        def vector(values, blocks, slices):
            return Vector(
                [
                    Block(pd.Series(values[slices[b]], index=idx), name=b)
                    for b, idx in blocks.items()
                ]
            )

        def blocks_2d(values, rows, cols, rs, cs, skip=()):
            return [
                MatrixBlock(
                    pd.DataFrame(values[rs[r], cs[c]], index=ri, columns=ci),
                    row_block=r,
                    col_block=c,
                )
                for r, ri in rows.items()
                for c, ci in cols.items()
                if (r, c) not in skip
            ]

        return InverseProblem(
            obs=vector(self.z, self.obs, os_),
            prior=vector(self.x_0, self.state, ss),
            forward_operator=blocks_2d(
                self.H, self.obs, self.state, os_, ss, skip={("co2", "bias")}
            ),
            prior_error=blocks_2d(self.S_0, self.state, self.state, ss, ss),
            modeldata_mismatch=blocks_2d(self.S_z, self.obs, self.obs, os_, os_),
            constant=vector(self.c, self.obs, os_),
        )


def _block_values(structure, key, index, columns=None) -> np.ndarray:
    """Read one block of a solved output by its name(s), checking its labels."""
    if columns is None:
        series = structure[key]
        assert series.index.equals(index), f"{key}: wrong index"
        assert list(series.index.names) == list(index.names), f"{key}: level names"
        return series.to_numpy()
    frame = structure[key]
    assert frame.index.equals(index), f"{key}: wrong row index"
    assert frame.columns.equals(columns), f"{key}: wrong column index"
    assert list(frame.index.names) == list(index.names), f"{key}: row level names"
    assert list(frame.columns.names) == list(columns.names), (
        f"{key}: column level names"
    )
    return frame.to_numpy()


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("seed", range(5))
def test_multi_block(estimator, seed):
    """Block by block, read back by label, the posterior equals the closed form."""
    case = MultiBlockCase.random(seed)
    expected = closed_form(case.z, case.x_0, case.H, case.S_0, case.S_z, case.c)
    problem = case.problem().solve(estimator)
    sc = expected.scale
    ss, os_ = case.state_slices, case.obs_slices

    for b, idx in case.state.items():
        what = f"seed {seed}: posterior[{b}]"
        got = _block_values(problem.posterior, b, idx)
        assert_close(got, expected.x_hat[ss[b]], sc["x_hat"], what)
        for b2, idx2 in case.state.items():
            what = f"seed {seed}: [{b}, {b2}]"
            got = _block_values(problem.posterior_error, (b, b2), idx, idx2)
            assert_close(got, expected.S_hat_cov[ss[b], ss[b2]], sc["S_hat"], what)
            got = _block_values(problem.averaging_kernel, (b, b2), idx, idx2)
            assert_close(got, expected.A[ss[b], ss[b2]], sc["A"], what)
        for o, oidx in case.obs.items():
            what = f"seed {seed}: kalman_gain[{b}, {o}]"
            got = _block_values(problem.kalman_gain, (b, o), idx, oidx)
            assert_close(got, expected.K[ss[b], os_[o]], sc["K"], what)

    for o, oidx in case.obs.items():
        what = f"seed {seed}: posterior_obs[{o}]"
        got = _block_values(problem.posterior_obs, o, oidx)
        assert_close(got, expected.y_hat[os_[o]], sc["y"], what)
        for o2, oidx2 in case.obs.items():
            what = f"seed {seed}: prior_obs_error[{o}, {o2}]"
            got = _block_values(problem.prior_obs_error, (o, o2), oidx, oidx2)
            assert_close(got, expected.S_y0[os_[o], os_[o2]], sc["S_y0"], what)

    assert_close(problem.estimator.DOFS, expected.DOFS, sc["DOFS"], "DOFS")
    assert_close(
        problem.estimator.reduced_chi2, expected.reduced_chi2, sc["chi2"], "chi2"
    )


# ---------------------------------------------------------------------------
# 3. Permutation
# ---------------------------------------------------------------------------


def _shuffled(obj, rng: np.random.Generator):
    """Return a Series in a random row order, or a DataFrame with rows and columns each in their own."""
    if isinstance(obj, pd.Series):
        return obj.iloc[rng.permutation(len(obj))]
    return obj.iloc[rng.permutation(obj.shape[0]), rng.permutation(obj.shape[1])]


def _aligned(structure, index, columns=None) -> np.ndarray:
    """Return a structure's values in the order of `index` (and `columns`), by label."""
    data = structure.data
    assert data.index.sort_values().equals(index.sort_values()), "row labels differ"
    if columns is None:
        return data.reindex(index).to_numpy()
    assert data.columns.sort_values().equals(columns.sort_values()), (
        "column labels differ"
    )
    return data.reindex(index=index, columns=columns).to_numpy()


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("seed", range(5))
def test_permutation(estimator, seed):
    """Each input in its own random order gives the same labeled posterior."""
    case = MultiBlockCase.random(seed)
    ref = case.problem().solve(estimator)
    rng = np.random.default_rng(1000 + seed)

    # The assembled inputs, with block levels, each shuffled on its own
    shuffled = InverseProblem(
        obs=_shuffled(ref.obs.data, rng),
        prior=_shuffled(ref.prior.data, rng),
        forward_operator=_shuffled(ref.forward_operator.data, rng),
        prior_error=_shuffled(ref.prior_error.data, rng),
        modeldata_mismatch=_shuffled(ref.modeldata_mismatch.data, rng),
        constant=_shuffled(ref.constant.data, rng),
    ).solve(estimator)

    s, o = ref.state_index, ref.obs_index
    expected = closed_form(case.z, case.x_0, case.H, case.S_0, case.S_z, case.c)
    sc = expected.scale
    pairs = [
        ("posterior", s, None, "x_hat"),
        ("posterior_error", s, s, "S_hat"),
        ("averaging_kernel", s, s, "A"),
        ("kalman_gain", s, o, "K"),
        ("posterior_obs", o, None, "y"),
        ("prior_obs_error", o, o, "S_y0"),
    ]
    for name, rows, cols, key in pairs:
        assert_close(
            _aligned(getattr(shuffled, name), rows, cols),
            getattr(ref, name).to_dense().values
            if cols is not None
            else getattr(ref, name).values,
            sc[key],
            f"seed {seed}: {name}",
        )
    for diag in ["DOFS", "reduced_chi2", "R2", "RMSE", "uncertainty_reduction"]:
        got = getattr(shuffled.estimator, diag)
        want = getattr(ref.estimator, diag)
        assert_close(got, want, abs(want), f"seed {seed}: {diag}")


# ---------------------------------------------------------------------------
# 4. Adjoint (dot) tests
# ---------------------------------------------------------------------------


def dot_test(
    forward: Callable[[np.ndarray], np.ndarray],
    adjoint: Callable[[np.ndarray], np.ndarray],
    n: int,
    m: int,
    rng: np.random.Generator,
    rtol: float = 1e-12,
    repeats: int = 3,
) -> None:
    """
    Check that `adjoint` is the adjoint of `forward`, a linear map from R^n to R^m.

    For random ``x`` and ``y``, ``<A x, y>`` must equal ``<x, A^T y>`` to within
    `rtol` of ``|A x| |y| + |x| |A^T y|``, the size of the terms the two inner
    products are summed from.
    """
    for _ in range(repeats):
        x, y = rng.standard_normal(n), rng.standard_normal(m)
        Ax, ATy = np.asarray(forward(x)), np.asarray(adjoint(y))
        assert Ax.shape == (m,), f"forward returned shape {Ax.shape}, not ({m},)"
        assert ATy.shape == (n,), f"adjoint returned shape {ATy.shape}, not ({n},)"
        lhs, rhs = Ax @ y, x @ ATy
        bound = np.linalg.norm(Ax) * np.linalg.norm(y)
        bound += np.linalg.norm(x) * np.linalg.norm(ATy)
        assert abs(lhs - rhs) <= rtol * bound, (
            f"<Ax, y> = {lhs!r} but <x, A^T y> = {rhs!r}"
        )


def test_dot_test_catches_a_wrong_adjoint():
    """The dot test fails for a map paired with something that is not its adjoint."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 3))
    dot_test(lambda x: A @ x, lambda y: A.T @ y, 3, 4, rng)
    with pytest.raises(AssertionError):
        dot_test(lambda x: A @ x, lambda y: 1.01 * A.T @ y, 3, 4, rng)
    B = rng.standard_normal((3, 3))
    with pytest.raises(AssertionError):
        dot_test(lambda x: B @ x, lambda y: B @ y, 3, 3, rng)


def _labeled_map(frame: pd.DataFrame, rng: np.random.Generator):
    """
    Return the map ``x -> frame @ x`` through :func:`fips.convolve`.

    ``x`` is in the order of ``frame.columns`` and the result in the order of
    ``frame.index``; the state is handed to ``convolve`` in a random order, so
    the product goes through the label alignment.
    """

    def apply(x: np.ndarray) -> np.ndarray:
        perm = rng.permutation(len(x))
        state = pd.Series(x[perm], index=frame.columns[perm], name="x")
        y = convolve(state, frame)
        assert y.index.equals(frame.index)
        return y.to_numpy()

    return apply


def _matrix_structure(kind: str, sparse: bool, rng: np.random.Generator):
    """Return a two-block structure of class `kind`, dense or sparse."""
    flux = pd.MultiIndex.from_product([[0, 1, 2], [0, 1]], names=["x", "y"])
    bias = pd.Index(["north", "south"], name="site")
    co2 = pd.Index([f"a{k}" for k in range(5)], name="obs_id")
    ch4 = pd.Index(["north", "south"], name="site")

    def block(rows, cols, r, c):
        values = rng.standard_normal((len(rows), len(cols)))
        values[np.abs(values) < 0.7] = 0.0
        frame = pd.DataFrame(values, index=rows, columns=cols)
        return MatrixBlock(frame, row_block=r, col_block=c)

    if kind == "CovarianceMatrix":
        blocks = [
            block(flux, flux, "flux", "flux"),
            block(flux, bias, "flux", "bias"),
            block(bias, bias, "bias", "bias"),
        ]
    else:
        blocks = [
            block(co2, flux, "co2", "flux"),
            block(co2, bias, "co2", "bias"),
            block(ch4, flux, "ch4", "flux"),
        ]
    cls = {
        "Matrix": Matrix,
        "ForwardOperator": ForwardOperator,
        "CovarianceMatrix": CovarianceMatrix,
    }[kind]
    return cls(blocks, sparse=sparse)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("kind", ["Matrix", "ForwardOperator", "CovarianceMatrix"])
def test_dot_test_structures(kind, sparse):
    """``convolve`` with a structure and with its transpose are adjoint, by label."""
    rng = np.random.default_rng(1)
    M = _matrix_structure(kind, sparse, rng)
    assert M.is_sparse == sparse
    frame = M.to_frame()
    m, n = frame.shape
    dot_test(_labeled_map(frame, rng), _labeled_map(frame.T, rng), n, m, rng)
    # And the map is the matrix the structure holds
    x = rng.standard_normal(n)
    np.testing.assert_allclose(
        _labeled_map(frame, rng)(x), M.to_dense().values @ x, rtol=1e-12
    )


def _obs_index(blocks: list[str], seed: int) -> pd.MultiIndex:
    """Six-hourly observations at two sites over three days, in a random order."""
    times = pd.date_range("2024-01-01", periods=12, freq="6h")
    idx = pd.MultiIndex.from_product(
        [blocks, ["north", "south"], times], names=["block", "site", "time"]
    )
    return idx[np.random.default_rng(seed).permutation(len(idx))]


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize(
    "aggregator, blocks",
    [
        (ObsAggregator(level="time", freq="D"), ["conc"]),
        (ObsAggregator(by="site", func="sum"), ["conc"]),
        (ObsAggregator(level="time", freq="D", blocks="co2"), ["co2", "ch4"]),
    ],
    ids=["daily-mean", "site-sum", "one-block"],
)
def test_dot_test_obs_aggregator(aggregator, blocks, sparse):
    """
    The aggregation of the observations and of the forward operator are one map ``W``.

    ``W z`` comes from the aggregated observations. ``W`` itself comes from
    aggregating the identity as a forward operator, so ``W^T y`` is that
    frame's transpose applied to ``y`` by label. The aggregated mismatch is
    ``W S_z W^T`` with the same ``W``.
    """
    rng = np.random.default_rng(2)
    obs_index = _obs_index(blocks, seed=3)
    m = len(obs_index)
    # Each input in its own order, so W is right only if apply aligns them
    identity = Matrix(
        _shuffled(pd.DataFrame(np.eye(m), index=obs_index, columns=obs_index), rng),
        sparse=sparse,
    )
    S_z = random_spd(rng, m)
    S_z_frame = pd.DataFrame(S_z, index=obs_index, columns=obs_index)
    _, W, S_z_agg, _ = aggregator.apply(
        obs=_shuffled(pd.Series(np.zeros(m), index=obs_index, name="z"), rng),
        forward_operator=identity,
        modeldata_mismatch=_shuffled(S_z_frame, rng),
    )
    W_frame = W.to_dense().to_frame()
    agg_index = W_frame.index
    k = len(agg_index)
    assert k < m

    def forward(x):
        perm = rng.permutation(m)
        z = pd.Series(x[perm], index=obs_index[perm], name="z")
        z_agg, _, _, _ = aggregator.apply(
            obs=z, forward_operator=identity, modeldata_mismatch=S_z_frame
        )
        assert z_agg.index.sort_values().equals(agg_index.sort_values())
        return z_agg.reindex(agg_index).to_numpy()

    def adjoint(y):
        perm = rng.permutation(k)
        y_agg = pd.Series(y[perm], index=agg_index[perm])
        return W_frame.T.dot(y_agg).reindex(obs_index).to_numpy()

    dot_test(forward, adjoint, m, k, rng)

    W_values = W_frame.reindex(columns=obs_index).to_numpy()
    np.testing.assert_allclose(
        S_z_agg.reindex(index=agg_index, columns=agg_index).to_numpy(),
        W_values @ S_z @ W_values.T,
        rtol=1e-12,
        atol=1e-12 * np.abs(S_z).max(),
    )
