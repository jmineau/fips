"""Tests for fips.operator: each form against its dense equivalent, and the estimator on every form."""

import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.linalg import LinAlgError
from scipy.sparse.linalg import LinearOperator

from fips import CovarianceMatrix, ForwardOperator, InverseProblem, Matrix
from fips.estimators import BayesianSolver
from fips.operator import (
    DenseOperator,
    DiagonalOperator,
    Operator,
    SparseOperator,
    as_operator,
    dottest,
)

RTOL = 1e-12


def spd_dense(rng, n):
    """A well-conditioned dense SPD matrix."""
    B = rng.standard_normal((n, n))
    return B @ B.T + n * np.eye(n)


def spd_banded(rng, n):
    """A sparse, diagonally dominant (so SPD) tridiagonal matrix, as a dense array."""
    off = rng.uniform(-1, 1, n - 1)
    return np.diag(rng.uniform(3, 4, n)) + np.diag(off, 1) + np.diag(off, -1)


def sparse_random(rng, m, n, density):
    """A random dense array with most entries zero."""
    A = rng.standard_normal((m, n))
    A[rng.random((m, n)) > density] = 0.0
    return A


def make(form: str, A: np.ndarray) -> Operator:
    """Return `A` (dense) as an operator of the given form."""
    if form == "dense":
        return DenseOperator(A)
    if form == "sparse":
        return SparseOperator(sp.csr_matrix(A))
    assert np.array_equal(A, np.diag(np.diag(A)))
    return DiagonalOperator(np.diag(A))


FORMS = ["dense", "sparse", "diagonal"]
KIND = {"dense": DenseOperator, "sparse": SparseOperator, "diagonal": DiagonalOperator}


def square(form, rng, n=6):
    """A square SPD matrix that the form can hold."""
    if form == "diagonal":
        return np.diag(rng.uniform(0.5, 2.0, n))
    return spd_banded(rng, n) if form == "sparse" else spd_dense(rng, n)


# ---------------------------------------------------------------------------
# Each form against its dense equivalent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
class TestEachForm:
    """matvec, rmatvec, matmat, transpose, diagonal, trace and solve."""

    def test_products_with_arrays(self, form):
        """``@``, ``matvec``, ``rmatvec``, ``matmat`` and ``rmatmat`` match the array."""
        rng = np.random.default_rng(0)
        A = square(form, rng)
        op = make(form, A)
        x, X = rng.standard_normal(6), rng.standard_normal((6, 3))
        assert isinstance(op, LinearOperator)
        assert op.shape == (6, 6) and op.dtype == np.float64
        np.testing.assert_allclose(op.matvec(x), A @ x, rtol=RTOL)
        np.testing.assert_allclose(op.rmatvec(x), A.T @ x, rtol=RTOL)
        np.testing.assert_allclose(op.matmat(X), A @ X, rtol=RTOL)
        np.testing.assert_allclose(op.rmatmat(X), A.T @ X, rtol=RTOL)
        for result, expected in [
            (op @ x, A @ x),
            (op @ X, A @ X),
            (x @ op, x @ A),
            (X.T @ op, X.T @ A),
        ]:
            assert type(result) is np.ndarray
            np.testing.assert_allclose(result, expected, rtol=RTOL)

    def test_transpose_dense_diagonal_trace(self, form):
        """``.T`` keeps the form; ``dense``, ``diagonal``, ``trace`` and ``np.asarray`` match."""
        rng = np.random.default_rng(1)
        A = square(form, rng)
        A[0, 1] += 0.5 if form != "diagonal" else 0.0  # not symmetric
        op = make(form, A)
        assert isinstance(op.T, KIND[form])
        np.testing.assert_array_equal(op.T.dense(), A.T)
        np.testing.assert_array_equal(op.dense(), A)
        np.testing.assert_array_equal(np.asarray(op), A)
        np.testing.assert_array_equal(op.diagonal(), np.diag(A))
        assert op.trace() == pytest.approx(np.trace(A), rel=RTOL)
        assert dottest(op) and dottest(op.T)

    def test_solve(self, form):
        """``solve`` gives ``A^-1 B`` for a vector, a matrix and an operator."""
        rng = np.random.default_rng(3)
        A = square(form, rng)
        op = make(form, A)
        b, B = rng.standard_normal(6), rng.standard_normal((6, 4))
        np.testing.assert_allclose(op.solve(b), np.linalg.solve(A, b), rtol=1e-10)
        np.testing.assert_allclose(op.solve(B), np.linalg.solve(A, B), rtol=1e-10)
        Bop = SparseOperator(sp.csr_matrix(B))
        np.testing.assert_allclose(op.solve(Bop), np.linalg.solve(A, B), rtol=1e-10)

    def test_scalar_arithmetic_keeps_form(self, form):
        """``*``, ``/`` and negation by a scalar keep the form."""
        rng = np.random.default_rng(4)
        A = square(form, rng)
        op = make(form, A)
        for result, expected in [
            (op * 2.5, A * 2.5),
            (2.5 * op, 2.5 * A),
            (op / 4.0, A / 4.0),
            (-op, -A),
        ]:
            assert isinstance(result, KIND[form])
            np.testing.assert_array_equal(result.dense(), expected)

    def test_zero_dim_array_is_a_scalar(self, form):
        """A 0-d array (``gamma=np.array(0.5)``) scales as the scalar it holds."""
        A = square(form, np.random.default_rng(4))
        op = make(form, A)
        g = np.array(0.5)
        for result, expected in [
            (op * g, A * 0.5),
            (g * op, 0.5 * A),
            (op / g, A / 0.5),
            (op.dot(g), A * 0.5),
        ]:
            assert isinstance(result, KIND[form])
            np.testing.assert_array_equal(result.dense(), expected)
        with pytest.raises(ValueError, match="Scalar operands"):
            op.__rmatmul__(g)

    def test_array_copies_when_asked(self, form):
        """``np.array(op)`` never shares memory with the operator; ``np.asarray`` may."""
        A = square(form, np.random.default_rng(7))
        op = make(form, A)
        copy = np.array(op)
        copy[0, 0] = 99.0
        np.testing.assert_array_equal(op.dense(), A)
        assert not np.shares_memory(np.array(op, copy=True), op.dense())
        np.testing.assert_array_equal(np.asarray(op), A)
        np.testing.assert_array_equal(np.array(op, dtype=np.float32), A.astype("f4"))
        if form == "dense":
            assert np.asarray(op) is op.dense()  # no copy unless asked

    def test_to_frame(self, form):
        """``to_frame`` labels the matrix; sparse forms give a sparse frame."""
        rng = np.random.default_rng(5)
        A = square(form, rng)
        idx = pd.Index(list("abcdef"), name="cell")
        frame = make(form, A).to_frame(idx)
        assert frame.index.equals(idx) and frame.columns.equals(idx)
        is_sparse = all(isinstance(dt, pd.SparseDtype) for dt in frame.dtypes)
        assert is_sparse == (form != "dense")
        dense = frame.sparse.to_dense() if is_sparse else frame
        np.testing.assert_array_equal(dense.to_numpy(), A)

    def test_pickle(self, form):
        """An operator survives a pickle round trip."""
        A = square(form, np.random.default_rng(6))
        op = pickle.loads(pickle.dumps(make(form, A)))
        assert isinstance(op, KIND[form])
        np.testing.assert_array_equal(op.dense(), A)


@pytest.mark.parametrize("form", ["dense", "sparse"])
def test_rectangular(form):
    """A rectangular dense or sparse operator multiplies and transposes."""
    rng = np.random.default_rng(2)
    A = sparse_random(rng, 5, 8, 0.4)
    op = make(form, A)
    assert op.shape == (5, 8) and op.T.shape == (8, 5)
    np.testing.assert_allclose(op @ np.ones(8), A @ np.ones(8), rtol=RTOL)
    np.testing.assert_allclose(op.T @ np.ones(5), A.T @ np.ones(5), rtol=RTOL)
    assert dottest(op)


def test_diagonal_solve_raises_on_zero():
    """A zero on the diagonal is singular."""
    with pytest.raises(LinAlgError, match="singular"):
        DiagonalOperator([1.0, 0.0]).solve(np.ones(2))


def test_singular_raises_linalgerror_in_every_form():
    """A singular sparse matrix raises ``LinAlgError``, as a dense one does (not SuperLU's ``RuntimeError``)."""
    S = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for op in (DenseOperator(S), SparseOperator(sp.csr_matrix(S))):
        with pytest.raises(LinAlgError, match="singular"):
            op.solve(np.ones(3))


def test_dense_solve_keeps_ldl_fallback(caplog):
    """A dense operator that Cholesky refuses is solved by ``LDL^T``, with a warning."""
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric, not positive-definite
    with caplog.at_level("WARNING", logger="fips.operator"):
        x = DenseOperator(A).solve(np.array([3.0, 3.0]))
    np.testing.assert_allclose(x, [1.0, 1.0], rtol=RTOL)
    assert any("non-positive-definite" in r.message for r in caplog.records)


def test_dottest_catches_a_wrong_adjoint():
    """``dottest`` fails for an operator whose rmatvec is not its adjoint."""
    A = np.arange(6.0).reshape(2, 3)
    wrong = LinearOperator(
        (2, 3), matvec=lambda x: A @ x, rmatvec=lambda y: 2 * A.T @ y, dtype=float
    )
    assert not dottest(wrong)
    assert dottest(DenseOperator(A))


# ---------------------------------------------------------------------------
# Products and sums between forms
# ---------------------------------------------------------------------------

PRODUCT_KIND = {
    ("dense", "dense"): DenseOperator,
    ("dense", "sparse"): DenseOperator,
    ("dense", "diagonal"): DenseOperator,
    ("sparse", "dense"): DenseOperator,
    ("sparse", "sparse"): SparseOperator,
    ("sparse", "diagonal"): SparseOperator,
    ("diagonal", "dense"): DenseOperator,
    ("diagonal", "sparse"): SparseOperator,
    ("diagonal", "diagonal"): DiagonalOperator,
}


@pytest.mark.parametrize("left, right", list(PRODUCT_KIND))
def test_product_between_forms(left, right):
    """``A @ B`` takes the cheapest exact form and equals the dense product."""
    rng = np.random.default_rng(7)
    n = 40  # large enough that a product of two tridiagonals stays sparse
    A, B = square(left, rng, n), square(right, rng, n)
    product = make(left, A) @ make(right, B)
    assert type(product) is PRODUCT_KIND[left, right]
    np.testing.assert_allclose(product.dense(), A @ B, rtol=RTOL, atol=1e-12)
    # A SciPy sparse matrix on either side counts as a sparse operator
    if right == "sparse":
        assert isinstance(make(left, A) @ sp.csr_matrix(B), Operator)


def test_sparse_product_that_fills_in_is_dense():
    """A product of sparse operators that fills more than a quarter is dense."""
    rng = np.random.default_rng(8)
    A = sparse_random(rng, 30, 30, 0.2)
    product = SparseOperator(sp.csr_matrix(A)) @ SparseOperator(sp.csr_matrix(A.T))
    assert isinstance(product, DenseOperator)
    np.testing.assert_allclose(product.dense(), A @ A.T, rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize("left", FORMS)
@pytest.mark.parametrize("right", FORMS)
def test_sum_between_forms(left, right):
    """``A + B`` and ``A - B`` are dense with a dense side, else sparse or diagonal."""
    rng = np.random.default_rng(9)
    A, B = square(left, rng), square(right, rng)
    expected = DenseOperator
    if "dense" not in (left, right):
        expected = DiagonalOperator if left == right == "diagonal" else SparseOperator
    for result, values in [
        (make(left, A) + make(right, B), A + B),
        (make(left, A) - make(right, B), A - B),
    ]:
        assert type(result) is expected
        np.testing.assert_allclose(result.dense(), values, rtol=RTOL)


def test_arithmetic_with_arrays():
    """An array on either side of ``+`` or ``-`` counts as a dense operator."""
    rng = np.random.default_rng(10)
    A, B = spd_dense(rng, 4), spd_dense(rng, 4)
    op = DiagonalOperator(np.diag(A))
    D = np.diag(np.diag(A))
    for result, expected in [
        (op + B, D + B),
        (B + op, B + D),
        (B - op, B - D),
        (op - B, D - B),
    ]:
        assert isinstance(result, DenseOperator)
        np.testing.assert_allclose(result.dense(), expected, rtol=RTOL)


def test_shape_mismatch_raises():
    """Operators of incompatible shapes refuse to multiply or add."""
    with pytest.raises(ValueError, match="multiply"):
        DiagonalOperator(np.ones(3)) @ DiagonalOperator(np.ones(4))
    with pytest.raises(ValueError, match="add"):
        DiagonalOperator(np.ones(3)) + DiagonalOperator(np.ones(4))


# ---------------------------------------------------------------------------
# as_operator and Structure2D.operator
# ---------------------------------------------------------------------------


class TestAsOperator:
    """``as_operator`` picks the form from the storage."""

    def test_forms(self):
        """An array is dense, a sparse matrix sparse, a sparse diagonal diagonal."""
        A = np.array([[1.0, 2.0], [0.0, 3.0]])
        assert isinstance(as_operator(A), DenseOperator)
        assert isinstance(as_operator(sp.csr_matrix(A)), SparseOperator)
        diag = as_operator(sp.diags([1.0, 2.0]))
        assert isinstance(diag, DiagonalOperator)
        np.testing.assert_array_equal(diag.values, [1.0, 2.0])
        # Square but not diagonal, and diagonal-looking but rectangular
        assert isinstance(as_operator(sp.csr_matrix(np.eye(2, 3))), SparseOperator)
        op = DenseOperator(A)
        assert as_operator(op) is op

    def test_integer_data_becomes_float(self):
        """Every form holds float64."""
        assert as_operator(np.eye(2, dtype=int)).dtype == np.float64
        assert DiagonalOperator([1, 2]).values.dtype == np.float64
        assert SparseOperator(sp.eye(2, dtype=int, format="csr")).matrix.dtype == float

    def test_not_2d_raises(self):
        """A 1D array is refused, with a pointer to DiagonalOperator."""
        with pytest.raises(ValueError, match="DiagonalOperator"):
            as_operator(np.ones(3))

    def test_structure(self):
        """A structure gives its ``operator``."""
        idx = pd.MultiIndex.from_product([["b"], ["x", "y"]], names=["block", "c"])
        M = Matrix(pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=idx))
        np.testing.assert_array_equal(as_operator(M).dense(), M.values)


class TestStructureOperator:
    """``Structure2D.operator`` gives the data without a dense copy of a sparse frame."""

    @staticmethod
    def index(n=4):
        """A one-block index."""
        return pd.MultiIndex.from_product(
            [["s"], [f"c{k}" for k in range(n)]], names=["block", "cell"]
        )

    def test_dense_is_a_view(self):
        """A dense float structure's operator holds its array, not a copy."""
        idx = self.index()
        M = CovarianceMatrix(pd.DataFrame(np.eye(4) * 2.0, index=idx, columns=idx))
        op = M.operator
        assert isinstance(op, DenseOperator)
        assert np.shares_memory(op.array, M.data.to_numpy())

    def test_sparse(self):
        """A sparse structure gives a SparseOperator equal to ``values``."""
        idx = self.index()
        A = spd_banded(np.random.default_rng(11), 4)
        M = CovarianceMatrix(pd.DataFrame(A, index=idx, columns=idx), sparse=True)
        op = M.operator
        assert isinstance(op, SparseOperator)
        np.testing.assert_array_equal(op.dense(), A)
        assert op.matrix.nnz == M.values.nnz

    def test_sparse_diagonal(self):
        """``diagonal(..., sparse=True)`` gives a DiagonalOperator; dense gives a dense one."""
        idx = self.index()
        S = CovarianceMatrix.diagonal([1.0, 2.0, 3.0, 4.0], index=idx, sparse=True)
        assert isinstance(S.operator, DiagonalOperator)
        np.testing.assert_array_equal(S.operator.values, [1.0, 2.0, 3.0, 4.0])
        assert isinstance(
            CovarianceMatrix.diagonal(1.0, index=idx).operator, DenseOperator
        )

    def test_follows_inplace_changes(self):
        """The operator is built on access, so it follows ``inplace=True`` changes (#70)."""
        idx = self.index()
        S = CovarianceMatrix.diagonal([1.0, 2.0, 3.0, 4.0], index=idx, sparse=True)
        S.reindex(idx[::-1], columns=idx[::-1], inplace=True)
        np.testing.assert_array_equal(S.operator.diagonal(), [4.0, 3.0, 2.0, 1.0])

    def test_integer_data_computes_in_float64(self):
        """Integer structures give float64 from a sum, ``variances`` and ``convolve``."""
        from fips import Vector, convolve

        idx = self.index()[:2]
        o = pd.MultiIndex.from_product(
            [["z"], ["o1", "o2", "o3"]], names=["block", "obs"]
        )
        M = Matrix(pd.DataFrame(np.array([[1, 2], [3, 4]]), index=idx, columns=idx))
        assert set(map(str, (M + M).data.dtypes)) == {"float64"}
        np.testing.assert_array_equal((M + M).values, [[2.0, 4.0], [6.0, 8.0]])
        C = CovarianceMatrix(
            pd.DataFrame(np.array([[2, 1], [1, 3]]), index=idx, columns=idx)
        )
        assert C.variances.data.dtype == np.float64
        H = ForwardOperator(
            pd.DataFrame(np.array([[1, 0], [0, 1], [1, 1]]), index=o, columns=idx)
        )
        y = convolve(Vector(pd.Series([1, 2], index=idx, name="prior")), H)
        assert y.dtype == np.float64
        np.testing.assert_array_equal(y.values, [1.0, 2.0, 3.0])

    def test_forward_operator(self):
        """A sparse forward operator's ``operator`` is rectangular and sparse."""
        o = pd.MultiIndex.from_product([["z"], ["o1", "o2"]], names=["block", "obs"])
        A = np.array([[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0]])
        H = ForwardOperator(pd.DataFrame(A, index=o, columns=self.index()), sparse=True)
        assert isinstance(H.operator, SparseOperator)
        np.testing.assert_array_equal(H.operator.dense(), A)


# ---------------------------------------------------------------------------
# The estimator on every combination of dense and sparse inputs (#79)
# ---------------------------------------------------------------------------

N, M = 30, 20


def _inputs(cov_shape: str):
    """Arrays of a problem whose ``H`` is sparse and whose covariances are banded or diagonal."""
    rng = np.random.default_rng(12)
    H = np.zeros((M, N))
    for i in range(M):  # each observation sees two cells
        H[i, rng.choice(N, 2, replace=False)] = rng.uniform(0.5, 2.0, 2)
    if cov_shape == "banded":
        S_0, S_z = spd_banded(rng, N), spd_banded(rng, M) / 4
    else:
        S_0, S_z = np.diag(rng.uniform(1, 3, N)), np.diag(rng.uniform(0.1, 0.5, M))
    x_0 = rng.normal(5.0, 1.0, N)
    z = H @ x_0 + rng.normal(0, 1.0, M)
    return z, x_0, H, S_0, S_z, rng.normal(0.5, 0.1, M)


def _problem(cov_shape: str, sparse: tuple[bool, bool, bool]) -> InverseProblem:
    """The labeled problem, with each of ``H``, ``S_0``, ``S_z`` dense or sparse."""
    z, x_0, H, S_0, S_z, c = _inputs(cov_shape)
    s = pd.MultiIndex.from_product(
        [["flux"], [f"c{k}" for k in range(N)]], names=["block", "cell"]
    )
    o = pd.MultiIndex.from_product(
        [["conc"], [f"o{k}" for k in range(M)]], names=["block", "obs"]
    )
    return InverseProblem(
        obs=pd.Series(z, index=o),
        prior=pd.Series(x_0, index=s),
        forward_operator=ForwardOperator(
            pd.DataFrame(H, index=o, columns=s), sparse=sparse[0]
        ),
        prior_error=CovarianceMatrix(
            pd.DataFrame(S_0, index=s, columns=s), sparse=sparse[1]
        ),
        modeldata_mismatch=CovarianceMatrix(
            pd.DataFrame(S_z, index=o, columns=o), sparse=sparse[2]
        ),
        constant=pd.Series(c, index=o),
    )


ESTIMATOR_OUTPUTS = [
    "x_hat", "S_hat", "K", "A", "y_hat", "y_0", "S_y0", "desroziers", "U_red",
    "DOFS", "reduced_chi2", "R2", "RMSE", "uncertainty_reduction",
]  # fmt: skip
POSTERIOR_OUTPUTS = [
    "estimate", "mean", "covariance", "variances", "modeled_obs", "prior_modeled_obs",
    "prior_obs_covariance", "kalman_gain", "averaging_kernel", "desroziers",
    "dofs", "reduced_chi2", "r2", "rmse", "uncertainty_reduction",
]  # fmt: skip
COMBINATIONS = [
    (h, s0, sz) for h in (False, True) for s0 in (False, True) for sz in (False, True)
]


def _close(actual, expected, what):
    """Compare at 1e-10 relative, with an absolute floor of 1e-10 of the largest entry."""
    expected = np.asarray(expected, dtype=float)
    floor = 1e-10 * max(np.abs(expected).max(), 1.0)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), expected, rtol=1e-10, atol=floor, err_msg=what
    )


@pytest.mark.parametrize("cov_shape", ["banded", "diagonal"])
@pytest.mark.parametrize(
    "sparse", COMBINATIONS, ids=lambda c: "".join("sd"[not f] for f in c)
)
def test_every_combination_matches_dense(cov_shape, sparse):
    """
    Each of the eight dense/sparse combinations of ``H``, ``S_0``, ``S_z`` gives the dense solve.

    Every estimator output and every ``Posterior`` output is compared, with a
    ``gamma`` other than 1. ``ids`` read ``H S_0 S_z``: ``s`` sparse, ``d`` dense.
    """
    reference = _problem(cov_shape, (False, False, False))
    problem = _problem(cov_shape, sparse)
    assert problem.forward_operator.is_sparse == sparse[0]
    assert problem.prior_error.is_sparse == sparse[1]
    assert problem.modeldata_mismatch.is_sparse == sparse[2]
    ref = reference.solve("bayesian", gamma=0.5).result
    post = problem.solve("bayesian", gamma=0.5).result
    for name in ESTIMATOR_OUTPUTS:
        value = getattr(post.estimator, name)
        if isinstance(value, np.ndarray):
            assert type(value) is np.ndarray, f"{name} is {type(value)}"
        _close(value, getattr(ref.estimator, name), name)
    for name in POSTERIOR_OUTPUTS:
        value, expected = getattr(post, name), getattr(ref, name)
        if callable(value):  # averaging_kernel, with no aggregation
            value, expected = value(), expected()
        if hasattr(value, "to_dense"):  # a labeled structure
            assert value.index.equals(expected.index), name
            value, expected = value.to_dense().values, expected.to_dense().values
        elif hasattr(value, "index"):
            assert value.index.equals(expected.index), name
            value, expected = value.values, expected.values
        _close(value, expected, f"posterior {name}")
    x = np.linspace(1.0, 2.0, N)
    _close(post.estimator.cost(x), ref.estimator.cost(x), "cost")


def test_all_sparse_diagonal_problem_stays_sparse():
    """Sparse ``H`` with diagonal covariances: the gain solve goes through the sparse path."""
    post = _problem("diagonal", (True, True, True)).solve("bayesian").result
    est = post.estimator
    assert isinstance(est._H_op, SparseOperator)
    assert isinstance(est._S_0_op, DiagonalOperator) and isinstance(
        est._S_z_op, DiagonalOperator
    )
    assert isinstance(est._HS_0, SparseOperator)
    assert isinstance(est._HS_0H + est._S_z_op, SparseOperator)


def test_estimator_takes_arrays_and_sparse_matrices():
    """The estimator takes plain arrays and SciPy sparse matrices, as before."""
    z, x_0, H, S_0, S_z, c = _inputs("diagonal")
    dense = BayesianSolver(z, x_0, H, S_0, S_z, c=c)
    mixed = BayesianSolver(z, x_0, sp.csr_matrix(H), sp.csr_matrix(S_0), S_z, c=c)
    assert isinstance(mixed._H_op, SparseOperator) and isinstance(
        mixed._S_0_op, DiagonalOperator
    )
    _close(mixed.x_hat, dense.x_hat, "x_hat")
    _close(mixed.S_hat, dense.S_hat, "S_hat")
    # The matrices read back as arrays
    np.testing.assert_array_equal(np.asarray(dense.H), H)
    np.testing.assert_array_equal(mixed.S_0.toarray(), S_0)


def test_gamma_as_zero_dim_array():
    """``gamma`` given as a 0-d array solves as the float it holds."""
    z, x_0, H, S_0, S_z, c = _inputs("banded")
    ref = BayesianSolver(z, x_0, H, S_0, S_z, c=c, gamma=0.5)
    est = BayesianSolver(z, x_0, H, S_0, S_z, c=c, gamma=np.array(0.5))
    np.testing.assert_array_equal(est.x_hat, ref.x_hat)
    assert est.reduced_chi2 == ref.reduced_chi2


class TestEstimatorAttributes:
    """``H``, ``S_0``, ``S_z`` and ``_S_z_orig`` give what they gave before operators."""

    def test_dense_inputs_give_arrays(self):
        """Dense inputs read back as arrays, which index and multiply elementwise."""
        z, x_0, H, S_0, S_z, c = _inputs("banded")
        est = BayesianSolver(z, x_0, H, S_0, S_z, c=c, gamma=0.5)
        for name, expected in (
            ("H", H),
            ("S_0", S_0),
            ("S_z", S_z / 0.5),
            ("_S_z_orig", S_z),
        ):
            value = getattr(est, name)
            assert type(value) is np.ndarray, name
            np.testing.assert_array_equal(value, expected)
        v = np.arange(1.0, N + 1)
        np.testing.assert_array_equal(est.S_0 * v, S_0 * v)  # elementwise
        assert est.S_0[0, 1] == S_0[0, 1]
        np.testing.assert_array_equal(est.H.sum(axis=0), H.sum(axis=0))

    def test_sparse_inputs_give_csr_matrices(self):
        """A sparse or diagonal input reads back as a CSR matrix, as ``.values`` gives it."""
        z, x_0, H, S_0, S_z, c = _inputs("diagonal")
        est = BayesianSolver(
            z, x_0, sp.csr_matrix(H), sp.csr_matrix(S_0), sp.csr_matrix(S_z), c=c
        )
        for name, expected in (("H", H), ("S_0", S_0), ("S_z", S_z)):
            value = getattr(est, name)
            assert isinstance(value, sp.csr_matrix), name
            np.testing.assert_array_equal(value.toarray(), expected)

    def test_an_edit_does_not_reach_the_estimator(self):
        """The arrays are read-only, and ``np.array`` of one is a copy."""
        z, x_0, H, S_0, S_z, c = _inputs("banded")
        est = BayesianSolver(z, x_0, H, S_0, S_z, c=c)
        with pytest.raises(ValueError, match="read-only"):
            est.H[0, 0] = 99.0
        copy = np.array(est.H)
        copy[:] = 0.0
        np.testing.assert_array_equal(est.H, H)
        np.testing.assert_allclose(est.A, BayesianSolver(z, x_0, H, S_0, S_z, c=c).A)

    def test_problem_estimator_attributes(self):
        """Through ``fips.solve``: arrays for dense structures, CSR for sparse ones."""
        dense = _problem("banded", (False, False, False)).solve("bayesian").estimator
        sparse = _problem("banded", (True, True, True)).solve("bayesian").estimator
        for name in ("H", "S_0", "S_z"):
            assert type(getattr(dense, name)) is np.ndarray, name
            assert isinstance(getattr(sparse, name), sp.csr_matrix), name
            np.testing.assert_allclose(
                getattr(sparse, name).toarray(), getattr(dense, name)
            )


class TestEstimatorPickle:
    """A pickle stores the matrices as arrays, so it reads with or without operators."""

    def test_round_trip(self):
        """A solved estimator pickles and keeps its operators and results."""
        z, x_0, H, S_0, S_z, c = _inputs("diagonal")
        est = BayesianSolver(z, x_0, sp.csr_matrix(H), sp.csr_matrix(S_0), S_z, c=c)
        _ = est.x_hat, est.reduced_chi2
        state = est.__getstate__()
        assert not any(isinstance(v, Operator) for v in state.values())
        assert {"H", "S_0", "S_z", "_S_z_orig"} <= set(state)
        loaded = pickle.loads(pickle.dumps(est))
        assert isinstance(loaded._H_op, SparseOperator)
        assert isinstance(loaded._S_0_op, DiagonalOperator)
        np.testing.assert_array_equal(loaded.x_hat, est.x_hat)
        assert loaded.uncertainty_reduction == est.uncertainty_reduction

    def test_state_without_operators(self):
        """State written by a fips without operators (plain arrays, ``_H_T``) still computes."""
        z, x_0, H, S_0, S_z, c = _inputs("banded")
        est = BayesianSolver(z, x_0, H, S_0, S_z, c=c)
        old = {k: v for k, v in est.__getstate__().items()}
        old["_H_T"] = H.T
        loaded = BayesianSolver.__new__(BayesianSolver)
        loaded.__setstate__(old)
        assert isinstance(loaded._H_op, DenseOperator) and "_H_T" not in vars(loaded)
        assert type(loaded.H) is np.ndarray
        _close(loaded.reduced_chi2, est.reduced_chi2, "reduced_chi2")
        _close(loaded.U_red, est.U_red, "U_red")
