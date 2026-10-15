"""Tests for fips.linalg: each form against its dense equivalent, and the solve on every form."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.linalg import LinAlgError, block_diag
from scipy.sparse.linalg import LinearOperator as SciPyLinearOperator

import fips
from fips import Covariance, Operator, Problem, Space, Vector
from fips.linalg import (
    BlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    KroneckerOperator,
    LinearOperator,
    SparseOperator,
    as_linear_operator,
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


BLOCK_TAIL = 2
"""The size of the diagonal block that ends a block diagonal test matrix."""


def make(form: str, A: np.ndarray) -> LinearOperator:
    """Return `A` (dense) as an operator of the given form."""
    if form == "dense":
        return DenseOperator(A)
    if form == "sparse":
        return SparseOperator(sp.csr_matrix(A))
    if form == "block":
        k = len(A) - BLOCK_TAIL
        assert not A[:k, k:].any() and not A[k:, :k].any()
        return BlockDiagonalOperator(
            [DenseOperator(A[:k, :k]), DiagonalOperator(np.diag(A[k:, k:]))]
        )
    assert np.array_equal(A, np.diag(np.diag(A)))
    return DiagonalOperator(np.diag(A))


FORMS = ["dense", "sparse", "diagonal"]
ALL_FORMS = [*FORMS, "block"]
KIND = {
    "dense": DenseOperator,
    "sparse": SparseOperator,
    "diagonal": DiagonalOperator,
    "block": BlockDiagonalOperator,
}


def square(form, rng, n=6):
    """A square SPD matrix that the form can hold."""
    if form == "diagonal":
        return np.diag(rng.uniform(0.5, 2.0, n))
    if form == "block":
        tail = np.diag(rng.uniform(0.5, 2.0, BLOCK_TAIL))
        return block_diag(spd_dense(rng, n - BLOCK_TAIL), tail)
    return spd_banded(rng, n) if form == "sparse" else spd_dense(rng, n)


def masked(A, rows, cols):
    """The entries of `A` at `rows` and `cols`, zero where a position is -1."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    out = A[np.ix_(np.maximum(rows, 0), np.maximum(cols, 0))]
    out[rows < 0] = 0.0
    out[:, cols < 0] = 0.0
    return out


# ---------------------------------------------------------------------------
# Each form against its dense equivalent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ALL_FORMS)
class TestEachForm:
    """matvec, rmatvec, matmat, transpose, diagonal, trace, solve, logdet and sqrt."""

    def test_products_with_arrays(self, form):
        """``@``, ``matvec``, ``rmatvec``, ``matmat`` and ``rmatmat`` match the array."""
        rng = np.random.default_rng(0)
        A = square(form, rng)
        op = make(form, A)
        x, X = rng.standard_normal(6), rng.standard_normal((6, 3))
        assert isinstance(op, SciPyLinearOperator)
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

    def test_logdet(self, form):
        """``logdet`` matches NumPy's ``slogdet`` of the dense matrix."""
        A = square(form, np.random.default_rng(13))
        sign, expected = np.linalg.slogdet(A)
        assert sign == 1.0
        assert make(form, A).logdet() == pytest.approx(expected, rel=1e-12)

    def test_sqrt(self, form):
        """``sqrt`` gives ``L`` with ``L L^T = A``: a diagonal stays diagonal, blocks stay blocks."""
        A = square(form, np.random.default_rng(14))
        L = make(form, A).sqrt()
        kinds = {"diagonal": DiagonalOperator, "block": BlockDiagonalOperator}
        assert type(L) is kinds.get(form, DenseOperator)
        np.testing.assert_allclose(L.dense() @ L.dense().T, A, rtol=1e-12)
        # The lower Cholesky factor (of each block)
        np.testing.assert_allclose(L.dense(), np.linalg.cholesky(A), rtol=1e-12)

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
    with pytest.raises(LinAlgError):  # not SuperLU's RuntimeError
        SparseOperator(sp.csr_matrix(S)).logdet()


def test_logdet_and_sqrt_refuse_a_matrix_that_is_not_positive_definite():
    """A form that is not positive-definite raises ``LinAlgError`` from ``logdet`` and ``sqrt``."""
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    for op in (
        DenseOperator(indefinite),
        SparseOperator(sp.csr_matrix(indefinite)),
        SparseOperator(sp.csr_matrix(-np.eye(3))),  # det > 0, but not positive
        DiagonalOperator([1.0, -1.0]),
        BlockDiagonalOperator([np.eye(2), DenseOperator(indefinite)]),
    ):
        with pytest.raises(LinAlgError, match="positive"):
            op.logdet()
        with pytest.raises(LinAlgError):
            op.sqrt()
    # A zero variance has a square root
    np.testing.assert_array_equal(DiagonalOperator([0.0, 4.0]).sqrt().values, [0, 2])


def test_sparse_logdet_refuses_an_indefinite_matrix_with_a_zero_diagonal():
    """A sparse form's ``logdet`` refuses an indefinite matrix whose diagonal is zero, as a dense one does."""
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(LinAlgError):
        DenseOperator(swap).logdet()
    with pytest.raises(LinAlgError):
        SparseOperator(sp.csr_matrix(swap)).logdet()


def test_dense_solve_keeps_ldl_fallback(caplog):
    """A dense operator that Cholesky refuses is solved by ``LDL^T``, with a warning."""
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric, not positive-definite
    with caplog.at_level("WARNING", logger="fips.linalg"):
        x = DenseOperator(A).solve(np.array([3.0, 3.0]))
    np.testing.assert_allclose(x, [1.0, 1.0], rtol=RTOL)
    assert any("non-positive-definite" in r.message for r in caplog.records)


def test_dottest_catches_a_wrong_adjoint():
    """``dottest`` fails for an operator whose rmatvec is not its adjoint."""
    A = np.arange(6.0).reshape(2, 3)
    wrong = SciPyLinearOperator(
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
    # A SciPy sparse matrix on the right counts as a sparse operator
    if right == "sparse":
        product = make(left, A) @ sp.csr_matrix(B)
        assert type(product) is PRODUCT_KIND[left, right]


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
# _select: rows and columns by position, -1 for a row or column of zeros
# ---------------------------------------------------------------------------


class TestSelect:
    """``_select`` takes positions, -1 giving a row or column of zeros, and keeps the form."""

    ROWS = np.array([3, -1, 0, 5])
    COLS = np.array([-1, 2, 0, 4, -1])

    @pytest.mark.parametrize(
        "form, kind",
        [
            ("dense", DenseOperator),
            ("sparse", SparseOperator),
            ("diagonal", SparseOperator),
        ],
    )
    def test_rows_and_columns_with_zeros(self, form, kind):
        """Other rows than columns, some -1: dense stays dense, sparse or diagonal is sparse."""
        A = square(form, np.random.default_rng(15))
        out = make(form, A)._select(self.ROWS, self.COLS)
        assert type(out) is kind
        np.testing.assert_array_equal(out.dense(), masked(A, self.ROWS, self.COLS))

    @pytest.mark.parametrize("form", FORMS)
    def test_same_rows_and_columns_with_zeros(self, form):
        """The same positions on both axes keep the form; a diagonal stays diagonal."""
        A = square(form, np.random.default_rng(16))
        positions = np.array([4, -1, 3, 0])  # 4 and 3 are neighbors in the band
        out = make(form, A)._select(positions, positions)
        assert type(out) is KIND[form]
        np.testing.assert_array_equal(out.dense(), masked(A, positions, positions))
        assert out.diagonal()[1] == 0.0

    def test_sparse_zeros_are_not_stored(self):
        """A -1 row or column of a sparse form stores nothing."""
        A = square("sparse", np.random.default_rng(17))
        out = make("sparse", A)._select(self.ROWS, self.COLS)
        assert isinstance(out, SparseOperator)
        stored = out.matrix.tocoo()
        assert not (stored.row == 1).any()
        assert not np.isin(stored.col, [0, 4]).any()
        assert (stored.data != 0).all()


# ---------------------------------------------------------------------------
# BlockDiagonalOperator
# ---------------------------------------------------------------------------


def kronecker_block():
    """A 6 x 6 Kronecker covariance: a 2 x 2 correlation in time, a 3 x 3 in space."""
    time = np.array([[1.0, 0.6], [0.6, 1.0]])
    space = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]])
    return KroneckerOperator([time, space], std=np.linspace(1.0, 2.0, 6))


def block_operator():
    """A Kronecker block, a dense one and a diagonal one (6 + 3 + 2 rows), and its dense matrix."""
    rng = np.random.default_rng(20)
    blocks = [
        kronecker_block(),
        DenseOperator(spd_dense(rng, 3)),
        DiagonalOperator([2.0, 3.0]),
    ]
    return BlockDiagonalOperator(blocks), block_diag(*(b.dense() for b in blocks))


class TestBlockDiagonal:
    """Each block in its own form, zeros between them (``TestEachForm`` covers a simple one)."""

    def test_blocks_are_kept_as_given(self):
        """Arrays and sparse matrices become forms; each block keeps its form."""
        banded = spd_banded(np.random.default_rng(0), 4)
        B = BlockDiagonalOperator([np.eye(2), sp.csr_matrix(banded), sp.eye(3)])
        kinds = [DenseOperator, SparseOperator, DiagonalOperator]
        assert [type(b) for b in B.blocks] == kinds
        assert B.shape == (9, 9)
        np.testing.assert_array_equal(
            B.dense(), block_diag(np.eye(2), banded, np.eye(3))
        )
        assert type(block_operator()[0].blocks[0]) is KroneckerOperator

    def test_needs_square_blocks(self):
        """No block, or a block that is not square, raises."""
        with pytest.raises(ValueError, match="at least one block"):
            BlockDiagonalOperator([])
        with pytest.raises(ValueError, match="square"):
            BlockDiagonalOperator([np.eye(2), np.ones((2, 3))])

    def test_products_with_arrays(self):
        """Products with vectors and matrices on both sides match the dense matrix."""
        B, A = block_operator()
        rng = np.random.default_rng(21)
        x, X = rng.standard_normal(11), rng.standard_normal((11, 4))
        np.testing.assert_allclose(B @ x, A @ x, rtol=RTOL)
        np.testing.assert_allclose(B @ X, A @ X, rtol=RTOL)
        np.testing.assert_allclose(x @ B, x @ A, rtol=RTOL)
        np.testing.assert_allclose(X.T @ B, X.T @ A, rtol=RTOL)
        assert dottest(B)

    @pytest.mark.parametrize("kind", [sp.csr_array, sp.csr_matrix, sp.csc_array])
    def test_products_with_sparse_matrices(self, kind):
        """A sparse matrix on either side gives the dense product, block by block."""
        B, A = block_operator()
        X = sparse_random(np.random.default_rng(22), 11, 5, 0.3)
        for right in (kind(X), SparseOperator(kind(X))):
            product = B @ right
            assert isinstance(product, DenseOperator)  # some blocks are not sparse
            np.testing.assert_allclose(product.dense(), A @ X, rtol=RTOL)
        product = SparseOperator(kind(X.T)) @ B
        assert isinstance(product, DenseOperator)
        np.testing.assert_allclose(product.dense(), X.T @ A, rtol=RTOL)
        # A bare sparse matrix on the left: SciPy's own product, an array
        np.testing.assert_allclose(np.asarray(kind(X.T) @ B), X.T @ A, rtol=RTOL)

    def test_sparse_blocks_give_sparse_products(self):
        """Blocks that are all sparse or diagonal make a sparse product with a sparse matrix."""
        rng = np.random.default_rng(23)
        S, d = spd_banded(rng, 20), np.arange(1.0, 6.0)
        B = BlockDiagonalOperator([sp.csr_array(S), DiagonalOperator(d)])
        A = block_diag(S, np.diag(d))
        H = sparse_random(rng, 10, 25, 0.1)
        for product, expected in [
            (B @ sp.csr_array(H.T), A @ H.T),
            (SparseOperator(sp.csr_array(H)) @ B, H @ A),
        ]:
            assert isinstance(product, SparseOperator)
            np.testing.assert_allclose(product.dense(), expected, rtol=RTOL)

    def test_transpose_transposes_each_block(self):
        """``.T`` keeps the blocks, each transposed."""
        C = np.random.default_rng(24).standard_normal((3, 3))  # not symmetric
        B = BlockDiagonalOperator([C, sp.csr_array(np.triu(np.ones((2, 2))))])
        assert isinstance(B.T, BlockDiagonalOperator)
        assert [type(b) for b in B.T.blocks] == [DenseOperator, SparseOperator]
        np.testing.assert_array_equal(B.T.dense(), B.dense().T)

    def test_solve_diagonal_logdet_sqrt(self):
        """Block by block, each matches the dense matrix (a Kronecker block among them)."""
        B, A = block_operator()
        rng = np.random.default_rng(25)
        b, X = rng.standard_normal(11), rng.standard_normal((11, 3))
        np.testing.assert_allclose(B.solve(b), np.linalg.solve(A, b), rtol=1e-10)
        np.testing.assert_allclose(B.solve(X), np.linalg.solve(A, X), rtol=1e-10)
        np.testing.assert_allclose(B.diagonal(), np.diag(A), rtol=RTOL)
        assert B.logdet() == pytest.approx(np.linalg.slogdet(A)[1], rel=1e-12)
        L = B.sqrt()
        assert isinstance(L, BlockDiagonalOperator)
        kinds = [DenseOperator, DenseOperator, DiagonalOperator]
        assert [type(b) for b in L.blocks] == kinds
        np.testing.assert_allclose(L.dense() @ L.dense().T, A, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize(
        "positions, kinds",
        [
            ([1, 0, 4, 6, 8, 10], [KroneckerOperator, DenseOperator, DiagonalOperator]),
            ([2, 7, 6, 9], [KroneckerOperator, DenseOperator, DiagonalOperator]),
            ([9, 10, 0, 1], [DiagonalOperator, KroneckerOperator]),
        ],
        ids=["within-blocks", "reordered-in-a-block", "blocks-reordered"],
    )
    def test_select_keeps_the_blocks(self, positions, kinds):
        """The same positions on both axes, each block's together, keep each block's form."""
        B, A = block_operator()
        out = B._select(np.array(positions), np.array(positions))
        assert isinstance(out, BlockDiagonalOperator)
        assert [type(b) for b in out.blocks] == kinds
        np.testing.assert_allclose(out.dense(), masked(A, positions, positions))

    @pytest.mark.parametrize(
        "positions, kind",
        [([0, 1, 2, 3, 4, 5], KroneckerOperator), ([10, 9], DiagonalOperator)],
        ids=["first-block", "last-block"],
    )
    def test_select_of_one_block_is_that_block(self, positions, kind):
        """Positions within one block give that block's own form."""
        B, A = block_operator()
        out = B._select(np.array(positions), np.array(positions))
        assert type(out) is kind
        np.testing.assert_allclose(out.dense(), masked(A, positions, positions))

    @pytest.mark.parametrize(
        "rows, cols",
        [([0, 9, 1], [0, 9, 1]), ([0, -1, 9], [0, -1, 9]), ([0, 1, 9], [1, 0, 9])],
        ids=["block-split", "minus-one", "rows-not-cols"],
    )
    def test_other_selections_are_dense(self, rows, cols):
        """Any other selection forms its entries, as the base class does."""
        B, A = block_operator()
        out = B._select(np.array(rows), np.array(cols))
        assert isinstance(out, DenseOperator)
        np.testing.assert_allclose(out.dense(), masked(A, rows, cols))

    def test_HSHT_with_a_sparse_jacobian(self):
        """``H @ S @ H.T`` with a sparse ``H`` equals the dense product, unlabeled and labeled."""
        B, A = block_operator()
        H = sparse_random(np.random.default_rng(26), 8, 11, 0.3)
        Hs = SparseOperator(sp.csr_array(H))
        expected = H @ A @ H.T
        for HSHT in (Hs @ B @ Hs.T, (Hs @ B) @ sp.csr_array(H).T):
            assert isinstance(HSHT, DenseOperator)
            np.testing.assert_allclose(HSHT.dense(), expected, rtol=1e-12, atol=1e-12)

        state = Space({"x": pd.RangeIndex(11, name="cell")})
        obs = Space({"z": pd.RangeIndex(8, name="obs")})
        H_op, S = Operator(sp.csr_array(H), obs, state), Covariance(B, state)
        assert S.data is B
        labeled = H_op @ S @ H_op.T
        assert labeled.rows == obs and labeled.cols == obs
        np.testing.assert_allclose(labeled.dense(), expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# as_linear_operator, and the dtype of a labeled Operator
# ---------------------------------------------------------------------------


class TestAsLinearOperator:
    """``as_linear_operator`` picks the form from the storage."""

    def test_forms(self):
        """An array is dense, a sparse matrix sparse, a sparse diagonal diagonal."""
        A = np.array([[1.0, 2.0], [0.0, 3.0]])
        assert isinstance(as_linear_operator(A), DenseOperator)
        assert isinstance(as_linear_operator(sp.csr_matrix(A)), SparseOperator)
        diag = as_linear_operator(sp.diags([1.0, 2.0]))
        assert isinstance(diag, DiagonalOperator)
        np.testing.assert_array_equal(diag.values, [1.0, 2.0])
        # Square but not diagonal, and diagonal-looking but rectangular
        rectangular = sp.csr_matrix(np.eye(2, 3))
        assert isinstance(as_linear_operator(rectangular), SparseOperator)
        op = DenseOperator(A)
        assert as_linear_operator(op) is op

    def test_integer_data_becomes_float(self):
        """Every form holds float64."""
        assert as_linear_operator(np.eye(2, dtype=int)).dtype == np.float64
        assert DiagonalOperator([1, 2]).values.dtype == np.float64
        assert SparseOperator(sp.eye(2, dtype=int, format="csr")).matrix.dtype == float

    def test_not_2d_raises(self):
        """A 1D array is refused, with a pointer to DiagonalOperator."""
        with pytest.raises(ValueError, match="DiagonalOperator"):
            as_linear_operator(np.ones(3))

    def test_operator_copies_a_float_array(self):
        """A labeled Operator holds its own copy of a float64 array."""
        space = Space({"s": pd.Index([f"c{k}" for k in range(4)], name="cell")})
        array = np.eye(4) * 2.0
        op = Operator(array, space, space)
        assert isinstance(op.data, DenseOperator)
        assert not np.shares_memory(op.data.array, array)
        np.testing.assert_array_equal(op.data.array, array)

    def test_integer_operator_computes_in_float64(self):
        """Integer matrices give float64 from a sum, ``variances`` and a product with a vector."""
        cells = Space({"s": pd.Index(["c0", "c1"], name="cell")})
        obs = Space({"z": pd.Index(["o1", "o2", "o3"], name="obs")})
        M = Operator(np.array([[1, 2], [3, 4]]), cells, cells)
        assert (M + M).dense().dtype == np.float64
        np.testing.assert_array_equal((M + M).dense(), [[2.0, 4.0], [6.0, 8.0]])
        C = Covariance(np.array([[2, 1], [1, 3]]), cells)
        assert C.variances.values.dtype == np.float64
        H = Operator(np.array([[1, 0], [0, 1], [1, 1]]), obs, cells)
        y = H @ Vector(np.array([1, 2]), cells)
        assert y.values.dtype == np.float64
        np.testing.assert_array_equal(y.values, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# A form owns its data: it copies its input and makes the copy read-only
# ---------------------------------------------------------------------------

_OWN = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])

# What a caller hands a form, how to change it in place, and the arrays the form holds
OWN_CASES = {
    "dense": (lambda: _OWN.copy(), DenseOperator, lambda g: g.fill(-1.0)),
    "sparse_csr": (lambda: sp.csr_array(_OWN), SparseOperator, lambda g: g.data.fill(-1)),
    "sparse_csc": (lambda: sp.csc_matrix(_OWN), SparseOperator, lambda g: g.data.fill(-1)),
    "diagonal": (lambda: np.diag(_OWN).copy(), DiagonalOperator, lambda g: g.fill(-1.0)),
}  # fmt: skip


def _held(op: LinearOperator) -> list[np.ndarray]:
    """The arrays a form holds."""
    if isinstance(op, DenseOperator):
        return [op.array]
    if isinstance(op, SparseOperator):
        return [op.matrix.data, op.matrix.indices, op.matrix.indptr]
    assert isinstance(op, DiagonalOperator)
    return [op.values]


def _arrays(given) -> list[np.ndarray]:
    """The arrays of what a caller handed over: an array, or a sparse matrix's three."""
    if sp.issparse(given):
        return [given.data, given.indices, given.indptr]
    return [given]


@pytest.mark.parametrize("case", OWN_CASES)
class TestOwnsItsData:
    """A form copies what it is given and makes the copy read-only, as a Vector does."""

    def test_editing_the_input_changes_nothing(self, case):
        """Writing into the array or matrix handed over leaves the form as it was."""
        make_input, kind, edit = OWN_CASES[case]
        given = make_input()
        op = kind(given)
        before = op.dense().copy()
        edit(given)
        np.testing.assert_array_equal(op.dense(), before)
        assert not any(
            np.shares_memory(a, g) for a in _held(op) for g in _arrays(given)
        )

    def test_what_it_holds_is_read_only(self, case):
        """Every array the form holds refuses a write."""
        make_input, kind, _ = OWN_CASES[case]
        for array in _held(kind(make_input())):
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0

    def test_asarray_is_the_buffer_and_array_a_copy(self, case):
        """``np.asarray`` gives a dense form's read-only buffer; ``np.array`` a writable, independent copy."""
        make_input, kind, _ = OWN_CASES[case]
        op = kind(make_input())
        before = op.dense().copy()
        buffer = np.asarray(op)
        if kind is DenseOperator:
            assert buffer is op.array and not buffer.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                buffer[0, 0] = 99.0
        duplicate = np.array(op)
        assert duplicate.flags.writeable
        assert not any(np.shares_memory(duplicate, a) for a in _held(op))
        duplicate[0, 0] = 99.0
        np.testing.assert_array_equal(op.dense(), before)

    def test_a_pickle_or_copy_is_read_only_too(self, case):
        """A pickle keeps the values, not the flag: the form sets it again."""
        make_input, kind, _ = OWN_CASES[case]
        op = kind(make_input())
        for again in (
            pickle.loads(pickle.dumps(op)),
            copy.copy(op),
            copy.deepcopy(op),
        ):
            assert type(again) is kind
            np.testing.assert_array_equal(again.dense(), op.dense())
            assert not any(a.flags.writeable for a in _held(again))


@pytest.mark.parametrize("case", ["dense", "sparse_csr", "sparse_csc"])
def test_as_linear_operator_copies_what_it_wraps(case):
    """``as_linear_operator`` copies an array or a sparse matrix, and the copy is read-only."""
    make_input, _, edit = OWN_CASES[case]
    given = make_input()
    op = as_linear_operator(given)
    before = op.dense().copy()
    edit(given)
    np.testing.assert_array_equal(op.dense(), before)
    assert not any(a.flags.writeable for a in _held(op))


def test_a_block_diagonal_form_copies_its_arrays():
    """A block given as an array is copied, as ``as_linear_operator`` does; a form is kept."""
    first, second = _OWN.copy(), np.array([[2.0]])
    form = DiagonalOperator([5.0, 6.0])
    op = BlockDiagonalOperator([first, form, second])
    before = op.dense().copy()
    first.fill(-1.0)
    second.fill(-1.0)
    np.testing.assert_array_equal(op.dense(), before)
    assert op.blocks[1] is form
    assert not op.blocks[0].array.flags.writeable


def test_a_form_made_from_a_form_shares_its_arrays():
    """A transpose, a scalar multiple and a selection hold the parent's read-only arrays, not copies."""
    dense, sparse = DenseOperator(_OWN), SparseOperator(sp.csr_array(_OWN))
    assert np.shares_memory(dense.T.array, dense.array)
    assert np.shares_memory(sparse.T.matrix.data, sparse.matrix.data)
    assert np.shares_memory(sparse._csr().data, sparse.matrix.data)
    for derived in (dense.T, sparse.T, dense * 2.0, sparse * 2.0, dense @ dense):
        assert not any(a.flags.writeable for a in _held(derived))


def test_a_product_does_not_copy_its_operands_again(constructors):
    """
    Forms built from forms never go through the constructors that copy.

    Products, sums, transposes, scalar multiples, selections, square roots and
    the solve of every form, and of forms in one another's company, hold the
    arrays they computed (or their parents' read-only ones). A copy of a 430 MB
    forward operator at each step would double the memory of a solve.
    """
    rng = np.random.default_rng(5)
    n = 6
    dense = DenseOperator(spd_dense(rng, n))
    sparse = SparseOperator(sp.csr_array(spd_banded(rng, n)))
    diagonal = DiagonalOperator(rng.uniform(1, 2, n))
    kron = KroneckerOperator([spd_dense(rng, 2) / 8, spd_dense(rng, 3) / 12], std=1.5)
    block = BlockDiagonalOperator(
        [DenseOperator(spd_dense(rng, 4)), DiagonalOperator(rng.uniform(1, 2, 2))]
    )
    forms = [dense, sparse, diagonal, kron, block]
    constructors.clear()  # building the inputs copied them, once
    rows = np.array([3, 0, 5, 1])
    for a in forms:
        a.T, 2.0 * a, a / 4.0, -a, a.diagonal(), a.sqrt(), a.solve(np.ones(n))
        a._select(rows, rows), a._select(rows, rows[::-1]), a._select(rows, rows - 1)
        a.dense(), np.array(a), np.asarray(a), a.logdet()
        for b in forms:
            a @ b, a + b, a - b, a @ b.T
        as_linear_operator(a)
    assert constructors == []


# ---------------------------------------------------------------------------
# The solve on every combination of dense and sparse inputs (#79)
# ---------------------------------------------------------------------------

N, M = 30, 20
STATE = Space({"flux": pd.Index([f"c{k}" for k in range(N)], name="cell")})
OBS = Space({"conc": pd.Index([f"o{k}" for k in range(M)], name="obs")})


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


def _labeled(z, x_0, H, S_0, S_z, c) -> Problem:
    """The labeled problem of the arrays (each matrix an array, a sparse matrix or a form)."""
    return Problem(
        obs=Vector(z, OBS),
        prior=Vector(x_0, STATE),
        forward=Operator(H, OBS, STATE),
        prior_cov=Covariance(S_0, STATE),
        obs_cov=Covariance(S_z, OBS),
        offset=Vector(c, OBS),
    )


def _problem(cov_shape: str, sparse: tuple[bool, bool, bool]) -> Problem:
    """The labeled problem, with each of ``H``, ``S_0``, ``S_z`` dense or sparse."""
    z, x_0, H, S_0, S_z, c = _inputs(cov_shape)
    H, S_0, S_z = (
        sp.csr_array(A) if is_sparse else A
        for A, is_sparse in zip((H, S_0, S_z), sparse, strict=True)
    )
    return _labeled(z, x_0, H, S_0, S_z, c)


POSTERIOR_OUTPUTS = [
    "estimate", "mean", "cov", "variances", "modeled_obs",
    "innovation_cov", "kalman_gain", "averaging_kernel", "desroziers", "dofs", "chi2",
    "reduced_chi2", "r2", "rmse", "uncertainty_reduction",
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


def _assert_same_posterior(post, ref) -> None:
    """Every Posterior output of `post` matches `ref`."""
    for name in POSTERIOR_OUTPUTS:
        value, expected = getattr(post, name), getattr(ref, name)
        if callable(value):  # cov, averaging_kernel, desroziers: with no argument
            value, expected = value(), expected()
        if isinstance(value, pd.DataFrame):  # desroziers
            assert value.index.equals(expected.index), name
            value, expected = value.to_numpy(), expected.to_numpy()
        if isinstance(value, Operator):  # a Covariance among them
            assert type(value) is type(expected), name
            assert value.rows == expected.rows and value.cols == expected.cols, name
            value, expected = value.dense(), expected.dense()
        elif hasattr(value, "index"):
            assert value.index.equals(expected.index), name
            value, expected = value.values, expected.values
        _close(value, expected, f"posterior {name}")


@pytest.mark.parametrize("cov_shape", ["banded", "diagonal"])
@pytest.mark.parametrize(
    "sparse", COMBINATIONS, ids=lambda c: "".join("sd"[not f] for f in c)
)
def test_every_combination_matches_dense(cov_shape, sparse):
    """
    Each of the eight dense/sparse combinations of ``H``, ``S_0``, ``S_z`` gives the dense solve.

    Every ``Posterior`` output is compared, with a ``gamma`` other than 1. ``ids`` read ``H S_0 S_z``: ``s`` sparse, ``d`` dense.
    """
    reference = _problem(cov_shape, (False, False, False))
    problem = _problem(cov_shape, sparse)
    cov_kind = SparseOperator if cov_shape == "banded" else DiagonalOperator
    for data, is_sparse, kind in [
        (problem.forward.data, sparse[0], SparseOperator),
        (problem.prior_cov.data, sparse[1], cov_kind),
        (problem.obs_cov.data, sparse[2], cov_kind),
    ]:
        assert type(data) is (kind if is_sparse else DenseOperator)
    ref = fips.solve(reference, gamma=0.5)
    post = fips.solve(problem, gamma=0.5)
    _assert_same_posterior(post, ref)


def test_block_diagonal_prior_error_matches_dense():
    """A prior error kept as blocks (Kronecker factors and a diagonal) solves as its dense matrix."""
    z, x_0, H, _, S_z, c = _inputs("banded")
    K = KroneckerOperator(
        [spd_banded(np.random.default_rng(30), 4), np.eye(6) + 0.2], std=1.5
    )  # 24 cells
    S_0 = BlockDiagonalOperator([K, DiagonalOperator(np.linspace(1.0, 2.0, N - 24))])
    problem = _labeled(z, x_0, sp.csr_array(H), S_0, S_z, c)
    assert problem.prior_cov.data is S_0
    reference = _labeled(z, x_0, H, S_0.dense(), S_z, c)
    _assert_same_posterior(
        fips.solve(problem, gamma=0.5), fips.solve(reference, gamma=0.5)
    )


def test_all_sparse_diagonal_problem_stays_sparse():
    """Sparse ``H`` with diagonal covariances: the products keep the sparse forms."""
    problem = _problem("diagonal", (True, True, True))
    assert isinstance(problem.forward.data, SparseOperator)
    assert isinstance(problem.prior_cov.data, DiagonalOperator)
    assert isinstance(problem.obs_cov.data, DiagonalOperator)
    post = fips.solve(problem)
    assert isinstance(post._HS_0H, SparseOperator)
    assert isinstance(post._HS_0H + problem.obs_cov.data, SparseOperator)
