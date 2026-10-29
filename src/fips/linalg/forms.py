"""
The base class of the forms, and the dense, sparse and diagonal forms.

:class:`LinearOperator` is the base of every form. It is in one module with
:class:`DenseOperator`, :class:`SparseOperator` and :class:`DiagonalOperator`
because it builds them (a default method returns a dense form, a product of
sparse matrices a sparse one) while each of them is a subclass of it. The forms
made of other forms are in :mod:`fips.linalg.block` and
:mod:`fips.linalg.kronecker`. What a form can do is in :mod:`fips.linalg`.
"""

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Self, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, eigh, solve
from scipy.linalg.lapack import get_lapack_funcs
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator as _SciPyLinearOperator
from scipy.sparse.linalg import splu

logger = logging.getLogger(__name__)

LinearOperatorLike: TypeAlias = (
    "LinearOperator | sp.spmatrix | sp.sparray | npt.ArrayLike"
)
"""What :func:`as_linear_operator` takes: a form, a SciPy sparse matrix or a 2D array."""

_DENSE_FILL = 0.25
"""
The share of stored entries above which a product of sparse operators is dense.

A product of sparse matrices can fill in (``H S_0 H^T`` for a sparse ``H``
usually does). Past about a quarter of its entries, a dense array is faster to
add and to factor than CSR, and not much bigger.
"""


def _solve_psd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""
    Solve ``A X = B`` for a symmetric positive-definite ``A``, robust to roundoff.

    Covariance-like matrices such as ``A = H S_0 H^T + S_z`` are positive-definite
    in exact arithmetic (``S_z`` is positive-definite), so the fast Cholesky path
    (``assume_a="pos"``) is tried first. When ``A`` is ill-conditioned -- a wide
    eigenvalue spread, with ``H S_0 H^T`` large relative to ``S_z`` -- roundoff in
    forming ``A`` can tip it just off positive-definite, and Cholesky raises
    :class:`~numpy.linalg.LinAlgError`. In that case we symmetrize ``A`` and fall
    back to the symmetric-indefinite ``LDL^T`` factorization (``assume_a="sym"``),
    which tolerates the roundoff and stays backward-stable.

    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix, positive-definite in exact arithmetic.
    B : np.ndarray
        Right-hand side.

    Returns
    -------
    np.ndarray
        Solution ``X`` of ``A X = B``.
    """
    try:
        return solve(A, B, assume_a="pos")
    except LinAlgError:
        logger.warning(
            "Covariance matrix is numerically non-positive-definite (likely "
            "roundoff from a dominant H S_0 H^T term); falling back to a symmetric "
            "LDL^T solve. Inspect problem conditioning if results look unphysical."
        )
        A = 0.5 * (A + A.T)
        return solve(A, B, assume_a="sym")


def _as_array(B) -> np.ndarray:
    """Return `B` (an array, a SciPy sparse matrix or a form) as a dense array."""
    if isinstance(B, LinearOperator):
        return B.dense()
    if sp.issparse(B):
        return B.toarray()
    return np.asarray(B, dtype=float)


def _cholesky(A: np.ndarray) -> np.ndarray:
    """Return the lower Cholesky factor of `A`, raising LinAlgError naming why if it is not positive-definite."""
    try:
        return np.asarray(cholesky(A, lower=True))
    except LinAlgError as err:
        raise LinAlgError(
            f"The matrix is not positive-definite, so it has no Cholesky factor ({err})."
        ) from err


def _clip_eigenvalues(w: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Return the eigenvalues `w` of a covariance with the negative ones set to zero, and whether any was.

    An eigenvalue of a positive semi-definite matrix that comes out negative
    by roundoff is within ``len(w) * eps * max|w|`` of zero (the size LAPACK
    works to); a more negative one means the matrix is not a covariance.

    Raises
    ------
    numpy.linalg.LinAlgError
        If an eigenvalue is more negative than that.
    """
    tol = len(w) * np.finfo(float).eps * np.abs(w).max(initial=0.0)
    if w.min(initial=0.0) < -tol:
        raise LinAlgError(
            f"The matrix is not positive semi-definite, so it is not a covariance: "
            f"its smallest eigenvalue is {w.min():.3g}, more than the roundoff "
            f"{tol:.3g} below zero."
        )
    return np.maximum(w, 0.0), bool((w < 0).any())


def _dense_root(A: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Return a square `R` with ``R R^T = A`` for a symmetric positive semi-definite `A`, and whether it is semi-definite.

    The lower Cholesky factor when `A` has one. A semi-definite `A` (a
    ``Constant()`` block is rank one) has none, so `R` is then
    ``V sqrt(w)`` from its eigendecomposition ``A = V diag(w) V^T``, with the
    eigenvalues that are negative by roundoff set to zero, and the second
    value is True. `linalg` cannot import `methods`, so this calls SciPy's
    ``eigh`` itself, as the analytic method does.

    Raises
    ------
    numpy.linalg.LinAlgError
        If `A` has an eigenvalue that is clearly negative.
    """
    try:
        return np.asarray(cholesky(A, lower=True)), False
    except LinAlgError:
        w, V = eigh(A)
        w, _ = _clip_eigenvalues(np.asarray(w))
        return np.asarray(V) * np.sqrt(w), True


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return `array`, made read-only (a view of a read-only array already is)."""
    array.flags.writeable = False
    return array


_SYMMETRY_RTOL = 1e-10
"""How far from symmetric a covariance may be: ``max|A - A^T| <= 1e-10 * max|A|``."""

_SYMMETRY_CHUNK = 2**22
"""How many entries of a dense matrix the symmetry check reads at once (32 MB of float64)."""


def _entry(A: sp.csr_array, i: int, j: int) -> float:
    """Return entry ``(i, j)`` of a CSR array, read from its arrays (duplicates summed)."""
    row = slice(A.indptr[i], A.indptr[i + 1])
    return float(A.data[row][A.indices[row] == j].sum())


def _asymmetry(matrix) -> tuple[int, int, float, float, float] | None:
    """
    Return the largest asymmetry of a square array or SciPy sparse matrix, if it is past the tolerance.

    Returns ``(i, j, A[i, j], A[j, i], max|A|)`` at the largest
    ``|A[i, j] - A[j, i]|`` when that is more than ``_SYMMETRY_RTOL`` times
    ``max|A|``, else None. A dense matrix is read in blocks of rows and a
    sparse one is subtracted as sparse, so neither is copied whole.
    """
    if sp.issparse(matrix):
        A = sp.csr_array(matrix)
        if A.nnz == 0:
            return None
        scale = float(np.abs(A.data).max())
        D = sp.coo_array(A - A.T)
        if D.nnz == 0:
            return None
        k = int(np.argmax(np.abs(D.data)))
        if abs(D.data[k]) <= _SYMMETRY_RTOL * scale:
            return None
        i, j = int(D.row[k]), int(D.col[k])
        return i, j, _entry(A, i, j), _entry(A, j, i), scale
    A = np.asarray(matrix)
    n = len(A)
    step = max(1, _SYMMETRY_CHUNK // max(n, 1))
    worst, scale, at = 0.0, 0.0, (0, 0)
    for start in range(0, n, step):
        rows = A[start : start + step]
        scale = max(scale, float(np.abs(rows).max()))
        difference = np.abs(rows - A[:, start : start + step].T)
        k = int(np.argmax(difference))
        if difference.flat[k] > worst:
            worst = float(difference.flat[k])
            i, j = np.unravel_index(k, difference.shape)
            at = (start + int(i), int(j))
    if worst <= _SYMMETRY_RTOL * scale:
        return None
    i, j = at
    return i, j, float(A[i, j]), float(A[j, i]), scale


def _sparse_array(matrix, copy: bool) -> "sp.csr_array | sp.csc_array":
    """Return a SciPy sparse matrix as a float64 CSR array (CSC for a CSC input), copied if `copy`."""
    kind = sp.csc_array if matrix.format == "csc" else sp.csr_array
    return kind(matrix, dtype=float, copy=copy)


_THREADED_WORK = 10_000_000
"""
The multiply-adds in a sparse-times-dense product above which it runs on threads.

SciPy's takes about a nanosecond for each, so this is about 10 ms: below it a
pool of threads costs more than it saves.
"""


def _threads() -> int:
    """
    Return how many threads a product may use.

    The CPUs this process may run on, so a Slurm job's and not the node's, and
    no more than ``OMP_NUM_THREADS`` when that is set to a number, as for NumPy's
    BLAS: a run that asks for one thread gets one.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS and Windows have no affinity
        cpus = os.cpu_count() or 1
    asked = os.environ.get("OMP_NUM_THREADS", "")
    return max(1, min(cpus, int(asked))) if asked.isdigit() else cpus


def _csr_times(matrix: "sp.csr_array", X: np.ndarray) -> np.ndarray:
    """
    Return ``matrix @ X`` for a CSR array and a 2D float64 array, over threads when it is large.

    SciPy multiplies a sparse matrix by a dense one on a single thread, row by
    row, and releases the GIL while it does. Here the rows of `matrix` go in
    blocks, one block to a thread, each written to its rows of the result. Row
    ``i`` of the product is a sum over the stored entries of row ``i``, taken in
    the same order whichever rows share the call, so the result is SciPy's bit
    for bit. A small product, one thread, or an array that is not 2D float64 is
    SciPy's own call.
    """
    if type(X) is not np.ndarray or X.ndim != 2 or X.dtype != np.float64:
        return matrix @ X
    threads = _threads()
    if threads == 1 or matrix.nnz * X.shape[1] < _THREADED_WORK:
        return matrix @ X
    X = np.ascontiguousarray(X)  # once for every block, not in each call
    # the rows from the index pointer: pyrefly reads SciPy's `shape` as possibly None
    rows = matrix.indptr.size - 1
    out = np.empty((rows, X.shape[1]))
    # more blocks than threads, so that dense rows do not leave the others idle
    cuts = np.linspace(0, rows, min(4 * threads, rows) + 1).astype(int)

    def block(first: int, last: int) -> None:
        """Write rows `first` to `last` of the product."""
        out[first:last] = matrix[first:last] @ X

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(block, cuts[:-1], cuts[1:]))  # list() raises a block's error
    return out


class LinearOperator(_SciPyLinearOperator):
    """
    A real matrix in one of fips's forms, for multiplying, adding, transposing and solving.

    The base class of :class:`DenseOperator`, :class:`SparseOperator`,
    :class:`DiagonalOperator`, :class:`BlockDiagonalOperator` and
    :class:`~fips.linalg.KroneckerOperator`: build one of those, or let
    :func:`as_linear_operator` choose. A form has no labels; an
    :class:`~fips.Operator` puts a :class:`~fips.Space` on each of its axes.

    A new form subclasses this one and implements ``_matmat`` and
    ``_transpose``. Every other method has a default that is right for any
    form, which the subclass overrides where its form has a cheaper way. A
    form that is neither dense nor sparse also gets a SciPy sparse matrix in
    ``_matmat``, from a product with a :class:`SparseOperator`. A covariance
    form with a root cheaper than the Cholesky factor of its dense matrix
    (the one a draw is made with) overrides ``_make_root``. A form that
    keeps something out of its pickle (a cache) overrides ``__getstate__``
    starting from this class's, which leaves out SciPy's own attributes and
    the root a draw is made with.

    Parameters
    ----------
    shape : tuple of int
        ``(rows, columns)``.

    See Also
    --------
    as_linear_operator : The form that fits an array or a sparse matrix.
    fips.Operator : A form with a space on each axis.
    """

    def __init__(self, shape: tuple[int, ...]):
        super().__init__(dtype=np.float64, shape=shape)

    def _freeze(self) -> None:
        """Make the arrays the form holds read-only; a form that holds none has nothing to do."""

    # A pickle holds the form's attributes, not SciPy's array namespace: SciPy
    # 1.18 keeps that module on each operator, and its own pickle hooks (which
    # 1.17 does not have) cannot read a pickle written without it. Loading runs
    # SciPy's constructor again instead, so a pickle written on any SciPy
    # version loads on any other.
    def __getstate__(self) -> dict:
        """Return the state for pickling, without SciPy's array namespace (a module)."""
        state = self.__dict__.copy()
        state.pop("_xp", None)
        state.pop("_root", None)  # a cache, built again by the next draw
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle, with SciPy's attributes set again and the arrays read-only again (a pickle keeps the values, not the flag)."""
        self.__dict__.update(state)
        LinearOperator.__init__(self, self.shape)
        self._freeze()

    # SciPy is not annotated, so pyrefly takes the type of each default it
    # infers (a private SciPy class) as the type an override must return.
    def _transpose(self) -> "LinearOperator":  # pyrefly: ignore[bad-override]
        """Return the transpose; each form implements it."""
        raise NotImplementedError

    def _matvec(self, x):
        """Return ``self @ x`` for a vector."""
        return self._matmat(x.reshape(-1, 1)).reshape(-1)

    def _rmatvec(self, x):
        """Return ``self.T @ x`` for a vector."""
        return self._transpose().matvec(x)

    def _rmatmat(self, X):
        """Return ``self.T @ X``."""
        return self._transpose().matmat(X)

    def _rmatmul_array(self, X: np.ndarray) -> np.ndarray:
        """Return ``X @ self`` for a 1D or 2D array."""
        if X.ndim == 1:
            return self._transpose().matvec(X)
        return self._transpose().matmat(X.T).T

    def _elementwise(self, func) -> "LinearOperator":
        """
        Return the form with `func` applied to its entries.

        `func` multiplies by a scalar (``a * c``, ``a / c`` or ``-a``), so it
        maps 0 to 0 and a form can apply it to what it stores.
        """
        return DenseOperator._adopt(func(self.dense()))

    def _select(self, rows: np.ndarray, cols: np.ndarray) -> "LinearOperator":
        """
        Return the entries at `rows` and `cols` (integer positions), in that order.

        A position of -1 gives a row or column of zeros. A dense form gives a
        dense one, a sparse or diagonal form a sparse (or diagonal) one.
        """
        rows, cols = np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)
        r, c = np.maximum(rows, 0), np.maximum(cols, 0)
        csr = self._csr()
        if csr is None:
            out = self.dense()[np.ix_(r, c)]
            out[rows < 0] = 0.0
            out[:, cols < 0] = 0.0
            return DenseOperator._adopt(out)
        # SciPy is not annotated, so pyrefly finds no __getitem__ on a sparse array
        out = csr[r][:, c]  # pyrefly: ignore[bad-index]
        if (rows < 0).any() or (cols < 0).any():
            keep_r = sp.diags_array((rows >= 0).astype(float))
            keep_c = sp.diags_array((cols >= 0).astype(float))
            out = sp.csr_array(keep_r @ out @ keep_c)
            out.eliminate_zeros()
        return _adopt(out)

    def _csr(self) -> "sp.csr_array | None":
        """Return the form as a CSR array, or None if it is not sparse."""
        return None

    def dense(self) -> np.ndarray:
        """
        Return the form as a dense NumPy array.

        Returns
        -------
        np.ndarray
            The full matrix. For a :class:`DenseOperator`, the read-only array
            it holds, not a copy; any other form builds a new array.
        """
        return self.matmat(np.eye(self.shape[1]))

    def diagonal(self) -> np.ndarray:
        """
        Return the main diagonal; a sparse or diagonal form reads it from its entries.

        Returns
        -------
        np.ndarray
            A new 1D array of the entries ``A[i, i]``.
        """
        return np.diagonal(self.dense()).copy()

    def trace(self) -> float:
        """
        Return the sum of the main diagonal.

        Returns
        -------
        float
            The trace.
        """
        return float(self.diagonal().sum())

    def solve(self, B) -> np.ndarray:
        """
        Solve ``self @ X = B`` for a symmetric positive-definite form.

        A dense form solves by Cholesky, falling back to a symmetric ``LDL^T``
        solve, with a logged warning, when roundoff has tipped it off
        positive-definite. A diagonal one divides, and raises
        :class:`~numpy.linalg.LinAlgError` on a zero. A sparse one uses SuperLU
        (:func:`scipy.sparse.linalg.splu`): SciPy has no sparse Cholesky, and
        LU keeps the factors sparse, solves every column of `B` with one
        factorization, and does not need positive-definiteness. A block
        diagonal one solves block by block, and a Kronecker one through its
        factors. A singular form raises :class:`~numpy.linalg.LinAlgError` in
        every form.

        Parameters
        ----------
        B : array-like, SciPy sparse matrix or LinearOperator
            The right-hand side: a vector, or a matrix with as many rows as the
            form.

        Returns
        -------
        np.ndarray
            ``X``, dense, with the shape of `B`.
        """
        return _solve_psd(self.dense(), _as_array(B))

    def _solver(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return ``B -> self.solve(B)`` with the factorization made once, for many solves with one form.

        A form that factors inside :meth:`solve` (dense, sparse, block
        diagonal) factors here, and the function it returns only substitutes,
        so a loop over groups of columns pays for one factorization and holds
        no more than the right-hand side of the group at hand. The factor
        lives as long as the function: the form itself keeps none. A form
        that already keeps its factors (diagonal, Kronecker) returns
        :meth:`solve`.
        """
        return self.solve

    def logdet(self) -> float:
        """
        Return the logarithm of the determinant, for a symmetric positive-definite form.

        A dense form takes it from its Cholesky factor, a diagonal one from
        its entries, a block diagonal one block by block, a sparse one from
        its LU factors, and a Kronecker one that holds its whole product from
        the factors (any other dense).

        Returns
        -------
        float
            ``log(det(A))``.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the form is not positive-definite (a diagonal or Cholesky
            factor with an entry that is not positive).
        """
        L = _cholesky(self.dense())
        return float(2.0 * np.log(np.diagonal(L)).sum())

    def sqrt(self) -> "LinearOperator":
        r"""
        Return a square root :math:`L` with :math:`L L^T` equal to this symmetric positive-definite form.

        A diagonal form gives the diagonal of square roots, a block diagonal
        one the root of each block, and a Kronecker one whose rows are the
        whole product a root through its factors' Cholesky factors; every
        other form the lower Cholesky factor of its dense matrix, as a
        :class:`DenseOperator`.

        Returns
        -------
        LinearOperator
            :math:`L`, square.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the form is not positive-definite.
        """
        return DenseOperator._adopt(_cholesky(self.dense()))

    @cached_property
    def _root(self) -> "LinearOperator":
        """
        A matrix ``R`` (``n x r``) with ``R R^T`` equal to this symmetric positive semi-definite form.

        A draw from ``N(0, S)`` is ``R w`` for ``w`` standard normal of length
        ``r``. Built by the first draw and kept (as ``KroneckerOperator._inverses``
        is), so a loop of draws factors once; ``__getstate__`` leaves it out.
        """
        return self._make_root()

    def _make_root(self) -> "LinearOperator":
        """
        Return the root of the form: the Cholesky factor of its dense matrix, or its eigenvalue root when it is semi-definite.

        A form that is cheaper in another way overrides this. Raises
        :class:`~numpy.linalg.LinAlgError` if the form has a clearly negative eigenvalue.
        """
        R, semidefinite = _dense_root(self.dense())
        if semidefinite:
            logger.info(
                "The %d x %d covariance has no Cholesky factor (it is only "
                "positive semi-definite); drawing from its eigendecomposition, "
                "with the eigenvalues below roundoff set to zero.",
                *self.shape,
            )
        return DenseOperator._adopt(R)

    def _sample(self, rng: np.random.Generator) -> np.ndarray:
        """Return one draw from ``N(0, S)``: the root times ``r`` standard normals from `rng`."""
        R = self._root
        return R.matvec(rng.standard_normal(R.shape[1]))

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """
        Return :meth:`dense`, so NumPy functions take a form as its matrix.

        `copy` follows NumPy's protocol: ``np.array(op)`` (``copy=True``) never
        shares memory with the form and is writable, while ``np.asarray(op)``
        gives a :class:`DenseOperator`'s own read-only array. Every other form
        builds a new array either way.
        """
        a = self.dense()
        if dtype is not None and np.dtype(dtype) != a.dtype:
            if copy is False:
                raise ValueError(
                    f"A form is float64: converting it to {np.dtype(dtype)} "
                    "needs a copy."
                )
            return a.astype(dtype)
        if copy and isinstance(self, DenseOperator):
            return a.copy()
        return a

    def dot(self, x):
        """
        Return ``self @ x``, or the form scaled by a scalar ``x``.

        Parameters
        ----------
        x : LinearOperator, SciPy sparse matrix, array-like or scalar
            What to multiply by.

        Returns
        -------
        LinearOperator or np.ndarray
            A form, the cheapest exact one, for a form, a sparse matrix or a
            scalar; an array for an array, as :meth:`matvec` and
            :meth:`matmat` give.
        """
        if isinstance(x, LinearOperator) or sp.issparse(x):
            return _product(self, as_linear_operator(x))
        if np.ndim(x) == 0:  # a scalar, or a 0-d array
            return self._elementwise(lambda a: a * float(x))
        return super().dot(x)

    def __rmatmul__(self, other):
        """Return ``other @ self``."""
        if isinstance(other, LinearOperator) or sp.issparse(other):
            return _product(as_linear_operator(other), self)
        if np.ndim(other) == 0:
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return self._rmatmul_array(np.asarray(other))

    def __rmul__(self, x):
        """Return ``x * self`` for a scalar, else ``x @ self``."""
        if np.ndim(x) == 0:  # a scalar, or a 0-d array
            return self._elementwise(lambda a: float(x) * a)
        return self.__rmatmul__(x)

    # Returns a LinearOperator, not SciPy's private class (see _transpose)
    def __truediv__(self, other):  # pyrefly: ignore[bad-override]
        """Return the form divided by a scalar."""
        if np.ndim(other) != 0:  # a scalar, or a 0-d array
            raise ValueError("Can only divide a form by a scalar.")
        return self._elementwise(lambda a: a / float(other))

    # Returns a LinearOperator, not SciPy's private class (see _transpose)
    def __neg__(self):  # pyrefly: ignore[bad-override]
        """Return ``-self``."""
        return self._elementwise(lambda a: -a)

    def __add__(self, x):
        """Return ``self + x``."""
        return _sum(self, as_linear_operator(x))

    def __radd__(self, other):
        """Return ``other + self``."""
        return _sum(as_linear_operator(other), self)

    def __sub__(self, x):
        """Return ``self - x``."""
        return _sum(self, -as_linear_operator(x))

    def __rsub__(self, other):
        """Return ``other - self``."""
        return _sum(as_linear_operator(other), -self)


class DenseOperator(LinearOperator):
    """
    A matrix held as a dense 2D NumPy array.

    Parameters
    ----------
    array : array-like
        The matrix. It is copied, and the copy is read-only: changing `array`
        afterwards leaves the form as it was.

    Raises
    ------
    ValueError
        If `array` is not 2D.

    Examples
    --------
    >>> import numpy as np
    >>> from fips.linalg import DenseOperator
    >>> A = DenseOperator(np.array([[2.0, 1.0], [1.0, 2.0]]))
    >>> A @ np.array([1.0, 0.0])
    array([2., 1.])
    >>> A.solve(np.array([3.0, 3.0]))
    array([1., 1.])
    """

    array: np.ndarray
    """The matrix, a read-only array."""

    def __init__(self, array: npt.ArrayLike):
        self._hold(np.array(array, dtype=float))

    @classmethod
    def _adopt(cls, array: np.ndarray) -> Self:
        """Return the form that holds `array` itself, which fips just made, without copying it."""
        form = cls.__new__(cls)
        form._hold(array)
        return form

    def _hold(self, array: np.ndarray) -> None:
        """Hold `array`, read-only."""
        array = np.asarray(array, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"A DenseOperator takes a 2D array, got {array.ndim}D.")
        self.array = array
        self._freeze()
        super().__init__(array.shape)

    def _freeze(self) -> None:
        """Make the array read-only."""
        _read_only(self.array)

    def _matvec(self, x):
        """Return ``self @ x`` for a vector."""
        return self.array @ x

    def _matmat(self, X):
        """Return ``self @ X``."""
        return self.array @ X

    def _rmatmul_array(self, X):
        """Return ``X @ self``."""
        return X @ self.array

    def _transpose(self):
        """Return the transpose, a view of the same array."""
        return DenseOperator._adopt(self.array.T)

    def _elementwise(self, func):
        """Return the form with `func` applied to its entries."""
        return DenseOperator._adopt(func(self.array))

    def dense(self) -> np.ndarray:
        """Return the read-only array the form holds, not a copy."""
        return self.array

    def trace(self) -> float:
        """Return the sum of the main diagonal."""
        return float(np.trace(self.array))

    def solve(self, B) -> np.ndarray:
        """Solve ``self @ X = B`` by Cholesky, falling back to ``LDL^T`` (see :meth:`LinearOperator.solve`)."""
        return _solve_psd(self.array, _as_array(B))

    def _solver(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return the substitution of the Cholesky factor (see :meth:`LinearOperator._solver`).

        When roundoff has tipped the matrix off positive-definite it returns
        :meth:`solve`, which falls back to ``LDL^T`` with a logged warning.
        """
        # LAPACK directly: SciPy's cho_factor returns a C-ordered factor, and
        # cho_solve copies it to Fortran order at every call
        potrf, potrs = get_lapack_funcs(("potrf", "potrs"), (self.array,))
        factor, info = potrf(self.array, lower=False, clean=False)
        if info != 0:
            return self.solve

        def substitute(B) -> np.ndarray:
            """Return ``self^-1 B``."""
            B = _as_array(B)
            X, _ = potrs(factor, B.reshape(len(B), -1), lower=False)
            return X.reshape(B.shape)

        return substitute


class SparseOperator(LinearOperator):
    """
    A matrix held as a SciPy sparse matrix.

    Parameters
    ----------
    matrix : SciPy sparse matrix or array
        The matrix. It is copied, as a CSR array (CSC for a CSC input), and
        the copy is read-only: changing `matrix` afterwards leaves the form as
        it was.

    Raises
    ------
    TypeError
        If `matrix` is not a SciPy sparse matrix.

    Examples
    --------
    A product of sparse forms stays sparse while at most a quarter of its
    entries are stored, and is dense past that:

    >>> import scipy.sparse as sp
    >>> from fips.linalg import SparseOperator
    >>> A = SparseOperator(sp.diags([4.0, 3.0, 2.0, 2.0, 1.0, 1.0], format="csr"))
    >>> A.diagonal()
    array([4., 3., 2., 2., 1., 1.])
    >>> type(A @ A).__name__
    'SparseOperator'
    >>> B = SparseOperator(sp.csr_matrix([[4.0, 1.0], [1.0, 3.0]]))
    >>> type(B @ B).__name__
    'DenseOperator'
    """

    matrix: "sp.csr_array | sp.csc_array"
    """The matrix, as a SciPy sparse array whose arrays are read-only."""

    def __init__(self, matrix: "sp.spmatrix | sp.sparray"):
        if not sp.issparse(matrix):
            raise TypeError("A SparseOperator takes a SciPy sparse matrix.")
        self._hold(_sparse_array(matrix, copy=True))

    @classmethod
    def _adopt(cls, matrix: "sp.spmatrix | sp.sparray") -> Self:
        """Return the form that holds `matrix`'s own arrays, which fips just made, without copying them."""
        form = cls.__new__(cls)
        form._hold(_sparse_array(matrix, copy=False))
        return form

    def _hold(self, matrix: "sp.csr_array | sp.csc_array") -> None:
        """Hold `matrix`, read-only."""
        self.matrix = matrix
        self._freeze()
        # pyrefly reads SciPy's unannotated `shape` as possibly None (its initial value)
        super().__init__(matrix.shape)  # pyrefly: ignore[bad-argument-type]

    def _freeze(self) -> None:
        """Make the matrix's data, indices and index pointers read-only."""
        for array in (self.matrix.data, self.matrix.indices, self.matrix.indptr):
            _read_only(array)

    def _matvec(self, x):
        """Return ``self @ x`` for a vector."""
        return self.matrix @ x

    def _matmat(self, X):
        """Return ``self @ X``."""
        if isinstance(self.matrix, sp.csr_array):
            return _csr_times(self.matrix, X)
        return self.matrix @ X

    def _rmatmul_array(self, X):
        """Return ``X @ self``."""
        if isinstance(self.matrix, sp.csc_array) and X.ndim == 2:
            # SciPy computes it as (self.T @ X.T).T, with self.T a CSR array
            return _csr_times(self.matrix.T, X.T).T
        return X @ self.matrix

    def _transpose(self):
        """Return the transpose (CSC for a CSR matrix, without a copy)."""
        return SparseOperator._adopt(self.matrix.T)

    def _elementwise(self, func):
        """Return the form with `func` applied to its stored entries."""
        return SparseOperator._adopt(func(self.matrix))

    def _csr(self):
        """Return the matrix as a CSR array."""
        return sp.csr_array(self.matrix)

    def _lu(self):
        """Return the SuperLU factorization, raising LinAlgError if the matrix is singular."""
        # SciPy has no sparse Cholesky (CHOLMOD is in scikit-sparse, not a
        # dependency). SuperLU keeps the factors sparse, and an LU does not
        # need the matrix to be positive-definite.
        try:
            return splu(sp.csc_matrix(self.matrix))
        except RuntimeError as err:  # SuperLU's "Factor is exactly singular"
            raise LinAlgError(
                f"The sparse matrix is singular: its LU factorization failed ({err})."
            ) from err

    def _make_root(self) -> "SparseOperator":
        """
        Return the root of each independent group of rows, as one sparse matrix.

        The rows are split into the connected components of the sparsity
        pattern (a ``within=`` kernel's groups are a site-day each): rows in
        different components are uncorrelated, so the matrix is block diagonal
        up to a permutation, and the root is the dense root of each block
        (:func:`_dense_root`), put back at the rows' own places. A matrix whose
        rows are all connected has one group, its dense matrix.
        """
        A = sp.csr_array(self.matrix)
        n = self.shape[0]
        graph = A.astype(bool)
        graph.eliminate_zeros()  # a stored zero is not a correlation
        count, label = connected_components(graph, directed=False)
        order = np.argsort(label, kind="stable")  # each group's rows together
        sizes = np.bincount(label, minlength=count)
        ends = np.cumsum(sizes)
        starts = ends - sizes
        # SciPy is not annotated, so pyrefly finds no __getitem__ on a sparse array
        ordered = sp.csr_array(A[order][:, order])  # pyrefly: ignore[bad-index]

        # the groups of one row, all at once: the root is the square root of the entry
        alone = starts[sizes == 1]
        variances, semidefinite = _clip_eigenvalues(ordered.diagonal()[alone])
        roots = np.sqrt(variances)
        rows, cols, vals = [order[alone]], [order[alone]], [roots]
        for a, b in zip(starts[sizes > 1], ends[sizes > 1], strict=True):
            # pyrefly reads a slice of the array as the bare `sparray` (no toarray)
            block = ordered[a:b, a:b].toarray()  # pyrefly: ignore[missing-attribute]
            R, semi = _dense_root(block)
            semidefinite = semidefinite or semi
            i, j = np.nonzero(R)
            rows.append(order[a + i])
            cols.append(order[a + j])
            vals.append(R[i, j])
        if semidefinite:
            logger.info(
                "The %d x %d sparse covariance has a group of rows with no "
                "Cholesky factor (it is only positive semi-definite); drawing "
                "from its eigendecomposition, with the eigenvalues below "
                "roundoff set to zero.",
                n,
                n,
            )
        root = sp.csr_array(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )
        root.eliminate_zeros()
        return SparseOperator._adopt(root)

    def dense(self) -> np.ndarray:
        """Return the form as a dense NumPy array."""
        return self.matrix.toarray()

    def diagonal(self) -> np.ndarray:
        """Return the main diagonal, read from the stored entries."""
        return self.matrix.diagonal()

    def solve(self, B) -> np.ndarray:
        """Solve ``self @ X = B`` by a sparse LU factorization (see :meth:`LinearOperator.solve`)."""
        return self._lu().solve(_as_array(B))

    def _solver(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the LU factorization's substitution (see :meth:`LinearOperator._solver`)."""
        lu = self._lu()
        return lambda B: lu.solve(_as_array(B))

    def logdet(self) -> float:
        """
        Return the log-determinant from a symmetric sparse LU factorization.

        The rows and columns are permuted alike and pivots are taken on the
        diagonal, so the factorization is ``P A P^T = L D L^T`` with ``U = D
        L^T``: the matrix is positive-definite when every pivot is positive.

        Raises
        ------
        numpy.linalg.LinAlgError
            If a pivot is zero or negative: the matrix is not positive-definite.
        """
        try:
            lu = splu(
                sp.csc_matrix(self.matrix),
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        except RuntimeError as err:  # a zero pivot
            raise LinAlgError(
                f"The sparse matrix is not positive-definite ({err})."
            ) from err
        pivots = lu.U.diagonal()
        # SuperLU pivots off the diagonal only on a zero diagonal entry, which a
        # positive-definite matrix never has: the permutations then differ
        if not np.array_equal(lu.perm_r, lu.perm_c) or (pivots <= 0).any():
            raise LinAlgError(
                "The sparse matrix is not positive-definite: a pivot of its "
                "symmetric factorization is not positive."
            )
        return float(np.log(pivots).sum())


class DiagonalOperator(LinearOperator):
    """
    A square matrix that is zero off its main diagonal, held as the diagonal.

    Parameters
    ----------
    values : array-like
        The diagonal, 1D. It is copied, and the copy is read-only.

    Raises
    ------
    ValueError
        If `values` is not 1D.

    Examples
    --------
    >>> from fips.linalg import DiagonalOperator
    >>> D = DiagonalOperator([2.0, 4.0])
    >>> D.solve([1.0, 1.0])
    array([0.5 , 0.25])
    >>> D.trace()
    6.0
    """

    values: np.ndarray
    """The diagonal, a read-only array."""

    def __init__(self, values: npt.ArrayLike):
        self._hold(np.array(values, dtype=float))

    @classmethod
    def _adopt(cls, values: np.ndarray) -> Self:
        """Return the form that holds `values` itself, which fips just made, without copying it."""
        form = cls.__new__(cls)
        form._hold(values)
        return form

    def _hold(self, values: np.ndarray) -> None:
        """Hold `values`, read-only."""
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError(
                f"A DiagonalOperator takes its diagonal, 1D; got {values.ndim}D."
            )
        self.values = values
        self._freeze()
        super().__init__((len(values), len(values)))

    def _freeze(self) -> None:
        """Make the diagonal read-only."""
        _read_only(self.values)

    def _matvec(self, x):
        """Return ``self @ x`` for a vector."""
        return self.values * x

    def _matmat(self, X):
        """Return ``self @ X``: each row of `X` scaled."""
        return self.values[:, None] * X

    def _rmatmul_array(self, X):
        """Return ``X @ self``: each column of `X` scaled."""
        return X * self.values

    def _transpose(self):
        """Return the form itself, which is symmetric."""
        return self

    def _elementwise(self, func):
        """Return the form with `func` applied to the diagonal."""
        return DiagonalOperator._adopt(func(self.values))

    def _csr(self):
        """Return the form as a CSR array that stores its nonzero entries."""
        n = len(self.values)
        i = np.flatnonzero(self.values)
        return sp.csr_array((self.values[i], (i, i)), shape=(n, n))

    def dense(self) -> np.ndarray:
        """Return the form as a dense NumPy array."""
        return np.diag(self.values)

    def diagonal(self) -> np.ndarray:
        """Return a copy of the diagonal."""
        return self.values.copy()

    def solve(self, B) -> np.ndarray:
        """Solve ``self @ X = B`` by dividing each row of `B` by the diagonal."""
        if not np.all(self.values):
            raise LinAlgError("The diagonal has a zero, so the operator is singular.")
        B = _as_array(B)
        return B / (self.values if B.ndim == 1 else self.values[:, None])

    def logdet(self) -> float:
        """Return the sum of the logarithms of the diagonal."""
        if (self.values <= 0).any():
            raise LinAlgError("The diagonal has an entry that is not positive.")
        return float(np.log(self.values).sum())

    def _make_root(self) -> "DiagonalOperator":
        """Return the diagonal of square roots; an entry negative by roundoff is zero (a draw from it is 0)."""
        values, clipped = _clip_eigenvalues(self.values)
        if clipped:
            logger.info(
                "The diagonal covariance has an entry below zero by roundoff; "
                "drawing it as zero."
            )
        return DiagonalOperator._adopt(np.sqrt(values))

    def sqrt(self) -> "DiagonalOperator":
        """Return the diagonal of square roots."""
        if (self.values < 0).any():
            raise LinAlgError("The diagonal has a negative entry.")
        return DiagonalOperator._adopt(np.sqrt(self.values))


def _product(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Return ``a @ b`` in the cheapest form that is exact."""
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Cannot multiply operators of shapes {a.shape} and {b.shape}."
        )
    if isinstance(a, DiagonalOperator) and isinstance(b, DiagonalOperator):
        return DiagonalOperator._adopt(a.values * b.values)
    if isinstance(a, SparseOperator) and isinstance(b, SparseOperator):
        product = a.matrix @ b.matrix
        if product.nnz > _DENSE_FILL * product.shape[0] * product.shape[1]:
            return DenseOperator._adopt(product.toarray())
        return SparseOperator._adopt(product)
    a_csr, b_csr = a._csr(), b._csr()
    if a_csr is not None and b_csr is not None:  # one side diagonal: no fill-in
        return SparseOperator._adopt(a_csr @ b_csr)
    if isinstance(b, DenseOperator):
        return DenseOperator._adopt(a.matmat(b.array))
    if isinstance(a, DenseOperator):
        return DenseOperator._adopt(b._rmatmul_array(a.array))
    # Left: a form of its own (Kronecker, block diagonal) on at least one side.
    # A sparse matrix (a Jacobian) goes through that form's own product, never
    # its dense matrix; anything else gives a product as big as the dense one.
    if isinstance(b, SparseOperator):
        return DenseOperator._adopt(a._matmat(b.matrix))
    if isinstance(a, SparseOperator):
        return DenseOperator._adopt(b._transpose()._matmat(a.matrix.T).T)
    if b_csr is not None:  # a diagonal right side scales the columns
        return DenseOperator._adopt(b._rmatmul_array(a.dense()))
    return DenseOperator._adopt(a.matmat(b.dense()))


def _sum(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """Return ``a + b`` in the cheapest form that is exact."""
    if a.shape != b.shape:
        raise ValueError(f"Cannot add operators of shapes {a.shape} and {b.shape}.")
    if isinstance(a, DiagonalOperator) and isinstance(b, DiagonalOperator):
        return DiagonalOperator._adopt(a.values + b.values)
    a_csr, b_csr = a._csr(), b._csr()
    if a_csr is not None and b_csr is not None:
        return SparseOperator._adopt(a_csr + b_csr)
    return DenseOperator._adopt(a.dense() + b.dense())


def _diagonal_of(matrix) -> np.ndarray | None:
    """Return the diagonal of a square SciPy sparse matrix that stores entries only there, else None."""
    coo = matrix.tocoo()
    if matrix.shape[0] == matrix.shape[1] and np.array_equal(coo.row, coo.col):
        return matrix.diagonal()
    return None


def _matrix_2d(x) -> np.ndarray:
    """Return array-like `x` as a float64 array, raising ValueError unless it is 2D."""
    array = np.asarray(x, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"A matrix takes 2D data, got {array.ndim}D. For a diagonal, "
            "use DiagonalOperator(values)."
        )
    return array


def _adopt(x) -> LinearOperator:
    """
    Return `x`, an array or SciPy sparse matrix that fips just made, as the form that fits it.

    The form holds `x` itself and makes it read-only: no copy, so only for
    what nothing else refers to, or what is read-only already (the arrays of
    another form). What a caller gave fips goes through
    :func:`as_linear_operator`, which copies it. A form comes back as it is.
    """
    if isinstance(x, LinearOperator):
        return x
    if sp.issparse(x):
        diagonal = _diagonal_of(x)
        if diagonal is None:
            return SparseOperator._adopt(x)
        return DiagonalOperator._adopt(diagonal)
    return DenseOperator._adopt(_matrix_2d(x))


def as_linear_operator(x: LinearOperatorLike) -> LinearOperator:
    """
    Return `x` as the form that fits how it is stored.

    Parameters
    ----------
    x : LinearOperator, SciPy sparse matrix or 2D array-like
        What to wrap. A form comes back as it is; an array or a sparse matrix
        is copied, and the form's copy is read-only.

    Returns
    -------
    LinearOperator
        A :class:`DiagonalOperator` for a square sparse matrix that stores
        entries on its diagonal only, a :class:`SparseOperator` for any other
        sparse matrix, and a :class:`DenseOperator` for an array.

    Raises
    ------
    ValueError
        If `x` is not 2D. A diagonal given as its values is a
        :class:`DiagonalOperator`.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from fips.linalg import as_linear_operator
    >>> type(as_linear_operator(np.eye(3))).__name__
    'DenseOperator'
    >>> type(as_linear_operator(sp.eye(3))).__name__
    'DiagonalOperator'
    """
    if isinstance(x, LinearOperator):
        return x
    # isinstance, not issparse: the type checker narrows only this one
    if isinstance(x, (sp.spmatrix, sp.sparray)):
        diagonal = _diagonal_of(x)
        if diagonal is None:
            return SparseOperator(x)
        return DiagonalOperator(diagonal)
    return DenseOperator(_matrix_2d(x))


def dottest(
    op: _SciPyLinearOperator, rtol: float = 1e-12, seed: int | None = 0
) -> bool:
    """
    Check that ``op.rmatvec`` is the adjoint of ``op.matvec``.

    For random ``x`` and ``y``, ``<op x, y>`` must equal ``<x, op^T y>`` to
    within `rtol` of ``|op x| |y| + |x| |op^T y|``, the size of the terms the
    two inner products are summed from.

    Parameters
    ----------
    op : scipy.sparse.linalg.LinearOperator
        The operator to test: any SciPy linear operator, a fips form among
        them.
    rtol : float, default 1e-12
        Relative tolerance.
    seed : int, optional
        Seed of the random vectors.

    Returns
    -------
    bool
        True if the two inner products agree.

    Examples
    --------
    >>> import numpy as np
    >>> from fips.linalg import DenseOperator, dottest
    >>> dottest(DenseOperator(np.arange(6.0).reshape(2, 3)))
    True
    """
    rng = np.random.default_rng(seed)
    x, y = rng.standard_normal(op.shape[1]), rng.standard_normal(op.shape[0])
    Ax, ATy = op.matvec(x), op.rmatvec(y)
    size = np.linalg.norm(Ax) * np.linalg.norm(y) + np.linalg.norm(x) * np.linalg.norm(
        ATy
    )
    return bool(abs(Ax @ y - x @ ATy) <= rtol * size)
