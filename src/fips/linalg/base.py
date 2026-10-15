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
from typing import Self, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, solve
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


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return `array`, made read-only (a view of a read-only array already is)."""
    array.flags.writeable = False
    return array


def _sparse_array(matrix, copy: bool) -> "sp.csr_array | sp.csc_array":
    """Return a SciPy sparse matrix as a float64 CSR array (CSC for a CSC input), copied if `copy`."""
    kind = sp.csc_array if matrix.format == "csc" else sp.csr_array
    return kind(matrix, dtype=float, copy=copy)


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
    ``_matmat``, from a product with a :class:`SparseOperator`.

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

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle, with the arrays read-only again (a pickle keeps the values, not the flag)."""
        self.__dict__.update(state)
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
        one the root of each block; every other form the lower Cholesky
        factor of its dense matrix, as a :class:`DenseOperator`.

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
    >>> import scipy.sparse as sp
    >>> from fips.linalg import SparseOperator
    >>> A = SparseOperator(sp.csr_matrix([[4.0, 1.0], [1.0, 3.0]]))
    >>> A.diagonal()
    array([4., 3.])
    >>> type(A @ A).__name__
    'SparseOperator'
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
        return self.matrix @ X

    def _rmatmul_array(self, X):
        """Return ``X @ self``."""
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

    def dense(self) -> np.ndarray:
        """Return the form as a dense NumPy array."""
        return self.matrix.toarray()

    def diagonal(self) -> np.ndarray:
        """Return the main diagonal, read from the stored entries."""
        return self.matrix.diagonal()

    def solve(self, B) -> np.ndarray:
        """Solve ``self @ X = B`` by a sparse LU factorization (see :meth:`LinearOperator.solve`)."""
        return self._lu().solve(_as_array(B))

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
