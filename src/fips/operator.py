"""
Linear operators: the matrices of an inverse problem in their cheapest form.

A structure (:class:`~fips.Matrix`, :class:`~fips.CovarianceMatrix`, ...) keeps
its labels in a pandas DataFrame. Its :attr:`~fips.base.Structure2D.operator`
gives the same matrix as an :class:`Operator`, which the estimators compute
with: a :class:`DenseOperator` (a NumPy array), a :class:`SparseOperator` (a
SciPy sparse matrix) or a :class:`DiagonalOperator` (the diagonal only). A
covariance kept as Kronecker factors is a
:class:`~fips.covariance.KroneckerOperator`, and a structure holds one through
:meth:`~fips.base.Structure2D.from_operator`.

Each is a :class:`scipy.sparse.linalg.LinearOperator`, so ``matvec``,
``rmatvec``, ``matmat`` and ``.T`` work as in SciPy. On top of that:

- ``A @ B`` of two operators keeps the cheapest form that is exact: diagonal
  times diagonal is diagonal; sparse or diagonal times sparse is sparse; a
  product with a dense side is dense; a Kronecker operator times a sparse one
  (``H @ S_0``) is computed through the factors and is dense. ``A @ x`` with an
  array is an array, as in SciPy.
- ``A + B`` and ``A - B`` follow the same rule (an array counts as dense), and
  a scalar ``*`` or ``/`` keeps the form.
- :meth:`~Operator.dense`, :meth:`~Operator.diagonal`, :meth:`~Operator.trace`,
  :meth:`~Operator.to_frame`, and :meth:`~Operator.solve` for a symmetric
  positive-definite operator, each done the way its form makes cheap.
- ``np.asarray(A)`` is :meth:`~Operator.dense`, so NumPy functions take an
  operator as its matrix; ``np.array(A)`` is a copy, as for an array.

The operators are real (float64) and pickle with the standard protocol.
"""

import logging
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from scipy.sparse.linalg import LinearOperator, splu

from fips._sparse import frame_from_spmatrix

if TYPE_CHECKING:
    from fips.base import Structure2D

logger = logging.getLogger(__name__)

OperatorLike: TypeAlias = (
    "Operator | sp.spmatrix | sp.sparray | npt.ArrayLike | Structure2D"
)
"""What :func:`as_operator` takes: an operator, a SciPy sparse matrix, a 2D array or a 2D structure."""

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
    """Return `B` (an array, a SciPy sparse matrix or an operator) as a dense array."""
    if isinstance(B, Operator):
        return B.dense()
    if sp.issparse(B):
        return B.toarray()
    return np.asarray(B, dtype=float)


class Operator(LinearOperator):
    """
    A real linear operator that fips can multiply, add, transpose and solve with.

    The base class of :class:`DenseOperator`, :class:`SparseOperator` and
    :class:`DiagonalOperator`: build one of those, or let :func:`as_operator`
    choose. A new form subclasses this one and implements ``_matmat`` and
    ``_transpose``. Every other method has a default that is right for any
    form, which the subclass overrides where its form has a cheaper way. A
    form that is neither dense nor sparse (a
    :class:`~fips.covariance.KroneckerOperator`) also gets a SciPy sparse
    matrix in ``_matmat``, from a product with a :class:`SparseOperator`.

    Parameters
    ----------
    shape : tuple of int
        ``(rows, columns)``.

    See Also
    --------
    as_operator : The operator that fits an array, a sparse matrix or a structure.
    fips.base.Structure2D.operator : A structure's data as an operator.
    """

    def __init__(self, shape: tuple[int, ...]):
        super().__init__(dtype=np.float64, shape=shape)

    # SciPy is not annotated, so pyrefly takes the type of each default it
    # infers (a private SciPy class) as the type an override must return.
    def _transpose(self) -> "Operator":  # pyrefly: ignore[bad-override]
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

    def _elementwise(self, func) -> "Operator":
        """
        Return the operator with `func` applied to its entries.

        `func` multiplies by a scalar (``a * c``, ``a / c`` or ``-a``), so it
        maps 0 to 0 and a form can apply it to what it stores.
        """
        return DenseOperator(func(self.dense()))

    def _select(self, rows: np.ndarray, cols: np.ndarray) -> "Operator":
        """Return the operator's entries at `rows` and `cols` (integer positions), in that order."""
        csr = self._csr()
        if csr is None:
            return DenseOperator(self.dense()[np.ix_(rows, cols)])
        # SciPy is not annotated, so pyrefly finds no __getitem__ on a sparse array
        return _from_sparse(csr[rows][:, cols])  # pyrefly: ignore[bad-index]

    def _csr(self) -> "sp.csr_array | None":
        """Return the operator as a CSR array, or None if its form is dense."""
        return None

    def dense(self) -> np.ndarray:
        """
        Return the operator as a dense NumPy array.

        Returns
        -------
        np.ndarray
            The full matrix. For a :class:`DenseOperator`, the array it holds,
            not a copy.
        """
        return self.matmat(np.eye(self.shape[1]))

    def diagonal(self) -> np.ndarray:
        """
        Return the main diagonal; a sparse or diagonal operator reads it from its entries.

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
        Solve ``self @ X = B`` for a symmetric positive-definite operator.

        A dense operator solves by Cholesky, falling back to a symmetric
        ``LDL^T`` solve, with a logged warning, when roundoff has tipped it off
        positive-definite. A diagonal one divides, and raises
        :class:`~numpy.linalg.LinAlgError` on a zero. A sparse one uses SuperLU
        (:func:`scipy.sparse.linalg.splu`): SciPy has no sparse Cholesky, and
        LU keeps the factors sparse, solves every column of `B` with one
        factorization, and does not need positive-definiteness. A singular
        operator raises :class:`~numpy.linalg.LinAlgError` in every form.

        Parameters
        ----------
        B : array-like, SciPy sparse matrix or Operator
            The right-hand side: a vector, or a matrix with as many rows as the
            operator.

        Returns
        -------
        np.ndarray
            ``X``, dense, with the shape of `B`.
        """
        return _solve_psd(self.dense(), _as_array(B))

    def to_frame(
        self, index: pd.Index, columns: pd.Index | None = None
    ) -> pd.DataFrame:
        """
        Return the operator as a labeled DataFrame.

        Parameters
        ----------
        index : pd.Index
            Labels of the rows.
        columns : pd.Index, optional
            Labels of the columns; `index` if not given.

        Returns
        -------
        pd.DataFrame
            A dense frame for a dense operator. A pandas sparse frame (implicit
            entries 0.0) for a sparse or diagonal one, built without forming
            the dense matrix.
        """
        columns = index if columns is None else columns
        csr = self._csr()
        if csr is None:
            return pd.DataFrame(self.dense(), index=index, columns=columns)
        return frame_from_spmatrix(sp.csr_matrix(csr), index, columns)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """
        Return :meth:`dense`, so NumPy functions take an operator as its matrix.

        `copy` follows NumPy's protocol: ``np.array(op)`` (``copy=True``) never
        shares memory with the operator, while ``np.asarray(op)`` gives a
        :class:`DenseOperator`'s own array. Every other form builds a new
        array either way.
        """
        a = self.dense()
        if dtype is not None and np.dtype(dtype) != a.dtype:
            if copy is False:
                raise ValueError(
                    f"An operator is float64: converting it to {np.dtype(dtype)} "
                    "needs a copy."
                )
            return a.astype(dtype)
        if copy and isinstance(self, DenseOperator):
            return a.copy()
        return a

    def dot(self, x):
        """
        Return ``self @ x``, or the operator scaled by a scalar ``x``.

        Parameters
        ----------
        x : Operator, SciPy sparse matrix, array-like or scalar
            What to multiply by.

        Returns
        -------
        Operator or np.ndarray
            An operator, in the cheapest exact form, for an operator, a sparse
            matrix or a scalar; an array for an array, as :meth:`matvec` and
            :meth:`matmat` give.
        """
        if isinstance(x, Operator) or sp.issparse(x):
            return _product(self, as_operator(x))
        if np.ndim(x) == 0:  # a scalar, or a 0-d array
            return self._elementwise(lambda a: a * float(x))
        return super().dot(x)

    def __rmatmul__(self, other):
        """Return ``other @ self``."""
        if isinstance(other, Operator) or sp.issparse(other):
            return _product(as_operator(other), self)
        if np.ndim(other) == 0:
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return self._rmatmul_array(np.asarray(other))

    def __rmul__(self, x):
        """Return ``x * self`` for a scalar, else ``x @ self``."""
        if np.ndim(x) == 0:  # a scalar, or a 0-d array
            return self._elementwise(lambda a: float(x) * a)
        return self.__rmatmul__(x)

    # Returns an Operator, not SciPy's private class (see _transpose)
    def __truediv__(self, other):  # pyrefly: ignore[bad-override]
        """Return the operator divided by a scalar."""
        if np.ndim(other) != 0:  # a scalar, or a 0-d array
            raise ValueError("Can only divide an operator by a scalar.")
        return self._elementwise(lambda a: a / float(other))

    # Returns an Operator, not SciPy's private class (see _transpose)
    def __neg__(self):  # pyrefly: ignore[bad-override]
        """Return ``-self``."""
        return self._elementwise(lambda a: -a)

    def __add__(self, x):
        """Return ``self + x``."""
        return _sum(self, as_operator(x))

    def __radd__(self, other):
        """Return ``other + self``."""
        return _sum(as_operator(other), self)

    def __sub__(self, x):
        """Return ``self - x``."""
        return _sum(self, -as_operator(x))

    def __rsub__(self, other):
        """Return ``other - self``."""
        return _sum(as_operator(other), -self)


class DenseOperator(Operator):
    """
    An operator held as a dense 2D NumPy array.

    Parameters
    ----------
    array : array-like
        The matrix. A float64 array is kept as it is, not copied.

    Examples
    --------
    >>> import numpy as np
    >>> from fips.operator import DenseOperator
    >>> A = DenseOperator(np.array([[2.0, 1.0], [1.0, 2.0]]))
    >>> A @ np.array([1.0, 0.0])
    array([2., 1.])
    >>> A.solve(np.array([3.0, 3.0]))
    array([1., 1.])
    """

    array: np.ndarray
    """The matrix."""

    def __init__(self, array: npt.ArrayLike):
        array = np.asarray(array, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"A DenseOperator takes a 2D array, got {array.ndim}D.")
        self.array = array
        super().__init__(array.shape)

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
        return DenseOperator(self.array.T)

    def _elementwise(self, func):
        """Return the operator with `func` applied to its entries."""
        return DenseOperator(func(self.array))

    def dense(self) -> np.ndarray:
        """Return the array the operator holds, not a copy."""
        return self.array

    def trace(self) -> float:
        """Return the sum of the main diagonal."""
        return float(np.trace(self.array))

    def solve(self, B) -> np.ndarray:
        """Solve ``self @ X = B`` by Cholesky, falling back to ``LDL^T`` (see :meth:`Operator.solve`)."""
        return _solve_psd(self.array, _as_array(B))


class SparseOperator(Operator):
    """
    An operator held as a SciPy sparse matrix.

    Parameters
    ----------
    matrix : SciPy sparse matrix or array
        The matrix. It is kept as a CSR array (CSC for a CSC input), sharing
        the input's data when that is already float64 in that format.

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> from fips.operator import SparseOperator
    >>> A = SparseOperator(sp.csr_matrix([[4.0, 1.0], [1.0, 3.0]]))
    >>> A.diagonal()
    array([4., 3.])
    >>> type(A @ A).__name__
    'SparseOperator'
    """

    matrix: "sp.csr_array | sp.csc_array"
    """The matrix, as a SciPy sparse array."""

    def __init__(self, matrix: "sp.spmatrix | sp.sparray"):
        if not sp.issparse(matrix):
            raise TypeError("A SparseOperator takes a SciPy sparse matrix.")
        if isinstance(matrix, (sp.csc_matrix, sp.csc_array)):
            self.matrix = sp.csc_array(matrix, dtype=float)
        else:
            self.matrix = sp.csr_array(matrix, dtype=float)
        # pyrefly reads SciPy's unannotated `shape` as possibly None (its initial value)
        super().__init__(self.matrix.shape)  # pyrefly: ignore[bad-argument-type]

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
        return SparseOperator(self.matrix.T)

    def _elementwise(self, func):
        """Return the operator with `func` applied to its stored entries."""
        return SparseOperator(func(self.matrix))

    def _csr(self):
        """Return the matrix as a CSR array."""
        return sp.csr_array(self.matrix)

    def dense(self) -> np.ndarray:
        """Return the operator as a dense NumPy array."""
        return self.matrix.toarray()

    def diagonal(self) -> np.ndarray:
        """Return the main diagonal, read from the stored entries."""
        return self.matrix.diagonal()

    def solve(self, B) -> np.ndarray:
        """Solve ``self @ X = B`` by a sparse LU factorization (see :meth:`Operator.solve`)."""
        # SciPy has no sparse Cholesky (CHOLMOD is in scikit-sparse, not a
        # dependency). SuperLU keeps the factors sparse, and an LU does not
        # need the matrix to be positive-definite.
        try:
            lu = splu(sp.csc_matrix(self.matrix))
        except RuntimeError as err:  # SuperLU's "Factor is exactly singular"
            raise LinAlgError(
                f"The sparse matrix is singular: its LU factorization failed ({err})."
            ) from err
        return lu.solve(_as_array(B))


class DiagonalOperator(Operator):
    """
    A square operator that is zero off its main diagonal, held as the diagonal.

    Parameters
    ----------
    values : array-like
        The diagonal, 1D.

    Examples
    --------
    >>> from fips.operator import DiagonalOperator
    >>> D = DiagonalOperator([2.0, 4.0])
    >>> D.solve([1.0, 1.0])
    array([0.5 , 0.25])
    >>> D.trace()
    6.0
    """

    values: np.ndarray
    """The diagonal."""

    def __init__(self, values: npt.ArrayLike):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError(
                f"A DiagonalOperator takes its diagonal, 1D; got {values.ndim}D."
            )
        self.values = values
        super().__init__((len(values), len(values)))

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
        """Return the operator itself, which is symmetric."""
        return self

    def _elementwise(self, func):
        """Return the operator with `func` applied to the diagonal."""
        return DiagonalOperator(func(self.values))

    def _csr(self):
        """Return the operator as a CSR array that stores its nonzero entries."""
        n = len(self.values)
        i = np.flatnonzero(self.values)
        return sp.csr_array((self.values[i], (i, i)), shape=(n, n))

    def dense(self) -> np.ndarray:
        """Return the operator as a dense NumPy array."""
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


def _product(a: Operator, b: Operator) -> Operator:
    """Return ``a @ b`` in the cheapest form that is exact."""
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Cannot multiply operators of shapes {a.shape} and {b.shape}."
        )
    if isinstance(a, DiagonalOperator) and isinstance(b, DiagonalOperator):
        return DiagonalOperator(a.values * b.values)
    if isinstance(a, SparseOperator) and isinstance(b, SparseOperator):
        product = a.matrix @ b.matrix
        if product.nnz > _DENSE_FILL * product.shape[0] * product.shape[1]:
            return DenseOperator(product.toarray())
        return SparseOperator(product)
    a_csr, b_csr = a._csr(), b._csr()
    if a_csr is not None and b_csr is not None:  # one side diagonal: no fill-in
        return SparseOperator(a_csr @ b_csr)
    if isinstance(b, DenseOperator):
        return DenseOperator(a.matmat(b.array))
    if isinstance(a, DenseOperator):
        return DenseOperator(b._rmatmul_array(a.array))
    # Left: a form of its own (a Kronecker covariance) on at least one side. A
    # sparse matrix (a Jacobian) goes through that form's own product, never
    # its dense matrix; anything else gives a product as big as the dense one.
    if isinstance(b, SparseOperator):
        return DenseOperator(a._matmat(b.matrix))
    if isinstance(a, SparseOperator):
        return DenseOperator(b._transpose()._matmat(a.matrix.T).T)
    if b_csr is not None:  # a diagonal right side scales the columns
        return DenseOperator(b._rmatmul_array(a.dense()))
    return DenseOperator(a.matmat(b.dense()))


def _sum(a: Operator, b: Operator) -> Operator:
    """Return ``a + b`` in the cheapest form that is exact."""
    if a.shape != b.shape:
        raise ValueError(f"Cannot add operators of shapes {a.shape} and {b.shape}.")
    if isinstance(a, DiagonalOperator) and isinstance(b, DiagonalOperator):
        return DiagonalOperator(a.values + b.values)
    a_csr, b_csr = a._csr(), b._csr()
    if a_csr is not None and b_csr is not None:
        return SparseOperator(a_csr + b_csr)
    return DenseOperator(a.dense() + b.dense())


def _from_sparse(matrix) -> Operator:
    """Return a SciPy sparse matrix as a DiagonalOperator if it is one, else a SparseOperator."""
    coo = matrix.tocoo()
    if matrix.shape[0] == matrix.shape[1] and np.array_equal(coo.row, coo.col):
        return DiagonalOperator(matrix.diagonal())
    return SparseOperator(matrix)


def as_operator(x: OperatorLike) -> Operator:
    """
    Return `x` as the :class:`Operator` that fits how it is stored.

    Parameters
    ----------
    x : Operator, SciPy sparse matrix, 2D array-like, or a 2D fips structure
        What to wrap. An operator comes back as it is, and a structure gives
        its :attr:`~fips.base.Structure2D.operator`.

    Returns
    -------
    Operator
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
    >>> from fips.operator import as_operator
    >>> type(as_operator(np.eye(3))).__name__
    'DenseOperator'
    >>> type(as_operator(sp.eye(3))).__name__
    'DiagonalOperator'
    """
    from fips.base import Structure2D  # fips.base imports this module

    if isinstance(x, Operator):
        return x
    if isinstance(x, Structure2D):
        return x.operator
    if sp.issparse(x):
        return _from_sparse(x)
    array = np.asarray(x, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"An operator takes 2D data, got {array.ndim}D. For a diagonal, "
            "use DiagonalOperator(values)."
        )
    return DenseOperator(array)


def dottest(op: LinearOperator, rtol: float = 1e-12, seed: int | None = 0) -> bool:
    """
    Check that ``op.rmatvec`` is the adjoint of ``op.matvec``.

    For random ``x`` and ``y``, ``<op x, y>`` must equal ``<x, op^T y>`` to
    within `rtol` of ``|op x| |y| + |x| |op^T y|``, the size of the terms the
    two inner products are summed from.

    Parameters
    ----------
    op : LinearOperator
        The operator to test.
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
    >>> from fips.operator import DenseOperator, dottest
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
