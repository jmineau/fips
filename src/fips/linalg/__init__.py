"""
Linear algebra on matrices kept in their cheapest form.

A :class:`~fips.Operator` holds its matrix in one of the forms here, and the
solvers compute with them: a :class:`DenseOperator` (a NumPy array), a
:class:`SparseOperator` (a SciPy sparse matrix), a :class:`DiagonalOperator`
(the diagonal only), a :class:`BlockDiagonalOperator` (a form on each diagonal
block, zeros between them) or a :class:`KroneckerOperator` (Kronecker factors).
None of them has labels: the labels are the
:class:`~fips.Space` on each axis of an :class:`~fips.Operator`.

Each is a :class:`scipy.sparse.linalg.LinearOperator`, so ``matvec``,
``rmatvec``, ``matmat`` and ``.T`` work as in SciPy. On top of that:

- ``A @ B`` of two forms keeps the cheapest form that is exact: diagonal
  times diagonal is diagonal; sparse or diagonal times sparse is sparse; a
  product with a dense side is dense; a Kronecker form times a sparse one
  (``H @ S_0``) is computed through the factors and is dense. ``A @ x`` with an
  array is an array, as in SciPy.
- ``A + B`` and ``A - B`` follow the same rule (an array counts as dense), and
  a scalar ``*`` or ``/`` keeps the form.
- :meth:`~LinearOperator.dense`, :meth:`~LinearOperator.diagonal`,
  :meth:`~LinearOperator.trace`, and for a symmetric positive-definite form
  :meth:`~LinearOperator.solve`, :meth:`~LinearOperator.logdet` and
  :meth:`~LinearOperator.sqrt`, each done the way its form makes cheap.
- ``np.asarray(A)`` is :meth:`~LinearOperator.dense`, so NumPy functions take
  a form as its matrix; ``np.array(A)`` is a copy, as for an array.

A form owns its data, as a :class:`~fips.Vector` does: it copies the array or
matrix it is given and makes the copy read-only, so changing the original
afterwards leaves the form as it was, and writing into the form's own array
raises :class:`ValueError`. The forms fips builds from one another (a product,
a transpose, a selection) hold the arrays they computed or the read-only ones
of their parents, never a second copy.

The forms are real (float64) and pickle with the standard protocol.
"""

from fips.linalg.base import (
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    LinearOperatorLike,
    SparseOperator,
    as_linear_operator,
    dottest,
)
from fips.linalg.block import BlockDiagonalOperator
from fips.linalg.kronecker import KroneckerOperator

__all__ = [
    "LinearOperator",
    "LinearOperatorLike",
    "DenseOperator",
    "SparseOperator",
    "DiagonalOperator",
    "BlockDiagonalOperator",
    "KroneckerOperator",
    "as_linear_operator",
    "dottest",
]
