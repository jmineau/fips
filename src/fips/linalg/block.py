"""
Block diagonal matrices: a form on each diagonal block, zeros between.

A :class:`BlockDiagonalOperator` is one of the forms of :mod:`fips.linalg`.
A covariance whose blocks are independent (a flux block and a bias block, say)
keeps each block in its own form, so it costs what its blocks do.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag

from fips.linalg.base import LinearOperator, _as_array, as_linear_operator


class BlockDiagonalOperator(LinearOperator):
    """
    A square matrix made of forms on its diagonal blocks, zero between them.

    A covariance whose blocks are independent (a flux block with Kronecker
    factors and a diagonal bias block, say) keeps each block in its own form:
    products, solves, the diagonal and the log-determinant go block by block,
    and the full matrix is formed only by :meth:`dense`.

    Parameters
    ----------
    blocks : sequence of LinearOperator, SciPy sparse matrix or 2D array
        The diagonal blocks, in order, each square. A form is kept as it is
        (it holds its own copy); an array or a sparse matrix is copied, as
        :func:`as_linear_operator` does.

    Raises
    ------
    ValueError
        If there is no block or a block is not square.

    Examples
    --------
    >>> import numpy as np
    >>> from fips.linalg import BlockDiagonalOperator, DiagonalOperator
    >>> S = BlockDiagonalOperator(
    ...     [np.array([[2.0, 1.0], [1.0, 2.0]]), DiagonalOperator([4.0])]
    ... )
    >>> S.dense()
    array([[2., 1., 0.],
           [1., 2., 0.],
           [0., 0., 4.]])
    >>> S.solve(np.array([3.0, 3.0, 4.0]))
    array([1., 1., 1.])
    """

    blocks: tuple[LinearOperator, ...]
    """The diagonal blocks, each in its own form."""

    def __init__(self, blocks):
        forms = tuple(as_linear_operator(b) for b in blocks)
        if not forms:
            raise ValueError("A BlockDiagonalOperator needs at least one block.")
        for form in forms:
            if form.shape[0] != form.shape[1]:
                raise ValueError(f"Each block must be square; got shape {form.shape}.")
        self.blocks = forms
        self._offsets = np.concatenate([[0], np.cumsum([f.shape[0] for f in forms])])
        n = int(self._offsets[-1])
        super().__init__((n, n))

    def _parts(self):
        """Yield each block with the slice of rows it covers."""
        for k, form in enumerate(self.blocks):
            yield form, slice(int(self._offsets[k]), int(self._offsets[k + 1]))

    def _matmat(self, X):
        """Return ``self @ X``, block by block."""
        out = np.empty((self.shape[0], X.shape[1]))
        if sp.issparse(X):
            X = sp.csr_array(X)
        for form, rows in self._parts():
            part = X[rows]
            csr = form._csr()
            if sp.issparse(part) and csr is not None:  # sparse times sparse
                out[rows] = _as_array(csr @ part)
            else:  # a dense form densifies, a Kronecker one goes by columns
                out[rows] = _as_array(form._matmat(part))
        return out

    def _transpose(self):
        """Return the transpose: each block transposed."""
        return BlockDiagonalOperator([form._transpose() for form in self.blocks])

    def _elementwise(self, func):
        """Return the form with `func` applied to each block."""
        return BlockDiagonalOperator([form._elementwise(func) for form in self.blocks])

    def _select(self, rows, cols):
        """
        Return the entries at `rows` and `cols`.

        The same rows and columns that keep each block's rows together (in
        any order of blocks, and any order within one) keep the blocks: a
        block diagonal form, or the one block that is left.
        """
        rows = np.asarray(rows, dtype=np.intp)
        if np.array_equal(rows, cols) and (rows >= 0).all():
            owner = np.searchsorted(self._offsets, rows, side="right") - 1
            starts = np.flatnonzero(np.diff(owner, prepend=-1))  # where each run begins
            runs = owner[starts]
            if len(np.unique(runs)) == len(runs):  # each block's rows stay together
                kept = []
                for k in runs:
                    mine = rows[owner == k] - self._offsets[k]
                    kept.append(self.blocks[k]._select(mine, mine))
                return kept[0] if len(kept) == 1 else BlockDiagonalOperator(kept)
        return super()._select(rows, cols)

    def _csr(self):
        """Return the form as a CSR array, if every block is sparse or diagonal."""
        parts = [form._csr() for form in self.blocks]
        if any(part is None for part in parts):
            return None
        return sp.csr_array(sp.block_diag(parts, format="csr"))

    def dense(self) -> np.ndarray:
        """Return the form as a dense NumPy array."""
        return block_diag(*(form.dense() for form in self.blocks))

    def diagonal(self) -> np.ndarray:
        """Return the main diagonal, block by block."""
        return np.concatenate([form.diagonal() for form in self.blocks])

    def solve(self, B) -> np.ndarray:
        """Solve ``self @ X = B`` block by block (see :meth:`LinearOperator.solve`)."""
        B = _as_array(B)
        X = np.empty(B.shape)
        for form, rows in self._parts():
            X[rows] = form.solve(B[rows])
        return X

    def logdet(self) -> float:
        """Return the sum of the blocks' log-determinants."""
        return float(sum(form.logdet() for form in self.blocks))

    def sqrt(self) -> "BlockDiagonalOperator":
        """Return the square root of each block."""
        return BlockDiagonalOperator([form.sqrt() for form in self.blocks])
