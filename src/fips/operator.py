"""
Operators: matrices with a space of labels on each axis.

An :class:`Operator` is a linear map from one :class:`~fips.Space` (its
columns) to another (its rows): a forward operator from the state to the
observations, an aggregation, a gain. Its matrix is kept in the cheapest form
that is exact (:mod:`fips.linalg`): a dense array, a SciPy sparse matrix, a
diagonal, Kronecker factors. The labels are beside the matrix, never inside
it, and every product, sum and selection checks them: two operators combine
only when their spaces have the same labels, in any order (an exact join).

A DataFrame is a way in (:meth:`Operator.from_frame`) and a way out
(:meth:`Operator.to_frame`), not the store.
"""

import logging
from collections.abc import Hashable, Mapping
from typing import Any, Self

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fips.linalg import (
    BlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    LinearOperatorLike,
    SparseOperator,
    as_linear_operator,
)
from fips.linalg.base import _adopt
from fips.space import Space, _listed
from fips.vector import Vector

logger = logging.getLogger(__name__)


def _stored_values(form: LinearOperator) -> np.ndarray:
    """
    Return the values a form stores: a dense one's array, a sparse one's entries, a diagonal.

    A block diagonal form gives its blocks' values; any other form (Kronecker
    factors, which check themselves when built) gives none.
    """
    if isinstance(form, DenseOperator):
        return form.array.ravel()
    if isinstance(form, SparseOperator):
        return form.matrix.data
    if isinstance(form, DiagonalOperator):
        return form.values
    if isinstance(form, BlockDiagonalOperator):
        return np.concatenate([_stored_values(block) for block in form.blocks])
    return np.empty(0)


def _axis_space(
    index: pd.Index, block: Hashable | None, axis: str
) -> tuple[Space, np.ndarray]:
    """
    Return the space of one axis of a frame, and the order that puts the frame's labels in it.

    With a ``block`` level, the blocks come from it (:meth:`Space.from_index`)
    and the rows of each block are put together; without one, the axis is the
    one block named `block`.
    """
    argument = "row_block" if axis == "rows" else "col_block"
    if "block" in index.names:
        if block is not None:
            raise ValueError(
                f"The frame's {axis} have a 'block' level, which names their "
                f"blocks; leave `{argument}` out."
            )
        space = Space.from_index(index)
        rank = pd.Index(space.names).get_indexer(
            index.get_level_values("block").astype(str)
        )
        return space, np.argsort(rank, kind="stable")
    if block is None:
        raise ValueError(
            f"The frame's {axis} have no 'block' level, so `{argument}` must "
            "name their one block."
        )
    return Space({str(block): index}), np.arange(len(index))


def _frame_matrix(frame: pd.DataFrame) -> "np.ndarray | sp.csr_array":
    """
    Return a frame's entries as an array, or as a CSR array if a column is pandas-sparse.

    The implicit entries of a sparse column are zeros: its fill value must be
    0 or NaN (pandas 3 fills a frame from ``from_spmatrix`` with NaN).

    Raises
    ------
    ValueError
        If a sparse column has another fill value, or an entry is NaN.
    """
    if not any(isinstance(dt, pd.SparseDtype) for dt in frame.dtypes):
        return frame.to_numpy(dtype=float)
    rows, cols, values = [], [], []
    for j, (label, column) in enumerate(frame.items()):
        array = column.array
        if isinstance(array, pd.arrays.SparseArray):
            fill = array.fill_value
            if not (fill == 0 or pd.isna(fill)):
                raise ValueError(
                    f"Column {label!r} is sparse with fill value {fill!r}. fips "
                    "reads the entries a sparse column does not store as zeros, "
                    "so its fill value must be 0 (or NaN)."
                )
            stored = array.sp_index.indices
            entries = np.asarray(array.sp_values, dtype=float)
        else:
            dense = column.to_numpy(dtype=float)
            stored = np.flatnonzero(dense)  # NaN is nonzero, so it is kept
            entries = dense[stored]
        rows.append(stored)
        cols.append(np.full(len(stored), j))
        values.append(entries)
    return sp.csr_array(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=frame.shape,
    )


def _permuted(
    matrix: "np.ndarray | sp.csr_array", rows: np.ndarray, cols: np.ndarray
) -> "np.ndarray | sp.csr_array":
    """Return `matrix` with its rows taken in the order `rows` and its columns in the order `cols`."""
    if not sp.issparse(matrix):
        return np.asarray(matrix)[np.ix_(rows, cols)]
    new_row, new_col = np.empty_like(rows), np.empty_like(cols)
    new_row[rows], new_col[cols] = np.arange(len(rows)), np.arange(len(cols))
    coo = sp.coo_array(matrix)
    return sp.csr_array(
        (coo.data, (new_row[coo.row], new_col[coo.col])), shape=matrix.shape
    )


def _where(target: pd.Index, given: pd.Index, block: str, what: str) -> np.ndarray:
    """
    Return where each label of `given` is among the labels `target` of `block`.

    Raises
    ------
    ValueError
        If `given` has other levels than `target`, or a label `target` does
        not have; the message names them.
    """
    if set(given.names) != set(target.names):
        raise ValueError(
            f"The {what} of block {block!r} have levels {list(given.names)}; the "
            f"block has {list(target.names)}."
        )
    if isinstance(given, pd.MultiIndex) and given.names != target.names:
        given = given.reorder_levels(list(target.names))
    found = target.get_indexer(given)
    if (found < 0).any():
        raise ValueError(
            f"The {what} of block {block!r} have labels the block does not "
            f"have: {_listed(given[found < 0])}."
        )
    return found


def _placed(
    op: "Operator", rows: pd.Index, cols: pd.Index, key: tuple[str, str]
) -> LinearOperator:
    """
    Return the form of `op` (on one row and one column block) placed on the labels `rows` and `cols`.

    A label `op` has no entry for is a zero row or column. When `op` covers
    every label, its form is kept (reordered); otherwise it is sparse if it
    was (or is diagonal), dense if not.
    """
    r, c = key
    where = f"matrix of ({r!r}, {c!r})"
    r_pos = _where(rows, op.rows.select(r), r, f"rows of the {where}")
    c_pos = _where(cols, op.cols.select(c), c, f"columns of the {where}")
    form = op.data
    if len(r_pos) == len(rows) and len(c_pos) == len(cols):
        if np.array_equal(r_pos, np.arange(len(rows))) and np.array_equal(
            c_pos, np.arange(len(cols))
        ):
            return form
        inverse_r, inverse_c = np.empty_like(r_pos), np.empty_like(c_pos)
        inverse_r[r_pos], inverse_c[c_pos] = (
            np.arange(len(r_pos)),
            np.arange(len(c_pos)),
        )
        return form._select(inverse_r, inverse_c)
    csr = form._csr()
    if csr is not None:
        coo = sp.coo_array(csr)
        return SparseOperator._adopt(
            sp.csr_array(
                (coo.data, (r_pos[coo.row], c_pos[coo.col])),
                shape=(len(rows), len(cols)),
            )
        )
    dense = np.zeros((len(rows), len(cols)))
    dense[np.ix_(r_pos, c_pos)] = form.dense()
    return DenseOperator._adopt(dense)


class Operator:
    """
    A matrix with a :class:`~fips.Space` of labels on each axis.

    The matrix is kept in its cheapest form (:attr:`data`, one of the forms
    of :mod:`fips.linalg`): a dense array, a SciPy sparse matrix, a diagonal,
    Kronecker factors. Its rows are labeled by :attr:`rows` and its columns
    by :attr:`cols`, in the order of each space. Products, sums and
    selections check the labels: two operators combine only when the spaces
    they meet on have the same labels, in any order (an exact join).

    An operator is not changed in place: every method returns a new one.
    NumPy defers to it (``__array_ufunc__ = None``), so ``np.float64(2) * op``
    is an operator and ``array + op`` raises, as ``op + array`` does.

    Parameters
    ----------
    data : array-like, SciPy sparse matrix or LinearOperator
        The matrix, with rows in the order of `rows` and columns in the
        order of `cols`. An array or a sparse matrix is copied, and the copy
        is read-only: changing `data` afterwards leaves the operator as it
        was. A sparse matrix stays sparse (a diagonal one becomes a
        :class:`~fips.linalg.DiagonalOperator`); a form of :mod:`fips.linalg`
        already holds its own copy and is kept as it is. A DataFrame is
        refused, since its labels would be ignored: use :meth:`from_frame`.
    rows : Space
        The labels of the rows.
    cols : Space
        The labels of the columns.
    name : str, optional
        What the matrix is ("forward", "prior_cov", ...). A name that is not
        a string is converted to one.

    Raises
    ------
    TypeError
        If `data` is a pandas object, or `rows` or `cols` is not a Space.
    ValueError
        If the shape of `data` does not match the spaces, or it stores a NaN.

    See Also
    --------
    Operator.from_frame : An operator from a labeled DataFrame.
    Operator.from_blocks : An operator from its blocks.
    fips.Covariance : A symmetric operator on one space.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from fips import Operator, Space, Vector
    >>> state = Space({"flux": pd.Index(["a", "b"], name="cell")})
    >>> obs = Space({"co2": pd.Index([1, 2, 3], name="obs_id")})
    >>> H = Operator(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), obs, state)
    >>> H.shape
    (3, 2)
    >>> x = Vector([2.0, 3.0], state)
    >>> (H @ x).values
    array([2., 3., 5.])
    """

    __array_ufunc__ = None
    """NumPy defers to the operator's own arithmetic, which checks labels."""

    def __init__(
        self,
        data: LinearOperatorLike,
        rows: Space,
        cols: Space,
        name: str | None = None,
    ):
        if isinstance(data, (pd.DataFrame, pd.Series, pd.Index)):
            raise TypeError(
                f"{type(self).__name__} takes the matrix as an array, in the "
                "spaces' order. Build one from a DataFrame with "
                f"{type(self).__name__}.from_frame, which reads its labels."
            )
        for space, what in ((rows, "rows"), (cols, "cols")):
            if not isinstance(space, Space):
                raise TypeError(f"{what} must be a Space, got {type(space).__name__}.")
        form = as_linear_operator(data)
        if form.shape != (len(rows), len(cols)):
            raise ValueError(
                f"The matrix has shape {form.shape}, but the spaces have "
                f"{len(rows)} rows and {len(cols)} columns."
            )
        if np.isnan(_stored_values(form)).any():
            raise ValueError("The matrix has NaN entries.")
        self._data = form
        self._rows = rows
        self._cols = cols
        self._name = None if name is None else str(name)

    @classmethod
    def _build(
        cls, data: LinearOperatorLike, rows: Space, cols: Space, name: str | None
    ) -> Self:
        """Return an instance of this class on `rows` and `cols`; a constructor's last step."""
        return cls(data, rows, cols, name=name)

    def _like(self, data: LinearOperatorLike, rows: Space, cols: Space) -> "Operator":
        """Return an operator like this one (its class where the spaces allow, its name) with new data or labels."""
        return Operator(data, rows, cols, name=self._name)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        row_block: str | None = None,
        col_block: str | None = None,
        name: str | None = None,
    ) -> Self:
        """
        Build an operator from a DataFrame, reading its labels.

        Parameters
        ----------
        frame : pd.DataFrame
            The matrix. An axis with a ``block`` level has those blocks (see
            :meth:`Space.from_index`), and the rows (or columns) of each
            block are put together, in the order the blocks first appear; an
            axis without one is one block, named by `row_block` or
            `col_block`. A frame with a pandas-sparse column is kept sparse;
            the entries a sparse column does not store are zeros.
        row_block, col_block : str, optional
            The name of the one block of an axis without a ``block`` level.
        name : str, optional
            The operator's name.

        Returns
        -------
        Operator
            The matrix on the spaces of the frame's labels.

        Raises
        ------
        ValueError
            If an axis with a ``block`` level is given its block name too, or
            one without is not; its labels cannot make a :class:`Space`; a
            sparse column has a fill value other than 0 or NaN; or an entry
            is NaN.

        Examples
        --------
        >>> import pandas as pd
        >>> from fips import Operator
        >>> frame = pd.DataFrame(
        ...     [[1.0, 0.5], [0.0, 2.0]],
        ...     index=pd.Index(["o1", "o2"], name="obs_id"),
        ...     columns=pd.Index(["a", "b"], name="cell"),
        ... )
        >>> H = Operator.from_frame(frame, row_block="co2", col_block="flux")
        >>> H.rows
        Space(co2: 2 x ['obs_id'])
        >>> H.to_frame().index.names
        FrozenList(['block', 'obs_id'])
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"from_frame takes a DataFrame, got {type(frame).__name__}."
            )
        rows, row_order = _axis_space(frame.index, row_block, "rows")
        cols, col_order = _axis_space(frame.columns, col_block, "columns")
        # the permuted matrix is a new array, not a view of the frame's
        matrix = _adopt(_permuted(_frame_matrix(frame), row_order, col_order))
        return cls._build(matrix, rows, cols, name)

    @classmethod
    def from_blocks(
        cls,
        blocks: Mapping[tuple[str, str], Any],
        rows: Space | None = None,
        cols: Space | None = None,
        name: str | None = None,
    ) -> Self:
        """
        Build an operator from its blocks, each named by its row and its column block.

        A pair of blocks that is not given is zero. A labeled block is placed
        by its labels: it may cover part of its blocks' labels (the rest is
        zero), as a transport Jacobian covers the cells a footprint touched.

        Parameters
        ----------
        blocks : mapping of (str, str) to a matrix
            Each ``(row_block, col_block)`` pair mapped to its matrix, labeled
            or not. Labeled: a DataFrame (dense or pandas-sparse) whose index
            and columns are the blocks' own labels (no ``block`` level); a
            tuple ``(matrix, row_labels, col_labels)`` of an array or a SciPy
            sparse matrix and two pandas Indexes; or an :class:`Operator` on
            a row space of the one block ``row_block`` and a column space of
            the one block ``col_block``. Unlabeled: an array, a SciPy sparse
            matrix or a form of :mod:`fips.linalg`, whose rows and columns are
            in the order of the blocks' labels in `rows` and `cols`. An array
            or a sparse matrix is copied, as for :class:`Operator`.
        rows, cols : Space, optional
            The labels of the rows and columns. By default each block's labels
            are those of the first labeled matrix in its row or column; an
            unlabeled matrix needs its blocks' labels from here or from a
            labeled matrix in the same row or column block.
        name : str, optional
            The operator's name.

        Returns
        -------
        Operator
            The assembled matrix: sparse if every block is sparse (or
            diagonal), dense otherwise. A labeled block keeps its form
            when it covers its blocks' labels (a pandas-sparse frame is
            sparse); a single block that covers both spaces keeps its form.

        Raises
        ------
        ValueError
            If a pair is given twice, a block's labels are found nowhere, a
            labeled matrix has a label its block does not have (the message
            names it), a labeled matrix is on other blocks than its key
            names, or an unlabeled matrix has the wrong shape.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fips import Operator, Space
        >>> obs = pd.Index(["o1", "o2"], name="obs_id")
        >>> H = Operator.from_blocks(
        ...     {
        ...         ("co2", "flux"): pd.DataFrame(
        ...             [[1.0, 0.5], [0.0, 2.0]],
        ...             index=obs,
        ...             columns=pd.Index(["a", "b"], name="cell"),
        ...         ),
        ...         ("co2", "bias"): pd.DataFrame(
        ...             1.0, index=obs, columns=pd.Index(["A"], name="site")
        ...         ),
        ...     }
        ... )
        >>> H.cols
        Space(flux: 2 x ['cell'], bias: 1 x ['site'])
        >>> H.dense()
        array([[1. , 0.5, 1. ],
               [0. , 2. , 1. ]])

        A Jacobian of the cells a footprint touched, placed on every cell:

        >>> cells = Space({"flux": pd.Index(["a", "b", "c"], name="cell")})
        >>> J = pd.DataFrame(
        ...     [[0.5], [0.2]], index=obs, columns=pd.Index(["c"], name="cell")
        ... )
        >>> Operator.from_blocks({("co2", "flux"): J}, cols=cells).dense()
        array([[0. , 0. , 0.5],
               [0. , 0. , 0.2]])
        """
        labeled: dict[tuple[str, str], Operator] = {}
        unlabeled: dict[tuple[str, str], LinearOperator] = {}
        for key, block in blocks.items():
            r, c = (str(part) for part in key)
            if (r, c) in labeled or (r, c) in unlabeled:
                raise ValueError(f"The pair ({r!r}, {c!r}) is given twice.")
            if isinstance(block, pd.DataFrame):
                block = Operator.from_frame(block, row_block=r, col_block=c)
            elif isinstance(block, tuple):
                if len(block) != 3:
                    raise ValueError(
                        f"The matrix of ({r!r}, {c!r}) is a tuple of {len(block)}; a "
                        "labeled matrix is (matrix, row_labels, col_labels)."
                    )
                matrix, row_labels, col_labels = block
                block = Operator(matrix, Space({r: row_labels}), Space({c: col_labels}))
            if isinstance(block, Operator):
                if block.rows.names != (r,) or block.cols.names != (c,):
                    raise ValueError(
                        f"The matrix of ({r!r}, {c!r}) is on rows {block.rows.names} "
                        f"and columns {block.cols.names}; it must be on the one "
                        "block its key names."
                    )
                labeled[(r, c)] = block
            else:
                unlabeled[(r, c)] = as_linear_operator(block)

        def labels(given: Space | None, axis: int) -> Space:
            """Return the space of one axis: `given`, or the labeled blocks' labels."""
            if given is not None:
                return given
            found: dict[str, pd.Index] = {}
            for key, op in labeled.items():
                space = op.rows if axis == 0 else op.cols
                found.setdefault(key[axis], space.select(key[axis]))
            for key in [*labeled, *unlabeled]:
                if key[axis] not in found:
                    keyword = "rows" if axis == 0 else "cols"
                    raise ValueError(
                        f"Block {key[axis]!r} has no labels: give a labeled matrix "
                        f"for one of its pairs, or `{keyword}`."
                    )
            order = list(dict.fromkeys(key[axis] for key in blocks))
            return Space({str(b): found[str(b)] for b in order})

        row_space, col_space = labels(rows, 0), labels(cols, 1)
        forms: dict[tuple[str, str], LinearOperator] = {}
        for (r, c), op in labeled.items():
            forms[(r, c)] = _placed(
                op, row_space.select(r), col_space.select(c), (r, c)
            )
        for (r, c), form in unlabeled.items():
            shape = (len(row_space.select(r)), len(col_space.select(c)))
            if form.shape != shape:
                raise ValueError(
                    f"The matrix of ({r!r}, {c!r}) has shape {form.shape}; its "
                    f"blocks have {shape[0]} and {shape[1]} labels."
                )
            forms[(r, c)] = form

        if len(forms) == 1 and len(row_space.names) == len(col_space.names) == 1:
            return cls._build(next(iter(forms.values())), row_space, col_space, name)
        parts = {key: form._csr() for key, form in forms.items()}
        if all(part is not None for part in parts.values()):
            grid = [
                [parts.get((r, c)) for c in col_space.names] for r in row_space.names
            ]
            # bmat needs a size for each row and column of blocks
            for i, r in enumerate(row_space.names):
                for j, c in enumerate(col_space.names):
                    if grid[i][j] is None and (i == 0 or j == 0):
                        grid[i][j] = sp.csr_array(
                            (len(row_space.select(r)), len(col_space.select(c)))
                        )
            matrix = sp.csr_array(sp.bmat(grid, format="csr"))
            return cls._build(_adopt(matrix), row_space, col_space, name)
        dense = np.zeros((len(row_space), len(col_space)))
        for (r, c), form in forms.items():
            dense[row_space.slice(r), col_space.slice(c)] = form.dense()
        return cls._build(_adopt(dense), row_space, col_space, name)

    @property
    def data(self) -> LinearOperator:
        """
        The matrix, in its form (a :mod:`fips.linalg` operator, without labels).

        The rows are in the order of :attr:`rows` and the columns in the
        order of :attr:`cols`. The form's arrays are read-only.
        """
        return self._data

    @property
    def rows(self) -> Space:
        """The labels of the rows."""
        return self._rows

    @property
    def cols(self) -> Space:
        """The labels of the columns."""
        return self._cols

    @property
    def name(self) -> str | None:
        """What the matrix is: a string, or None."""
        return self._name

    @property
    def shape(self) -> tuple[int, int]:
        """The number of rows and of columns."""
        return (len(self._rows), len(self._cols))

    @property
    def T(self) -> "Operator":
        """The transpose: the columns become the rows, labels and all."""
        return self._like(self._data._transpose(), self._cols, self._rows)

    def __getitem__(self, key: tuple[str, str]) -> "Operator":
        """
        Return the block of one row block and one column block.

        Parameters
        ----------
        key : (str, str)
            The names of the row block and the column block.

        Returns
        -------
        Operator
            The block, on spaces of those single blocks, labeled as each
            block was given. A diagonal block of a :class:`~fips.Covariance`
            is a Covariance, and keeps the form (Kronecker factors stay
            factors).

        Raises
        ------
        KeyError
            If either space has no block of that name.
        """
        r, c = key
        rs, cs = self._rows.slice(r), self._cols.slice(c)
        form = self._data._select(
            np.arange(rs.start, rs.stop), np.arange(cs.start, cs.stop)
        )
        rows = Space({r: self._rows.select(r)})
        cols = Space({c: self._cols.select(c)})
        return self._like(form, rows, cols)

    def _on(self, rows: Space, cols: Space) -> LinearOperator:
        """
        Return the form with its rows and columns in the order of `rows` and `cols`.

        Raises
        ------
        ValueError
            If `rows` or `cols` do not have exactly this operator's labels
            (:meth:`Space.align`).
        """
        r = rows.align(self._rows) if rows is not self._rows else None
        c = cols.align(self._cols) if cols is not self._cols else None
        if (r is None or np.array_equal(r, np.arange(len(r)))) and (
            c is None or np.array_equal(c, np.arange(len(c)))
        ):
            return self._data
        r = np.arange(self.shape[0]) if r is None else r
        c = np.arange(self.shape[1]) if c is None else c
        return self._data._select(r, c)

    def align(self, rows: Space, cols: Space) -> "Operator":
        """
        Return the operator on `rows` and `cols`, which must have exactly its labels.

        Parameters
        ----------
        rows, cols : Space
            The labels to put the rows and columns on, in any order (see
            :meth:`Space.align`).

        Returns
        -------
        Operator
            The same matrix, its rows and columns reordered.

        Raises
        ------
        ValueError
            If a space does not have the same labels as the operator's; the
            message names those missing from either.
        """
        return self._like(self._on(rows, cols), rows, cols)

    def round_labels(self, decimals: int) -> "Operator":
        """
        Return the operator with its float labels rounded (:meth:`Space.round_labels`).

        Parameters
        ----------
        decimals : int
            Number of decimal places.

        Returns
        -------
        Operator
            The same matrix on the rounded labels.

        Raises
        ------
        ValueError
            If rounding makes two labels of a block equal.
        """
        rows = self._rows.round_labels(decimals)
        cols = rows if self._cols is self._rows else self._cols.round_labels(decimals)
        return self._like(self._data, rows, cols)

    def __matmul__(self, other: "Operator | Vector") -> "Operator | Vector":
        """
        Return the product with an operator or a vector, aligned by label.

        Parameters
        ----------
        other : Operator or Vector
            An operator whose rows, or a vector whose space, have exactly the
            labels of this operator's columns, in any order.

        Returns
        -------
        Operator or Vector
            An operator on this one's rows and `other`'s columns, in the
            cheapest form that is exact (:mod:`fips.linalg`), or a vector on
            this one's rows.

        Raises
        ------
        TypeError
            If `other` is an array: it has no labels to check. Multiply
            :attr:`data` by it to compute by position.
        ValueError
            If the labels do not match exactly.
        """
        if isinstance(other, Vector):
            values = self._data @ other.align(self._cols).values
            return Vector(values, self._rows)
        if isinstance(other, Operator):
            # Put this operator's columns in the order of the other's rows:
            # the other may be a form (Kronecker) that a reordering would densify
            left = self._on(self._rows, other.rows)
            return Operator(left @ other.data, self._rows, other.cols)
        raise TypeError(
            f"An Operator multiplies an Operator or a Vector, not "
            f"{type(other).__name__}: an array has no labels to check. Use "
            "op.data @ x to compute by position."
        )

    def _combine(self, other: "Operator", sign: float) -> "Operator":
        """Return ``self + sign * other``, with `other` aligned to this operator's labels."""
        if not isinstance(other, Operator):
            raise TypeError(
                f"An Operator adds an Operator, not {type(other).__name__}."
            )
        theirs = other._on(self._rows, self._cols)
        total = self._data + theirs if sign > 0 else self._data - theirs
        name = self._name if self._name == other.name else None
        return self._sum_like(other, total, name)

    def _sum_like(
        self, other: "Operator", total: LinearOperator, name: str | None
    ) -> "Operator":
        """Return the sum `total` of this operator and `other`: an operator."""
        return Operator(total, self._rows, self._cols, name=name)

    def __add__(self, other: "Operator") -> "Operator":
        """Return the sum with an operator on the same labels (in any order)."""
        if not isinstance(other, Operator):
            # a covariance model adds itself (CovarianceModel.__radd__); for
            # anything else Python raises a TypeError
            return NotImplemented
        return self._combine(other, 1.0)

    def __sub__(self, other: "Operator") -> "Operator":
        """Return the difference with an operator on the same labels (in any order)."""
        return self._combine(other, -1.0)

    def __mul__(self, scalar: float) -> "Operator":
        """Return the operator times a scalar, in the same form."""
        if np.ndim(scalar) != 0:
            raise TypeError("An Operator is multiplied by a scalar with '*'; use '@'.")
        return self._like(self._data * float(scalar), self._rows, self._cols)

    def __rmul__(self, scalar: float) -> "Operator":
        """Return a scalar times the operator."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Operator":
        """Return the operator divided by a scalar."""
        if np.ndim(scalar) != 0:
            raise TypeError("An Operator can only be divided by a scalar.")
        return self._like(self._data / float(scalar), self._rows, self._cols)

    def __neg__(self) -> "Operator":
        """Return ``-self``."""
        return self._like(-self._data, self._rows, self._cols)

    def dense(self) -> np.ndarray:
        """
        Return the matrix as a new dense array, rows and columns in the spaces' order.

        This forms the full matrix, whatever its form.
        """
        return np.array(self._data)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """
        Return the dense matrix, so NumPy functions take an operator as its matrix.

        ``np.asarray(op)`` is a dense operator's own read-only array (any
        other form builds a new one); ``np.array(op)`` is a writable copy.
        """
        return self._data.__array__(dtype=dtype, copy=copy)

    def to_frame(self) -> pd.DataFrame:
        """
        Return the matrix as a dense DataFrame.

        The index is :attr:`rows`'s :attr:`~Space.index` and the columns
        :attr:`cols`'s, each with the ``block`` level first. This forms the
        full matrix, whatever its form.

        Returns
        -------
        pd.DataFrame
            A copy of the matrix, labeled.
        """
        return pd.DataFrame(
            self.dense(), index=self._rows.index, columns=self._cols.index
        )

    def __repr__(self) -> str:
        """Return the name, shape and form, then each space."""
        return (
            f"{type(self).__name__}(name={self._name!r}, shape={self.shape}, "
            f"form={type(self._data).__name__})\n"
            f"  rows: {self._rows!r}\n  cols: {self._cols!r}"
        )
