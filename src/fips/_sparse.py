"""
Sparse storage helpers.

fips stores sparse blocks as pandas sparse DataFrames whose ``fill_value`` is
``0.0``: the implicit entries of a covariance matrix or a Jacobian are zeros,
not missing data. ``Structure2D.values`` relies on that when it calls
``DataFrame.sparse.to_coo()``, and ``_validate`` rejects NaN outright.

``pandas.DataFrame.sparse.from_spmatrix`` used to agree, but as of pandas 3 it
builds float frames with ``fill_value=NaN``:

* announcement: https://pandas.pydata.org/docs/whatsnew/v3.0.0.html#sparse
* change: https://github.com/pandas-dev/pandas/pull/59064
* upstream bug: https://github.com/pandas-dev/pandas/issues/59212

scipy.sparse defines unstored entries as zero, so the conversion loses that.
Left alone, the zeros of a sparse covariance read back as NaN, equality checks
against the dense equivalent fail, and any frame built the ordinary pandas way
is refused by fips as containing NaN.

:func:`normalize_fill_value` re-bases such a frame onto ``fill_value=0.0``.
Sparsity is preserved -- only the fill changes, nothing is densified.

Filling with zero would also turn a NaN that is *stored* in the matrix into a
zero. A stored NaN is data (a missing variance, say), and on the dense path
``_validate`` refuses it. The two kinds of NaN can be told apart before the
fill: a stored entry is in the column's ``sp_values``, an implicit one is not.
So :func:`normalize_fill_value` raises on a stored NaN instead of filling it.

Any other fill (``1.0``, say) cannot be re-based: its implicit entries are not
zeros, and every consumer in fips reads them as zeros (``to_coo()`` on pandas 3
ignores the fill and returns the stored entries only). Changing the dtype to
``fill_value=0.0`` would turn those entries into zeros too. So
:func:`check_fill_value`, which :func:`normalize_fill_value` calls first, raises
on such a fill, and the caller builds the frame with a zero fill or hands over a
dense frame.
"""

from __future__ import annotations

import pandas as pd
from scipy.sparse import spmatrix

__all__ = [
    "check_fill_value",
    "frame_from_spmatrix",
    "is_sparse_frame",
    "normalize_fill_value",
]


def is_sparse_frame(data: pd.DataFrame) -> bool:
    """Return True if every column of ``data`` uses pandas sparse storage."""
    return len(data.columns) > 0 and all(
        isinstance(dt, pd.SparseDtype) for dt in data.dtypes
    )


def _stores_nan(column: pd.Series) -> bool:
    """Return True if a sparse column stores a NaN (its implicit entries aside)."""
    values = column.array
    return isinstance(values, pd.arrays.SparseArray) and bool(
        pd.isna(values.sp_values).any()
    )


def check_fill_value(data: pd.DataFrame) -> None:
    """
    Raise if a sparse column of ``data`` has a fill that is neither 0.0 nor NaN.

    Parameters
    ----------
    data : pandas.DataFrame
        Any DataFrame. Dense columns have no fill to check, so a frame with no
        sparse column never raises.

    Raises
    ------
    ValueError
        If a sparse column has a ``fill_value`` other than 0.0 or NaN: its
        implicit entries are not zeros, and fips reads them as zeros. This
        holds for a frame that mixes sparse and dense columns too.

    Notes
    -----
    :func:`normalize_fill_value` calls this first, so every sparse frame that
    enters a structure is checked. Call it directly where a bare frame skips
    the structures and no re-based frame is needed, as in
    ``ObsAggregator.apply``.
    """
    fills = [dt.fill_value for dt in data.dtypes if isinstance(dt, pd.SparseDtype)]
    nonzero = [fill for fill in fills if not (pd.isna(fill) or fill == 0.0)]
    if nonzero:
        found = ", ".join(sorted({str(fill) for fill in nonzero}))
        raise ValueError(
            "A sparse frame must have fill_value 0.0 (or NaN, which fips reads as "
            f"0.0), but fill_value {found} was found in {len(nonzero)} of "
            f"{len(fills)} sparse columns. fips reads the implicit entries of a "
            "sparse matrix as zeros, as a covariance or a Jacobian has no other "
            "kind, so it cannot take a frame whose implicit entries are not "
            "zeros. Build the frame with fill_value=0.0, or pass a dense frame "
            "(sparse=True stores it sparse). Do not convert it with "
            "astype(pd.SparseDtype(float, fill_value=0.0)): that keeps the stored "
            "entries and turns the implicit ones into zeros."
        )


def normalize_fill_value(data: pd.DataFrame) -> pd.DataFrame:
    """
    Re-base a sparse DataFrame onto ``fill_value=0.0``.

    Parameters
    ----------
    data : pandas.DataFrame
        Any DataFrame. Dense frames and sparse frames that already fill with
        zero are returned unchanged.

    Returns
    -------
    pandas.DataFrame
        A sparse frame whose implicit entries are ``0.0``. Density is
        unchanged; the stored values are the same numbers, and the implicit
        entries, NaN before, are zeros.

    Raises
    ------
    ValueError
        If a sparse column has a ``fill_value`` that is neither 0.0 nor NaN
        (see :func:`check_fill_value`).
    ValueError
        If a column stores a NaN. Filling would make it a zero, where the
        dense path refuses it.

    Notes
    -----
    Works on the values rather than on the dtype: assigning a new
    ``SparseDtype`` would keep the NaNs as stored entries, whereas filling
    them folds them back into the implicit zeros. Every column is filled, one
    by one: a column that stores every row has no NaN to fill, and
    ``DataFrame.fillna`` would leave its ``fill_value`` as NaN, so the frame
    would mix fills (and warn when pandas combines the columns).

    The check for stored NaN reads each column's stored values only, so it
    never densifies; it runs only when a fill is needed.
    """
    check_fill_value(data)
    if not is_sparse_frame(data):
        return data
    if all(
        dt.fill_value == 0.0 for dt in data.dtypes if isinstance(dt, pd.SparseDtype)
    ):
        return data
    # filling would also fill a NaN stored in the matrix, which is data
    if any(_stores_nan(column) for _, column in data.items()):
        raise ValueError("Data contains NaN values.")
    # Fill each column's array, not the frame: on pandas 3 DataFrame.fillna
    # returns a column that holds no NaN as it is, fill_value included, and a
    # column that stores every row holds none. It has no implicit entry, so
    # its new fill changes no cell.
    filled = pd.DataFrame(
        {i: column.array.fillna(0.0) for i, (_, column) in enumerate(data.items())},
        index=data.index,
        copy=False,
    )
    filled.columns = data.columns
    return filled


def frame_from_spmatrix(
    matrix: spmatrix, index: pd.Index, columns: pd.Index
) -> pd.DataFrame:
    """
    Build a sparse DataFrame whose implicit entries are ``0.0`` from a scipy matrix.

    This is ``pandas.DataFrame.sparse.from_spmatrix`` followed by
    :func:`normalize_fill_value`, which pandas 3 makes necessary. Build every
    sparse frame from a scipy matrix here, so none skips the second step.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        The matrix to wrap. Its unstored entries are zeros.
    index : pandas.Index
        Row labels.
    columns : pandas.Index
        Column labels.

    Returns
    -------
    pandas.DataFrame
        A sparse frame with ``fill_value=0.0`` and the density of `matrix`.

    Raises
    ------
    ValueError
        If `matrix` stores a NaN and pandas built the frame with
        ``fill_value=NaN`` (pandas 3). On pandas 2 the NaN stays in the frame,
        and a structure built from it refuses it.
    """
    return normalize_fill_value(
        # pandas-stubs do not type DataFrame.sparse
        # pyrefly: ignore[missing-attribute]
        pd.DataFrame.sparse.from_spmatrix(matrix, index=index, columns=columns)
    )
