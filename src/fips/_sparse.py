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
"""

from __future__ import annotations

import pandas as pd
from scipy.sparse import spmatrix

__all__ = ["frame_from_spmatrix", "is_sparse_frame", "normalize_fill_value"]


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
        If a column stores a NaN. Filling would make it a zero, where the
        dense path refuses it.

    Notes
    -----
    Works on the values rather than on the dtype: assigning a new
    ``SparseDtype`` would keep the NaNs as stored entries, whereas filling
    them folds them back into the implicit zeros.

    The check for stored NaN reads each column's stored values only, so it
    never densifies; it runs only when a fill is needed.
    """
    if not is_sparse_frame(data):
        return data
    if all(
        dt.fill_value == 0.0 for dt in data.dtypes if isinstance(dt, pd.SparseDtype)
    ):
        return data
    # fillna would also fill a NaN stored in the matrix, which is data
    if any(_stores_nan(column) for _, column in data.items()):
        raise ValueError("Data contains NaN values.")
    return data.fillna(0.0)


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
        pd.DataFrame.sparse.from_spmatrix(matrix, index=index, columns=columns)
    )
