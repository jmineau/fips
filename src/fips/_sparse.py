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
        unchanged; the values are the same numbers, with NaN read as zero.

    Notes
    -----
    Works on the values rather than on the dtype: assigning a new
    ``SparseDtype`` would keep the NaNs as stored entries, whereas filling
    them folds them back into the implicit zeros.
    """
    if not is_sparse_frame(data):
        return data
    if all(
        dt.fill_value == 0.0 for dt in data.dtypes if isinstance(dt, pd.SparseDtype)
    ):
        return data
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
    """
    return normalize_fill_value(
        pd.DataFrame.sparse.from_spmatrix(matrix, index=index, columns=columns)
    )
