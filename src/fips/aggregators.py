"""
Data aggregation utilities for inverse problems.

This module provides functions for aggregating and integrating data over time
and space, particularly useful for processing observations and state vectors
into compatible resolutions.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from fips.covariance import Covariance
from fips.linalg import as_linear_operator
from fips.operator import Operator, _axis_space, _frame_matrix
from fips.space import Space
from fips.vector import Block, Vector


def integrate_over_time_bins(
    data: pd.DataFrame | pd.Series, time_bins: pd.IntervalIndex, time_dim: str = "time"
) -> pd.DataFrame | pd.Series:
    """
    Integrate data over time bins.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Data to integrate.
    time_bins : pd.IntervalIndex
        Time bins for integration.
    time_dim : str, optional
        Time dimension name, by default 'time'

    Returns
    -------
    pd.DataFrame | pd.Series
        Integrated footprint. The bin labels are set to the left edge of the bin.
    """
    is_series = isinstance(data, pd.Series)

    dims = data.index.names
    if time_dim not in dims:
        raise ValueError(f"time_dim '{time_dim}' not found in data index levels {dims}")
    other_levels = [lvl for lvl in dims if lvl != time_dim]

    data = data.reset_index()

    # Use pd.cut to bin the data by time into time bins.
    # Cast the time column to match the resolution of time_bins (e.g. datetime64[us]
    # vs datetime64[ns]) to avoid a ValueError on pandas 3.x when the dtypes differ.
    time_col = data[time_dim]
    if hasattr(time_bins, "dtype") and hasattr(time_bins.dtype, "subtype"):
        target_dtype = time_bins.dtype.subtype
        if hasattr(time_col, "dt") and time_col.dtype != target_dtype:
            time_col = time_col.astype(target_dtype)
    data[time_dim] = pd.cut(time_col, bins=time_bins, include_lowest=True, right=False)

    # Set Intervals to the left edge of the bin (start of time interval).
    # Use cat.rename_categories() instead of .apply() so that NaN values
    # (observations outside all bins) are left untouched — .apply() in
    # pandas 3.x also passes np.nan through the mapper, which has no .left.
    data[time_dim] = data[time_dim].cat.rename_categories(lambda x: x.left)

    # Group the date by the time bins & any other existing levels
    grouped = data.groupby([time_dim] + other_levels, observed=True)

    # Sum over the groups
    integrated = grouped.sum()

    # Order the index levels if MultiIndex
    if isinstance(integrated.index, pd.MultiIndex):
        integrated = integrated.reorder_levels(list(dims))

    if is_series:
        # Return a Series if the input was a Series
        return integrated.iloc[:, 0]
    return integrated


class ObsAggregator:
    """
    Aggregates the observation space of an inverse problem.

    Builds a sparse (n_agg x n_obs) weight matrix W and applies it to each
    component of the problem::

        z_agg = W @ z  # aggregated observations
        H_agg = W @ H  # aggregated forward operator
        S_z_agg = W @ S_z @ W.T  # covariance propagation
        c_agg = W @ c  # aggregated constant (if vector)

    For ``func='mean'`` each non-zero entry in row i equals 1/nᵢ (the
    reciprocal of the group size), so ``W @ z`` yields group means and
    ``W @ S_z @ W.T`` scales variances by 1/nᵢ². For ``func='sum'`` every
    entry is 1. Only ``'mean'`` and ``'sum'`` are supported because other
    functions do not have a well-defined covariance propagation rule.

    Grouping interface
    ------------------
    Exactly one of ``by`` or ``level`` must be provided:

    - ``by`` : an index level name (str), a list of level names, or a
      callable that accepts the obs ``pd.Index`` and returns group labels.
    - ``level`` + ``freq`` : resample a datetime index level at the given
      pandas offset alias (e.g. ``level='obs_time', freq='D'``). All other
      index levels are preserved as exact-match grouping keys.

    When the obs index has a ``'block'`` level it is always prepended as a
    grouping key, ensuring observations from different blocks are never
    merged.

    Partial aggregation
    -------------------
    ``blocks`` restricts aggregation to the named block(s). Observations
    belonging to other blocks are passed through unchanged via identity rows
    in W, so the returned arrays cover the full observation space.

    Parameters
    ----------
    by : str | list[str] | Callable, optional
        Explicit grouping specification. Mutually exclusive with ``level``.
    level : str, optional
        Index level to group / resample. Requires either a matching level
        name in the obs index or use alongside ``freq``.
    freq : str, optional
        Pandas offset alias for resampling ``level`` (e.g. ``'D'``, ``'h'``).
    func : {'mean', 'sum'}
        Aggregation function. Default ``'mean'``.
    blocks : str | list[str], optional
        Block name(s) to aggregate. Unlisted blocks pass through as-is.

    Apply to inverse problem components
    -----------------------------------
    apply(obs, forward_operator, modeldata_mismatch, constant)
        Apply the aggregation to the inverse problem components.
    """

    by: str | list[str] | Callable | None
    """Explicit grouping specification. Mutually exclusive with 'level'."""
    level: str | None
    """Index level to group / resample. Requires either a matching level name in the obs index or use alongside 'freq'."""
    freq: str | None
    """Pandas offset alias for resampling 'level' (e.g. 'D', 'h')."""
    func: str
    """Aggregation function, either 'mean' or 'sum'. Default is 'mean'."""
    blocks: list[str] | None
    """Block name(s) to aggregate. Unlisted blocks pass through as-is."""

    def __init__(
        self,
        by: str | list[str] | Callable | None = None,
        level: str | None = None,
        freq: str | None = None,
        func: str = "mean",
        blocks: str | list[str] | None = None,
    ):
        if func not in {"mean", "sum"}:
            raise ValueError(
                "func must be 'mean' or 'sum' for valid covariance propagation."
            )

        if by is None and level is None:
            raise ValueError("Must provide either 'by' or 'level'.")

        self.by = by
        self.level = level
        self.freq = freq
        self.func = func
        self.blocks = [blocks] if isinstance(blocks, str) else blocks

    def _build_operator(self, obs_index: pd.Index) -> tuple[csr_matrix, pd.Index]:
        """
        Build W and the aggregated index from an obs ``pd.Index``.

        Constructs W in COO format (data, row, col) — one entry per input
        observation — then converts to CSR for efficient matrix products.
        Returns the (n_agg x n_obs) weight matrix and the new row index.
        """
        n_obs = len(obs_index)

        # Which observations are targeted for aggregation?
        # Untargeted observations become identity (passthrough) rows in W.
        if self.blocks is not None and "block" in obs_index.names:
            mask = obs_index.get_level_values("block").isin(self.blocks)
        else:
            mask = np.ones(n_obs, dtype=bool)

        # COO triplets — one entry per obs; col_indices[i] = i always
        # (each obs maps to exactly one output row).
        row_indices = np.empty(n_obs, dtype=int)
        col_indices = np.arange(n_obs)
        data = np.empty(n_obs, dtype=float)

        new_index_list = []  # index labels for each output row
        n_agg_targets = 0  # rows produced by aggregation (excludes passthrough)

        # --- Build aggregated rows for the targeted block(s) ---
        if mask.any():
            target_idx = obs_index[mask]
            # dummy_series is just a carrier for the index so we can use groupby
            dummy_series = pd.Series(np.arange(len(target_idx)), index=target_idx)

            # Build grouping keys. "block" is always first to prevent
            # cross-block merging even when other index values coincide.
            keys = []
            if "block" in target_idx.names:
                keys.append(pd.Grouper(level="block"))

            if self.by is not None:
                # Use explicit user instructions if provided
                user_keys = self.by if isinstance(self.by, (list, tuple)) else [self.by]
                for k in user_keys:
                    if isinstance(k, str):
                        keys.append(pd.Grouper(level=k))
                    elif callable(k):
                        keys.append(k(target_idx))
                    else:
                        keys.append(k)
            else:
                for lvl in target_idx.names:
                    if lvl == "block":
                        continue  # already grouped as the first key
                    if lvl == self.level:
                        if self.freq is not None:
                            keys.append(pd.Grouper(level=lvl, freq=self.freq))
                        else:
                            keys.append(pd.Grouper(level=lvl))
                    else:
                        keys.append(pd.Grouper(level=lvl))

            # pandas groupby requires a scalar (not a list) for a single key
            if len(keys) == 1:
                keys = keys[0]

            grouper = dummy_series.groupby(keys, sort=True)

            # ngroup() maps each obs to a consecutive integer row id in W.
            agg_target_idx = grouper.size().index
            target_group_ids = grouper.ngroup().to_numpy()
            n_agg_targets = len(agg_target_idx)

            row_indices[mask] = target_group_ids

            if self.func == "mean":
                counts = np.bincount(target_group_ids, minlength=n_agg_targets)
                data[mask] = 1.0 / counts[target_group_ids]  # weight = 1/nᵢ
            else:  # "sum"
                data[mask] = 1.0

            new_index_list.append(agg_target_idx)

        # --- Identity passthrough rows for untargeted observations ---
        if (~mask).any():
            other_idx = obs_index[~mask]
            n_other = (~mask).sum()

            # Row ids continue after the aggregated rows.
            other_group_ids = np.arange(n_agg_targets, n_agg_targets + n_other)
            row_indices[~mask] = other_group_ids
            data[~mask] = 1.0

            new_index_list.append(other_idx)

        W = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_agg_targets + (~mask).sum(), n_obs),
        )

        # Aggregated rows first, passthrough rows second.
        if len(new_index_list) == 1:
            agg_index = new_index_list[0]
        else:
            agg_index = new_index_list[0].append(new_index_list[1])

        return W, agg_index

    def apply(
        self,
        obs: pd.Series | Block | Vector,
        forward_operator: "pd.DataFrame | Operator",
        modeldata_mismatch: "pd.DataFrame | Operator",
        constant: float | pd.Series | Block | Vector | None = None,
    ):
        """
        Apply W to the inverse problem components.

        Inputs may be bare pandas objects or fips types (``Vector``,
        :class:`~fips.Operator`, :class:`~fips.Covariance`); return types
        mirror the inputs. See the class docstring for the mathematical
        transforms.

        The rows of the forward operator and both axes of the mismatch are
        matched to the observations by label before W is applied: an
        operator must have exactly the observations' labels (in any order),
        and a DataFrame is reindexed onto them, zero where it has no entry.

        Parameters
        ----------
        obs : pd.Series | Block | Vector
            Observation vector to be aggregated.
        forward_operator : pd.DataFrame | Operator
            Forward operator matrix to be aggregated.
        modeldata_mismatch : pd.DataFrame | Operator
            Model-data mismatch covariance matrix to be aggregated.
        constant : float | pd.Series | Block | Vector | None, optional
            Optional constant offset vector to be aggregated. Scalars are invariant to aggregation. Default is None.

        Returns
        -------
        tuple
            Aggregated (obs, forward_operator, modeldata_mismatch, constant) in the same types as the inputs.

        Raises
        ------
        ValueError
            If an operator's rows do not have exactly the observations'
            labels, or a DataFrame has a sparse column whose `fill_value` is
            not 0.0 or NaN: its implicit entries are not zeros, which is how
            the aggregation reads them.
        """
        z = obs.data if isinstance(obs, (Block, Vector)) else obs
        if len(z) == 0:
            raise ValueError("Input data contains empty dimensions, cannot aggregate.")
        W, agg_idx = self._build_operator(z.index)
        z_agg = pd.Series(W @ z.to_numpy(dtype=float), index=agg_idx, name=z.name)

        def on_obs(matrix, square: bool):
            """Return `matrix` aggregated: ``W M`` (or ``W M W^T`` if `square`), in its own type."""
            if isinstance(matrix, Operator):
                block = matrix.rows.names[0] if len(matrix.rows.names) == 1 else None
                space, order = _axis_space(
                    z.index, None if "block" in z.index.names else block, "rows"
                )
                found = np.empty(len(z), dtype=np.intp)
                found[order] = space.align(matrix.rows)
                cols = found if square else np.arange(matrix.shape[1])
                form = as_linear_operator(W) @ matrix.data._select(found, cols)
                agg_space = _agg_space(agg_idx, block)
                if square:
                    form = form @ as_linear_operator(W.T)
                    return Covariance(form, agg_space, name=matrix.name)
                return Operator(form, agg_space, matrix.cols, name=matrix.name)
            frame = matrix.reindex(
                index=z.index, columns=z.index if square else None, fill_value=0.0
            )
            if any(len(axis) == 0 for axis in frame.axes):
                raise ValueError(
                    "Input data contains empty dimensions, cannot aggregate."
                )
            values = W @ _frame_matrix(frame)
            if square:
                values = values @ W.T
            values = values.toarray() if hasattr(values, "toarray") else values
            columns = agg_idx if square else frame.columns
            return pd.DataFrame(values, index=agg_idx, columns=columns)

        def repack(orig_obj, new_series):
            """Re-wrap a Series in the type of `orig_obj` if it is a fips vector."""
            if isinstance(orig_obj, (Block, Vector)):
                return type(orig_obj)(new_series, name=orig_obj.name)
            return new_series

        # c_agg = W @ c. A scalar constant (or none) is unchanged by aggregation.
        if constant is None or isinstance(constant, (int, float, np.number)):
            c_agg = constant
        else:
            c = constant.data if isinstance(constant, (Block, Vector)) else constant
            c = c.reindex(index=z.index, fill_value=0.0)
            c_vals = pd.Series(W @ c.to_numpy(dtype=float), index=agg_idx, name=c.name)
            c_agg = repack(constant, c_vals)

        return (
            repack(obs, z_agg),
            on_obs(forward_operator, square=False),
            on_obs(modeldata_mismatch, square=True),
            c_agg,
        )


def _agg_space(index: pd.Index, block: str | None) -> Space:
    """Return the space of the aggregated observations: the blocks of `index`, or its one block."""
    if "block" in index.names:
        return Space.from_index(index)
    if block is None:
        raise ValueError(
            "Observations without a 'block' level aggregate an operator of one block."
        )
    return Space({block: index})
