from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from fips.matrix import Matrix, MatrixBlock
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

    # Use pd.cut to bin the data by time into time bins
    data[time_dim] = pd.cut(
        data[time_dim], bins=time_bins, include_lowest=True, right=False
    )

    # Set Intervals to the left edge of the bin (start of time interval)
    data[time_dim] = data[time_dim].apply(lambda x: x.left)

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
    """Aggregates the observation space of an inverse problem.

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
    """

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
        """Build W and the aggregated index from an obs ``pd.Index``.

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
        forward_operator: pd.DataFrame | MatrixBlock | Matrix,
        modeldata_mismatch: pd.DataFrame | MatrixBlock | Matrix,
        constant: float | pd.Series | Block | Vector | None = None,
    ):
        """Apply W to the inverse problem components.

        Inputs may be bare pandas objects or fips wrapper types (``Vector``,
        ``ForwardOperator``, ``CovarianceMatrix``); return types mirror the
        inputs. See the class docstring for the mathematical transforms.

        The aggregator ensures all inputs are properly aligned to obs.index
        before building the weight matrix W.
        """

        # Unwrap fips types to the underlying pandas object for arithmetic.
        def unwrap(obj):
            return obj.data if hasattr(obj, "data") else obj

        z_df = unwrap(obs)
        H_df = unwrap(forward_operator)
        S_z_df = unwrap(modeldata_mismatch)

        # Ensure matrices are aligned to obs.index before aggregation
        # This guarantees W has compatible dimensions for matrix operations
        H_df = H_df.reindex(index=z_df.index, fill_value=0.0)
        S_z_df = S_z_df.reindex(index=z_df.index, columns=z_df.index, fill_value=0.0)
        if constant is not None and not np.isscalar(constant):
            c_df = unwrap(constant)
            c_df = c_df.reindex(index=z_df.index, fill_value=0.0)
        else:
            c_df = None
        if any(len(x) == 0 for x in [z_df, H_df, S_z_df]):
            raise ValueError("Input data contains empty dimensions, cannot aggregate.")

        W, agg_idx = self._build_operator(z_df.index)

        # Obs vector aggregation
        # z_agg = W @ z
        z_agg = pd.Series(W @ z_df.values, index=agg_idx, name=z_df.name)

        # Forward operator aggregation
        # H_agg = W @ H  (preserve SparseDtype if present)
        if all(isinstance(dt, pd.SparseDtype) for dt in H_df.dtypes):
            H_agg_vals = W @ H_df.sparse.to_coo().tocsr()
            H_agg = pd.DataFrame.sparse.from_spmatrix(
                H_agg_vals, index=agg_idx, columns=H_df.columns
            )
        else:
            H_agg = pd.DataFrame(W @ H_df.values, index=agg_idx, columns=H_df.columns)

        # Covariance aggregation with full propagation
        # S_z_agg = W @ S_z @ W.T  — for diagonal S_z with variance σ² and
        # mean aggregation of n obs this yields σ²/n on the diagonal.
        S_z_agg_vals = W @ S_z_df.values @ W.T
        if hasattr(S_z_agg_vals, "toarray"):
            S_z_agg_vals = S_z_agg_vals.toarray()
        S_z_agg = pd.DataFrame(S_z_agg_vals, index=agg_idx, columns=agg_idx)

        def repack(orig_obj, new_df):
            """Re-wrap new_df in the same type as orig_obj if it is a fips wrapper."""
            return (
                type(orig_obj)(new_df, name=orig_obj.name)
                if hasattr(orig_obj, "data")
                else new_df
            )

        # c_agg = W @ c  (scalars are invariant to aggregation)
        if constant is None:
            c_agg = None
        elif np.isscalar(constant):
            c_agg = constant  # scalars pass through unchanged
        else:
            c_vals = pd.Series(W @ c_df.values, index=agg_idx, name=c_df.name)
            c_agg = repack(constant, c_vals)

        return (
            repack(obs, z_agg),
            repack(forward_operator, H_agg),
            repack(modeldata_mismatch, S_z_agg),
            c_agg,
        )
