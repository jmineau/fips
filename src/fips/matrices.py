from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from fips.indices import check_overlap, promote_index, sanitize_index
from fips.vectors import Vector

# ==============================================================================
# BASE MATRIX INTERFACE
# ==============================================================================


class Matrix:
    """
    Base class for all matrix-like objects in the inversion framework.
    Wraps a pandas DataFrame and ensures consistent index handling.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        # Ensure indices are clean (standard numeric types)
        self.data.index = sanitize_index(self.data.index)
        self.data.columns = sanitize_index(self.data.columns)

    @property
    def values(self) -> np.ndarray:
        return self.data.to_numpy()

    @property
    def index(self) -> pd.Index:
        return self.data.index

    @property
    def columns(self) -> pd.Index:
        return self.data.columns

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def __repr__(self):
        return f"<{self.__class__.__name__} shape={self.shape}>"


def prepare_matrix(
    matrix: pd.DataFrame | Matrix,
    matrix_class: type[Matrix],
    row_index: pd.Index,
    col_index: pd.Index,
    float_precision: int | None = None,
) -> Matrix:
    """
    Prepare a matrix by sanitizing, promoting if needed, checking overlap, and reindexing.

    Assumes row_index and col_index are already promoted (i.e., in their final form).
    Will promote the matrix indices if they don't match the first level of the target indices.
    Always warns if overlap is incomplete.

    Parameters
    ----------
    matrix : pd.DataFrame | Matrix
        Input matrix data
    matrix_class : type[Matrix]
        Class to wrap result in (e.g., ForwardOperator, CovarianceMatrix)
    row_index : pd.Index
        Target row index (assumed already promoted)
    col_index : pd.Index
        Target column index (assumed already promoted)
    float_precision : int | None
        Decimals to round float indices to

    Returns
    -------
    Matrix
        Instance of matrix_class wrapping the prepared DataFrame
    """

    # Unwrap if already a Matrix
    if isinstance(matrix, Matrix):
        df = matrix.data.copy()
    else:
        df = matrix.copy()

    # Sanitize all indices
    df.index = sanitize_index(index=df.index, decimals=float_precision)
    df.columns = sanitize_index(index=df.columns, decimals=float_precision)
    row_index = sanitize_index(index=row_index, decimals=float_precision)
    col_index = sanitize_index(index=col_index, decimals=float_precision)

    # Handle block level promotion for both row and column indices
    for axis, (target_idx, name) in enumerate(
        [(row_index, "row"), (col_index, "column")]
    ):
        matrix_idx = df.index if axis == 0 else df.columns

        matrix_has_block = (
            isinstance(matrix_idx, pd.MultiIndex) and "block" in matrix_idx.names
        )
        target_has_block = (
            isinstance(target_idx, pd.MultiIndex) and "block" in target_idx.names
        )

        if not matrix_has_block and target_has_block:
            # Matrix doesn't have block level, but target does
            blocks = target_idx.get_level_values("block").unique()

            if len(blocks) > 1:
                raise ValueError(
                    f"Cannot automatically assign block to matrix: {name} index has multiple blocks {list(blocks)}. "
                    "Please provide a matrix with explicit block levels."
                )

            # Only one block, so promote the matrix to use this block
            promoted_idx = promote_index(
                index=matrix_idx, promotion=blocks[0], promotion_level="block"
            )
            if axis == 0:
                df.index = promoted_idx
            else:
                df.columns = promoted_idx

        elif matrix_has_block and target_has_block:
            # Both have blocks - ensure they're compatible
            matrix_blocks = matrix_idx.get_level_values("block").unique()
            target_blocks = target_idx.get_level_values("block").unique()
            invalid_blocks = set(matrix_blocks) - set(target_blocks)

            if invalid_blocks:
                raise ValueError(
                    f"Matrix {name} blocks {invalid_blocks} not found in target {name} index blocks {list(target_blocks)}"
                )

    # Check for overlapping indices
    check_overlap(target_idx=row_index, available_idx=df.index, name="Row")
    check_overlap(target_idx=col_index, available_idx=df.columns, name="Column")

    # Reindex to target indices, filling missing with zeros
    df = df.reindex(index=row_index, columns=col_index).fillna(0.0)

    # Wrap in appropriate matrix class and return
    return matrix_class(df)


# ==============================================================================
# COVARIANCE MATRIX
# ==============================================================================


class CovarianceMatrix(Matrix):
    """
    Represents a symmetric Covariance Matrix (Prior Error or Obs Error).

    Can be initialized from a DataFrame directly, or created from a Vector
    to enable block-based construction methods using set_block().
    """

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    @classmethod
    def from_vector(cls, vector: Vector) -> "CovarianceMatrix":
        """
        Factory method to create a zero-filled CovarianceMatrix aligned to a Vector.
        Allows usage of set_block() and set_interaction().
        """
        size = vector.n
        data = pd.DataFrame(
            np.zeros((size, size)), index=vector.data.index, columns=vector.data.index
        )
        return cls(data)

    def __add__(self, other):
        """Allows adding two CovarianceMatrices (e.g. Instrument Error + Transport Error)."""
        if not isinstance(other, CovarianceMatrix):
            return NotImplemented

        # We rely on pandas alignment for the addition
        new_data = self.data + other.data
        return CovarianceMatrix(new_data)

    # --- Internal Helpers ---

    def _get_block_slice(self, block_name: str) -> slice | np.ndarray:
        """
        Resolves the integer slice/indices for a block name using the index.
        Assumes the index is a MultiIndex with 'block' as level 0.
        """
        try:
            # get_loc on a MultiIndex returns a slice if sorted, or boolean array/indices
            return self.data.index.get_loc(block_name)
        except KeyError:
            raise KeyError(f"Block '{block_name}' not found in CovarianceMatrix index.")

    def _get_inner_index(self, slc: slice) -> pd.Index:
        """Returns the inner index (dropping the block name) for a slice."""
        full_idx = self.data.index[slc]
        if isinstance(full_idx, pd.MultiIndex):
            return full_idx.droplevel(0)
        return full_idx

    def _normalize_sigma(self, sigma: Any, size: int) -> np.ndarray:
        if isinstance(sigma, (pd.Series, pd.DataFrame)):
            s_vals = sigma.values
        else:
            s_vals = np.array(sigma)

        if s_vals.ndim == 0 or (s_vals.ndim == 1 and len(s_vals) == 1):
            s_vals = np.full(size, float(s_vals))

        s_vals = s_vals.flatten()
        if len(s_vals) != size:
            raise ValueError(
                f"Sigma mismatch: expected length {size}, got {len(s_vals)}"
            )
        return s_vals

    def _compute_from_sigma(
        self,
        sigma_row: Any,
        sigma_col: Any,
        idx_row: pd.Index,
        idx_col: pd.Index,
        kernel: Callable | None,
    ) -> np.ndarray:
        r_size = len(idx_row)
        c_size = len(idx_col)

        s_row = self._normalize_sigma(sigma_row, r_size)
        s_col = self._normalize_sigma(sigma_col, c_size)

        if kernel is not None:
            core_matrix = kernel(idx_row, idx_col)
        else:
            if r_size == c_size and np.array_equal(idx_row, idx_col):
                core_matrix = np.eye(r_size)
            else:
                core_matrix = np.ones((r_size, c_size))

        if core_matrix.shape != (r_size, c_size):
            raise ValueError(
                f"Kernel shape mismatch: expected ({r_size}, {c_size}), got {core_matrix.shape}"
            )

        return s_row[:, None] * s_col[None, :] * core_matrix

    def _compute_from_covariance(
        self, cov: Any, idx_row: pd.Index, idx_col: pd.Index
    ) -> np.ndarray:
        r_size = len(idx_row)
        c_size = len(idx_col)

        if callable(cov):
            try:
                data = cov(idx_row, idx_col)
            except TypeError:
                data = cov(idx_row)
        elif isinstance(cov, (pd.Series, pd.DataFrame)):
            data = cov.values
        else:
            data = np.array(cov)

        if isinstance(data, (int, float)) or (data.ndim == 0):
            if r_size == c_size:
                data = np.eye(r_size) * float(data)
            else:
                data = np.full((r_size, c_size), float(data))
        elif data.ndim == 1 and len(data) == r_size and r_size == c_size:
            data = np.diag(data)

        return data

    # --- Public Construction API ---

    def set_block(
        self,
        block: str,
        *,
        covariance: float | np.ndarray | pd.Series | pd.DataFrame | Callable = None,
        sigma: float | np.ndarray | pd.Series | pd.DataFrame = None,
        kernel: Callable | None = None,
    ):
        """Set a diagonal block (auto-covariance)."""
        if covariance is not None and sigma is not None:
            raise ValueError("Specify 'covariance' OR 'sigma', not both.")
        if covariance is None and sigma is None:
            raise ValueError("Must specify either 'covariance' or 'sigma'.")

        slc = self._get_block_slice(block)
        idx = self._get_inner_index(slc)
        block_size = len(idx)

        if sigma is not None:
            data = self._compute_from_sigma(sigma, sigma, idx, idx, kernel)
        else:
            if kernel is not None:
                raise ValueError(
                    "Cannot apply 'kernel' to 'covariance'. Use 'sigma' instead."
                )
            data = self._compute_from_covariance(covariance, idx, idx)

        if data.shape != (block_size, block_size):
            raise ValueError(
                f"Final shape mismatch for block '{block}': expected ({block_size}, {block_size}), got {data.shape}"
            )

        # Modify underlying numpy array using the slice
        self.data.values[slc, slc] = data
        return self

    def set_interaction(
        self,
        block_row: str,
        block_col: str,
        *,
        covariance: float | np.ndarray | Callable = None,
        sigma: float | tuple[Any, Any] | Any = None,
        kernel: Callable | None = None,
    ):
        """Set an off-diagonal block (cross-covariance)."""
        if block_row == block_col:
            return self.set_block(
                block_row, covariance=covariance, sigma=sigma, kernel=kernel
            )

        r_slc = self._get_block_slice(block_row)
        c_slc = self._get_block_slice(block_col)

        r_idx = self._get_inner_index(r_slc)
        c_idx = self._get_inner_index(c_slc)
        r_size = len(r_idx)
        c_size = len(c_idx)

        if sigma is not None:
            if isinstance(sigma, tuple) and len(sigma) == 2:
                s_row, s_col = sigma
            else:
                s_row = s_col = sigma
            data = self._compute_from_sigma(s_row, s_col, r_idx, c_idx, kernel)

        elif covariance is not None:
            if kernel is not None:
                raise ValueError("Cannot apply 'kernel' to 'covariance'.")
            data = self._compute_from_covariance(covariance, r_idx, c_idx)
        else:
            raise ValueError("Must specify 'covariance' or 'sigma'.")

        if data.shape != (r_size, c_size):
            raise ValueError(
                f"Interaction shape mismatch: expected ({r_size}, {c_size}), got {data.shape}"
            )

        self.data.values[r_slc, c_slc] = data
        self.data.values[c_slc, r_slc] = data.T
        return self


# ==============================================================================
# FORWARD OPERATOR
# ==============================================================================


class ForwardOperator(Matrix):
    """
    Represents the Jacobian / Forward Operator (H).
    Columns = State Space, Rows = Obs Space.
    """

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.state_index = self.columns
        self.obs_index = self.index

    def convolve(
        self, state: Vector | pd.Series | np.ndarray, float_precision: int | None = None
    ) -> pd.Series:
        """Convolve (project) a state vector through the forward operator."""
        if isinstance(state, Vector):
            s = state.data.copy()
            s.index = sanitize_index(s.index, float_precision)
        elif isinstance(state, pd.Series):
            s = state.copy()
            s.index = sanitize_index(s.index, float_precision)
        elif isinstance(state, np.ndarray):
            if state.shape[0] != self.data.shape[1]:
                raise ValueError(
                    f"Shape mismatch: Operator expects {self.data.shape[1]}, got {state.shape[0]}"
                )
            x_vals = state
            return pd.Series(
                self.data.values @ x_vals, index=self.obs_index, name="modeled_obs"
            )
        else:
            raise TypeError("State must be a Vector, pandas Series, or numpy array.")

        x_vals = s.reindex(self.state_index).fillna(0.0).values
        y_values = self.data.values @ x_vals
        return pd.Series(y_values, index=self.obs_index, name=f"{s.name}_obs")


def convolve(
    state: Vector | pd.Series | np.ndarray,
    forward_operator: ForwardOperator | pd.DataFrame,
    float_precision: int | None = None,
) -> pd.Series:
    """Helper to convolve a state vector with a forward operator matrix."""
    if isinstance(forward_operator, pd.DataFrame):
        forward_operator = ForwardOperator(forward_operator)

    return forward_operator.convolve(state=state, float_precision=float_precision)
