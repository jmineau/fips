from typing import Optional, Union, Tuple, Any, Callable

import numpy as np
import pandas as pd

from fips.indices import sanitize_index
from fips.vectors import Vector


# --- Matrix Helpers ---


def prepare_matrix(matrix: pd.DataFrame,
                   row_promote: bool, row_asm: Optional[Vector],
                   col_promote: bool, col_asm: Optional[Vector],
                   float_precision: Optional[int]) -> pd.DataFrame:
    """Helper to sanitize matrix indices and promote to MultiIndex structure."""
    df = matrix.copy()
    df.index = sanitize_index(df.index, float_precision)
    df.columns = sanitize_index(df.columns, float_precision)

    def promote_index(index, block_name):
        """Promote an index by adding the block level while preserving inner coordinates."""
        if isinstance(index, pd.MultiIndex):
            new_levels = [[block_name]] + list(index.levels)
            new_codes = [pd.array([0] * len(index), dtype='int8')] + [index.codes[i] for i in range(index.nlevels)]
            new_names = ['block'] + list(index.names)
            return pd.MultiIndex(levels=new_levels, codes=new_codes, names=new_names)
        else:
            names = ['block', index.name or 'index']
            return pd.MultiIndex.from_product([[block_name], index], names=names)

    # Avoid double promotion: if already has a 'block' level, skip promotion
    if row_promote and row_asm is not None and 'block' not in (df.index.names or []):
        df.index = promote_index(df.index, row_asm.block_order[0])
    if col_promote and col_asm is not None and 'block' not in (df.columns.names or []):
        df.columns = promote_index(df.columns, col_asm.block_order[0])
    return df


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
        return self.data.values

    @property
    def index(self) -> pd.Index:
        return self.data.index

    @property
    def columns(self) -> pd.Index:
        return self.data.columns

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def __repr__(self):
        return f"<{self.__class__.__name__} shape={self.shape}>"


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
    def from_vector(cls, vector: Vector) -> 'CovarianceMatrix':
        """
        Factory method to create a zero-filled CovarianceMatrix aligned to a Vector.
        Allows usage of set_block() and set_interaction().
        """
        size = vector.size
        data = pd.DataFrame(
            np.zeros((size, size)),
            index=vector.data.index,
            columns=vector.data.index
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

    def _get_block_slice(self, block_name: str) -> Union[slice, np.ndarray]:
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
             raise ValueError(f"Sigma mismatch: expected length {size}, got {len(s_vals)}")
        return s_vals

    def _compute_from_sigma(self, 
                            sigma_row: Any, sigma_col: Any, 
                            idx_row: pd.Index, idx_col: pd.Index, 
                            kernel: Optional[Callable]) -> np.ndarray:
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
             raise ValueError(f"Kernel shape mismatch: expected ({r_size}, {c_size}), got {core_matrix.shape}")

        return s_row[:, None] * s_col[None, :] * core_matrix

    def _compute_from_covariance(self, 
                                 cov: Any, 
                                 idx_row: pd.Index, idx_col: pd.Index) -> np.ndarray:
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
        covariance: Union[float, np.ndarray, pd.Series, pd.DataFrame, Callable] = None,
        sigma: Union[float, np.ndarray, pd.Series, pd.DataFrame] = None,
        kernel: Optional[Callable] = None
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
                raise ValueError("Cannot apply 'kernel' to 'covariance'. Use 'sigma' instead.")
            data = self._compute_from_covariance(covariance, idx, idx)

        if data.shape != (block_size, block_size):
            raise ValueError(f"Final shape mismatch for block '{block}': expected ({block_size}, {block_size}), got {data.shape}")
        
        # Modify underlying numpy array using the slice
        self.data.values[slc, slc] = data
        return self

    def set_interaction(
        self, 
        block_row: str, 
        block_col: str, 
        *,
        covariance: Union[float, np.ndarray, Callable] = None,
        sigma: Union[float, Tuple[Any, Any], Any] = None,
        kernel: Optional[Callable] = None
    ):
        """Set an off-diagonal block (cross-covariance)."""
        if block_row == block_col:
            return self.set_block(block_row, covariance=covariance, sigma=sigma, kernel=kernel)
            
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
            raise ValueError(f"Interaction shape mismatch: expected ({r_size}, {c_size}), got {data.shape}")
            
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

    def convolve(self, state: Union[Vector, pd.Series, np.ndarray],
                 float_precision: Optional[int] = None) -> pd.Series:
        """Convolve (project) a state vector through the forward operator."""
        if isinstance(state, Vector):
            s = state.data.copy()
            s.index = sanitize_index(s.index, float_precision)
        elif isinstance(state, pd.Series):
            s = state.copy()
            s.index = sanitize_index(s.index, float_precision)
        elif isinstance(state, np.ndarray):
            if state.shape[0] != self.data.shape[1]:
                 raise ValueError(f"Shape mismatch: Operator expects {self.data.shape[1]}, got {state.shape[0]}")
            x_vals = state
            return pd.Series(self.data.values @ x_vals, index=self.obs_index, name="modeled_obs")
        else:
            raise TypeError("State must be a Vector, pandas Series, or numpy array.")

        x_vals = s.reindex(self.state_index).fillna(0.0).values
        y_values = self.data.values @ x_vals
        return pd.Series(y_values, index=self.obs_index, name=f"{s.name}_obs")


def convolve(state: Union[Vector, pd.Series, np.ndarray],
             forward_operator: Union[ForwardOperator, pd.DataFrame],
             float_precision: Optional[int] = None) -> pd.Series:
    """Helper to convolve a state vector with a forward operator matrix."""
    if isinstance(forward_operator, pd.DataFrame):
        forward_operator = ForwardOperator(forward_operator)

    return forward_operator.convolve(state=state, float_precision=float_precision)
