import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import reduce

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix

from fips.matrix import Matrix
from fips.vector import Vector

logger = logging.getLogger(__name__)


class CovarianceMatrix(Matrix):
    """
    Represents a symmetric Covariance Matrix.

    Covariance matrices are used to represent error covariances in the inversion framework.
    They can be constructed from variances and correlation matrices.
    """

    @property
    def variances(self) -> Vector:
        """Returns the variances (diagonal elements) of the covariance matrix."""
        return Vector(np.diag(self.data.values), index=self.data.index, name="variance")

    def force_symmetry(self, keep: str = "lower") -> "CovarianceMatrix":
        """
        Forces the matrix to be perfectly symmetric by copying one triangle
        to the other. Useful for eliminating floating-point asymmetry.

        Parameters
        ----------
        keep : {'lower', 'upper'}, default 'lower'
            Which triangle of the matrix to preserve and copy.

        Returns
        -------
        CovarianceMatrix
            A new, perfectly symmetric covariance matrix.
        """
        mat = self.values.copy()

        if keep == "lower":
            # Extract lower triangle and add its transpose (minus the diagonal so it doesn't double)
            lower = np.tril(mat)
            sym_mat = lower + lower.T - np.diag(np.diag(lower))
        elif keep == "upper":
            upper = np.triu(mat)
            sym_mat = upper + upper.T - np.diag(np.diag(upper))
        else:
            raise ValueError("keep must be 'lower' or 'upper'")

        return type(self)(sym_mat, index=self.index, name=self.name)


class ErrorComponent(ABC):
    def __init__(self, name: str, variances: float | pd.Series):
        self.name = name
        self.variances = variances

    def _align_variances(self, index: pd.MultiIndex) -> pd.Series:
        if isinstance(self.variances, pd.Series):
            if self.variances.index.equals(index):
                return self.variances.fillna(0.0)

            # Check if the series has names for all its levels
            target_levels = self.variances.index.names
            if any(lvl is None for lvl in target_levels):
                raise ValueError(
                    "All levels in the variance Series index must be named."
                )

            # Check if the target index has any unnamed levels
            if any(name is None for name in index.names):
                raise ValueError(
                    "All levels in the target index must be named when aligning variances."
                )

            # Ensure the Series has a name for join operation
            variances = self.variances
            if variances.name is None:
                variances = variances.copy()
                variances.name = "variances"

            # Convert the target index to a dataframe just to use its join capabilities
            target_df = pd.DataFrame(index=index)
            aligned = target_df.join(variances, on=target_levels)
            return aligned[variances.name].fillna(0.0)

        return pd.Series(self.variances, index=index)

    @abstractmethod
    def build(self, index: pd.MultiIndex, **kwargs) -> pd.DataFrame:
        """Must return a pd.DataFrame with the given index on both rows and columns."""
        pass

    def __add__(self, other):
        if isinstance(other, ErrorComponent):
            # Adding two components creates a new Builder
            return CovarianceBuilder([self, other])
        elif isinstance(other, CovarianceBuilder):
            # Adding a component to a Builder appends it
            other.components.append(self)
            return other
        raise TypeError(f"Cannot add ErrorComponent to {type(other)}")


class CovarianceBuilder:
    def __init__(self, components: list[ErrorComponent]):
        self.components = components

    def build(
        self, index: pd.MultiIndex, sparse: bool = False, **kwargs
    ) -> pd.DataFrame:
        """Builds and sums all error components into a single DataFrame.

        Parameters
        ----------
        index : pd.MultiIndex
            The state/observation index to build the covariance over.
        sparse : bool, default False
            If True, return a pandas sparse DataFrame.  Apply threshold zeroing
            in each ErrorComponent.build() before using this option.
        """
        if not self.components:
            raise ValueError("No components provided to build.")

        S = np.add.reduce(
            [c.build(index, **kwargs).to_numpy(dtype=float) for c in self.components]
        )

        if sparse:
            return pd.DataFrame.sparse.from_spmatrix(
                csr_matrix(S), index=index, columns=index
            )
        return pd.DataFrame(S, index=index, columns=index)

    def __add__(self, other):
        if isinstance(other, ErrorComponent):
            return CovarianceBuilder(self.components + [other])
        elif isinstance(other, CovarianceBuilder):
            return CovarianceBuilder(self.components + other.components)
        raise TypeError(f"Cannot add CovarianceBuilder to {type(other)}")

    def __radd__(self, other):
        if isinstance(other, ErrorComponent):
            return CovarianceBuilder([other] + self.components)
        raise TypeError(f"Cannot add {type(other)} to CovarianceBuilder")


class DiagonalError(ErrorComponent):
    def build(self, index: pd.MultiIndex, **kwargs) -> pd.DataFrame:
        variances = self._align_variances(index)
        cov_matrix = np.diag(variances)
        return pd.DataFrame(cov_matrix, index=index, columns=index)


class BlockDecayError(ErrorComponent):
    def __init__(
        self,
        name: str,
        variances: float | pd.Series,
        groupers: list,
        corr_func: Callable,
    ):
        super().__init__(name, variances)
        self.groupers = groupers
        self.corr_func = corr_func

    def _compute_block(self, group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Helper function to execute safely in a joblib worker."""
        idx = group.index.to_numpy()
        return idx, self.corr_func(group)

    def build(self, index: pd.MultiIndex, n_jobs: int = 1, **kwargs) -> pd.DataFrame:
        variances = self._align_variances(index)
        N = len(index)
        corr_matrix = np.eye(N)

        coords = index.to_frame(index=False)

        # Parallelize the loop over groups using joblib
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_block)(group)
            for _, group in coords.groupby(self.groupers)
        )

        for idx, block in results:
            corr_matrix[np.ix_(idx, idx)] = block

        std_dev = np.sqrt(variances.to_numpy())
        cov_matrix = std_dev[:, None] * corr_matrix * std_dev[None, :]

        return pd.DataFrame(cov_matrix, index=index, columns=index)


class KroneckerError(ErrorComponent):
    """
    Builds a full covariance matrix for a strict grid using Kronecker products
    of N arbitrary marginal correlation matrices.
    """

    def __init__(
        self,
        name: str,
        variances: float | pd.Series,
        marginal_kernels: list[tuple[str | list[str], Callable]],
    ):
        """
        Parameters
        ----------
        marginal_kernels : list of tuples
            Each tuple contains:
            1. The dimension name(s) as a string or list of strings.
            2. The Callable kernel function that takes a DataFrame of those
               unique coordinates and returns a 2D correlation matrix.
            ORDER MATTERS: These must be provided in the exact order
            that the dimensions appear in the matrix's MultiIndex!
        """
        super().__init__(name, variances)
        self.marginal_kernels = marginal_kernels

    def build(self, index: pd.MultiIndex, **kwargs) -> pd.DataFrame:
        variances = self._align_variances(index)

        corr_matrices = []
        for dims, kernel_func in self.marginal_kernels:
            # Standardize string vs list of strings
            if isinstance(dims, str):
                dims = [dims]

            # Extract the unique coordinates for this specific marginal dimension
            # We use drop_duplicates to isolate the strict grid points
            coords = index.to_frame(index=False)[dims].drop_duplicates()

            # Build the marginal correlation matrix
            C_marginal = kernel_func(coords)
            corr_matrices.append(C_marginal)

        # Combine all marginals using consecutive Kronecker products
        # reduce(np.kron, [A, B, C]) is mathematically equivalent to np.kron(A, np.kron(B, C))
        corr_matrix = reduce(np.kron, corr_matrices)

        # Scale by variances
        std_dev = np.sqrt(variances.to_numpy())
        cov_matrix = std_dev[:, None] * corr_matrix * std_dev[None, :]

        return pd.DataFrame(cov_matrix, index=index, columns=index)
