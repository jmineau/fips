"""
Covariance matrix structures and generation.

This module provides the `CovarianceMatrix` class and related utilities for
representing, building, and manipulating error covariance matrices used in
Bayesian estimation.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.sparse import csr_matrix

from fips._sparse import frame_from_spmatrix
from fips.base import diagonal_frame
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
        """
        The variances (diagonal elements) of the covariance matrix.

        Read from the stored entries of a sparse covariance, without forming
        the dense matrix.
        """
        return Vector(self.operator.diagonal(), index=self.data.index, name="variance")

    def force_symmetry(self, keep: str = "lower") -> "CovarianceMatrix":
        """
        Force the matrix to be perfectly symmetric by copying one triangle to the other.

        Useful for eliminating floating-point asymmetry.

        Parameters
        ----------
        keep : {'lower', 'upper'}, default 'lower'
            Which triangle of the matrix to preserve and copy.

        Returns
        -------
        CovarianceMatrix
            A new, perfectly symmetric covariance matrix, sparse if this one is.
        """
        if keep not in ("lower", "upper"):
            raise ValueError("keep must be 'lower' or 'upper'")

        if self.is_sparse:
            # The kept triangle, diagonal included, plus the transpose of the
            # same triangle without its diagonal
            mat = self.values
            if keep == "lower":
                sym = sparse.tril(mat) + sparse.tril(mat, k=-1).T
            else:
                sym = sparse.triu(mat) + sparse.triu(mat, k=1).T
            frame = frame_from_spmatrix(sparse.csr_matrix(sym), self.index, self.index)
            return type(self)(frame, name=self.name)

        mat = self.values.copy()

        if keep == "lower":
            # Extract lower triangle and add its transpose (minus the diagonal so it doesn't double)
            lower = np.tril(mat)
            sym_mat = lower + lower.T - np.diag(np.diag(lower))
        else:
            upper = np.triu(mat)
            sym_mat = upper + upper.T - np.diag(np.diag(upper))

        return type(self)(sym_mat, index=self.index, name=self.name)


class ErrorComponent(ABC):
    """
    Abstract base class for error components in covariance matrices.
    """

    name: str
    """Name of the error component."""
    variances: float | pd.Series
    """Variances for the diagonal of the covariance matrix. Can be a single float or a Series indexed by the same MultiIndex as the covariance matrix."""

    def __init__(self, name: str, variances: float | pd.Series):
        self.name = name
        self.variances = variances

    def _align_variances(self, index: pd.MultiIndex) -> pd.Series:
        """Return the variances reindexed onto ``index``, filling gaps with zeros."""
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
            # pandas-stubs take only str names in join(on=), though a level name
            # is any Hashable
            # pyrefly: ignore[bad-argument-type]
            aligned = target_df.join(variances, on=target_levels)
            return aligned[variances.name].fillna(0.0)

        return pd.Series(self.variances, index=index)

    @abstractmethod
    def build(self, index: pd.MultiIndex, **kwargs) -> pd.DataFrame:
        """Must return a pd.DataFrame with the given index on both rows and columns."""
        pass

    def __add__(self, other):
        """Add error components to create a CovarianceBuilder."""
        if isinstance(other, ErrorComponent):
            # Adding two components creates a new Builder
            return CovarianceBuilder([self, other])
        elif isinstance(other, CovarianceBuilder):
            # Adding a component to a Builder appends it
            other.components.append(self)
            return other
        raise TypeError(f"Cannot add ErrorComponent to {type(other)}")


class CovarianceBuilder:
    """
    Builds covariance matrices by combining multiple error components.
    """

    components: list[ErrorComponent]
    """The list of error components to build and sum."""

    def __init__(self, components: list[ErrorComponent]):
        self.components = components

    def build(
        self, index: pd.MultiIndex, sparse: bool = False, **kwargs
    ) -> pd.DataFrame:
        """
        Build and sum all error components into a single DataFrame.

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
            return frame_from_spmatrix(csr_matrix(S), index=index, columns=index)
        return pd.DataFrame(S, index=index, columns=index)

    def __add__(self, other):
        """Add error components or builders together."""
        if isinstance(other, ErrorComponent):
            return CovarianceBuilder(self.components + [other])
        elif isinstance(other, CovarianceBuilder):
            return CovarianceBuilder(self.components + other.components)
        raise TypeError(f"Cannot add CovarianceBuilder to {type(other)}")

    def __radd__(self, other):
        """Right-hand addition for error components."""
        if isinstance(other, ErrorComponent):
            return CovarianceBuilder([other] + self.components)
        raise TypeError(f"Cannot add {type(other)} to CovarianceBuilder")

    def __repr__(self) -> str:
        """Return string representation."""
        names = [c.name for c in self.components]
        return f"CovarianceBuilder(n_components={len(self.components)}, components={names})"


class DiagonalError(ErrorComponent):
    """
    Error component that produces diagonal covariance matrices.
    """

    def build(self, index: pd.MultiIndex, **kwargs) -> pd.DataFrame:
        """Build diagonal covariance matrix."""
        variances = self._align_variances(index)
        return diagonal_frame(variances.to_numpy(), index)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiagonalError(name='{self.name}')"


class BlockDecayError(ErrorComponent):
    """
    Error component with block-structured correlation decay.
    """

    groupers: list
    """List of groupers to use for block structure."""

    corr_func: Callable
    """Function that takes a DataFrame group and returns a correlation matrix for that block."""

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
        """Compute block correlation matrix for a group."""
        idx = group.index.to_numpy()
        return idx, self.corr_func(group)

    def build(self, index: pd.MultiIndex, n_jobs: int = 1, **kwargs) -> pd.DataFrame:
        """Build block-structured covariance matrix with correlation decay."""
        variances = self._align_variances(index)
        N = len(index)
        corr_matrix = np.eye(N)

        coords = index.to_frame(index=False)

        # Parallelize the loop over groups using joblib
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_block)(group)
            for _, group in coords.groupby(self.groupers)
        )

        for idx, block in results or []:
            corr_matrix[np.ix_(idx, idx)] = block

        std_dev = np.sqrt(variances.to_numpy())
        cov_matrix = std_dev[:, None] * corr_matrix * std_dev[None, :]

        return pd.DataFrame(cov_matrix, index=index, columns=index)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BlockDecayError(name='{self.name}', groupers={self.groupers})"


class KroneckerError(ErrorComponent):
    """
    Builds a covariance matrix from a product of marginal correlation matrices.

    Each marginal kernel gives the correlation along some of the index's levels
    (time, or ``lat`` and ``lon`` together), and the correlation between two
    elements is the product of their marginal correlations. On the full product
    of the marginal coordinates, in the kernels' order, this is the Kronecker
    product of the marginal correlation matrices.

    Parameters
    ----------
    name : str
        Name of the error component.
    variances : float or pd.Series
        Variances for the diagonal of the covariance matrix.
        Can be a single float or a Series indexed by the same MultiIndex as the covariance matrix.
    marginal_kernels : list of tuples
        Each tuple contains:

        1. The dimension name(s) as a string or list of strings.
        2. The Callable kernel function that takes a DataFrame of those
           unique coordinates and returns a 2D correlation matrix.

    Notes
    -----
    The index can hold its rows in any order, and any subset of the product of
    the marginal coordinates: a grid with some cells left out, at every time or
    at only some. Entry ``(i, j)`` of the result is
    ``std[i] * std[j] * C_1[i, j] * C_2[i, j] * ...``, where ``C_k[i, j]`` is
    the correlation the k-th kernel gives between the coordinates of rows i and
    j on that kernel's levels. Each kernel receives the unique coordinates of
    its levels in the order they first appear in the index. Every row must have
    its own combination of the kernels' coordinates: two rows that differ only
    on a level no kernel names (two blocks on the same grid, say) would be
    perfectly correlated, so `build` raises a `ValueError` instead.
    """

    marginal_kernels: list[tuple[str | list[str], Callable]]
    """List of marginal kernels, each defined by a tuple of dimension name(s) and a kernel function."""

    def __init__(
        self,
        name: str,
        variances: float | pd.Series,
        marginal_kernels: list[tuple[str | list[str], Callable]],
    ):
        super().__init__(name, variances)
        self.marginal_kernels = marginal_kernels

    def build(self, index: pd.MultiIndex, **kwargs) -> pd.DataFrame:
        """
        Build covariance matrix using product of marginal kernels.

        Parameters
        ----------
        index : pd.MultiIndex
            The MultiIndex of the covariance matrix. It must contain the
            levels named in ``marginal_kernels``; its rows can be in any order.
        **kwargs
            Additional keyword arguments that may be needed by the kernel functions.

        Returns
        -------
        pd.DataFrame
            The constructed covariance matrix with the given index on both rows and columns.

        Raises
        ------
        ValueError
            If no marginal kernels were given, if two rows of the index have the
            same coordinates on every kernel's levels, or if a kernel does not
            return a square matrix with one row per unique coordinate.
        """
        if not self.marginal_kernels:
            raise ValueError("KroneckerError needs at least one marginal kernel.")

        variances = self._align_variances(index)
        coords = index.to_frame(index=False)

        dims_list = [
            [dims] if isinstance(dims, str) else list(dims)
            for dims, _ in self.marginal_kernels
        ]
        kernel_dims = list(dict.fromkeys(d for dims in dims_list for d in dims))
        n_repeated = int(coords.duplicated(subset=kernel_dims).sum())
        if n_repeated:
            raise ValueError(
                f"KroneckerError needs each row of the index to have its own "
                f"combination of the kernels' coordinates {kernel_dims}, but "
                f"{n_repeated} rows repeat one. Rows that differ only on another "
                f"level would be perfectly correlated."
            )

        # Each marginal is its correlation matrix C over the unique coordinates
        # of its levels, and pos, the position of each row's coordinates among
        # them (numbered in order of first appearance).
        marginals = []
        for dims, (_, kernel_func) in zip(
            dims_list, self.marginal_kernels, strict=True
        ):
            pos = coords.groupby(dims, sort=False, dropna=False).ngroup().to_numpy()
            first = np.unique(pos, return_index=True)[1]
            C = np.asarray(kernel_func(coords[dims].iloc[first]), dtype=float)
            if C.shape != (len(first), len(first)):
                raise ValueError(
                    f"The kernel for {dims} returned a matrix of shape {C.shape} "
                    f"for {len(first)} unique coordinates; expected "
                    f"({len(first)}, {len(first)})."
                )
            marginals.append((C, pos))

        # corr[i, j] = prod_k C_k[pos_k[i], pos_k[j]], for any row order and any
        # subset of the product. On the full product in the kernels' order it
        # equals reduce(np.kron, [C_1, C_2, ...]).
        C, pos = marginals[0]
        corr_matrix = C.take(pos, axis=0).take(pos, axis=1)
        for C, pos in marginals[1:]:
            corr_matrix *= C.take(pos, axis=0).take(pos, axis=1)

        # Scale by variances
        std_dev = np.sqrt(variances.to_numpy())
        corr_matrix *= std_dev[:, None]
        corr_matrix *= std_dev[None, :]

        return pd.DataFrame(corr_matrix, index=index, columns=index)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"KroneckerError(name='{self.name}', "
            f"n_marginals={len(self.marginal_kernels)})"
        )
