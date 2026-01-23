"""Generic test data generators for inverse problems."""

import numpy as np
import pandas as pd

from fips import CovarianceMatrix, InverseProblem

from .utils import create_correlated_noise, normalize_jacobian


def _parse_dimensions(n, prefix=None):
    """Parse dimension specification into size and index.

    Parameters
    ----------
    n : int, list, or dict
        Dimension specification:
        - int: returns (int, RangeIndex)
        - list: returns (product of list, MultiIndex with names dim1, dim2, ... or {prefix}_dim1, {prefix}_dim2, ...)
        - dict: returns (product of values, MultiIndex with provided names)
    prefix : str, optional
        Prefix for auto-generated dimension names when n is a list.
        If None, uses "dim1", "dim2", etc. If provided, uses "{prefix}_dim1", etc.

    Returns
    -------
    tuple of (int, pd.Index or pd.MultiIndex)
        Total size and corresponding index.
    """
    if isinstance(n, int):
        return n, pd.RangeIndex(n)

    elif isinstance(n, (list, tuple)):
        # Create MultiIndex with dim1, dim2, dim3, ... or {prefix}_dim1, etc.
        if prefix:
            dim_names = [f"{prefix}_dim{i + 1}" for i in range(len(n))]
        else:
            dim_names = [f"dim{i + 1}" for i in range(len(n))]
        index = pd.MultiIndex.from_product(
            [pd.RangeIndex(size) for size in n], names=dim_names
        )
        return len(index), index

    elif isinstance(n, dict):
        # Create MultiIndex with custom names
        index = pd.MultiIndex.from_product(
            [pd.RangeIndex(size) for size in n.values()], names=list(n.keys())
        )
        return len(index), index

    else:
        raise TypeError(f"n must be int, list, or dict, got {type(n)}")


def create_jacobian(
    n_obs, n_state, obs_index=None, state_index=None, structure="dense", seed=None
):
    """Create a synthetic Jacobian (forward operator / sensitivity matrix).

    Parameters
    ----------
    n_obs : int
        Number of observations.
    n_state : int
        Number of state variables.
    obs_index : pd.Index or pd.MultiIndex, optional
        Index for observations (rows). If None, uses RangeIndex.
    state_index : pd.Index or pd.MultiIndex, optional
        Index for state variables (columns). If None, uses RangeIndex.
    structure : {"dense", "sparse"}, optional
        Structure of the Jacobian (default: "dense").
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Jacobian matrix as DataFrame.
    """
    if seed is not None:
        np.random.seed(seed)

    jacobian = np.random.randn(n_obs, n_state)

    if structure == "sparse":
        # Zero out 70% of entries randomly
        mask = np.random.rand(n_obs, n_state) > 0.3
        jacobian = jacobian * mask

    # Normalize to reasonable magnitudes
    jacobian = normalize_jacobian(jacobian, scale=0.5)

    # Create indices if not provided
    if obs_index is None:
        obs_index = pd.RangeIndex(n_obs)
    if state_index is None:
        state_index = pd.RangeIndex(n_state)

    return pd.DataFrame(jacobian, index=obs_index, columns=state_index)


def create_covariance_matrix(n, index=None, corr_type="identity", seed=None):
    """Create a covariance matrix as SymmetricMatrix.

    Parameters
    ----------
    n : int
        Size of the covariance matrix.
    index : pd.Index or pd.MultiIndex, optional
        Index for the covariance matrix. If None, uses RangeIndex.
    corr_type : {"identity", "diagonal", "correlated"}, optional
        Type of covariance structure (default: "identity").
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SymmetricMatrix
        Covariance matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    if corr_type == "identity":
        # Identity matrix (no correlation)
        cov_matrix = np.eye(n)

    elif corr_type == "diagonal":
        # Diagonal matrix with varied values
        values = np.exp(np.random.randn(n) * 0.5)  # Log-normal distribution
        cov_matrix = np.diag(values)

    elif corr_type == "correlated":
        # Create a correlated covariance matrix
        noise = create_correlated_noise((n, n), correlation="spatial", seed=seed)
        cov_matrix = (noise @ noise.T) / n
        # Rescale to reasonable magnitudes
        cov_matrix = cov_matrix / np.max(np.diag(cov_matrix)) * 2.0

    else:
        raise ValueError(f"Unknown covariance type: {corr_type}")

    # Create index if not provided
    if index is None:
        index = pd.RangeIndex(n)

    cov_df = pd.DataFrame(cov_matrix, index=index, columns=index)
    return CovarianceMatrix(cov_df)


def _get_dimension_names(index):
    """Extract dimension names from a pandas Index or MultiIndex."""
    if isinstance(index, pd.MultiIndex):
        return set(index.names)
    else:
        return set()


def create_problem_data(
    n_state=10,
    n_obs=15,
    jacobian_structure="dense",
    prior_error_type="identity",
    obs_error_type="diagonal",
    seed=None,
):
    """Create a complete synthetic Problem dataset.

    Parameters
    ----------
    n_state : int, list, or dict, optional
        State dimensions (default: 10).
        - int: 1D with that size (e.g., 10)
        - list: ND with those sizes (e.g., [5, 5] for 2D)
        - dict: ND with custom dimension names (e.g., {'x': 5, 'y': 5})
    n_obs : int, list, or dict, optional
        Observation dimensions (default: 15).
        - int: 1D with that size
        - list: ND with those sizes
        - dict: ND with custom dimension names
    jacobian_structure : {"dense", "sparse"}, optional
        Structure of the Jacobian (default: "dense").
    prior_error_type : {"identity", "diagonal", "correlated"}, optional
        Type of prior error covariance (default: "identity").
    obs_error_type : {"identity", "diagonal", "correlated"}, optional
        Type of observation error covariance (default: "diagonal").
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys: "state", "obs", "jacobian", "prior_error", "obs_error".

    Raises
    ------
    ValueError
        If state and observation dimension names overlap.
    """
    if seed is not None:
        np.random.seed(seed)

    # Parse dimensions
    n_state_total, state_idx = _parse_dimensions(n_state, prefix="state")
    n_obs_total, obs_idx = _parse_dimensions(n_obs, prefix="obs")

    # Check for overlapping dimension names
    state_names = _get_dimension_names(state_idx)
    obs_names = _get_dimension_names(obs_idx)
    overlap = state_names & obs_names
    if overlap:
        raise ValueError(
            f"State and observation dimension names overlap: {overlap}. "
            f"Use dict with custom names or ensure prefixes are different."
        )

    # Create data
    state = pd.Series(np.exp(np.random.randn(n_state_total) * 0.3), index=state_idx)
    obs = pd.Series(np.random.randn(n_obs_total), index=obs_idx)

    jacobian_df = create_jacobian(
        n_obs_total,
        n_state_total,
        obs_index=obs_idx,
        state_index=state_idx,
        structure=jacobian_structure,
        seed=seed,
    )

    prior_error = create_covariance_matrix(
        n_state_total, index=state_idx, corr_type=prior_error_type, seed=seed
    )
    obs_error = create_covariance_matrix(
        n_obs_total, index=obs_idx, corr_type=obs_error_type, seed=seed
    )

    return {
        "state": state,
        "obs": obs,
        "jacobian": jacobian_df,
        "prior_error": prior_error,
        "obs_error": obs_error,
    }


def create_problem(
    n_state=10,
    n_obs=15,
    jacobian_structure="dense",
    prior_error_type="identity",
    obs_error_type="diagonal",
    background=None,
    seed=None,
):
    """Create a complete Problem instance with synthetic data.

    Supports flexible dimension specification for 1D and ND cases.

    Parameters
    ----------
    n_state : int, list, or dict, optional
        State dimensions (default: 10).
        - int: 1D with that size (e.g., 10)
        - list: ND with those sizes (e.g., [5, 5] for 2D)
        - dict: ND with custom dimension names (e.g., {'x': 5, 'y': 5})
    n_obs : int, list, or dict, optional
        Observation dimensions (default: 15).
        - int: 1D with that size
        - list: ND with those sizes
        - dict: ND with custom dimension names
    jacobian_structure : {"dense", "sparse"}, optional
        Structure of the Jacobian (default: "dense").
    prior_error_type : {"identity", "diagonal", "correlated"}, optional
        Type of prior error covariance (default: "identity").
    obs_error_type : {"identity", "diagonal", "correlated"}, optional
        Type of observation error covariance (default: "diagonal").
    background : float or pd.Series, optional
        Background (baseline) value for observations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Problem
        Initialized Problem instance.
    """
    data = create_problem_data(
        n_state=n_state,
        n_obs=n_obs,
        jacobian_structure=jacobian_structure,
        prior_error_type=prior_error_type,
        obs_error_type=obs_error_type,
        seed=seed,
    )

    return InverseProblem(
        state=data["state"],
        obs=data["obs"],
        forward_operator=data["jacobian"],
        prior_error=data["prior_error"],
        obs_error=data["obs_error"],
        background=background,
    )
