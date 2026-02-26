from typing import Any

import numpy as np
import pandas as pd

from fips.covariance import CovarianceMatrix
from fips.indexes import assign_block


def promote_index(
    index: pd.Index, block_name: str, level_name: str = "block"
) -> pd.Index:
    """Promote an index by adding a block level.

    Parameters
    ----------
    index : pd.Index
        The index to promote.
    block_name : str
        The name/value of the block to assign.
    level_name : str
        The name of the level to add (default "block").

    Returns
    -------
    pd.Index
        The promoted index with block level added.
    """
    return assign_block(index, block_name)


def _parse_dimensions(
    dims: int | list[int] | dict[str, int], name_prefix: str
) -> tuple[pd.Index, np.ndarray, tuple[int, ...]]:
    """Helper to parse flexible dimension definitions into Index and Coordinates."""

    # Normalize input to (shape, names) pair
    if isinstance(dims, int):
        shape = (dims,)
        names = [f"{name_prefix}_0"]
    elif isinstance(dims, dict):
        names = list(dims.keys())
        shape = tuple(dims.values())
    elif isinstance(dims, list):
        shape = tuple(dims)
        names = [f"{name_prefix}_{i}" for i in range(len(shape))]
    else:
        raise TypeError(f"Dimensions must be int, list, or dict. Got {type(dims)}")

    # Create index and coordinates
    axes = [np.arange(s) for s in shape]
    index = pd.MultiIndex.from_product(axes, names=names)
    coords = np.stack(
        [index.get_level_values(i).to_numpy() for i in range(len(shape))], axis=1
    )

    return index, coords, shape


def generate_test_data(
    n_state: int | list[int] | dict[str, int] = 10,
    n_obs: int | list[int] | dict[str, int] = 15,
    prior_sigma: float = 1.0,
    obs_sigma: float = 0.5,
    correlation_len: float = 3.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Generates synthetic data components for an inverse problem.

    Parameters
    ----------
    n_state : int, list, or dict
        State dimensions (e.g. 10, [10,10], or {'lat':10}).
    n_obs : int, list, or dict
        Observation dimensions.
    prior_sigma : float
        Prior error standard deviation.
    obs_sigma : float
        Observation error standard deviation.
    correlation_len : float
        Correlation length scale for prior covariance and forward operator.
    seed : int
        Random seed.

    Returns
    -------
    Dict containing:
        - 'prior': pd.Series (Prior state x_a)
        - 'obs': pd.Series (Synthetic observations y_obs)
        - 'forward_operator': pd.DataFrame (Forward operator H)
        - 'prior_error': CovarianceMatrix (S_a)
        - 'modeldata_mismatch': CovarianceMatrix (S_z)
        - 'truth': pd.Series (Hidden truth x_true)
    """
    rng = np.random.default_rng(seed)

    # 1. Parse Dimensions
    idx_state, coords_state, shape_state = _parse_dimensions(n_state, "state")
    n_state_tot = len(idx_state)
    ndim_state = coords_state.shape[1]

    if isinstance(n_obs, int):
        idx_obs = pd.Index(np.arange(n_obs), name="obs_0")
        n_obs_tot = n_obs
        bounds = np.array(shape_state) - 1
        coords_obs = rng.uniform(0, bounds, size=(n_obs, ndim_state))
    else:
        idx_obs, coords_obs, shape_obs = _parse_dimensions(n_obs, "obs")
        n_obs_tot = len(idx_obs)

    # 2. Truth State (Gaussian Blobs)
    truth_vals = np.zeros(n_state_tot)
    n_blobs = max(1, int(np.log(n_state_tot))) + 1

    for _ in range(n_blobs):
        center = rng.uniform(0, np.array(shape_state), size=ndim_state)
        width = rng.uniform(0.5, np.max(shape_state) / 3)
        amp = rng.uniform(5, 10)
        dist_sq = np.sum((coords_state - center) ** 2, axis=1)
        truth_vals += amp * np.exp(-dist_sq / (2 * width**2))

    truth_vals += 10.0
    truth_state = pd.Series(truth_vals, index=idx_state, name="truth")

    # 3. Prior State (Distorted Truth)
    noise = rng.normal(0, prior_sigma, n_state_tot)
    prior_vals = truth_vals * rng.uniform(0.8, 1.2) + noise
    prior_state = pd.Series(prior_vals, index=idx_state, name="prior")

    # 4. Forward Operator (Distance-based)
    # Pad to max dimensions so distances include all spatial dims
    max_dims = max(coords_obs.shape[1], coords_state.shape[1])
    c_obs = np.pad(
        coords_obs, ((0, 0), (0, max_dims - coords_obs.shape[1])), mode="constant"
    )
    c_state = np.pad(
        coords_state, ((0, 0), (0, max_dims - coords_state.shape[1])), mode="constant"
    )

    diff = c_obs[:, None, :] - c_state[None, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    H_vals = np.exp(-dists_sq / (2 * correlation_len**2))

    # Normalize rows (avoid zero div)
    row_sums = H_vals.sum(axis=1, keepdims=True) + 1e-12
    H_vals = H_vals / row_sums

    H_df = pd.DataFrame(H_vals, index=idx_obs, columns=idx_state)
    # Promote to include block level so H aligns with Vector/Covariance indices
    # Use "state" and "obs" as block names to match what InverseProblem expects
    H_df.index = promote_index(H_df.index, "obs", "block")
    H_df.columns = promote_index(H_df.columns, "state", "block")

    # 5. Covariances (CovarianceMatrix)
    # Build correlation matrix using spatial distance
    # For prior covariance, use spatial correlation
    def build_spatial_corr(coords, length_scale):
        """Build spatial correlation matrix from coordinates."""
        n = len(coords)
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                corr[i, j] = np.exp(-dist / length_scale)
        return corr

    spatial_corr = build_spatial_corr(coords_state, correlation_len)
    prior_variances = np.full(n_state_tot, prior_sigma**2)

    # Build Prior Error (S_0) from variances and correlation
    sigma_prior = np.diag(np.sqrt(prior_variances))
    S_prior_vals = sigma_prior @ spatial_corr @ sigma_prior
    S_prior_idx = promote_index(idx_state, "state", "block")
    S_prior_df = pd.DataFrame(S_prior_vals, index=S_prior_idx, columns=S_prior_idx)
    S_prior = CovarianceMatrix(S_prior_df)

    # Build Obs Error (S_z) - diagonal
    obs_variances = np.full(n_obs_tot, obs_sigma**2)
    S_obs_idx = promote_index(idx_obs, "obs", "block")
    S_obs_df = pd.DataFrame(np.diag(obs_variances), index=S_obs_idx, columns=S_obs_idx)
    S_obs = CovarianceMatrix(S_obs_df)
    # 6. Synthetic Observations
    # y = Hx + noise
    # We use raw values since H and Truth are constructed aligned
    y_clean = H_df.values @ truth_state.values

    # Noise based on S_obs
    noise = rng.multivariate_normal(
        mean=np.zeros(n_obs_tot), cov=S_obs.values, check_valid="warn"
    )
    y_obs_vals = y_clean + noise
    obs_state = pd.Series(y_obs_vals, index=idx_obs, name="obs")

    return {
        "prior": prior_state,
        "obs": obs_state,
        "forward_operator": H_df,
        "prior_error": S_prior,
        "modeldata_mismatch": S_obs,
        "truth": truth_state,
    }
