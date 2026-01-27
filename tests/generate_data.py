import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, List, Any

from fips.vectors import Vector, Block
from fips.matrices import CovarianceMatrix


def _parse_dimensions(
    dims: Union[int, List[int], Dict[str, int]], 
    name_prefix: str
) -> Tuple[pd.Index, np.ndarray, Tuple[int, ...]]:
    """Helper to parse flexible dimension definitions into Index and Coordinates."""
    
    # Normalize input to (shape, names) pair
    if isinstance(dims, int):
        shape = (dims,)
        names = [name_prefix]
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
    coords = np.stack([index.get_level_values(i).values for i in range(len(shape))], axis=1)
    
    return index, coords, shape


def generate_test_data(
    n_state: Union[int, List[int], Dict[str, int]] = 10,
    n_obs: Union[int, List[int], Dict[str, int]] = 15,
    prior_sigma: float = 1.0,
    obs_sigma: float = 0.5,
    correlation_len: float = 3.0,
    seed: int = 42
) -> Dict[str, Any]:
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
        Correlation length scale for prior covariance and Jacobian.
    seed : int
        Random seed.
        
    Returns
    -------
    Dict containing:
        - 'prior': pd.Series (Prior state x_a)
        - 'obs': pd.Series (Synthetic observations y_obs)
        - 'jacobian': pd.DataFrame (Forward operator H)
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
        idx_obs = pd.Index(np.arange(n_obs), name="obs")
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
        width = rng.uniform(0.5, np.max(shape_state)/3)
        amp = rng.uniform(5, 10)
        dist_sq = np.sum((coords_state - center)**2, axis=1)
        truth_vals += amp * np.exp(-dist_sq / (2 * width**2))
        
    truth_vals += 10.0
    truth_state = pd.Series(truth_vals, index=idx_state, name="state_var")

    # 3. Prior State (Distorted Truth)
    noise = rng.normal(0, prior_sigma, n_state_tot)
    prior_vals = truth_vals * rng.uniform(0.8, 1.2) + noise
    prior_state = pd.Series(prior_vals, index=idx_state, name="state_var")

    # 4. Jacobian (Distance-based)
    # Pad to max dimensions so distances include all spatial dims
    max_dims = max(coords_obs.shape[1], coords_state.shape[1])
    c_obs = np.pad(coords_obs, ((0, 0), (0, max_dims - coords_obs.shape[1])), mode='constant')
    c_state = np.pad(coords_state, ((0, 0), (0, max_dims - coords_state.shape[1])), mode='constant')
    
    diff = c_obs[:, None, :] - c_state[None, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    H_vals = np.exp(-dists_sq / (2 * correlation_len**2))
    
    # Normalize rows (avoid zero div)
    row_sums = H_vals.sum(axis=1, keepdims=True) + 1e-12
    H_vals = H_vals / row_sums
    
    H_df = pd.DataFrame(H_vals, index=idx_obs, columns=idx_state)

    # 5. Covariances (CovarianceMatrix)
    # Temporary vectors for builder logic
    x_vec = Vector([Block("state_var", prior_state)])
    # Placeholder obs series just for structure
    obs_placeholder = pd.Series(np.zeros(n_obs_tot), index=idx_obs, name="obs_var")
    obs_vec = Vector([Block("obs_var", obs_placeholder)])

    # Generic N-D Kernel
    def nd_decay(sigma, length_scale):
        def kernel(idx1, idx2):
            def get_coords(idx):
                if isinstance(idx, pd.MultiIndex):
                    return np.stack([idx.get_level_values(i).values for i in range(len(idx.levels))], axis=1)
                else:
                    return idx.values[:, None]
            c1 = get_coords(idx1)
            c2 = get_coords(idx2)
            # Pad to max dimensions
            max_d = max(c1.shape[1], c2.shape[1])
            c1 = np.pad(c1, ((0, 0), (0, max_d - c1.shape[1])), mode='constant')
            c2 = np.pad(c2, ((0, 0), (0, max_d - c2.shape[1])), mode='constant')
            diff = c1[:, None, :] - c2[None, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            return (sigma**2) * np.exp(-np.sqrt(dist_sq) / length_scale)
        return kernel

    # Build Prior Error (S_0)
    S_prior = CovarianceMatrix.from_vector(x_vec)
    S_prior.set_block("state_var", sigma=prior_sigma, kernel=nd_decay(1.0, correlation_len))
    
    # Build Obs Error (S_z)
    S_obs = CovarianceMatrix.from_vector(obs_vec)
    S_obs.set_block("obs_var", sigma=obs_sigma)

    # 6. Synthetic Observations
    # y = Hx + noise
    # We use raw values since H and Truth are constructed aligned
    y_clean = H_df.values @ truth_state.values
    
    # Noise based on S_obs
    noise = rng.multivariate_normal(
        mean=np.zeros(n_obs_tot), 
        cov=S_obs.values, 
        check_valid='warn'
    )
    y_obs_vals = y_clean + noise
    obs_state = pd.Series(y_obs_vals, index=idx_obs, name="obs_var")

    return {
        "prior": prior_state,
        "obs": obs_state,
        "jacobian": H_df,
        "prior_error": S_prior,
        "modeldata_mismatch": S_obs,
        "truth": truth_state
    }
