"""Flux-specific test data generators for atmospheric flux inversion."""

import numpy as np
import pandas as pd

from fips.problems.flux import FluxProblem, ModelDataMismatch, PriorError

from .utils import (
    create_correlated_noise,
    create_obs_multiindex,
    create_state_multiindex,
    normalize_jacobian,
)


def create_transport_jacobian(
    n_obs_locations,
    n_obs_times,
    n_times,
    n_lat,
    n_lon,
    transport_range=(0.0, 1.0),
    seed=None,
):
    """Create a synthetic STILT-like transport Jacobian.

    The Jacobian represents sensitivity of observed concentrations to surface fluxes,
    typically derived from atmospheric transport models like STILT.

    Parameters
    ----------
    n_obs_locations, n_obs_times : int
        Dimensions of observations.
    n_times, n_lat, n_lon : int
        Dimensions of state (fluxes).
    transport_range : tuple
        Min/max values for transport sensitivities (typically 0-1 after normalization).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Transport Jacobian matrix of shape (n_obs, n_state).
    """
    if seed is not None:
        np.random.seed(seed)

    n_obs = n_obs_locations * n_obs_times
    n_state = n_times * n_lat * n_lon

    # Create sparse, localized sensitivities (realistic for transport models)
    jacobian = np.zeros((n_obs, n_state))

    # For each observation, create a localized sensitivity pattern
    for i in range(n_obs):
        # Random center of influence
        state_idx = np.random.choice(n_state, size=max(1, n_state // 10), replace=False)
        # Gaussian-like decay from center
        distances = np.abs(np.arange(n_state)[:, np.newaxis] - state_idx[np.newaxis, :])
        sensitivities = np.exp(-(distances**2) / (n_state // 10) ** 2).sum(axis=1)
        jacobian[i] = (
            sensitivities / np.max(sensitivities) if np.max(sensitivities) > 0 else 0
        )

    # Normalize to target range
    jacobian = normalize_jacobian(jacobian, scale=np.mean(transport_range))
    jacobian = np.clip(jacobian, transport_range[0], transport_range[1])

    return jacobian


def create_prior_flux(
    n_times,
    n_lat,
    n_lon,
    time_start="2020-01-01",
    time_freq="D",
    lat_range=(30, 45),
    lon_range=(-120, -105),
    mean_flux=1.0,
    variability=0.3,
    seed=None,
):
    """Create a synthetic prior flux inventory (state variable).

    Parameters
    ----------
    n_times, n_lat, n_lon : int
        Dimensions of the flux grid.
    time_start : str
        Start date for time index.
    time_freq : str
        Frequency for time index.
    lat_range, lon_range : tuple
        Ranges for spatial dimensions.
    mean_flux : float, optional
        Mean flux value (default: 1.0).
    variability : float, optional
        Coefficient of variation (default: 0.3).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Prior flux inventory with MultiIndex (time, lat, lon).
    """
    if seed is not None:
        np.random.seed(seed)

    state_idx = create_state_multiindex(
        n_times,
        n_lat,
        n_lon,
        time_start=time_start,
        time_freq=time_freq,
        lat_range=lat_range,
        lon_range=lon_range,
    )

    # Log-normal distribution (realistic for fluxes)
    log_fluxes = np.log(mean_flux) + np.random.randn(len(state_idx)) * variability
    fluxes = np.exp(log_fluxes)

    return pd.Series(fluxes, index=state_idx)


def create_observed_concentrations(
    n_obs_locations,
    n_obs_times,
    time_start="2020-01-01",
    time_freq="D",
    lat_range=(30, 45),
    lon_range=(-120, -105),
    mean_conc=400.0,
    conc_std=5.0,
    seed=None,
):
    """Create synthetic observed atmospheric concentrations.

    Parameters
    ----------
    n_obs_locations, n_obs_times : int
        Number of observation locations and times.
    time_start : str
        Start date for time index.
    time_freq : str
        Frequency for time index.
    lat_range, lon_range : tuple
        Ranges for observation location coordinates.
    mean_conc : float, optional
        Mean concentration value (default: 400.0, e.g., ppm for CO2).
    conc_std : float, optional
        Standard deviation of concentrations (default: 5.0).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Observed concentrations with MultiIndex (obs_location, obs_time).
    """
    if seed is not None:
        np.random.seed(seed)

    obs_idx = create_obs_multiindex(
        n_obs_locations,
        n_obs_times,
        time_start=time_start,
        time_freq=time_freq,
        lat_range=lat_range,
        lon_range=lon_range,
    )

    # Create temporally correlated concentrations
    n_obs = len(obs_idx)
    noise = create_correlated_noise((n_obs,), correlation="temporal", seed=seed)
    concs = mean_conc + conc_std * noise

    return pd.Series(concs, index=obs_idx)


def create_prior_error_covariance(
    state_idx,
    error_frac=0.2,
    correlation_type="exponential",
    time_decay_days=7,
    space_decay_km=500,
    seed=None,
):
    """Create a prior error covariance matrix for flux uncertainty.

    Parameters
    ----------
    state_idx : pd.MultiIndex
        State index with (time, lat, lon) levels.
    error_frac : float, optional
        Fractional uncertainty relative to flux values (default: 0.2 = 20%).
    correlation_type : {"exponential", "none"}, optional
        Type of correlation structure (default: "exponential").
    time_decay_days : float, optional
        Time scale for correlation decay in days (default: 7).
    space_decay_km : float, optional
        Space scale for correlation decay in km (default: 500).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    PriorError
        Prior error covariance matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(state_idx)

    # Create diagonal elements (variances)
    variances = np.ones(n) * (error_frac**2)

    if correlation_type == "exponential":
        # Create correlated prior error matrix
        cov_matrix = np.diag(variances)

        # Add exponential decay correlations
        for i in range(n):
            for j in range(i + 1, n):
                # Extract indices
                time_i, lat_i, lon_i = state_idx[i]
                time_j, lat_j, lon_j = state_idx[j]

                # Time distance
                time_dist = abs((time_j - time_i).days) / time_decay_days

                # Spatial distance (rough Euclidean)
                space_dist = np.sqrt((lat_j - lat_i) ** 2 + (lon_j - lon_i) ** 2) / (
                    space_decay_km / 111
                )  # ~111 km per degree

                # Combined exponential decay
                correlation = np.exp(-(time_dist**2 + space_dist**2))
                cov_matrix[i, j] = correlation * np.sqrt(variances[i] * variances[j])
                cov_matrix[j, i] = cov_matrix[i, j]

    else:
        # Diagonal (no correlation)
        cov_matrix = np.diag(variances)

    cov_df = pd.DataFrame(cov_matrix, index=state_idx, columns=state_idx)
    return PriorError(cov_df)


def create_model_data_mismatch_covariance(
    obs_idx,
    error_frac=0.1,
    correlation_type="exponential",
    time_decay_days=1,
    spatial_corr=False,
    seed=None,
):
    """Create observation/model error covariance matrix.

    Parameters
    ----------
    obs_idx : pd.MultiIndex
        Observation index with (obs_location, obs_time) levels.
    error_frac : float, optional
        Fractional uncertainty relative to observation values (default: 0.1 = 10%).
    correlation_type : {"exponential", "none"}, optional
        Type of correlation structure (default: "exponential").
    time_decay_days : float, optional
        Time scale for correlation decay in days (default: 1).
    spatial_corr : bool, optional
        Whether to include spatial correlation between stations (default: False).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ModelDataMismatch
        Model-data mismatch error covariance matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(obs_idx)

    # Create diagonal elements (variances)
    variances = np.ones(n) * (error_frac**2)

    if correlation_type == "exponential":
        cov_matrix = np.diag(variances)

        # Add exponential decay correlations in time
        for i in range(n):
            for j in range(i + 1, n):
                obs_loc_i, obs_time_i = obs_idx[i]
                obs_loc_j, obs_time_j = obs_idx[j]

                # Time distance
                time_dist = abs((obs_time_j - obs_time_i).days) / time_decay_days
                correlation = np.exp(-(time_dist**2))

                # Optionally add spatial correlation
                if spatial_corr and obs_loc_i != obs_loc_j:
                    correlation *= 0.5  # Reduce correlation between different stations

                cov_matrix[i, j] = correlation * np.sqrt(variances[i] * variances[j])
                cov_matrix[j, i] = cov_matrix[i, j]

    else:
        cov_matrix = np.diag(variances)

    cov_df = pd.DataFrame(cov_matrix, index=obs_idx, columns=obs_idx)
    return ModelDataMismatch(cov_df)


def create_flux_problem_data(
    n_times=5,
    n_lat=3,
    n_lon=3,
    n_obs_locations=2,
    n_obs_times=10,
    time_start="2020-01-01",
    time_freq="D",
    lat_range=(30, 45),
    lon_range=(-120, -105),
    mean_flux=1.0,
    mean_conc=400.0,
    prior_error_frac=0.2,
    obs_error_frac=0.1,
    seed=None,
):
    """Create a complete synthetic flux problem dataset.

    Parameters
    ----------
    n_times, n_lat, n_lon : int
        Dimensions of the state (flux) grid.
    n_obs_locations, n_obs_times : int
        Dimensions of the observations.
    time_start : str
        Start date for time index.
    time_freq : str
        Frequency for time index.
    lat_range, lon_range : tuple
        Ranges for spatial dimensions.
    mean_flux : float, optional
        Mean prior flux value (default: 1.0).
    mean_conc : float, optional
        Mean observed concentration (default: 400.0).
    prior_error_frac : float, optional
        Fractional prior flux uncertainty (default: 0.2).
    obs_error_frac : float, optional
        Fractional observation uncertainty (default: 0.1).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys: "inventory", "concentrations", "jacobian", "prior_error", "obs_error".
    """
    if seed is not None:
        np.random.seed(seed)

    # Create indices
    state_idx = create_state_multiindex(
        n_times,
        n_lat,
        n_lon,
        time_start=time_start,
        time_freq=time_freq,
        lat_range=lat_range,
        lon_range=lon_range,
    )
    obs_idx = create_obs_multiindex(
        n_obs_locations,
        n_obs_times,
        time_start=time_start,
        time_freq=time_freq,
        lat_range=lat_range,
        lon_range=lon_range,
    )

    # Create flux scenario
    inventory = create_prior_flux(
        n_times,
        n_lat,
        n_lon,
        time_start=time_start,
        time_freq=time_freq,
        lat_range=lat_range,
        lon_range=lon_range,
        mean_flux=mean_flux,
        seed=seed,
    )

    # Create observations
    concentrations = create_observed_concentrations(
        n_obs_locations,
        n_obs_times,
        time_start=time_start,
        time_freq=time_freq,
        lat_range=lat_range,
        lon_range=lon_range,
        mean_conc=mean_conc,
        seed=seed,
    )

    # Create transport Jacobian
    n_obs = len(obs_idx)
    n_state = len(state_idx)
    transport_jacobian = create_transport_jacobian(
        n_obs_locations,
        n_obs_times,
        n_times,
        n_lat,
        n_lon,
        seed=seed,
    )
    jacobian_df = pd.DataFrame(transport_jacobian, index=obs_idx, columns=state_idx)

    # Create error covariances
    prior_error = create_prior_error_covariance(
        state_idx,
        error_frac=prior_error_frac,
        seed=seed,
    )

    obs_error = create_model_data_mismatch_covariance(
        obs_idx,
        error_frac=obs_error_frac,
        seed=seed,
    )

    return {
        "inventory": inventory,
        "concentrations": concentrations,
        "jacobian": jacobian_df,
        "prior_error": prior_error,
        "obs_error": obs_error,
    }


def create_flux_problem(
    n_times=5,
    n_lat=3,
    n_lon=3,
    n_obs_locations=2,
    n_obs_times=10,
    time_start="2020-01-01",
    time_freq="D",
    lat_range=(30, 45),
    lon_range=(-120, -105),
    mean_flux=1.0,
    mean_conc=400.0,
    prior_error_frac=0.2,
    obs_error_frac=0.1,
    background=None,
    seed=None,
):
    """Create a complete FluxProblem instance with synthetic data.

    Parameters are the same as create_flux_problem_data, plus:

    Parameters
    ----------
    background : float, optional
        Background (baseline) concentration value.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    FluxProblem
        Initialized FluxProblem instance.
    """
    data = create_flux_problem_data(
        n_times=n_times,
        n_lat=n_lat,
        n_lon=n_lon,
        n_obs_locations=n_obs_locations,
        n_obs_times=n_obs_times,
        time_start=time_start,
        time_freq=time_freq,
        lat_range=lat_range,
        lon_range=lon_range,
        mean_flux=mean_flux,
        mean_conc=mean_conc,
        prior_error_frac=prior_error_frac,
        obs_error_frac=obs_error_frac,
        seed=seed,
    )

    prob = FluxProblem(
        inventory=data["inventory"],
        concentrations=data["concentrations"],
        jacobian=data["jacobian"],
        prior_error=data["prior_error"],
        obs_error=data["obs_error"],
        background=background,
    )

    return prob
