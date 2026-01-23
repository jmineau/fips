"""Test data generators for fips package."""

from .flux import (
    create_flux_problem,
    create_flux_problem_data,
    create_model_data_mismatch_covariance,
    create_observed_concentrations,
    create_prior_error_covariance,
    create_prior_flux,
    create_transport_jacobian,
)
from .generic import (
    create_covariance_matrix,
    create_jacobian,
    create_problem,
    create_problem_2d,
    create_problem_data,
)
from .utils import (
    create_correlated_noise,
    create_obs_multiindex,
    create_spatial_index,
    create_state_multiindex,
    create_time_index,
    normalize_jacobian,
)

__all__ = [
    # Utils
    "create_time_index",
    "create_spatial_index",
    "create_state_multiindex",
    "create_obs_multiindex",
    "create_correlated_noise",
    "normalize_jacobian",
    # Generic
    "create_jacobian",
    "create_covariance_matrix",
    "create_problem_data",
    "create_problem",
    "create_problem_2d",
    # Flux
    "create_transport_jacobian",
    "create_prior_flux",
    "create_observed_concentrations",
    "create_prior_error_covariance",
    "create_model_data_mismatch_covariance",
    "create_flux_problem_data",
    "create_flux_problem",
]
