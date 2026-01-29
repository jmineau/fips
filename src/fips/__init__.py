"""fips

Flexible Inverse Problem Solver (FIPS)
"""

__version__ = "2025.10.0"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"


from .estimators import Estimator
from .kernels import (
    exponential_decay_kernel,
    exponential_with_masking_kernel,
    location_specific_kernel,
    same_day_groupwise_kernel,
    spatial_decay_kernel,
    spatio_temporal_kernel,
)
from .matrices import CovarianceMatrix, ForwardOperator, convolve
from .parallel import parallelize
from .problem import InverseProblem
from .spacetime import (
    dataframe_to_xarray,
    enough_obs_per_interval,
    filter_intervals,
    haversine_matrix,
    integrate_over_time_bins,
    series_to_xarray,
    time_difference_matrix,
)
from .vectors import Block, Vector
from .visualization import compute_credible_interval, plot_comparison, plot_error_norm

__all__ = [
    "Block",
    "Vector",
    "Estimator",
    "ForwardOperator",
    "CovarianceMatrix",
    "InverseProblem",
    "convolve",
    "plot_comparison",
    "plot_error_norm",
    "compute_credible_interval",
    "exponential_decay_kernel",
    "spatial_decay_kernel",
    "spatio_temporal_kernel",
    "same_day_groupwise_kernel",
    "location_specific_kernel",
    "exponential_with_masking_kernel",
    "haversine_matrix",
    "integrate_over_time_bins",
    "time_difference_matrix",
    "filter_intervals",
    "enough_obs_per_interval",
    "parallelize",
    "series_to_xarray",
    "dataframe_to_xarray",
]
