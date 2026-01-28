"""fips

Flexible Inverse Problem Solver (FIPS)
"""

__version__ = "2025.10.0"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"


from .estimators import Estimator
from .matrices import CovarianceMatrix, ForwardOperator, convolve
from .plotting import compute_credible_interval, plot_comparison, plot_error_norm
from .problem import InverseProblem
from .vectors import Block, Vector

# import problems  #TODO should this be here?


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
]
