"""fips

Flexible Inverse Problem Solver (FIPS)
"""

__version__ = "2025.10.0"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"


from .estimators import Estimator
from .matrices import CovarianceMatrix
from .operator import ForwardOperator, convolve
from .problem import InverseProblem

# import .estimators  # import to register estimators
# import problems


__all__ = [
    "Estimator",
    "ForwardOperator",
    "CovarianceMatrix",
    "InverseProblem",
    "convolve",
]
