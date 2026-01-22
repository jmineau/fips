"""fips

Flexible Inverse Problem Solver (FIPS)
"""

__version__ = "2025.10.0"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"


from .estimators import Estimator
from .matrices import CovarianceMatrix
from .operators import ForwardOperator, convolve
from .problem import InverseProblem

# import problems  #TODO should this be here?


__all__ = [
    "Estimator",
    "ForwardOperator",
    "CovarianceMatrix",
    "InverseProblem",
    "convolve",
]
