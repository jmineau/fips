"""fips

Flexible Inverse Problem Solver (FIPS)
"""

__version__ = "2025.10.0"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"


from .core import (
    Estimator,
    ESTIMATOR_REGISTRY,
    SymmetricMatrix,
    InverseProblem,
)
# import .estimators  # import to register estimators
# import problems


__all__ = [
    "Estimator",
    "ESTIMATOR_REGISTRY",
    "SymmetricMatrix",
    "InverseProblem",
]