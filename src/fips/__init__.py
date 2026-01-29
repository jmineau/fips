"""Flexible Inverse Problem Solver (FIPS).

A Pythonic framework for solving linear inverse problems using Bayesian estimation.
Provides data structures for state vectors, observations, forward operators, and
covariance matrices; estimators for computing posteriors; and interfaces for
serialization, visualization, and specialized applications like atmospheric
flux inversion.
"""

__version__ = "2025.10.0"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"


from .estimators import Estimator
from .matrices import CovarianceMatrix, ForwardOperator, convolve
from .problem import InverseProblem
from .vectors import Block, Vector

__all__ = [
    "Block",
    "Vector",
    "Estimator",
    "ForwardOperator",
    "CovarianceMatrix",
    "InverseProblem",
    "convolve",
]
