"""Flexible Inverse Problem Solver (FIPS).

A Pythonic framework for solving linear inverse problems using Bayesian estimation.
Provides data structures for state vectors, observations, forward operators, and
covariance matrices; estimators for computing posteriors; and interfaces for
serialization, visualization, and specialized applications like atmospheric
flux inversion.
"""

import logging

__version__ = "2025.10.0"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"


from .covariance import CovarianceMatrix
from .estimators import Estimator
from .matrix import Matrix, MatrixBlock
from .operators import ForwardOperator, convolve
from .problem import InverseProblem
from .vector import Block, Vector

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Block",
    "Vector",
    "Matrix",
    "MatrixBlock",
    "Estimator",
    "ForwardOperator",
    "CovarianceMatrix",
    "InverseProblem",
    "convolve",
]
