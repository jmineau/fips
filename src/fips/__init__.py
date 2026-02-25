"""Flexible Inverse Problem Solver (FIPS).

A Pythonic framework for solving linear inverse problems using Bayesian estimation.
Provides data structures for state vectors, observations, forward operators, and
covariance matrices; estimators for computing posteriors; and interfaces for
serialization, visualization, and specialized applications like atmospheric
flux inversion.
"""

import logging

from .covariance import CovarianceMatrix
from .estimators import Estimator, available_estimators
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
    "available_estimators",
    "ForwardOperator",
    "CovarianceMatrix",
    "InverseProblem",
    "convolve",
]
