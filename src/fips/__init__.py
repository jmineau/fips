"""
Flexible Inverse Problem Solver (FIPS).

A Pythonic framework for solving linear inverse problems using Bayesian estimation.
Labels live in a :class:`Space`; values on one are a :class:`Vector`; a matrix
between two is an :class:`Operator` (a :class:`Covariance` from one to
itself), kept in its cheapest form. A covariance is written as a correlation
model with standard deviations: ``Exponential("time", "30D").with_std(std)``. Estimators compute posteriors; there are
interfaces for serialization, visualization, and specialized applications like
atmospheric flux inversion.
"""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .correlation import Independent, Kronecker
from .covariance import Covariance
from .estimators import Estimator, available_estimators
from .kernels import Constant, Exponential, Gaussian, Matern
from .operator import Operator, convolve
from .pipeline import InversionPipeline
from .posterior import Posterior
from .problem import InverseProblem, solve
from .space import Space
from .vector import Vector

try:
    __version__ = _version("fips")  # set by setuptools-scm from git tags
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0+unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "Space",
    "Vector",
    "Operator",
    "Covariance",
    "Exponential",
    "Gaussian",
    "Matern",
    "Constant",
    "Independent",
    "Kronecker",
    "Estimator",
    "available_estimators",
    "InversionPipeline",
    "InverseProblem",
    "Posterior",
    "solve",
    "convolve",
]
