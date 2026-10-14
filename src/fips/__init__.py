"""
Flexible Inverse Problem Solver (FIPS).

A Pythonic framework for solving linear inverse problems using Bayesian estimation.
Labels live in a :class:`Space`; values on one are a :class:`Field`; a matrix
between two is an :class:`Operator` (a :class:`Covariance` from one to
itself), kept in its cheapest form. Estimators compute posteriors; there are
interfaces for serialization, visualization, and specialized applications like
atmospheric flux inversion.
"""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .covariance import Covariance
from .estimators import Estimator, available_estimators
from .field import Field
from .operator import Operator, convolve
from .pipeline import InversionPipeline
from .posterior import Posterior
from .problem import InverseProblem, solve
from .space import Space
from .vector import Block, Vector

try:
    __version__ = _version("fips")  # set by setuptools-scm from git tags
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0+unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "Space",
    "Field",
    "Operator",
    "Covariance",
    "Block",
    "Vector",
    "Estimator",
    "available_estimators",
    "InversionPipeline",
    "InverseProblem",
    "Posterior",
    "solve",
    "convolve",
]
