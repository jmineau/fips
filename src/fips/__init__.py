"""
Flexible Inverse Problem Solver (FIPS).

A Pythonic framework for solving linear inverse problems using Bayesian estimation.
Labels live in a :class:`Space`; values on one are a :class:`Vector`; a matrix
between two is an :class:`Operator` (a :class:`Covariance` from one to
itself), kept in its cheapest form. A covariance is written as a correlation
model with standard deviations: ``Exponential("time", "30D").with_std(std)``.
A :class:`Problem` puts the prior, the observations, the forward operator and
the covariances together, and :func:`solve` returns its :class:`Posterior`.
:mod:`fips.aggregate` sums or averages a space by group (the observations to
daily means, the state to monthly totals), :func:`offset_block` adds a block
of offsets to the state, and :mod:`fips.scales` sizes a model-data mismatch
that grows with the signal.
"""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import aggregate, scales
from .aggregate import offset_block
from .correlation import Independent, Kronecker
from .covariance import Covariance
from .kernels import Constant, Exponential, Gaussian, Matern
from .operator import Operator
from .pipeline import InversionPipeline
from .posterior import Posterior
from .problem import Problem
from .solve import available_methods, solve
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
    "Problem",
    "solve",
    "available_methods",
    "Posterior",
    "aggregate",
    "offset_block",
    "scales",
    "InversionPipeline",
]
