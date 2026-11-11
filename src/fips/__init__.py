"""
Flexible Inverse Problem Solver (FIPS).

A Pythonic framework for solving inverse problems using Bayesian estimation.
Labels live in a :class:`Space`; values on one are a :class:`Vector`; a matrix
between two is an :class:`Operator` (a :class:`Covariance` from one to
itself), kept in its cheapest form. A covariance is written as a correlation
model with standard deviations: ``Exponential("time", "30D").with_std(std)``.
A :class:`Problem` puts the prior, the observations, the forward operator (or
a :class:`ForwardModel` that is not linear, with its Jacobian;
:class:`LogNormalForward` and :class:`SoftplusForward` keep a block positive)
and the covariances together, and :func:`solve` returns its :class:`Posterior`.
With priors on the covariance hyperparameters (:mod:`fips.distributions`),
:func:`hierarchical` draws them from their posterior and the state at each
draw (:class:`Hierarchical`).
A finding that changes how far an answer can be trusted is logged under the
``fips`` logger and issued as a :class:`FipsWarning`.
:mod:`fips.aggregate` sums or averages a space by group (the observations to
daily means, the state to monthly totals), and :func:`offset_block` adds a
block of offsets to the state.
"""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import aggregate, distributions
from ._errors import ConvergenceError, FipsWarning
from .aggregate import offset_block
from .covariance import (
    Constant,
    Covariance,
    Exponential,
    Gaussian,
    Independent,
    Kronecker,
    Matern,
)
from .forward import ForwardModel, taylortest
from .hyperparameters import Hierarchical, hierarchical
from .methods import available_methods, solve
from .operator import Operator
from .posterior import Posterior, UnavailableError
from .problem import Problem
from .space import Space
from .transforms import LogNormalForward, SoftplusForward
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
    "ForwardModel",
    "taylortest",
    "LogNormalForward",
    "SoftplusForward",
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
    "UnavailableError",
    "FipsWarning",
    "ConvergenceError",
    "hierarchical",
    "Hierarchical",
    "distributions",
    "aggregate",
    "offset_block",
]
