"""
Covariances, and the models that write them down before their labels exist.

A :class:`Covariance` is a symmetric :class:`~fips.Operator` from a space to
itself, with the solves, log-determinant and square root a covariance needs.
It is written as a correlation model with standard deviations:
``Exponential("time", "30D").with_std(std)``. The kernels
(:class:`Exponential`, :class:`Gaussian`, :class:`Matern`, :class:`Constant`)
correlate labels by their distance, measured with the functions of
:mod:`fips.covariance.distances`; :class:`Independent` and :class:`Kronecker`
combine models; a :class:`CovarianceModel` is a covariance waiting for its
space.
"""

from fips.covariance.covariance import Covariance
from fips.covariance.kernels import Constant, Exponential, Gaussian, Matern
from fips.covariance.models import (
    CorrelationModel,
    CovarianceModel,
    Independent,
    Kronecker,
)

__all__ = [
    "Covariance",
    "CorrelationModel",
    "CovarianceModel",
    "Independent",
    "Kronecker",
    "Exponential",
    "Gaussian",
    "Matern",
    "Constant",
]
