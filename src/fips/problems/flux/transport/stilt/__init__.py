"""
STILT atmospheric transport model integration.

This package provides tools for building the Jacobian (sensitivity) matrix
from STILT atmospheric transport simulations, including footprint loading
and simulation management.
"""

from fips.problems.flux.transport.stilt.builder import JacobianBuilder

__all__ = ["JacobianBuilder"]
