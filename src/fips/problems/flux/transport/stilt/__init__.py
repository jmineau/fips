"""
STILT atmospheric transport model integration.

This package provides tools for building the Jacobian (sensitivity) matrix
from STILT atmospheric transport simulations, including footprint loading
and simulation management.

Note: Requires the 'stilt' package to be installed.
"""

__all__ = []

# Optional imports - requires stilt package
try:
    from fips.problems.flux.transport.stilt.builder import JacobianBuilder  # noqa: F401

    __all__.append("JacobianBuilder")
except ImportError:
    pass
