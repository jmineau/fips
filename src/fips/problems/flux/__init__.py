"""
Atmospheric flux inversion problem domain.

This package provides specialized classes and utilities for solving
atmospheric flux inversion problems, including the `FluxProblem` class,
transport models, and visualization tools.
"""

from fips.problems.flux.pipeline import FluxInversionPipeline
from fips.problems.flux.problem import FluxProblem
from fips.problems.flux.visualization import FluxPlotter

__all__ = [
    "FluxProblem",
    "FluxPlotter",
    "FluxInversionPipeline",
]

# Optional STILT transport backend (requires stilt package)
try:
    from fips.problems.flux.transport.stilt import JacobianBuilder  # noqa: F401

    __all__.append("JacobianBuilder")
except ImportError:
    pass
